from .utils import *

from librosa.filters import *

import numpy as np
import librosa
import torch

class LHVQT(torch.nn.Module):
    def __init__(self, fs = 22050, harmonics = [0.5, 1, 2, 3, 4, 5], hop_length = 256, fmin = None,
                       n_bins = 360, bins_per_octave = 60, filter_scale = 1, gamma = 0, norm = 1,
                       window = 'hann', scale = True, norm_length = True, random = False, max_p = 1,
                       log = True, batch_norm = True, log_scale = True):
        super(LHVQT, self).__init__()

        if fmin is None:
            # C1 by default
            fmin = librosa.note_to_hz('C1')

        # Make all parameters accessible
        self.fs = fs
        self.harmonics = harmonics
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.filter_scale = filter_scale
        self.gamma = gamma
        self.norm = norm
        self.window = window
        self.scale = scale
        self.norm_length = norm_length
        self.random = random
        self.max_p = max_p
        self.log = log
        self.batch_norm = batch_norm
        self.log_scale = log_scale

        # Add CQTs as elements of a Torch module
        self.tfs = torch.nn.Module()
        for h_num in range(len(harmonics)):
            h = harmonics[h_num]
            mod_name = 'lcqt%d' % h_num # Name according to index of harmonic
            self.tfs.add_module(mod_name,
                                LVQT(fs, hop_length, fmin * float(h), n_bins,
                                     bins_per_octave, filter_scale, gamma, norm,
                                     window, scale, norm_length, random, max_p,
                                     log, batch_norm, log_scale))

    def forward(self, wav):
        # Lists to hold cqts and shapes of cqts
        cqt_list = []
        shapes = []

        for h in range(len(self.harmonics)):
            # Take the transform at each harmonic
            tf = torch.nn.Sequential(*list(self.tfs.children()))[h]
            cqt = tf(wav)
            cqt = cqt.unsqueeze(0)
            cqt_list.append(cqt)
            shapes.append(cqt.shape)

        # Combine the CQTs together along dimension H
        hcqts = torch.cat(cqt_list, dim = 0)
        # Re-organize so we have B x H x F x T
        hcqts = hcqts.transpose(1, 0)
        return hcqts

class LVQT(torch.nn.Module):
    def __init__(self, fs = 22050, hop_length = 256, fmin = None, n_bins = 360,
                 bins_per_octave = 60, filter_scale = 1, gamma = 0, norm = 1,
                 window = 'hann', scale = True, norm_length = True, random = False,
                 max_p = 1, log = True, batch_norm = True, log_scale = True):
        super(LVQT, self).__init__()

        if fmin is None:
            # C1 by default
            fmin = librosa.note_to_hz('C1')

        # Make all parameters accessible
        self.fs = fs
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.filter_scale = filter_scale
        self.gamma = gamma
        self.norm = norm
        self.window = window
        self.scale = scale
        self.norm_length = norm_length
        self.random = random
        self.max_p = max_p
        self.log = log
        self.batch_norm = batch_norm
        self.log_scale = log_scale

        # Get complex bases with log center frequencies and their respective lengths
        self.basis, self.lengths = constant_q(sr=self.fs,
                                              fmin=self.fmin,
                                              n_bins=self.n_bins,
                                              bins_per_octave=self.bins_per_octave,
                                              window=self.window,
                                              filter_scale=self.filter_scale,
                                              pad_fft=False,
                                              norm=self.norm,
                                              gamma=self.gamma)

        # Get the real weights from the complex valued bases
        real_weights = np.real(self.basis)

        # Create the convolutional filterbank with the weights
        self.time_conv = torch.nn.Conv1d(1, n_bins, self.basis.shape[1],
                                         hop_length // max_p, self.basis.shape[1] // 2, bias = False)

        if random:
            # Weights are initialized randomly by default - but they must be normalized
            pass
        else:
            # If CQT/VQT initialization is desired, manually set the Conv1d parameters with the complex weights
            self.time_conv.weight = torch.nn.Parameter(torch.Tensor(real_weights).unsqueeze(1))

        # Initialize L2 pooling to recombine separate real/imag filter channels into complex coefficients.
        # while also dealing with bad gradients when the power is zero
        self.l2_pool = torch.nn.LPPool1d(norm_type = 2, kernel_size = 2, stride = 2)

        self.mp = torch.nn.MaxPool1d(max_p)
        self.bn = torch.nn.BatchNorm1d(n_bins)

    def forward(self, wav):
        missing = self.hop_length - (wav.size(-1) % self.hop_length)
        if (missing != self.hop_length):
            remaining = np.expand_dims(np.zeros(missing), 0)
            remaining = torch.Tensor([remaining] * wav.size(0)).to(wav.device)
            wav = torch.cat((wav, remaining), dim=-1)
        num_frames = (wav.size(-1) - 1) // self.hop_length + 1
        C_real = self.time_conv(wav)
        C_real_weights = self.time_conv.weight
        C_imag_weights = torch_hilbert(C_real_weights)[:, :, :, 1]
        C_imag = torch.nn.functional.conv1d(wav, C_imag_weights, None, self.hop_length // self.max_p, self.basis.shape[1] // 2)
        C = torch.cat((C_real.view(C_real.size(0), -1).unsqueeze(-1), C_imag.view(C_imag.size(0), -1).unsqueeze(-1)), dim = -1)
        C = self.l2_pool(C).view(C_real.size())

        if self.norm_length:
            # Compensate for different filter lengths of CQT/VQT
            C *= torch.Tensor(self.lengths[np.newaxis, :, np.newaxis]).to(wav.device)

        if self.scale:
            # Scale the CQT response by square-root the length of each channel’s filter
            C /= torch.sqrt(torch.Tensor(self.lengths)).unsqueeze(1).to(wav.device)

        C = self.mp(C)

        if self.log:
            C = take_log(C, scale = self.log_scale)

        if self.batch_norm:
            C = self.bn(C)

        return C[:, :, :num_frames]


def torch_hilbert(x, N = None):
    if N is None:
        N = x.size(-1)
    if N <= 0:
        raise ValueError("N must be positive.")

    zs = torch.zeros(x.size()).to(x.device)
    x = torch.cat((x.unsqueeze(-1), zs.unsqueeze(-1)), dim = -1)
    Xf = torch.fft(x, signal_ndim = 1)
    h = torch.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1 : N // 2] = 2
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2

    h = h.unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(x.device)
    x = torch.ifft(Xf * h, signal_ndim = 1)
    return x
