from .utils import *

from librosa.filters import constant_q
import numpy as np
import librosa
import torch


class LHVQT(torch.nn.Module):
    """
    Implements a slight adaptation/modernization of the original module
    presented in my Master's Thesis (https://scholarworks.rit.edu/theses/10143/).
    """

    def __init__(self, fs=22050, harmonics=[0.5, 1, 2, 3, 4, 5], hop_length=256, fmin=None,
                 n_bins=360, bins_per_octave=60, filter_scale=1, gamma=0, norm=1, window='hann',
                 scale=True, norm_length=True, random=False, max_p=1):
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

        # Add VQTs as elements of a Torch module
        self.tfs = torch.nn.Module()
        for h_num in range(len(harmonics)):
            h = harmonics[h_num]
            mod_name = 'lvqt%d' % h_num # Name according to index of harmonic
            self.tfs.add_module(mod_name,
                                LVQT(fs=self.fs,
                                     hop_length=self.hop_length,
                                     fmin=self.fmin * float(h),
                                     n_bins=self.n_bins,
                                     bins_per_octave=self.bins_per_octave,
                                     filter_scale=self.filter_scale,
                                     gamma=self.gamma,
                                     norm=self.norm,
                                     window=self.window,
                                     scale=self.scale,
                                     norm_length=self.norm_length,
                                     random=self.random,
                                     max_p=self.max_p))

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

    def norm_weights(self):
        for h in range(len(self.harmonics)):
            # Normalize the weights at each harmonic
            torch.nn.Sequential(*list(self.tfs.children()))[h].norm_weights()


class LVQT(torch.nn.Module):
    def __init__(self, fs=22050, hop_length=256, fmin=None, n_bins=360, bins_per_octave=60,
                 filter_scale=1, gamma=0, norm=1, window='hann', scale=True, norm_length=True,
                 random=False, max_p=1):
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

        # Get complex bases with log-spaced center frequencies and their respective lengths
        basis, self.lengths = constant_q(sr=fs,
                                         fmin=fmin,
                                         n_bins=n_bins,
                                         bins_per_octave=bins_per_octave,
                                         window=window,
                                         filter_scale=filter_scale,
                                         gamma=gamma,
                                         pad_fft=False,
                                         norm=norm)

        # Create the convolutional filterbank with the weights
        self.time_conv = torch.nn.Conv1d(1, 2 * n_bins, basis.shape[1], hop_length // max_p,
                                         basis.shape[1] // 2, bias = False)

        if random:
            # Weights are initialized randomly by default - but they must be normalized
            self.norm_weights()
        else:
            # Split the complex valued bases into real and imaginary weights
            real_weights, imag_weights = np.real(basis), np.imag(basis)
            # Stack them together into one representation
            complex_weights = np.array([[real_weights[i]] + [imag_weights[i]] for i in range(n_bins)])
            complex_weights = np.reshape(complex_weights, (2 * n_bins, basis.shape[1]))
            # If CQT/VQT initialization is desired, manually set the Conv1d parameters with the complex weights
            self.time_conv.weight = torch.nn.Parameter(torch.Tensor(complex_weights).unsqueeze(1))

        # Initialize L2 pooling to recombine separate real/imag filter channels into complex coefficients
        self.l2_pool = torch.nn.LPPool1d(norm_type = 2, kernel_size = 2, stride = 2)

        self.mp = torch.nn.MaxPool1d(max_p)
        self.bn = torch.nn.BatchNorm1d(n_bins)

    def forward(self, wav):
        missing = self.hop_length - (wav.size(-1) % self.hop_length)
        if (missing != self.hop_length):
            wav = torch.cat((wav, torch.zeros(missing).unsqueeze(0).unsqueeze(0).to(wav.device)), dim=-1)
        C = self.time_conv(wav)
        C = self.l2_pool(C.transpose(1, 2)).transpose(1, 2)

        if self.norm_length:
            # Compensate for different filter lengths of CQT/VQT
            C *= torch.Tensor(self.lengths[np.newaxis, :, np.newaxis]).to(wav.device)

        if self.scale:
            # Scale the CQT response by square-root the length of each channel’s filter
            C /= torch.sqrt(torch.Tensor(self.lengths)).unsqueeze(1).to(wav.device)

        C = take_log(self.mp(C), scale=True)

        if C.size(-1) > 1:
            C = self.bn(C)

        return C

    def norm_weights(self):
        with torch.no_grad():
            complex_weights = self.time_conv.weight.view(self.n_bins, 2, -1)
            abs_weights = torch.sqrt(complex_weights[:, 0] ** 2 + complex_weights[:, 1] ** 2)
            divisor = torch.sum(abs_weights, dim = 1).unsqueeze(1).repeat(2, 1, abs_weights.size(1)).transpose(0, 1)
            norm_weights = complex_weights / divisor
            self.time_conv.weight = torch.nn.Parameter(norm_weights.view(2 * self.n_bins, 1, -1))
