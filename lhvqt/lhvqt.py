import numpy as np
import librosa
import torch

class LHVQT(torch.nn.Module):
    def __init__(self, fs = 22050, harmonics = [0.5, 1, 2, 3, 4, 5], hop_length = 256, fmin = None,
                       n_bins = 360, bins_per_octave = 60, filter_scale = 1, gamma = 0, norm = 1,
                       window = 'hann', scale = True, norm_length = True, random = False, max_p = 1):
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

        # Add CQTs as elements of a Torch module
        self.tfs = torch.nn.Module()
        for h_num in range(len(harmonics)):
            h = harmonics[h_num]
            mod_name = 'lcqt%d' % h_num # Name according to index of harmonic
            self.tfs.add_module(mod_name,
                                LVQT(fs, hop_length, fmin * float(h), n_bins,
                                     bins_per_octave, filter_scale, gamma, norm,
                                     window, scale, norm_length, random, max_p))

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
                 window = 'hann', scale = True, norm_length = True, random = False, max_p = 1):
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

        # Get complex bases with log center frequencies and their respective lengths
        self.basis, self.lengths = variable_q(fs, fmin, n_bins, bins_per_octave, 0,
                                         window, filter_scale, gamma, False, norm)

        # Get the real weights from the complex valued bases
        real_weights = np.real(self.basis)

        # Create the convolutional filterbank with the weights
        self.time_conv = torch.nn.Conv1d(1, n_bins, self.basis.shape[1], hop_length // max_p, self.basis.shape[1] // 2, bias = False)

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

        C = self.bn(take_log(self.mp(C)))
        return C[:, :, :num_frames]

"""
This was adapted from the Librosa contant_q function. The only new parameter
is gamma, which is passed to variable_q_lengths. Setting gamma = 0 will result
in constant-q filters, whereas positive gamma will slowly decrease Q-factor at
lower frequencies.
"""
def variable_q(sr, fmin = None, n_bins = 84, bins_per_octave = 12, tuning = 0.0,
               window = 'hann', filter_scale = 1, gamma = 0, pad_fft = True, norm = 1,
               dtype = np.complex64, **kwargs):

    if fmin is None:
        fmin = librosa.note_to_hz('C1')

    # Pass-through parameters to get the filter lengths (assuming constant-q)
    lengths = variable_q_lengths(sr, fmin, n_bins = n_bins, bins_per_octave = bins_per_octave,
                                 tuning = tuning, window = window, filter_scale = filter_scale,
                                 gamma = 0)

    # Apply tuning correction
    correction = 2.0 ** (float(tuning) / bins_per_octave)
    fmin = correction * fmin

    # Q should be capitalized here, so we suppress the name warning
    # pylint: disable=invalid-name
    Q = float(filter_scale) / (2.0 ** (1. / bins_per_octave) - 1)

    # Convert lengths back to frequencies (constant-q)
    freqs = Q * sr / lengths

    # Recalculate the lengths in case gamma is nonzero (variable-q)
    lengths = variable_q_lengths(sr, fmin, n_bins = n_bins, bins_per_octave = bins_per_octave,
                                 tuning = tuning, window = window, filter_scale = filter_scale,
                                 gamma = gamma)

    # Build the filters
    filters = []
    for ilen, freq in zip(lengths, freqs):
        # Build the filter: note, length will be ceil(ilen)
        sig = np.exp(np.arange(-ilen // 2, ilen // 2, dtype = float) * 1j * 2 * np.pi * freq / sr)

        # Apply the windowing function
        sig = sig * librosa.filters.__float_window(window)(len(sig))

        # Normalize
        sig = librosa.util.normalize(sig, norm = norm)

        filters.append(sig)

    # Pad and stack
    max_len = max(lengths)
    if pad_fft:
        max_len = int(2.0 ** (np.ceil(np.log2(max_len))))
    else:
        max_len = int(np.ceil(max_len))

    filters = np.asarray([librosa.util.pad_center(filt, max_len, **kwargs) for filt in filters], dtype = dtype)

    return filters, np.asarray(lengths)

"""
This was adapted from the Librosa contant_q_lengths function. The only new
parameter is gamma, which varies the Q-factor at lower frequencies. In this
context, Q is a vector used to calculate the filter length of each channel.
It is important to note that the center frequencies of each channel still
align with a constant Q-factor. That is, they are logarithmically spaced
by a factor of  (2.0 ** (1 / bins_per_octave)).
"""
def variable_q_lengths(sr, fmin, n_bins = 84, bins_per_octave = 12,
                       tuning = 0.0, window = 'hann', filter_scale = 1,
                       gamma = 0):
    if fmin <= 0:
        raise Exception('fmin must be positive')
    if gamma < 0:
        raise Exception('gamma must be positive')
    if bins_per_octave <= 0:
        raise Exception('bins_per_octave must be positive')
    if filter_scale <= 0:
        raise Exception('filter_scale must be positive')
    if n_bins <= 0 or not isinstance(n_bins, int):
        raise Exception('n_bins must be a positive integer')

    correction = 2.0 ** (float(tuning) / bins_per_octave)

    fmin = correction * fmin

    # Calculate the constant Q-factor
    Q = float(filter_scale) / (2.0 ** (1. / bins_per_octave) - 1)

    # Compute the constant-q frequencies
    freq = fmin * (2.0 ** (np.arange(n_bins, dtype = float) / bins_per_octave))

    # Determine if support of maximum center frequency is possible
    if freq[-1] * (1 + 0.5 * librosa.filters.window_bandwidth(window) / Q) > sr / 2.0:
        raise Exception('Filter pass-band lies beyond Nyquist')

    # Calculate the Q factor of each frequency channel
    Q = freq * float(filter_scale) / ((2.0 ** (1. / bins_per_octave) - 1) * freq + gamma)

    # Convert frequencies to filter lengths
    lengths = Q * sr / freq

    return lengths

"""
Take the log of a time-frequency representation using differentiable
PyTorch functions, scaling values to be between 0 and 1. Adapted from
the Librosa amplitude_to_db function.

Expected input: batch x harmonic x frequency x time
                              OR
                batch x frequency x time
"""
def take_log(tfr, amin = (1e-5) ** 2, top_db = 80.0):
    if len(tfr.size()) != 4 and len(tfr.size()) != 3:
        raise Exception('Expected input - B x H x F x T OR B x F x T')

    batch_size = tfr.size(0)

    if len(tfr.size()) == 4:
        # View input as B x F x T
        h_size = tfr.size(1)
        tfr = tfr.contiguous().view(batch_size * h_size, tfr.size(2), tfr.size(3))
    else:
        h_size = None

    # Convert amplitude to power spectrogram
    magnitude = torch.abs(tfr)
    power = magnitude * magnitude

    num_tfrs = tfr.size(0)
    # Get reference values (maximums) from power spectrogram for each tfr
    ref_value = torch.max(magnitude.view(num_tfrs, -1), dim = -1)[0] ** 2

    # Clamp power spectrogram at specified minimum - effectively max(amin, power)
    power[power < amin] = amin

    # Convert to dB
    log_tfr = 10.0 * torch.log10(power)

    # Make sure reference values are above minimum amplitude - effectively max(amin, ref_value)
    amin = torch.Tensor([amin] * num_tfrs).to(power.device)
    amin[amin < ref_value] = ref_value[amin < ref_value]
    maxes = amin.unsqueeze(-1).unsqueeze(-1).to(power.device) # Add dimensions to broadcast over tfr

    # Combined with previous log, we are performing 10 * log10(power / ref)
    log_tfr = log_tfr - 10.0 * torch.log10(maxes)

    # Clamp the dB values at the specified top - effectively max(log_tfr, log_tfr.max() - top_db)
    log_tfr = log_tfr.view(num_tfrs, -1) # Combine F and T dimension temporarily
    newVals = (torch.max(log_tfr, dim = -1)[0] - top_db).unsqueeze(-1).to(power.device)
    log_tfr[log_tfr < newVals] = newVals.repeat(1, log_tfr.size(-1))[log_tfr < newVals]

    # Scale values and offset to be between 0 and 1
    tfr = (log_tfr / top_db) + 1.0
    tfr = tfr.view(magnitude.size())

    if h_size is not None:
        # Match input dimensions to original HCQT
        tfr = tfr.contiguous().view(batch_size, h_size, tfr.size(1), tfr.size(2))

    return tfr

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
