# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# Regular imports
import numpy as np
import librosa
import torch

EPSILON = torch.finfo(torch.float32).eps


def torch_amplitude_to_db(feats, amin=1e-10, top_db=80.0, to_prob=False):
    """
    Take the log of a time-frequency representation with differentiable
    PyTorch functions, scaling values to be between 0 and 1. Adapted from
    the Librosa amplitude_to_db function.

    Parameters
    ----------
    feats : Tensor (B x H x F x T) or (B x F x T)
      Set of amplitude features,
      B - batch size
      H - number of harmonics (a.k.a. channels)
      F - dimensionality of features
      T - number of time steps (frames)
    amin : float
      Minimum amplitude possible within a time-frequency bin
    top_db : float
      Maximum difference from dB ceiling (defines dB floor)
    to_prob : bool
      Scale decibel values to be between 0 and 1

    Returns
    ----------
    log_feats : Tensor (B x H x F x T) or (B x F x T)
      Set of dB or probability features
    """

    # Perform processing on a copy of original features
    feats_copy = feats.clone()

    # Get handles for the feature dimensionality
    B, H, F, T = feats.size(0), 1, feats.size(-2), feats.size(-1)

    # Collapse the channel dimension if it exists
    if len(feats.size()) > 3:
        H = feats.size(1)
        feats_copy = feats_copy.view(B * H, F, T)

    # Get handle for pseudo batch size
    PB = B * H

    # Convert amplitude features to power
    power = torch.abs(feats_copy) ** 2

    # Get reference values (maximums) from power spectrogram for each 2D transform
    ref_value = torch.max(power.view(PB, -1), dim=-1)[0]

    # Clamp power spectrogram at specified minimum - effectively max(amin, power)
    power[power < amin] = amin

    # Convert power to dB
    log_feats = 10.0 * torch.log10(power)

    # Make sure reference values are above minimum amplitude
    amin = torch.Tensor([amin] * PB).to(power.device)
    amin[amin < ref_value] = ref_value[amin < ref_value]
    # Add dimensions for broadcasting
    amin = amin.unsqueeze(-1).unsqueeze(-1)

    # Combined with previous log, we are performing 10 * log10(power / ref)
    log_feats = log_feats - 10.0 * torch.log10(amin)

    # Collapse the last two dimensions temporarily to make the following easier
    log_feats = log_feats.view(PB, -1)
    # Determine the dB floor for each 2D transform
    dB_floor = (torch.max(log_feats, dim=-1)[0] - top_db).unsqueeze(-1)
    # Clamp the values at the specified dB floor
    log_feats[log_feats < dB_floor] = dB_floor.repeat(1, F * T)[log_feats < dB_floor]

    if to_prob:
        # Scale values and offset to be between 0 and 1
        log_feats = (log_feats / top_db) + 1.0

    # Reshape the log-scaled features using the original feature dimensions
    log_feats = log_feats.view(feats.size())

    return log_feats


def torch_hilbert(x_real, n_fft=None):
    """
    Obtain imaginary counterpart to a real signal such that there are no negative frequency
    components when represented as a complex signal. This is done by using the Hilbert transform.
    We end up with an analytic signal and return only the imaginary part. Most importantly,
    this procedure is fully differentiable. Adapted from the SciPy signal.hilbert function.

    TODO - seems to be a tiny bit of energy in negative frequencies - numerical stability?

    Parameters
    ----------
    x_real : Tensor (F x T)
      Real counterpart of an analytic signal,
      F - number of independent signals
      T - number of time steps (samples)
    n_fft : int
      Number of Fourier components

    Returns
    ----------
    x_imag : Tensor (F x T)
      Imaginary counterpart of an analytic signal,
      F - number of independent signals
      T - number of time steps (samples)
    """

    # Default to the length of the input signal
    if n_fft is None:
        n_fft = x_real.size(-1)

    # Create the transfer function for an analytic signal
    h = torch.zeros(n_fft).to(x_real.device)
    if n_fft % 2 == 0:
        h[0] = h[n_fft // 2] = 1
        h[1 : n_fft // 2] = 2
    else:
        h[0] = 1
        h[1 : (n_fft + 1) // 2] = 2

    # TODO - zero-pad x_real to n_fft or use new fft.rfft functions when released
    # Take the Fourier transform of the real part
    Xf = torch.rfft(x_real, signal_ndim=1, onesided=False)
    # Apply the transfer function to the Fourier transform
    Xfh = Xf * h.unsqueeze(-1)
    # Take the inverse Fourier Transform to obtain the analytic signal
    x_alyt = torch.ifft(Xfh, signal_ndim=1)
    # Take the imaginary part of the analytic signal to obtain the Hilbert transform
    x_imag = x_alyt[..., -1]

    return x_imag


def fft_response(signal, sample_rate, n_fft=None, decibels=False):
    """
    Obtain a signal's FFT response in a plot-friendly format.

    Parameters
    ----------
    signal : ndarray
      Signal to transform
    sample_rate : int
      Number of samples per second
    n_fft : int or None (optional)
      See np.fft.fft documentation...
    decibels : bool
      Whether to convert to dB or leave as amplitude

    Returns
    ----------
    frequencies : ndarray
      Ordered frequencies corresponding to each response
    response : ndarray
      Magnitude response of signal for each basis
    """

    # Take the FFT and calculate the magnitude of the response
    response = np.abs(np.fft.fft(signal, n=n_fft))

    if decibels:
        # Convert the frequency response from amplitude to decibels
        response = librosa.amplitude_to_db(response, ref=np.max)

    # Determine the number of bins in the response
    num_bins = response.shape[-1]

    # Get the frequencies corresponding to the FFT indices
    frequencies = np.fft.fftfreq(num_bins, (1 / sample_rate))

    # Re-order the FFT and frequencies so they go from most negative to most positive
    response = np.roll(response, response.shape[-1] // 2, axis=-1)
    frequencies = np.roll(frequencies, frequencies.shape[-1] // 2)

    return frequencies, response


def filter_kwargs(keywords, **kwargs):
    """
    Filter provided keyword arguments and remove all but
    those matching a provided list of relevant keywords.

    Parameters
    ----------
    keywords : list of string
      Keywords to search for within the provided keyword arguments
    kwargs : dict
      Dictionary of provided keywords

    Returns
    ----------
    filtered_kwargs : dict
      Keyword arguments matching provided keyword list entries
    """

    # Create an empty dictionary for keyword matches
    filtered_kwargs = dict()

    # Loop through relevant keywords
    for key in keywords:
        # Check if there is a match
        if key in kwargs:
            # Add the keyword argument to the matches dictionary
            filtered_kwargs[key] = kwargs[key]

    return filtered_kwargs


def tensor_to_array(tensor):
    """
    Simple helper function to convert a PyTorch tensor
    into a NumPy array in order to keep code readable.

    Parameters
    ----------
    tensor : PyTorch tensor
      Tensor to convert to array

    Returns
    ----------
    array : NumPy ndarray
      Converted array
    """

    # Change device to CPU,
    # detach from gradient graph,
    # and convert to NumPy array
    array = tensor.cpu().detach().numpy()

    return array


def array_to_tensor(array, device=None):
    """
    Simple helper function to convert a NumPy array
    into a PyTorch tensor in order to keep code readable.

    Parameters
    ----------
    array : NumPy ndarray
      Array to convert to tensor
    device : string, or None (optional)
      Add tensor to this device, if specified

    Returns
    ----------
    tensor : PyTorch tensor
      Converted tensor
    """

    # Convert to PyTorch tensor
    tensor = torch.from_numpy(array)

    # Add tensor to device, if specified
    if device is not None:
        tensor = tensor.to(device)

    return tensor
