# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# Regular imports
import numpy as np
import librosa
import torch

EPSILON = torch.finfo(torch.float32).eps


def torch_amplitude_to_db(feats, amin=1e-5, top_db=80.0, to_prob=False):
    """
    Convert amplitude features to decibels or probability-like
    values sample-wise with differentiable PyTorch functions.

    Adapted from the Librosa amplitude_to_db function.

    Parameters
    ----------
    feats : Tensor (B x H x F x T) or (B x F x T)
      Set of magnitude of amplitude features,
      B - batch size
      H - number of harmonics (a.k.a. channels)
      F - dimensionality of features
      T - number of time steps (frames)
    amin : float
      Threshold for minimum amplitude
    top_db : float
      Maximum difference from dB ceiling
    to_prob : bool
      Convert decibels to probability-like features

    Returns
    ----------
    decibels : Tensor (B x H x F x T) or (B x F x T)
      Set of decibel or probability-like features
    """

    # Determine the number of samples in the batch
    B = feats.size(0)

    # Make sure the features represent magnitude
    magnitude = torch.abs(feats.clone())

    # Convert magnitude to power
    power = torch.square(magnitude)

    # Get reference values for each sample in the batch
    ref_values = power.reshape(B, -1).max(dim=-1)[0]

    # Add dimensions to reference values for broadcasting
    ref_values = ref_values.view((-1,) + tuple([1] * (len(feats.shape) - 1)))

    # Convert threshold to power
    amin = torch.tensor(amin) ** 2

    # Convert power to dB, clamping power at specified threshold
    log_feats = 10.0 * torch.log10(torch.maximum(amin, power))

    # Combine with previous operation to perform 10 * log10(power / ref)
    log_feats -= 10.0 * torch.log10(torch.maximum(amin, ref_values))

    # Clamp all decibels at the corresponding floor
    decibels = torch.maximum(log_feats, -torch.tensor(top_db))

    # Make sure features for silence are at decibel floor
    decibels[ref_values.squeeze() == 0.] = -top_db

    if to_prob:
        # Scale values and offset to be between 0 and 1
        decibels = (decibels / top_db) + 1.0

    return decibels


def torch_hilbert(x_real, n_fft=None):
    """
    Obtain imaginary counterpart to a real signal such that there are no negative frequency
    components when represented as a complex signal. This is done by using the Hilbert transform.
    We end up with an analytic signal and return only the imaginary part. Most importantly,
    this procedure is fully differentiable. Adapted from the SciPy signal.hilbert function.

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

    # Take the Fourier transform of the real part
    Xf = torch.fft.fft(x_real, n=n_fft, dim=-1)
    # Apply the transfer function to the Fourier transform
    Xfh = Xf * h.unsqueeze(-2)
    # Take the inverse Fourier Transform to obtain the analytic signal
    x_alyt = torch.fft.ifft(Xfh, dim=-1)
    # Take the imaginary part of the analytic signal to obtain the Hilbert transform
    x_imag = x_alyt.imag

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
