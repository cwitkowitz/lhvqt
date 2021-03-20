# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
# None of my imports used

# Regular imports
import numpy as np
import librosa


def hvqt_similarity(a, b):
    """
    Compute the average cosine similarity measure
    across the harmonic dimension of two HVQTs.
    HVQTs must be of the same shape.

    Parameters
    ----------
    a : ndarray (H x F x T)
      First HVQT to compare
      H - number of harmonics
      F - number of frequency bins
      T - number of time steps
    b : ndarray (H x F x T)
      Second HVQT to compare
      H - number of harmonics
      F - number of frequency bins
      T - number of time steps

    Returns
    ----------
    hvqt_sim : float
      HVQT similarity measurement
    """

    assert len(a.shape) == 3

    # Compute the cosine similarity for each harmonic
    hvqt_sim = [cosine_similarity(a[i], b[i]) for i in range(a.shape[0])]
    # Average across harmonic dimension
    hvqt_sim = np.mean(hvqt_sim)

    return hvqt_sim


def cosine_similarity(a, b):
    """
    Compute the cosine similarity measure for two matrices.
    Matrices must be of the same shape.

    Parameters
    ----------
    a : ndarray (R x C)
      First matrix to compare
      R - number of rows
      C - number of columns
    b : ndarray (R x C)
      Second matrix to compare
      R - number of rows
      C - number of columns

    Returns
    ----------
    cos_sim : float
      Cosine similarity measurement
    """

    assert len(a.shape) == 2
    assert a.shape == b.shape

    # Compute the dot product of matrix a and b as if they were vectors
    ab_dot = np.trace(np.dot(a.T, b))
    # Compute the norms of each matrix
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    # Compute cosine similarity
    cos_sim = ab_dot / (a_norm * b_norm)

    return cos_sim


def librosa_hvqt(audio, harmonics, sample_rate, hop_length, fmin, n_bins, bins_per_octave, gamma):
    """
    Compute an HVQT using Librosa.

    Parameters
    ----------
    audio : ndarray (N)
      Audio to transform,
      N - number of samples
    harmonics : list of ints
      Specific harmonics to stack across the harmonic dimension
    sample_rate : int or float
      Number of samples per second of audio
    hop_length : int
      Number of samples between frames
    fmin : float
      Lowest center frequency in basis
    n_bins : int
      Number of basis functions in the filterbank
    bins_per_octave : int
      Number of basis functions per octave
    gamma : float
      Bandwidth offset to smoothly vary Q-factor

    Returns
    ----------
    hvqt : ndarray (H x F x T)
      Harmonic Variable-Q Transform (HVQT) for the provided audio,
      H - number of harmonics
      F - number of bins
      T - number of time steps (frames)
    """

    # Initialize a list to hold the harmonic-wise transforms
    hvqt = list()

    # Initialize a list to hold the number of frames for each transform
    frames = list()

    # Loop through harmonics
    for h in range(len(harmonics)):
        # Compute the true minimum center frequency for this harmonic
        h_fmin = harmonics[h] * fmin

        # Compute the VQT for this harmonic
        vqt = librosa.vqt(audio, sr=sample_rate, hop_length=hop_length, fmin=h_fmin,
                          n_bins=n_bins, gamma=gamma, bins_per_octave=bins_per_octave)

        # Keep track of the number of frames produced
        frames.append(vqt.shape[-1])

        # Add the VQT to the collection
        hvqt.append(np.expand_dims(vqt, axis=0))

    # Determine the maximum number of frames that can be concatenated
    max_frames = min(frames)

    # Perform any trimming and concatenate
    hvqt = np.concatenate([vqt[..., :max_frames] for vqt in hvqt])

    # Take the magnitude
    hvqt = np.abs(hvqt)

    return hvqt
