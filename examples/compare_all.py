# My imports
from lhvqt.lvqt_orig import LVQT as LVQT_C
from lhvqt.lvqt_hilb import LVQT as LVQT_H
from lhvqt.lvqt_real import LVQT as LVQT_R

from lhvqt import LHVQT

# Regular imports
from librosa.display import specshow
from time import time

import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch


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


def main():
    """
    Compare the VQT convolutional implementations with the Librosa implementation.
    """

    # Select parameters to use across all implementations
    n_bins = 216  # 6 octaves
    gamma = None  # default gamma
    hop_length = 512
    bins_per_octave = 36
    fmin = librosa.note_to_hz('C1')
    harmonics = [0.5, 1, 2, 3, 4, 5]

    # Load an example piece of audio
    y, sr = librosa.load(librosa.util.example_audio_file())

    # Calculate the HVQT using librosa
    lib_start = time()
    lib_hvqt = librosa_hvqt(y, harmonics, sr, hop_length, fmin, n_bins, bins_per_octave, gamma)
    print(f'Librosa time : {time() - lib_start}')

    # Set the device for the convolutional implementations
    device = 0
    device = torch.device(f'cuda:{device}'
                          if torch.cuda.is_available() else 'cpu')

    # Add a batch and channel dimension to the audio, and make it a tensor
    y = torch.Tensor([[y]]).to(device)

    # Construct the classic variant
    lhvqt_classic = LHVQT(lvqt=LVQT_C,
                          harmonics=harmonics,
                          fs=sr,
                          hop_length=hop_length,
                          fmin=fmin,
                          n_bins=n_bins,
                          bins_per_octave=bins_per_octave,
                          gamma=gamma,
                          to_db=False,
                          db_to_prob=False,
                          batch_norm=False).to(device)

    # Construct the Hilbert transform variant
    lhvqt_hilbert = LHVQT(lvqt=LVQT_H,
                          harmonics=harmonics,
                          fs=sr,
                          hop_length=hop_length,
                          fmin=fmin,
                          n_bins=n_bins,
                          bins_per_octave=bins_per_octave,
                          gamma=gamma,
                          to_db=False,
                          db_to_prob=False,
                          batch_norm=False).to(device)

    # Construct the real-only variant
    lhvqt_real = LHVQT(lvqt=LVQT_R,
                       harmonics=harmonics,
                       fs=sr,
                       hop_length=hop_length,
                       fmin=fmin,
                       n_bins=n_bins,
                       bins_per_octave=bins_per_octave,
                       gamma=gamma,
                       to_db=False,
                       db_to_prob=False,
                       batch_norm=False).to(device)

    # Compute the response from the classic variant
    cla_start = time()
    cla_hvqt = lhvqt_classic(y)
    print(f'Classic time : {time() - cla_start}')

    # Compute the response from the hilbert variant
    hil_start = time()
    hil_hvqt = lhvqt_hilbert(y)
    print(f'Hilbert time : {time() - hil_start}')

    # Compute the response from the real variant
    rea_start = time()
    rea_hvqt = lhvqt_real(y)
    print(f'Real time : {time() - rea_start}')

    # Print a new line
    print()

    # Remove from the device and convert back to ndarray
    cla_hvqt = cla_hvqt.squeeze(0).cpu().detach().numpy()
    hil_hvqt = hil_hvqt.squeeze(0).cpu().detach().numpy()
    rea_hvqt = rea_hvqt.squeeze(0).cpu().detach().numpy()

    # Convert all HVQTs to decibels - done out here as not to factor into timing
    lib_hvqt = librosa.amplitude_to_db(lib_hvqt, ref=np.max)
    cla_hvqt = librosa.amplitude_to_db(cla_hvqt, ref=np.max)
    hil_hvqt = librosa.amplitude_to_db(hil_hvqt, ref=np.max)
    rea_hvqt = librosa.amplitude_to_db(rea_hvqt, ref=np.max)

    # Compute similarities
    print('HVQT Similariy (Classic): %1.2f%%' % (100 * hvqt_similarity(cla_hvqt, lib_hvqt)))
    print('HVQT Similariy (Hilbert): %1.2f%%' % (100 * hvqt_similarity(hil_hvqt, lib_hvqt)))
    # TODO - fix dimensionality mismatch
    #print('HVQT Similariy (Real): %1.2f %%' % 100 * hvqt_similarity(rea_hvqt, lib_hvqt))

    # Plot the nth harmonic of each transform
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # Index of harmonic within list to plot
    h_idx = 0

    plt.sca(ax1)
    specshow(lib_hvqt[h_idx],
             sr=sr,
             hop_length=hop_length,
             fmin=fmin,
             bins_per_octave=bins_per_octave,
             x_axis='time',
             y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    ax1.set_title('Librosa HVQT')

    plt.sca(ax2)
    specshow(cla_hvqt[h_idx],
             sr=sr,
             hop_length=hop_length,
             fmin=fmin,
             bins_per_octave=bins_per_octave,
             x_axis='time',
             y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    ax2.set_title(f'Classic HVQT (h = {harmonics[h_idx]})')

    plt.sca(ax3)
    specshow(hil_hvqt[h_idx],
             sr=sr,
             hop_length=hop_length,
             fmin=fmin,
             bins_per_octave=bins_per_octave,
             x_axis='time',
             y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    ax3.set_title(f'Hilbert HVQT (h = {harmonics[h_idx]})')

    plt.sca(ax4)
    specshow(rea_hvqt[h_idx],
             sr=sr,
             hop_length=hop_length,
             fmin=fmin,
             bins_per_octave=bins_per_octave,
             x_axis='time',
             y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    ax4.set_title(f'Real HVQT (h = {harmonics[h_idx]})')

    # Resize and show the figure
    fig.set_size_inches(16, 8)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
