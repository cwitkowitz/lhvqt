from librosa.display import specshow

import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch

from lhvqt.lvqt_orig import *


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


def main():
    y, sr = librosa.load(librosa.util.example_audio_file())
    V = np.abs(librosa.vqt(y, sr=sr, gamma=5, hop_length=256, n_bins=60))
    V = librosa.amplitude_to_db(V, ref=np.max)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.sca(ax1)
    specshow(V, sr=sr, x_axis='time', y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    ax1.set_title('Variable-Q power spectrum (Librosa)')

    plt.sca(ax2)
    vqt = LVQT(fs=sr, gamma=5, db_to_prob=False, batch_norm=False,
               bins_per_octave=12, n_bins=60)(torch.Tensor([[y]]))
    vqt = vqt[0].cpu().detach().numpy()
    specshow(vqt, sr=sr, x_axis='time', y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    ax2.set_title('Variable-Q power spectrum (LHVQT)')

    fig.set_size_inches(15, 6)
    fig.tight_layout()
    plt.show()

    print('Similarity: %1.4f' % cosine_similarity(V, vqt))


if __name__ == '__main__':
    main()
