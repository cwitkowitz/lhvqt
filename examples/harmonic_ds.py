from librosa.display import specshow

import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch

from lhvqt.lhvqt_ds import *
from lhvqt.lhvqt import *


def similarity(A, B):
    return np.trace(np.dot(A.T, B)) / (np.linalg.norm(A) * np.linalg.norm(B))


def main():
    y, sr = librosa.load(librosa.util.example_audio_file())


    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.sca(ax1)
    V1 = LHVQT(fs=sr,
               harmonics=[1, 2, 3, 4, 5],
               n_bins=24,
               bins_per_octave=24,
               db_to_prob=False,
               batch_norm=False)(torch.Tensor([[y]]))
    V1 = V1.view(5*24, -1).cpu().detach().numpy()
    specshow(V1, sr=sr, x_axis='time', y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    ax1.set_title('Standard HVQT')

    plt.sca(ax2)
    V2 = LHVQT_DS(fs=sr,
                  harmonics=[1, 2, 3, 4, 5],
                  n_bins=24,
                  bins_per_octave=24,
                  db_to_prob=False,
                  batch_norm=False)(torch.Tensor([[y]]))
    V2 = V2.view(5*24, -1).cpu().detach().numpy()
    specshow(V2, sr=sr, x_axis='time', y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    ax2.set_title('Harmonic Downsampler')

    fig.set_size_inches(15, 6)
    fig.tight_layout()
    plt.show()

    print('Similarity: %1.4f' % similarity(V1, V2))


if __name__ == '__main__':
    main()
