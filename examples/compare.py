from librosa.display import specshow

import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch

from lhvqt import *

def similarity(A, B):
    return np.trace(np.dot(A.T, B)) / (np.linalg.norm(A) * np.linalg.norm(B))

def main():
    y, sr = librosa.load(librosa.util.example_audio_file())
    V = np.abs(librosa.vqt(y, sr=sr, gamma=5, hop_length=256, n_bins = 60))
    V = librosa.amplitude_to_db(V, ref=np.max)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.sca(ax1)
    specshow(V, sr=sr, x_axis='time', y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    ax1.set_title('Variable-Q power spectrum (Librosa)')

    plt.sca(ax2)
    vqt = LVQT(sr, gamma=5, batch_norm=False, log_scale=False,
               bins_per_octave = 12, n_bins = 60)(torch.Tensor([[y]]))
    vqt = vqt[0].cpu().detach().numpy()
    specshow(vqt, sr=sr, x_axis='time', y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    ax2.set_title('Variable-Q power spectrum (LHVQT)')

    fig.tight_layout()

    plt.show()

    print('Similarity: %1.4f' % similarity(V, vqt))

if __name__ == '__main__':
    main()
