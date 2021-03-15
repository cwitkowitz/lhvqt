# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from lhvqt.lvqt_real import LVQT

from lhvqt import LHVQT

# Regular imports
import librosa
import os


def main():
    """
    Simple visualization example for real variant.
    """

    # Construct the path to the directory for saving images
    save_dir = os.path.join('..', '..', 'generated', 'visualization', 'real')
    os.makedirs(save_dir, exist_ok=True)

    # Select parameters to use
    sr = 22050
    n_bins = 192  # 8 octaves
    gamma = None  # default gamma
    hop_length = 512
    bins_per_octave = 24
    fmin = librosa.note_to_hz('C1')
    harmonics = [1]

    # Construct the real variant
    lhvqt_real = LHVQT(lvqt=LVQT,
                       harmonics=harmonics,
                       fs=sr,
                       hop_length=hop_length,
                       fmin=fmin,
                       n_bins=n_bins,
                       bins_per_octave=bins_per_octave,
                       gamma=gamma,
                       to_db=False,
                       db_to_prob=False,
                       batch_norm=False)

    # Visualize the real variant
    lhvqt_real.visualize(save_dir,
                         idcs=[0, 45, 90, 135, 180],
                         fix_scale=True,
                         include_axis=True,
                         n_fft=None,
                         scale_freqs=True,
                         decibels=True,
                         include_negative=True,
                         separate=False,
                         sort_by_centroid=True)


if __name__ == '__main__':
    main()
