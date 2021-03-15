# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from lhvqt.lvqt_orig import LVQT

from lhvqt import LHVQT

# Regular imports
import librosa
import os


def main():
    """
    Simple visualization example for classic variant.
    """

    # Construct the path to the directory for saving images
    save_dir = os.path.join('..', '..', 'generated', 'visualization', 'classic')
    os.makedirs(save_dir, exist_ok=True)

    # Select parameters to use
    sr = 22050
    n_bins = 96  # 8 octaves
    gamma = None  # default gamma
    hop_length = 512
    bins_per_octave = 12
    fmin = librosa.note_to_hz('C1')
    harmonics = [0.5, 1, 2, 3, 4, 5]

    # Construct the classic variant
    lhvqt_classic = LHVQT(lvqt=LVQT,
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

    # Visualize the classic variant
    lhvqt_classic.visualize(save_dir,
                            idcs=None,
                            fix_scale=True,
                            include_axis=True,
                            n_fft=None,
                            scale_freqs=False,
                            decibels=True,
                            include_negative=False,
                            separate=True,
                            sort_by_centroid=True)


if __name__ == '__main__':
    main()
