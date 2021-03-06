# My imports
from lhvqt.lvqt_orig import LVQT

from lhvqt import LHVQT

# Regular imports
import librosa
import os


def main():
    """
    Simple visualization example for all variants.
    """

    # Construct the path to the directory for saving images
    save_dir = os.path.join('..', '..', 'generated', 'classic')
    os.makedirs(save_dir, exist_ok=True)

    # Select parameters to use across all variants
    sr = 22050
    n_bins = 216  # 6 octaves
    gamma = None  # default gamma
    hop_length = 512
    bins_per_octave = 36
    fmin = librosa.note_to_hz('C1')
    harmonics = [1]

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

    lhvqt_classic.visualize(save_dir, fix_scale=True, include_axis=True, decibels=True, include_negative=True)


if __name__ == '__main__':
    main()
