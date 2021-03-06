# My imports
from lhvqt.lvqt_orig import LVQT as LVQT_C
from lhvqt.lvqt_hilb import LVQT as LVQT_H
from lhvqt.lvqt_real import LVQT as LVQT_R

from lhvqt import LHVQT, LHVQT_COMB

# Regular imports
import librosa


def main():
    """
    Simple visualization example for all variants.
    """

    # Select parameters to use across all variants
    sr = 22050
    n_bins = 216  # 6 octaves
    gamma = None  # default gamma
    hop_length = 512
    bins_per_octave = 36
    fmin = librosa.note_to_hz('C1')
    harmonics = [0.5, 1, 2, 3, 4, 5]

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
                          batch_norm=False)

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
                          batch_norm=False)

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
                       batch_norm=False)





if __name__ == '__main__':
    main()
