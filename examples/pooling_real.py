# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from lhvqt.lvqt_real import LVQT as LVQT_R

from lhvqt import LHVQT

from compare_utils import *

# Regular imports
from time import time

import numpy as np
import librosa
import torch


def main():
    """
    Compare the real variant with different max pooling settings to librosa HVQT.
    """

    # Select parameters to use across all implementations
    n_bins = 216  # 6 octaves
    gamma = None  # default gamma
    hop_length = 512
    bins_per_octave = 36
    fmin = librosa.note_to_hz('C1')
    harmonics = [0.5, 1, 2, 3, 4, 5]

    # Load an example piece of audio
    y, sr = librosa.load(librosa.ex('trumpet'))

    # Calculate the HVQT using librosa
    lib_start = time()
    lib_hvqt = librosa_hvqt(y, harmonics, sr, hop_length, fmin, n_bins, bins_per_octave, gamma)
    print(f'Processing Time (Librosa): {time() - lib_start}')

    # Print a new line
    print()

    # Convert librosa HVQT to decibels
    lib_hvqt = librosa.amplitude_to_db(lib_hvqt, ref=np.max)

    # Set the device for the convolutional implementations
    device = 0
    device = torch.device(f'cuda:{device}'
                          if torch.cuda.is_available() else 'cpu')

    # Add a batch and channel dimension to the audio, and make it a tensor
    y = torch.Tensor([[y]]).to(device)

    # Construct the real-only variant
    lhvqt_real = LHVQT(lvqt=LVQT_R,
                       harmonics=harmonics,
                       fs=sr,
                       hop_length=hop_length,
                       fmin=fmin,
                       n_bins=n_bins,
                       bins_per_octave=bins_per_octave,
                       gamma=gamma,
                       max_p=1,
                       random=False,
                       update=False,
                       to_db=False,
                       db_to_prob=False,
                       batch_norm=False,
                       var_drop=False).to(device)

    # Compute the response from the real variant
    rea_start = time()
    rea_hvqt = lhvqt_real(y)
    print(f'Processing Time (Real w/ MP=1): {time() - rea_start}')

    # Remove from the device and convert back to ndarray
    rea_hvqt = rea_hvqt.squeeze(0).cpu().detach().numpy()

    # Convert HVQT to decibels
    rea_hvqt = librosa.amplitude_to_db(rea_hvqt, ref=np.max)

    # Compute similarities after putting all transforms on dB scale
    print('Real Variant Similariy (MP=1): %1.2f%%' % (100 * hvqt_similarity(rea_hvqt, lib_hvqt)))

    # Print a new line
    print()

    # Construct the real-only variant
    lhvqt_real = LHVQT(lvqt=LVQT_R,
                       harmonics=harmonics,
                       fs=sr,
                       hop_length=hop_length,
                       fmin=fmin,
                       n_bins=n_bins,
                       bins_per_octave=bins_per_octave,
                       gamma=gamma,
                       max_p=2,
                       to_db=False,
                       db_to_prob=False,
                       batch_norm=False,
                       var_drop=False).to(device)

    # Compute the response from the real variant
    rea_start = time()
    rea_hvqt = lhvqt_real(y)
    print(f'Processing Time (Real w/ MP=2): {time() - rea_start}')

    # Remove from the device and convert back to ndarray
    rea_hvqt = rea_hvqt.squeeze(0).cpu().detach().numpy()

    # Convert HVQT to decibels
    rea_hvqt = librosa.amplitude_to_db(rea_hvqt, ref=np.max)

    # Compute similarities after putting all transforms on dB scale
    print('Real Variant Similariy (MP=2): %1.2f%%' % (100 * hvqt_similarity(rea_hvqt, lib_hvqt)))

    # Print a new line
    print()

    # Construct the real-only variant
    lhvqt_real = LHVQT(lvqt=LVQT_R,
                       harmonics=harmonics,
                       fs=sr,
                       hop_length=hop_length,
                       fmin=fmin,
                       n_bins=n_bins,
                       bins_per_octave=bins_per_octave,
                       gamma=gamma,
                       max_p=4,
                       to_db=False,
                       db_to_prob=False,
                       batch_norm=False,
                       var_drop=False).to(device)

    # Compute the response from the real variant
    rea_start = time()
    rea_hvqt = lhvqt_real(y)
    print(f'Processing Time (Real w/ MP=4): {time() - rea_start}')

    # Remove from the device and convert back to ndarray
    rea_hvqt = rea_hvqt.squeeze(0).cpu().detach().numpy()

    # Convert HVQT to decibels
    rea_hvqt = librosa.amplitude_to_db(rea_hvqt, ref=np.max)

    # Compute similarities after putting all transforms on dB scale
    print('Real Variant Similariy (MP=4): %1.2f%%' % (100 * hvqt_similarity(rea_hvqt, lib_hvqt)))

    # Print a new line
    print()

    # Construct the real-only variant
    lhvqt_real = LHVQT(lvqt=LVQT_R,
                       harmonics=harmonics,
                       fs=sr,
                       hop_length=hop_length,
                       fmin=fmin,
                       n_bins=n_bins,
                       bins_per_octave=bins_per_octave,
                       gamma=gamma,
                       max_p=8,
                       to_db=False,
                       db_to_prob=False,
                       batch_norm=False,
                       var_drop=False).to(device)

    # Compute the response from the real variant
    rea_start = time()
    rea_hvqt = lhvqt_real(y)
    print(f'Processing Time (Real w/ MP=8): {time() - rea_start}')

    # Remove from the device and convert back to ndarray
    rea_hvqt = rea_hvqt.squeeze(0).cpu().detach().numpy()

    # Convert HVQT to decibels
    rea_hvqt = librosa.amplitude_to_db(rea_hvqt, ref=np.max)

    # Compute similarities after putting all transforms on dB scale
    print('Real Variant Similariy (MP=8): %1.2f%%' % (100 * hvqt_similarity(rea_hvqt, lib_hvqt)))

    # Print a new line
    print()

    # Construct the real-only variant
    lhvqt_real = LHVQT(lvqt=LVQT_R,
                       harmonics=harmonics,
                       fs=sr,
                       hop_length=hop_length,
                       fmin=fmin,
                       n_bins=n_bins,
                       bins_per_octave=bins_per_octave,
                       gamma=gamma,
                       max_p=16,
                       to_db=False,
                       db_to_prob=False,
                       batch_norm=False,
                       var_drop=False).to(device)

    # Compute the response from the real variant
    rea_start = time()
    rea_hvqt = lhvqt_real(y)
    print(f'Processing Time (Real w/ MP=16): {time() - rea_start}')

    # Remove from the device and convert back to ndarray
    rea_hvqt = rea_hvqt.squeeze(0).cpu().detach().numpy()

    # Convert HVQT to decibels
    rea_hvqt = librosa.amplitude_to_db(rea_hvqt, ref=np.max)

    # Compute similarities after putting all transforms on dB scale
    print('Real Variant Similariy (MP=16): %1.2f%%' % (100 * hvqt_similarity(rea_hvqt, lib_hvqt)))

    # Print a new line
    print()

    # Construct the real-only variant
    lhvqt_real = LHVQT(lvqt=LVQT_R,
                       harmonics=harmonics,
                       fs=sr,
                       hop_length=hop_length,
                       fmin=fmin,
                       n_bins=n_bins,
                       bins_per_octave=bins_per_octave,
                       gamma=gamma,
                       max_p=32,
                       to_db=False,
                       db_to_prob=False,
                       batch_norm=False,
                       var_drop=False).to(device)

    # Compute the response from the real variant
    rea_start = time()
    rea_hvqt = lhvqt_real(y)
    print(f'Processing Time (Real w/ MP=32): {time() - rea_start}')

    # Remove from the device and convert back to ndarray
    rea_hvqt = rea_hvqt.squeeze(0).cpu().detach().numpy()

    # Convert HVQT to decibels
    rea_hvqt = librosa.amplitude_to_db(rea_hvqt, ref=np.max)

    # Compute similarities after putting all transforms on dB scale
    print('Real Variant Similariy (MP=32): %1.2f%%' % (100 * hvqt_similarity(rea_hvqt, lib_hvqt)))


if __name__ == '__main__':
    main()
