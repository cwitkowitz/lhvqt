# My imports
from .lhvqt import *
from .lvqt_orig import *

# Regular imports
import librosa
import torch.nn as nn
import torch
import os


class LHVQT_COMB(LHVQT):
    """
    Harmonic learnable filterbank where the filters
    associated with each harmonic are collapsed into one.
    """

    def __init__(self, fmin=None, harmonics=None, lvqt=None, **kwargs):
        """
        Initialize parameters necessary to construct a harmonic learnable filterbank.

        Parameters
        ----------
        fmin : float
          Lowest center frequency in basis
        harmonics : list of int
          Harmonics to compute
        lvqt : type
          Class definition of chosen lower-level LVQT module
        **kwargs : N/A
          Any parameters intended for the lower-level LVQT module
        """

        # Load PyTorch Module properties
        super(LHVQT_COMB, self).__init__(fmin=fmin, harmonics=harmonics, lvqt=lvqt, **kwargs)

        # Obtain a pointer to the lower-level modules
        lvqt_modules = self.get_modules()

        # TODO - overwrite self.tfs

        self.comb = lvqt_modules[0]
        weights = self.comb.time_conv.weight

        for h in range(1, len(self.harmonics)):
            basis = lvqt_modules[h].time_conv.weight

            pad_amt = weights.size(-1) - basis.size(-1)
            pad = (pad_amt // 2, pad_amt - pad_amt // 2)

            basis = nn.functional.pad(basis, pad=pad)
            weights = weights + basis

        self.comb.time_conv.weight = torch.nn.Parameter(weights)

    def forward(self, wav):
        """
        Perform the main processing steps for the harmonic filterbank.

        Parameters
        ----------
        wav : Tensor (B x 1 x T)
          Audio for a batch of tracks,
          B - batch size
          T - number of samples (a.k.a. sequence length)

        Returns
        ----------
        feats : Tensor (B x H x F x T)
          Input features for a batch of tracks,
          B - batch size
          H - number of harmonics (a.k.a. channels)
          F - dimensionality of features
          T - number of time steps (frames)
        """

        feats = self.comb(wav).unsqueeze(0)
        # Switch harmonic and batch dimension
        feats = feats.transpose(1, 0)

        return feats

    def get_expected_frames(self, audio):
        """
        Determine the number of frames we expect from provided audio.

        Parameters
        ----------
        audio : Tensor (B x 1 x T)
          Audio for a batch of tracks,
          B - batch size
          T - number of samples (a.k.a. sequence length)

        Returns
        ----------
        num_frames : int
          Number of frames which will be generated for given audio
        """

        # Number of hops in the audio plus one
        num_frames = self.comb.get_expected_frames(audio)

        return num_frames

    def visualize(self, save_dir, **kwargs):
        # Visualize the harmonic comb
        self.comb.visualize(save_dir, **kwargs)
