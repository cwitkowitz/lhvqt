# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .lhvqt import *

# Regular imports
import torch.nn as nn
import torch


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

        # Reset the collection of filterbanks
        self.tfs = nn.Module()
        # Use the filterbank associated with the lowest harmonic to match hyperparameters
        self.tfs.add_module('lvqt', lvqt_modules[0])

        # Get the weights from the lowest harmonic
        harmonic_weights = lvqt_modules[0].time_conv.weight

        # Loop through harmonics
        for h in range(1, len(self.harmonics)):
            # Get the filterbank weights associated with the harmonic
            weights = lvqt_modules[h].time_conv.weight

            # Pad the weights to match the size of the lowest harmonic
            pad_amt = harmonic_weights.size(-1) - weights.size(-1)
            # Distribute the padding among both sides of the weights
            pad = (pad_amt // 2, pad_amt - pad_amt // 2)
            # Pad and add the weights to the summed harmonic weights
            harmonic_weights = harmonic_weights + nn.functional.pad(weights, pad=pad)

        # Replace the filterbank weights with the summer harmonic weights
        self.tfs.lvqt.time_conv.weight = torch.nn.Parameter(harmonic_weights)

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

        # Run the audio through the singular module
        feats = self.tfs.lvqt(wav).unsqueeze(0)
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
        num_frames = self.tfs.lvqt.get_expected_frames(audio)

        return num_frames

    def visualize(self, save_dir, **kwargs):
        """
        Perform visualization steps.

        Parameters
        ----------
        save_dir : string
          Top-level directory to hold images of all plots
        **kwargs : N/A
          Arguments for generating plots
        """

        # Visualize the singular module
        self.tfs.lvqt.visualize(save_dir, **kwargs)

    def sonify(self, save_dir, **kwargs):
        """
        Perform sonification steps.

        Parameters
        ----------
        save_dir : string
          Top-level directory to hold images of all plots
        **kwargs : N/A
          Arguments for generating plots
        """

        # Sonify the singular module
        self.tfs.lvqt.sonify(save_dir, **kwargs)
