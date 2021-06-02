# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .lvqt_orig import *

# Regular imports
import librosa
import torch
import os


class LHVQT(torch.nn.Module):
    """
    Learnable filterbank with various HVQT initialization capabilties.
    """

    def __init__(self, fmin=None, harmonics=None, lvqt=None, **kwargs):
        """
        Initialize parameters necessary to construct a harmonic learnable filterbank.

        Parameters
        ----------
        fmin : float
          Lowest center frequency in basis
        harmonics : list of ints
          Specific harmonics to stack across the harmonic dimension
        lvqt : type
          Class definition of chosen lower-level LVQT module
        **kwargs : N/A
          Any parameters intended for the lower-level LVQT module
        """

        # Load PyTorch Module properties
        super(LHVQT, self).__init__()

        # Default the minimum frequency
        if fmin is None:
            # Note C1
            fmin = librosa.note_to_hz('C1')
        self.fmin = fmin

        # Default the harmonics to those used in the DeepSalience paper
        if harmonics is None:
            harmonics = [0.5, 1, 2, 3, 4, 5]
        harmonics.sort()
        self.harmonics = harmonics

        # Default the class definition for the lower-level module
        if lvqt is None:
            # Classic LVQT module
            lvqt = LVQT

        # Create a PyTorch Module to hold LVQTs
        self.tfs = torch.nn.Module()

        # Loop through harmonics
        for h in range(len(self.harmonics)):
            # Name module according to index of harmonic
            mod_name = 'lvqt%d' % (h + 1)
            # Calculate the lowest center frequency of this LVQT
            fmin = float(harmonics[h]) * self.fmin
            # Create and add the LVQT to the LHVQT module
            self.tfs.add_module(mod_name, lvqt(fmin=fmin, **kwargs))

    def get_modules(self):
        """
        Helper function to extract pointers to the lower-level modules.

        Returns
        ----------
        lvqt_modules : list of _LVQT
          Pointers to each module in the harmonic filterbank
        """

        # Obtain a pointer to the lower-level modules
        lvqt_modules = torch.nn.Sequential(*list(self.tfs.children()))

        return lvqt_modules

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

        # Initialize a list to hold each harmonic transform
        tf_list = []

        # Obtain a pointer to the lower-level modules
        lvqt_modules = self.get_modules()

        # Loop through harmonics
        for h in range(len(self.harmonics)):
            # Take the transform at each harmonic
            tf = lvqt_modules[h](wav)
            # Add a harmonic dimension
            tf = tf.unsqueeze(0)
            # Add the transform to the list
            tf_list.append(tf)

        # Combine the transforms together along harmonic dimension
        feats = torch.cat(tf_list, dim=0)
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

        # Obtain a pointer to the lower-level modules
        lvqt_modules = self.get_modules()

        # Gather expected counts from lower-level modules
        num_frames = [lvqt_modules[h].get_expected_frames(audio)
                      for h in range(len(self.harmonics))]

        # Check if count of each module is the same
        if len(set(num_frames)) == 1:
            # Collapse the expected frame counts
            num_frames = num_frames[0]

        return num_frames

    def visualize(self, save_dir, **kwargs):
        """
        Perform visualization steps for each lower-level module.

        Parameters
        ----------
        save_dir : string
          Top-level directory to hold images of all plots
        **kwargs : N/A
          Arguments for generating plots
        """

        # Obtain a pointer to the lower-level modules
        lvqt_modules = self.get_modules()

        # Loop through harmonics
        for h in range(len(self.harmonics)):
            # Construct a path to the directory for the harmonic
            h_dir = os.path.join(save_dir, f'h_{self.harmonics[h]}') if len(self.harmonics) > 1 else save_dir
            # Visualize the harmonic
            lvqt_modules[h].visualize(h_dir, **kwargs)

    def sonify(self, save_dir, **kwargs):
        """
        Perform sonification steps for each lower-level module.

        Parameters
        ----------
        save_dir : string
          Top-level directory to hold images of all plots
        **kwargs : N/A
          Arguments for generating audio
        """

        # Obtain a pointer to the lower-level modules
        lvqt_modules = self.get_modules()

        # Loop through harmonics
        for h in range(len(self.harmonics)):
            # Construct a path to the directory for the harmonic
            h_dir = os.path.join(save_dir, f'h_{self.harmonics[h]}') if len(self.harmonics) > 1 else save_dir
            # Sonify the harmonic
            lvqt_modules[h].sonify(h_dir, **kwargs)
