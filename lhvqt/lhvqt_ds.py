# My imports
from .lvqt_orig import *

# Regular imports
import librosa
import torch.nn as nn
import torch
import os


class LHVQT_DS(torch.nn.Module):
    """
    Harmonic learnable filterbank where only the top harmonic is learned.
    The response for lower harmonics is inferred by downsampling the signal.
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
        super(LHVQT_DS, self).__init__()

        # Default the minimum frequency
        if fmin is None:
            # Note C1
            fmin = librosa.note_to_hz('C1')
        self.fmin = fmin

        # Default the harmonics to those used in the DeepSalience paper
        if harmonics is None:
            harmonics = [0.5, 1, 2, 3, 4, 5]
        harmonics.sort()
        # TODO - how to handle decimal harmonics (e.g. 0.5)?
        self.harmonics = harmonics

        # Default the class definition for the lower-level module
        if lvqt is None:
            # Original LVQT module
            lvqt = LVQT

        # Initialize the learnable set of base filters
        top_fmin = self.fmin * self.harmonics[-1]
        self.top = lvqt(fmin=top_fmin, **kwargs)

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
        # and add the first harmonic to start
        top_h = self.top(wav).unsqueeze(0)
        tf_list = [top_h]

        # Obtain target number of frames for remaining harmonics
        num_frames = self.top.get_expected_frames(wav)

        # Loop through remaining harmonics in reverse order
        for i, h in enumerate(self.harmonics[:-1][::-1]):
            us_rate = self.harmonics[-i-2]
            ds_rate = self.harmonics[-i-1]
            # Low-pass filter to remove frequency content above the current harmonic
            # TODO - do I need this low-pass step?
            # Upsample the audio by the current harmonic
            wav = nn.functional.interpolate(wav, scale_factor=us_rate, mode='linear', align_corners=True)
            # Downsample the audio by the last harmonic
            wav = wav[..., range(0, wav.shape[-1], ds_rate)]

            # Take the transform again
            tf = self.top(wav)
            # Resample the TFR to the same frame amount as the top harmonic
            tf = nn.functional.interpolate(tf, size=num_frames, mode='linear', align_corners=True)
            # Add a harmonic dimension
            tf = tf.unsqueeze(0)
            # Add the transform to the list
            tf_list.append(tf)

        # Reverse the order of the harmonics in the resulting transform
        tf_list.reverse()

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

        # Number of hops in the audio plus one
        num_frames = self.top.get_expected_frames(audio)

        return num_frames

    # TODO - comment
    def plot_time_weights(self, dir_path, mag=False):
        dir_path = os.path.join(dir_path, 'top')
        self.top.plot_time_weights(dir_path, mag)

    # TODO - comment
    def plot_freq_weights(self, dir_path):
        dir_path = os.path.join(dir_path, 'top')
        self.top.plot_freq_weights(dir_path)
