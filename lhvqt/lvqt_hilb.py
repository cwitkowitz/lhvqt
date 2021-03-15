# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .lvqt_real import LVQT as _LVQT
from .utils import *

# Regular imports
import torch


class LVQT(_LVQT):
    """
    Implements an extension of the real-only LVQT variant. In this version, the imaginary
    weights of the transform are inferred from the real weights by using the Hilbert transform.
    """

    def __init__(self, **kwargs):
        """
        Initialize LVQT parameters and the PyTorch processing modules.

        Parameters
        ----------
        See _LVQT class...
        """

        super(LVQT, self).__init__(**kwargs)

        # Initialize l2 pooling to recombine real/imag filter channels
        self.l2_pool = torch.nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)

    def forward(self, audio):
        """
        Perform the main processing steps for the filterbank.

        Parameters
        ----------
        audio : Tensor (B x 1 x T)
          Audio for a batch of tracks,
          B - batch size
          T - number of samples (a.k.a. sequence length)

        Returns
        ----------
        feats : Tensor (B x F x T)
          Features calculated for a batch of tracks,
          B - batch size
          F - dimensionality of features (number of bins)
          T - number of time steps (frames)
        """

        # Run the standard convolution steps (call grandparent function)
        real_feats = super(type(self).__bases__[0], self).forward(audio)

        # We manually do the padding for the convolutional
        # layer to allow for different front/back padding
        padded_audio = torch.nn.functional.pad(audio, self.pd1)

        # Obtain the imaginary weights
        imag_weights = self.get_imag_weights().unsqueeze(1)
        # Convolve the audio with the filterbank of imaginary weights
        imag_feats = torch.nn.functional.conv1d(padded_audio,
                                                weight=imag_weights,
                                                stride=self.sd1,
                                                padding=0)

        # Add an extra dimension to both sets of features
        real_feats = real_feats.unsqueeze(-1)
        imag_feats = imag_feats.unsqueeze(-1)

        # Concatenate the features along a new dimension
        feats = torch.cat((real_feats, imag_feats), dim=-1)
        # Switch filter and frame dimension
        feats = feats.transpose(1, 2)
        # Collapse the last dimension to zip the features
        # and make the real/imag responses adjacent
        feats = feats.reshape(tuple(list(feats.shape[:-2]) + [-1]))
        # Perform l2 pooling across the filter dimension
        feats = self.l2_pool(feats)
        # Switch the frame and filter dimension back
        feats = feats.transpose(1, 2)

        # Perform post-processing steps
        feats = self.post_proc(feats)

        return feats

    def get_imag_weights(self):
        """
        Obtain the weights of the imaginary part of the transform using Hilbert transform.

        Returns
        ----------
        imag_weights : Tensor (F x T)
          Weights of the imaginary part of the transform,
          F - number of frequency bins
          T - number of time steps (samples)
        """

        real_weights = self.get_real_weights()

        if self.update:
            # Calculate HT as normal
            imag_weights = torch_hilbert(real_weights)
        else:
            # Don't allow gradients to propagate through HT
            with torch.no_grad():
                imag_weights = torch_hilbert(real_weights)

        return imag_weights
