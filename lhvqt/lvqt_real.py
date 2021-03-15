# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .lvqt import _LVQT
from .variational import *

# Regular imports
import numpy as np
import torch


class LVQT(_LVQT):
    """
    Implements an extension of the original LVQT module. In this version, only the real channel is used.
    """

    def __init__(self, **kwargs):
        """
        Initialize LVQT parameters and the PyTorch processing modules.

        Parameters
        ----------
        See _LVQT class...
        """

        super(LVQT, self).__init__(**kwargs)

        # One channel (audio) coming in
        nf_in = 1
        # One channel (real) for each bin going out
        nf_out = self.n_bins

        # Initialize the 1D convolutional filterbank
        if self.var_drop:
            self.time_conv = VariationalDropoutConv1d(in_channels=nf_in, out_channels=nf_out,
                                                      kernel_size=self.ks1, stride=self.sd1,
                                                      log_sigma2=self.var_drop)
        else:
            self.time_conv = torch.nn.Conv1d(in_channels=nf_in, out_channels=nf_out,
                                             kernel_size=self.ks1, stride=self.sd1,
                                             bias=False)

        if not self.random:
            # Get the real weights from the complex valued bases
            real_weights = np.real(self.basis)
            # Add a channel dimension to the weights
            real_weights = torch.Tensor(real_weights).view(nf_out, 1, self.ks1)
            # Manually set the Conv1d parameters to the weights
            self.time_conv.weight = torch.nn.Parameter(real_weights)

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

        # Run the standard convolution steps
        feats = super().forward(audio)

        # Perform post-processing steps
        feats = self.post_proc(feats)

        return feats

    def get_real_weights(self):
        """
        Obtain the real-valued weights.

        Returns
        ----------
        real_weights : Tensor (F x T)
          Weights of the real part of the transform,
          F - number of frequency bins
          T - number of time steps (samples)
        """

        real_weights = self.time_conv.weight
        real_weights = real_weights.squeeze()

        return real_weights

    def get_imag_weights(self):
        """
        Return zeros matching size of real weights

        Returns
        ----------
        imag_weights : Tensor (F x T)
          Weights of the imaginary part of the transform (all zeros here),
          F - number of frequency bins
          T - number of time steps (samples)
        """

        real_weights = self.get_real_weights()
        imag_weights = torch.zeros(real_weights.size())
        imag_weights = imag_weights.to(real_weights.device)

        return imag_weights
