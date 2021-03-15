# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .lvqt import _LVQT
from .variational import *

# Regular imports
import numpy as np
import torch


class LVQT(_LVQT):
    """
    Implements a slight adaptation/modernization of the original module presented in my Master's
    Thesis (https://scholarworks.rit.edu/theses/10143/). This variant is referred to as the
    classic variant.
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
        # Two channels (real/imag) for each bin going out
        nf_out = 2 * self.n_bins

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
            # Split the complex valued bases into real and imaginary weights
            real_weights, imag_weights = np.real(self.basis), np.imag(self.basis)
            # Zip them together as separate channels so both components of each filter are adjacent
            complex_weights = np.array([[real_weights[i]] + [imag_weights[i]]
                                        for i in range(self.n_bins)])
            # View the real/imag channels as independent filters
            complex_weights = torch.Tensor(complex_weights).view(nf_out, 1, self.ks1)
            # Manually set the Conv1d parameters with the real/imag weights
            self.time_conv.weight = torch.nn.Parameter(complex_weights)

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

        # Run the standard convolution steps
        feats = super().forward(audio)

        # Switch filter and frame dimension
        feats = feats.transpose(1, 2)
        # Perform l2 pooling across the filter dimension
        feats = self.l2_pool(feats)
        # Switch the frame and filter dimension back
        feats = feats.transpose(1, 2)

        # Perform post-processing steps
        feats = self.post_proc(feats)

        return feats

    def get_weights(self):
        """
        Obtain the weights of the transform split by real/imag.

        Returns
        ----------
        comp_weights : Tensor (F x 2 x T)
          Weights of the transform split by real/imag,
          F - number of frequency bins
          T - number of time steps (samples)
        """

        weights = self.time_conv.weight
        weights = weights.view(self.n_bins, 2, -1)

        return weights

    def get_real_weights(self):
        """
        Obtain the weights of the real part of the transform.

        Returns
        ----------
        real_weights : Tensor (F x T)
          Weights of the real part of the transform,
          F - number of frequency bins
          T - number of time steps (samples)
        """

        comp_weights = self.get_weights()
        real_weights = comp_weights[:, 0]

        return real_weights

    def get_imag_weights(self):
        """
        Obtain the weights of the imaginary part of the transform.

        Returns
        ----------
        imag_weights : Tensor (F x T)
          Weights of the imaginary part of the transform,
          F - number of frequency bins
          T - number of time steps (samples)
        """

        comp_weights = self.get_weights()
        imag_weights = comp_weights[:, 1]

        return imag_weights
