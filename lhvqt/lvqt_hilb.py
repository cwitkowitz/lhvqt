# My imports
from .lvqt import _LVQT
from .utils import *

# Regular imports
from librosa.filters import constant_q
import numpy as np
import librosa
import torch


class LVQT(_LVQT):
    """
    Implements an extension of the original LVQT module. In this version, the imaginary
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

        # One channel (audio) coming in
        nf_in = 1
        # One channel (real) for each bin going out
        nf_out = self.n_bins
        # Kernel must be as long as longest basis
        ks1 = self.basis.shape[1]
        # Stride the amount of samples necessary to take 'max_p' responses per frame
        self.sd1 = self.hop_length // self.max_p
        # Padding to start centered around the first real sample,
        # and end centered around the last real sample
        pd1 = self.basis.shape[1] // 2
        self.pd1 = (pd1, self.basis.shape[1] - pd1)

        # Initialize the 1D convolutional filterbank
        self.time_conv = Conv1d(in_channels=nf_in,
                                out_channels=nf_out,
                                kernel_size=ks1,
                                stride=self.sd1,
                                dropout=True)

        if not self.random:
            # Get the real weights from the complex valued bases
            real_weights = np.real(self.basis)
            # View the real/imag channels as independent filters
            real_weights = torch.Tensor(real_weights).view(nf_out, 1, ks1)
            # Manually set the Conv1d parameters with the real/imag weights
            self.time_conv.weight = torch.nn.Parameter(real_weights)

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

        # Pad the audio so the last frame of samples can be used
        #audio = super().pad_audio(audio)
        # We manually do the padding for the convolutional
        # layer to allow for different front/back padding
        padded_audio = torch.nn.functional.pad(audio, self.pd1)
        # Convolve the audio with the filterbank of real weights
        real_feats = self.time_conv(padded_audio)

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

        real_weights = self.time_conv.weight
        real_weights = real_weights.squeeze()
        return real_weights

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
        imag_weights = torch_hilbert(real_weights)
        return imag_weights
