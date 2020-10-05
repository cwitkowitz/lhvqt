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
        # Kernel must be as long as longest basis
        ks1 = self.basis.shape[1]
        # Stride the amount of samples necessary to take 'max_p' responses per frame
        self.sd1 = self.hop_length // self.max_p
        # Padding to start centered around the first real sample,
        # and end centered around the last real sample
        pd1 = self.basis.shape[1] // 2
        self.pd1 = (pd1, self.basis.shape[1] - pd1)

        # Initialize the 1D convolutional filterbank
        self.time_conv = torch.nn.Conv1d(in_channels=nf_in,
                                         out_channels=nf_out,
                                         kernel_size=ks1,
                                         stride=self.sd1,
                                         padding=0,
                                         bias=False)

        if not self.random:
            # Get the real weights from the complex valued bases
            real_weights = np.real(self.basis)
            # View the real/imag channels as independent filters
            real_weights = torch.Tensor(real_weights).view(nf_out, 1, ks1)
            # Manually set the Conv1d parameters with the real/imag weights
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

        # Pad the audio so the last frame of samples can be used
        #audio = super().pad_audio(audio)
        # We manually do the padding for the convolutional
        # layer to allow for different front/back padding
        padded_audio = torch.nn.functional.pad(audio, self.pd1)
        # Convolve the audio with the filterbank of real weights
        feats = self.time_conv(padded_audio)

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
