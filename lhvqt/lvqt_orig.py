# My imports
from .lvqt import _LVQT

# Regular imports
import numpy as np
import torch


class LVQT(_LVQT):
    """
    Implements a slight adaptation/modernization of the original module
    presented in my Master's Thesis (https://scholarworks.rit.edu/theses/10143/).
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
        # Kernel must be as long as longest basis
        ks1 = self.basis.shape[1]
        # Stride the amount of samples necessary to take 'max_p' responses per frame
        sd1 = self.hop_length // self.max_p
        # Padding to start centered around the first real sample,
        # and end centered around the last real sample
        pd1 = self.basis.shape[1] // 2
        self.pd1 = (pd1, self.basis.shape[1] - pd1)

        # Initialize the 1D convolutional filterbank
        self.time_conv = torch.nn.Conv1d(in_channels=nf_in,
                                         out_channels=nf_out,
                                         kernel_size=ks1,
                                         stride=sd1,
                                         padding=0,
                                         bias=False)

        if self.random:
            # Weights are initialized randomly by default - but they must be normalized
            self.norm_weights()
        else:
            # Split the complex valued bases into real and imaginary weights
            real_weights, imag_weights = np.real(self.basis), np.imag(self.basis)
            # Zip them together as separate channels so both components of each filter are adjacent
            complex_weights = np.array([[real_weights[i]] + [imag_weights[i]]
                                        for i in range(self.n_bins)])
            # View the real/imag channels as independent filters
            complex_weights = torch.Tensor(complex_weights).view(nf_out, 1, ks1)
            # Manually set the Conv1d parameters with the real/imag weights
            self.time_conv.weight = torch.nn.Parameter(complex_weights + 1e-10)

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
        # Convolve the audio with the filterbank
        feats = self.time_conv(padded_audio)
        # Switch filter and frame dimension
        feats = feats.transpose(1, 2)
        # Perform l2 pooling across the filter dimension
        feats = self.l2_pool(feats)
        # Switch the frame and filter dimension back
        feats = feats.transpose(1, 2)

        #feats *= torch.Tensor(self.lengths[np.newaxis, :, np.newaxis]).to(audio.device)
        #feats /= torch.sqrt(torch.Tensor(self.lengths)).unsqueeze(1).to(audio.device)

        # Perform post-processing steps
        feats = self.post_proc(feats)

        return feats

    def get_comp_weights(self):
        """
        Obtain the weights of the transform split by real/imag.

        Returns
        ----------
        comp_weights : Tensor (F x 2 x T)
          Weights of the transform split by real/imag,
          F - number of frequency bins
          T - number of time steps (samples)
        """

        comp_weights = self.time_conv.weight
        comp_weights = comp_weights.view(self.n_bins, 2, -1)
        return comp_weights

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

        comp_weights = self.get_comp_weights()
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

        comp_weights = self.get_comp_weights()
        imag_weights = comp_weights[:, 1]
        return imag_weights

    def get_mag_weights(self):
        """
        Obtain the magnitude of the complex weights.

        Returns
        ----------
        mag_weights : Tensor (F x T)
          Magnitude of the complex weights,
          F - number of frequency bins
          T - number of time steps (samples)
        """

        real_weights = self.get_real_weights()
        imag_weights = self.get_imag_weights()
        mag_weights = torch.sqrt(real_weights ** 2 + imag_weights ** 2)
        return mag_weights

    def norm_feats(self, feats):
        """
        Normalize the features based on the l2 norm of filter weights.

        Parameters
        ----------
        feats : Tensor (B x F x T)
          Features calculated for a batch of tracks,
          B - batch size
          F - dimensionality of features (number of bins)
          T - number of time steps (frames)

        Returns
        ----------
        feats : Tensor (B x F x T)
          Normalized features for a batch of track.
        """

        # TODO - this produces NaNs in the gradient somehow - fixed by adding epsilon to zero weights?
        mag_weights = self.get_mag_weights()
        # Get the l2 norm of the weight magnitude
        norm = torch.norm(mag_weights, p=2, dim=1, keepdim=True)
        #norm = torch.zeros(norm.size()).to(feats.device)
        # Divide the features by the norm
        feats = torch.div(feats, norm)

        return feats

    def norm_weights(self):
        """
        Normalize the weights based on the l1 norm of filter weights.
        """

        # Turn off gradient management
        with torch.no_grad():
            complex_weights = self.get_comp_weights()
            mag_weights = self.get_mag_weights()
            # Calculate the l1 norm of the magnitude weights
            norm = torch.norm(mag_weights, p=1, dim=1, keepdim=True).unsqueeze(-1)
            # Normalize the complex weights
            norm_weights = complex_weights / norm
            # Re-zip the weights
            norm_weights = norm_weights.view(2 * self.n_bins, 1, -1)
            # Insert the new weights into the 1D conv layer
            self.time_conv.weight = torch.nn.Parameter(norm_weights)
