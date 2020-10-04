# My imports
from .utils import *

# Regular imports
from librosa.filters import constant_q
from abc import abstractmethod

import librosa
import torch


class _LVQT(torch.nn.Module):
    """
    Abstract class to implement common functionality across LVQT variants.
    """

    def __init__(self, fs=22050, hop_length=256, fmin=None, n_bins=360, bins_per_octave=60,
                 gamma=0, random=False, max_p=1, to_db=True, db_to_prob=True, batch_norm=True):
        """
        Initialize parameters common to all LVQT variants.

        Parameters
        ----------
        fs : int or float
          Number of samples per second of audio
        hop_length : int
          Number of samples between frames
        fmin : float
          Lowest center frequency in basis
        n_bins : int
          Number of basis functions in the filterbank
        bins_per_octave : int
          Number of basis functions per octave
        gamma : float
          Bandwidth offset to smoothly vary Q-factor
        random : bool
          Keep the weights random instead of loading in the bases
        max_p : int
          Kernel size and stride for max pooling operation (1 to disable)
        to_db : bool
          Convert features from amplitude to decibels
        db_to_prob : bool
          Scale decibel values to be between 0 and 1 if log is taken
        batch_norm : bool
          Perform batch normalization
        """

        # Load PyTorch Module properties
        super(_LVQT, self).__init__()

        # Default the minimum frequency
        if fmin is None:
            # Note C1
            fmin = librosa.note_to_hz('C1')
        self.fmin = fmin

        # Make parameters accessible
        self.fs = fs
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.gamma = gamma
        self.random = random
        self.max_p = max_p
        self.to_db = to_db
        self.db_to_prob = db_to_prob
        self.batch_norm = batch_norm

        # Get complex bases and their respective lengths for a variable-Q transform
        self.basis, self.lengths = constant_q(sr=fs,
                                              fmin=fmin,
                                              n_bins=n_bins,
                                              bins_per_octave=bins_per_octave,
                                              gamma=gamma,
                                              pad_fft=False)

        # Initialize max pooling to take 'max_p' responses per frame and aggregate with max operation
        self.mp = torch.nn.MaxPool1d(self.max_p)

        # Initialize batch normalization to normalize the output of each channel
        self.bn = torch.nn.BatchNorm1d(self.n_bins)

    @abstractmethod
    def forward(self, audio):
        """
        Perform the main processing steps for the filterbank.

        Parameters
        ----------
        audio : Tensor (B x 1 x T)
          Audio for a batch of tracks,
          B - batch size
          T - number of samples (a.k.a. sequence length)
        """

        return NotImplementedError

    @abstractmethod
    def post_proc(self, feats):
        """
        Perform the post-processing steps for the filterbank.

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
          Post-processed features for a batch of track.
        """

        # Normalize the filterbank features
        #feats = self.norm_feats(feats)

        # Perform max pooling operation
        feats = self.mp(feats)

        if self.to_db:
            # Convert the raw filterbank output to decibels
            feats = torch_amplitude_to_db(feats, to_prob=self.db_to_prob)

        # Number of frames obtained from the filterbank
        num_frames = feats.size(-1)

        if self.batch_norm and not (self.training and num_frames <= 1):
            # Perform batch normalization
            feats = self.bn(feats)

        return feats

    @abstractmethod
    def norm_feats(self, feats):
        """
        Perform the main processing steps for the filterbank.

        Parameters
        ----------
        feats : Tensor (B x F x T)
          Features calculated for a batch of tracks,
          B - batch size
          F - dimensionality of features (number of bins)
          T - number of time steps (frames)
        """

        return NotImplementedError

    @abstractmethod
    def norm_weights(self):
        """
        Perform any steps to normalize output.
        """

        return NotImplementedError

    # TODO - make padding for extra frame optional with self.pad param and put padding
    #        into forward of this abstract class - it will also affect expected frames
    def pad_audio(self, audio):
        """
        Pad audio to squeeze another frame out of trailing samples.

        Parameters
        ----------
        audio : Tensor (B x 1 x T)
          Audio for a batch of tracks,
          B - batch size
          T - number of samples (a.k.a. sequence length)
        """

        # Determine the remaining number of samples required for an additional frame
        remaining = self.hop_length - ((audio.shape[-1]) % self.hop_length)

        # If there are trailing samples...
        if (remaining != self.hop_length):
            # Pad the audio with zeros to fill the remaining samples
            shape = tuple(audio.shape[:-1]) + tuple([remaining])
            filler = torch.zeros(shape).to(audio.device)
            audio = torch.cat((audio, filler), dim=-1)

        return audio

    def get_expected_frames(self, audio, padded=True):
        """
        Determine the number of frames we expect from provided audio.

        Parameters
        ----------
        audio : Tensor (B x 1 x T)
          Audio for a batch of tracks,
          B - batch size
          T - number of samples (a.k.a. sequence length)
        padded : bool
          Whether to factor in padding

        Returns
        ----------
        num_frames : int
          Number of frames which will be generated for given audio
        """

        #if padded:
            # Pad the audio before calculating expected frames (should add one more)
        #    audio = self.pad_audio(audio)

        # Number of hops in the audio plus one
        num_frames = audio.shape[-1] // self.hop_length + 1

        return num_frames
