# My imports
from .utils import *

# Regular imports
from librosa.filters import constant_q
from matplotlib import pyplot as plt
from abc import abstractmethod

import numpy as np
import librosa
import torch
import os


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
                                              pad_fft=False,
                                              norm=None)

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

        # Perform max pooling operation
        # TODO - fix max pooling
        #feats = self.mp(feats)

        if self.to_db:
            # Convert the raw filterbank output to decibels
            feats = torch_amplitude_to_db(feats, to_prob=self.db_to_prob)

        # Number of frames obtained from the filterbank
        num_frames = feats.size(-1)

        if self.batch_norm and not (self.training and num_frames <= 1):
            # Perform batch normalization
            feats = self.bn(feats)

        return feats

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
        if remaining != self.hop_length:
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
        #    # Pad the audio before calculating expected frames (should add one more)
        #    audio = self.pad_audio(torch.Tensor(audio))

        # Number of hops in the audio plus one
        num_frames = audio.shape[-1] // self.hop_length + 1

        return num_frames

    @abstractmethod
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

        return NotImplementedError

    @abstractmethod
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

        return NotImplementedError

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

    def get_comp_weights(self):
        """
        Obtain the complex weights.

        Returns
        ----------
        comp_weights : ndarray (F x T)
          Complex-valued weights as NumPy array,
          F - number of frequency bins
          T - number of time steps (samples)
        """

        real_weights = self.get_real_weights().cpu().detach().numpy()
        imag_weights = self.get_imag_weights().cpu().detach().numpy()
        comp_weights = real_weights + 1j * imag_weights
        return comp_weights

    # TODO - comment
    def plot_time_weights(self, dir_path, mag=False):
        os.makedirs(dir_path, exist_ok=True)

        mag_weights = self.get_mag_weights().cpu().detach().numpy()
        real_weights = self.get_real_weights().cpu().detach().numpy()
        imag_weights = self.get_imag_weights().cpu().detach().numpy()

        for k in range(self.n_bins):
            # TODO - do cpu.detach.numpy in functions?
            # TODO - add y axis - or just make max/min same for all
            if mag:
                plt.plot(mag_weights[k], color='black', label='Magn')
            else:
                plt.plot(real_weights[k], color='black', label='Real', alpha=0.5)
                plt.plot(imag_weights[k], color='red', label='Imag', alpha=0.5)

            plt.axis('off')

            path = os.path.join(dir_path, f'f-{k}.jpg')
            plt.savefig(path)
            plt.clf()

    # TODO - comment
    def plot_freq_weights(self, dir_path, n_fft=2048*16):
        os.makedirs(dir_path, exist_ok=True)

        comp_weights = self.get_comp_weights()

        nyquist = self.fs // 2

        # TODO - CQT basis instead?
        freqs = np.fft.fftfreq(n_fft, (1 / self.fs))

        # Take the magnitude of the FFT of the complex weights (freq response)
        freq_resp = np.abs(np.fft.fft(comp_weights, n=n_fft))

        freq_resp = librosa.amplitude_to_db(freq_resp, ref=np.max)
        freq_resp = freq_resp.T

        # Normalize frequency response
        norm_freq_resp = (freq_resp + 80) / 80
        centroids = np.dot(freqs[:n_fft // 2], norm_freq_resp[:n_fft // 2])
        centroids = centroids / np.sum(norm_freq_resp[:n_fft // 2:], axis=0)

        # Sort by ascending spectral centroid # TODO - make optional
        freq_resp = freq_resp[:, np.argsort(-centroids)]

        #freqs = np.roll(np.fft.fftfreq(n_fft, (1 / self.fs)), n_fft // 2)
        #freq_resp = np.roll(freq_resp, n_fft // 2, axis=0)
        freq_resp = np.concatenate((np.flip(freq_resp[:n_fft // 2]), np.flip(freq_resp[n_fft // 2:])))

        plt.imshow(freq_resp, extent=[0, self.n_bins, -nyquist, nyquist], aspect='auto')
        #plt.yticks(np.linspace(nyquist, -nyquist, 9))
        plt.yticks(np.linspace(nyquist, 0, 2))
        #plt.xticks(np.linspace(0, self.n_bins, ((self.n_bins - 1) // 10) + 1).astype('uint16'))
        # TODO - should only remove negative freq for hilbert
        plt.ylim([0, nyquist])
        plt.title('Frequency Response')
        plt.ylabel('Frequency')
        plt.xlabel('Filter Index')
        plt.colorbar(format='%+2.0f dB')
        #plt.grid(True, color='black', linestyle='--', axis='x')

        path = os.path.join(dir_path, 'freq.jpg')
        plt.savefig(path)
        plt.clf()
