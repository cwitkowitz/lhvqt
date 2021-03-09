# My imports
from .utils import *

# Regular imports
from mpl_toolkits.axisartist.axislines import SubplotZero
from librosa.filters import constant_q
from matplotlib import pyplot as plt
from matplotlib import rcParams
from abc import abstractmethod

import numpy as np
import librosa
import torch
import math
import os

# TODO - customize font for visualization?
#rcParams['font.family'] = 'sans-serif'
#rcParams['font.sans-serif'] = ['Tahoma']


class _LVQT(torch.nn.Module):
    """
    Abstract class to implement common functionality across LVQT variants.
    """

    def __init__(self, fs=22050, hop_length=256, fmin=None, n_bins=360, bins_per_octave=60, gamma=0,
                 random=False, max_p=1, to_db=True, db_to_prob=True, batch_norm=True, var_drop=True):
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
        var_drop : bool
          Perform variational dropout after the 1D convolutional layer
        """

        # Load PyTorch Module properties
        super(_LVQT, self).__init__()

        # Make parameters accessible
        self.fs = fs
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.random = random
        self.max_p = max_p
        self.to_db = to_db
        self.db_to_prob = db_to_prob
        self.batch_norm = batch_norm
        self.var_drop = var_drop

        # Default the minimum frequency
        if fmin is None:
            # Note C1
            fmin = librosa.note_to_hz('C1')
        self.fmin = fmin

        # Default gamma using the procedure defined in
        # librosa.filters.constant_q.vqt documentation
        if gamma is None:
            alpha = 2.0 ** (1.0 / bins_per_octave) - 1
            gamma = 24.7 * alpha / 0.108
        self.gamma = gamma

        # Make sure no filters will be invalid
        Q = 1 / (2.0 ** (1.0 / bins_per_octave) - 1.0)
        center_freqs = fmin * (2.0 ** (np.arange(n_bins) / bins_per_octave))
        band_bounds = center_freqs * (1 + 0.5 * librosa.filters.window_bandwidth('hann') / Q)
        num_invalid = int(np.sum(band_bounds > fs / 2.0))

        # Get complex bases and their respective lengths for a variable-Q transform
        basis, lengths = constant_q(sr=fs,
                                    fmin=fmin,
                                    n_bins=n_bins - num_invalid,
                                    bins_per_octave=bins_per_octave,
                                    gamma=gamma,
                                    pad_fft=False,
                                    norm=None)

        zero_filters = np.zeros((num_invalid, basis.shape[-1]))
        zero_lengths = np.zeros(num_invalid)

        self.basis = np.concatenate((basis, zero_filters), axis=0)
        self.lengths = np.concatenate((lengths, zero_lengths), axis=0)

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
        # TODO - think this is related to max pooling breaking
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

        real_weights = tensor_to_array(self.get_real_weights())
        imag_weights = tensor_to_array(self.get_imag_weights())

        comp_weights = real_weights + 1j * imag_weights

        return comp_weights

    def visualize(self, save_dir, **kwargs):
        time_dir = os.path.join(save_dir, 'time')

        time_kwargs = filter_kwargs(['idcs',
                                     'fix_scale',
                                     'include_axis'], **kwargs)

        self.visualize_time_domain_complex(time_dir, **time_kwargs)

        fft_dir = os.path.join(save_dir, 'fft')

        fft_1d_kwargs = filter_kwargs(['idcs',
                                       'n_fft',
                                       'include_axis',
                                       'scale_freqs',
                                       'decibels',
                                       'include_negative',
                                       'separate'], **kwargs)

        self.visualize_freq_domain_fft_1d(fft_dir, **fft_1d_kwargs)

        fft_2d_path = os.path.join(fft_dir, f'all_2d.jpg')

        fft_2d_kwargs = filter_kwargs(['idcs',
                                       'n_fft',
                                       'sort_by_centroid',
                                       'include_axis',
                                       'scale_freqs',
                                       'include_negative'], **kwargs)
        self.visualize_freq_domain_fft_2d(fft_2d_path, **fft_2d_kwargs)

    def visualize_time_domain_complex(self, save_dir, idcs=None, fix_scale=False, include_axis=False):
        """
        Plot the time domain filters of the filterbank.

        Parameters
        ----------
        save_dir : string
          Directory under which to save images for each plot
        idcs : list, ndarray or None (optional)
          Specific filter indices to plot rather than plotting all of them
        fix_scale : bool
          Whether to place all filters on the same amplitude scale
        include_axis : bool
          Whether to add X and Y axis and a grid along the Y axis
        """

        # Make sure the provided save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Obtain the real and imaginary filterbank weights
        real_weights = tensor_to_array(self.get_real_weights())
        imag_weights = tensor_to_array(self.get_imag_weights())

        # Determine the maximum weight magnitude for scaling
        max_weight = max(np.max(np.abs(real_weights)), np.max(np.abs(imag_weights)))

        # Determine the amount of weights in each filter
        num_weights = real_weights.shape[-1]

        # Create ascending indices to loop through
        filter_idcs = np.arange(self.n_bins)

        if idcs is not None:
            # Reduce the indices to those specified by user
            filter_idcs = np.intersect1d(filter_idcs, idcs)

        # Create a figure and axis for plotting
        fig = plt.figure()
        ax = SubplotZero(fig, 111)
        fig.add_subplot(ax)

        # Loop through the filter indices
        for k in filter_idcs:
            if fix_scale:
                # Make the y boundaries of each filter 10% of the maximum weight magnitude
                ax.set_ylim([-1.1 * max_weight, 1.1 * max_weight])

            # Remove right, top, and bottom border
            ax.axis['right'].set_visible(False)
            ax.axis['top'].set_visible(False)
            ax.axis['bottom'].set_visible(False)

            if include_axis:
                # Add X axes at origin
                ax.axis['xzero'].set_visible(True)
                # Only add X tick to show number of weights in the plot
                ax.set_xticks([num_weights])
                # Remove space padding along X axis
                ax.set_xlim([0, num_weights])
                # Add a grid to the axis
                ax.grid(axis='y')
            else:
                # Remove the left border
                ax.axis['left'].set_visible(False)

            # Minimize free space
            fig.tight_layout()

            # Plot the real and imaginary weights separately
            ax.plot(real_weights[k], color='black', label='Real', alpha=0.75)
            ax.plot(imag_weights[k], color='purple', label='Imag', alpha=0.75)

            # Construct a path to save an image of the plot
            save_path = os.path.join(save_dir, f'f_{k}.jpg')

            # Save the figure
            fig.savefig(save_path)

            # Clear the plot in preparation for the next filter
            ax.cla()

    def visualize_freq_domain_fft_1d(self, save_dir, idcs=None, n_fft=None, include_axis=False,
                                     scale_freqs=False, decibels=False, include_negative=False, separate=True):
        """
        Plot the FFT response of the filterbank and display in 1D fashion.

        Parameters
        ----------
        save_dir : string
          Directory under which to save images for each plot
        idcs : list, ndarray or None (optional)
          Specific filter indices to plot rather than plotting all of them
        n_fft : int or None (optional)
          See np.fft.fft documentation...
        include_axis : bool
          Whether to add X and Y axis and a grid along the Y axis
        scale_freqs : bool
          Whether to leave frequencies as they are or scale to be between [-1, 1]
        decibels : bool
          Whether to convert to dB or leave as amplitude
        include_negative : bool
          Whether to include negative frequencies in the plot
        separate : bool
          Whether to plot the response of each filter separately or all in one plot
        """

        # Make sure the provided save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Obtain the complex filterbank weights
        comp_weights = self.get_comp_weights()

        # Determine the Nyquist
        nyquist = self.fs / 2

        # Create ascending indices to loop through
        filter_idcs = np.arange(self.n_bins)

        if idcs is not None:
            # Reduce the indices to those specified by user
            filter_idcs = np.intersect1d(filter_idcs, idcs)

        # Scale the X axis to be twice the length of Y axis if negative frequencies are included
        figsize = rcParams['figure.figsize']
        figsize = [(2 ** include_negative) * figsize[0], figsize[1]]

        # Create a figure and axis for plotting
        fig = plt.figure(figsize=figsize)
        ax = SubplotZero(fig, 111)
        fig.add_subplot(ax)

        if include_negative:
            # Keep negative frequencies and remove space padding along X axis
            ax.set_xlim(-nyquist ** (not scale_freqs), nyquist ** (not scale_freqs))
        else:
            # Remove negative frequencies and remove space padding along X axis
            ax.set_xlim(0, nyquist ** (not scale_freqs))

        # Remove right and top border
        ax.axis['right'].set_visible(False)
        ax.axis['top'].set_visible(False)

        # Calculate the FFT response for all filters at once
        freqs, resp = fft_response(comp_weights, self.fs, n_fft, decibels)

        # Remove space padding along Y axis
        ax.set_ylim(bottom=-80) if decibels else ax.set_ylim(bottom=0)

        # Make the top boundary of Y axis 10% above maximum
        if decibels:
            ax.set_ylim(top=10 * math.log10(1.1) + np.max(resp))
        else:
            ax.set_ylim(top=1.1 * np.max(resp))

        if include_axis:
            # Add an appropriate label to the Y axis
            ax.set_ylabel('dB') if decibels else ax.set_ylabel('A')
            # Add a grid to the axis
            ax.grid(axis='y')
        else:
            # Remove the left and bottom border
            ax.axis['left'].set_visible(False)
            ax.axis['bottom'].set_visible(False)

        # Minimize free space
        fig.tight_layout()

        if scale_freqs:
            # Scale the frequencies to be within [-1, 1]
            freqs = freqs / nyquist

        # Loop through the filter indices
        for k in filter_idcs:
            # Plot the FFT response
            ax.plot(freqs, resp[k], color='purple', label='FFT Response', alpha=0.75)

            if separate:
                # Construct a path to save an image of the plot
                save_path = os.path.join(save_dir, f'f_{k}.jpg')
                # Save the figure
                fig.savefig(save_path)
                # Clear the plot in preparation for the next filter
                ax.lines[0].remove()

        if not separate:
            # Construct a path to save an image of the plot
            save_path = os.path.join(save_dir, f'all_1d.jpg')
            # Save the figure, now that it is complete
            fig.savefig(save_path)

    def visualize_freq_domain_fft_2d(self, save_path, idcs=None, n_fft=None, sort_by_centroid=False,
                                     include_axis=False, scale_freqs=False, include_negative=False):
        """
        Plot the FFT response (dB) of the filterbank and display in 2D fashion.

        Parameters
        ----------
        save_path : string
          Path to use when saving an image of the plot
        idcs : list, ndarray or None (optional)
          Specific filter indices to plot rather than plotting all of them
        n_fft : int or None (optional)
          See np.fft.fft documentation...
        sort_by_centroid : bool
          Whether to order the filters by ascending spectral centroid;
        include_axis : bool
          Whether to add X and Y axis and a grid along the Y axis
        scale_freqs : bool
          Whether to leave frequencies as they are or scale to be between [-1, 1]
        include_negative : bool
          Whether to include negative frequencies in the plot
        """

        # Make sure the provided save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Obtain the complex filterbank weights
        comp_weights = self.get_comp_weights()

        # Calculate the FFT response for all filters at once
        freqs, resp = fft_response(comp_weights, self.fs, n_fft, False)

        # Determine the Nyquist
        nyquist = self.fs / 2

        # Create ascending indices to loop through
        filter_idcs = np.arange(self.n_bins)

        if idcs is not None:
            # Reduce the indices to those specified by user
            filter_idcs = np.intersect1d(filter_idcs, idcs)

        # Keep only filters matching specified indices
        resp = resp[filter_idcs]

        # Determine the number of bins in the response
        num_bins = resp.shape[-1]

        if sort_by_centroid:
            # Get the normalized response in case the filters are not normalized
            norm_resp = resp / (np.expand_dims(np.sum(resp, axis=-1), axis=-1) + EPSILON)

            # Compute the spectral centroid of each filter
            centroids = np.dot(norm_resp[..., num_bins // 2:], freqs[num_bins // 2:])
            centroids = centroids / (np.sum(norm_resp[..., num_bins // 2:], axis=-1) + EPSILON)

            # Sort the filters by spectral centroid of positive frequency response
            resp = resp[np.argsort(centroids)]

        # Convert the frequency response from amplitude to decibels
        resp = librosa.amplitude_to_db(resp, ref=np.max)

        # Line up the response properly for the image
        resp = np.transpose(np.flip(resp, axis=-1))

        # Scale the X axis to be twice the length of Y axis. If negative
        # frequencies are included, the Y axis length will be scaled by two as well
        figsize = rcParams['figure.figsize']
        figsize = [2 * figsize[0], (2 ** include_negative) * figsize[1]]

        # Create a figure and axis for plotting
        fig = plt.figure(figsize=figsize)
        ax = SubplotZero(fig, 111)
        fig.add_subplot(ax)

        # Determine the frequency boundaries we are operating with
        y_bounds = [-nyquist ** (not scale_freqs), nyquist ** (not scale_freqs)]

        # Plot the response for all filters as an image
        img = ax.imshow(resp, extent=[0, len(filter_idcs), y_bounds[0], y_bounds[1]], aspect='auto')

        # Remove right and top border
        ax.axis['right'].set_visible(False)
        ax.axis['top'].set_visible(False)

        if include_axis:
            # Only add X tick to show number of filters in plot
            ax.set_xticks([len(filter_idcs)])
            # Add a grid to the axis
            ax.grid(axis='y')
            # Add a colorbar to the figure
            fig.colorbar(img, format='%+2.0f dB')
        else:
            # Remove the left and bottom border
            ax.axis['left'].set_visible(False)
            ax.axis['bottom'].set_visible(False)

        if not include_negative:
            # Trim response for negative frequencies
            ax.set_ylim([0, y_bounds[1]])

        # Minimize free space
        fig.tight_layout()

        # Save the figure
        fig.savefig(save_path)
