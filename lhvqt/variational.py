# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .utils import *

# Regular imports
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.nn.functional as F
import torch.nn as nn
import torch
import math


class VariationalDropoutConv1d(nn.Conv1d):
    """
    Implements a standard 1D convolutional layer (no bias) with variational dropout,
    as proposed in http://proceedings.mlr.press/v70/molchanov17a.html with heavy
    inspiration from https://github.com/kefirski/variational_dropout.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, threshold=math.inf, log_sigma2=-20):
        """
        Initialize the convolutional layer and establish parameter defaults in function signature.

        Parameters
        ----------
        in_channels : int
          Number of feature channels incoming
        out_channels : int
          Number of filters in the convolutional layer
        kernel_size : int
          Number of weights in each filter (receptive field)
        stride : int
          Number of samples between hops
        threshold : float
          Value for log_alpha above which corresponding weights become zero, e.g. 3 -> alpha /approx 20
        log_sigma2 : float
          Initial value of log sigma ^ 2 (determines initial value of alpha), e.g. -10 -> sigma ^ 2 /approx 4.5e-5
          Note : sigma ^ 2 can be thought of as the variance of the weights
               - this can cause problems if set too low
        """

        super(VariationalDropoutConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, bias=False)

        self.threshold = threshold

        # Initialize the variance term with same dimensionality as weights
        self.log_sigma2 = Parameter(torch.FloatTensor(self.weight.size()).fill_(log_sigma2))

        # Initialize the field for the dropout rate
        self.log_alpha = None

    def forward(self, in_feats):
        """
        Feed features through the convolutional layer, adding Gaussian noise
        (with learned individual rates) to each weight during training. During
        inference, weights with corresponding log alpha (dropout rate) beyond
        the chosen threshold are zeroed.

        Parameters
        ----------
        in_feats : Tensor (B x C x T)
          Incoming features for a batch,
          B - batch size
          C - number of input channels
          T - number of samples (a.k.a. sequence length)

        Returns
        ----------
        out_feats : Tensor (B x F x N)
          Outgoing features calculated for a batch,
          B - batch size
          F - dimensionality of features (number of filters)
          N - number of time/spatial steps (frames)
        """

        # Infer log alpha from log sigma ^ 2 and the current weights
        # sigma ^ 2 = alpha * theta ^ 2 -> log alpha = log sigma ^ 2 - log theta ^ 2
        self.log_alpha = self.log_sigma2 - torch.log(self.weight ** 2 + EPSILON)

        if self.training:
            # Convolve the input features with the mean of the weights to obtain the mean of the output
            gamma = F.conv1d(in_feats, weight=self.weight, stride=self.stride)

            # Compute the variance of the weights
            sigma2 = torch.exp(self.log_sigma2)

            # Convolve the (squared) input features with the variance to obtain the variance of the output
            delta = F.conv1d(in_feats ** 2, weight=sigma2, stride=self.stride)

            # Compute the standard deviation of the output
            sqrt_delta = torch.sqrt(delta + EPSILON)

            # Sample from standard normal distribution
            noise = Variable(torch.randn(*gamma.size())).to(in_feats.device)

            # Add the Gaussian noise to the outgoing features
            out_feats = gamma + sqrt_delta * noise
        else:
            # Determine which weights should be zeroed
            mask = self.log_alpha > self.threshold
            # Feed the input features through the convolutional layer with masked weights
            out_feats = F.conv1d(in_feats,
                                 weight=self.weight.masked_fill(mask, 0),
                                 stride=self.stride)

        return out_feats

    def kld(self):
        """
        Compute the approximate KL-divergence of the current weights.

        Returns
        ----------
        kld : float
          KL-divergence averaged across all weights of the layer
        """

        # Terms for approximate KL-divergence
        k = [0.63576, 1.87320, 1.48695]

        # Compute the approximate negative KL-divergence
        nkl = (k[0] * torch.sigmoid(k[1] + k[2] * self.log_alpha) -
               0.5 * torch.log(1 + torch.exp(-self.log_alpha)) - k[0])

        # Flip sign and average the KL-divergence across the weights
        kld = -torch.sum(nkl) / (self.in_channels * self.out_channels * self.kernel_size[0])

        return kld
