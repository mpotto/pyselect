# src/pyselect/layers.py
"""Custom layers for pyselect."""
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, Generator


class HadamardLayer(nn.Module):
    """Hadamard layer of bandwidths with constant initialization.

    Args:
        in_features (int): Number of features in the input tensor.
        val (float): Initial value of the bandwidths vector entries. Default value is 0.0
    """

    # TODO: consistent broadcasting, bandwidth initialization, named tensors
    def __init__(self, in_features: int, val=0.0):
        """Constructor of the Hadamard layer of bandwidths."""
        super(HadamardLayer, self).__init__()
        self.in_features = in_features
        self.val = val
        self.bandwidths = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(self.val)

    def forward(self, x):
        """Scale the input vector by the bandwidths."""
        return x * self.bandwidths

    def reset_parameters(self, val):
        """Initialize vector of bandwidths with specified value."""
        nn.init.constant_(self.bandwidths, val)

    def __repr__(self):
        return f"HadamardLayer(in_features={self.in_features})"


class RandomFourierFeaturesLayer(nn.Module):
    """Random Fourier Features layer implementation.

    Attributes:
        in_features: Number of features in the input tensor.
        out_features: Number of features in the output tensor.
        sampler: Callable that samples from the chosen distribution. Determines
        the kernel the RFF method is aproximating.
        generator: torch generator object that manages the state of algorithm
        that produes pseudo random numbers
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sampler: Callable[[int, int, Generator], Tensor] = torch.randn,
        generator: Generator = torch.default_generator,
    ):
        """Constructor of Random Fourier Features Layer."""
        super(RandomFourierFeaturesLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "_omega_sample",
            sampler(self.in_features, self.out_features, generator=generator),
            persistent=True,
        )
        self.register_buffer(
            "_unif_sample",
            torch.rand(self.out_features, generator=generator) * 2 * np.pi,
            persistent=True,
        )

    def forward(self, x):
        """Compute the approximate feature map for the input vector."""
        # normalizing factor for LLN.
        norm = torch.tensor(2.0 / self.out_features)
        output = torch.sqrt(norm) * torch.cos(
            x @ self._omega_sample + self._unif_sample
        )
        return output

    def __repr__(self):
        return f"RandomFourierFeaturesLayer(in_features={self.in_features}, out_features={self.out_features})"
