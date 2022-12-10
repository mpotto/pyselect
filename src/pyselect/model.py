from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class RFFLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sampler: Callable[[int, int], Tensor] = torch.randn,
    ):
        """Constructor of Random Fourier Features Layer."""
        super(RFFLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters
        self.precisions = nn.Parameter(torch.empty(in_features))

        # Buffers
        self.register_buffer(
            "_omega_sample",
            sampler(self.in_features, self.out_features),
            persistent=True,
        )
        self.register_buffer(
            "_unif_sample",
            torch.rand(self.out_features) * 2 * np.pi,
            persistent=True,
        )
        self.reset_parameters()

    def forward(self, x):
        # normalizing factor
        norm = torch.tensor(2.0 / self.out_features)
        output = torch.sqrt(norm) * torch.cos(
            (x * self.precisions) @ self._omega_sample + self._unif_sample
        )

        return output

    def reset_parameters(self, val=0.0):
        nn.init.constant_(self.precisions, val)

    def __repr__(self):
        return f"RFFLayer(in_features={self.in_features}, out_features={self.out_features})"


class RFFNet(nn.Module):
    def __init__(self, dims, sampler=torch.randn):
        """Constructor of the Random Fourier Features Network."""
        super(RFFNet, self).__init__()

        self.rff = RFFLayer(dims[0], dims[1], sampler)
        self.linear = nn.Linear(dims[1], dims[2], bias=False)

    def forward(self, x):
        """Perform a forward pass on the complete network."""
        random_features = self.rff(x)
        return self.linear(random_features)

    def cpu_state_dict(self):
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}

    def get_precisions(self):
        return self.rff.precisions.detach().cpu().numpy()


class ReluRFFNet(nn.Module):
    def __init__(self, dims, sampler=torch.randn, dropout=None):
        """Constructor of the Random Fourier Features Network."""
        super(ReluRFFNet, self).__init__()

        n_dims = len(dims)

        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.rff = RFFLayer(dims[0], dims[1], sampler)
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(1, n_dims - 1)]
        )

    def forward(self, x):
        """Perform a forward pass on the complete network."""
        current_layer = self.rff(x)
        for theta in self.layers:
            current_layer = theta(current_layer)
            if theta is not self.layers[-1]:
                if self.dropout is not None:
                    current_layer = self.dropout(current_layer)
                current_layer = F.relu(current_layer)
        return current_layer

    def cpu_state_dict(self):
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}

    def get_precisions(self):
        return self.rff.precisions.detach().cpu().numpy()
