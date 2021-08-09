# src/pyselect/synthesizer.py
"""Data synthesizer for running experiments."""
# TODO: add a class that inherits from Dataset and DataLoader
import numpy as np
import torch


def make_regression_with_tensors(
    n_samples: int = 100,
    n_features: int = 100,
    n_informative: int = 10,
    bias: float = 0.0,
    noise: float = 1.0,
):
    """Create a synthetic regression dataset. The covariates are sampled from a standard normal distribution. The response variable is a linear model with non-zero coefficients only for a subset of `n_informative` covariates plus a normally distributed noise.

    Args:
        n_samples: Number of samples in the dataset.
        n_features: Number of features (covariates) in the dataset.
        n_informative: Number of informative features.
        n_targets: Number of targets (response variables)
        bias: Bias of the linear model.
        noise: Standard deviation of the noise.

    Returns:
        X: Design matrix.
        y: Response variable.
        coef: Coefficients of the linear model.
    """
    X = torch.randn(n_samples, n_features)
    coef = torch.zeros(n_features)
    relevant_feat = np.random.choice(n_features, n_informative, replace=False)
    coef[relevant_feat] = torch.randn(n_informative)
    y = X @ coef.unsqueeze_(-1) + noise * torch.randn(n_samples).unsqueeze(-1) + bias
    return X, y, coef


def gregorova_se1(n_samples: int = 100, n_features: int = 18, noise: float = 0.0):
    """Gregorova *et al* article Synthetic Experiment 1."""
    X = torch.randn(n_samples, n_features)

    y = torch.sin((X[:, 0] + X[:, 2]) ** 2) * torch.sin(X[:, 6] * X[:, 7] * X[:, 8])
    y_noise = y.unsqueeze(-1) + noise * torch.randn(n_samples).unsqueeze(-1)
    return X, y_noise
