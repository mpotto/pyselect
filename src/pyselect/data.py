# src/pyselect/synthesizer.py
"""Data synthesizer for running experiments."""
import numpy as np
import torch
from torch import Generator


def make_regression_with_tensors(
    n_samples: int = 100,
    n_features: int = 100,
    n_informative: int = 10,
    bias: float = 0.0,
    noise: float = 1.0,
    generator: Generator = torch.default_generator,
):
    """Create a synthetic regression dataset. The covariates are sampled from
    a standard normal distribution. The response variable is a linear model
    with non-zero coefficients only for a subset of `n_informative` covariates
    plus a normally distributed noise.

    Args:
        n_samples: Number of samples in the dataset.
        n_features: Number of features (covariates) in the dataset.
        n_informative: Number of informative features.
        n_targets: Number of targets (response variables)
        bias: Bias of the linear model.
        noise: Standard deviation of the noise.
        seed: integer random state for dataset generation.

    Returns:
        X: Design matrix.
        y: Response variable.
        coef: Coefficients of the linear model.
    """
    X = torch.randn(n_samples, n_features, generator=generator)
    coef = torch.zeros(n_features)
    relevant_feat = torch.multinomial(
        torch.ones(n_features) / n_features, n_informative, generator=generator
    )
    coef[relevant_feat] = torch.randn(n_informative, generator=generator)
    y = (
        X @ coef.unsqueeze_(-1)
        + noise * torch.randn(n_samples, generator=generator).unsqueeze(-1)
        + bias
    )
    return X, y, coef


def gregorova_se1(
    n_samples: int = 2000,
    noise: float = 0.1,
    generator: Generator = torch.default_generator,
):
    """Gregorova *et al* article Synthetic Experiment 1."""
    X = torch.randn(n_samples, 18, generator=generator)
    y_noise = torch.sin(torch.square(X[:, 0] + X[:, 2])) * torch.sin(
        X[:, 6] * X[:, 7] * X[:, 8]
    ) + noise * torch.randn(n_samples, generator=generator)
    y = y_noise.unsqueeze(-1)
    return X, y


def gregorova_se2(
    n_samples: int = 2000,
    noise: float = 0.1,
    generator: Generator = torch.default_generator,
):
    """Gregorova *et al* article Synthetic Experiment 2."""
    X = torch.randn(n_samples, 100, generator=generator)
    y_noise = torch.log(
        torch.square(torch.sum(X[:, 10:15], axis=1))
    ) + noise * torch.randn(n_samples, generator)
    y = y_noise.unsqueeze(-1)
    return X, y


def jordan_se1(
    n_samples: int = 200,
    n_features: int = 2,
    rho: float = 0.5,
    noise_level: float = 0.1,
    generator: Generator = torch.default_generator,
):
    """Jordan *et al* article Synthetic Experiment 1."""
    features_indices = torch.arange(1, n_features + 1)
    col_indices, row_indices = torch.meshgrid(
        features_indices, features_indices, indexing="xy"
    )
    power_matrix = torch.abs(col_indices - row_indices)
    cov = torch.Tensor(rho ** power_matrix)
    loc = torch.zeros(n_features)

    rng = np.random.default_rng(seed=generator.get_state().numpy())
    X = torch.Tensor(rng.multivariate_normal(loc, cov, size=n_samples))

    y_noise = X[:, 0] + noise_level * torch.randn(n_samples, generator=generator)
    y = y_noise.unsqueeze(-1)
    return X, y


def jordan_se2(
    n_samples: int = 300,
    n_features: int = 10,
    rho: float = 0.5,
    noise_level: float = 0.1,
    generator: Generator = torch.default_generator,
):
    """Jordan *et al* article Synthetic Experiment 2."""
    features_indices = torch.arange(1, n_features + 1)
    col_indices, row_indices = torch.meshgrid(
        features_indices, features_indices, indexing="xy"
    )
    power_matrix = torch.abs(col_indices - row_indices)
    cov = torch.Tensor(rho ** power_matrix)
    loc = torch.zeros(n_features)

    rng = np.random.default_rng(seed=generator.get_state().numpy())
    X = torch.Tensor(rng.multivariate_normal(loc, cov, size=n_samples))

    y_noise = (
        X[:, 0] ** 3
        + X[:, 1] ** 3
        + noise_level * torch.randn(n_samples, generator=generator)
    )
    y = y_noise.unsqueeze(-1)
    return X, y


def jordan_se3(
    n_samples: int = 200,
    n_features: int = 10,
    noise_level: float = 0.1,
    generator: Generator = torch.default_generator,
):
    """Jordan *et al* article Synthetic Experiment 3."""
    X = torch.randn((n_samples, n_features), generator=generator)
    y_noise = X[:, 0] * X[:, 1] + noise_level * torch.randn(
        n_samples, generator=generator
    )
    y = y_noise.unsqueeze(-1)
    return X, y
