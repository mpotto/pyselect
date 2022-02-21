# src/pyselect/synthesizer.py
"""Data synthesizer for running experiments."""
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


def gregorova_se1(train_size: int = 10 ** 3, test_size: int = 500):
    """Gregorova *et al* article Synthetic Experiment 1."""
    noise = 0.1
    n_samples = train_size + 2 * test_size
    X = torch.randn(n_samples, 18)
    y = torch.sin(torch.square(X[:, 0] + X[:, 2])) * torch.sin(
        X[:, 6] * X[:, 7] * X[:, 8]
    )
    y_noise = y.unsqueeze(-1) + noise * torch.randn(n_samples).unsqueeze(-1)

    X_train = X[:train_size, :]
    y_train = y_noise[:train_size]

    y_train_mean = y_train.mean()

    y_train = y_train - y_train_mean
    X_val = X[train_size : train_size + test_size, :]
    y_val = y_noise[train_size : train_size + test_size] - y_train_mean
    X_test = X[train_size + test_size :, :]
    y_test = y_noise[train_size + test_size :] - y_train_mean

    return X_train, y_train, X_val, y_val, X_test, y_test


def gregorova_se2(train_size: int = 10 ** 3, test_size: int = 500):
    """Gregorova *et al* article Synthetic Experiment 2."""
    noise = 0.1
    n_samples = train_size + 2 * test_size
    X = torch.randn(n_samples, 100)
    y = torch.log(torch.square(torch.sum(X[:, 10:15], axis=1)))
    y_noise = y.unsqueeze(-1) + noise * torch.randn(n_samples)

    X_train = X[:train_size, :]
    y_train = y_noise[:train_size]

    y_train_mean = y_train.mean()

    y_train = y_train - y_train_mean
    X_val = X[train_size : train_size + test_size, :]
    y_val = y_noise[train_size : train_size + test_size] - y_train_mean
    X_test = X[train_size + test_size :, :]
    y_test = y_noise[train_size + test_size :] - y_train_mean
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_val_test_split(X, y, train_size, val_size, test_size):
    """Split dataset in train, validation and test parts."""
    X_train = X[:train_size, :]
    y_train = y[:train_size]
    X_val = X[train_size : train_size + val_size, :]
    y_val = y[train_size : train_size + val_size]
    X_test = X[train_size + val_size : train_size + val_size + test_size, :]
    y_test = y[train_size + val_size : train_size + val_size + test_size]
    return X_train, y_train, X_val, y_val, X_test, y_test
