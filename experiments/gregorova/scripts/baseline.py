# experiments/gregorova/scripts/baseline.py
"""High-level script to run SRFF algorithm from *Gregorová et al.*, which we consider as a baseline for performance.
"""
import os
from typing import Dict, Tuple

import hydra
import joblib
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import random_split

from pyselect.gregorova.gregorova import srf_run
from pyselect.training import get_datasets, get_n_samples


def split_datasets(
    config: DictConfig,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Get datasets and split into training, validation and test tensors for features and labels.

    Args:
        config (DictConfig): configuration dictionary.
    """

    train_val, test = get_datasets(config)

    # Seed should be the same as used in the rff method.
    rng = torch.Generator().manual_seed(config.seed)

    train, val = random_split(
        train_val, [config.dataset.train_size, config.dataset.val_size], generator=rng
    )

    # Train and test are Subsets, hence the following.
    X_train, y_train = train.dataset[train.indices]
    X_val, y_val = val.dataset[val.indices]

    # Test is a TensorDataset, has attribute `tensors`
    X_test, y_test = test.tensors

    return X_train, y_train, X_val, y_val, X_test, y_test


def save_results(config: DictConfig, train_map, val_map, test_map) -> None:
    """Save SRFF results as pickle objects.

    Args:
        config: configuration dictionary.
        train_map: dictionary of results from training.
        val_map: dictionary of results from validation.
        test_map: dictionary of results from testing.
    """
    if not os.path.exists("../baselines"):
        os.mkdir("../baselines")
    n_samples = get_n_samples(config)
    base_name = f"../baselines/{config.dataset.name}-{n_samples}-"
    joblib.dump(train_map, base_name + "train.pkl")
    joblib.dump(val_map, base_name + "val.pkl")
    joblib.dump(test_map, base_name + "test.pkl")


@hydra.main(config_path="../config/baseline", config_name="gregorova")
def main(config: DictConfig) -> None:

    os.chdir(get_original_cwd())
    torch.manual_seed(config.seed)

    X_train, y_train, X_val, y_val, X_test, y_test = split_datasets(config)

    # Gregorova globals.
    max_iter_gamma = config.max_iter_gamma
    max_iter_srf = config.max_iter_srf
    num_avg_samples = config.num_avg_samples
    update_threshold = config.update_threshold
    out_features = config.out_features
    kernel_param = config.kernel_param
    lambda_min = config.lambda_min
    lambda_max = config.lambda_max
    lambda_step = config.lambda_step

    train_map, val_map, test_map = srf_run(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    save_results(config, train_map, val_map, test_map)


if __name__ == "__main__":
    main()
