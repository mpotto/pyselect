# experiments/gregorova/scripts/data.py
import os
from typing import Callable, Tuple

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Subset, TensorDataset, random_split

from pyselect.data import gregorova_se1, gregorova_se2


def generate_path(config: DictConfig) -> None:
    """Generate folder where data will be stored."""
    if not os.path.exists(config.path):
        os.mkdir(config.path)


def _label_transform(
    train_dataset: Subset, test_dataset: Subset
) -> Tuple[TensorDataset, TensorDataset]:
    """Transform label from train and test datasets for *Gregorova et al* datasets."""
    X_train, y_train = train_dataset.dataset[train_dataset.indices]
    X_test, y_test = test_dataset.dataset[test_dataset.indices]

    y_train_mean = y_train.mean()

    y_train -= y_train_mean
    y_test -= y_train_mean

    train = TensorDataset(X_train, y_train)
    test = TensorDataset(X_test, y_test)

    return train, test


def generate_dataset(config: DictConfig) -> None:
    """Generate *Gregorova et al* datasets according to configuration. All
    datasets are saved as TensorDataset.
    """
    name = config.name
    if name == "se1":
        _func = gregorova_se1
    elif name == "se2":
        _func = gregorova_se2

    # This is strictly needed.
    rng = torch.Generator().manual_seed(config.seed)

    train_size = config.train_size
    test_size = config.test_size
    n_samples = train_size + test_size
    X, y = _func(n_samples, generator=rng)

    greg_full = TensorDataset(X, y)
    greg_train, greg_test = random_split(
        greg_full, [train_size, test_size], generator=rng
    )

    greg_train_processed, greg_test_processed = _label_transform(greg_train, greg_test)

    train_name = f"{config.name}-{n_samples}-train.pt"
    test_name = f"{config.name}-{n_samples}-test.pt"

    print(
        f"Saving {config.name} train and test datasets with {n_samples} samples and seed {config.seed}."
    )
    generate_path(config)

    torch.save(greg_train_processed, os.path.join(config.path, train_name))
    torch.save(greg_test_processed, os.path.join(config.path, test_name))


@hydra.main(config_path="../config/data", config_name="data")
def main(config: DictConfig) -> None:

    os.chdir(hydra.utils.get_original_cwd())

    generate_dataset(config)


if __name__ == "__main__":
    main()
