# experiments/gregorova/scripts/training.py
import functools
import logging
import os

import hydra
import optuna
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from pyselect.training import (
    best_model_callback,
    check_study_exists_and_load,
    configurable_objetive,
    get_n_samples,
    save_study,
    get_dataloaders,
)

logger = logging.getLogger("ignite.engine.engine")
logger.setLevel(logging.WARNING)


@hydra.main(config_path="../config/rffnet", config_name="rffnet")
def main(config: DictConfig) -> None:

    os.chdir(get_original_cwd())

    # Configure torch global seeding to keep layer initialization stable.
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    n_samples = get_n_samples(config)
    study_name = f"{config.dataset.name}-{n_samples}.pkl"
    study = check_study_exists_and_load(study_name=study_name)

    if study is None:
        sampler = optuna.samplers.TPESampler(seed=config.tuner.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

    train_loader, val_loader = get_dataloaders(config)

    objective = functools.partial(
        configurable_objetive,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    study.optimize(
        objective, n_trials=config.tuner.n_trials, callbacks=[best_model_callback]
    )

    save_study(study, study_name=study_name)


if __name__ == "__main__":
    main()
