# src/pyselect/training.py
"""
High-level functions to ease model training from hydra configuration files.
"""
import os
from typing import Tuple, Union

import joblib
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.engine.engine import Engine
from ignite.handlers import BasicTimeProfiler, EarlyStopping, LRScheduler
from ignite.metrics import Loss, RootMeanSquaredError
from omegaconf import DictConfig
from optuna.integration.pytorch_ignite import PyTorchIgnitePruningHandler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split

from pyselect.networks import RandomFourierFeaturesNet


def check_study_exists_and_load(
    study_path: str = "../studies", study_name: str = None
) -> Union[optuna.Study, None]:
    """Check if Optuna study exists.

    Args:
        study_path (str): path to studies folder.
        study_name (str): filename of study whose existence is checked.
    """
    full_path = os.path.join(study_path, study_name)
    study = None
    if os.path.exists(full_path):
        study = joblib.load(full_path)
    return study


def save_study(
    study: optuna.Study, study_path: str = "../studies", study_name: str = None
) -> None:
    """Save Optuna study."""
    if not os.path.exists(study_path):
        os.mkdir(study_path)
    full_name = os.path.join(study_path, study_name)
    joblib.dump(study, full_name)


def get_n_samples(config: DictConfig) -> int:
    """Get number of samples on full dataset.

    Args:
        config (DictConfig): training configuration dictionary.
    """
    n_samples = (
        config.dataset.train_size + config.dataset.test_size + config.dataset.val_size
    )
    return n_samples


def get_device(config: DictConfig) -> torch.device:
    """Check configuration device and its torch availability.

    Args:
        config (DictConfig): configuration dictionary containing device key.
    """
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_datasets(config: DictConfig) -> Tuple[Dataset, Dataset]:
    """Load datasets saved on memory as TensorDataset.

    Args:
        config (DictConfig): configuration dictionary.
    """
    n_samples = get_n_samples(config)

    train_name = f"{config.dataset.name}-{n_samples}-train.pt"
    test_name = f"{config.dataset.name}-{n_samples}-test.pt"

    train = torch.load(os.path.join(config.dataset.path, train_name))
    test = torch.load(os.path.join(config.dataset.path, test_name))

    return train, test


def get_dataloaders(config: DictConfig) -> Tuple[DataLoader, DataLoader]:
    """Load train and test datasets saved on memory as TensorDataset and configure DataLoaders for training and validation by splitting the train dataset.

    Args:
        config (DictConfig): configuration dictionary.
    """
    train_val, _ = get_datasets(config)

    # Caution: random object. Seed should be the same as in the baseline.
    rng = torch.Generator().manual_seed(config.seed)
    train, val = random_split(
        train_val,
        [config.dataset.train_size, config.dataset.val_size],
        generator=rng,
    )

    train_loader = DataLoader(
        train,
        batch_size=config.dataloader.batch_size,
    )
    val_loader = DataLoader(
        val,
        batch_size=config.dataloader.batch_size,
    )

    return train_loader, val_loader


def get_optimizer(
    config: DictConfig, model: nn.Module, trial: optuna.Trial
) -> optim.Optimizer:
    """Configure optimizer for hyperarmeter tuning.

    Args:
        config (DictConfig): configuration dictionary.
        model (nn.Module): RandomFourierFeaturesNet model.
        trial (optuna.Trial): tuning trial from Optuna study.

    Note: the regularization parameter is not divided by two in our optimization problem. That's why `weight_decay` is 2 * reg_param.
    """

    lr = trial.suggest_float(
        "learning_rate",
        config.optimizer.lr.low,
        config.optimizer.lr.high,
        log=config.optimizer.lr.log,
    )
    reg_param = trial.suggest_float(
        "reg_param",
        config.optimizer.reg_param.low,
        config.optimizer.reg_param.high,
        log=config.optimizer.reg_param.log,
    )

    optimizer = optim.SGD(
        [
            {"params": model.rff_net[0].parameters()},
            {"params": model.rff_net[2].weight, "weight_decay": 2 * reg_param},
            {"params": model.rff_net[2].bias},
        ],
        lr=lr,
        momentum=0.9,
    )

    return optimizer


def get_scheduler(
    config: DictConfig, optimizer: optim.Optimizer, train_loader: DataLoader
) -> LRScheduler:
    """Configure learning rate decay scheduler.

    Args:
        config (DictConfig): configuration dictionary.
        optimizer (optim.Optimizer): optimizer object whose learning rate decay is being scheduled.
        train_loader (DataLoader): dataloader for training dataset.
    """
    step_size = len(train_loader) * config.n_epochs // config.scheduler.n_steps
    torch_scheduler = StepLR(
        optimizer, step_size=step_size, gamma=config.scheduler.gamma
    )
    scheduler = LRScheduler(torch_scheduler)
    return scheduler


def score_function(engine: Engine) -> float:
    """Score function for Ignite's early stopping handler.

    Args:
        engine (Engine): ignite engine where the early stopping handler will be attached, e.g. a validation evaluator.
    """
    val_loss = engine.state.metrics["rmse"]
    return -val_loss


def get_early_stopping(config: DictConfig, trainer: Engine) -> EarlyStopping:
    """Configure early stopping handler.

    Args:
        config (DictConfig): configuration dictionary.
        trainer (Engine): trainer object where early stopping will possibly trigger training interruption.
    """
    early_stopping = EarlyStopping(
        patience=config.early_stopping.patience,
        score_function=score_function,
        trainer=trainer,
    )
    return early_stopping


def get_model(config: DictConfig, trial: optuna.Trial) -> nn.Module:
    """Configure RandomFourierFeaturesNet model with uniform bandwidths initialization.

    Args:
        config (DictConfig): configuration dictionary.
        trial (optuna.Trial): tuning trial from Optuna study.

    Notes: weights fom the `Linear` layer are randomly initialized, but the generator object used in the process cannot be set locally. Consequently, these weights can only be reproduced across multiple runs if the torch.default_generator is apropriately seeded.
    """
    device = get_device(config)

    # Experience suggests out_features choice can regularize the model.
    out_features = trial.suggest_int(
        "out_features",
        config.model.out_features.low,
        config.model.out_features.high,
        log=config.model.out_features.log,
    )
    model = RandomFourierFeaturesNet(config.model.in_features, out_features)

    # Initialization policy for bandwidths.
    model.rff_net[0].reset_parameters(1 / config.model.in_features)
    return model.to(device)


def get_time_profiler() -> BasicTimeProfiler:
    """Configure a basic (non-detaile) time profiler handler."""
    return BasicTimeProfiler()


def get_pruner(trainer: Engine, trial: optuna.Trial) -> PyTorchIgnitePruningHandler:
    """Configure Optuna pruner handler for Ignite.

    Args:
        trainer (Engine): trainer object containing metrics that will be used to decide if trial should be pruned.
        trial (optuna.Trial): tuning trial from Optuna study.
    """
    pruner = PyTorchIgnitePruningHandler(trial, "rmse", trainer)
    return pruner


def best_model_callback(study: optuna.Study, trial: optuna.Trial) -> None:
    """Callback to save best model (not only tuning best params) during Optuna hyperparameter tuning.

    Args:
        study (optuna.Study): Optuna study containing objective value during tuning.
        trial (optuna.Trial): current trial on Optuna study.
    """
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["model"])


def log_training_results(engine: Engine, evaluator: Engine, loader: DataLoader) -> None:
    """Log training results."""
    evaluator.run(loader)
    loss = evaluator.state.metrics["loss"]
    rmse = evaluator.state.metrics["rmse"]
    print(f"Training - Epoch: {engine.state.epoch} Loss: {loss:.5f} RMSE: {rmse:.5f}")


def log_validation_results(
    engine: Engine, evaluator: Engine, loader: DataLoader
) -> None:
    """Log validation results."""
    evaluator.run(loader)
    loss = evaluator.state.metrics["loss"]
    rmse = evaluator.state.metrics["rmse"]
    print(f"Validation - Epoch: {engine.state.epoch} Loss: {loss:.5f} RMSE: {rmse:.5f}")


def log_lr(optimizer: optim.Optimizer) -> None:
    """Log learning rate."""
    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.4f}")


def configurable_objetive(
    trial: optuna.Trial,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    config: DictConfig = None,
) -> float:
    """Expand usual Optuna objective's parameters to allow further configuration. We construct the usual objective function by partial application of additional parameters (loaders and configuration).

    Args:
        trial (optuna.Trial): current tuning trial from Optuna study.
        train_loader (DataLoader): dataloader for training dataset.
        val_loader (DataLoader): dataloader for validation dataset.
        config (DictConfig): configuration dictionary.

    Notes: Ignite logging is intentionally disabled. Only optuna logging will be visible.
    """
    device = get_device(config)
    model = get_model(config, trial)

    optimizer = get_optimizer(config, model, trial)

    loss_fn = nn.MSELoss()

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device)

    # TODO (mpotto): should we select hyperpameters using the Ridge Loss?
    metrics = {"rmse": RootMeanSquaredError(), "loss": Loss(loss_fn)}
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    scheduler = get_scheduler(config, optimizer, train_loader)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config.log_rate),
        log_training_results,
        trainer,
        train_evaluator,
        train_loader,
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config.log_rate),
        log_validation_results,
        trainer,
        val_evaluator,
        val_loader,
    )

    time_profiler = get_time_profiler()
    time_profiler.attach(trainer)

    trainer.run(train_loader, max_epochs=config.n_epochs)

    val_evaluator.run(val_loader)
    val_loss = val_evaluator.state.metrics["rmse"]

    trial.set_user_attr(key="model", value=model)
    trial.set_user_attr(key="time_profiler", value=time_profiler)

    return val_loss
