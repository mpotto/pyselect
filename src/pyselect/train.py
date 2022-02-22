# src/pyselect/train.py
"""Functions used for training and optimizing pyselect models."""
import torch


def ridge_loss(y_pred, y_true, model, reg_param):
    weights = model.rff_net[2].weight.squeeze()
    loss = torch.mean(torch.square(y_pred - y_true)) + reg_param * weights.dot(weights)
    return loss


def score_function(engine):
    val_loss = engine.state.metrics["loss"]
    return -val_loss


def best_model_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])
