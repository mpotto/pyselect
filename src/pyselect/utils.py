import numpy as np


def get_mse_confidence_interval(y_true, y_pred):
    diff_squared = np.square(y_true - y_pred)
    center = diff_squared.mean()

    sample_variance = diff_squared.var()
    band = 2 * np.sqrt(sample_variance / len(y_true))
    return center, band


def best_model_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["model"])
