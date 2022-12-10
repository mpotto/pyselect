import os

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


def get_folder(folder_path, verbose=True):
    """Creates folder, if it doesn't exist, and returns folder path.
    Args:
        folder_path (str): Folder path, either existing or to be created.
    Returns:
        str: folder path.
    """
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        if verbose:
            print(f"-created directory {folder_path}")
    return folder_path
