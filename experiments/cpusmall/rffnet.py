import os
import time
import joblib

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from pyselect import RFFNetRegressor
from pyselect.utils import best_model_callback, get_mse_confidence_interval, get_folder

test_size = 10 ** 3
train_size = 6 * 10 ** 3
val_size = 10 ** 3

metrics = []

seed_sequence = np.random.SeedSequence(entropy=0)
seed = seed_sequence.generate_state(1)[0]

# Dataset
data = pd.read_csv("data/processed/cpusmall.csv")
X = data.drop(["target"], axis=1).to_numpy()
y = np.ravel(data[["target"]]).to_numpy()

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, train_size=train_size + val_size, test_size=test_size, random_state=0
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 1e-7, 1, log=True)

    # Model
    model = RFFNetRegressor(
        lr=lr,
        alpha=alpha,
        validation_fraction=0.1,
        n_iter_no_change=30,
        bath_size=train_size // 10,
        torch_seed=seed,
        random_state=0,
    )

    t_start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - t_start

    model_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, model_pred)
    trial.set_user_attr("model", value=model)
    trial.set_user_attr("fitting_time", value=elapsed_time)

    return mse


search_space = {
    "lr": [1e-4, 1e-3, 1e-2],
    "alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
}

study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, callbacks=[best_model_callback])

best_model = study.user_attrs["best_model"]

best_model_pred = best_model.predict(X_test)

center, band = get_mse_confidence_interval(y_test, best_model_pred)
metrics.append([center, band, study.best_trial.user_attrs["fitting_time"]])

# Save results
relevances_folder = get_folder("eval/benchmarks/rffnet/cpusmall/precisions")
metrics_folder = get_folder("eval/benchmarks/rffnet/cpusmall/metrics")
models_folder = get_folder("eval/benchmarks/rffnet/cpusmall/models")

precisions = best_model.precisions_

np.savetxt(f"{relevances_folder}/precisions.txt", precisions)
np.savetxt(f"{metrics_folder}/metrics.txt", metrics)
joblib.dump(best_model, f"{models_folder}/model.joblib")
