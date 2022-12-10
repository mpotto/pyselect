import time
import joblib

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ARDRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from pyselect.utils import best_model_callback, get_mse_confidence_interval, get_folder

test_size = 10 ** 3
train_size = 6 * 10 ** 3
val_size = 10 ** 3

metrics = []

# Dataset
data = pd.read_csv("data/processed/cpusmall.csv")
X = data.drop(["target"], axis=1)
y = np.ravel(data[["target"]])

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
    lambda_1 = trial.suggest_float("lambda_1", 1e-8, 1e-4, log=True)
    lambda_2 = trial.suggest_float("lambda_2", 1e-8, 1e-4, log=True)

    # Model
    model = ARDRegression(tol=1e-5, lambda_1=lambda_1, lambda_2=lambda_2)

    t_start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - t_start

    model_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, model_pred)
    trial.set_user_attr("model", value=model)
    trial.set_user_attr("elapsed_time", value=elapsed_time)

    return mse


# Model
search_space = {
    "lambda_1": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
    "lambda_2": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
}

study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, callbacks=[best_model_callback])

best_model = study.user_attrs["best_model"]

best_model_pred = best_model.predict(X_test)

center, band = get_mse_confidence_interval(y_test, best_model_pred)
metrics.append([center, band, study.best_trial.user_attrs["elapsed_time"]])

precisions = best_model.lambda_

# Save results
relevances_folder = get_folder("eval/benchmarks/ard/cpusmall/precisions")
metrics_folder = get_folder("eval/benchmarks/ard/cpusmall/metrics")
models_folder = get_folder("eval/benchmarks/ard/cpusmall/models")

np.savetxt(f"{relevances_folder}/precisions.txt", precisions)
np.savetxt(f"{metrics_folder}/metrics.txt", metrics)
joblib.dump(best_model, f"{models_folder}/model.joblib")
