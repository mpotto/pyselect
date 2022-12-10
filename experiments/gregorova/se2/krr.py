import time
import joblib

import numpy as np
import optuna
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from pyselect.datasets import make_gregorova_se2
from pyselect.utils import best_model_callback, get_mse_confidence_interval, get_folder

n_samples = 54 * 10 ** 3
test_size = 2 * 10 ** 3
val_size = 2 * 10 ** 3

metrics = []

# 2) Generate data
X, y = make_gregorova_se2(n_samples, random_state=0)
y = y.ravel()

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=test_size, random_state=0
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 1e-7, 1e-1, log=True)

    # Model
    model = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)

    t_start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - t_start

    model_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, model_pred)
    trial.set_user_attr("model", value=model)
    trial.set_user_attr("elapsed_time", value=elapsed_time)

    return mse


search_space = {
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "gamma": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
}

study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, callbacks=[best_model_callback])

best_model = study.user_attrs["best_model"]

best_model_pred = best_model.predict(X_test)

center, band = get_mse_confidence_interval(y_test, best_model_pred)
metrics.append([center, band, study.best_trial.user_attrs["elapsed_time"]])

# Save results
metrics_folder = get_folder("eval/benchmarks/krr/se2/metrics")
models_folder = get_folder("eval/benchmarks/krr/se2/models")

np.savetxt(f"{metrics_folder}/metrics.txt", metrics)
joblib.dump(best_model, f"{models_folder}/model.joblib")
