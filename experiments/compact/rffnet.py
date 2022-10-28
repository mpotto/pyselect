import os
import time
import joblib

import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from pyselect import RFFNetRegressor
from pyselect.utils import best_model_callback, get_mse_confidence_interval

test_size = 10 ** 3
train_size = 6 * 10 ** 3
val_size = 10 ** 3

results = []

seed_sequence = np.random.SeedSequence(entropy=0)
seed = seed_sequence.generate_state(1)[0]
rng = np.random.RandomState(0)

# Dataset
filename = os.path.join("data/compact/compact/ComputerActivity", "cpu_act.data")
data = np.loadtxt(filename, delimiter=",")

X, y = data[:, 0:21], data[:, 21].reshape(-1, 1)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, train_size=train_size + val_size, test_size=test_size, random_state=rng
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size, random_state=rng
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 1e-7, 1e-1, log=True)

    # Model
    model = RFFNetRegressor(
        lr=lr,
        alpha=alpha,
        validation_fraction=0.1,
        n_iter_no_change=30,
        torch_seed=seed,
        random_state=rng,
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
    "lr": [1e-5, 1e-4, 1e-3, 1e-2],
    "alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
}

study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, callbacks=[best_model_callback])

best_model = study.user_attrs["best_model"]

best_model_pred = best_model.predict(X_test)
center, band = get_mse_confidence_interval(y_test, best_model_pred)
results.append([center, band, study.best_trial.user_attrs["fitting_time"]])

# 7) Plot RFF precisions
labels = np.arange(0, X.shape[1])
plt.figure()
plt.stem(np.abs(best_model.precisions_))
plt.ylabel(r"$\lambda$")
plt.xticks(labels[::2], labels[::2] + 1)

plt.tight_layout()
plt.show()

# 8) Save results.

np.savetxt("experiments/compact/results/rffnet_results.txt", results)
np.savetxt("experiments/compact/results/rffnet_precisions.txt", best_model.precisions_)
joblib.dump(best_model, "experiments/compact/models/reg_rff.joblib")
