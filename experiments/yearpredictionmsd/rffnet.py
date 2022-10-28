import time
import joblib

import numpy as np
import optuna
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

from pyselect.estimators import RFFNetRegressor
from pyselect.utils import best_model_callback, get_mse_confidence_interval

val_size = 5 * 10 ** 3

results = []

# 1) Generate seeds
# torch
seed_sequence = np.random.SeedSequence(entropy=0)
seed = seed_sequence.generate_state(1)[0]

# sklearn
rng = np.random.RandomState(0)

# Dataset
X_train = np.load("data/yearpredictionmsd/processed/X_train.npy")
X_val = np.load("data/yearpredictionmsd/processed/X_val.npy")
X_test = np.load("data/yearpredictionmsd/processed/X_test.npy")
y_train = np.load("data/yearpredictionmsd/processed/y_train.npy").reshape(-1, 1)
y_val = np.load("data/yearpredictionmsd/processed/y_val.npy").reshape(-1, 1)
y_test = np.load("data/yearpredictionmsd/processed/y_test.npy").reshape(-1, 1)

# Subsample for finding best parameters
X_sub, y_sub = resample(
    X_train, y_train, n_samples=10 ** 4, stratify=y_train, random_state=rng
)


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

    model.fit(X_sub, y_sub)

    model_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, model_pred)
    trial.set_user_attr("model", value=model)

    return mse


# Model
search_space = {
    "lr": [1e-5, 1e-4, 1e-3, 1e-2],
    "alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
}

study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, callbacks=[best_model_callback])

best_model = study.user_attrs["best_model"]
best_model.verbose = True
best_model.log_rate = 1

t_start = time.time()
best_model.fit(X_train, y_train)
elapsed_time = time.time() - t_start

# Evaluate on test
best_model_pred = best_model.predict(X_test)
center, band = get_mse_confidence_interval(y_test, best_model_pred)
results.append([center, band, elapsed_time])

# 7) Plot RFF bandwidths
labels = np.arange(0, X_train.shape[1])
plt.figure()
plt.stem(np.abs(best_model.precisions_))
plt.ylabel(r"$\sigma$")
plt.xticks(labels[::1], labels[::1] + 1)

plt.tight_layout()
plt.show()

# 8) Save results.
np.savetxt("experiments/yearpredictionmsd/results/rffnet_results.txt", results)
np.savetxt(
    "experiments/yearpredictionmsd/results/rffnet_precisions.txt",
    best_model.precisions_,
)
joblib.dump(best_model, "experiments/yearpredictionmsd/models/reg_rff.joblib")
