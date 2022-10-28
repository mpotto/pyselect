import time
import joblib

import numpy as np
import optuna
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

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
X_train_val = np.load("data/amazon/processed2/X_train_reg.npy")
X_test = np.load("data/amazon/processed2/X_test_reg.npy")
y_train_val = np.load("data/amazon/processed2/y_train_reg.npy")
y_test = np.load("data/amazon/processed2/y_test_reg.npy")

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size, random_state=rng
)

# Subsample for finding best parameters
X_sub, y_sub = resample(
    X_train, y_train, n_samples=10 ** 4, stratify=y_train, random_state=rng
)


def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-7, 1e1, log=True)

    # Model
    model = Lasso(alpha=alpha)

    model.fit(X_sub, y_sub)

    model_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, model_pred)
    trial.set_user_attr("model", value=model)

    return mse


# Model
search_space = {
    "alpha": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
}

study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, callbacks=[best_model_callback])

best_model = study.user_attrs["best_model"]

t_start = time.time()
best_model.fit(X_train, y_train)
elapsed_time = time.time() - t_start

# Evaluate on test
best_model_pred = best_model.predict(X_test)
center, band = get_mse_confidence_interval(y_test, best_model_pred)
results.append([center, band, elapsed_time])

# 8) Save results.
np.savetxt("experiments/amazon_reg/results/lasso_results.txt", results)

joblib.dump(best_model, "experiments/amazon_reg/models/best_lasso.joblib")
