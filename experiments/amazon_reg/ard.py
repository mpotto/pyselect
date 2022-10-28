import time
import joblib

import numpy as np
import optuna
from sklearn.linear_model import ARDRegression
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
    lambda_1 = trial.suggest_float("lambda_1", 1e-8, 1e-4, log=True)
    lambda_2 = trial.suggest_float("lambda_2", 1e-8, 1e-4, log=True)

    # Model
    model = ARDRegression(tol=1e-5, lambda_1=lambda_1, lambda_2=lambda_2)

    model.fit(X_sub, y_sub)

    model_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, model_pred)
    trial.set_user_attr("model", value=model)

    return mse


# Model
search_space = {
    "lambda_1": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
    "lambda_2": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
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
np.savetxt("experiments/amazon_reg/results/ard_results.txt", results)
np.savetxt(
    "experiments/amazon_reg/results/ard_precisions.txt",
    best_model.lambda_,
)
joblib.dump(best_model, "experiments/amazon_reg/models/ard_rff.joblib")
