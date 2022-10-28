import time
import joblib
from matplotlib import pyplot as plt

import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from gregorova import srf_algo, predict_linear
from pyselect.utils import get_mse_confidence_interval

val_size = 5 * 10 ** 3

results = []

# 1) Generate seeds
# torch
seed_sequence = np.random.SeedSequence(entropy=0)
seed = seed_sequence.generate_state(1)[0]

# sklearn
rng = np.random.RandomState(0)

# Dataset
# Dataset
X_train_val = np.load("data/amazon/processed2/X_train_reg.npy")
X_test = np.load("data/amazon/processed2/X_test_reg.npy")
y_train_val = np.load("data/amazon/processed2/y_train_reg.npy").reshape(-1, 1)
y_test = np.load("data/amazon/processed2/y_test_reg.npy").reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size, random_state=rng
)

X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_val = torch.FloatTensor(y_val)
y_test = torch.FloatTensor(y_test)

# Subsample for finding best parameters
X_sub, y_sub = resample(
    X_train, y_train, n_samples=10 ** 4, stratify=y_train, random_state=rng
)


def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-7, 1e-1, log=True)

    # Model

    result_map = srf_algo(X_sub, y_sub, alpha)

    ones_mat = torch.ones([X_val.size(0), 1])
    Z_valid = torch.cos(
        X_val.mm(torch.transpose(result_map["omg"], 1, 0))
        + ones_mat.mm(torch.transpose(result_map["b"], 1, 0))
    )

    _, val_error = predict_linear(Z_valid, y_val, result_map["a"])

    return val_error


search_space = {"alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}

study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective)

best_alpha = study.best_params["alpha"]

t_start = time.time()
train_results = srf_algo(X_train, y_train, best_alpha)
elapsed_time = time.time() - t_start

ones_mat = torch.ones([X_test.size(0), 1])
Z_test = torch.cos(
    X_test.mm(torch.transpose(train_results["omg"], 1, 0))
    + ones_mat.mm(torch.transpose(train_results["b"], 1, 0))
)
test_preds, _ = predict_linear(Z_test, y_test, train_results["a"])

srff_pred = test_preds.numpy()
center, band = get_mse_confidence_interval(y_test.numpy(), srff_pred)


results.append([center, band, elapsed_time])

precisions = train_results["gamma"].detach().numpy()

np.savetxt(f"experiments/amazon_reg/results/srff_precisions.txt", precisions)
joblib.dump(train_results, f"experiments/amazon_reg/models/reg_srff.joblib")

# 7) Plot RFF bandwidths
labels = np.arange(0, X_train.shape[1])
plt.figure()
plt.stem(precisions)
plt.ylabel(r"$\lambda$")
plt.xticks(labels, labels + 1)

plt.tight_layout()
plt.show()

# 8) Save results.
np.savetxt("experiments/amazon_reg/results/srff.txt", results)
