import time
import joblib

import numpy as np
import optuna
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample

from pyselect.utils import best_model_callback

val_size = 5 * 10 ** 3

results = []

# 1) Generate seeds
# torch
seed_sequence = np.random.SeedSequence(entropy=0)
seed = seed_sequence.generate_state(1)[0]

# sklearn
rng = np.random.RandomState(0)

# Dataset
X_train_val = np.load("data/amazon/processed2/X_train.npy")
X_test = np.load("data/amazon/processed2/X_test.npy")
y_train_val = np.load("data/amazon/processed2/y_train.npy")
y_test = np.load("data/amazon/processed2/y_test.npy")

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size, random_state=rng
)

# Subsample for finding best parameters
X_sub, y_sub = resample(
    X_train, y_train, n_samples=10 ** 4, stratify=y_train, random_state=rng
)


def objective(trial):
    C = trial.suggest_float("C", 1e-7, 1e2, log=True)

    # Model
    model = LogisticRegression(C=C, penalty="l1", solver="liblinear")

    model.fit(X_sub, y_sub)

    model_proba = model.predict_proba(X_val)

    roc_auc = roc_auc_score(y_val, model_proba[:, -1])

    trial.set_user_attr("model", value=model)

    return roc_auc


search_space = {"C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}

study = optuna.create_study(
    sampler=optuna.samplers.GridSampler(search_space), direction="maximize"
)
study.optimize(objective, callbacks=[best_model_callback])

best_model = study.user_attrs["best_model"]

t_start = time.time()
best_model.fit(X_train, y_train)
elapsed_time = time.time() - t_start

# Evaluate on test
model_pred = best_model.predict(X_test)
model_proba = best_model.predict_proba(X_test)

acc = accuracy_score(y_test, model_pred)
f1 = f1_score(y_test, model_pred)
roc_auc = roc_auc_score(y_test, model_proba[:, -1])

results.append([acc, f1, roc_auc, elapsed_time])

# 8) Save results.
np.savetxt("experiments/amazon/results/logreg_l1_results.txt", results)
joblib.dump(best_model, "experiments/amazon/models/best_logreg_l1.joblib")
