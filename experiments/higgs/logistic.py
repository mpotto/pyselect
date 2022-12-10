import time
import joblib

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample

from pyselect.utils import best_model_callback, get_folder

metrics = []

# Dataset
X_train = np.load("data/splitted/higgs/X_train.npy")
X_val = np.load("data/splitted/higgs/X_val.npy")
X_test = np.load("data/splitted/higgs/X_test.npy")
y_train = np.load("data/splitted/higgs/y_train.npy")
y_val = np.load("data/splitted/higgs/y_val.npy")
y_test = np.load("data/splitted/higgs/y_test.npy")

# Subsample for finding best parameters
X_sub, y_sub = resample(
    X_train, y_train, n_samples=10 ** 4, stratify=y_train, random_state=0
)


def objective(trial):
    C = trial.suggest_float("C", 1e-5, 1e2, log=True)

    # Model
    model = LogisticRegression(C=C)

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

metrics.append([acc, f1, roc_auc, elapsed_time])

# Results
metrics_folder = get_folder("eval/benchmarks/logistic-l2/higgs/metrics")
models_folder = get_folder("eval/benchmarks/logistic-l2/higgs/models")

np.savetxt(f"{metrics_folder}/metrics.txt", metrics)
joblib.dump(best_model, f"{models_folder}/model.joblib")
