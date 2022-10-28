import time
import joblib

import numpy as np
import optuna
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample

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
    X_train, y_train, n_samples=5 * 10 ** 3, stratify=y_train, random_state=rng
)


def best_model_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["model"])


def objective(trial):
    max_depth = trial.suggest_int("max_depth", 3, 20)
    eta = trial.suggest_float("eta", 0, 1)
    gamma = trial.suggest_float("gamma", 1e-5, 100, log=True)
    min_child_weight = trial.suggest_float("min_child_weight", 0, 10)

    # Model
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        max_depth=max_depth,
        eta=eta,
        gamma=gamma,
        min_child_weight=min_child_weight,
    )

    model.fit(X_sub, y_sub)

    model_proba = model.predict_proba(X_val)

    roc_auc = roc_auc_score(y_val, model_proba[:, -1])

    trial.set_user_attr("model", value=model)

    return roc_auc


study = optuna.create_study(
    sampler=optuna.samplers.RandomSampler(), direction="maximize"
)
study.optimize(objective, callbacks=[best_model_callback], n_trials=50)

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
# 7) Plot feature importance
importances = best_model.feature_importances_

labels = np.arange(0, X_train.shape[1])
plt.figure()
plt.stem(importances)
plt.ylabel(r"Importances")
plt.xticks(labels, labels + 1)

plt.tight_layout()
plt.show()

# 8) Save results.
np.savetxt("experiments/amazon/results/xgb_results.txt", results)
np.savetxt("experiments/amazon/results/xgb_precisions.txt", importances)
joblib.dump(best_model, "experiments/amazon/models/best_xgb.joblib")
