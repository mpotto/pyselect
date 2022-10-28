import time
import joblib

import numpy as np
import optuna
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample

from pyselect.estimators import RFFNetClassifier
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
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 1e-7, 1e-1, log=True)

    # Model
    model = RFFNetClassifier(
        lr=lr,
        alpha=alpha,
        validation_fraction=0.1,
        n_iter_no_change=5,
        batch_size=32,
        torch_seed=seed,
        random_state=rng,
    )

    model.fit(X_sub, y_sub)

    model_proba = model.predict_proba(X_val)

    roc_auc = roc_auc_score(y_val, model_proba[:, -1])

    trial.set_user_attr("model", value=model)

    return roc_auc


# Model
search_space = {
    "lr": [1e-5, 1e-4, 1e-3, 1e-2],
    "alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
}

study = optuna.create_study(
    sampler=optuna.samplers.GridSampler(search_space), direction="maximize"
)
study.optimize(objective, callbacks=[best_model_callback])

best_model = study.user_attrs["best_model"]
best_model.verbose = True
best_model.log_rate = 1

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

# 7) Plot RFF bandwidths
labels = np.arange(0, X_train.shape[1])
plt.figure()
plt.stem(np.abs(best_model.precisions_))
plt.ylabel(r"$\sigma$")
plt.xticks(labels[::200], labels[::200] + 1)

plt.tight_layout()
plt.show()

# 8) Save results.
np.savetxt("experiments/amazon/results/rffnet_results.txt", results)
np.savetxt("experiments/amazon/results/rffnet_precisions.txt", best_model.precisions_)
joblib.dump(best_model, "experiments/amazon/models/reg_rff.joblib")
