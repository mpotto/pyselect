import time
import joblib

import numpy as np
import optuna
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample

from pyselect.estimators import RFFNetClassifier
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

seed_sequence = np.random.SeedSequence(entropy=0)
seed = seed_sequence.generate_state(1)[0]


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
        random_state=0,
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
best_model.log_rate = 5
best_model.n_iter_no_change = 10
best_model.batch_size = len(X_train) // 10

t_start = time.time()
best_model.fit(X_train, y_train)
elapsed_time = time.time() - t_start

# Evaluate on test
model_pred = best_model.predict(X_test)
model_proba = best_model.predict_proba(X_test)

acc = accuracy_score(y_test, model_pred)
f1 = f1_score(y_test, model_pred)
roc_auc = roc_auc_score(y_test, model_proba[:, -1])

precisions = best_model.precisions_

metrics.append([acc, f1, roc_auc, elapsed_time])

# Results
precisions_folder = get_folder("eval/benchmarks/rffnet/higgs/precisions")
metrics_folder = get_folder("eval/benchmarks/rffnet/higgs/metrics")
models_folder = get_folder("eval/benchmarks/rffnet/higgs/models")

np.savetxt(f"{precisions_folder}/precisions.txt", precisions)
np.savetxt(f"{metrics_folder}/metrics.txt", metrics)
joblib.dump(best_model, f"{models_folder}/model.joblib")
