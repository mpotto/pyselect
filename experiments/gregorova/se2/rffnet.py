import time
import joblib

import numpy as np
import optuna
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from pyselect.estimators import RFFNetRegressor
from pyselect.datasets import make_gregorova_se2
from pyselect.utils import best_model_callback, get_mse_confidence_interval

seed_sequence = np.random.SeedSequence(entropy=0)
seed = seed_sequence.generate_state(1)[0]
rng = np.random.RandomState(0)

results = []

n_samples = 54 * 10 ** 3
test_size = 2 * 10 ** 3
val_size = 2 * 10 ** 3

# 2) Generate data
X, y = make_gregorova_se2(n_samples, random_state=rng)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=test_size, random_state=rng
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

# Save unscaled precisions.
np.savetxt(
    "experiments/gregorova/se2/results/rffnet_precisions.txt",
    best_model.precisions_,
)
joblib.dump(best_model, "experiments/gregorova/se2/models/reg_rff.joblib")

# 7) Plot RFF unscaled precisions
labels = np.arange(0, X.shape[1])
plt.figure()
plt.stem(np.abs(best_model.precisions_))
plt.ylabel(r"$\lambda$")
plt.xticks(labels, labels + 1)

plt.tight_layout()
plt.show()

np.savetxt("experiments/gregorova/se2/results/rffnet.txt", np.array(results))
