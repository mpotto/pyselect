import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import ARDRegression

from pyselect import RFFNetRegressor
from pyselect.datasets import make_vehtari

# 1) Generate seeds
# torch
seed_sequence = np.random.SeedSequence(entropy=1234)
seed = seed_sequence.generate_state(1)[0]

# sklearn
rng = np.random.RandomState(5)

# 2) Generate datasets
X, y = make_vehtari(n_samples=10_000, random_state=rng)

# 3) Process data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1_000, random_state=rng
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4) Define models
# RFF (hand-tuned)
reg_rff = RFFNetRegressor(
    alpha=1e-4,
    lr=5e-3,
    verbose=True,
    log_rate=10,
    n_iter_no_change=10,
    torch_seed=seed,
    random_state=rng,
)

reg_ard = ARDRegression()

# 5) Fit models
t_start = time.time()
reg_rff.fit(X_train, y_train)
print("Time elapsed for RFF: {:2f}".format(time.time() - t_start))

t_start = time.time()
reg_ard.fit(X_train, y_train.ravel())
print("Time elapsed for ARD: {:2f}".format(time.time() - t_start))

# 6) Evaluate on test
rff_pred = reg_rff.predict(X_test)
ard_pred = reg_ard.predict(X_test)
print("RFF test loss: {:.5f}".format(mean_squared_error(y_test, rff_pred)))
print("ARD test loss: {:.5f}".format(mean_squared_error(y_test, ard_pred)))

# 7) Plot RFF bandwidths
labels = np.arange(0, X.shape[1])

scaler = MinMaxScaler()
bandwidths = np.abs(reg_rff.model.precisions_).reshape(-1, 1)
bandwidths = scaler.fit_transform(bandwidths).ravel()

precisions = reg_ard.lambda_.reshape(-1, 1)
precisions = scaler.fit_transform(precisions).ravel()
print(precisions.shape)


plt.figure()
plt.title(f"Vehtari ARD")
plt.stem(bandwidths, markerfmt="C0o", linefmt="grey", label="RFFNet")
plt.stem(precisions, markerfmt="C2o", linefmt="grey", label="ARD")
plt.axhline(y=1, linestyle="dashed", color="k")
plt.ylabel(r"$\beta$")
plt.xticks(labels, labels + 1)
plt.legend()
plt.tight_layout()
plt.show()
