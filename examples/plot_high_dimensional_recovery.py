import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from pyselect import RFFNetRegressor
from pyselect.datasets import make_correlated_data

# 1) Generate seeds
# torch
seed_sequence = np.random.SeedSequence(entropy=1234)
seed = seed_sequence.generate_state(1)[0]

# sklearn
rng = np.random.RandomState(0)

# 2) Generate datasets
n_samples, n_features, rho = 1000, 2000, 0.5
w_true = np.zeros((n_features, 1))
w_true[:5] = np.linspace(0.1, 5, 5).reshape(-1, 1)
X, y, w = make_correlated_data(
    n_samples, n_features, rho=rho, w_true=w_true, random_state=rng
)
y = y.reshape(-1, 1)

# 3) Process data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=rng
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4) Define models
# RFF (hand-tuned)
reg_rff = RFFNetRegressor(
    lr=1e-3,
    alpha=0e-2,
    batch_size=32,
    verbose=True,
    log_rate=1,
    n_iter_no_change=40,
    torch_seed=seed,
    random_state=rng,
)

# 5) Fit model
t_start = time.time()
reg_rff.fit(X_train, y_train)
print("Time elapsed for RFF: {:2f}".format(time.time() - t_start))

precisions = np.abs(reg_rff.precisions_).reshape(-1, 1)
minmax = MinMaxScaler()
precisions = minmax.fit_transform(precisions).flatten()

plt.figure()
plt.stem(precisions)
plt.ylabel(r"$\beta$")

plt.tight_layout()
plt.show()
