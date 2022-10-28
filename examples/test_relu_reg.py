import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from pyselect import ReluRFFNetRegressor
from pyselect.datasets import make_gregorova_se2

# 1) Generate seeds
# torch
seed_sequence = np.random.SeedSequence(entropy=1234)
seed = seed_sequence.generate_state(1)[0]

# sklearn
rng = np.random.RandomState(0)

# 2) Generate datasets
n_samples = 1000000
X, y = make_gregorova_se2(n_samples, random_state=rng)

# 3) Process data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=rng
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4) Define models
reg_rff = ReluRFFNetRegressor(
    lr=1e-3,
    hidden_layer_size=(300, 20, 10),
    torch_seed=seed,
    random_state=rng,
    n_iter_no_change=10,
    verbose=True,
    log_rate=1,
    dropout=0.6,
)

# 5) Fit model
reg_rff.fit(X_train, y_train)

precisions = np.abs(reg_rff.precisions_).reshape(-1, 1)
minmax = MinMaxScaler()
precisions = minmax.fit_transform(precisions).flatten()

plt.figure()
plt.stem(precisions)
plt.ylabel(r"$\beta$")

plt.tight_layout()
plt.show()
