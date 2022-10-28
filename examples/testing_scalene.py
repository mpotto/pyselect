import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from pyselect import RFFNetRegressor
from pyselect.datasets import make_jordan_se1

# 1) Generate seeds
# torch
seed_sequence = np.random.SeedSequence(entropy=1234)
seed = seed_sequence.generate_state(1)[0]

# sklearn
rng = np.random.RandomState(0)

# 2) Generate datasets
n_samples, n_features, rho = 1000, 20, 0.1
X, y = make_jordan_se1(n_samples, n_features, rho, random_state=rng)

# 3) Process data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=rng
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

# 4) Define models
# RFF (hand-tuned)
reg_rff = RFFNetRegressor(
    alpha=1e-5,
    lr=1e-4,
    batch_size=32,
    verbose=True,
    log_rate=5,
    n_iter_no_change=20,
    torch_seed=seed,
    random_state=rng,
)

# 5) Fit models
t_start = time.time()
reg_rff.fit(X_train, y_train)
print("Time elapsed for RFF: {:2f}".format(time.time() - t_start))

# 6) Evaluate on test
rff_pred = reg_rff.predict(X_test)
print("RFF test loss: {:.5f}".format(mean_squared_error(y_test, rff_pred)))

# 7) Plot RFF bandwidths
labels = np.arange(0, X.shape[1])
plt.figure()
plt.title(f"Jordan SE2")
plt.stem(np.abs(reg_rff.precisions_))
plt.ylabel(r"$\beta$")
plt.xticks(labels[::5], labels[::5] + 1)
plt.tight_layout()
plt.show()
