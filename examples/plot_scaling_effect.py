"""Effect of covariate scale on answer. Can it blur the bandwidth determination?"""
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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
n_samples, n_features, rho = 2000, 10, 0.6
X, y = make_jordan_se1(
    n_samples=n_samples, n_features=n_features, rho=rho, random_state=rng
)

design_matrices = [
    X,
    X * np.random.randn(10),
    X * np.r_[1e-2, np.ones(9)],
    X * np.r_[1e2, np.ones(9)],
    X * np.r_[1, 1e-2, np.ones(8)],
    X * np.r_[1, 1e2, np.ones(8)],
]

names = [
    "Original",
    "Std. Scaled",
    "Shrink important",
    "Increase important",
    "Shrink unimp.",
    "Increase unimp.",
]
scaler = StandardScaler()

for i, X_scaled in enumerate(design_matrices):
    ax = plt.subplot(2, 3, i + 1)

    # 3) Process data.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=500, random_state=rng
    )

    # Std scale randomly scaled data.
    if i == 1:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 4) Define model
    reg_rff = RFFNetRegressor(lr=1e-2, torch_seed=seed, random_state=rng)

    # 5) Fit model
    # RFF
    t_start = time.time()
    reg_rff.fit(X_train, y_train)
    print("Time elapsed for RFF: {:2f}".format(time.time() - t_start))

    # 6) Evaluate on test
    rff_pred = reg_rff.predict(X_test)
    print("RFF test loss: {:.5f}".format(mean_squared_error(y_test, rff_pred)))

    # 7) Plot RFF bandwidths
    labels = np.arange(0, X.shape[1])
    ax.set_title(names[i])
    ax.stem(np.abs(reg_rff.model.get_bandwidths()))
    ax.set_ylabel(r"$\sigma$")
    ax.set_xticks(labels, labels + 1)

plt.tight_layout()
plt.show()
