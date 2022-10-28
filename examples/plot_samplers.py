import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.distributions import Cauchy, Laplace

from pyselect import RFFNetRegressor
from pyselect.datasets import make_correlated_data

# 1) Generate seeds
# torch
seed_sequence = np.random.SeedSequence(entropy=1234)
seed = seed_sequence.generate_state(1)[0]

# sklearn
rng = np.random.RandomState(0)

# 2) Generate datasets
n_samples, n_features, rho = 2000, 20, 0.6
w_true = np.zeros((n_features, 1))
w_true[:3] = 1
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

# 4) Define samplers
samplers = [
    torch.randn,
    lambda x, y: Cauchy(0, 1).sample(sample_shape=(x, y)),
    lambda x, y: Laplace(0, 1).sample(sample_shape=(x, y)),
]

for name, sampler in zip(["Normal", "Cauchy", "Laplace"], samplers):
    reg_rff = RFFNetRegressor(
        lr=5e-4,
        alpha=1e-4,
        verbose=True,
        log_rate=20,
        torch_seed=seed,
        random_state=rng,
        sampler=sampler,
    )

    # 5) Fit model
    t_start = time.time()
    reg_rff.fit(X_train, y_train)
    print("Time elapsed for RFF: {:2f}".format(time.time() - t_start))

    y_pred = reg_rff.predict(X_test)
    print(
        f"Mean squared error for RFF with {name} sampler: {mean_squared_error(y_test, y_pred):.3f}"
    )

    plt.figure()
    plt.title(name)
    plt.stem(np.abs(reg_rff.model.get_bandwidths()))
    plt.ylabel(r"$\beta$")

plt.tight_layout()
plt.show()
