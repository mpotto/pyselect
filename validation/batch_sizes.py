import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pyselect.datasets import make_gregorova_se1
from pyselect.model import RFFNet


def prox_2_squared(x, alpha):
    "Proximal operator for l2 norm squared."
    return x / (1 + 2 * alpha)


def adam_palm(batch_size):
    torch.manual_seed(SEED)
    model = RFFNet(dims=[n_features, n_random_features, 1])

    optim_linear = torch.optim.Adam(model.linear.parameters(), lr=LR)
    optim_rff = torch.optim.Adam(model.rff.parameters(), lr=LR)

    model.train()

    history = []

    for epoch in range(N_EPOCHS):
        indices = torch.randperm(train_size)
        model.train()

        for i in range(train_size // batch_size):
            batch = indices[i * batch_size : (i + 1) * batch_size]
            pred = model(X_train[batch])

            loss = CRITERION(pred, y_train[batch])

            optim_linear.zero_grad()
            loss.backward()
            optim_linear.step()

            with torch.no_grad():
                curr_lr = optim_linear.param_groups[0]["lr"]
                model.linear.weight.data = prox_2_squared(
                    model.linear.weight.data, ALPHA * curr_lr
                )

        for i in range(train_size // batch_size):
            batch = indices[i * batch_size : (i + 1) * batch_size]
            pred = model(X_train[batch])

            loss = CRITERION(pred, y_train[batch])

            optim_rff.zero_grad()
            loss.backward()
            optim_rff.step()

        with torch.no_grad():
            val_loss = CRITERION(model(X_val), y_val)
            history.append(val_loss)

    return history


SEED = 67891101
rng = np.random.RandomState(SEED)

# Data
n_samples = 6 * 10 ** 3
train_size = 5 * 10 ** 3

X, y = make_gregorova_se1(n_samples=n_samples, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=train_size, random_state=SEED
)
n_features = X.shape[1]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
y_train = torch.FloatTensor(y_train)
y_val = torch.FloatTensor(y_val)

# Number of random features
# n_random_features = int(np.sqrt(train_size) * np.log(train_size))
n_random_features = int(np.sqrt(train_size))

N_EPOCHS = 100
LR = 1e-2
ALPHA = 1.0
CRITERION = torch.nn.MSELoss()

# Run
histories = []
batch_sizes = [train_size // i for i in [1, 2, 3, 4, 5, 10, 15, 20, 40, 50]]
for bs in batch_sizes:
    print(bs)
    start_time = time.time()
    palm_adam_hist = adam_palm(bs)
    print(time.time() - start_time)
    histories.append(palm_adam_hist)

# Plot
for i, hist in enumerate(histories):
    plt.plot(hist, label="{:d}".format(batch_sizes[i]))
plt.yscale("log")
plt.legend()
plt.show()

# Large batches sizes work without regularization. Smaller batches require regularization and converge much faster: could benefit from early stopping.
