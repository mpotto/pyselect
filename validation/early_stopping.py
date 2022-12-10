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


def adam_palm():
    torch.manual_seed(SEED)
    model = RFFNet(dims=[n_features, n_random_features, 1])

    optim_linear = torch.optim.Adam(model.linear.parameters(), lr=LR)
    optim_rff = torch.optim.Adam(model.rff.parameters(), lr=LR)

    no_improvement_count = 0
    best_val_loss = torch.Tensor([float("Inf")])

    history = []

    for epoch in range(N_EPOCHS):
        indices = torch.randperm(train_size)
        model.train()

        for i in range(train_size // BATCH_SIZE):
            batch = indices[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
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

        for i in range(train_size // BATCH_SIZE):
            batch = indices[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            pred = model(X_train[batch])

            loss = CRITERION(pred, y_train[batch])

            optim_rff.zero_grad()
            loss.backward()
            optim_rff.step()

        with torch.no_grad():
            val_loss = CRITERION(model(X_val), y_val)
            history.append(val_loss)

        if val_loss < MIN_DELTA_FRACTION * best_val_loss:
            best_model_state_dict = model.state_dict()
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count == N_ITER_NO_CHANGE:
            break

    return history


SEED = 0
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
n_random_features = int(np.sqrt(train_size) * np.log(train_size))

N_ITER_NO_CHANGE = 20
MIN_DELTA_FRACTION = 0.99
N_EPOCHS = 100
LR = 1e-2
ALPHA = 0
BATCH_SIZE = train_size // 10
CRITERION = torch.nn.MSELoss()

# Run
adam_hist = adam_palm()

# Plot
plt.plot(adam_hist, label="{:.2E}".format(0.0))
plt.yscale("log")
plt.legend()
plt.show()

# Decision: keep regularization.
