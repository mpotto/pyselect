import numpy as np
import torch

from numpy.linalg import norm
from sklearn.model_selection import train_test_split


def prox_2_squared(x, alpha):
    "Proximal operator for l2 norm squared."
    return x / (1 + 2 * alpha)


def ST_vec(x, u):
    """Entrywise soft-thresholding of array x at level u."""
    return np.sign(x) * np.maximum(0.0, np.abs(x) - u)


def palm_solver(
    X,
    y,
    model,
    criterion,
    *,
    alpha=1e-4,
    lr=5e-4,
    batch_size=32,
    max_iter=10 ** 3,
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=0.1,
    random_state=None,
    verbose=False,
    log_rate=5,
):
    optim_linear = torch.optim.Adam(model.linear.parameters(), lr=lr)
    optim_rff = torch.optim.Adam(model.rff.parameters(), lr=lr)

    X, X_val, y, y_val = train_test_split(
        X, y, test_size=validation_fraction, random_state=random_state
    )

    no_improvement_count = 0
    best_val_loss = torch.Tensor([float("Inf")])

    train_size = len(X)
    for epoch in range(max_iter):
        indices = torch.randperm(train_size)

        model.train()

        for i in range(train_size // batch_size):
            batch = indices[i * batch_size : (i + 1) * batch_size]
            pred = model(X[batch])

            loss = criterion(pred, y[batch])

            optim_linear.zero_grad()
            loss.backward()
            optim_linear.step()

            with torch.no_grad():
                curr_lr = optim_linear.param_groups[0]["lr"]
                model.linear.weight.data = prox_2_squared(
                    model.linear.weight.data, alpha * curr_lr
                )

        for i in range(train_size // batch_size):
            batch = indices[i * batch_size : (i + 1) * batch_size]
            pred = model(X[batch])

            loss = criterion(pred, y[batch])

            optim_rff.zero_grad()
            loss.backward()
            optim_rff.step()

        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)

        if val_loss < best_val_loss:
            best_model_state_dict = model.state_dict()
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if early_stopping and no_improvement_count == n_iter_no_change:
            break

        if verbose and epoch % log_rate == 0:
            print(f"Epoch {epoch}, Val. loss {val_loss:.3e}")

    model.load_state_dict(best_model_state_dict)

    return model
