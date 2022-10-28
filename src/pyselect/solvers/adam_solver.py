from sklearn.model_selection import train_test_split
import torch


def adam_solver(
    X,
    y,
    model,
    criterion,
    *,
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
    optim = torch.optim.Adam(model.parameters(), lr=lr)

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

            optim.zero_grad()
            loss.backward()
            optim.step()

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
