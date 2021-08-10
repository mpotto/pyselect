# scripts/hyperparameters.py
"""Hyperparameter tuning for the synthetic datasets. Mainly based on <Optuna Examples https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py>_."""
import logging

import optuna
from optuna.trial import TrialState
import torch
import torch.optim as optim
from torch.nn import MSELoss

from pyselect import networks, datasets

# This is poor code practice and you know it.
BATCH_SIZE = 32
IN_FEATURES = 100
SAMPLER = torch.randn
TRAIN_FILENAME = "data/synth-2000-100-10-train.pt"
TEST_FILENAME = "data/synth-2000-100-10-test.pt"
N_EPOCHS = 1000


def define_model(trial):
    # R is the only parameter.
    out_features = trial.suggest_int("out_features", 100, 600)
    model = networks.RandomFourierFeaturesNet(
        in_features=IN_FEATURES, out_features=out_features, sampler=SAMPLER
    )

    return model


def get_dataset(train_filename=TRAIN_FILENAME, test_filename=TEST_FILENAME):
    train_dataset = datasets.SynthDataset(train_filename)
    test_dataset = datasets.SynthDataset(test_filename)

    return train_dataset, test_dataset


def get_dataloader(train_filename=TRAIN_FILENAME, test_filename=TEST_FILENAME):
    train_dataset, test_dataset = get_dataset(train_filename, test_filename)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, test_loader


def objective(trial):

    model = define_model(trial)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    train_loader, test_loader = get_dataloader()

    loss_fn = torch.nn.MSELoss()

    for epoch in range(N_EPOCHS):
        model.train()  # start in training mode
        for batch_idx, (data, target) in enumerate(train_loader):

            output = model(data)

            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    test_loss = 0
    with torch.no_grad():
        train_dataset, test_dataset = get_dataset()
        data = test_dataset.X
        target = test_dataset.y

        y_pred = model(data)

        test_loss = loss_fn(y_pred, target)

    trial.report(test_loss, epoch)

    return test_loss


# study
if __name__ == "__main__":
    optuna.logging.get_logger("optuna").addHandler(
        logging.FileHandler("optimization/optuna-quickstart.txt")
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2, timeout=600)

    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
