# optimization/synthetic-data/optimization.py
"""Optimize hyperparameters and log to Neptune."""
import yaml
import os

import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pyselect import layers, networks

params = yaml.safe_load(open("optimization/synthetic-data/params.yaml"))

train_dataset = torch.load(params["train_filename"])
test_dataset = torch.load(params["test_filename"])


def objective(trial):

    train_dataloader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )

    test_dataloader = DataLoader(test_dataset, batch_size=params["batch_size"])

    out_features = trial.suggest_int("out_features", 100, 600)
    model = networks.RandomFourierFeaturesNet(
        in_features=params["n_features"], out_features=out_features, sampler=torch.randn
    )

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_fn = nn.MSELoss()

    for epoch in range(params["n_epochs"]):
        model.train()
        for batch_idx, (data, target) in enumerate(train_dataloader):

            output = model(data)

            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    test_loss = 0
    with torch.no_grad():
        data, target = test_dataset.tensors

        y_pred = model(data)

        test_loss = loss_fn(y_pred, target)

    trial.report(test_loss, epoch)

    return test_loss


run = neptune.init(
    project="mpotto/pyselect",
    tags="Optimization",
    api_token=os.environ.get("NEPTUNE_API_TOKEN"),
    source_files=["optimization/synthetic-data/*.py"],
)


neptune_callback = optuna_utils.NeptuneCallback(run)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1, callbacks=[neptune_callback])


optuna_utils.log_study_metadata(study, run, log_plot_intermediate_values=False)

best_params = study.best_params
params["out_features"] = best_params["out_features"]
params["learning_rate"] = best_params["learning_rate"]

with open("optimization/synthetic-data/best-params.yaml", "w") as file:
    yaml.dump(params, file)

run["config/hyperparameters"] = params

run.stop()
