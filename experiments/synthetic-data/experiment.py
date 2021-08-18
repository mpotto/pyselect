# experiments/synthetic-data/experiment.py
"""Run experiments on the synthetic datasets and log results using Neptune."""
import yaml
import os

import neptune.new as neptune
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pyselect import layers, networks

run = neptune.init(
    project="mpotto/pyselect",
    tags="Experiment",
    api_token=os.environ.get("NEPTUNE_API_TOKEN"),
    source_files=["experiments/synthetic-data/*.py"],
)
params = yaml.safe_load(open("experiments/synthetic-data/params.yaml"))

run["model/params"] = params

data_dir = params["data_dir"]

train_dataset = torch.load(params["train_filename"])
test_dataset = torch.load(params["test_filename"])

train_dataloader = DataLoader(
    train_dataset, batch_size=params["batch_size"], shuffle=True
)

test_dataloader = DataLoader(test_dataset, batch_size=params["batch_size"])

torch.manual_seed(params["seed"])

model = networks.RandomFourierFeaturesNet(
    params["n_features"], params["out_features"], torch.randn
)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"])

run["config/datasets/path"] = data_dir
run["config/datasets/size"] = {len(train_dataset), len(test_dataset)}
run["config/model"] = type(model).__name__
run["config/loss"] = type(loss_fn).__name__
run["config/optimizer"] = type(optimizer).__name__
run["config/hyperparameters"] = params

for epoch in range(params["n_epochs"]):
    model.train()
    for epoch, (x, y) in enumerate(train_dataloader):
        output = model(x)
        loss = loss_fn(output, y)

        run["logs/training/batch/loss"].log(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), params["model_name"])
run["model/dict"].upload(params["model_name"])

model.eval()
with torch.no_grad():
    test_loss = 0
    X_test, y_test = test_dataset.tensors

    y_pred = model(X_test)
    test_loss = loss_fn(y_pred, y_test)

run["test/test_loss"] = test_loss

run.stop()
