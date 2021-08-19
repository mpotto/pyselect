# experiments/synthetic-data/experiment.py
"""Run experiments on the synthetic datasets and log results using Neptune."""
import yaml
import os

import matplotlib.pyplot as plt
import numpy as np
import neptune.new as neptune
from neptune.new.types import File
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pyselect import layers, networks

run = neptune.init(
    project="mpotto/pyselect",
    tags=["Experiment", "Synthetic data"],
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
    for batch, (x, y) in enumerate(train_dataloader):
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


run["logs/test/test_loss"] = test_loss

with torch.no_grad():
    for batch, (x, y) in enumerate(test_dataloader, 0):
        output = model(x)
        loss = loss_fn(output, y)

        run["logs/test/batch/loss"].log(loss)

## Add histogram with bandwidths of model.
coefs = torch.load(params["coef_filename"]).squeeze(1).numpy()
bands = model.state_dict()["rff_net.0.bandwidths"].numpy()

inf_features = np.abs(coefs) >= 1e-4

inf_bands = bands[inf_features]
noninf_bands = bands[~inf_features]

lim = np.abs(bands).max()
bins = np.linspace(-lim * 1.1, lim * 1.1, 40)

plt.style.use("figures/pyselect.mplstyle")
fig = plt.figure()
plt.hist(inf_bands, bins=bins, rwidth=0.8, color="crimson", label="Informative")
plt.hist(
    noninf_bands,
    bins=bins,
    rwidth=0.8,
    color="black",
    alpha=0.4,
    label="Non-informative",
)
plt.xlabel(r"1/$\sigma$")
plt.ylabel("Frequency")
plt.legend(loc="best")

run["figures/band-histogram"] = File.as_image(fig)

index = np.array(list(range(0, params["n_features"])))
inf_index = index[inf_features]
noninf_index = index[~inf_features]

fig = plt.figure()
plt.bar(
    inf_index,
    np.abs(bands[inf_features]),
    color="crimson",
    width=0.9,
    label="Informative",
)
plt.bar(
    noninf_index,
    np.abs(bands[~inf_features]),
    color="black",
    alpha=0.4,
    label="Non-informative",
)
ax = plt.gca()
ax.ticklabel_format(axis="x", style="plain")
ax.set_xlabel("Coefficient Index")
ax.set_ylabel(r"$| 1/\sigma |$")

run["figures/coefficients-bands"] = File.as_image(fig)

run.stop()
