# scripts/experiments.py
"""Run experiments on the synthetic datasets and log results using Sacred. Heavily based on `Sacred Examples <https://github.com/maartjeth/sacred-example-pytorch/blob/master/train_nn.py>`_."""
import torch
import torch.nn as nn

from sacred import Experiment
from sacred.observers import FileStorageObserver

from pyselect import layers, networks, synthesizer, datasets

ex = Experiment("make_regression")

ex.observers.append(FileStorageObserver("experiments"))

# TODO: implement GPU compatibility.
device = "cpu"


class Trainer:
    def __init__(self):
        self.model = self.make_model()
        self.optimizer = self.make_optimizer()
        self.loss_fn = nn.MSELoss()
        self.train_dataset, self.test_dataset = self.get_datasets()
        self.train_loader, self.test_loader = self.get_dataloaders()

    @ex.capture
    def make_model(self, n_features, R, sampler):
        model = networks.RandomFourierFeaturesNet(n_features, R, sampler).to(device)
        return model

    @ex.capture
    def make_optimizer(self, learning_rate):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        return optimizer

    @ex.capture
    def get_datasets(self, train_filename, test_filename):
        train_dataset = datasets.SynthDataset(train_filename)
        test_dataset = datasets.SynthDataset(test_filename)
        return train_dataset, test_dataset

    @ex.capture
    def get_dataloaders(self, batch_size):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset, batch_size=batch_size, shuffle=False
        )

        return train_loader, test_loader

    @ex.capture
    def train(self, num_epochs, _run):
        total_step = len(self.train_loader)
        for epoch in range(num_epochs):
            for i, (x_obs, y_obs) in enumerate(self.train_loader):
                x_obs = x_obs.to(device=device)
                y_obs = y_obs.to(device=device)

                output = self.model(x_obs)

                loss = self.loss_fn(output, y_obs)
                _run.log_scalar("loss", float(loss.data))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) % 30 == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}"
                    )

    @ex.capture
    def test(self):
        with torch.no_grad():
            test_loss = 0
            X_test, y_test = self.test_dataset.X, self.test_dataset.y

            y_pred = self.model(X_test)
            test_loss = self.loss_fn(y_pred, y_test)
            print(f"Total loss on test: {test_loss:.4f}")
            return test_loss

    @ex.capture
    def run(self, model_file):
        self.train()

        loss = self.test()
        torch.save(self.model.state_dict(), model_file)

        print(f"Model saved in {model_file}")

        return loss


@ex.config
def config():
    train_filename = "./data/synth-2000-100-10-train.pt"
    test_filename = "./data/synth-2000-100-10-test.pt"
    learning_rate = 0.004216025368528044
    n_features = 100
    R = 486
    sampler = torch.randn
    num_epochs = 1000
    batch_size = 32
    model_file = "models/model.ckpt"


@ex.automain
def main(_run):
    trainer = Trainer()
    loss = trainer.run()
    return {"loss": loss}
