# scripts/data-preparation.py
import torch

from pyselect import synthesizer

# TODO: generate synth dataset and save on data.
class GenerateSynthDataset:
    def __init__(self, make_regression, *args, **kwargs):
        self.X, self.y, self.coef = make_regression(*args, **kwargs)

    def save_data(
        self, training_fraction: float, train_filename: str, test_filename: str
    ) -> None:
        training_size = int(training_fraction * self.X.size(0))

        X_train, y_train = self.X[:-training_size, :], self.y[:-training_size]
        X_test, y_test = self.X[-training_size:, :], self.y[-training_size:]

        train_as_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        torch.save(train_as_dataset, train_filename)

        test_as_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        torch.save(test_as_dataset, test_filename)

    def save_coef(self, filename):
        torch.save(self.coef, filename)


def generate_synth_1():
    synth_data_gen = GenerateSynthDataset(
        synthesizer.make_regression_with_tensors,
        n_samples=2000,
        n_features=100,
        n_informative=10,
    )
    synth_data_gen.save_data(
        training_fraction=0.7,
        train_filename="data/synth-2000-100-10-train.pt",
        test_filename="data/synth-2000-100-10-test.pt",
    )
    synth_data_gen.save_coef("data/synth-coef-2000-100-10.pt")


if __name__ == "__main__":
    generate_synth_1()
