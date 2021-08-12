# scripts/data-preparation.py
import os

import torch

from pyselect import synthesizer
from pyselect.datasets import GenerateSynthDataset


def generate_synth_1():
    os.makedirs("data/synthetic/synth-2000-100-10/")

    synth_data_gen = GenerateSynthDataset(
        synthesizer.make_regression_with_tensors,
        n_samples=2000,
        n_features=100,
        n_informative=10,
    )
    synth_data_gen.save_data(
        training_fraction=0.7,
        train_filename="data/synthetic/synth-2000-100-10/synth-2000-100-10-train.pt",
        test_filename="data/synthetic/synth-2000-100-10/synth-2000-100-10-test.pt",
    )
    synth_data_gen.save_coef(
        "data/synthetic/synth-2000-100-10/synth-coef-2000-100-10.pt"
    )


if __name__ == "__main__":
    generate_synth_1()
