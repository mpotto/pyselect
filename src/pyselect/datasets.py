# src/pyselect/datasets.py
import torch
from torch.utils.data import Dataset


class SynthDataset(Dataset):
    def __init__(self, data_filename, transform=None, target_transform=None):
        self.X, self.y = torch.load(data_filename).tensors
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        X_obs = self.X[idx, :]
        y_obs = self.y[idx]
        if self.transform:
            X_obs = self.transform(X_obs)
        if self.target_transform:
            y_obs = self.target_transform(y_obs)
        return X_obs, y_obs
