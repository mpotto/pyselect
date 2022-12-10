import joblib

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gregorova import srf_run
from pyselect.utils import get_mse_confidence_interval, get_folder
from pyselect.datasets import make_gregorova_se1

n_samples = 54 * 10 ** 3
test_size = 2 * 10 ** 3
val_size = 2 * 10 ** 3

metrics = []

# 1) Generate seeds
# torch
seed_sequence = np.random.SeedSequence(entropy=0)
seed = seed_sequence.generate_state(1)[0]
torch.manual_seed(seed)

# 2) Generate data
X, y = make_gregorova_se1(n_samples, random_state=0)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=test_size, random_state=0
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_val = torch.FloatTensor(y_val)
y_test = torch.FloatTensor(y_test)

# Model
train, valid, test = srf_run(X_train, y_train, X_val, y_val, X_test, y_test)

elapsed_times = np.array([train[i]["time"] / 1e3 for i in range(len(train))])
mean_elapsed_time = np.mean(elapsed_times)
print("Mean elapsed time for SRF: {:2f}".format(mean_elapsed_time))

srf_pred = test["preds"].numpy()
center, band = get_mse_confidence_interval(y_test.numpy(), srf_pred)
metrics.append([center, band, mean_elapsed_time])

precisions = test["gamma"].detach().numpy()

# Save results
relevances_folder = get_folder("eval/benchmarks/srff/se1/precisions")
metrics_folder = get_folder("eval/benchmarks/srff/se1/metrics")
models_folder = get_folder("eval/benchmarks/rffnet/se1/models")

np.savetxt(f"{relevances_folder}/precisions.txt", precisions)
np.savetxt(f"{metrics_folder}/metrics.txt", metrics)
joblib.dump(test, f"{models_folder}/model.joblib")
