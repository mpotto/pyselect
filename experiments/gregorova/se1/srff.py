import joblib
from matplotlib import pyplot as plt

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gregorova import srf_run
from pyselect.utils import get_mse_confidence_interval
from pyselect.datasets import make_gregorova_se1

n_samples = 54 * 10 ** 3
test_size = 2 * 10 ** 3
val_size = 2 * 10 ** 3

results = []

# 1) Generate seeds
# torch
seed_sequence = np.random.SeedSequence(entropy=0)
seed = seed_sequence.generate_state(1)[0]
torch.manual_seed(seed)

# sklearn
rng = np.random.RandomState(0)

# 2) Generate data
X, y = make_gregorova_se1(n_samples, random_state=rng)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=test_size, random_state=rng
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size, random_state=rng
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
results.append([center, band, mean_elapsed_time])

precisions = test["gamma"].detach().numpy()

np.savetxt(f"experiments/gregorova/se1/results/srff_precisions.txt", precisions)
joblib.dump(test, f"experiments/gregorova/se1/models/reg_srff.joblib")

# 7) Plot RFF bandwidths
labels = np.arange(0, X.shape[1])
plt.figure()
plt.stem(precisions)
plt.ylabel(r"$\lambda$")
plt.xticks(labels, labels + 1)

plt.tight_layout()
plt.show()

# 8) Save results.
np.savetxt("experiments/gregorova/se1/results/srff.txt", results)
