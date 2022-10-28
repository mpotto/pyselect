import numpy as np

from sklearn.preprocessing import StandardScaler

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

data = np.loadtxt("data/higgs/HIGGS.csv", delimiter=",")

X, y = resample(data[:, 1:], data[:, 0], n_samples=10 ** 6, stratify=data[:, 0])

rng = np.random.RandomState(0)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, train_size=85 * 10 ** 4, test_size=15 * 10 ** 4, random_state=rng
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, train_size=70 * 10 ** 4, random_state=rng
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

np.save("data/higgs/processed/X_train.npy", X_train)
np.save("data/higgs/processed/X_val.npy", X_val)
np.save("data/higgs/processed/X_test.npy", X_test)
np.save("data/higgs/processed/y_train.npy", y_train)
np.save("data/higgs/processed/y_val.npy", y_val)
np.save("data/higgs/processed/y_test.npy", y_test)
