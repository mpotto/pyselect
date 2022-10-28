import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = np.loadtxt("data/yearpredictionmsd/YearPredictionMSD.txt", delimiter=",")

train_index = 463715
test_index = 51630

X_train, y_train = data[:train_index, 1:], data[:train_index, 0]
X_test, y_test = data[-test_index:, 1:], data[-test_index:, 0]

rng = np.random.RandomState(0)

scaler = StandardScaler()

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=rng
)

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

np.save("data/yearpredictionmsd/processed/X_train.npy", X_train)
np.save("data/yearpredictionmsd/processed/X_val.npy", X_val)
np.save("data/yearpredictionmsd/processed/X_test.npy", X_test)
np.save("data/yearpredictionmsd/processed/y_train.npy", y_train)
np.save("data/yearpredictionmsd/processed/y_val.npy", y_val)
np.save("data/yearpredictionmsd/processed/y_test.npy", y_test)
