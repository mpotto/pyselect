import time
import joblib

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from pyselect.utils import get_mse_confidence_interval, get_folder

# Configuration to train the model
seed_sequence = np.random.SeedSequence(entropy=1234)
seed = seed_sequence.generate_state(1)[0]

test_size = 10 ** 3
train_size = 11 * 10 ** 3
val_size = 10 ** 3

metrics = []

# Dataset
data = pd.read_csv("data/processed/ailerons.csv")
X = data.drop(["target"], axis=1)
y = np.ravel(data[["target"]])

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, train_size=train_size + val_size, test_size=test_size, random_state=0
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

tf.random.set_seed(seed)
model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(300, activation="relu"),
                tf.keras.layers.Dense(20, activation="relu"),
                tf.keras.layers.Dense(10, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.compile(optimizer="adam", loss="mean_squared_error")

t_start = time.time()
model.fit(X_train, y_train, epochs=100, batch_size=64, callbacks=[callback])
elapsed_time = time.time() - t_start

model_pred = model.predict(X_test)
print(model_pred.shape)

center, band = get_mse_confidence_interval(y_test, model_pred)
metrics.append([center, band, elapsed_time])

# Save results
metrics_folder = get_folder("eval/benchmarks/nn/ailerons/metrics")
models_folder = get_folder("eval/benchmarks/nn/ailerons/models")

np.savetxt(f"{metrics_folder}/metrics.txt", metrics)
joblib.dump(model, f"{models_folder}/model.joblib") 
