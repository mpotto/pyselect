import time
import joblib

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from pyselect.datasets import make_gregorova_se2
from pyselect.utils import get_mse_confidence_interval, get_folder

# Configuration to train the model
seed_sequence = np.random.SeedSequence(entropy=1234)
seed = seed_sequence.generate_state(1)[0]

n_samples = 54 * 10 ** 3
train_size = 50 * 10 ** 3
test_size = 4 * 10 ** 3

metrics = []

# 2) Generate data
X, y = make_gregorova_se2(n_samples, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

tf.random.set_seed(seed)
model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(300, activation="relu"),
                tf.keras.layers.Dense(20, activation="relu"),
                tf.keras.layers.Dense(10, activation="relu"),
            ]
        )
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.compile(optimizer="adam", loss="mean_squared_error")

t_start = time.time()
model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=[callback])
elapsed_time = time.time() - t_start

model_pred = model.predict(X_test)

center, band = get_mse_confidence_interval(y_test, model_pred)
metrics.append([center, band, elapsed_time])

# Save results
metrics_folder = get_folder("eval/benchmarks/nn/se2/metrics")
models_folder = get_folder("eval/benchmarks/nn/se2/models")

np.savetxt(f"{metrics_folder}/metrics.txt", metrics)
joblib.dump(model, f"{models_folder}/model.joblib") 
