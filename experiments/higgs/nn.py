import time
import joblib

import numpy as np
import tensorflow as tf
from sklearn.base import MultiOutputMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample

from pyselect.utils import best_model_callback, get_folder
from pyselect.estimators import BaseRFFNet
from pyselect.model import ReluRFFNet
from pyselect.solvers import adam_solver

metrics = []

# Dataset
X_train = np.load("data/splitted/higgs/X_train.npy")
X_val = np.load("data/splitted/higgs/X_val.npy")
X_test = np.load("data/splitted/higgs/X_test.npy")
y_train = np.load("data/splitted/higgs/y_train.npy")
y_val = np.load("data/splitted/higgs/y_val.npy")
y_test = np.load("data/splitted/higgs/y_test.npy")

# Configuration to train the model
seed_sequence = np.random.SeedSequence(entropy=1234)
seed = seed_sequence.generate_state(1)[0]

tf.random.set_seed(seed)
model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(300, activation="relu"),
                tf.keras.layers.Dense(20, activation="relu"),
                tf.keras.layers.Dense(10, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid")
            ]
        )
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

t_start = time.time()
model.fit(X_train, y_train, batch_size=128, epochs=10, callbacks=[callback])
elapsed_time = time.time() - t_start

# model_pred = model.predict_classes(X_test)
model_proba = model.predict(X_test)
model_pred = (model_proba > 0.5).ravel().astype(np.int32)

acc = accuracy_score(y_test, model_pred)
f1 = f1_score(y_test, model_pred)
roc_auc = roc_auc_score(y_test, model_proba)

metrics.append([acc, f1, roc_auc, elapsed_time])

# Results
metrics_folder = get_folder("eval/benchmarks/nn/higgs/metrics")
models_folder = get_folder("eval/benchmarks/nn/higgs/models")

np.savetxt(f"{metrics_folder}/metrics.txt", metrics)
joblib.dump(model, f"{models_folder}/model.joblib")
