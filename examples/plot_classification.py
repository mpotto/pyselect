import time

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

from pyselect import RFFNetClassifier


# 1) Generate seeds
# torch
seed_sequence = np.random.SeedSequence(entropy=1234)
seed = seed_sequence.generate_state(1)[0]

# sklearn
rng = np.random.RandomState(0)

# 2) Generate datasets
n_samples = 2000
X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=rng)

# 3) Process data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=500, random_state=rng
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4) Define models
# RFF (hand-tuned)
classif_rff = RFFNetClassifier(
    lr=1e-2,
    alpha=1e-4,
    verbose=True,
    log_rate=5,
    torch_seed=seed,
    max_iter=200,
    random_state=rng,
)

# 5) Fit models
t_start = time.time()
classif_rff.fit(X_train, y_train)
print("Time elapsed for RFF: {:2f}".format(time.time() - t_start))

# 6) Metrics
rff_pred = classif_rff.predict(X_test)
acc = accuracy_score(y_test, rff_pred)
roc_auc = roc_auc_score(y_test, classif_rff.predict_proba(X_test)[:, -1])
print(f"Accuracy {acc:.2f} ROC AUC {roc_auc:.2f}.")


# 7) Plot RFF bandwidths
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
ax = plt.subplot(1, 1, 1)
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
# Plot the testing points
ax.scatter(
    X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
)
DecisionBoundaryDisplay.from_estimator(
    classif_rff, X_test, cmap=cm, alpha=0.5, ax=ax, eps=0.5
)
plt.show()
