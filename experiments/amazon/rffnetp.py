import time
import joblib

import numpy as np
import torch
from sklearn.base import MultiOutputMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample
from sklearn.utils.validation import check_is_fitted

from pyselect.utils import best_model_callback, get_folder
from pyselect.estimators import BaseRFFNet
from pyselect.model import ReluRFFNet
from pyselect.solvers import adam_solver


metrics = []

# Dataset
X_train = np.load("data/splitted/amazon/X_train.npy")
X_val = np.load("data/splitted/amazon/X_val.npy")
X_test = np.load("data/splitted/amazon/X_test.npy")
y_train = np.load("data/splitted/amazon/y_train.npy")
y_val = np.load("data/splitted/amazon/y_val.npy")
y_test = np.load("data/splitted/amazon/y_test.npy")

class ReluRFFNetClassifier(ClassifierMixin, MultiOutputMixin, BaseRFFNet):

    criterion = torch.nn.CrossEntropyLoss()

    def _convert_y(self, y):
        y = torch.LongTensor(y)
        assert len(y.shape) == 1, "y must be 1D"
        return y

    @staticmethod
    def _output_shape(y):
        return (y.max() + 1).item()

    def _init_model(self, X, y):
        if self.torch_seed is not None:
            torch.manual_seed(self.torch_seed)

        output_shape = self._output_shape(y)

        self.model = ReluRFFNet(
            [X.shape[1], *self.hidden_layer_size, output_shape],
            sampler=self.sampler,
            dropout=self.dropout,
        )

    def decision_function(self, X):
        check_is_fitted(self)
        with torch.no_grad():
            scores = self.model(self._cast_input(X))
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        ans = scores.argmax(dim=1)
        if isinstance(X, np.ndarray):
            ans.cpu().numpy()
        return ans

    def predict_proba(self, X):
        with torch.no_grad():
            ans = torch.softmax(self.model(self._cast_input(X)), -1)
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans

    def fit(self, X, y):
        self._init_model(X, y)
        X, y = self._cast_input(X, y)

        self.model = adam_solver(
            X,
            y,
            self.model,
            self.criterion,
            lr=self.lr,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
            verbose=self.verbose,
            log_rate=self.log_rate,
        )

        self.model.eval()

        self.precisions_ = self.model.get_precisions()

        return self


# Configuration to train the model
seed_sequence = np.random.SeedSequence(entropy=1234)
seed = seed_sequence.generate_state(1)[0]


model = ReluRFFNetClassifier(
    lr=1e-4,
    batch_size=128,
    hidden_layer_size=(300, 20, 10),
    torch_seed=seed,
    random_state=0,
    n_iter_no_change=5,
    verbose=True,
    log_rate=1,
    max_iter=30,
)

t_start = time.time()
model.fit(X_train, y_train)
elapsed_time = time.time() - t_start

model_pred = model.predict(X_test)
model_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, model_pred)
f1 = f1_score(y_test, model_pred)
roc_auc = roc_auc_score(y_test, model_proba[:, -1])

metrics.append([acc, f1, roc_auc, elapsed_time])

importances = model.precisions_

# Results
precisions_folder = get_folder("eval/benchmarks/rffnetp/amazon/precisions")
metrics_folder = get_folder("eval/benchmarks/rffnetp/amazon/metrics")
models_folder = get_folder("eval/benchmarks/rffnetp/amazon/models")

np.savetxt(f"{precisions_folder}/precisions.txt", importances)
np.savetxt(f"{metrics_folder}/metrics.txt", metrics)
#joblib.dump(model, f"{models_folder}/model.joblib")
