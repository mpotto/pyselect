import time
import joblib

import numpy as np
import torch
from sklearn.base import MultiOutputMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from pyselect.datasets import make_gregorova_se2
from pyselect.estimators import BaseRFFNet
from pyselect.model import ReluRFFNet
from pyselect.solvers import adam_solver
from pyselect.utils import get_mse_confidence_interval, get_folder


class ReluRFFNetRegressor(RegressorMixin, MultiOutputMixin, BaseRFFNet):

    criterion = torch.nn.MSELoss()

    def _convert_y(self, y):
        y = torch.FloatTensor(y)
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        return y

    @staticmethod
    def _output_shape(y):
        return y.shape[1]

    def _init_model(self, X, y):
        if self.torch_seed is not None:
            torch.manual_seed(self.torch_seed)

        output_shape = self._output_shape(y)

        self.model = ReluRFFNet(
            [X.shape[1], *self.hidden_layer_size, output_shape],
            sampler=self.sampler,
            dropout=self.dropout,
        )

    def predict(self, X):
        with torch.no_grad():
            ans = self.model(self._cast_input(X))
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


model = ReluRFFNetRegressor(
    lr=1e-3,
    hidden_layer_size=(300, 20, 10),
    torch_seed=seed,
    random_state=0,
    n_iter_no_change=10,
    verbose=True,
    log_rate=1,
)

t_start = time.time()
model.fit(X_train, y_train)
elapsed_time = time.time() - t_start

model_pred = model.predict(X_test)

center, band = get_mse_confidence_interval(y_test, model_pred)
metrics.append([center, band, elapsed_time])

precisions = model.precisions_

# Save results
relevances_folder = get_folder("eval/benchmarks/nn/se2/precisions")
metrics_folder = get_folder("eval/benchmarks/nn/se2/metrics")
models_folder = get_folder("eval/benchmarks/nn/se2/models")

np.savetxt(f"{relevances_folder}/precisions.txt", precisions)
np.savetxt(f"{metrics_folder}/metrics.txt", metrics)

# Can't pickle because the class is defined in this file
# joblib.dump(model, f"{models_folder}/model.joblib") 
