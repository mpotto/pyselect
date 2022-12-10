from abc import ABCMeta, abstractmethod, abstractstaticmethod

import numpy as np
import torch
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
)
from sklearn.utils.validation import check_is_fitted

from pyselect.model import ReluRFFNet, RFFNet
from pyselect.solvers import adam_solver, palm_solver


class BaseRFFNet(BaseEstimator, metaclass=ABCMeta):
    def __init__(
        self,
        hidden_layer_size=None,
        *,
        sampler=torch.randn,
        alpha=1e-4,
        dropout=None,
        batch_size=32,
        lr=5e-4,
        max_iter=200,
        early_stopping=True,
        min_delta_fraction=1.0,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=None,
        torch_seed=None,
        verbose=False,
        log_rate=5,
    ):
        self.hidden_layer_size = hidden_layer_size
        self.sampler = sampler

        self.alpha = alpha
        self.dropout = dropout

        self.lr = lr
        self.batch_size = batch_size

        self.max_iter = max_iter

        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.min_delta_fraction = min_delta_fraction

        self.random_state = random_state
        self.torch_seed = torch_seed

        self.verbose = verbose
        self.log_rate = log_rate

    @abstractmethod
    def _convert_y(self, y):
        raise NotImplementedError

    @abstractstaticmethod
    def _output_shape(cls, y):
        raise NotImplementedError

    @property
    @abstractmethod
    def criterion(cls):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def _init_model(self, X, y):
        if self.hidden_layer_size is None:
            n = len(X)
            self.hidden_layer_size = int(np.sqrt(n) * np.log(n))

        if self.torch_seed is not None:
            torch.manual_seed(self.torch_seed)

        output_shape = self._output_shape(y)

        self.model = RFFNet(
            [X.shape[1], self.hidden_layer_size, output_shape], sampler=self.sampler
        )

    def _cast_input(self, X, y=None):
        X = torch.FloatTensor(X)
        if y is None:
            return X
        y = self._convert_y(y)
        return X, y

    def fit(self, X, y):
        self._init_model(X, y)
        X, y = self._cast_input(X, y)

        self.model = palm_solver(
            X,
            y,
            self.model,
            self.criterion,
            alpha=self.alpha,
            lr=self.lr,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            min_delta_fraction=self.min_delta_fraction,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
            verbose=self.verbose,
            log_rate=self.log_rate,
        )

        self.model.eval()

        self.precisions_ = self.model.get_precisions()
        self.coefs_ = self.model.linear.weight.detach().numpy()

        return self


class RFFNetRegressor(RegressorMixin, MultiOutputMixin, BaseRFFNet):

    criterion = torch.nn.MSELoss()

    def _convert_y(self, y):
        y = torch.FloatTensor(y)
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        return y

    @staticmethod
    def _output_shape(y):
        return y.shape[1]

    def predict(self, X):
        with torch.no_grad():
            ans = self.model(self._cast_input(X))
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans


class RFFNetClassifier(ClassifierMixin, BaseRFFNet):

    criterion = torch.nn.CrossEntropyLoss()

    def _convert_y(self, y):
        y = torch.LongTensor(y)
        assert len(y.shape) == 1, "y must be 1D"
        return y

    @staticmethod
    def _output_shape(y):
        return (y.max() + 1).item()

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
