# tests/test_synthesizer.py
"""Test suite for data synth."""
from unittest.mock import Mock

import numpy as np
import pytest
from pytest_mock import MockFixture
import torch

from pyselect import synthesizer


@pytest.fixture
def mock_make_regression(mocker: MockFixture) -> Mock:
    """Fixture for mocking call to make_regression."""
    mock = mocker.patch("sklearn.datasets.make_regression")
    mock.return_value = (
        np.ones((5, 5)),
        np.ones(5),
        np.ones(5),
    )
    return mock


def test_make_regression_with_tensors_returns_tensor(
    mock_make_regression: Mock,
) -> None:
    """It returns a torch.Tensor."""
    result = synthesizer.make_regression_with_tensors()
    X, y, coef = result
    assert torch.is_tensor(X) and torch.is_tensor(y) and torch.is_tensor(coef)


def test_make_regression_uses_arguments():
    # TODO: dimensions should match matrix multiplication.
    """It uses the n_samples and n_features arguments."""
    X, y, coef = synthesizer.make_regression_with_tensors(
        n_samples=5,
        n_features=5,
    )
    assert (
        X.size() == torch.Size([5, 5])
        and y.size() == torch.Size([5])
        and coef.size() == torch.Size([5])
    )
