import numpy as np
import pytest
from sklearn.linear_model import (
    HuberRegressor,
    QuantileRegressor,
    TheilSenRegressor,
    SGDClassifier,
)
from diagonalpy.convert import convert
from diagonalpy.sklearn.linear_model import LINEAR_MODELS, CLASSIFICATION_MODELS


def setup_regression_data():
    """Create random regression data."""
    X = np.random.randn(1000, 10)
    y = np.sum(X, axis=1) + np.random.randn(1000)
    return X, y


def setup_classification_data():
    """Create random classification data."""
    X = np.random.randn(1000, 10)
    y = (np.sum(X, axis=1) / 1.5).astype(int)
    y = y - np.min(y)
    y[y < (np.min(y) + 4)] = np.min(y) + 4
    y[y > (np.max(y) - 4)] = np.max(y) - 4
    return X, y


def setup_binary_classification_data():
    """Create random classification data."""
    X = np.random.randn(1000, 10)
    y = (np.sum(X, axis=1) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("model_class", LINEAR_MODELS)
def test_convert_regression_models(model_class):
    """Test the convert function with regression models."""
    # Some models may need specific parameters
    if model_class == QuantileRegressor:
        model = model_class(solver="highs")
    elif model_class in [HuberRegressor, TheilSenRegressor]:
        model = model_class(max_iter=100)
    else:
        model = model_class()

    X, y = setup_regression_data()
    model.fit(X, y)

    # Test the convert function
    converted_model = convert(model)
    assert converted_model is not None, f"Conversion failed for {model_class.__name__}"


@pytest.mark.parametrize("model_class", CLASSIFICATION_MODELS)
def test_convert_classification_models(model_class):
    """Test the convert function with classification models."""
    # Some classification models may need specific parameters
    if model_class == SGDClassifier:
        model = model_class(max_iter=100)
    else:
        model = model_class()

    X, y = setup_classification_data()
    model.fit(X, y)

    # Test the convert function
    converted_model = convert(model)
    assert converted_model is not None, f"Conversion failed for {model_class.__name__}"


@pytest.mark.parametrize("model_class", CLASSIFICATION_MODELS)
def test_convert_binary_classification_models(model_class):
    """Test the convert function with classification models."""
    # Some classification models may need specific parameters
    if model_class == SGDClassifier:
        model = model_class(max_iter=100)
    else:
        model = model_class()

    X, y = setup_binary_classification_data()
    model.fit(X, y)

    # Test the convert function
    converted_model = convert(model)
    assert converted_model is not None, f"Conversion failed for {model_class.__name__}"
