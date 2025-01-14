import torch
import torch.nn as nn
import numpy as np
from typing import Any, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from diagonalpy.models.linear_model import (
    LinearRegressionPyTorch,
    LogisticRegressionPyTorch,
)


def convert_sklearn_linear_to_pytorch(
    sklearn_model: LinearRegression,
) -> Tuple[LinearRegressionPyTorch, dict]:
    """
    Convert a trained scikit-learn LinearRegression model to PyTorch.

    Parameters:
    -----------
    sklearn_model : LinearRegression
        Trained scikit-learn linear regression model

    Returns:
    --------
    pytorch_model : LinearRegressionPyTorch
        Equivalent PyTorch model
    conversion_info : dict
        Dictionary containing conversion details and verification results

    Example:
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> import numpy as np
    >>>
    >>> # Create and train sklearn model
    >>> X = np.random.randn(100, 3)
    >>> y = X @ np.array([1, 2, 3]) + 0.5 + np.random.randn(100) * 0.1
    >>> sklearn_model = LinearRegression().fit(X, y)
    >>>
    >>> # Convert to PyTorch
    >>> pytorch_model, info = convert_sklearn_linear_to_pytorch(sklearn_model)
    """
    if not isinstance(sklearn_model, LinearRegression):
        raise TypeError("Model must be a scikit-learn LinearRegression instance")

    if not hasattr(sklearn_model, "coef_"):
        raise ValueError("Model must be trained before conversion")

    # Create PyTorch model
    input_dim = sklearn_model.coef_.shape[0]
    pytorch_model = LinearRegressionPyTorch(input_dim, bias=True)

    # Convert weights and bias
    weights = torch.FloatTensor(sklearn_model.coef_)
    bias = torch.FloatTensor([sklearn_model.intercept_])

    # Assign parameters
    pytorch_model.linear.weight.data = weights.view(1, -1)
    pytorch_model.linear.bias.data = bias

    # Verify conversion
    conversion_info = {
        "input_dim": input_dim,
        "weights_shape": tuple(weights.shape),
        "original_coefficients": sklearn_model.coef_.tolist(),
        "converted_coefficients": weights.view(-1).tolist(),
        "original_intercept": float(sklearn_model.intercept_),
        "converted_intercept": float(bias.item()),
    }

    return pytorch_model, conversion_info


def verify_conversion_linear_regression(
    sklearn_model: LinearRegression,
    pytorch_model: LinearRegressionPyTorch,
    X: np.ndarray,
    rtol: float = 1e-5,
) -> bool:
    """
    Verify that the converted PyTorch model produces the same predictions
    as the original scikit-learn model.

    Parameters:
    -----------
    sklearn_model : LinearRegression
        Original scikit-learn model
    pytorch_model : LinearRegressionPyTorch
        Converted PyTorch model
    X : np.ndarray
        Input data for verification
    rtol : float
        Relative tolerance for numerical comparison

    Returns:
    --------
    bool
        True if predictions match within tolerance
    """
    # Get predictions from both models
    sklearn_pred = sklearn_model.predict(X)

    # Convert input to PyTorch tensor
    X_torch = torch.FloatTensor(X)
    with torch.no_grad():
        pytorch_pred = pytorch_model(X_torch).numpy().flatten()

    # Compare predictions
    return np.allclose(sklearn_pred, pytorch_pred, rtol=rtol)


def convert_linear_regression(model: Any) -> nn.Module:
    test_array = np.random.randn(100, model.coef_.shape[0])

    rtol = 1e-10
    while rtol <= 1e-2:
        pytorch_model, conversion_info = convert_sklearn_linear_to_pytorch(model)
        if (
            verify_conversion_linear_regression(model, pytorch_model, test_array, rtol)
            is False
        ):
            print(f"Conversion failed at {rtol:.1e}")
            rtol *= 10
        else:
            print(f"Conversion succeeded at {rtol:.1e}")
            return pytorch_model, model.coef_.shape[0]


def convert_sklearn_logistic_to_pytorch(
    sklearn_model: LogisticRegression,
) -> Tuple[LogisticRegressionPyTorch, dict]:
    """
    Convert a trained scikit-learn LogisticRegression model to PyTorch.

    Parameters:
    -----------
    sklearn_model : LogisticRegression
        Trained scikit-learn logistic regression model

    Returns:
    --------
    pytorch_model : LogisticRegressionPyTorch
        Equivalent PyTorch model
    conversion_info : dict
        Dictionary containing conversion details and verification results

    Example:
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>>
    >>> # Create and train sklearn model for binary classification
    >>> X = np.random.randn(100, 3)
    >>> y = (X @ np.array([1, 2, 3]) + 0.5 > 0).astype(int)
    >>> sklearn_model = LogisticRegression().fit(X, y)
    >>>
    >>> # Convert to PyTorch
    >>> pytorch_model, info = convert_sklearn_logistic_to_pytorch(sklearn_model)
    """
    if not isinstance(sklearn_model, LogisticRegression):
        raise TypeError("Model must be a scikit-learn LogisticRegression instance")

    if not hasattr(sklearn_model, "coef_"):
        raise ValueError("Model must be trained before conversion")

    # Get input dimension and number of classes
    input_dim = sklearn_model.coef_.shape[1]

    # Handle both binary and multilabel cases
    is_binary = len(sklearn_model.classes_) == 2
    output_dim = 1 if is_binary else len(sklearn_model.classes_)

    # Create PyTorch model
    pytorch_model = LogisticRegressionPyTorch(input_dim, output_dim=output_dim)

    # Convert weights and bias
    if is_binary:
        # For binary case, sklearn stores one set of coefficients
        weights = torch.FloatTensor(sklearn_model.coef_)
        bias = torch.FloatTensor([sklearn_model.intercept_])
    else:
        # For multilabel case, sklearn stores coefficients for each class
        weights = torch.FloatTensor(sklearn_model.coef_)
        bias = torch.FloatTensor(sklearn_model.intercept_)

    # Assign parameters
    pytorch_model.linear.weight.data = weights
    pytorch_model.linear.bias.data = bias

    # Verify conversion
    conversion_info = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "is_binary": is_binary,
        "weights_shape": tuple(weights.shape),
        "original_coefficients": sklearn_model.coef_.tolist(),
        "converted_coefficients": weights.view(-1).tolist(),
        "original_intercept": sklearn_model.intercept_.tolist(),
        "converted_intercept": bias.tolist(),
        "classes": sklearn_model.classes_.tolist(),
    }

    return pytorch_model, conversion_info


def verify_conversion_logistic_regression(
    sklearn_model: LogisticRegression,
    pytorch_model: LogisticRegressionPyTorch,
    X: np.ndarray,
    rtol: float = 1e-5,
) -> bool:
    """
    Verify that the converted PyTorch model produces the same predictions
    as the original scikit-learn model.

    Parameters:
    -----------
    sklearn_model : LogisticRegression
        Original scikit-learn model
    pytorch_model : LogisticRegressionPyTorch
        Converted PyTorch model
    X : np.ndarray
        Input data for verification
    rtol : float
        Relative tolerance for numerical comparison

    Returns:
    --------
    bool
        True if predictions match within tolerance
    """
    # Get predictions from both models
    sklearn_pred = sklearn_model.predict_proba(X)

    # Convert input to PyTorch tensor
    X_torch = torch.FloatTensor(X)
    with torch.no_grad():
        pytorch_pred = pytorch_model(X_torch).numpy()

        # For binary case, convert pytorch output to match sklearn format
        if len(sklearn_model.classes_) == 2:
            pytorch_pred = np.column_stack([1 - pytorch_pred, pytorch_pred])

    # Compare predictions
    return np.allclose(sklearn_pred, pytorch_pred, rtol=rtol)


def convert_logistic_regression(model: Any) -> nn.Module:
    """
    Convert a scikit-learn LogisticRegression model to PyTorch with automatic
    verification.

    Parameters:
    -----------
    model : LogisticRegression
        Trained scikit-learn logistic regression model

    Returns:
    --------
    nn.Module
        Converted PyTorch model
    """
    test_array = np.random.randn(100, model.coef_.shape[1])

    rtol = 1e-10
    while rtol <= 1e-2:
        pytorch_model, conversion_info = convert_sklearn_logistic_to_pytorch(model)
        if (
            verify_conversion_logistic_regression(
                model, pytorch_model, test_array, rtol
            )
            is False
        ):
            print(f"Conversion failed at {rtol:.1e}")
            rtol *= 10
        else:
            print(f"Conversion succeeded at {rtol:.1e}")
            return pytorch_model, model.coef_.shape[0]

    raise ValueError("Could not convert model within acceptable tolerance")
