from sklearn.linear_model import LinearRegression, LogisticRegression
from typing import Any
from diagonalpy.sklearn.linear_model import (
    convert_linear_regression,
    convert_logistic_regression,
)


def convert(model: Any) -> None:
    if isinstance(model, LinearRegression):
        pytorch_model = convert_linear_regression(model)
    elif isinstance(model, LogisticRegression):
        pytorch_model = convert_logistic_regression(model)
    else:
        raise NotImplementedError(
            f"Convert not currently implemented for {type(model)}"
        )

    return pytorch_model
