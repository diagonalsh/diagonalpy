from sklearn.linear_model import LinearRegression
from typing import Any
from diagonalpy.sklearn.linear_model import export_linear_regression


def export(model: Any) -> None:
    if isinstance(model, LinearRegression):
        pytorch_model = export_linear_regression(model)
    else:
        raise NotImplementedError(f"Export not currently implemented for {type(model)}")

    return pytorch_model
