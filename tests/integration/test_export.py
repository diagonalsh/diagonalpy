import numpy as np
from sklearn.linear_model import LinearRegression
from diagonalpy.export import export
from diagonalpy.sklearn.linear_model import (
    convert_sklearn_linear_to_pytorch,
)


def test_export():
    lr = LinearRegression()
    X = np.random.randn(50, 10)
    y = np.sum(X, axis=1) + np.random.randn(50)
    lr.fit(X, y)
    export(lr)
    convert_sklearn_linear_to_pytorch(lr)
