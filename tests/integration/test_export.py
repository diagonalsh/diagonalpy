import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from diagonalpy.export import export
from diagonalpy.sklearn.linear_model import (
    convert_sklearn_linear_to_pytorch,
)


def test_export_linear_regression():
    lr = LinearRegression()
    X = np.random.randn(50, 10)
    y = np.sum(X, axis=1) + np.random.randn(50)
    lr.fit(X, y)
    export(lr)
    convert_sklearn_linear_to_pytorch(lr)


def test_export_logistic_regression_multiclass():
    clf = LogisticRegression()
    X = np.random.randn(50, 10)
    y = ((np.sum(X, axis=1) + np.random.randn(50)) * 0.5).astype(int)
    y = y - np.min(y)
    y[y < 2] = 2
    y[y > 4] = 4
    y = y - np.min(y)
    clf.fit(X, y)
    export(clf)


def test_export_logistic_regression_binary():
    clf = LogisticRegression()
    X = np.random.randn(50, 10)
    y = ((np.sum(X, axis=1) + np.random.randn(50)) * 0.5).astype(int)
    y = y - np.min(y)
    y[y < 2] = 2
    y[y > 3] = 3
    y = y - np.min(y)
    clf.fit(X, y)
    export(clf)
