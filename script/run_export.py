import numpy as np
from sklearn.linear_model import LinearRegression
from diagonalpy.deploy import deploy

if __name__ == "__main__":
    lr = LinearRegression()
    X = np.random.randn(50, 10)
    y = np.sum(X, axis=1) + np.random.randn(50)

    lr.fit(X, y)

    deploy(lr, "test-model3")
