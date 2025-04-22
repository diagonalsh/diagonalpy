# diagonalpy

`diagonalpy` is a Python library for converting scikit-learn linear models to PyTorch and exporting them for production use.

## Features

- Export scikit-learn linear models the diagonal.sh inference platform
- Delete models deployed on the diagonal.sh inference platform

## Installation

Currently, `diagonalpy` is available for Python 3.9, 3.10, 3.11 and 3.12. Support for 3.13 will be added as soon as dependencies allow it.

```bash
pip install diagonalpy
```

torch is a dependency of `diagonalpy`, so if it isn't installed in the installation environment, you'll also have to run

```bash
pip install torch
```

## Quick Start

### Export a Model

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from diagonalpy.export import export

# Train a scikit-learn model
model = LinearRegression()
X = np.random.randn(100, 10)
y = np.sum(X, axis=1) + np.random.randn(100)
model.fit(X, y)

# Export the model
export(model, "my-wonderful-model")
```

### Delete a deployed model
```python
from diagonalpy.delete import delete

delete("model-id-from-export")
```

## Supported Models
### Regression Models:

 - LinearRegression
 - Ridge
 - RidgeCV
 - Lasso
 - LassoCV
 - ElasticNet
 - ElasticNetCV
 - Lars
 - LarsCV
 - LassoLars
 - LassoLarsCV
 - LassoLarsIC
 - OrthogonalMatchingPursuit
 - OrthogonalMatchingPursuitCV
 - BayesianRidge
 - ARDRegression
 - HuberRegressor
 - QuantileRegressor
 - TheilSenRegressor
 - TweedieRegressor

### Classification Models

 - LogisticRegression
 - LogisticRegressionCV
 - SGDClassifier
 - Perceptron
 - PassiveAggressiveClassifier
 - RidgeClassifier
 - RidgeClassifierCV

## Environment Variables

DIAGONALSH_API_KEY: Your Diagonal.sh API key (required)

DIAGONALSH_REGION: AWS region for deployment (required) - currently, only "eu-west-3" is valid

#### Environment Setup
```bash
export DIAGONALSH_API_KEY="your_api_key"
export DIAGONALSH_REGION="your_aws_region"
```

## License
This package is distributed under CC BY-ND license, which allows commercial use of the unmodified software and prohibits the distribution of any modifications of this software.
