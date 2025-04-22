# diagonalpy

A Python library for converting scikit-learn linear models to PyTorch and exporting them for production use.

## Features

- Export scikit-learn linear models the diagonal.sh inference platform
- Delete models deployed on the diagonal.sh inference platform

## Installation

```bash
pip install diagonalpy
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
