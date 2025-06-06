# diagonalpy

`diagonalpy` is a Python library for deploying scikit-learn linear models to the inference platform diagonal.sh

## Features

- Save scikit-learn linear models locally, as `onnx` file
- Deploy scikit-learn linear models to the diagonal.sh inference platform
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

### Save a Model

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from diagonalpy.save import save

# Train a scikit-learn model
model = LinearRegression()
X = np.random.randn(100, 10)
y = np.sum(X, axis=1) + np.random.randn(100)
model.fit(X, y)

save(model, "/my-path/model.onnx")

```

You can then upload the model manually at [console.diagonal.sh](https://console.diagonal.sh).


### Deploy a Model

Once you have an account *and created a console key*, you can set it to the env variable `DIAGONALSH_API_KEY`, and deploy directly. You'll also have to set the env variable `DIAGONALSH_REGION` to `eu-west-3`. This is the snippet:

```python
import os
os.environ['DIAGONALSH_API_KEY'] = 'dia_console_...'
os.environ['DIAGONALSH_REGION'] = 'eu-west-3'
```

Then, you can deploy directly from `diagonalpy`:
```python
from diagonalpy.deploy import deploy

deploy(model, "my-wonderful-model")
```

The deployed model will be available for inference usually after 10 to 15 minutes.

### Inferring your deployed model

If you want to get the model output for the model you deployed, you can do this via a `HTTP` `POST` request, like this:

```bash
curl -X POST -H "Content-Type: application/json" -H "X-API-Key: YOUR_INFERENCE_API_KEY" -d "{\"data\": YOUR_DATA, \"model\":\"YOUR_MODEL_ID\"}" "https://infer.diagonal.sh/YOUR_MODEL_ROUTE" &
```

The data for a model like the one trained above would look like this:

`[-5.0, 2.0, 0.0, 1.0, 3.0, -2.0, 1.0, 0.0, 2.0, 5.0]`

`YOUR_DATA` should be replaced with an unnested list containing comma-separated values that can be integers or floats, but will be interpreted as floats either way.

The `inference_api_key` can be created under `Settings` on [console.diagonal.sh](https://console.diagonal.sh). The `model_id` and `model_route` are visible on the models overview on [console.diagonal.sh](https://console.diagonal.sh).


### Delete a deployed model

Deleting a model on `diagonal.sh` is easy with `diagonalpy`:

```python
from diagonalpy.delete import delete

delete("model-id-from-deployment")
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
export DIAGONALSH_REGION="eu-west-3"
```

## License
This package is distributed under CC BY-ND license, which allows commercial use of the unmodified software and prohibits the distribution of any modifications of this software.
