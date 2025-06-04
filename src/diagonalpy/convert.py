from typing import Any
from diagonalpy.sklearn.linear_model import (
    convert_linear_regression,
    convert_logistic_regression,
    LINEAR_MODELS,
    CLASSIFICATION_MODELS,
)


def convert(model: Any) -> None:
    is_linear_regression = isinstance(model, LINEAR_MODELS)
    is_linear_classifier = isinstance(model, CLASSIFICATION_MODELS)
    if is_linear_regression:
        pytorch_model, input_size = convert_linear_regression(model)
        model_type = "regression"
    elif is_linear_classifier:
        pytorch_model, input_size = convert_logistic_regression(model)
        model_type = "classification"
    else:
        raise NotImplementedError(
            f"Convert not currently implemented for {type(model)}"
        )

    pytorch_model.eval()

    return pytorch_model, input_size, model_type
