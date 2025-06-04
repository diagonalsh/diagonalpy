from typing import Any
import torch
from diagonalpy.convert import convert


def save(model: Any, model_path: str) -> None:
    if not model_path.endswith(".onnx"):
        raise ValueError("Only paths with file ending '.onnx' are allowed")

    try:
        pytorch_model, input_size, _ = convert(model)
    except Exception as e:
        raise ValueError(f"Failed to convert model: {str(e)}")

    try:
        torch.onnx.export(
            pytorch_model, torch.randn(1, input_size), model_path, opset_version=19
        )
    except Exception as e:
        raise ValueError(f"Failed to export model to ONNX: {str(e)}")
