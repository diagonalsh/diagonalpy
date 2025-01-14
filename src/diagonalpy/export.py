import os
import requests
from pathlib import Path
import tempfile
from typing import Any
import torch
from diagonalpy.convert import convert


def export(model: Any, model_name: str) -> dict[str, Any]:
    api_key = os.getenv("DIAGONALSH_API_KEY")
    if api_key is None:
        raise EnvironmentError("Please set DIAGONALSH_API_KEY")

    # Use a temporary directory that's automatically cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            pytorch_model, input_size = convert(model)

            onnx_path = Path(temp_dir) / f"{model_name}.onnx"

            torch.onnx.export(pytorch_model, torch.randn(1, input_size), onnx_path)

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/octet-stream",
            }

            # Use context manager for file handling
            with open(onnx_path, "rb") as f:
                files = {"model": (model_name, f)}
                response = requests.post(
                    "https://api.diagonal.sh/v1/models", headers=headers, files=files
                )
                response.raise_for_status()
                return response.json()

        except torch.onnx.ExportError as e:
            raise ValueError(f"Failed to export model to ONNX: {str(e)}")
