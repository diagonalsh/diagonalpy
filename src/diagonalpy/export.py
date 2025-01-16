import os
import requests
from pathlib import Path
import tempfile
from typing import Any
import torch
from requests.exceptions import RequestException
from diagonalpy.convert import convert

API_TIMEOUT = 600  # seconds
API_URL = "http://localhost:8000/model-upload-api"


def export(
    model: Any,
    model_name: str,
) -> dict[str, Any]:
    api_key = os.getenv("DIAGONALSH_API_KEY")
    if api_key is None:
        raise EnvironmentError("Please set DIAGONALSH_API_KEY")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            pytorch_model, input_size = convert(model)
        except Exception as e:
            raise ValueError(f"Failed to convert model: {str(e)}")

        onnx_path = Path(temp_dir) / f"{model_name}.onnx"

        try:
            torch.onnx.export(pytorch_model, torch.randn(1, input_size), onnx_path)
        except Exception as e:
            raise ValueError(f"Failed to export model to ONNX: {str(e)}")

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        try:
            with open(onnx_path, "rb") as f:
                files = {"file": (onnx_path.name, f)}  # Changed from "model" to "file"
                response = requests.post(
                    API_URL, headers=headers, files=files, timeout=API_TIMEOUT
                )
                response.raise_for_status()
                return response.json()
        except RequestException as e:
            raise RuntimeError(f"Failed to upload model: {str(e)}")
