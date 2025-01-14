import os
import requests
import shutil
from typing import Any
import torch
from diagonalpy.convert import convert


def export(model: Any, model_name: str) -> None:
    api_key = os.getenv("DIAGONALSH_API_KEY")
    if api_key is None:
        raise EnvironmentError("Please set DIAGONALSH_API_KEY")

    os.makedirs(".diagonalsh_model_export")
    pytorch_model, input_size = convert(model)

    onnx_path = os.path.join(".diagonalsh_model_export", model_name)
    torch.onnx.export(pytorch_model, torch.randn(1, input_size), onnx_path)

    try:
        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/octet-stream",
        }

        # Read the ONNX file in binary mode
        with open(onnx_path, "rb") as f:
            files = {"model": (model_name, f)}

            # Send the POST request to the API
            response = requests.post(
                "https://api.diagonal.sh/v1/models", headers=headers, files=files
            )

            # Check if the request was successful
            response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to upload model: {str(e)}")

    finally:
        shutil.rmtree(".diagonalsh_model_export", ignore_errors=True)
