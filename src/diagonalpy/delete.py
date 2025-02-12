import os
import requests
from typing import Any
from requests.exceptions import RequestException

API_TIMEOUT = 600  # seconds
API_URL = "https://api.diagonal.sh/model-delete-api"


def delete(model_id: str) -> dict[str, Any]:
    api_key = os.getenv("DIAGONALSH_API_KEY")
    if api_key is None:
        raise EnvironmentError("Please set DIAGONALSH_API_KEY")

    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.delete(
            f"{API_URL}/{model_id}", headers=headers, timeout=API_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        raise RuntimeError(f"Failed to delete model: {str(e)}")
