import json
import os
import zipfile
from io import BytesIO
from typing import Dict, List, Optional

import requests


class ServerClient:
    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url or os.environ.get(
            "SERVER_URL", "http://127.0.0.1:9924")

    @staticmethod
    def parse_model_ref(model_ref: str) -> tuple[str, str]:
        """Parse "project_id/model_version" into (project_id, version)."""
        parts = model_ref.split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid model ref format: '{model_ref}'. "
                "Expected 'project_id/model_version'"
            )
        return parts[0], parts[1]

    def download_model(self, project_id_or_ref: str, version: Optional[str] = None,
                       cache_dir: str = "/app/cache") -> str:
        """Download model from server.

        Supports two calling conventions:
          1. Combined: download_model("proj/v1", cache_dir="/app/cache")
          2. Legacy:   download_model("proj", "v1", cache_dir="/app/cache")
        """
        if version is None:
            project_id, version = self.parse_model_ref(project_id_or_ref)
        else:
            project_id = project_id_or_ref
        url = f"{self.server_url}/api/model/download"
        params = {"project": project_id, "version": version}
        response = requests.get(url, params=params, stream=True, timeout=300)
        response.raise_for_status()

        model_dir = f"{cache_dir}/models/{project_id}_{version}"
        os.makedirs(model_dir, exist_ok=True)

        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            zf.extractall(model_dir)

        return model_dir

    def list_model_versions(self, project_id: str) -> List[Dict]:
        """List available model versions for a project."""
        url = f"{self.server_url}/api/model/versions"
        params = {"project": project_id}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("versions", [])

    def list_workflows(self, project_id: str) -> List[Dict]:
        """List available workflows for a project from the server."""
        url = f"{self.server_url}/api/workflow/list"
        params = {"project": project_id}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("workflows", data if isinstance(data, list) else [])
