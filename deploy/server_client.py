import json
import os
import zipfile
from io import BytesIO
from typing import Dict, List, Optional

import requests


class ServerClient:
    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url or os.environ.get(
            "SERVER_URL", "http://xclabel-server:5000")

    def download_model(self, project_id: str, version: str,
                       cache_dir: str) -> str:
        url = f"{self.server_url}/api/model/download"
        params = {"project": project_id, "version": version}
        response = requests.get(url, params=params, stream=True, timeout=300)
        response.raise_for_status()

        model_dir = f"{cache_dir}/models/{project_id}_{version}"
        os.makedirs(model_dir, exist_ok=True)

        # Extract zip
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            zf.extractall(model_dir)

        return model_dir

    def list_model_versions(self, project_id: str) -> List[Dict]:
        url = f"{self.server_url}/api/model/versions"
        params = {"project": project_id}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("versions", [])

    def list_workflows(self) -> List[Dict]:
        """List nndeploy workflows from server."""
        url = f"{self.server_url}/api/nndeploy/workflows"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        # nndeploy-app returns {flag, message, result: [...]}
        workflows = data.get("result", []) if "result" in data else data.get("workflows", [])
        return workflows

    def download_workflow(self, workflow_id: str, cache_dir: str) -> str:
        """Download nndeploy workflow JSON by id."""
        url = f"{self.server_url}/api/nndeploy/workflow/download"
        params = {"id": workflow_id}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        workflow_path = f"{cache_dir}/workflows/{workflow_id}.json"
        os.makedirs(os.path.dirname(workflow_path), exist_ok=True)

        with open(workflow_path, "w", encoding="utf-8") as f:
            json.dump(response.json(), f, ensure_ascii=False, indent=2)

        return workflow_path
