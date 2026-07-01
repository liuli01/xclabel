import os
import zipfile
from io import BytesIO
from typing import Optional

import requests


class ServerClient:
    def __init__(self, server_url: Optional[str] = None):
        self.server_url = (server_url or os.environ.get(
            "SERVER_URL", "http://127.0.0.1:9924")).rstrip("/")

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

    @staticmethod
    def _model_dir(cache_dir: str, project_id: str, version: str) -> str:
        return f"{cache_dir}/models/{project_id}_{version}"

    @staticmethod
    def _has_cached_model(model_dir: str) -> bool:
        """Check if model_dir exists and contains at least one model file."""
        if not os.path.isdir(model_dir):
            return False
        for fname in os.listdir(model_dir):
            if fname.endswith(('.pt', '.engine', '.onnx')):
                return True
        return False

    @staticmethod
    def _model_cache_keep_files() -> set:
        """Files to keep when extracting model zip; everything else is deleted."""
        return {
            'best.pt', 'best.engine', 'best.onnx',
            'model_info.json', 'deploy_metadata.json', 'args.yaml',
        }

    @staticmethod
    def _clean_model_extraction(model_dir: str):
        """Remove training artifacts, keep only inference-essential files."""
        keep = ServerClient._model_cache_keep_files()
        for fname in os.listdir(model_dir):
            fpath = os.path.join(model_dir, fname)
            if fname in keep:
                continue
            try:
                if os.path.isdir(fpath):
                    import shutil
                    shutil.rmtree(fpath)
                else:
                    os.remove(fpath)
            except Exception as e:
                print(f"[model_cache] Failed to remove '{fname}': {e}", flush=True)

    def download_model(self, project_id_or_ref: str, version: Optional[str] = None,
                       cache_dir: str = "/app/cache") -> str:
        """Download model from server with disk cache.

        If the model is already on disk under {cache_dir}/models/{project}_{version}/,
        skip the HTTP download and return the cached path directly.

        Only keeps inference-essential files; training artifacts are cleaned up.

        Supports two calling conventions:
          1. Combined: download_model("proj/v1", cache_dir="/app/cache")
          2. Legacy:   download_model("proj", "v1", cache_dir="/app/cache")
        """
        if version is None:
            project_id, version = self.parse_model_ref(project_id_or_ref)
        else:
            project_id = project_id_or_ref

        model_dir = self._model_dir(cache_dir, project_id, version)

        # ── Disk cache check ──
        if self._has_cached_model(model_dir):
            # Clean up any old training artifacts that may linger from previous versions
            self._clean_model_extraction(model_dir)
            print(f"[model_cache] HIT '{project_id}/{version}' at {model_dir}", flush=True)
            return model_dir

        # ── Download from server ──
        print(f"[model_cache] MISS '{project_id}/{version}', downloading from server...", flush=True)
        url = f"{self.server_url}/api/model/download"
        params = {"project": project_id, "version": version}
        response = requests.get(url, params=params, stream=True, timeout=300)
        response.raise_for_status()

        os.makedirs(model_dir, exist_ok=True)

        # Extract to temp dir first, move only essential files, then clean up
        import tempfile
        tmp_extract = tempfile.mkdtemp(dir=model_dir)
        try:
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                zf.extractall(tmp_extract)

            keep = self._model_cache_keep_files()
            for fname in os.listdir(tmp_extract):
                if fname in keep:
                    src = os.path.join(tmp_extract, fname)
                    dst = os.path.join(model_dir, fname)
                    # Handle nested model_info.json (may be in subdirs like runs/train/exp/)
                    if os.path.isfile(src):
                        os.replace(src, dst)
                    elif os.path.isdir(src):
                        import shutil
                        shutil.copytree(src, dst, dirs_exist_ok=True)
        finally:
            import shutil
            shutil.rmtree(tmp_extract, ignore_errors=True)

        # Clean up any remaining non-essential files from model_dir
        self._clean_model_extraction(model_dir)

        print(f"[model_cache] Downloaded '{project_id}/{version}' to {model_dir}", flush=True)
        return model_dir
