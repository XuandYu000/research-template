from __future__ import annotations

import urllib.request
from pathlib import Path

import swanlab
from swanlab.api.metric import Metric

from utils.swanlab_utils import CHECKPOINT_FILENAME


def is_run_id(run_id: str) -> bool:
    """Check if a string is a SwanLab run ID rather than a checkpoint path."""
    if "/" in run_id or "\\" in run_id or run_id.endswith(".ckpt"):
        return False
    forbidden = set("/\\#?%:")
    return 1 <= len(run_id) <= 64 and not any(char in forbidden for char in run_id)


def download_latest_checkpoint(run_path: str, download_dir: Path) -> Path:
    """Download the latest model checkpoint from a SwanLab cloud run."""
    api = swanlab.Api()
    run = api.run(run_path)
    experiment_id = run.run_id
    if not experiment_id:
        raise ValueError(f"SwanLab run not found: {run_path}")

    ckpt_path = None
    resp = run._get(f"/experiment/{experiment_id}/files/list")
    if resp.ok and isinstance(resp.data, dict):
        files = resp.data.get("files") or resp.data.get("list") or []
        ckpt_files = [
            file_info
            for file_info in files
            if str(file_info.get("path", file_info.get("name", ""))).endswith(".ckpt")
        ]
        if ckpt_files:
            latest = max(
                ckpt_files,
                key=lambda file_info: file_info.get("updatedAt", file_info.get("updated_at", 0)),
            )
            ckpt_path = latest.get("cosKey") or latest.get("path") or latest.get("name")

    if ckpt_path is None:
        ckpt_path = CHECKPOINT_FILENAME

    url_map = Metric._fetch_file_presigned_urls(run, [ckpt_path])
    download_url = url_map.get(ckpt_path)
    if not download_url:
        raise FileNotFoundError(
            f"Could not download checkpoint from SwanLab run {run_path}. Tried file path: {ckpt_path}"
        )

    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_path
    root.mkdir(exist_ok=True, parents=True)
    dest = root / CHECKPOINT_FILENAME
    urllib.request.urlretrieve(download_url, dest)
    return dest
