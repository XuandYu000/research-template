"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research
template [repo](https://github.com/buoyancy99/research-template).
By its MIT license, you must keep the above sentence in `README.md`
and the `LICENSE` file to credit the author.
"""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Union

import swanlab
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.types import _PATH
from swanlab.integration.pytorch_lightning import SwanLabLogger
from typing_extensions import override

CHECKPOINT_FILENAME = "model.ckpt"
SYNC_COMMAND_DIR = Path(".swanlab_sync_command_dir")


def _map_mode(offline: bool, mode: Optional[str] = None) -> str:
    if mode is not None:
        if mode == "dryrun":
            return "disabled"
        return mode
    return "offline" if offline else "online"


class SpaceEfficientSwanLabLogger(SwanLabLogger):
    """
    A SwanLab logger that uploads checkpoints via swanlab.save.
    Checkpoints are copied to a stable filename before upload so they can be
    downloaded later with download_latest_checkpoint().
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        offline: bool = False,
        mode: Optional[str] = None,
        project: Optional[str] = None,
        workspace: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        id: Optional[str] = None,
        checkpoint_name: Optional[str] = None,
        expiration_days: Optional[int] = 5,
        config: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        init_kwargs: dict[str, Any] = {
            "project": project,
            "workspace": workspace,
            "experiment_name": name,
            "save_dir": str(save_dir),
            "mode": _map_mode(offline, mode),
        }
        if config is not None:
            init_kwargs["config"] = config
        if id is not None:
            init_kwargs["id"] = id
            init_kwargs["resume"] = "allow"
        init_kwargs.update(kwargs)
        super().__init__(**init_kwargs)

        self._log_model = log_model
        self._checkpoint_name = checkpoint_name or CHECKPOINT_FILENAME
        self._checkpoint_callbacks: dict[int, ModelCheckpoint] = {}
        self._logged_model_time: dict[str, float] = {}
        self._offline = offline or init_kwargs["mode"] == "offline"
        self.expiration_days = expiration_days

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        if isinstance(params, Mapping):
            self.experiment.config.update(dict(params))
        else:
            super().log_hyperparams(params, *args, **kwargs)

    @override
    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        if self._log_model == "all" or (self._log_model is True and checkpoint_callback.save_top_k == -1):
            self._scan_and_log_checkpoints(checkpoint_callback)
        elif self._log_model is True:
            self._checkpoint_callbacks[id(checkpoint_callback)] = checkpoint_callback

    @override
    @rank_zero_only
    def finalize(self, status: Optional[str] = None) -> None:
        if status in (None, "success"):
            for checkpoint_callback in self._checkpoint_callbacks.values():
                self._scan_and_log_checkpoints(checkpoint_callback)
        super().finalize(status)

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)
        checkpoint_dir = Path(checkpoint_callback.dirpath)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        upload_path = checkpoint_dir / self._checkpoint_name

        for t, p, s, tag in checkpoints:
            del s, tag
            shutil.copy2(p, upload_path)
            swanlab.save(str(upload_path), base_path=checkpoint_dir, policy="now")
            self._logged_model_time[p] = t


class OfflineSwanLabLogger(SpaceEfficientSwanLabLogger):
    """
    Wraps SwanLabLogger to queue offline run directories for syncing on login nodes.
    This is useful when running on slurm clusters that only have internet on login nodes.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        offline: bool = False,
        mode: Optional[str] = None,
        project: Optional[str] = None,
        workspace: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        id: Optional[str] = None,
        checkpoint_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            offline=True,
            mode=mode or "offline",
            project=project,
            workspace=workspace,
            log_model=log_model,
            id=id,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        SYNC_COMMAND_DIR.mkdir(parents=True, exist_ok=True)
        self.last_sync_time = 0.0
        self.min_sync_interval = 60

    def _queue_sync(self) -> None:
        run_dir = Path(self.experiment.dir)
        command_file = SYNC_COMMAND_DIR / f"{run_dir.name}-{uuid.uuid4().hex[:8]}"
        command_file.write_text(str(run_dir.resolve()))

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        out = super().log_metrics(metrics, step)
        if time.time() - self.last_sync_time > self.min_sync_interval:
            self._queue_sync()
            self.last_sync_time = time.time()
        return out
