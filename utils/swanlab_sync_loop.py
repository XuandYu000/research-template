"""
Background loop to sync offline SwanLab runs from compute nodes to the cloud.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import swanlab

from utils.swanlab_utils import SYNC_COMMAND_DIR


def main() -> None:
    command_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else SYNC_COMMAND_DIR
    command_dir.mkdir(parents=True, exist_ok=True)
    synced: set[str] = set()

    while True:
        for cmd_file in sorted(command_dir.glob("*")):
            try:
                run_dir = Path(cmd_file.read_text().strip())
                run_key = str(run_dir.resolve())
                if run_dir.is_dir() and run_key not in synced:
                    swanlab.sync(run_dir)
                    synced.add(run_key)
            except Exception as exc:
                print(f"Failed to sync {cmd_file}: {exc}", file=sys.stderr)
            finally:
                cmd_file.unlink(missing_ok=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
