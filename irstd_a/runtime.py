"""Safe, reproducible experiment runtime helpers for module A."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def prepare_new_run(path: Path) -> None:
    """Create an empty run directory and refuse to reuse artifact directories."""
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"run path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"run directory is not empty: {path}")
        return
    path.mkdir(parents=True, exist_ok=False)


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def atomic_json_dump(data: dict, path: Path) -> None:
    """Write UTF-8 JSON atomically in the destination directory."""
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    """Atomically save a PyTorch payload and return its SHA-256 digest."""
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(destination)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _git_value(arguments: list[str]) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        return result.stdout.strip(), None
    except (OSError, subprocess.SubprocessError) as error:
        return None, str(error)


def environment_manifest() -> dict[str, Any]:
    """Capture runtime and source provenance without requiring CUDA or Git."""
    git_root, git_root_error = _git_value(["rev-parse", "--show-toplevel"])
    git_commit, git_commit_error = _git_value(["rev-parse", "HEAD"])
    git_status, git_status_error = _git_value(["status", "--porcelain"])
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "gpu_names": gpu_names,
        "git": {
            "root": git_root,
            "root_error": git_root_error,
            "commit": git_commit,
            "commit_error": git_commit_error,
            "dirty": bool(git_status) if git_status is not None else None,
            "status_error": git_status_error,
        },
    }
