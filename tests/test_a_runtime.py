from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from irstd_a.runtime import (
    _git_value,
    atomic_json_dump,
    capture_rng_state,
    prepare_new_run,
    restore_rng_state,
    save_checkpoint,
)


def test_prepare_new_run_accepts_new_or_empty_and_rejects_artifacts(tmp_path: Path) -> None:
    new_run = tmp_path / "new"
    prepare_new_run(new_run)
    assert new_run.is_dir()
    prepare_new_run(new_run)

    (new_run / "train.jsonl").write_text("record", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_new_run(new_run)


def test_atomic_json_dump_leaves_valid_json_and_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"

    atomic_json_dump({"value": 3, "text": "红外"}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 3, "text": "红外"}
    assert list(tmp_path.iterdir()) == [path]


def test_save_checkpoint_round_trips_and_returns_sha256(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    payload = {"epoch": 7, "tensor": torch.arange(5)}

    digest = save_checkpoint(path, payload)
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    assert loaded["epoch"] == 7
    assert torch.equal(loaded["tensor"], payload["tensor"])
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not any(item.suffix == ".tmp" for item in tmp_path.iterdir())


def test_rng_state_round_trip_restores_python_numpy_and_torch() -> None:
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    state = capture_rng_state()

    expected = (random.random(), float(np.random.rand()), torch.rand(4))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(4))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_git_value_decodes_non_ascii_paths_as_utf8(tmp_path: Path) -> None:
    """Catches locale-dependent decoding of Git's UTF-8 path output."""
    if shutil.which("git") is None:
        pytest.skip("git is unavailable")
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "core.quotepath", "false"],
        check=True,
    )
    (repository / "分解.txt").write_text("A", encoding="utf-8")

    output, error = _git_value(["-C", str(repository), "status", "--porcelain"])

    assert error is None
    assert "分解.txt" in output
