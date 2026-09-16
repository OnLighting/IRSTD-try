from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


SCRIPT = Path("run_a_v5_1_matrix.sh")


def test_matrix_runner_contains_guarded_complete_workflow() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for token in (
        "set -euo pipefail",
        "BASH_SOURCE[0]",
        "torch.cuda.is_available",
        "a_v5_1_irstd1k.py",
        "a_v5_1_sirst_uavb.py",
        "a_v5_1_sirst4.py",
        "a_last.pt",
        "--resume",
        "train_a.py",
        "eval_a.py",
        "summarize_a_matrix.py",
        "matrix_manifest.sha256",
        "sha256sum",
        "tar -czf",
    ):
        assert token in source


def test_matrix_runner_uses_expected_dataset_counts_and_never_deletes_runs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for count in (800, 201, 2400, 600, 2285, 1067):
        assert str(count) in source
    assert "rm -rf" not in source
    assert "git reset" not in source


def test_matrix_runner_has_valid_bash_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        return

    result = subprocess.run(
        [bash, "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, result.stderr
