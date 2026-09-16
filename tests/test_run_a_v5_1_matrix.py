from __future__ import annotations

import os
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
        "checkpoint config mismatch",
        "completion marker mismatch",
        "checkpoint hash mismatch",
    ):
        assert token in source


def test_matrix_runner_uses_expected_dataset_counts_and_never_deletes_runs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for count in (800, 201, 2400, 600, 2285, 1067):
        assert str(count) in source
    assert "rm -rf" not in source
    assert "git reset" not in source


def test_matrix_runner_resolves_its_location_without_external_dirname() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "dirname" not in source
    assert "basename" not in source
    assert "${BASH_SOURCE[0]%/*}" in source
    assert "find . -type f" in source


def _write_fake_python(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$FAKE_LOG"
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  if [[ "${2:-}" == *metrics.json && ! -f "$2" ]]; then exit 1; fi
  if [[ -d "${2:-}" && "${2:-}" == *train_* ]]; then
    printf '{}\\n' > "$2/formal_complete.json"
  fi
  exit 0
fi
if [[ "${1:-}" == "-m" ]]; then exit 0; fi
command_name="${1:-}"
run_dir=""
run_root=""
previous=""
for argument in "$@"; do
  if [[ "$previous" == "--run-dir" ]]; then run_dir="$argument"; fi
  if [[ "$previous" == "--run-root" ]]; then run_root="$argument"; fi
  previous="$argument"
done
if [[ "$command_name" == "train_a.py" ]]; then
  if [[ "$*" == *"${FAIL_CONFIG:-__never__}"* ]]; then exit 23; fi
  mkdir -p "$run_dir"
  : > "$run_dir/a_best.pt"
  : > "$run_dir/a_last.pt"
elif [[ "$command_name" == "eval_a.py" ]]; then
  mkdir -p "$run_dir"
  printf '{}\\n' > "$run_dir/metrics.json"
  printf 'sample_id\\n' > "$run_dir/per_image_metrics.csv"
  printf '{}\\n' > "$run_dir/stability.json"
elif [[ "$command_name" == "summarize_a_matrix.py" ]]; then
  printf '{}\\n' > "$run_root/matrix_summary.json"
  printf 'train_dataset\\n' > "$run_root/matrix_summary.csv"
  printf 'train_dataset\\n' > "$run_root/per_image_metrics.csv"
fi
""",
        encoding="utf-8",
        newline="\n",
    )
    path.chmod(0o755)


def test_matrix_runner_invokes_three_trains_and_evals_and_preserves_first_on_failure(
    tmp_path: Path,
) -> None:
    bash = shutil.which("bash")
    if bash is None:
        return
    fake = tmp_path / "fake-python"
    log = tmp_path / "calls.log"
    _write_fake_python(fake)
    common = {
        **os.environ,
        "PYTHON_BIN": fake.as_posix(),
        "DEVICE": "cpu",
        "SMOKE": "1",
        "LIMIT_TRAIN": "1",
        "LIMIT_VAL": "1",
        "EPOCHS": "1",
        "FAKE_LOG": log.as_posix(),
    }
    success_root = tmp_path / "debug-success"
    success_root_arg = os.path.relpath(success_root, SCRIPT.parent).replace("\\", "/")
    result = subprocess.run(
        [bash, str(SCRIPT.resolve())],
        cwd=SCRIPT.parent,
        env={**common, "RUN_ROOT": success_root_arg},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("train_a.py ") for line in calls) == 3
    assert sum(line.startswith("eval_a.py ") for line in calls) == 3

    failure_root = tmp_path / "debug-failure"
    failure_root_arg = os.path.relpath(failure_root, SCRIPT.parent).replace("\\", "/")
    failed = subprocess.run(
        [bash, str(SCRIPT.resolve())],
        cwd=SCRIPT.parent,
        env={
            **common,
            "RUN_ROOT": failure_root_arg,
            "FAIL_CONFIG": "a_v5_1_sirst_uavb.py",
        },
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert failed.returncode == 23
    assert (failure_root / "train_irstd1k_seed42" / "a_best.pt").is_file()


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
