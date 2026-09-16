#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
if [[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then
  SCRIPT_DIR="."
fi
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd -P)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FORCE_EVAL="${FORCE_EVAL:-0}"
SMOKE="${SMOKE:-0}"
LIMIT_TRAIN="${LIMIT_TRAIN:-8}"
LIMIT_VAL="${LIMIT_VAL:-4}"
EPOCHS="${EPOCHS:-2}"
if [[ -z "${RUN_ROOT:-}" ]]; then
  if [[ "$SMOKE" == "1" ]]; then
    RUN_ROOT="runs/a_v5_1_matrix/debug/smoke"
  else
    RUN_ROOT="runs/a_v5_1_matrix"
  fi
fi
RUN_PARENT="${RUN_ROOT%/*}"
if [[ "$RUN_PARENT" == "$RUN_ROOT" ]]; then
  RUN_PARENT="."
fi
RUN_NAME="${RUN_ROOT##*/}"

if [[ "$SEED" != "42" ]]; then
  echo "[A-v5.1-matrix] ERROR: formal matrix identity requires SEED=42" >&2
  exit 2
fi
if [[ "$SMOKE" != "0" && "$SMOKE" != "1" ]]; then
  echo "[A-v5.1-matrix] ERROR: SMOKE must be 0 or 1" >&2
  exit 2
fi
if [[ "$FORCE_EVAL" != "0" && "$FORCE_EVAL" != "1" ]]; then
  echo "[A-v5.1-matrix] ERROR: FORCE_EVAL must be 0 or 1" >&2
  exit 2
fi
if [[ "$SMOKE" == "1" && "$RUN_ROOT" != *debug* ]]; then
  echo "[A-v5.1-matrix] ERROR: smoke RUN_ROOT must contain a debug path component" >&2
  exit 2
fi

declare -a DATASETS=("irstd1k" "sirst_uavb" "sirst4")
declare -A CONFIGS=(
  [irstd1k]="configs/a_v5_1_irstd1k.py"
  [sirst_uavb]="configs/a_v5_1_sirst_uavb.py"
  [sirst4]="configs/a_v5_1_sirst4.py"
)
declare -A RUN_NAMES=(
  [irstd1k]="train_irstd1k_seed42"
  [sirst_uavb]="train_sirst_uavb_seed42"
  [sirst4]="train_sirst4_seed42"
)

echo "[A-v5.1-matrix] preflight python=$PYTHON_BIN device=$DEVICE run_root=$RUN_ROOT"
"$PYTHON_BIN" - "$DEVICE" <<'PY'
import sys
import torch
from irstd_g0.data import DatasetSpec, build_dataset

device = sys.argv[1]
if device.startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
expected = {
    "irstd1k": ("data/IRSTD-1K", 800, 201),
    "sirst_uavb": ("data/SIRST-UAVB_OnlyUAV_Form", 2400, 600),
    "sirst4": ("data/SIRST4-ForLiTE", 2285, 1067),
}
for name, (root, train_count, test_count) in expected.items():
    train = build_dataset(DatasetSpec(name, root, "train", False))
    test = build_dataset(DatasetSpec(name, root, "test", False))
    actual = (len(train), len(test))
    wanted = (train_count, test_count)
    if actual != wanted:
        raise SystemExit(f"{name} split counts mismatch: expected {wanted}, got {actual}")
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
PY

"$PYTHON_BIN" -m pytest -q \
  tests/test_a_v5_1.py \
  tests/test_a_matrix_data.py \
  tests/test_a_matrix_summary.py \
  tests/test_eval_a.py \
  tests/test_train_a.py \
  tests/test_run_a_v5_1_matrix.py
"$PYTHON_BIN" -m compileall -q \
  irstd_a irstd_g0 train_a.py eval_a.py summarize_a_matrix.py configs

validate_checkpoint() {
  local checkpoint="$1"
  local dataset="$2"
  local config_path="$3"
  "$PYTHON_BIN" - "$checkpoint" "$dataset" "$SEED" "$config_path" "$SMOKE" "$LIMIT_TRAIN" "$LIMIT_VAL" <<'PY'
import sys
import torch
from train_a import build_training_datasets, load_config

path, expected_dataset, expected_seed, config_path = sys.argv[1:5]
expected_seed = int(expected_seed)
smoke, limit_train, limit_val = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
config = checkpoint.get("config", {})
expected_config = load_config(config_path)
if config != expected_config:
    raise SystemExit(f"checkpoint config mismatch in {path}")
actual_dataset = checkpoint.get("dataset_name", config.get("data", {}).get("name"))
actual_objective = checkpoint.get("objective_version", config.get("loss", {}).get("objective_version"))
actual_seed = checkpoint.get("seed", config.get("run", {}).get("seed"))
if actual_dataset != expected_dataset:
    raise SystemExit(f"checkpoint dataset mismatch in {path}: {actual_dataset!r}")
if actual_objective != "v5.1-ur":
    raise SystemExit(f"checkpoint objective mismatch in {path}: {actual_objective!r}")
if int(actual_seed) != expected_seed:
    raise SystemExit(f"checkpoint seed mismatch in {path}: {actual_seed!r}")
_, _, train_ids, val_ids = build_training_datasets(
    expected_config["data"], expected_seed,
    limit_train if smoke else 0, limit_val if smoke else 0,
)
if checkpoint.get("train_ids") != train_ids or checkpoint.get("val_ids") != val_ids:
    raise SystemExit(f"checkpoint split mismatch in {path}")
PY
}

validate_completion_marker() {
  local marker="$1"
  local checkpoint="$2"
  local dataset="$3"
  "$PYTHON_BIN" - "$marker" "$checkpoint" "$dataset" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

marker_path, checkpoint_path = map(Path, sys.argv[1:3])
dataset = sys.argv[3]
payload = json.loads(marker_path.read_text(encoding="utf-8"))
actual_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
expected = {
    "experiment": "a_v5_1_cross_dataset_matrix",
    "source_dataset": dataset,
    "objective_version": "v5.1-ur",
    "seed": 42,
    "checkpoint_sha256": actual_hash,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"completion marker mismatch for {key}: {marker_path}")
PY
}

write_completion_marker() {
  local run_dir="$1"
  local dataset="$2"
  "$PYTHON_BIN" - "$run_dir" "$dataset" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

run_dir, dataset = Path(sys.argv[1]), sys.argv[2]
checkpoint = run_dir / "a_best.pt"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
payload = {
    "experiment": "a_v5_1_cross_dataset_matrix",
    "source_dataset": dataset,
    "objective_version": "v5.1-ur",
    "seed": 42,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": digest,
}
handle, temporary = tempfile.mkstemp(prefix=".formal_complete.", suffix=".tmp", dir=run_dir)
try:
    with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, run_dir / "formal_complete.json")
finally:
    Path(temporary).unlink(missing_ok=True)
PY
}

run_train() {
  local dataset="$1"
  local config="${CONFIGS[$dataset]}"
  local run_dir="$RUN_ROOT/${RUN_NAMES[$dataset]}"
  local marker="$run_dir/formal_complete.json"
  local best="$run_dir/a_best.pt"
  local last="$run_dir/a_last.pt"
  local console_log="${run_dir}_console.log"

  mkdir -p "$RUN_ROOT"
  if [[ -f "$marker" && -f "$best" ]]; then
    validate_checkpoint "$best" "$dataset" "$config"
    validate_completion_marker "$marker" "$best" "$dataset"
    echo "[A-v5.1-matrix] skip completed training: $dataset"
    return
  fi

  local -a command=(
    "$PYTHON_BIN" train_a.py
    --config "$config"
    --run-dir "$run_dir"
    --device "$DEVICE"
    --seed "$SEED"
    --num-workers "$NUM_WORKERS"
  )
  if [[ "$SMOKE" == "1" ]]; then
    command+=(
      --debug
      --limit-train "$LIMIT_TRAIN"
      --limit-val "$LIMIT_VAL"
      --epochs "$EPOCHS"
      --batch-size 1
      --num-workers 0
    )
  fi
  if [[ -f "$last" ]]; then
    validate_checkpoint "$last" "$dataset" "$config"
    command+=(--resume "$last")
    echo "[A-v5.1-matrix] resume training: $dataset"
  elif [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "[A-v5.1-matrix] ERROR: non-empty non-resumable run: $run_dir" >&2
    exit 3
  else
    echo "[A-v5.1-matrix] start training: $dataset"
  fi
  "${command[@]}" 2>&1 | tee -a "$console_log"
  if [[ ! -f "$best" ]]; then
    echo "[A-v5.1-matrix] ERROR: training produced no best checkpoint: $best" >&2
    exit 4
  fi
  validate_checkpoint "$best" "$dataset" "$config"
  write_completion_marker "$run_dir" "$dataset"
}

metrics_are_complete() {
  local metrics="$1"
  local dataset="$2"
  local checkpoint="$3"
  "$PYTHON_BIN" - "$metrics" "$dataset" "$checkpoint" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path, source, checkpoint = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("source_dataset") != source or payload.get("objective_version") != "v5.1-ur":
    raise SystemExit(1)
if payload.get("checkpoint_sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
    raise SystemExit(f"checkpoint hash mismatch in {path}")
required = {"irstd1k", "sirst_uavb", "sirst4_all", "sirst4_xdu", "sirst4_non_xdu"}
if not required.issubset(payload.get("datasets", {})):
    raise SystemExit(1)
PY
}

run_eval() {
  local dataset="$1"
  local run_dir="$RUN_ROOT/${RUN_NAMES[$dataset]}"
  local checkpoint="$run_dir/a_best.pt"
  local metrics="$run_dir/metrics.json"
  if [[ "$FORCE_EVAL" == "0" ]] && metrics_are_complete "$metrics" "$dataset" "$checkpoint"; then
    echo "[A-v5.1-matrix] skip completed evaluation: $dataset"
    return
  fi
  local -a command=(
    "$PYTHON_BIN" eval_a.py
    --checkpoint "$checkpoint"
    --run-dir "$run_dir"
    --datasets irstd1k sirst_uavb sirst4
    --device "$DEVICE"
    --probe-count 8
  )
  if [[ "$SMOKE" == "1" ]]; then
    command+=(--limit 2 --probe-count 1)
  fi
  echo "[A-v5.1-matrix] evaluate: $dataset"
  "${command[@]}" 2>&1 | tee -a "${run_dir}_eval_console.log"
  metrics_are_complete "$metrics" "$dataset" "$checkpoint"
}

for dataset in "${DATASETS[@]}"; do
  run_train "$dataset"
done
for dataset in "${DATASETS[@]}"; do
  run_eval "$dataset"
done

"$PYTHON_BIN" summarize_a_matrix.py --run-root "$RUN_ROOT"
(
  cd "$RUN_ROOT"
  find . -type f ! -name 'matrix_manifest.sha256' -print0 \
    | sort -z \
    | xargs -0 sha256sum > matrix_manifest.sha256
)
tar -czf "${RUN_ROOT}.tar.gz" -C "$RUN_PARENT" "$RUN_NAME"

echo "[A-v5.1-matrix] complete summary=$RUN_ROOT/matrix_summary.json archive=${RUN_ROOT}.tar.gz"
