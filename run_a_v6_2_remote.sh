#!/usr/bin/env bash
set -Eeuo pipefail

# Stage-A v6.2: test -> fresh train/resume -> evaluate -> gate -> package.
# Run from the new_model_v3 repository root on a Linux CUDA machine.

RUN_DIR="${RUN_DIR:-runs/a_psf/irstd1k_seed42_v6_2}"
CONFIG="${CONFIG:-configs/a_psf_irstd1k.py}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"

CONSOLE_LOG="${RUN_DIR}_console.log"
TEST_REPORT="${RUN_DIR}_pytest.xml"
ARCHIVE="${RUN_DIR}.tar.gz"

export PYTHONUNBUFFERED=1
# Let the self-test follow this script when users upload it under another name
# such as run.sh.
export A_V6_RUN_SCRIPT="${BASH_SOURCE[0]}"
mkdir -p "$(dirname "$RUN_DIR")"
exec > >(tee -a "$CONSOLE_LOG") 2>&1

echo "[A-v6.2] run_dir=$RUN_DIR config=$CONFIG device=$DEVICE seed=$SEED"

"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to start the paid training run")
print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"gpu={torch.cuda.get_device_name(0)}"
)
PY

echo "[A-v6.2] running tests"
"$PYTHON" -m pytest -q tests --junitxml="$TEST_REPORT"
"$PYTHON" -m compileall -q irstd_a train_a.py eval_a.py visualize_a.py "$CONFIG"

TRAIN_ARGS=(
  --config "$CONFIG"
  --run-dir "$RUN_DIR"
  --device "$DEVICE"
  --seed "$SEED"
)

if [[ -f "$RUN_DIR/a_last.pt" ]]; then
  echo "[A-v6.2] resuming from $RUN_DIR/a_last.pt"
  TRAIN_ARGS+=(--resume "$RUN_DIR/a_last.pt")
elif [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty run without a_last.pt: $RUN_DIR" >&2
  exit 2
else
  echo "[A-v6.2] starting a fresh seed-$SEED run"
fi

"$PYTHON" train_a.py "${TRAIN_ARGS[@]}"

echo "[A-v6.2] evaluating all datasets"
"$PYTHON" eval_a.py \
  --checkpoint "$RUN_DIR/a_best.pt" \
  --run-dir "$RUN_DIR" \
  --datasets irstd1k sirst_uavb sirst4 \
  --probe-count 8 \
  --device "$DEVICE"

echo "[A-v6.2] rendering diagnostics"
"$PYTHON" visualize_a.py \
  --checkpoint "$RUN_DIR/a_best.pt" \
  --metrics "$RUN_DIR/per_image_metrics.csv" \
  --output-dir "$RUN_DIR/visualizations" \
  --max-panels 12 \
  --device "$DEVICE"

cp "$TEST_REPORT" "$RUN_DIR/test-results.xml"
cp "$CONSOLE_LOG" "$RUN_DIR/console.log"

"$PYTHON" - <<PY
from pathlib import Path

root = Path("$RUN_DIR")
required = [
    "a_best.pt",
    "a_last.pt",
    "metrics.json",
    "stability.json",
    "per_image_metrics.csv",
    "train.jsonl",
    "config_snapshot.json",
    "environment.json",
    "test-results.xml",
    "console.log",
    "visualizations/psf_kernels.png",
]
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit(f"run finished but required artifacts are missing: {missing}")
PY

# Package only scientifically accepted runs.
"$PYTHON" -m irstd_a.acceptance --run-dir "$RUN_DIR"
cp "$CONSOLE_LOG" "$RUN_DIR/console.log"

echo "[A-v6.2] packaging accepted run"
tar -czf "$ARCHIVE" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"
test -s "$ARCHIVE"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"

echo "[A-v6.2] complete: $RUN_DIR"
echo "[A-v6.2] packaged output files:"
echo "  archive:  $(realpath "$ARCHIVE")"
echo "  checksum: $(realpath "${ARCHIVE}.sha256")"
ls -lh "$ARCHIVE" "${ARCHIVE}.sha256"
