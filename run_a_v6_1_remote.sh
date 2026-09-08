#!/usr/bin/env bash
set -Eeuo pipefail

# Stage-A v6.1 corrected: test -> train/resume -> evaluate -> gate -> package.
# Run from the new_model_v3 repository root on a Linux CUDA machine.

RUN_DIR="${RUN_DIR:-runs/a_psf/irstd1k_seed42_v6_1_corrected}"
CONFIG="${CONFIG:-configs/a_psf_irstd1k.py}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"

CONSOLE_LOG="${RUN_DIR}_console.log"
TEST_REPORT="${RUN_DIR}_pytest.xml"
ARCHIVE="${RUN_DIR}.tar.gz"

export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$RUN_DIR")"

# Capture environment checks, tests, training, evaluation and packaging in a
# single log outside RUN_DIR. It is copied into RUN_DIR before archiving.
exec > >(tee -a "$CONSOLE_LOG") 2>&1

echo "[A-v6.1] run_dir=$RUN_DIR config=$CONFIG device=$DEVICE seed=$SEED"

"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to start the paid training run")
print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"gpu={torch.cuda.get_device_name(0)}"
)
PY

echo "[A-v6.1] running tests"
"$PYTHON" -m pytest -q tests --junitxml="$TEST_REPORT"
"$PYTHON" -m compileall -q irstd_a train_a.py eval_a.py visualize_a.py "$CONFIG"

TRAIN_ARGS=(
  --config "$CONFIG"
  --run-dir "$RUN_DIR"
  --device "$DEVICE"
  --seed "$SEED"
)

if [[ -f "$RUN_DIR/a_last.pt" ]]; then
  echo "[A-v6.1] resuming from $RUN_DIR/a_last.pt"
  TRAIN_ARGS+=(--resume "$RUN_DIR/a_last.pt")
elif [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty run without a_last.pt: $RUN_DIR" >&2
  exit 2
else
  echo "[A-v6.1] starting a fresh seed-$SEED run"
fi

"$PYTHON" train_a.py "${TRAIN_ARGS[@]}"

echo "[A-v6.1] evaluating all datasets"
"$PYTHON" eval_a.py \
  --checkpoint "$RUN_DIR/a_best.pt" \
  --run-dir "$RUN_DIR" \
  --datasets irstd1k sirst_uavb sirst4 \
  --probe-count 8 \
  --device "$DEVICE"

echo "[A-v6.1] rendering diagnostics"
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

# Fails closed: no archive is produced when any scientific gate fails or a
# required measurement is missing.
"$PYTHON" -m irstd_a.acceptance --run-dir "$RUN_DIR"
cp "$CONSOLE_LOG" "$RUN_DIR/console.log"

echo "[A-v6.1] packaging accepted run"
tar -czf "$ARCHIVE" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"
test -s "$ARCHIVE"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"

echo "[A-v6.1] complete: $RUN_DIR"
echo "[A-v6.1] archive: $ARCHIVE"
echo "[A-v6.1] checksum: ${ARCHIVE}.sha256"
