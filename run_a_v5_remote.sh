#!/usr/bin/env bash
set -euo pipefail

# One paid-GPU invocation for Stage-A v5: verify, train/resume, evaluate, visualize.
# Run from the new_model_v3 repository root.

RUN_DIR="${RUN_DIR:-runs/a_psf/irstd1k_seed42_v5_dualsource}"
CONFIG="${CONFIG:-configs/a_psf_irstd1k.py}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

export PYTHONUNBUFFERED=1

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to start the paid training run")
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY

python -m pytest -q tests
python -m compileall -q irstd_a train_a.py eval_a.py visualize_a.py "$CONFIG"

mkdir -p "$(dirname "$RUN_DIR")"
TRAIN_ARGS=(
  --config "$CONFIG"
  --run-dir "$RUN_DIR"
  --device "$DEVICE"
  --seed "$SEED"
)

if [[ -f "$RUN_DIR/a_last.pt" ]]; then
  echo "[A-v5] resuming from $RUN_DIR/a_last.pt"
  TRAIN_ARGS+=(--resume "$RUN_DIR/a_last.pt")
elif [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty run without a_last.pt: $RUN_DIR" >&2
  exit 2
else
  echo "[A-v5] starting a new seed-$SEED run in $RUN_DIR"
fi

python train_a.py "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${RUN_DIR}_console.log"

python eval_a.py \
  --checkpoint "$RUN_DIR/a_best.pt" \
  --run-dir "$RUN_DIR" \
  --datasets irstd1k sirst_uavb sirst4 \
  --probe-count 8 --device "$DEVICE"

python visualize_a.py \
  --checkpoint "$RUN_DIR/a_best.pt" \
  --metrics "$RUN_DIR/per_image_metrics.csv" \
  --output-dir "$RUN_DIR/visualizations" \
  --max-panels 12 --device "$DEVICE"

python - <<PY
from pathlib import Path

required = [
    "a_best.pt", "a_last.pt", "metrics.json", "stability.json",
    "per_image_metrics.csv", "visualizations/psf_kernels.png",
]
root = Path("$RUN_DIR")
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit(f"run finished but required artifacts are missing: {missing}")
print(f"[A-v5] complete: {root}")
PY

# Keep the complete console stream with the scientific artifacts, then create
# one archive outside RUN_DIR so it cannot recursively include itself.
cp "${RUN_DIR}_console.log" "$RUN_DIR/console.log"
ARCHIVE="${RUN_DIR}.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"
test -s "$ARCHIVE"
echo "[A-v5] packaged: $ARCHIVE"
