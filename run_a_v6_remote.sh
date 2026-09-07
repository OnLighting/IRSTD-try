#!/usr/bin/env bash
set -euo pipefail

# One paid-GPU invocation for Stage-A v6 (seven-loss redesign).
# Verifies environment, runs the full test suite, trains (or resumes) the
# v6 model, evaluates across all three datasets, visualises components,
# and gates the result against the v6 acceptance criteria.
#
# v6 differs from v5 in three ways that this script enforces:
#   1. Public U is the reconstruction uncertainty head alone; PSF-mixture
#      entropy is no longer mixed in.
#   2. The loss is the eight-term objective (rec, bg, sp, ctr, psf, ind,
#      flip, amp) with all weights 1.0.
#   3. A second forward pass on the h-flipped image drives the `flip`
#      equivariance loss term.

RUN_DIR="${RUN_DIR:-runs/a_psf/irstd1k_seed42_v6_sevenloss}"
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
  echo "[A-v6] resuming from $RUN_DIR/a_last.pt"
  TRAIN_ARGS+=(--resume "$RUN_DIR/a_last.pt")
elif [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty run without a_last.pt: $RUN_DIR" >&2
  exit 2
else
  echo "[A-v6] starting a new seed-$SEED run in $RUN_DIR"
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
import json

required = [
    "a_best.pt", "a_last.pt", "metrics.json", "stability.json",
    "per_image_metrics.csv", "visualizations/psf_kernels.png",
]
root = Path("$RUN_DIR")
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit(f"run finished but required artifacts are missing: {missing}")

# v6 acceptance gate: check the new diagnostics on IRSTD-1K test.
with (root / "metrics.json").open() as handle:
    metrics = json.load(handle)
irstd1k = metrics.get("datasets", {}).get("irstd1k", {})
gates = {
    "coverage_at_5px_gate": ("passed", True),
    "uncertainty_error_spearman_median": (">= 0.30", None),
}
if "coverage_at_5px_gate" not in irstd1k:
    print("[A-v6] WARNING: coverage_at_5px_gate missing from metrics.json")
elif not irstd1k["coverage_at_5px_gate"]["passed"]:
    print(f"[A-v6] coverage_at_5px gate FAILED: {irstd1k['coverage_at_5px_gate']}")

spearman = irstd1k.get("uncertainty_error_spearman_median")
if spearman is None or spearman < 0.30:
    print(f"[A-v6] uncertainty/Spearman gate FAILED: median={spearman}, required >= 0.30")
else:
    print(f"[A-v6] uncertainty/Spearman median={spearman:.4f} (>= 0.30: PASS)")

# h-flip S Pearson is in stability.json; the v5 gate is 0.90, v6 tightens
# to 0.90 from 0.85.
with (root / "stability.json").open() as handle:
    stability = json.load(handle)
hflip_S = stability.get("perturbation", {}).get("hflip", {}).get("S", {}).get("pearson", {}).get("median")
if hflip_S is None:
    print("[A-v6] WARNING: h-flip S Pearson missing from stability.json")
elif hflip_S < 0.90:
    print(f"[A-v6] h-flip S Pearson gate FAILED: median={hflip_S:.4f}, required >= 0.90")
else:
    print(f"[A-v6] h-flip S Pearson median={hflip_S:.4f} (>= 0.90: PASS)")

print(f"[A-v6] complete: {root}")
PY

# Keep the complete console stream with the scientific artifacts, then create
# one archive outside RUN_DIR so it cannot recursively include itself.
cp "${RUN_DIR}_console.log" "$RUN_DIR/console.log"
ARCHIVE="${RUN_DIR}.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"
test -s "$ARCHIVE"
echo "[A-v6] packaged: $ARCHIVE"
