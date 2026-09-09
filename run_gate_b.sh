#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-}"
DATA_ROOT="${DATA_ROOT:-data/SIRST4-ForLiTE}"
SEED="${SEED:-42}"
RUN_DIR="${RUN_DIR:-runs/gaussamr_gate_b_seed${SEED}}"
EPOCHS="${EPOCHS:-200}"
MAX_STEPS="${MAX_STEPS:-3200}"
LR="${LR:-1e-3}"
SUBSET_SIZE="${SUBSET_SIZE:-16}"
SKIP_TESTS="${SKIP_TESTS:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"

mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/gate_b.log") 2>&1

if [[ -z "$DEVICE" ]]; then
  DEVICE="$("$PYTHON" - <<'EOF'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
EOF
)"
fi

cleanup() {
  EXIT_CODE=$?
  set +Eeuo pipefail
  trap - EXIT
  cp "$SCRIPT_DIR/run_gate_b.sh" "$RUN_DIR/run_gate_b.sh"
  "$PYTHON" -m pip freeze > "$RUN_DIR/environment.txt"
  git rev-parse HEAD > "$RUN_DIR/git_revision.txt"
  git status --short > "$RUN_DIR/git_status.txt"
  # ponytail: find|sort|xargs instead of a hashing loop; GNU-only -z/-0 is fine on the Linux remote
  (
    cd "$RUN_DIR" &&
    find . -type f ! -name sha256sums.txt -print0 |
      LC_ALL=C sort -z |
      xargs -0 -r sha256sum > sha256sums.txt
  )
  (
    cd "$(dirname "$RUN_DIR")" &&
    tar -czf "$(basename "$RUN_DIR").tar.gz.tmp" "$(basename "$RUN_DIR")" &&
    mv -f "$(basename "$RUN_DIR").tar.gz.tmp" "$(basename "$RUN_DIR").tar.gz"
  )
  exit "$EXIT_CODE"
}
trap cleanup EXIT

echo "=== Gate B run starting: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "PYTHON=$PYTHON DEVICE=$DEVICE DATA_ROOT=$DATA_ROOT RUN_DIR=$RUN_DIR"
echo "SEED=$SEED EPOCHS=$EPOCHS MAX_STEPS=$MAX_STEPS LR=$LR SUBSET_SIZE=$SUBSET_SIZE"

if [[ "$SKIP_TESTS" != "1" ]]; then
  "$PYTHON" -m unittest discover -s tests -v
fi

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --run-dir "$RUN_DIR"
  --seed "$SEED"
  --subset-size "$SUBSET_SIZE"
  --epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --lr "$LR"
  --device "$DEVICE"
)

if [[ "$SMOKE_TEST" == "1" ]]; then
  "$PYTHON" train_gaussamr_gate_b.py "${COMMON_ARGS[@]}" --smoke-test
else
  "$PYTHON" train_gaussamr_gate_b.py "${COMMON_ARGS[@]}"
  "$PYTHON" train_gaussamr_gate_b.py \
    --data-root "$DATA_ROOT" \
    --subset-size "$SUBSET_SIZE" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --verify-only \
    --checkpoint "$RUN_DIR/gaussamr_v1_gate_b.pt"
fi

echo "=== Gate B run finished: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
