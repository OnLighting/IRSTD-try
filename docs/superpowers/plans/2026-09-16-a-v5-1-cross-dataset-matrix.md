# Stage-A v5.1 Cross-Dataset Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible Linux CUDA runner that trains historical Stage-A v5.1 separately on IRSTD-1K, SIRST-UAVB, and SIRST4 and produces a complete 3x3 evaluation matrix.

**Architecture:** Keep the current v5.2a path intact while adding explicit objective-version dispatch for the recovered v5.1 model, targets, loss, and checkpoint score. Generalize the existing dataset factory and Stage-A trainer, then add a pure summary module and a guarded shell orchestrator that owns the three formal run directories.

**Tech Stack:** Python 3.12+, PyTorch, NumPy, SciPy, Pillow, pytest, Bash, existing `irstd_a` and `irstd_g0` packages.

**Spec:** `docs/superpowers/specs/2026-09-16-a-v5-1-cross-dataset-matrix-design.md`

## Global Constraints

- The formal objective identity is exactly `v5.1-ur`; never relabel v5.2a output as v5.1.
- Seed is 42 for model initialization, data order, and deterministic validation selection.
- Fixed train/validation counts are 720/80, 2160/240, and 2056/229 for IRSTD-1K, SIRST-UAVB, and SIRST4.
- Official test sets never select checkpoints.
- Existing artifacts below `runs/a_psf/`, `runs/g0/`, and `runs/gaussamr_*` are read-only.
- The deferred `.worktrees/codex-g1-i-background-v5` checkout is not modified.
- Formal outputs live only below `runs/a_v5_1_matrix/` plus the sibling archive `runs/a_v5_1_matrix.tar.gz`.
- SIRST4 reports `sirst4_all` (1067), `sirst4_xdu` (201), and `sirst4_non_xdu` (866); the official matrix cell is `sirst4_all`.
- The public runner must stop on provenance mismatch and must not delete an existing run.

---

### Task 1: Recover and lock the historical v5.1 objective

**Files:**
- Create: `irstd_a/objectives.py`
- Modify: `irstd_a/model.py`
- Modify: `irstd_a/targets.py`
- Modify: `irstd_a/losses.py`
- Modify: `train_a.py`
- Modify: `eval_a.py`
- Create: `tests/test_a_v5_1.py`
- Modify: `tests/test_a_model.py`
- Modify: `tests/test_a_targets.py`
- Modify: `tests/test_a_losses.py`
- Modify: `tests/test_train_a.py`

**Interfaces:**
- Produces: `V5_1_OBJECTIVE = "v5.1-ur"` and `V5_2A_OBJECTIVE = "v5.2a-signed-r"`.
- Produces: `validate_objective_version(value: str) -> str`.
- Extends: `build_a_model(..., objective_version: str = V5_2A_OBJECTIVE) -> APSFUnmixingNet`.
- Extends: `build_weak_targets(..., objective_version: str = V5_2A_OBJECTIVE) -> dict[str, Tensor]`.
- Preserves: `APSFUnmixingLoss(weights, objective_version=...)` with version-specific forward behavior.
- Produces: `validation_score(summary, objective_version=V5_2A_OBJECTIVE) -> float`.

- [ ] **Step 1: Write failing regression tests for v5.1 identity and model behavior**

Add `tests/test_a_v5_1.py` with fixed assertions:

```python
from pathlib import Path

import torch

from irstd_a.model import build_a_model
from irstd_a.objectives import V5_1_OBJECTIVE, V5_2A_OBJECTIVE


def test_v51_uses_nonnegative_residual_and_direct_uncertainty() -> None:
    model = build_a_model(
        dims=(8, 16, 32), num_psf=2, kernel_size=5,
        objective_version=V5_1_OBJECTIVE,
    ).eval()
    with torch.no_grad():
        output = model(torch.rand(1, 1, 32, 32), return_aux=True)
    assert torch.all(output["R"] >= 0)
    torch.testing.assert_close(output["U"], output["u_rec"])


def test_v51_and_v52a_residual_semantics_are_distinct() -> None:
    v51 = build_a_model(dims=(8, 16, 32), num_psf=2, kernel_size=5,
                        objective_version=V5_1_OBJECTIVE)
    v52 = build_a_model(dims=(8, 16, 32), num_psf=2, kernel_size=5,
                        objective_version=V5_2A_OBJECTIVE)
    assert torch.all(v51.residual_head.bias < 0)
    assert torch.count_nonzero(v52.residual_head.bias) == 0


def test_historical_v51_checkpoint_loads_strictly() -> None:
    path = Path("runs/a_psf/irstd1k_seed42_v5_1_ur/a_best.pt")
    if not path.exists():
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_a_model(
        **checkpoint["config"]["model"],
        objective_version=checkpoint["config"]["loss"]["objective_version"],
    )
    model.load_state_dict(checkpoint["model"], strict=True)
```

Also add an inference regression using sample `XDU189`: load the historical
checkpoint when present, run the first IRSTD-1K test sample, and assert `R.min()
>= 0`, `R.max() < 1e-3`, and reconstruction is not identically zero. This
catches accidental tanh loading, which currently drives the legacy residual to
`-1` and the reconstruction to zero.

- [ ] **Step 2: Run the focused tests and record the expected failures**

Run:

```bash
python -m pytest -q tests/test_a_v5_1.py tests/test_a_model.py tests/test_a_targets.py tests/test_a_losses.py tests/test_train_a.py
```

Expected: failures because objective constants and versioned model/target/score
dispatch do not exist, and the current model interprets the v5.1 residual head
with `tanh`.

- [ ] **Step 3: Implement explicit objective validation**

Create `irstd_a/objectives.py`:

```python
from __future__ import annotations

V5_1_OBJECTIVE = "v5.1-ur"
V5_2A_OBJECTIVE = "v5.2a-signed-r"
SUPPORTED_OBJECTIVES = frozenset((V5_1_OBJECTIVE, V5_2A_OBJECTIVE))


def validate_objective_version(value: str) -> str:
    version = str(value)
    if version not in SUPPORTED_OBJECTIVES:
        supported = ", ".join(sorted(SUPPORTED_OBJECTIVES))
        raise ValueError(f"unsupported Stage-A objective {version!r}; expected one of: {supported}")
    return version
```

- [ ] **Step 4: Restore v5.1 model semantics without changing v5.2a**

Extend `APSFUnmixingNet` and `build_a_model` with `objective_version`. Store the
validated value. For v5.1 initialize the residual head with normal weights
(`mean=0`, `std=0.01`) and bias `-6.0`, then use
`torch.sigmoid(self.residual_head(features))`. For v5.2a preserve zero
initialization and `torch.tanh`. Both objectives publish direct supervised
`U=u_rec`, and `uncertainty_head` consumes `features.detach()`.

All train/eval model construction must pass:

```python
objective_version = config["loss"]["objective_version"]
model = build_a_model(**config["model"], objective_version=objective_version)
```

- [ ] **Step 5: Restore v5.1 target and loss branches**

In `build_weak_targets`, always return `center`, `support`, `psf_support`,
`local_background`, `target_proxy`, and `source_proxy`. Only v5.2a computes and
returns `psf_teacher` and `residual_teacher`.

In `APSFUnmixingLoss.forward`, dispatch to focused private methods. The v5.1
target terms are the historical equations:

```python
target_output = psf + residual
target_fit = (
    (target_output.float() - target_proxy.float()).abs() * support.float()
).sum(dim=spatial_dims) / safe_proxy_energy
target_energy = (target_output.float() * support.float()).sum(dim=spatial_dims)
target_energy_error = (target_energy - proxy_energy).abs() / safe_proxy_energy
target_leakage = torch.log1p(
    (target_output.float().abs() * psf_outside.float()).sum(dim=spatial_dims)
    / safe_proxy_energy
)
present_target_loss = target_fit + target_energy_error + 2.0 * target_leakage
empty_target_loss = target_output.float().abs().mean(dim=spatial_dims)
residual_loss = (
    2.0 * _masked_mean(residual.abs(), outside)
    + 0.1 * residual.abs().mean()
    + 0.2 * _edge_aware_tv(residual, image)
)
overlap = (psf.abs() * residual.abs()).sum() / torch.sqrt(
    psf.square().sum() * residual.square().sum()
).clamp_min(1e-6)
independence = overlap + 0.25 * detail_overlap
```

Preserve the current signed teacher-based equations for v5.2a. Keep the common
reconstruction, background, presence, amplitude, sparsity, PSF diversity, and
uncertainty terms shared.

- [ ] **Step 6: Restore the exact v5.1 checkpoint-selection score**

Version `validation_score`. For v5.1 use the historical base score plus the two
repairs proven by the saved v5.1 training log:

```python
score = (
    reconstruction
    + 0.5 * (1.0 - min(precision, 1.0))
    + 0.25 * recall_error
    + 0.5 * source_false
    + 0.5 * (1.0 - min(max(centroid_recall, 0.0), 1.0))
    + 0.5 * background_leakage
    + 0.1 * overlap
    + 0.25 * residual_target
    + 0.25 * max(0.0, 0.30 - uncertainty_spearman)
    + 0.1 * max(0.0, 1.0 - residual_meaningful / 0.05)
)
```

Add a regression that loads epochs 46 and 49 from the historical `train.jsonl`
when available and recomputes exactly `0.2625969584949904` and
`0.20267867070305798` within `1e-12`.

- [ ] **Step 7: Run the objective tests**

Run the Step 2 command. Expected: all selected tests pass and the existing
v5.2a tests remain unchanged.

- [ ] **Step 8: Commit the objective compatibility layer**

```bash
git add irstd_a/objectives.py irstd_a/model.py irstd_a/targets.py irstd_a/losses.py train_a.py eval_a.py tests/test_a_v5_1.py tests/test_a_model.py tests/test_a_targets.py tests/test_a_losses.py tests/test_train_a.py
git commit -m "feat: restore Stage-A v5.1 objective"
```

---

### Task 2: Generalize all three datasets for Stage-A training

**Files:**
- Modify: `irstd_g0/data.py`
- Modify: `train_a.py`
- Create: `configs/a_v5_1_irstd1k.py`
- Create: `configs/a_v5_1_sirst_uavb.py`
- Create: `configs/a_v5_1_sirst4.py`
- Create: `tests/test_a_matrix_data.py`
- Modify: `tests/test_train_a.py`

**Interfaces:**
- Extends: `SIRSTUAVBDataset(root, split="test", augment=False)`.
- Extends: `SIRST4Dataset(root, split="test", augment=False)`.
- Preserves: `build_dataset(DatasetSpec) -> Dataset` for all names and train/eval modes.
- Produces: `build_training_datasets(data_config, seed, limit_train=0, limit_val=0) -> tuple[Dataset, Dataset, list[str], list[str]]`.
- Produces three configs with `data["name"]`, source `data["root"]`, all evaluation roots, and exact validation counts.

- [ ] **Step 1: Write failing loader, split, and routing tests**

Create temporary miniature copies of each dataset layout with asymmetric image
and mask pixels. Patch `np.random.rand`/`np.random.randint` so augmentation is
deterministic and assert image/mask flips remain aligned. Assert:

```python
expected = {
    "irstd1k": (800, 80, 720),
    "sirst_uavb": (2400, 240, 2160),
    "sirst4": (2285, 229, 2056),
}
for config_path, (official_train, val_count, effective_train) in cases:
    config = load_config(config_path)
    train, val, train_ids, val_ids = build_training_datasets(config["data"], seed=42)
    assert len(train_ids) == effective_train
    assert len(val_ids) == val_count
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(train_ids) + len(val_ids) == official_train
```

Add failure cases for unknown names, duplicate split IDs, missing split files,
and missing image/mask pairs.

- [ ] **Step 2: Run tests and observe hard-coded IRSTD failures**

Run:

```bash
python -m pytest -q tests/test_a_matrix_data.py tests/test_train_a.py
```

Expected: SIRST train augmentation and generic Stage-A construction tests fail.

- [ ] **Step 3: Make the canonical dataset factory train-capable**

Add `augment` to both SIRST constructors, set `self.augment = augment and
split == "train"`, and call `_augment(img, msk)` before normalization. Remove
the eval-only assertions in `build_dataset`. Validate IDs once at construction:

```python
def _validate_ids(ids: list[str], split_path: str) -> list[str]:
    if not ids:
        raise ValueError(f"split is empty: {split_path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"split contains duplicate IDs: {split_path}")
    return ids
```

Raise `FileNotFoundError` naming the missing image or mask from `__getitem__`.

- [ ] **Step 4: Generalize Stage-A dataset construction**

Replace the direct `IRSTD1KDataset` import with `DatasetSpec`, `build_dataset`,
and the abstract `Dataset` type. Implement:

```python
def build_training_datasets(data_config, seed, limit_train=0, limit_val=0):
    name = str(data_config["name"])
    root = str(data_config["root"])
    split = str(data_config["train_split"])
    full = build_dataset(DatasetSpec(name=name, root=root, split=split, augment=False))
    train_ids, val_ids = split_ids(
        full.ids,
        val_count=int(data_config["val_count"]),
        seed=int(data_config.get("split_seed", seed)),
    )
    if limit_train:
        train_ids = train_ids[:limit_train]
    if limit_val:
        val_ids = val_ids[:limit_val]
    train = build_dataset(DatasetSpec(name=name, root=root, split=split, augment=True))
    val = build_dataset(DatasetSpec(name=name, root=root, split=split, augment=False))
    train.ids = train_ids
    val.ids = val_ids
    return train, val, train_ids, val_ids
```

Include `dataset_name`, `objective_version`, and `seed` at checkpoint top level
as redundant provenance, and validate them on resume in addition to full config
and split equality.

- [ ] **Step 5: Add the three immutable v5.1 configs**

Each config copies the historical v5.1 settings: model widths `(32, 64, 128)`,
six PSFs, kernel size 15, source flux 16, AdamW `2e-4`, weight decay `1e-4`,
500 warmup steps, 150 epochs, patience 20, batch size 4, and loss weights from
the saved v5.1 snapshot including uncertainty `0.10`.

Use these exact data fields:

```python
# IRSTD-1K
name="irstd1k", root="data/IRSTD-1K", val_count=80
# SIRST-UAVB
name="sirst_uavb", root="data/SIRST-UAVB_OnlyUAV_Form", val_count=240
# SIRST4
name="sirst4", root="data/SIRST4-ForLiTE", val_count=229
```

All configs also expose `irstd1k_root`, `sirst_uavb_root`, and `sirst4_root`
for common evaluation.

- [ ] **Step 6: Run loader/trainer tests**

Run the Step 2 command and then:

```bash
python -m pytest -q tests/test_a_v5_1.py tests/test_a_matrix_data.py tests/test_train_a.py
```

Expected: pass.

- [ ] **Step 7: Commit generic training support**

```bash
git add irstd_g0/data.py train_a.py configs/a_v5_1_irstd1k.py configs/a_v5_1_sirst_uavb.py configs/a_v5_1_sirst4.py tests/test_a_matrix_data.py tests/test_train_a.py
git commit -m "feat: train Stage-A on all three datasets"
```

---

### Task 3: Produce overlap-aware evaluation and the 3x3 summary

**Files:**
- Modify: `eval_a.py`
- Create: `summarize_a_matrix.py`
- Modify: `tests/test_eval_a.py`
- Create: `tests/test_a_matrix_summary.py`

**Interfaces:**
- Extends: `build_eval_datasets(data_config, names)` to return `sirst4_all`, `sirst4_xdu`, and `sirst4_non_xdu`.
- Produces: `build_matrix_summary(run_root: Path) -> dict[str, Any]`.
- Produces CLI: `python summarize_a_matrix.py --run-root runs/a_v5_1_matrix`.

- [ ] **Step 1: Write failing SIRST4 and matrix-summary tests**

Update the routing assertion to:

```python
assert sizes == {
    "irstd1k": 201,
    "sirst_uavb": 600,
    "sirst4_all": 1067,
    "sirst4_xdu": 201,
    "sirst4_non_xdu": 866,
}
```

Create three temporary run directories containing minimal `metrics.json`,
`config_snapshot.json`, and checkpoint hash fields. Assert the summary contains
exactly nine official cells ordered by source
`irstd1k,sirst_uavb,sirst4` and target `irstd1k,sirst_uavb,sirst4_all`, plus
three clean SIRST4 companion rows. Assert duplicate or missing source runs,
wrong objective versions, and missing metrics are errors.

- [ ] **Step 2: Run tests and observe missing aggregate/summary failures**

```bash
python -m pytest -q tests/test_eval_a.py tests/test_a_matrix_summary.py
```

Expected: failures for `sirst4_all` and absent summary module.

- [ ] **Step 3: Preserve SIRST4 aggregate evaluation**

Build one base `SIRST4Dataset`, then independent dataset instances for subsets
so mutating `.ids` cannot alter `sirst4_all`. Keep ordering from `test.txt`.
Add `source_dataset` and `objective_version` from the checkpoint config to the
top-level metrics artifact and every per-image record.

Eval must construct the versioned model and versioned weak targets from the
checkpoint config. It must reject CLI config/checkpoint objective or model
mismatches rather than loading non-strictly.

- [ ] **Step 4: Implement the pure matrix summarizer**

Use constants:

```python
SOURCE_RUNS = {
    "irstd1k": "train_irstd1k_seed42",
    "sirst_uavb": "train_sirst_uavb_seed42",
    "sirst4": "train_sirst4_seed42",
}
OFFICIAL_TARGETS = ("irstd1k", "sirst_uavb", "sirst4_all")
CLEAN_COMPANION = "sirst4_non_xdu"
PRIMARY_METRICS = (
    "target_energy_precision_median",
    "target_contrast_recall_median",
    "centroid_recall_5px_mean",
    "source_false_activation_median",
    "background_target_leakage_median",
    "reconstruction_mae_mean",
    "reconstruction_psnr_mean",
    "uncertainty_error_spearman_median",
    "model_latency_ms_per_image",
    "end_to_end_imgs_per_s",
)
```

`build_matrix_summary` validates source identity, seed, objective, checkpoint
hash, and required targets. Write JSON atomically. Write CSV with stable columns
`train_dataset,test_dataset,subset_kind,n_images,checkpoint_sha256` followed by
`PRIMARY_METRICS`. Concatenate source per-image CSV files while adding or
validating `train_dataset`.

- [ ] **Step 5: Run evaluation/summary tests**

Run the Step 2 command. Expected: pass.

- [ ] **Step 6: Commit matrix reporting**

```bash
git add eval_a.py summarize_a_matrix.py tests/test_eval_a.py tests/test_a_matrix_summary.py
git commit -m "feat: summarize Stage-A generalization matrix"
```

---

### Task 4: Add the guarded Linux CUDA runner

**Files:**
- Create: `run_a_v5_1_matrix.sh`
- Create: `tests/test_run_a_v5_1_matrix.py`
- Modify: `README_A.md`

**Interfaces:**
- Produces: `bash run_a_v5_1_matrix.sh`.
- Accepts environment overrides: `PYTHON_BIN`, `DEVICE`, `SEED`, `RUN_ROOT`, `NUM_WORKERS`, `FORCE_EVAL`, `SMOKE`, `LIMIT_TRAIN`, `LIMIT_VAL`, and `EPOCHS`.
- Produces: completion markers, `matrix_summary.json`, `matrix_summary.csv`, combined per-image CSV, checksum manifest, and archive.

- [ ] **Step 1: Write failing static and failure-path tests**

Tests read the shell source and assert it contains `set -euo pipefail`, resolves
`SCRIPT_DIR`, checks CUDA, checks exact split counts, declares all three config
paths, invokes training and evaluation, uses `a_last.pt` for resume, calls the
summarizer, creates SHA-256 output, and archives without removing the run root.

Add an executable smoke test using a fake `PYTHON_BIN` shim in a temporary
directory. The shim records argv and creates the requested checkpoint/metrics
sentinels. Assert all three train and all three eval invocations occur. Make the
second training invocation fail and assert the script returns that status while
the first run's artifacts remain.

- [ ] **Step 2: Run tests and observe missing-runner failures**

```bash
python -m pytest -q tests/test_run_a_v5_1_matrix.py
bash -n run_a_v5_1_matrix.sh
```

Expected: pytest fails and `bash -n` reports the missing file.

- [ ] **Step 3: Implement preflight and idempotent training functions**

The script header and defaults are:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
RUN_ROOT="${RUN_ROOT:-runs/a_v5_1_matrix}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FORCE_EVAL="${FORCE_EVAL:-0}"
SMOKE="${SMOKE:-0}"
LIMIT_TRAIN="${LIMIT_TRAIN:-8}"
LIMIT_VAL="${LIMIT_VAL:-4}"
EPOCHS="${EPOCHS:-2}"
```

Preflight calls a Python helper block that imports torch, verifies CUDA when
requested, instantiates each official split, and asserts counts
`800/201`, `2400/600`, and `2285/1067`. Do not install or upgrade PyTorch.

Implement `run_train dataset config run_dir` with three states:

- valid `formal_complete.json` and `a_best.pt`: skip;
- `a_last.pt` without completion: invoke `train_a.py --resume`;
- empty/nonexistent run: invoke a fresh training command.

If an existing directory contains artifacts but neither a valid resumable
checkpoint nor matching completion marker, exit with an actionable error.

- [ ] **Step 4: Implement evaluation, summary, hashes, and archive**

For every completed run invoke `eval_a.py` once with
`--datasets irstd1k sirst_uavb sirst4`. Redirect each console through `tee`
to a sibling log without hiding the Python exit status (`pipefail` provides the
guard). Then run `summarize_a_matrix.py`.

Generate the manifest from sorted relative paths below the run root:

```bash
find "$RUN_ROOT" -type f ! -name 'matrix_manifest.sha256' -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$RUN_ROOT/matrix_manifest.sha256"
tar -czf "${RUN_ROOT}.tar.gz" -C "$(dirname "$RUN_ROOT")" "$(basename "$RUN_ROOT")"
```

Formal mode uses the immutable configured epochs. Smoke mode appends `--debug`,
limits, epoch override, batch size 1, and workers 0, and requires a run root
whose path contains `debug`.

- [ ] **Step 5: Document the exact remote invocation and outputs**

Add to `README_A.md`:

```bash
bash run_a_v5_1_matrix.sh
```

Document resume as the same command, smoke as
`SMOKE=1 RUN_ROOT=runs/a_v5_1_matrix/debug/smoke bash
run_a_v5_1_matrix.sh`, environment requirements, expected dataset counts, run
tree, and that Stage A reports decomposition metrics rather than segmentation
IoU.

- [ ] **Step 6: Run runner tests and syntax validation**

Run the Step 2 commands. Expected: pass.

- [ ] **Step 7: Commit the public runner**

```bash
git add run_a_v5_1_matrix.sh tests/test_run_a_v5_1_matrix.py README_A.md
git commit -m "feat: add v5.1 matrix CUDA runner"
```

---

### Task 5: End-to-end verification and handoff

**Files:**
- Modify only if verification exposes a scoped defect in files from Tasks 1-4.

**Interfaces:**
- Verifies the complete public workflow without launching the unavailable formal CUDA training locally.

- [ ] **Step 1: Run the full Python test suite**

Run with the Python installation that contains PyTorch:

```bash
python -m pytest -q
```

Expected: all tests pass. If an unrelated historical test is environment-bound,
record its exact command and traceback; do not weaken it silently.

- [ ] **Step 2: Run compilation and shell syntax checks**

```bash
python -m compileall -q irstd_a irstd_g0 train_a.py eval_a.py summarize_a_matrix.py configs
bash -n run_a_v5_1_matrix.sh
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 3: Run a real CPU smoke matrix**

Use an isolated debug directory:

```bash
SMOKE=1 DEVICE=cpu NUM_WORKERS=0 LIMIT_TRAIN=4 LIMIT_VAL=2 EPOCHS=1 \
RUN_ROOT=runs/a_v5_1_matrix/debug/local_smoke \
bash run_a_v5_1_matrix.sh
```

Expected: three tiny checkpoints, three multi-target metrics files, nine
official matrix cells, SIRST4 companion rows, a checksum manifest, and a tar.gz
archive. Confirm the script did not touch existing `runs/a_psf` timestamps.

- [ ] **Step 4: Verify historical v5.1 inference numerically**

Load `runs/a_psf/irstd1k_seed42_v5_1_ur/a_best.pt`, evaluate `XDU189`, and
compare `B`, `S`, `T_psf`, `R`, and `U` summary statistics to the existing
`per_image_metrics.csv`. Use tolerances `rtol=5e-3`, `atol=1e-6` to allow CPU
and library-version differences. Confirm residual remains nonnegative and the
reconstruction is not zero.

- [ ] **Step 5: Verify repository and deferred worktree integrity**

```bash
git status --short
git -C .worktrees/codex-g1-i-background-v5 status --short
git diff --stat f190e15 -- runs/a_psf runs/g0 irstd_g1 train_g1.py eval_g1.py
```

Expected: only intended task files differ from the plan baseline, the deferred
worktree is clean, and no existing experiment artifacts changed.

- [ ] **Step 6: Commit verification-only fixes, if any**

Stage only the scoped files changed to repair verification, run Steps 1-5
again, then commit:

```bash
git commit -m "test: verify v5.1 matrix workflow"
```

Do not create an empty commit when no fixes were needed.
