# GaussAMR V1 Gate B Design

**Date:** 2026-09-09

**Status:** approved for specification review

**Parent design:** `2026-09-08-gaussamr-v1-design.md`, Section 12, Gate B

## 1. Objective

Implement and run a deterministic optimization-sanity gate for the packaged
GaussAMR V1 path. Gate B must prove that the existing architecture and enabled
losses can overfit one fixed 16-image SIRST4 training subset without changing
the architecture or using validation/test images for model selection.

The gate passes only when all of the following are true on the same unaugmented
16 training images:

- every enabled loss and gradient is finite for positive, multi-target, and
  empty-mask cases;
- full composed-mask nIoU is at least `0.90`;
- router coverage@24 is exactly `1.00`;
- router coverage@8 is at least `0.95`.

## 2. Selected approach

Use a deterministic two-stage overfit from zero initialization:

1. Train the router and Gaussian composition until both routing coverage gates
   pass.
2. Freeze that router, initialize a fresh DetailRefiner, and train the local and
   full-mask residual losses until full-mask nIoU passes.

This follows the already validated staged implementation, isolates routing and
segmentation failures, and avoids unstable joint optimization through discrete
top-k selection. An optional resume checkpoint is supported for interrupted
remote runs, but the default Gate B result starts from zero initialization.

Rejected alternatives are direct fine-tuning of the existing Gate C checkpoint,
which is a weaker test of learnability, and joint router/detail training, which
expands the experiment beyond the smallest Gate B sanity check.

## 3. Deterministic data contract

- Dataset: `SIRST4Dataset(data_root, split="train")`.
- Subset: the first 16 IDs produced by existing `fixed_subset(ids, 16, seed=42)`.
- Training and evaluation use exactly those IDs.
- Training uses the existing paired horizontal flip, vertical flip, and 90-degree
  rotation augmentation with a seeded generator.
- Gate evaluation uses the unaugmented images in stable ID order.
- Batch size remains 1 so variable native image sizes are preserved.
- The script records the exact selected IDs in `gate_b_summary.json`.

No test-split image participates in Gate B.

## 4. Training stages

### 4.1 Router stage

- Initialize `GaussianFeatureBank` and `GaussianRouter` from a fresh default
  random initialization; the fixed bank remains non-trainable.
- Train the existing router focal, center, sigma, and coverage losses plus the
  Gaussian full-mask BCE/Dice loss.
- Evaluate coverage@8 and coverage@24 after every epoch.
- Stop as soon as coverage@24 is `1.00` and coverage@8 is at least `0.95`.
- Save `router_best.pt` and `router_last.pt`.

### 4.2 Detail stage

- Load `router_best.pt` and freeze the router.
- Initialize a fresh `DetailRefiner` unless an explicit resume checkpoint is
  supplied.
- Select K2=8 proposals with the existing probability/uncertainty priority.
- Train `2 * local BCE/Dice + full composed-mask BCE/Dice`, using strict
  Mahalanobis-radius-3 residual truncation.
- Evaluate the packaged full output after every epoch.
- Stop as soon as nIoU is at least `0.90`, provided both frozen-router coverage
  gates still pass.
- Save `detail_best.pt`, `detail_last.pt`, and a consolidated
  `gaussamr_v1_gate_b.pt` loadable by `GaussAMRV1.from_probe_checkpoint`.

Both stages default to at most 200 epochs and 3,200 optimizer steps. These are
safety ceilings rather than targets. Failure to reach the thresholds produces a
failed summary and a nonzero process status; the runner does not silently add
epochs or change hyperparameters.

## 5. Finite-loss validation

Expose the Gate B training-loss computation as testable Python functions rather
than duplicating it inside tests. Synthetic single-target, multi-target, and
empty-mask inputs exercise every loss enabled in the router and detail stages.
Tests assert finite scalar losses, finite input gradients where applicable, and
finite gradients for every trainable parameter participating in that stage.

Any non-finite runtime loss or gradient aborts immediately with the stage,
epoch, step, and sample ID in the error message.

## 6. Components and files

- `irstd_gaussamr/gate_b.py`: subset construction, stage losses, evaluation,
  gate decisions, checkpointing, and summary helpers.
- `train_gaussamr_gate_b.py`: command-line entry point for the deterministic
  two-stage run and resume support.
- `tests/test_gaussamr_gate_b.py`: finite-loss cases, deterministic subset,
  early-stop decisions, one-step smoke run, and failed-gate exit behavior.
- `run_gate_b.sh`: remote-server orchestration, test execution, training,
  independent final verification, provenance capture, and archive creation.
- `docs/superpowers/specs/2026-09-08-gaussamr-v1-design.md`: status and measured
  Gate B result after the real run completes.

Existing Gate A and Gate C artifacts are not overwritten.

## 7. Command-line and remote execution contract

`train_gaussamr_gate_b.py` accepts at least:

- `--data-root`;
- `--run-dir`;
- `--seed` and `--subset-size`;
- `--epochs` and `--max-steps`;
- `--lr`;
- `--device`;
- optional router/detail resume checkpoints.

`run_gate_b.sh` uses `set -Eeuo pipefail`, resolves the repository root from the
script location, and supports environment overrides including `PYTHON`,
`DEVICE`, `DATA_ROOT`, `RUN_DIR`, `SEED`, `EPOCHS`, `MAX_STEPS`, and `LR`.
When `DEVICE` is unset it chooses CUDA only if the selected Python environment
reports CUDA availability; otherwise it uses CPU.

The runner writes commands and output through `tee` to `gate_b.log`. It runs the
Gate B unit/integration tests before training and performs a final fresh
checkpoint evaluation after training. A failed threshold returns a nonzero exit
status suitable for remote job schedulers.

## 8. Outputs and packaging

The run directory contains:

```text
gate_b.log
gate_b_summary.json
selected_ids.json
router_best.pt
router_last.pt
detail_best.pt
detail_last.pt
gaussamr_v1_gate_b.pt
environment.txt
git_revision.txt
git_status.txt
sha256sums.txt
run_gate_b.sh
```

On shell exit, a packaging function computes SHA-256 checksums for available
result files and creates `<run-dir>.tar.gz` beside the run directory. Packaging
is attempted for both passing and failed runs, so partial logs and checkpoints
can be retrieved for diagnosis. The archive never contains itself.

The summary records configuration, selected IDs, per-stage best metrics,
thresholds, pass/fail booleans, stopping reasons, and final overall Gate B
status. The consolidated checkpoint contains router and detail state dicts plus
the same configuration and final metrics.

## 9. Verification and completion

Before Gate B is reported complete:

1. the complete unit/integration suite must pass;
2. a one-step temporary-data smoke run must create all expected outputs;
3. the consolidated checkpoint must load through `GaussAMRV1` and preserve the
   input image size;
4. `run_gate_b.sh` must pass `bash -n`;
5. the real fixed-16 run must produce `gate_b_summary.json` and the tar archive;
6. completion is reported as a pass only if all numerical thresholds are met.

If the current local CPU environment is too slow for the real run, implementation
and smoke verification may complete locally, but the numerical Gate B status
remains pending until the provided script finishes on the remote server.
