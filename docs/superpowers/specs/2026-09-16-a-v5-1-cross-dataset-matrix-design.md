# Stage-A v5.1 Three-Dataset Generalization Matrix

Date: 2026-09-16
Status: Approved design

## Objective

Run the Stage-A `v5.1-ur` decomposition model as three independent training
experiments. Train once on each of IRSTD-1K, SIRST-UAVB, and SIRST4, then
evaluate every best checkpoint on all three official test sets. The six
off-diagonal evaluations are the primary zero-shot generalization results; the
three diagonal evaluations provide the in-domain reference, yielding a complete
3x3 train-to-test matrix.

This experiment concerns Stage A only. It does not connect Stage A to G0, add a
segmentation head, or implement the deferred reduced-information segmentation
experiment.

## Version identity and reproducibility

The experiment must run the historical objective identified in the existing
artifact as `v5.1-ur`. The current main-branch Stage-A source implements the
later `v5.2a-signed-r` objective and must not silently stand in for v5.1.

Implementation will add an explicit, source-controlled v5.1 compatibility path
for the model, weak targets, and loss behavior. The historical IRSTD-1K v5.1
checkpoint, its configuration snapshot, training log, and locally preserved
Python bytecode are recovery and validation evidence. Bytecode is not a remote
runtime dependency: the recovered behavior must be represented by ordinary
Python source and covered by regression tests.

Every v5.1 checkpoint and result artifact must record:

- `objective_version="v5.1-ur"`;
- the complete effective configuration;
- training dataset and seed;
- train/validation sample IDs;
- Git revision and dirty status;
- Python, PyTorch, CUDA, cuDNN, and GPU information;
- SHA-256 digests for checkpoints and final result files.

Checkpoint loading and resume must fail clearly if the objective version,
dataset identity, split IDs, model configuration, or optimizer configuration
does not match the active experiment.

## Dataset protocol

Use the repository's existing per-image min-max normalization and paired random
horizontal flip, vertical flip, and 90-degree rotation. Masks remain weak
training supervision and evaluation annotations; they are never model inputs.

The three official train/test splits are:

| Dataset | Official train | Fixed validation | Effective train | Official test |
| --- | ---: | ---: | ---: | ---: |
| IRSTD-1K | 800 | 80 | 720 | 201 |
| SIRST-UAVB | 2400 | 240 | 2160 | 600 |
| SIRST4 | 2285 | 229 | 2056 | 1067 |

For each dataset, select validation IDs deterministically from the official
training split with seed 42. The official test split must never influence
checkpoint selection or early stopping. The split function must preserve the
existing ID order after deterministic index selection and must reject duplicate
IDs, missing samples, empty partitions, and train/validation overlap.

SIRST-UAVB and SIRST4 must gain the same training augmentation option already
used by IRSTD-1K. Their existing file naming remains unchanged, including the
`_mask` suffix for SIRST-UAVB masks.

## Experiment configurations

Provide one immutable configuration for each training source:

- `configs/a_v5_1_irstd1k.py`;
- `configs/a_v5_1_sirst_uavb.py`;
- `configs/a_v5_1_sirst4.py`.

All three configurations use seed 42 and the historical v5.1 model, loss,
optimizer, augmentation, weak-target, diagnostic, and early-stopping settings.
Only dataset identity, root, fixed validation count, and output directory may
differ. Configuration loading must reject an objective other than
`v5.1-ur` for this matrix runner.

The run directories are:

```text
runs/a_v5_1_matrix/
  train_irstd1k_seed42/
  train_sirst_uavb_seed42/
  train_sirst4_seed42/
```

Each directory owns its checkpoints, console log, structured training log,
split manifests, environment snapshot, per-image evaluation table, aggregate
metrics, stability results, and file hashes. Existing `runs/a_psf/` experiments
must not be changed or overwritten.

## Training behavior

The generalized Stage-A trainer builds its dataset through a named dataset
factory instead of directly constructing `IRSTD1KDataset`. Aside from dataset
selection and explicit v5.1 dispatch, the established training behavior is
preserved:

- AdamW with the historical learning rate and weight decay;
- warmup followed by the existing cosine schedule;
- AMP on CUDA;
- gradient clipping and finite-value checks;
- validation after every epoch;
- the historical v5.1 validation score for best-checkpoint selection;
- early stopping with the historical patience;
- atomic best, last, and milestone checkpoints;
- exact RNG and split restoration when resuming.

The trainer must support lightweight limits and epoch overrides for tests and
smoke runs without changing the saved formal configuration identity.

## Evaluation and matrix reporting

Each best checkpoint is evaluated once across IRSTD-1K, SIRST-UAVB, and SIRST4
with the existing Stage-A diagnostic definitions. Stage A is not a segmentation
model, so this experiment does not report IoU as a primary metric. It reports
the decomposition and target-recovery evidence relevant to v5.1:

- target-energy precision and target-contrast recall;
- centroid recall within five pixels;
- source false activation and non-center mass;
- background target leakage;
- reconstruction MAE and PSNR;
- uncertainty/error Spearman correlation;
- component stability under flip, intensity, and noise perturbations;
- degeneration flags, parameter count, latency, throughput, and peak memory.

For SIRST4, retain the aggregate official-test result and additionally report:

- `sirst4_xdu` for the 201 `XDU*` samples duplicated from IRSTD-1K;
- `sirst4_non_xdu` for the remaining 866 samples.

The official `sirst4_all` result occupies the 3x3 matrix cell. The
`sirst4_non_xdu` result is a mandatory clean-domain companion and must be used
when interpreting IRSTD-1K-to-SIRST4 generalization.

The matrix summarizer writes:

- `matrix_summary.json`, containing complete scalar summaries and provenance;
- `matrix_summary.csv`, one row per train/test pair with the primary metrics;
- `per_image_metrics.csv`, with source and target dataset columns;
- `matrix_manifest.sha256`, covering configurations, checkpoints, logs, and
  summaries.

Rows must distinguish the nine official train/test cells and the two SIRST4
subsets without double-counting subset rows as additional experiments.

## Remote runner

Provide `run_a_v5_1_matrix.sh` as the single public entry point for a Linux
CUDA host. It must use `set -euo pipefail`, resolve the repository root from the
script location, and expose environment overrides for Python executable,
device, seed, run root, worker count, and optional smoke limits.

The script performs, in order:

1. verify Python, CUDA availability, all dataset roots, split files, and sample
   counts;
2. run the focused Stage-A/dataset/matrix tests and Python compilation checks;
3. train the three source models sequentially;
4. resume from `a_last.pt` when a run is incomplete and skip training when a
   valid `a_best.pt` plus completion marker already exists;
5. evaluate each best checkpoint on all three test datasets;
6. build the JSON/CSV matrix summaries and SHA-256 manifest;
7. create `runs/a_v5_1_matrix.tar.gz` without deleting the uncompressed runs.

An explicit force environment flag may rerun evaluation and summary generation,
but the script must never delete or overwrite a non-matching training run. A
configuration or provenance mismatch is a hard error with the conflicting path
shown to the operator.

## Tests

Implementation follows test-driven development. Tests must cover:

1. all three dataset loaders support train augmentation and eval-only loading
   while preserving image/mask alignment and naming conventions;
2. factory construction returns the requested dataset and rejects unknown
   names, missing roots, invalid splits, and duplicate IDs;
3. deterministic split counts are exactly 720/80, 2160/240, and 2056/229 with
   no overlap;
4. v5.1 configuration dispatch uses the recovered historical behavior and is
   observably distinct from v5.2a signed-residual targets/losses;
5. existing IRSTD-1K v5.1 checkpoints load strictly through the compatibility
   path;
6. generalized training uses the configured dataset rather than hard-coding
   IRSTD-1K;
7. resume validation rejects dataset, split, objective, and configuration
   mismatches;
8. SIRST4 aggregate, XDU, and non-XDU partitions contain 1067, 201, and 866
   unique samples respectively;
9. matrix summarization maps all nine train/test cells correctly and preserves
   provenance and SIRST4 clean-subset metrics;
10. a tiny CPU smoke run completes train, resume, evaluation, and summarization
    without writing into existing experiment directories;
11. `bash -n run_a_v5_1_matrix.sh` succeeds and failure paths preserve partial
    artifacts for diagnosis.

## Acceptance criteria

The implementation is ready for remote execution when:

- all new tests are first observed failing and then passing;
- the existing Stage-A, G0, and GaussAMR tests still pass;
- the historical IRSTD-1K v5.1 checkpoint loads and evaluates without state
  mismatch;
- a local CPU smoke matrix produces structurally valid checkpoints and summary
  files;
- the shell script passes syntax validation;
- no existing run artifact or the deferred G1 worktree is modified.

The scientific experiment is complete only after the CUDA runner produces
three formal best checkpoints, all nine official evaluation cells, the SIRST4
clean-subset results, the final summaries, and the checksum manifest.

## Explicit non-goals

- Connecting v5.1, v5.2a, or any Stage-A output to G0.
- Implementing reduced-information segmentation or claiming inference speedup.
- Joint multi-dataset training or mixed-domain batches.
- Hyperparameter tuning per dataset.
- Multiple-seed uncertainty estimation.
- Treating SIRST4's duplicated XDU subset as independent generalization
  evidence.
- Modifying, merging, or deleting `.worktrees/codex-g1-i-background-v5`.
