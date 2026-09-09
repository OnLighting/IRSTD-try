# GaussAMR V1 Gate B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic two-stage 16-image overfit gate, a remote runner, and a recoverable archive containing all Gate B outputs.

**Architecture:** A focused `irstd_gaussamr.gate_b` module owns stage losses, evaluation, thresholds, and checkpoint payloads. A Python CLI runs or independently verifies the two stages; a thin Bash runner handles the remote environment, logging, provenance, failure-safe packaging, and scheduler-friendly exit status.

**Tech Stack:** Python 3, PyTorch, SciPy, Pillow, `unittest`, Bash, GNU `tar`, and `sha256sum`.

**Spec:** `docs/superpowers/specs/2026-09-09-gaussamr-gate-b-design.md`

## Global Constraints

- Use only the fixed 16-image SIRST4 training subset selected by `fixed_subset(ids, 16, seed=42)` for both training and Gate B evaluation.
- Keep batch size 1, K1=24, K2=8, strict Mahalanobis-radius-3 residual truncation, and ContextRefiner bypassed.
- Default ceilings are 200 epochs and 3,200 optimizer steps; stop earlier when the current stage passes.
- Gate B passes only with finite enabled losses/gradients, nIoU >= 0.90, coverage@24 == 1.00, and coverage@8 >= 0.95.
- The default run starts from fresh random router/detail initialization; resume checkpoints must be explicit and recorded.
- A threshold failure returns a nonzero status and must not trigger automatic extra epochs or architecture changes.
- `run_gate_b.sh` must package available outputs on both successful and failed exits.

---

### Task 1: Gate B loss, evaluation, and decision core

**Files:**
- Create: `irstd_gaussamr/gate_b.py`
- Create: `tests/test_gaussamr_gate_b.py`

**Interfaces:**
- Consumes: `GaussianFeatureBank`, `GaussianRouter`, `DetailRefiner`, `SparseGaussianComposer`, `BCEDiceLoss`, `MetricAccumulator`, and existing router/refiner utilities.
- Produces: `GateBThresholds`, `router_stage_loss(...)`, `detail_stage_loss(...)`, `evaluate_gate_b(...)`, `router_gate_passes(...)`, `detail_gate_passes(...)`, `scatter_detail_residual(...)`, `seed_everything(...)`, and `build_overfit_loaders(...)`.

- [ ] **Step 1: Write failing deterministic-subset and gate-decision tests**

```python
class GateBDecisionTest(unittest.TestCase):
    def test_requires_all_router_thresholds(self):
        self.assertTrue(router_gate_passes({"coverage_at_24": 1.0, "coverage_at_8": 0.96}))
        self.assertFalse(router_gate_passes({"coverage_at_24": 0.99, "coverage_at_8": 0.96}))

    def test_requires_detail_niou_and_router_thresholds(self):
        passing = {"detail_n_iou": 0.90, "coverage_at_24": 1.0, "coverage_at_8": 0.96}
        self.assertTrue(detail_gate_passes(passing))
        self.assertFalse(detail_gate_passes({**passing, "detail_n_iou": 0.899}))
```

- [ ] **Step 2: Run the decision tests and verify missing-module/import failures**

Run: `python -m unittest tests.test_gaussamr_gate_b.GateBDecisionTest -v`

Expected: FAIL because `irstd_gaussamr.gate_b` does not exist.

- [ ] **Step 3: Implement thresholds and exact gate predicates**

```python
@dataclass(frozen=True)
class GateBThresholds:
    n_iou: float = 0.90
    coverage_at_24: float = 1.00
    coverage_at_8: float = 0.95

def router_gate_passes(metrics, thresholds=GateBThresholds()):
    return metrics["coverage_at_24"] >= thresholds.coverage_at_24 and metrics["coverage_at_8"] >= thresholds.coverage_at_8

def detail_gate_passes(metrics, thresholds=GateBThresholds()):
    return router_gate_passes(metrics, thresholds) and metrics["detail_n_iou"] >= thresholds.n_iou
```

- [ ] **Step 4: Run the decision tests and verify they pass**

Run: `python -m unittest tests.test_gaussamr_gate_b.GateBDecisionTest -v`

Expected: PASS.

- [ ] **Step 5: Write failing positive, multi-target, and empty-mask finite-loss tests**

```python
def _cases():
    empty = torch.zeros(1, 1, 64, 64)
    single = empty.clone(); single[..., 30:33, 30:33] = 1
    multi = single.clone(); multi[..., 8:10, 50:52] = 1
    return {"empty": empty, "single": single, "multi": multi}

def test_router_and_detail_losses_have_finite_gradients_for_gate_b_cases(self):
    for name, mask in _cases().items():
        # Fresh modules per case; call each stage loss, backward, and inspect every
        # trainable parameter gradient for presence and finiteness.
```

- [ ] **Step 6: Run the finite-loss test and verify the missing functions fail**

Run: `python -m unittest tests.test_gaussamr_gate_b.GateBLossTest -v`

Expected: FAIL because stage-loss functions are missing.

- [ ] **Step 7: Implement reusable stage computations**

`router_stage_loss` pads image to stride 8, extracts mask instances, computes
router losses, decodes K1=24, composes Gaussian logits at the original mask
size, and returns named router plus full-mask losses. `detail_stage_loss` runs
the frozen bank/router under `torch.no_grad()`, selects K2=8, computes strict
local residual logits, scatters residuals into K1 slots, composes full logits,
and returns local/full/total losses.

```python
return {
    "total": router_losses["total"] + full_mask,
    **{f"router_{k}": v for k, v in router_losses.items() if k != "total"},
    "full_mask": full_mask,
}
```

- [ ] **Step 8: Implement deterministic loaders and evaluator**

`build_overfit_loaders` creates two `SIRST4Dataset(split="train")` instances,
assigns the identical selected ID list, shuffles only the training loader with a
seeded generator, and returns `(train_loader, eval_loader, selected_ids)`.
`evaluate_gate_b` returns additive coverage diagnostics plus Gaussian and detail
IoU/nIoU/Pd/Fa metrics on the unaugmented evaluation loader.

- [ ] **Step 9: Run Task 1 and existing tests**

Run: `python -m unittest tests.test_gaussamr_gate_b tests.test_gaussamr_model tests.test_gaussamr_router_probe -v`

Expected: PASS with no warnings.

- [ ] **Step 10: Commit Task 1**

```bash
git add irstd_gaussamr/gate_b.py tests/test_gaussamr_gate_b.py
git commit -m "feat: add GaussAMR Gate B core"
```

---

### Task 2: Deterministic two-stage training and verification CLI

**Files:**
- Create: `train_gaussamr_gate_b.py`
- Modify: `tests/test_gaussamr_gate_b.py`

**Interfaces:**
- Consumes: all Task 1 interfaces.
- Produces: `run_gate_b(args) -> dict`, `verify_gate_b_checkpoint(...) -> dict`, CLI `main() -> int`, `gate_b_summary.json`, router/detail best/last checkpoints, `selected_ids.json`, and `gaussamr_v1_gate_b.pt`.

- [ ] **Step 1: Write a failing one-step CLI smoke test on a temporary SIRST4 tree**

The test creates one 64x64 image/mask and a train split, invokes:

```python
subprocess.run([
    sys.executable, "train_gaussamr_gate_b.py",
    "--data-root", str(root), "--run-dir", str(run_dir),
    "--subset-size", "1", "--epochs", "1", "--max-steps", "1",
    "--device", "cpu", "--smoke-test",
], check=False, capture_output=True, text=True)
```

It asserts exit 0, all checkpoint/JSON outputs exist, `overall_pass` is false,
and `mode == "smoke-test"` prevents the relaxed smoke thresholds from being
reported as a real Gate B pass.

- [ ] **Step 2: Run the smoke test and verify the missing CLI fails**

Run: `python -m unittest tests.test_gaussamr_gate_b.GateBCliTest.test_one_step_smoke_writes_complete_outputs -v`

Expected: FAIL because `train_gaussamr_gate_b.py` does not exist.

- [ ] **Step 3: Implement argument validation and deterministic setup**

Required CLI defaults: seed 42, subset size 16, epochs 200, max steps 3200,
learning rate `1e-3`, device `cuda` when explicitly requested, and fresh
initialization when resume paths are absent. Reject missing datasets, nonpositive
numeric values, `--detail-checkpoint` without compatible router weights, and
CUDA selection when CUDA is unavailable.

- [ ] **Step 4: Implement router stage with fresh/resume behavior**

Evaluate before training, save the best lexicographic
`(coverage_at_24, coverage_at_8, gaussian_n_iou)` checkpoint, run finite checks
after every backward pass, and early-stop only through `router_gate_passes`.
Always write `router_last.pt`, including a recorded stopping reason.

- [ ] **Step 5: Implement detail stage and consolidated checkpoint**

Reload `router_best.pt`, freeze router parameters, initialize or resume detail,
save the best nIoU checkpoint, and early-stop only through
`detail_gate_passes`. The consolidated payload must contain keys `router`,
`detail_refiner`, `config`, and `best`, matching
`GaussAMRV1.from_probe_checkpoint`.

- [ ] **Step 6: Implement summary and independent verification mode**

`gate_b_summary.json` contains mode, config, exact selected IDs, thresholds,
both histories, both best metrics, stopping reasons, resume provenance, and
`overall_pass`. `--verify-only --checkpoint PATH` reloads the consolidated
checkpoint, evaluates the deterministic subset from scratch, prints JSON, and
returns 0 only when the real thresholds pass.

- [ ] **Step 7: Run the smoke test and verify all artifacts load**

Run: `python -m unittest tests.test_gaussamr_gate_b.GateBCliTest -v`

Expected: PASS; the test loads `gaussamr_v1_gate_b.pt` through
`GaussAMRV1.from_probe_checkpoint` and checks a 65x70 input returns 65x70 logits.

- [ ] **Step 8: Run the full Python test suite**

Run: `python -m unittest discover -s tests -v`

Expected: PASS with no failures or warnings.

- [ ] **Step 9: Commit Task 2**

```bash
git add train_gaussamr_gate_b.py tests/test_gaussamr_gate_b.py
git commit -m "feat: add deterministic Gate B trainer"
```

---

### Task 3: Remote runner and failure-safe output archive

**Files:**
- Create: `run_gate_b.sh`
- Modify: `tests/test_gaussamr_gate_b.py`

**Interfaces:**
- Consumes: Task 2 CLI and its run-directory outputs.
- Produces: a scheduler-friendly Bash entry point, `gate_b.log`, provenance
  files, `sha256sums.txt`, and `<run-dir>.tar.gz` on both success and failure.

- [ ] **Step 1: Write failing shell contract tests**

Tests run `bash -n run_gate_b.sh` and inspect a real one-step smoke execution
with `SKIP_TESTS=1`, temporary `DATA_ROOT`/`RUN_DIR`, CPU, and smoke mode. They
assert the archive exists beside the run directory and contains
`gate_b.log`, `gate_b_summary.json`, checkpoints, provenance, copied runner,
and `sha256sums.txt` without containing the archive itself.

- [ ] **Step 2: Run shell tests and verify the missing script fails**

Run: `python -m unittest tests.test_gaussamr_gate_b.GateBShellTest -v`

Expected: FAIL because `run_gate_b.sh` does not exist.

- [ ] **Step 3: Implement strict environment and logging setup**

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-data/SIRST4-ForLiTE}"
RUN_DIR="${RUN_DIR:-runs/gaussamr_gate_b_seed${SEED:-42}}"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/gate_b.log") 2>&1
```

Determine `DEVICE` with a short read-only PyTorch availability command only
when it was not explicitly set.

- [ ] **Step 4: Implement an EXIT trap that packages partial or complete runs**

The trap saves its incoming exit code, disables `set -e`, copies the runner,
captures `pip freeze`, Git revision/status, writes deterministic SHA-256 entries
for existing files while excluding `sha256sums.txt`, creates a temporary archive
beside `RUN_DIR`, atomically renames it to `<run-dir>.tar.gz`, and finally exits
with the original status.

- [ ] **Step 5: Implement tests, training, and fresh verification commands**

Unless `SKIP_TESTS=1`, run the complete unittest suite. Invoke the trainer with
all environment-derived values. On a successful real run, invoke
`--verify-only` against `gaussamr_v1_gate_b.pt`. `SMOKE_TEST=1` is accepted only
for automated pipeline testing and is passed through as `--smoke-test`.

- [ ] **Step 6: Run shell syntax and archive tests**

Run: `bash -n run_gate_b.sh`

Run: `python -m unittest tests.test_gaussamr_gate_b.GateBShellTest -v`

Expected: both PASS; failed-child status is preserved after packaging.

- [ ] **Step 7: Run full tests and diff checks**

Run: `python -m unittest discover -s tests -v`

Run: `git diff --check`

Expected: PASS and no content errors.

- [ ] **Step 8: Commit Task 3**

```bash
git add run_gate_b.sh tests/test_gaussamr_gate_b.py
git commit -m "feat: add remote Gate B runner"
```

---

### Task 4: Verification, remote handoff, and status documentation

**Files:**
- Modify after a numerical run: `docs/superpowers/specs/2026-09-08-gaussamr-v1-design.md`
- Verify: all files from Tasks 1-3

**Interfaces:**
- Consumes: `run_gate_b.sh` and the real remote run archive.
- Produces: verified code status locally and, after the remote job completes,
  an evidence-backed Gate B pass/fail record in the parent design.

- [ ] **Step 1: Run a complete local smoke pipeline**

Run with the Python environment containing PyTorch:

```bash
SKIP_TESTS=1 SMOKE_TEST=1 DEVICE=cpu EPOCHS=1 MAX_STEPS=1 \
RUN_DIR=runs/gaussamr_gate_b_smoke ./run_gate_b.sh
```

Expected: exit 0 in smoke mode and `runs/gaussamr_gate_b_smoke.tar.gz` exists.

- [ ] **Step 2: Verify archive integrity and checkpoint loading**

Extract into a temporary directory, run `sha256sum -c sha256sums.txt`, load the
consolidated checkpoint through `GaussAMRV1`, and confirm arbitrary-size output.

- [ ] **Step 3: Run final fresh verification**

Run: `python -m unittest discover -s tests -v`

Run: `python -m compileall -q irstd_gaussamr tests train_gaussamr_gate_b.py`

Run: `bash -n run_gate_b.sh`

Run: `git diff --check`

Expected: all commands exit 0.

- [ ] **Step 4: Give the remote invocation**

```bash
PYTHON=python DEVICE=cuda DATA_ROOT=/path/to/SIRST4-ForLiTE \
RUN_DIR=runs/gaussamr_gate_b_seed42 ./run_gate_b.sh
```

The handoff must state that implementation/smoke verification is complete but
the numerical Gate B result remains pending until the remote summary is read.

- [ ] **Step 5: Record the real result only after evidence exists**

When the remote archive returns, read `gate_b_summary.json`, verify checksums,
and update the parent design with exact best epoch/step, nIoU, coverage@8,
coverage@24, stopping reasons, hardware, and pass/fail outcome. Never infer a
pass from job completion alone.
