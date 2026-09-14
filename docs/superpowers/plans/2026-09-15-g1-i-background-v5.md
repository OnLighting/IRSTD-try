# G1 Frozen A-v5 Background Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a controlled G1 experiment that warm-starts G0, concatenates the raw image with the frozen A-v5 background prediction, and trains/evaluates only the segmentation network.

**Architecture:** A focused `irstd_g1` package owns checkpoint provenance, G0-to-two-channel initialization, and a wrapper that keeps A-v5 frozen. Dedicated G1 train/eval entries reuse G0 data, loss, metrics, and optimization semantics while writing isolated artifacts.

**Tech Stack:** Python, PyTorch, pytest, existing `irstd_a` and `irstd_g0` packages.

**Spec:** `docs/superpowers/specs/2026-09-15-g1-i-background-v5-design.md`

## Global Constraints

- Stage A loads from `runs/a_psf/irstd1k_seed42_v5_dualsource/a_best.pt`, stays in `eval()`, uses `torch.no_grad()`, and is never optimized.
- G1 starts from `runs/g0/g0_best.pt`; the copied image stem is unchanged and the new background stem slice is exactly zero.
- Channel order is exactly `[I, B]`; no mask enters the conditioning or model interfaces.
- Reuse G0 data splits, augmentations, loss, optimizer, schedule, threshold, and target matching.
- Write only below `runs/g1_i_b_v5/` and record SHA-256 for both upstream checkpoints.
- Preserve current GaussAMR changes and existing run artifacts.

---

### Task 1: Frozen background-conditioning wrapper

**Files:**
- Create: `irstd_g1/__init__.py`
- Create: `irstd_g1/model.py`
- Test: `tests/test_g1_model.py`

**Interfaces:**
- Consumes: `APSFUnmixingNet.forward(image) -> dict`, `G0SwinUNet.forward(tensor) -> logits`.
- Produces: `FrozenBackgroundConditionedSegmenter(stage_a, segmenter)`, whose `forward(image: Tensor) -> Tensor` concatenates `[image, B]`.

- [ ] **Step 1: Write failing wrapper tests**

```python
def test_wrapper_keeps_stage_a_frozen_in_train_mode():
    wrapper = FrozenBackgroundConditionedSegmenter(TinyA(), TinySegmenter())
    wrapper.train()
    assert wrapper.training
    assert not wrapper.stage_a.training
    assert all(not p.requires_grad for p in wrapper.stage_a.parameters())

def test_wrapper_uses_image_and_background_only():
    stage_a = TinyA()
    segmenter = RecordingSegmenter()
    image = torch.rand(2, 1, 32, 32, requires_grad=True)
    logits = FrozenBackgroundConditionedSegmenter(stage_a, segmenter)(image)
    assert segmenter.seen.shape == (2, 2, 32, 32)
    torch.testing.assert_close(segmenter.seen[:, :1], image)
    torch.testing.assert_close(segmenter.seen[:, 1:], stage_a.background)
    logits.sum().backward()
    assert all(p.grad is None for p in stage_a.parameters())
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest -q tests/test_g1_model.py`

Expected: collection fails because `irstd_g1.model` does not exist.

- [ ] **Step 3: Implement the minimal wrapper**

```python
class FrozenBackgroundConditionedSegmenter(nn.Module):
    def __init__(self, stage_a: nn.Module, segmenter: nn.Module) -> None:
        super().__init__()
        self.stage_a = stage_a.eval()
        self.stage_a.requires_grad_(False)
        self.segmenter = segmenter

    def train(self, mode: bool = True):
        super().train(mode)
        self.stage_a.eval()
        return self

    def forward(self, image: Tensor) -> Tensor:
        with torch.no_grad():
            background = self.stage_a(image)["B"]
        return self.segmenter(torch.cat((image, background), dim=1))
```

- [ ] **Step 4: Run the wrapper tests and verify GREEN**

Run: `python -m pytest -q tests/test_g1_model.py`

Expected: all Task 1 tests pass.

### Task 2: Strict G0-to-G1 initialization and checkpoint provenance

**Files:**
- Modify: `irstd_g1/model.py`
- Create: `irstd_g1/runtime.py`
- Modify: `tests/test_g1_model.py`
- Test: `tests/test_g1_runtime.py`

**Interfaces:**
- Produces: `initialize_two_channel_segmenter(segmenter, g0_state_dict) -> None`.
- Produces: `load_frozen_stage_a(path, device) -> APSFUnmixingNet`.
- Produces: `sha256_file(path: str | Path) -> str`.
- Produces: `build_g1_from_checkpoints(g0_path, stage_a_path, segmenter_config, device) -> FrozenBackgroundConditionedSegmenter`.

- [ ] **Step 1: Write failing initialization and provenance tests**

```python
def test_two_channel_initialization_is_logit_equivalent():
    torch.manual_seed(7)
    g0 = build_g0_model(in_ch=1, dims=(8, 16, 32), depths=(1, 1, 1), num_heads=(1, 2, 4), window_size=4)
    g1 = build_g0_model(in_ch=2, dims=(8, 16, 32), depths=(1, 1, 1), num_heads=(1, 2, 4), window_size=4)
    initialize_two_channel_segmenter(g1, g0.state_dict())
    image, background = torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32)
    torch.testing.assert_close(g1(torch.cat((image, background), 1)), g0(image))

def test_sha256_file_returns_known_digest(tmp_path):
    path = tmp_path / "value.bin"
    path.write_bytes(b"g1")
    assert sha256_file(path) == hashlib.sha256(b"g1").hexdigest()
```

Also add tests that reject a missing checkpoint, a malformed Stage-A payload,
and any G0 state mismatch other than `stem.conv.weight` input width.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest -q tests/test_g1_model.py tests/test_g1_runtime.py`

Expected: imports fail for the new functions.

- [ ] **Step 3: Implement strict initialization and loaders**

Implement SHA-256 with chunked reads. Load Stage A using the checkpoint's
`config["model"]`, then load `checkpoint["model"]` strictly. For G0 expansion,
require identical key sets, require all non-stem tensor shapes to match, copy
the original stem slice, and fill only the second input slice with zero.

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `python -m pytest -q tests/test_g1_model.py tests/test_g1_runtime.py`

Expected: all Task 1–2 tests pass.

### Task 3: Dedicated G1 configuration and training entry

**Files:**
- Create: `configs/g1_i_b_v5_irstd1k.py`
- Create: `train_g1.py`
- Create: `tests/test_train_g1.py`

**Interfaces:**
- Consumes: `build_g1_from_checkpoints`, G0 `DatasetSpec`, `BCEDiceLoss`, and `MetricAccumulator`.
- Produces: `load_config(path: str) -> dict`, `build_checkpoint_payload(segmenter, config, epoch, validation_iou, provenance) -> dict`, and CLI training artifacts `g1_best.pt`, `g1_last.pt`, `config_snapshot.json`.

- [ ] **Step 1: Write failing training-contract tests**

```python
def test_g1_config_preserves_g0_protocol():
    g0 = load_python_config("configs/g0_irstd1k.py")
    g1 = load_config("configs/g1_i_b_v5_irstd1k.py")
    assert g1["model"] == {**g0["model"], "in_ch": 2}
    assert g1["optim"] == g0["optim"]
    assert g1["loss"] == g0["loss"]
    assert g1["condition"] == "B"

def test_checkpoint_payload_records_upstream_hashes():
    segmenter = torch.nn.Conv2d(2, 1, 1)
    payload = build_checkpoint_payload(
        segmenter=segmenter,
        config={"run_dir": "runs/g1_i_b_v5"},
        epoch=1,
        validation_iou=0.25,
        provenance={
            "g0_checkpoint": "runs/g0/g0_best.pt",
            "g0_sha256": "a" * 64,
            "stage_a_checkpoint": "runs/a_psf/irstd1k_seed42_v5_dualsource/a_best.pt",
            "stage_a_sha256": "b" * 64,
            "condition": "B",
            "channel_order": ["I", "B"],
        },
    )
    assert payload["provenance"]["channel_order"] == ["I", "B"]
    assert len(payload["provenance"]["g0_sha256"]) == 64
    assert len(payload["provenance"]["stage_a_sha256"]) == 64
```

Test that the optimizer receives only `wrapper.segmenter.parameters()` and
that the configured run directory is exactly `runs/g1_i_b_v5`.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest -q tests/test_train_g1.py`

Expected: `train_g1` and the G1 config do not exist.

- [ ] **Step 3: Implement the minimal training entry**

Mirror G0's loader, cosine warmup, BCE/Dice loss, IoU validation, early
stopping, and checkpoint semantics. Build the wrapper from the two configured
checkpoints. Pass only `model.segmenter.parameters()` to AdamW. Save the
segmenter state and complete provenance, not a duplicate Stage-A state dict.

- [ ] **Step 4: Run training tests and regression tests**

Run: `python -m pytest -q tests/test_train_g1.py tests/test_g1_model.py tests/test_g1_runtime.py tests/test_train_a.py`

Expected: all selected tests pass.

### Task 4: Cross-dataset evaluation, documentation, and real-checkpoint smoke

**Files:**
- Create: `eval_g1.py`
- Create: `README_G1.md`
- Create: `tests/test_eval_g1.py`
- Modify: `docs/superpowers/plans/2026-09-15-g1-i-background-v5.md`

**Interfaces:**
- Consumes: a G1 checkpoint containing `model`, `config`, and `provenance`.
- Produces: `runs/g1_i_b_v5/metrics.json` with experiment identity, upstream hashes, segmentation params, total online params, latency, throughput, and per-dataset G0 metrics.

- [ ] **Step 1: Write failing evaluation tests**

```python
def test_evaluation_metadata_identifies_g1_and_online_cost():
    metadata = build_evaluation_metadata(wrapper, checkpoint)
    assert metadata["experiment"] == "g1_i_b_v5"
    assert metadata["condition"] == "B"
    assert metadata["channel_order"] == ["I", "B"]
    assert metadata["online_params"] >= metadata["segmenter_params"]
```

Test that padding occurs before the wrapper so Stage A and G1 see the same
spatial tensor, predictions are cropped to the original size, and metrics are
written under the configured G1 run directory.

- [ ] **Step 2: Run evaluation tests and verify RED**

Run: `python -m pytest -q tests/test_eval_g1.py`

Expected: import fails because `eval_g1.py` does not exist.

- [ ] **Step 3: Implement evaluation and concise usage documentation**

Reuse G0 dataset factories and `MetricAccumulator`. Report segmentation-only
and complete online-pipeline parameter counts. Document training, evaluation,
the frozen-A invariant, and the fact that scientific gain requires a full
controlled run.

- [ ] **Step 4: Run the complete test suite**

Run: `python -m pytest -q`

Expected: all tests pass with no failures.

- [ ] **Step 5: Run syntax and real-checkpoint smoke checks**

Run:

```bash
python -m compileall -q irstd_g1 train_g1.py eval_g1.py configs/g1_i_b_v5_irstd1k.py
python train_g1.py --config configs/g1_i_b_v5_irstd1k.py --device cpu --limit-train 4 --epochs 1 --smoke-run-dir runs/g1_i_b_v5/smoke
python eval_g1.py --config configs/g1_i_b_v5_irstd1k.py --checkpoint runs/g1_i_b_v5/smoke/g1_last.pt --device cpu --datasets irstd1k --limit 1
```

Expected: both real upstream checkpoints load, one optimizer step completes,
Stage A has no gradients, and evaluation writes one-image metrics.

- [ ] **Step 6: Review diffs and commit implementation**

Run `git diff --check`, confirm G0/A-v5/GaussAMR artifacts are unchanged, then
commit only the G1 package, config, entries, tests, README, and updated plan.
