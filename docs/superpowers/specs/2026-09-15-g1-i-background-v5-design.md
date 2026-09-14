# G1: Frozen A-v5 Background-Conditioned Segmentation

Date: 2026-09-15
Status: Approved design

## Objective

G1 tests one narrow causal question: does the background map `B` predicted by
the frozen Stage-A v5 model improve the existing G0 infrared small-target
segmenter when the raw infrared image remains available?

G1 is an ablation, not a new decomposition-training experiment. It must not
update Stage A, change the segmentation loss, change the dataset split, or add
any Stage-A output other than `B`.

## Experiment contract

- G0 remains the image-only baseline `I -> mask logits`.
- G1 receives the two-channel tensor `[I, B_v5(I)]`.
- The binary mask is used only by the segmentation loss and evaluation
  metrics. It is never passed to Stage A or the segmentation model.
- Stage A is loaded from
  `runs/a_psf/irstd1k_seed42_v5_dualsource/a_best.pt`, placed in evaluation
  mode, and frozen for the entire run.
- G1 writes only below `runs/g1_i_b_v5/`; it must not overwrite G0 or Stage-A
  artifacts.

## Architecture and data flow

For every already-normalized and already-augmented input image `I`, G1 performs:

```text
I --no_grad--> frozen A-v5 --select B--> concatenate [I, B]
                                            |
                                            v
                                G0 Swin-UNet with 2-channel stem
                                            |
                                            v
                                      mask logits
```

Stage A runs online after the dataset augmentation. This avoids stale or
misaligned cached backgrounds for random flips and rotations. Offline caching
is outside G1 scope.

The segmentation backbone is the existing `G0SwinUNet`; only `in_ch` changes
from one to two. No additional fusion block, attention layer, or background
normalization is introduced.

## Initialization

G1 warm-starts from `runs/g0/g0_best.pt`.

- Every shape-compatible G0 parameter is copied exactly.
- The first stem-convolution input slice for `I` is copied from G0.
- The new stem-convolution input slice for `B` is initialized to zero.
- Missing or unexpected non-stem parameters are errors.

This construction gives an initialization invariant: for any valid image and
background tensors, G1 before optimization must produce the same logits as G0
on the image alone, within floating-point tolerance. It prevents the extra
channel from changing predictions before it has learned a useful contribution.

## Training and validation

Only G1 segmentation parameters are passed to AdamW. The Stage-A model:

- remains in `eval()` even when the G1 wrapper is put in training mode;
- has `requires_grad=False` for every parameter;
- is executed under `torch.no_grad()`;
- must not accumulate parameter gradients.

G1 reuses the G0 optimization and evaluation protocol unless an option is
strictly path-related:

- IRSTD-1K train/test splits;
- random flips and 90-degree rotations;
- BCE/Dice weights;
- AdamW settings, warmup, epochs, gradient clipping, and early stopping;
- validation checkpoint selection by IoU;
- evaluation threshold and target matching rule.

This intentionally preserves G0's existing test-split checkpoint-selection
convention for comparability, while documentation must continue to identify it
as mild test-set leakage.

## Configuration and artifacts

G1 has a dedicated config containing:

- the inherited G0 model/training/data values;
- `in_ch=2` for the segmenter;
- absolute or project-relative paths to the G0 and A-v5 checkpoints;
- `run_dir="runs/g1_i_b_v5"`;
- the condition name `B`.

Each saved G1 checkpoint and config snapshot records:

- G0 checkpoint path and SHA-256;
- A-v5 checkpoint path and SHA-256;
- condition name and channel order `[I, B]`;
- seed, epoch, validation score, and effective configuration.

Checkpoint loading must fail clearly for a missing checkpoint, an incompatible
Stage-A configuration, or G0 state-dict differences beyond the expected stem
input-channel expansion.

## Evaluation

G1 uses the existing G0 metrics and cross-dataset roots. Its metrics artifact
must identify the experiment as `g1_i_b_v5` and include both upstream
checkpoint hashes.

The first comparison is G0 versus G1 on identical datasets and metrics. No
claim that `B` helps is allowed from training loss or qualitative panels alone.
The primary evidence is the change in pixel IoU/nIoU and target Pd/Fa; model
parameters and latency are also reported because online Stage A is part of the
inference path.

## Tests

Implementation follows test-driven development. Tests must cover:

1. Loading G0 weights into the two-channel model copies all shared parameters,
   copies the image-channel stem weights, and zeroes the background channel.
2. Before training, G0 on `I` and initialized G1 on `[I, B]` produce matching
   logits for arbitrary `B`.
3. The wrapper returns full-resolution logits with the expected shape.
4. Calling `train()` on the wrapper leaves Stage A in evaluation mode.
5. Stage-A parameters are frozen and receive no gradients after G1 backward.
6. Stage A receives only the image tensor; masks are not accepted by the
   conditioning interface.
7. Missing or incompatible checkpoints raise actionable errors.
8. G1 artifacts use their dedicated output directory and record both SHA-256
   values.

## Acceptance criteria

G1 implementation is complete when:

- all new tests are observed failing before production implementation and
  passing afterward;
- the restored Stage-A tests and existing G0/GaussAMR tests still pass in a
  Python environment containing the project dependencies;
- a small CPU or CUDA smoke run loads both real checkpoints and completes at
  least one forward/backward optimization step;
- no Stage-A gradient is produced;
- the initialization-equivalence test passes;
- G0, A-v5, and existing GaussAMR artifacts remain unchanged.

Full 200-epoch training is a separate experiment execution and is not required
to establish that the G1 implementation is correct.

## Explicit non-goals

- Retraining or repairing Stage A.
- Using `S`, `T_psf`, `R`, or `U`.
- Offline condition caching.
- Joint fine-tuning of Stage A and G1.
- Changing G0 losses, metrics, splits, or backbone capacity.
- Claiming scientific improvement before a controlled G0-versus-G1 run.
