# GaussAMR-Net V1 Design

**Date:** 2026-09-08

**Status:** proposed for implementation review

**Repository:** `new_model_v3`

**Baseline:** existing G0 I-only Swin-UNet

## 1. Objective

Build a minimal single-frame infrared small-target detector that uses the
Gaussian appearance of thermal targets to reduce computation, rather than to
reconstruct every target as a fixed optical point-spread function.

V1 must answer two questions before any additional research features are
introduced:

1. Can a cheap Gaussian-aware router cover nearly all targets with a fixed,
   small proposal budget?
2. Can a heavy segmentation refiner operate only on routed regions and still
   beat a Gaussian-only mask while reducing end-to-end inference cost relative
   to G0?

V1 is a correctness and feasibility experiment. It is not intended to establish
state of the art.

## 2. Scope

### In scope

- Single-frame, single-channel infrared input.
- Full mask supervision during training.
- A fixed, separable Gaussian/DoG feature bank.
- A shallow full-image router at 1/8 resolution.
- A fixed number of Gaussian proposals per image.
- Two fixed-budget refinement levels.
- A Gaussian base mask plus a learned local residual mask.
- Full-image IoU/nIoU/Pd/Fa and batch-1 latency/FLOPs/peak-memory reporting.
- Reuse of G0 dataset loading, normalization, splits, and metric definitions.

### Out of scope

- PSF or background/target matrix decomposition.
- Reinforcement learning or sequential glimpse policies.
- Learned early stopping or variable iteration counts.
- Mixture-of-experts with independently parameterized experts.
- Video, tracking, Kalman filtering, temporal recurrence, or event cameras.
- Language, foundation-model, diffusion, or synthetic-data branches.
- Custom CUDA kernels.
- Claims of SOTA accuracy or guaranteed speedup.

These exclusions are hard V1 boundaries. They may be reconsidered only after
the correctness gate in Section 12 passes.

## 3. Non-negotiable contracts

- Training consumes `(I, Y)`. Inference consumes `I` only. `Y` may only be used
  by target construction, losses, and metrics; it must never enter model
  features or routing at inference.
- The expensive refinement path must never process a dense full-resolution
  feature map.
- Proposal counts and crop tensor shapes must be fixed by configuration so GPU
  work is predictable.
- The final output must be a full-resolution logit tensor with shape
  `(B, 1, H, W)` and must remain compatible with `irstd_g0.metrics`.
- Images may have arbitrary height and width. The runtime pads on the bottom and
  right to a multiple of 8, then crops the final logits back to the original
  shape before loss or metric computation.
- Every reported speed result must include the router, crop extraction,
  refinement, and full-image composition. Sparse-branch FLOPs alone are not an
  end-to-end speed result.

## 4. Architecture overview

```text
I: (B,1,H,W)
  |
  +-- GaussianFeatureBank: fixed separable Gaussian and normalized DoG channels
  |       output: (B, 1 + 2S, H, W)
  |
  +-- GaussianRouter: shallow stride-2 depthwise-separable CNN
  |       output at 1/8 resolution:
  |       objectness, sub-cell offset, log_sigma_x, log_sigma_y, uncertainty
  |
  +-- local-max suppression + fixed top-K1 proposals
  |       proposals: (B,K1,6) = [score, mu_x, mu_y, sigma_x, sigma_y, uncertainty]
  |
  +-- ContextRefiner on K1 half-resolution crops
  |       corrected Gaussian parameters + keep score
  |
  +-- fixed top-K2 proposals, K2 <= K1
  |       |
  |       +-- DetailRefiner on K2 full-resolution crops
  |              Gaussian parameter corrections + local residual logits
  |
  +-- SparseGaussianComposer
          truncated Gaussian logits + local residual logits -> (B,1,H,W)
```

The architecture is deliberately proposal-based, but differs from an ordinary
region-proposal detector in three testable ways: the proposal state is an
explicit Gaussian distribution, Gaussian uncertainty participates in both
ranking and refinement, and the same Gaussian is the analytic base used to
compose the final mask.

## 5. Gaussian feature bank

`GaussianFeatureBank` receives normalized `I` and constructs features using
fixed scales:

```text
sigmas = (0.8, 1.2, 1.8, 2.6) input pixels
kappa = 1.6
```

For every sigma it produces:

1. `G_sigma * I`;
2. a standardized center-surround response

```text
dog_sigma = (G_sigma * I - G_(kappa*sigma) * I) /
            sqrt(max(G_(kappa*sigma) * I^2 - (G_(kappa*sigma) * I)^2, eps))
```

Each two-dimensional Gaussian must be implemented as horizontal then vertical
one-dimensional depthwise convolution. Kernels are buffers, not trainable
parameters. Kernel radius is `ceil(3 * sigma)`. Variance is clamped below by
`eps = 1e-6`.

The output is the raw image concatenated with the four smoothed channels and
four standardized DoG channels, for nine channels total. Keeping the raw image
prevents the physical prior from deleting negative-contrast or non-Gaussian
evidence.

## 6. Gaussian router and proposal representation

The router uses three stride-2 depthwise-separable blocks with channel widths
`(16, 24, 32)`. A final 1x1 head predicts six maps at 1/8 resolution:

```text
objectness_logit: 1
offset_xy:        2, tanh constrained to [-0.5, 0.5] router cells
log_sigma_xy:     2
uncertainty_logit:1
```

Decoded input-pixel parameters are:

```text
mu_x = 8 * (grid_x + 0.5 + offset_x)
mu_y = 8 * (grid_y + 0.5 + offset_y)
sigma_xy = clamp(softplus(log_sigma_xy) + 0.5, 0.5, 8.0)
uncertainty = sigmoid(uncertainty_logit)
```

V1 uses diagonal covariance only. Correlation and rotation are excluded.

Inference applies 3x3 local-max suppression to objectness and selects exactly
`K1 = 16` cells per image with tensorized `topk`. Empty images still carry 16
slots; low-score slots remain valid tensor entries but are trained to produce
negative final logits. No Python loop may select proposals.

The routing priority used after context refinement is:

```text
priority = sigmoid(objectness_logit) * (1 + 0.5 * uncertainty)
```

This reserves detail computation for both likely and ambiguous candidates.

## 7. Fixed-budget refinement

### 7.1 Context refinement

- Capacity: `K1 = 16` proposals per image.
- Source: the nine-channel Gaussian-bank tensor downsampled to 1/2 resolution.
- Physical crop support: 64x64 input pixels centered at each `mu`.
- Tensor crop shape: 32x32 through `grid_sample`.
- Network: two residual depthwise-separable convolution blocks, width 32.
- Outputs per proposal: corrected objectness, `delta_mu_xy`,
  `delta_log_sigma_xy`, and corrected uncertainty.

Corrections are bounded to at most 4 input pixels for each center coordinate and
`log(2)` for each sigma coordinate.

After correction, the model selects exactly `K2 = 8` proposals per image by the
priority formula in Section 6.

### 7.2 Detail refinement

- Capacity: `K2 = 8` proposals per image.
- Source: full-resolution crops from the nine-channel Gaussian-bank tensor.
- Physical crop support: 48x48 input pixels.
- Tensor crop shape: 48x48.
- Network: a small U-Net with widths `(24, 48, 72)` and one output channel.
- Outputs: a 48x48 residual-logit patch and final bounded corrections to
  `mu_xy`, `log_sigma_xy`, and objectness.

The detail U-Net is shared by all proposals. There are no separate target and
background experts in V1.

## 8. Sparse Gaussian composition

For each final proposal, the composer creates a diagonal Gaussian only inside
its 48x48 crop:

```text
g_k(x, y) = -0.5 * (((x-mu_x)/sigma_x)^2 + ((y-mu_y)/sigma_y)^2)
base_logit_k = objectness_logit + g_k
patch_logit_k = base_logit_k + residual_logit_k
```

Values outside Mahalanobis radius 3 are replaced by a constant background logit
`-12`. Proposal patches are pasted to the full canvas with differentiable
bilinear sampling. Overlaps are combined with `logsumexp`; the background logit
is included as an additional component.

The composer performs no convolution on the full-resolution canvas. Its dense
work is limited to canvas initialization, paste/scatter, overlap reduction, and
the final crop to `(H, W)`.

## 9. Target construction and matching

Connected components in `Y` define target instances during training only. For
each component:

- `mu` is the binary-mask centroid;
- `sigma_x` and `sigma_y` are the square roots of the second central moments,
  clamped to `[0.5, 8.0]`;
- a one-pixel component receives `(sigma_x, sigma_y) = (0.75, 0.75)`;
- target objectness is 1 at the nearest router cell;
- all other router cells are negatives except a one-cell ignore ring around the
  positive cell.

Router targets are matched by cell assignment. Context and detail proposals are
matched one-to-one to ground-truth components by greedy minimum center distance,
with a maximum distance of 12 input pixels. Unmatched proposals are hard
negatives. A ground-truth component not matched by any selected proposal
contributes to the coverage loss.

## 10. Losses

The total training loss is:

```text
L = 1.0 * L_router_focal
  + 1.0 * L_center
  + 0.25 * L_sigma
  + 1.0 * L_coverage
  + 0.5 * L_proposal_cls
  + 1.0 * L_local_bce
  + 1.0 * L_local_dice
  + 0.5 * L_full_bce
  + 0.5 * L_full_dice
  + 0.05 * L_isotropy
```

- `L_router_focal`: alpha-balanced sigmoid focal loss with `alpha=0.75`,
  `gamma=2`.
- `L_center`: Smooth-L1 on matched centers, normalized by 8 pixels.
- `L_sigma`: Smooth-L1 on matched `log_sigma_xy`.
- `L_coverage`: for every target, `-log(max router probability over all cells
  within 12 input pixels + eps)`. The fixed-top-k coverage values are reported
  as diagnostics rather than used as a differentiable loss.
- `L_proposal_cls`: BCE on matched versus unmatched context/detail proposals.
- `L_local_bce` and `L_local_dice`: detail-patch segmentation losses.
- `L_full_bce` and `L_full_dice`: losses on composed full-image logits.
- `L_isotropy`: mean absolute `log_sigma_x - log_sigma_y` on positive
  proposals; it is a weak prior rather than an equality constraint.

## 11. Training protocol

Training has three explicit phases controlled by fractions of total steps:

1. **Oracle refinement, first 10%:** detail/context crops use ground-truth
   Gaussians with center jitter uniformly sampled from `[-4, 4]` pixels and
   sigma multiplier sampled from `[0.75, 1.25]`. Router losses are trained in
   parallel, but router proposals do not feed the refiners.
2. **Mixed routing, next 20%:** each image uses oracle proposals with probability
   linearly decaying from 1 to 0; remaining images use predicted proposals.
3. **Predicted routing, final 70%:** only predicted top-K proposals feed the
   refiners. Unmatched high-objectness proposals provide hard negatives.

Top-k indices are not required to be differentiable. Router supervision and
coverage loss train the router directly; gradients from crop contents may flow
to proposal coordinates through `grid_sample`, but not through the discrete
choice of proposal identities.

The first implementation uses the optimizer, augmentation policy, image
normalization, and dataset splits from `configs/g0_irstd1k.py` unless this design
explicitly overrides them.

The model forward contract is:

```text
forward(image, routing_mode="predicted", targets=None) -> {
    "logits":          (B,1,H,W),
    "gaussian_logits": (B,1,H,W),
    "router_maps":     dict[str, Tensor],
    "proposals_l1":    (B,16,6),
    "proposals_l2":    (B,8,6),
    "local_logits":    (B,8,1,48,48),
}
```

`targets` is required only for `routing_mode="oracle"` or `"mixed"`; passing
targets in predicted inference mode is an error. Evaluation calls predicted
mode without targets.

## 12. Fast correctness gate

Implementation is considered structurally correct only when all four gates pass.

### Gate A: deterministic geometry tests

- A synthetic centered Gaussian is recovered by the feature/router decoding
  utilities with center error <= 0.5 input pixel.
- Crop followed by paste places an impulse at the original coordinate with
  error <= 0.5 pixel, including all four image borders.
- Composing two overlapping proposals produces finite logits and gradients.
- Every model output has exactly the unpadded input height and width.

### Gate B: optimization sanity

- All enabled losses and gradients are finite for positive, multi-target, and
  empty images.
- The model overfits a deterministic 16-image subset to training nIoU >= 0.90.
- On that subset, router coverage@16 reaches 1.00 and coverage@8 reaches at
  least 0.95.

### Gate C: short real-data probe

Run a fixed-seed train/validation probe using 64 IRSTD-1K training images and 32
validation images. Stop at 10 epochs or 1,000 optimizer steps, whichever comes
first. Record:

- router coverage@16 and coverage@8;
- matched center error and sigma error;
- Gaussian-only nIoU;
- Gaussian-plus-residual nIoU;
- full IoU/nIoU/Pd/Fa;
- number of active positive and hard-negative proposals.

The probe passes when coverage@16 >= 0.95, the residual output improves nIoU
over the Gaussian-only output, and no output/loss collapses to a constant or
non-finite value. This gate tests correctness, not competitive accuracy.

### Gate D: efficiency sanity

At 512x512, batch size 1:

- DetailRefiner processes exactly `8 * 48 * 48 = 18,432` spatial sites, 7.03%
  of the 262,144 full-image sites.
- Report total parameters and end-to-end FLOPs for both G0 and GaussAMR V1.
- Measure CUDA latency after 50 warm-up runs over 200 timed runs with explicit
  synchronization; report median and p95.
- Report peak CUDA allocated memory.

V1 remains viable if its heavy-path spatial work is bounded as specified and
its end-to-end median latency is lower than G0. If FLOPs fall but latency does
not, crop/paste dispatch is the diagnosed bottleneck and must be optimized
before architectural expansion.

## 13. Required ablations

The first full experiment must include exactly these variants:

1. `G0`: existing dense Swin-UNet.
2. `G-only`: router Gaussian proposals composed without DetailRefiner residuals.
3. `Learned-router`: same fixed budget, but raw-image router input without the
   Gaussian feature bank.
4. `Gauss-router`: full V1 router and Gaussian composition without local
   residuals.
5. `GaussAMR-V1`: full router, both refinement levels, and residual composition.

This isolates whether gains arise from Gaussian evidence, sparse routing, or
the local refiner. No extra module may be added before these comparisons exist.

## 14. Diagnostic interpretation

- Low coverage@16: router/target assignment failure; do not tune the detail
  network.
- High coverage but poor Gaussian-only nIoU: parameter decoding or target
  moment mismatch.
- Good Gaussian-only result but no residual gain: local target extraction,
  matching, or residual-loss failure.
- Good validation nIoU with high Fa: insufficient unmatched-proposal negatives.
- Good FLOPs but poor latency: non-vectorized crop/paste or excessive small GPU
  operations.
- Good in-domain results but poor cross-dataset coverage: Gaussian scales or
  router normalization overfit the training sensor.

## 15. Planned implementation boundaries

Implementation will live beside G0 rather than modify it:

```text
configs/gaussamr_v1_irstd1k.py
irstd_gaussamr/__init__.py
irstd_gaussamr/gaussian_bank.py
irstd_gaussamr/router.py
irstd_gaussamr/refiners.py
irstd_gaussamr/composer.py
irstd_gaussamr/targets.py
irstd_gaussamr/losses.py
irstd_gaussamr/model.py
irstd_gaussamr/diagnostics.py
train_gaussamr_v1.py
eval_gaussamr_v1.py
tests/test_gaussian_bank.py
tests/test_gaussamr_geometry.py
tests/test_gaussamr_model.py
tests/test_gaussamr_losses.py
tests/test_gaussamr_smoke.py
README_GAUSSAMR_V1.md
```

`irstd_g0.data` and `irstd_g0.metrics` remain the canonical dataset and metric
implementations. G0 files are not refactored during V1.

## 16. Novelty boundary and provenance

The defensible contribution is not the existence of Gaussian heatmaps,
Gaussian convolution, sparse decomposition, or region proposals independently.
All already exist in or near IRSTD. The V1 research claim to test is:

> A learned Gaussian posterior can serve as a fixed-budget spatial and
> resolution router, so expensive IRSTD computation is allocated according to
> physically meaningful target probability and uncertainty, while sparse
> Gaussian composition restores a full-resolution segmentation output.

Closest prior directions that must be discussed in any later paper are
DISTA-Net's Gaussian-PSF sparse unmixing, Gaussian-prior convolution for IRSTD,
RPCANet's sparse-target deep unfolding, Mixture-of-Depths fixed-capacity token
routing, dynamic token-pass dense segmentation, uncertainty-aware probabilistic
keypoints, and compact Gaussian splatting. V1 intentionally tests their
cross-domain intersection rather than claiming that any constituent mechanism
is new.
