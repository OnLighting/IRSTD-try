# Stage-A seed-42 v2 AMP-fix audit

Verdict: **the localization collapse is fixed, but the physical decomposition
gate still fails. Do not freeze this checkpoint for G1--G5.**

The run completed normally and early-stopped at epoch 105. The selected
checkpoint is epoch 85 (`a_best.pt`, SHA-256
`d65eb4bf12cc0da8398a164eefdf49fff539bac935f5498ba075036daaeb01aa`).

## What improved over v1

- IRSTD-1K centroid recall increased from 0 to 0.969.
- background target leakage decreased from 0.277 to 0.138;
- uncertainty/error Spearman increased from 0.411 to 0.556;
- PSF/residual overlap decreased from 0.129 to 0.064;
- reconstruction remained good at 33.40 dB PSNR.

## Why v2 still fails

- mean source false activation is 0.811 (median 0.850), above the 0.20 gate;
- mean target energy precision is 0.148 (median 0.114), below the 0.80 gate;
- target-contrast recovery is over-scaled by 10.37x on average (8.88x median);
- `T_psf` saturates at 1.0 on average;
- `R` is nearly zero and its flip correlations are not meaningful (horizontal
  0.296, vertical 0.171);
- cross-domain false activations are visible on scene highlights, especially
  UAVB.

The high `S` and `T_psf` perturbation correlations therefore describe a stable
over-activation pattern, not a correct stable decomposition.

## Root cause and v3 correction

`S` was simultaneously treated as a binary centroid probability and as a
physical source amplitude. Balanced binary focal supervision pushed centroid
values toward one, while the PSF operator multiplied them by
`source_flux_scale=16`. The center and target losses consequently remained in
conflict throughout training (`L_center` about 0.40 and `L_target` about 0.53
near the selected epoch).

The v3 code:

1. builds a one-pixel `source_proxy` whose amplitude is local target-contrast
   flux divided by `source_flux_scale`;
2. regresses `S` to that physical amplitude instead of a binary one;
3. defines sparsity as the fraction of source energy away from centroids,
   directly exposing a dense sigmoid floor;
4. distinguishes narrow target support from the valid 15x15 PSF footprint;
5. scores target-energy over-recovery as an error instead of clipping every
   recall above one to a perfect score;
6. adds an explicit target-energy-miscalibration degeneration flag.

