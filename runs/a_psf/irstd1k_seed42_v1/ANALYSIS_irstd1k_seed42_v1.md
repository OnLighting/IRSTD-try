# Stage-A seed-42 v1 audit

Verdict: **failed decomposition gate; do not freeze this checkpoint for G1--G5.**

The run completed normally on an RTX 4090 and early-stopped at epoch 49. The
selected checkpoint is epoch 29 (`a_best.pt`, SHA-256
`6eda95a4d1078e891a5be1a8b0cb29583ac717812bfbb95aa41b7d42a752b431`). Its
IRSTD-1K reconstruction is numerically good (PSNR 34.06 dB), but reconstruction
quality is not evidence that the intended decomposition was identified.

## Component audit

- `B`: mostly follows the low-frequency scene, but still retains target signal.
  Mean background target leakage is 0.277, close to the 0.30 gate.
- `S`: collapsed to a low-amplitude dense floor. Its mean target-external energy
  fraction is 0.9993, maximum activation averages only 0.00182, and centroid
  recall is zero.
- `T_psf`: nonzero but diffuse over the image. Only 0.0109 of its combined
  target-output energy falls in target support on IRSTD-1K (0.0005 on UAVB).
- `R`: does not copy the full image globally, but visual panels show that it can
  take localized target structure. Horizontal-flip Pearson stability is only
  0.646. The old global residual ratio cannot measure this shortcut reliably.
- `U`: has useful error association (mean Spearman 0.411 on IRSTD-1K and 0.573--
  0.697 cross-domain), but is spatially close to a constant floor and drifts
  late in training (best-vs-last Pearson 0.628).
- PSF bank: all six kernels are finite, diverse, anisotropic, and used. The bank
  itself is interpretable, but the collapsed `S` means it is not being applied
  as a localized target formation model.

High perturbation correlations for `S` and `T_psf` are not accepted as evidence
of stability because an almost uniform collapsed map is trivially stable.

## Root cause and v2 correction

Two losses were diluted by image area:

1. focal center supervision averaged rare positive centers together with every
   background pixel, making its observed raw value only about 0.0003--0.0005;
2. residual regularization had no support-normalized target-region term, so `R`
   could cheaply absorb the target.

The v2 code normalizes positive and negative focal terms independently, adds a
support-normalized residual penalty, records `residual_target_fraction`, uses it
for checkpoint selection and collapse gating, uses an amplitude-aware centroid
threshold, and auto-scales sparse component audit panels.

