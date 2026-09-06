# Stage-A v4 source-gradient run analysis

## Decision

V4 is a substantial improvement over v3 but does **not** establish the Stage-A
baseline.  It establishes that the model can learn target locations, while the
single source head cannot yet provide both selective localization and calibrated
physical flux.

## Training and selection

- Training completed all 150 epochs; it did not early-stop.
- Best checkpoint: epoch 147, validation score 1.069664.
- Validation centroid recall reached 1.0 at epoch 2 and remained there.
- Validation source false activation fell from 0.9914 to 0.7208.
- Validation target precision rose from 0.0084 to 0.2932.
- Validation target contrast recall remained overestimated at 3.0415.
- Late checkpoints are numerically stable; epochs 140 and 150 are effectively
  identical to epoch 147 for all five maps.

## IRSTD-1K test evidence

- mean reconstruction MAE: 0.006113
- median centroid recall at 5 px: 1.0
- median source false activation: 0.800336 (fails <= 0.20)
- median target-energy precision: 0.209321 (fails >= 0.80)
- median target-contrast recall: 2.782272 (fails 0.50--2.00)
- median background target leakage: 0.134611 (passes <= 0.30)
- median PSF/residual overlap: 0.023299 (passes <= 0.50)
- median residual target fraction: 0.000059 (passes <= 0.50)
- median uncertainty/error Spearman: 0.626116 (passes >= 0.30)
- degeneration flags: `source_dense=true`,
  `target_energy_miscalibrated=true`

## Component interpretation

- `B`: good in-domain reconstruction and acceptable median target leakage, but
  some targets leave a dark subtraction mark.  Cross-domain backgrounds can be
  biased or nearly copy the input.
- `S`: target-centered in-domain, but extra bright structures receive point
  activations.  The false source energy is quantitatively and visually real.
- `T_psf`: compact PSF spots replace the diffuse v3 map, but their aggregate
  energy is too high and false spots remain.
- `R`: nearly zero.  It does not take a shortcut, but the experiment provides
  little evidence that this branch has a useful role.
- `U`: follows reconstruction error in-domain and cross-domain, with Spearman
  medians 0.626, 0.841, and 0.760 on IRSTD-1K, UAVB, and non-XDU SIRST4.

## Stability

Noise and intensity perturbations are stable for all components.  Horizontal
and vertical flips are stable for `B`, `S`, `T_psf`, and mostly `U`; `R` is not
stable (vertical-flip Pearson median 0.2334), consistent with its near-zero
energy.  Only one initialization was tested, as intended for exploration.

## Root-cause evidence

On four real validation images at epoch 147, non-centroid source mass was
5--85 times the expected source mass.  Target output outside the legal PSF
support was 0.96--75 times the true target energy.  Nevertheless, the current
per-background-pixel leakage loss was only 0.001277 because averaging over the
whole image dilutes sparse false positives.

The single sigmoid source map is also asked to solve two different tasks:

1. binary localization (target versus background); and
2. continuous physical source flux.

Soft amplitude supervision fixes flux semantics but does not strongly teach
binary rejection of target-like background structures.  Increasing the global
sparsity weight enough to remove them would again suppress dim true targets.

## Recommended next architecture

Do not continue v4 or only retune patience.  Split the source representation
into a presence gate and an amplitude head:

`S = sigmoid(presence_logits) * sigmoid(amplitude_logits)`.

Train the presence gate with separately normalized positive and negative
localization losses, and train amplitude only at true centroids against the
physical flux proxy.  Add target-energy and outside-support losses normalized
by each image's true target energy rather than by image area.  This directly
matches the acceptance metrics and removes the localization/amplitude conflict.

Because this changes the architecture, a new checkpoint family and a fresh run
are required; v4 checkpoints cannot be resumed.
