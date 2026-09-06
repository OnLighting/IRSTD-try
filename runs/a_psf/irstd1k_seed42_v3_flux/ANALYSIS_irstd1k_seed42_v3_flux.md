# Stage-A v3 flux run analysis

## Decision

This run does **not** establish an interpretable Stage-A baseline.  The best
checkpoint is epoch 8 only because background leakage temporarily improved;
the sparse source pathway never learned during any of the 28 epochs.

## IRSTD-1K test evidence

- mean reconstruction PSNR: 28.8466 dB
- median target-energy precision: 0.002729
- median target-contrast recall: 0.325797
- median source false activation: 0.999330
- median centroid recall at 5 px: 0.0
- median background target leakage: 0.153525
- median PSF/residual overlap: 0.398008
- median residual target fraction: 0.128546
- median uncertainty/error Spearman: 0.388875
- degeneration flags: `source_dense=true`,
  `target_energy_miscalibrated=true`

The same zero centroid recall and near-one source false-activation ratio occur
on SIRST-UAVB and both SIRST4 subsets.  High perturbation correlations therefore
measure a stable diffuse failure, not a stable target decomposition.

## Training trace

Across epochs 1--28:

- center loss remains 0.665--0.674;
- sparse loss remains 0.99999 under the old scale-invariant definition;
- validation centroid recall remains exactly 0;
- target-energy precision remains about 0.0024--0.0042.

The epoch-8 score is selected when median background leakage falls to 0.1097.
Centroid recall was still 0 and source false activation was 0.9992.  The old
validation score did not include centroid recall.

Visual panels confirm that `S` is a low-amplitude copy of scene structure and
`T_psf` is its blurred full-frame copy.  `B` is recognizable but retains the
target, `R` is small and unstable, and `U` is spatially meaningful but cannot
make the decomposition acceptable.

## Root cause and v4 correction

The source head starts with bias -6 (`S` about 0.0025).  Probability-space L1
center supervision multiplies its gradient by the saturated sigmoid derivative.
On four real validation images at the epoch-8 checkpoint, the weighted source
head gradient norms were 2.706 for reconstruction but only 0.00326 for center
supervision.  The center signal was therefore about 800 times weaker.

The correction uses soft-label BCE on the pre-sigmoid source logits.  It keeps
the physical flux target as the optimum while avoiding the saturated L1
gradient.  Sparse energy is normalized against expected source mass and passed
through `log1p`, with weight 0.01, so it remains scale-sensitive without pinning
the shared source bias at zero.  On the same checkpoint and batch, the corrected
weighted gradient norms are 8.836 for center, 2.706 for reconstruction, and
0.0437 for sparsity.  Checkpoint selection now explicitly penalizes missing
centroid recall, and resume rejects any checkpoint whose full training config
does not match.

The corrected experiment must start from scratch in
`runs/a_psf/irstd1k_seed42_v4_sourcegrad`; v3 optimizer/checkpoint state is not
compatible.
