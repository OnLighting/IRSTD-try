# Stage-A v5 dual-source result analysis

## Decision

V5 is a major in-domain improvement over v4 and fixes the two motivating
public-output failures, but it does **not** yet pass Stage A as a complete,
interpretable, stable decomposition. `B`, public `S`, and `T_psf` are useful on
IRSTD-1K; `R` is parsimoniously unused because PSF fit is good. The remaining
blocking failures are the meaning of `U`, latent `P/A` calibration, horizontal-
flip stability, and UAV cross-domain component collapse.

The selected checkpoint is epoch 45 (`val_score=0.099230`); training stopped at
epoch 65 after 20 non-improving epochs. All reported results are one seed and
do not establish initialization stability.

## In-domain comparison with v4

| IRSTD-1K test metric | v4 | v5 | Interpretation |
| --- | ---: | ---: | --- |
| reconstruction MAE mean | 0.0061 | 0.0180 | worse, but still small |
| centroid recall mean / median | 0.843 / 1.000 | 0.789 / 1.000 | median retained; long tail worsened |
| source false activation median | 0.800 | 0.047 | core dense-`S` failure fixed |
| target contrast recall median | 2.782 | 0.907 | energy overshoot fixed |
| target energy precision median | 0.209 | 0.958 | target-output leakage fixed at the median |
| background target leakage median | 0.135 | 0.091 | improved |
| PSF/residual overlap median | 0.023 | 0.006 | improved |
| uncertainty/error Spearman median | +0.626 | -0.207 | newly failed and sign-inverted |

The aggregate degeneration flags are all false on IRSTD-1K. This is necessary
but not sufficient: the uncertainty gate fails, flip stability fails, and
several median summaries hide a substantial failure tail.

## Long-tail audit on IRSTD-1K

- Only 149/201 images (74.1%) individually pass source false activation <=0.20.
- Only 149/201 images (74.1%) individually pass target precision >=0.80.
- 175/201 (87.1%) pass target recall in [0.5, 2.0].
- Only 36/201 (17.9%) pass uncertainty/error Spearman >=0.30.
- Source false activation has q90=0.650 and maximum approximately 1.0.
- Target outside-energy ratio has median 0.046, q90=1.104, and maximum 35.11.
- Non-centroid source mass ratio has median 1.099, q90=3.252, and maximum 35.68.

Thus v5 is clean for the typical test image but still has severe false-source
and leakage failures on roughly the worst decile. `XDU9`, `XDU999`, `XDU733`,
`XDU302`, and `XDU167` are priority outside-energy failure cases.

## Component interpretation

### B — mostly interpretable in-domain

Median background target leakage is 0.091 and input correlation is 0.99774,
below the 0.999 copy gate. Panels show scene structure retained while compact
targets are locally removed. Reconstruction MAE increased threefold from v4,
so background fidelity is no longer the strongest part of the objective.

### S — public map works; its factors are not independently calibrated

Public `S=P*A` is sparse and target-localized on the typical IRSTD image:
median false activation is 0.047 and median centroid recall is 1.0. However,
mean centroid recall is only 0.789 and the failure tail remains large.

The explanatory factors do not yet match their intended semantics:

- median `P` at centroids is 0.99999, which is desirable;
- median presence false activation is 0.437, so `P` alone is not a clean
  existence map; false locations are frequently suppressed by `A` instead;
- median centroid amplitude MAE is 0.409, so `A` is not a calibrated physical
  photometry estimate.

This means factorization improved the public product without fully separating
binary localization from amplitude.

### T_psf — useful in-domain, partially collapsed kernel usage

Median target recall 0.907, precision 0.958, outside-energy ratio 0.046, and
zero-PSF fraction 0 show a useful compact target component. Visual panels show
localized PSF responses rather than the v4 full-frame activation.

The six learned kernels remain geometrically distinct, but mean selection is
strongly concentrated: K1 usage is 0.850 and K4 is 0.091; the other four sum to
about 0.057. This is partial mixture-use collapse, not a zero-PSF collapse.

### R — effectively zero, conditionally acceptable

Median residual/target energy is 0.000287, median target fraction is
3.7e-7, and meaningful-residual fraction is 0. `R` is therefore not an
empirically active component. Under the approved v5 criterion this is
acceptable on IRSTD-1K because `T_psf` fit is good; it should be described as
"unused by this dataset/model", not as a validated non-PSF residual estimator.

### U — not an uncertainty map

Median uncertainty/error Spearman is -0.207 on IRSTD-1K, -0.354 on SIRST4
non-XDU, and -0.653 on UAV. High `U` therefore does not identify high absolute
reconstruction error. The public map mixes reconstruction uncertainty with PSF
mixture entropy, while the explicit uncertainty loss supervises only the
reconstruction sub-head. This construct mismatch is a plausible mechanism and
must be tested directly in the next revision.

## Stability

Median perturbation Pearson correlations for `(B,S,T_psf,U)` are:

- horizontal flip: `(0.998, 0.777, 0.945, 0.795)`;
- vertical flip: `(0.999, 0.937, 0.957, 0.777)`;
- noise: `(0.999, 0.997, 1.000, 0.983)`;
- intensity: `(1.000, 0.997, 0.999, 0.993)`.

`B`, `S`, and `T_psf` are robust to noise and intensity changes. `S` fails the
0.90 horizontal-flip correlation requirement. Flip normalized-L1 errors are
also large (`S`: 0.747 horizontal, 0.441 vertical; `T_psf`: 0.421 and 0.376),
showing material energy changes even where spatial correlation is high. `U`
fails both flip correlations.

Checkpoint comparisons show similar spatial structure near the best epoch but
unstable energy: epoch 40/50 versus best epoch 45 gives `S` median Pearson
0.929/0.937 but normalized L1 0.711/0.741; `T_psf` gives Pearson 0.985/0.991
but normalized L1 0.650/0.641. Best-checkpoint selection is therefore
meaningful, and the decomposition is not stationary in amplitude.

## Cross-domain evidence

SIRST4 non-XDU is a partial success at the median: source false activation
0.042, target precision 0.960, recall 0.594, and outside-energy ratio 0.005.
Its mean centroid recall is only 0.604 and its worst decile still collapses.

SIRST-UAVB fails: median centroid recall is 0, source false activation 0.995,
non-centroid mass ratio 9.97, target recall 0.044, target precision 0.0048, and
outside-energy ratio 9.87. Both `source_dense` and
`target_energy_miscalibrated` flags are true. The visualization confirms that
bright scene structures are selected as sources while the target is missed.

## Recommended next revision

1. Make public `U` represent one construct. Prefer `U=u_rec`; report PSF
   entropy separately, or supervise the exact blended public map. Add direct
   diagnostics for `u_rec` versus error before choosing.
2. Correct the amplitude/energy objective mismatch: amplitude proxy encodes
   total source flux, while current energy calibration counts output only in
   the narrower target support and excludes valid PSF tails. Calibrate total
   target energy over `psf_support`, retaining pixel fit on the narrow support.
3. Add a mass-normalized penalty for `P` itself so localization cannot delegate
   false-location rejection entirely to `A`.
4. Address flip consistency and dominant-kernel use with an explicit
   equivariance/paired-kernel constraint rather than relying only on random
   flip augmentation.
5. Keep `R` unforced. Reassess it only after corrected PSF energy accounting;
   forcing a nonzero residual would manufacture apparent interpretability.

V5 can be retained as a strong ablation checkpoint showing that dual-source
factorization fixes v4's public density and energy failure. It should not yet
be frozen as the final Stage-A baseline for G1--G5.
