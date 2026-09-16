# GaussAMR V3 novelty scan

Retrieved: 2026-09-15

## Question

Does the current GaussAMR V3 design duplicate existing infrared small-target
detection research?

## Search scope

Focused searches covered Gaussian/PSF priors, Gaussian-shaped convolution,
explicit Gaussian target parameters, Top-K proposal selection, sparse spatial
routing, coarse-to-fine proposal refinement, uncertainty-guided computation,
and analytic Gaussian masks with learned residual refinement. Searches used
publisher, conference, and arXiv pages available through 2026-09-15.

The configured `research-lookup` script was attempted first, but the local
Python runtime lacked `requests`; official web sources were used as fallback.

## Closest prior work

1. **Differentiable Sparse Mask Guided Infrared Small Target Fast Detection
   Network** (Sheng et al., 2026), DOI 10.11999/JEIT250989.
   Generates a differentiable sparse mask, samples candidate target regions,
   suppresses dense background computation, and refines sparse features. This
   is the closest overlap with GaussAMR's high-level sparse-computation story.
   https://jeit.ac.cn/cn/article/doi/10.11999/JEIT250989

2. **Denoising-Enhanced Coarse-to-Fine Infrared Small Target Detection with
   Attention Prior-Guided Knowledge Distillation / ECFNet** (Fang et al.,
   2026 preprint). Uses grid-based region proposals followed by a lightweight
   fine detector to avoid redundant background computation.
   https://arxiv.org/abs/2606.21956

3. **Interior Attention-Aware Network for Infrared Small Target Detection**
   (Wang et al., IEEE TGRS 2022), DOI 10.1109/TGRS.2022.3163410. Uses an RPN to
   obtain coarse target regions and then performs region-focused refinement.
   https://ieeexplore.ieee.org/document/9757265

4. **One-Stage Cascade Refinement Networks for Infrared Small Target Detection
   / OSCAR** (Dai et al., 2022/2023). Uses high-level soft proposals to drive
   low-level coarse-to-fine refinement.
   https://arxiv.org/abs/2212.08472

5. **Region Energy-Aware Learning with Gaussian-Prior Convolution for Infrared
   Small Target Detection** (Liu et al., ICASSP 2026), DOI
   10.1109/ICASSP55912.2026.11461594. Injects a Gaussian target prior into
   feature extraction and couples it with region-energy-aware learning.
   https://doi.org/10.1109/ICASSP55912.2026.11461594

6. **Revisiting the Scale Loss Function and Gaussian-Shape Convolution for
   Infrared Small Target Detection** (Li and Zhuo, 2026 preprint). Uses a
   learnable Gaussian-shaped convolution and orientation mask.
   https://arxiv.org/abs/2604.09991

7. **DISTA-Net: Dynamic Closely-Spaced Infrared Small Target Unmixing** (Han et
   al., ICCV 2025). Explicitly models targets using a 2-D Gaussian PSF and
   estimates sparse sub-pixel target quantities, locations, and intensities.
   Its task is target unmixing rather than full-image semantic mask
   segmentation, but its physical parameterization is close.
   https://openaccess.thecvf.com/content/ICCV2025/html/Han_DISTA-Net_Dynamic_Closely-Spaced_Infrared_Small_Target_Unmixing_ICCV_2025_paper.html

8. **IR-SAM2: Target Enhancement with SAM2 for Infrared Small Target
   Detection** (Remote Sensing 2026). Selects Top-K high-frequency candidate
   points and turns them into positional queries for SAM2-based segmentation.
   https://www.mdpi.com/2072-4292/18/12/1891

9. **Pick of the Bunch / SeRankDet** (Dai et al., 2024). Uses nonlinear Top-K
   selection to retain salient target responses at constant complexity.
   https://arxiv.org/abs/2408.03717

10. **SCOFM-Net: Saliency-guided sparse Mamba for infrared small-target
    detection** (Qiu et al., 2026). Uses saliency/high-frequency cues to route
    expensive state-space processing toward informative spatial locations.
    https://www.sciencedirect.com/science/article/pii/S0030399226006900

11. **STIFNet: A Scale-Guided Sparse Target Interaction and Context Feedback
    Network** (Xiang et al., 2026), DOI 10.1016/j.knosys.2026.116898. Uses
    scale guidance, Top-K sparse interaction, candidate routing, and context
    feedback, overlapping with the broader selective-computation narrative.
    https://doi.org/10.1016/j.knosys.2026.116898

12. **Boosting IRSTD via Logit-Domain Contrast and Adaptive Shape Refinement**
    (Zeng et al., 2026 preprint). Uses logit-domain supervision and adaptive
    shape/boundary refinement, adjacent to but not the same as GaussAMR's
    analytic Gaussian logit plus local residual composition.
    https://arxiv.org/abs/2607.01555

## Assessment

- **Already crowded:** Gaussian-like target priors, DoG/local saliency,
  Top-K/sparse selection, proposal-based coarse-to-fine refinement, and
  residual/shape refinement.
- **No exact one-to-one match found:** a fixed-budget hierarchy in which each
  proposal is an explicit `(score, mu_x, mu_y, sigma_x, sigma_y, uncertainty)`
  state, the same Gaussian state produces an analytic full-mask base, and a
  full-resolution local network predicts only residual logits that are fused
  by a sparse composer.
- **Potentially differentiating combination:** explicit Gaussian state reused
  consistently for routing, ranking, support definition, analytic mask
  composition, and measurable support-coverage contracts.
- **Weak/unsafe novelty claims:** merely using a Gaussian prior; merely using
  Top-K; merely cropping candidate patches; merely being coarse-to-fine; or
  merely using uncertainty.
- **Evidence limitation:** absence of an exact match in this focused search is
  not proof of worldwide novelty. Patent databases, Chinese-language
  databases, and paywalled full text require a separate formal novelty search.

## Recommended claim boundary

Describe the contribution as a unified, fixed-budget, Gaussian-state sparse
segmentation formulation and its support-consistent optimization, not as the
first Gaussian-prior, first sparse, or first coarse-to-fine IRSTD method.

