# G0 Baseline (I-only Swin-UNet)

Reliable I-only lightweight segmentation baseline for infrared small-target
detection.

## Contract

- **Train input:** `(I, Y)` — `I` is the single-frame IR image, `Y` is the
  full-image binary mask used as supervision only.
- **Eval input:** `I` only. `Y` is loaded solely for metric computation.
- **Output:** `P(Y|I)` at original resolution; binarize at `eval_threshold`.

No mask-derived feature is allowed at inference; otherwise the task collapses
into label leakage.

## Backbone

Minimal Swin-UNet with local window attention, high-resolution skip connections,
and no repeated mean/max pooling. Default
config: `dims=(48,96,192)`, `depths=(2,2,2)`, `window_size=8`, params ≈ 6 M
at 512×512.

## Loss

`0.5 · BCEWithLogits + 0.5 · Dice`. No target-level loss yet (added after G0).

## Reported metrics

G0 reports:

- **Pixel:** IoU (mean over images), nIoU (per-image IoU averaged).
- **Target:** Pd (image-level detection probability), Fa (false alarms per
  image). Matching rule: greedy nearest centroid, 8-connectivity, radius ≤ 5 px.
- **Efficiency:** params, FLOPs (at 512×512), per-image latency, throughput.

Error-structure analysis (漏检 / 孤立虚警 / 粘连 / 晕环) is intentionally
deferred until G0 numbers are stable.

## Files

| File | Purpose |
| --- | --- |
| `configs/g0_irstd1k.py` | Single source of truth for hyperparameters. |
| `irstd_g0/data.py` | Dataset wrappers (IRSTD-1K, SIRST-UAVB, SIRST4). |
| `irstd_g0/model.py` | Swin-UNet backbone + param/FLOPs probes. |
| `irstd_g0/losses.py` | BCE + Dice. |
| `irstd_g0/metrics.py` | IoU / nIoU / Pd / Fa + matching rule. |
| `train_g0.py` | Training entry. Saves `runs/g0/g0_last.pt`. |
| `eval_g0.py` | Cross-dataset eval. Writes `runs/g0/metrics.json`. |

## What G0 does NOT include

- No PSF decomposition or conditional diffusion.
- No target-level loss yet — added after G0 confirms pixel-level behaviour.
- No error-structure classification — added after G0 numbers are stable.
- No language / ViT branch.
- No tests written — the user explicitly requested none.

## How to run

```bash
# Train
python train_g0.py --config configs/g0_irstd1k.py

# Eval (writes runs/g0/metrics.json)
python eval_g0.py --checkpoint runs/g0/g0_last.pt
```

## Acceptance gate (before claiming G0 complete)

1. Loss converges on IRSTD-1K (visual eyeballing per epoch).
2. Cross-dataset eval writes a metrics JSON with explicit match_rule.
3. Params < 30 M, FLOPs reported (install `thop` for the count).
4. No code path reads `Y` outside `losses.py` / `metrics.py`.
