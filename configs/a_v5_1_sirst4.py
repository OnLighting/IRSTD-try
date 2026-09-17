"""Historical Stage-A v5.1 training on SIRST4."""

from configs.a_v5_1_irstd1k import diagnostics, loss, model, optim as base_optim, run

# SIRST4 contains hundreds of native resolutions. Process one image at a time
# and accumulate four gradients so training preserves geometry and effective
# batch size without padding image content into the decomposition losses.
optim = dict(base_optim, batch_size=1, grad_accum_steps=4)

data = dict(
    name="sirst4", root="data/SIRST4-ForLiTE", train_split="train",
    test_split="test", val_count=229, split_seed=42, dilation_radius=3,
    ring_radius=9, irstd1k_root="data/IRSTD-1K",
    sirst_uavb_root="data/SIRST-UAVB_OnlyUAV_Form",
    sirst4_root="data/SIRST4-ForLiTE",
)
