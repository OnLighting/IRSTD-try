"""Historical Stage-A v5.1 training on SIRST-UAVB."""

from configs.a_v5_1_irstd1k import diagnostics, loss, model, optim, run

data = dict(
    name="sirst_uavb", root="data/SIRST-UAVB_OnlyUAV_Form",
    train_split="train", test_split="test", val_count=240, split_seed=42,
    dilation_radius=3, ring_radius=9, irstd1k_root="data/IRSTD-1K",
    sirst_uavb_root="data/SIRST-UAVB_OnlyUAV_Form",
    sirst4_root="data/SIRST4-ForLiTE",
)
