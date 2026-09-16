"""G0 single config.

All hyperparameters for the G0 baseline live here so a reader sees the full
experimental setup in one file.
"""

# Data
data_root = "data/IRSTD-1K"
train_split = "train"
test_split = "test"

# Cross-dataset eval roots
sirst_uavb_root = "data/SIRST-UAVB_OnlyUAV_Form"
sirst4_root = "data/SIRST4-ForLiTE"

# Model
model = dict(
    in_ch=1,
    dims=(48, 96, 192),
    depths=(2, 2, 2),
    num_heads=(3, 6, 12),
    window_size=8,
)

# Optimization
optim = dict(
    name="adamw",
    lr=1e-4,
    weight_decay=0.05,
    warmup_steps=200,   # 1 epoch at bs=4 on ~800 train images
    epochs=200,
    early_stop_patience=30,
    batch_size=4,
    grad_clip=1.0,
)

# Loss
loss = dict(
    bce_weight=0.5,
    dice_weight=0.5,
)

# Binarization for eval
eval_threshold = 0.5

# Output
run_dir = "runs/g0"
