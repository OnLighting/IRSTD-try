"""Exploratory single-seed configuration for independent module-A training."""

data = dict(
    root="data/IRSTD-1K",
    train_split="train",
    test_split="test",
    val_count=80,
    split_seed=42,
    dilation_radius=3,
    ring_radius=9,
    sirst_uavb_root="data/SIRST-UAVB_OnlyUAV_Form",
    sirst4_root="data/SIRST4-ForLiTE",
)

model = dict(
    in_ch=1,
    dims=(32, 64, 128),
    num_psf=6,
    kernel_size=15,
    sigma_min=0.6,
    sigma_max=4.0,
    source_flux_scale=16.0,
)

loss = dict(
    # Bump whenever loss semantics change. Resume validation compares the
    # complete config and rejects checkpoints trained under another objective.
    objective_version="v6.2",
    weights=dict(
        rec=1.0,
        bg=1.0,
        sp=1.0,
        ctr=1.0,
        psf=1.0,
        ind=1.0,
        flip=1.0,
        amp=1.0,
    ),
)

optim = dict(
    name="adamw",
    lr=2e-4,
    weight_decay=1e-4,
    warmup_steps=500,
    epochs=150,
    early_stop_patience=20,
    batch_size=4,
    grad_accum_steps=1,
    grad_clip=1.0,
    amp=True,
    num_workers=4,
)

diagnostics = dict(
    centroid_radius_px=5,
    perturbations=dict(noise_std=0.01, intensity_scale=0.9, intensity_bias=0.05),
    gates=dict(
        target_energy_precision_min=0.80,
        target_contrast_recall_min=0.50,
        target_contrast_recall_max=2.00,
        source_false_activation_max=0.20,
        background_target_leakage_max=0.30,
        psf_residual_overlap_max=0.50,
        uncertainty_error_spearman_min=0.30,
        residual_energy_ratio_max=0.90,
        residual_target_fraction_max=0.50,
        target_psf_zero_fraction_max=0.25,
        background_input_correlation_max=0.999,
        flip_intensity_correlation_min=0.90,
        flip_S_pearson_min=0.90,
        noise_correlation_min=0.80,
        coverage_at_5px_min=0.85,
    ),
)

run = dict(
    seed=42,
    checkpoint_interval=10,
    keep_milestones=True,
)
