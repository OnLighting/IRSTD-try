from __future__ import annotations

from pathlib import Path

import pytest

from irstd_g0.data import DatasetSpec, SIRST4Dataset, SIRSTUAVBDataset, build_dataset
from train_a import build_training_datasets, load_config


CASES = (
    ("configs/a_v5_1_irstd1k.py", "irstd1k", 800, 80, 720),
    ("configs/a_v5_1_sirst_uavb.py", "sirst_uavb", 2400, 240, 2160),
    ("configs/a_v5_1_sirst4.py", "sirst4", 2285, 229, 2056),
)


@pytest.mark.parametrize(
    ("config_path", "name", "official_train", "val_count", "effective_train"),
    CASES,
)
def test_matrix_configs_produce_exact_deterministic_splits(
    config_path: str,
    name: str,
    official_train: int,
    val_count: int,
    effective_train: int,
) -> None:
    config = load_config(config_path)
    first = build_training_datasets(config["data"], seed=42)
    second = build_training_datasets(config["data"], seed=42)
    train, val, train_ids, val_ids = first

    assert config["data"]["name"] == name
    assert len(train_ids) == effective_train
    assert len(val_ids) == val_count
    assert len(train_ids) + len(val_ids) == official_train
    assert not set(train_ids) & set(val_ids)
    assert train.ids == train_ids and val.ids == val_ids
    assert second[2:] == (train_ids, val_ids)


def test_sirst_training_datasets_enable_only_train_augmentation() -> None:
    uav_root = "data/SIRST-UAVB_OnlyUAV_Form"
    sirst4_root = "data/SIRST4-ForLiTE"

    assert SIRSTUAVBDataset(uav_root, split="train", augment=True).augment is True
    assert SIRSTUAVBDataset(uav_root, split="test", augment=True).augment is False
    assert SIRST4Dataset(sirst4_root, split="train", augment=True).augment is True
    assert SIRST4Dataset(sirst4_root, split="test", augment=True).augment is False


def test_existing_v52a_config_remains_trainable() -> None:
    config = load_config("configs/a_psf_irstd1k.py")

    train, val, train_ids, val_ids = build_training_datasets(config["data"], seed=42)

    assert len(train) == len(train_ids) == 720
    assert len(val) == len(val_ids) == 80


def test_factory_builds_trainable_sirst_dataset() -> None:
    dataset = build_dataset(
        DatasetSpec(
            name="sirst_uavb",
            root="data/SIRST-UAVB_OnlyUAV_Form",
            split="train",
            augment=True,
        )
    )

    assert dataset.augment is True
    assert len(dataset) == 2400


def test_duplicate_split_ids_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "duplicate"
    (root / "80_20").mkdir(parents=True)
    (root / "80_20" / "train.txt").write_text("same\nsame\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        build_dataset(DatasetSpec("irstd1k", str(root), "train", False))


def test_unknown_dataset_name_is_actionable() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_dataset(DatasetSpec("not_a_dataset", "data", "train", False))
