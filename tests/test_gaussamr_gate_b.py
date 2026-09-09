import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from irstd_g0.losses import BCEDiceLoss
from irstd_gaussamr.composer import SparseGaussianComposer
from irstd_gaussamr.gate_b import (
    GateBThresholds,
    build_overfit_loaders,
    detail_stage_loss,
    detail_gate_passes,
    evaluate_gate_b,
    router_gate_passes,
    router_stage_loss,
    scatter_detail_residual,
    seed_everything,
)
from irstd_gaussamr.refiners import DetailRefiner
from irstd_gaussamr.router_probe import GaussianFeatureBank, GaussianRouter


def _write_sirst4(root: Path, ids: list[str]) -> None:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    (root / "img_idx").mkdir()
    for index, sid in enumerate(ids):
        image = np.full((32, 32), index, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        Image.fromarray(image).save(root / "images" / f"{sid}.png")
        Image.fromarray(mask).save(root / "masks" / f"{sid}.png")
    (root / "img_idx" / "train.txt").write_text(
        "\n".join(ids) + "\n", encoding="utf-8"
    )


class GateBDecisionTest(unittest.TestCase):
    def test_requires_all_router_thresholds(self):
        self.assertTrue(
            router_gate_passes({"coverage_at_24": 1.0, "coverage_at_8": 0.96})
        )
        self.assertFalse(
            router_gate_passes({"coverage_at_24": 0.99, "coverage_at_8": 0.96})
        )
        self.assertFalse(
            router_gate_passes({"coverage_at_24": 1.0, "coverage_at_8": 0.949})
        )

    def test_requires_detail_niou_and_router_thresholds(self):
        passing = {
            "detail_n_iou": 0.90,
            "coverage_at_24": 1.0,
            "coverage_at_8": 0.96,
        }
        self.assertTrue(detail_gate_passes(passing))
        self.assertFalse(detail_gate_passes({**passing, "detail_n_iou": 0.899}))
        self.assertFalse(detail_gate_passes({**passing, "coverage_at_8": 0.94}))

    def test_custom_thresholds_are_honored(self):
        relaxed = GateBThresholds(n_iou=0.0, coverage_at_24=0.0, coverage_at_8=0.0)

        self.assertTrue(
            router_gate_passes(
                {"coverage_at_24": 0.0, "coverage_at_8": 0.0}, relaxed
            )
        )
        self.assertTrue(
            detail_gate_passes(
                {"detail_n_iou": 0.0, "coverage_at_24": 0.0, "coverage_at_8": 0.0},
                relaxed,
            )
        )


class GateBLoaderTest(unittest.TestCase):
    def test_rejects_nonpositive_subset_size_before_reading_dataset(self):
        with self.assertRaisesRegex(ValueError, "subset_size must be positive"):
            build_overfit_loaders("missing-dataset", subset_size=0, seed=42)

    def test_rejects_training_split_smaller_than_requested_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sirst4"
            _write_sirst4(root, ["a", "b"])

            with self.assertRaisesRegex(
                ValueError, "requested 3 images, but train split contains 2"
            ):
                build_overfit_loaders(root, subset_size=3, seed=42)

    def test_train_and_eval_share_seeded_subset_but_only_train_shuffles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sirst4"
            _write_sirst4(root, ["a", "b", "c", "d", "e"])

            train, evaluation, selected = build_overfit_loaders(
                root, subset_size=3, seed=42
            )
            train_again, evaluation_again, selected_again = build_overfit_loaders(
                root, subset_size=3, seed=42
            )

            self.assertEqual(selected, ["d", "b", "c"])
            self.assertEqual(selected_again, selected)
            self.assertEqual(train.dataset.ids, selected)
            self.assertEqual(evaluation.dataset.ids, selected)
            self.assertIsNot(train.dataset, evaluation.dataset)
            self.assertIsInstance(train.sampler, RandomSampler)
            self.assertIsInstance(evaluation.sampler, SequentialSampler)
            self.assertEqual(
                [batch["id"][0] for batch in train],
                [batch["id"][0] for batch in train_again],
            )
            self.assertEqual(
                [batch["id"][0] for batch in evaluation],
                [batch["id"][0] for batch in evaluation_again],
            )


class GateBLossTest(unittest.TestCase):
    @staticmethod
    def _cases() -> dict[str, torch.Tensor]:
        empty = torch.zeros(1, 1, 64, 64)
        single = empty.clone()
        single[..., 30:33, 30:33] = 1
        multi = single.clone()
        multi[..., 8:10, 50:52] = 1
        return {"empty": empty, "single": single, "multi": multi}

    def test_scatter_places_k2_residuals_in_their_k1_slots(self):
        residual = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(
            2, 3, 1, 4, 4
        )
        indices = torch.tensor([[5, 1, 3], [0, 4, 2]])

        scattered = scatter_detail_residual(residual, indices, proposal_count=6)

        self.assertEqual(scattered.shape, (2, 6, 1, 4, 4))
        torch.testing.assert_close(scattered[0, 5], residual[0, 0])
        torch.testing.assert_close(scattered[0, 1], residual[0, 1])
        torch.testing.assert_close(scattered[1, 2], residual[1, 2])
        torch.testing.assert_close(scattered[0, 0], torch.zeros_like(scattered[0, 0]))

    def test_router_and_detail_losses_have_finite_gradients_for_gate_b_cases(self):
        for name, mask in self._cases().items():
            with self.subTest(stage="router", case=name):
                image = torch.randn_like(mask, requires_grad=True)
                bank = GaussianFeatureBank()
                router = GaussianRouter(widths=(4, 6, 8))
                losses = router_stage_loss(
                    bank,
                    router,
                    SparseGaussianComposer(),
                    BCEDiceLoss(),
                    image,
                    mask,
                )
                self.assertEqual(
                    set(losses),
                    {
                        "total",
                        "router_focal",
                        "router_center",
                        "router_sigma",
                        "router_coverage",
                        "full_mask",
                    },
                )
                self.assertTrue(all(value.ndim == 0 for value in losses.values()))
                self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                losses["total"].backward()
                self.assertIsNotNone(image.grad)
                self.assertTrue(torch.isfinite(image.grad).all())
                for parameter in router.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

            with self.subTest(stage="detail", case=name):
                bank = GaussianFeatureBank()
                router = GaussianRouter(widths=(4, 6, 8))
                detail = DetailRefiner(widths=(4, 8, 12))
                losses = detail_stage_loss(
                    bank,
                    router,
                    detail,
                    SparseGaussianComposer(),
                    BCEDiceLoss(),
                    torch.randn_like(mask),
                    mask,
                )
                self.assertEqual(set(losses), {"total", "local", "full"})
                self.assertTrue(all(value.ndim == 0 for value in losses.values()))
                self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                torch.testing.assert_close(
                    losses["total"], 2 * losses["local"] + losses["full"]
                )
                losses["total"].backward()
                for parameter in detail.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertTrue(
                    all(parameter.grad is None for parameter in router.parameters())
                )


class GateBSeedTest(unittest.TestCase):
    def test_reseeding_repeats_python_numpy_and_torch_streams(self):
        seed_everything(17)
        first = (random.random(), float(np.random.rand()), torch.rand(3))
        seed_everything(17)
        second = (random.random(), float(np.random.rand()), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2])
        self.assertTrue(torch.are_deterministic_algorithms_enabled())


class _TwoTargetRouter(nn.Module):
    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, _, height, width = features.shape
        map_height, map_width = height // 8, width // 8
        logits = features.new_full((batch, 1, map_height, map_width), -10.0)
        logits[:, :, 1, 1] = 10.0
        logits[:, :, 5, 5] = 9.0
        return {
            "objectness_logit": logits,
            "offset_xy": features.new_zeros(batch, 2, map_height, map_width),
            "log_sigma_xy": features.new_zeros(batch, 2, map_height, map_width),
            "uncertainty_logit": features.new_zeros(batch, 1, map_height, map_width),
        }


class GateBEvaluationTest(unittest.TestCase):
    def test_reports_target_additive_coverage_and_both_mask_metrics(self):
        image = torch.zeros(1, 64, 64)
        mask = torch.zeros(1, 64, 64)
        image[:, 11:14, 11:14] = 1
        mask[:, 11:14, 11:14] = 1
        image[:, 43:46, 43:46] = 1
        mask[:, 43:46, 43:46] = 1
        loader = DataLoader(
            [{"image": image, "mask": mask, "id": "two"}], batch_size=1
        )

        metrics = evaluate_gate_b(
            GaussianFeatureBank(),
            _TwoTargetRouter(),
            loader,
            "cpu",
            SparseGaussianComposer(),
            DetailRefiner(widths=(4, 8, 12)),
        )

        self.assertEqual(metrics["images"], 1)
        self.assertEqual(metrics["targets"], 2)
        self.assertEqual(metrics["hits_at_8"], 2)
        self.assertEqual(metrics["hits_at_24"], 2)
        self.assertEqual(metrics["coverage_at_8"], 1.0)
        self.assertEqual(metrics["coverage_at_24"], 1.0)
        for prefix in ("gaussian", "detail"):
            self.assertEqual(metrics[f"{prefix}_images"], 1)
            self.assertGreaterEqual(metrics[f"{prefix}_iou"], 0.0)
            self.assertLessEqual(metrics[f"{prefix}_iou"], 1.0)
            self.assertGreaterEqual(metrics[f"{prefix}_n_iou"], 0.0)
            self.assertLessEqual(metrics[f"{prefix}_n_iou"], 1.0)
            self.assertIn(f"{prefix}_pd", metrics)
            self.assertIn(f"{prefix}_fa_per_image", metrics)
        self.assertEqual(metrics["gaussian_n_iou"], metrics["detail_n_iou"])


if __name__ == "__main__":
    unittest.main()
