import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from irstd_gaussamr.composer import SparseGaussianComposer
from irstd_gaussamr.refiners import (
    ContextRefiner,
    DetailRefiner,
    context_refiner_loss,
    detail_crops,
    gaussian_patch_logits,
    select_detail_proposals,
)
from irstd_gaussamr.router_probe import (
    GaussianFeatureBank,
    GaussianRouter,
    augment_pair,
    decode_proposals,
    extract_instances,
    fixed_subset,
    proposal_diagnostics,
    router_loss,
)


class GaussianFeatureBankTest(unittest.TestCase):
    def test_constant_image_has_nine_channels_and_zero_dog(self):
        bank = GaussianFeatureBank()
        image = torch.ones(2, 1, 33, 35)

        features = bank(image)

        self.assertEqual(features.shape, (2, 9, 33, 35))
        self.assertEqual(sum(p.numel() for p in bank.parameters()), 0)
        torch.testing.assert_close(features[:, 5:], torch.zeros_like(features[:, 5:]), atol=2e-4, rtol=0)


class ProposalDecodeTest(unittest.TestCase):
    def test_decodes_cell_offset_in_input_pixels(self):
        objectness = torch.full((1, 1, 8, 8), -10.0)
        objectness[0, 0, 4, 3] = 10.0
        offset = torch.zeros(1, 2, 8, 8)
        offset[0, 0, 4, 3] = 0.25
        offset[0, 1, 4, 3] = -0.25
        maps = {
            "objectness_logit": objectness,
            "offset_xy": offset,
            "log_sigma_xy": torch.zeros(1, 2, 8, 8),
            "uncertainty_logit": torch.zeros(1, 1, 8, 8),
        }

        proposals = decode_proposals(maps, k=1)

        self.assertEqual(proposals.shape, (1, 1, 6))
        torch.testing.assert_close(proposals[0, 0, 1:3], torch.tensor([30.0, 34.0]))

    def test_small_images_still_return_the_fixed_budget(self):
        maps = {
            "objectness_logit": torch.zeros(1, 1, 4, 4),
            "offset_xy": torch.zeros(1, 2, 4, 4),
            "log_sigma_xy": torch.zeros(1, 2, 4, 4),
            "uncertainty_logit": torch.zeros(1, 1, 4, 4),
        }

        proposals = decode_proposals(maps, k=24)

        self.assertEqual(proposals.shape, (1, 24, 6))
        torch.testing.assert_close(proposals[:, 16:, 0], torch.zeros(1, 8))


class GaussianRouterTest(unittest.TestCase):
    def test_outputs_six_maps_at_one_eighth_resolution(self):
        maps = GaussianRouter()(torch.randn(2, 9, 65, 70))

        self.assertEqual(maps["objectness_logit"].shape, (2, 1, 9, 9))
        self.assertEqual(maps["offset_xy"].shape, (2, 2, 9, 9))
        self.assertEqual(maps["log_sigma_xy"].shape, (2, 2, 9, 9))
        self.assertEqual(maps["uncertainty_logit"].shape, (2, 1, 9, 9))
        self.assertLessEqual(float(maps["offset_xy"].abs().max()), 0.5)


class SparseGaussianComposerTest(unittest.TestCase):
    def test_overlapping_proposals_have_finite_logits_and_gradients(self):
        proposals = torch.tensor([[
            [0.99, 10.0, 12.0, 1.0, 1.0, 0.5],
            [0.80, 10.5, 12.0, 1.5, 1.0, 0.5],
        ]], requires_grad=True)

        logits = SparseGaussianComposer()(proposals, (24, 24))
        logits.sum().backward()

        self.assertEqual(logits.shape, (1, 1, 24, 24))
        self.assertEqual(int(logits[0, 0].argmax()), 12 * 24 + 10)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(proposals.grad).all())

    def test_zero_residual_is_identity_and_nonzero_residual_has_gradient(self):
        proposals = torch.tensor([[[0.9, 12.0, 12.0, 1.5, 1.5, 0.5]]])
        residual = torch.zeros(1, 1, 1, 48, 48, requires_grad=True)
        composer = SparseGaussianComposer()

        baseline = composer(proposals, (24, 24))
        refined = composer(proposals, (24, 24), residual)
        refined.sum().backward()

        torch.testing.assert_close(refined, baseline)
        self.assertTrue(torch.isfinite(residual.grad).all())

    def test_residual_cannot_create_logits_outside_gaussian_support(self):
        proposals = torch.tensor([[[0.99, 24.0, 24.0, 1.0, 1.0, 0.5]]])
        residual = torch.zeros(1, 1, 1, 48, 48)
        residual[0, 0, 0, 0, 0] = 30
        composer = SparseGaussianComposer()

        baseline = composer(proposals, (48, 48))
        refined = composer(proposals, (48, 48), residual)

        torch.testing.assert_close(refined[0, 0, 0, 0], baseline[0, 0, 0, 0])
        self.assertEqual(int(refined[0, 0].argmax()), 24 * 48 + 24)


class ContextRefinerTest(unittest.TestCase):
    def test_zero_initialized_refiner_preserves_fixed_budget_proposals(self):
        features = torch.randn(1, 9, 64, 80)
        proposals = torch.tensor([[[0.75, 30.0, 24.0, 1.5, 2.0, 0.4]]]).repeat(1, 24, 1)

        refined = ContextRefiner()(features, proposals)

        self.assertEqual(refined.shape, (1, 24, 6))
        torch.testing.assert_close(refined, proposals, atol=1e-6, rtol=1e-6)

    def test_context_loss_is_finite_and_updates_refiner(self):
        refiner = ContextRefiner()
        features = torch.randn(1, 9, 64, 64)
        proposals = torch.tensor([[[0.6, 20.0, 20.0, 1.0, 1.0, 0.5]]]).repeat(1, 24, 1)
        instances = torch.tensor([[21.0, 19.0, 1.5, 1.2]])

        refined = refiner(features, proposals)
        losses = context_refiner_loss(refined, instances)
        losses["total"].backward()

        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertTrue(any(parameter.grad is not None for parameter in refiner.parameters()))

    def test_context_classification_balances_one_positive_against_negatives(self):
        proposals = torch.tensor([[[0.5, 20.0, 20.0, 1.0, 1.0, 0.5]]]).repeat(1, 24, 1)
        proposals.requires_grad_()
        instances = torch.tensor([[20.0, 20.0, 1.0, 1.0]])

        losses = context_refiner_loss(proposals, instances)
        losses["classification"].backward()

        self.assertAlmostEqual(float(proposals.grad[..., 0].sum()), 0.0, places=5)


class DetailRefinerTest(unittest.TestCase):
    def test_selects_fixed_k2_by_probability_uncertainty_priority(self):
        proposals = torch.zeros(1, 24, 6)
        proposals[..., 0] = torch.linspace(0.9, 0.1, 24)
        proposals[..., 5] = 0.0
        proposals[0, 10, 0] = 0.65
        proposals[0, 10, 5] = 1.0

        selected, indices = select_detail_proposals(proposals, k=8)

        self.assertEqual(selected.shape, (1, 8, 6))
        self.assertEqual(indices.shape, (1, 8))
        self.assertIn(10, indices[0].tolist())

    def test_zero_initialized_detail_residual_preserves_gaussian_patch(self):
        features = torch.randn(1, 9, 64, 80)
        proposals = torch.tensor([[[0.8, 30.0, 24.0, 1.5, 2.0, 0.5]]]).repeat(1, 8, 1)
        refiner = DetailRefiner()

        residual = refiner(features, proposals)
        local_logits = gaussian_patch_logits(proposals) + residual

        self.assertEqual(residual.shape, (1, 8, 1, 48, 48))
        torch.testing.assert_close(residual, torch.zeros_like(residual))
        torch.testing.assert_close(local_logits, gaussian_patch_logits(proposals))

    def test_detail_crop_keeps_integer_center_aligned(self):
        image = torch.zeros(1, 1, 64, 64)
        image[0, 0, 20, 30] = 1
        proposals = torch.tensor([[[0.9, 30.0, 20.0, 1.0, 1.0, 0.5]]])

        crop = detail_crops(image, proposals)

        self.assertEqual(crop.shape, (1, 1, 1, 48, 48))
        self.assertEqual(int(crop[0, 0, 0].argmax()), 24 * 48 + 24)

    def test_local_residual_is_truncated_outside_gaussian_support(self):
        proposals = torch.tensor([[[0.99, 24.0, 24.0, 1.0, 1.0, 0.5]]])
        residual = torch.zeros(1, 1, 1, 48, 48)
        residual[0, 0, 0, 0, 0] = 30

        logits = gaussian_patch_logits(proposals, residual_logits=residual)

        self.assertEqual(float(logits[0, 0, 0, 0, 0]), -12.0)
        self.assertEqual(int(logits[0, 0, 0].argmax()), 24 * 48 + 24)


class TargetConstructionTest(unittest.TestCase):
    def test_extracts_centers_and_moments_with_single_pixel_fallback(self):
        mask = torch.zeros(1, 24, 24)
        mask[0, 4:7, 10:13] = 1
        mask[0, 20, 2] = 1

        instances = extract_instances(mask)

        expected_sigma = (2 / 3) ** 0.5
        expected = torch.tensor([
            [11.0, 5.0, expected_sigma, expected_sigma],
            [2.0, 20.0, 0.75, 0.75],
        ])
        torch.testing.assert_close(instances, expected)


class RouterLossTest(unittest.TestCase):
    def test_positive_target_has_finite_loss_and_increasing_logit_gradient(self):
        objectness = torch.zeros(1, 1, 4, 4, requires_grad=True)
        maps = {
            "objectness_logit": objectness,
            "offset_xy": torch.zeros(1, 2, 4, 4, requires_grad=True),
            "log_sigma_xy": torch.zeros(1, 2, 4, 4, requires_grad=True),
            "uncertainty_logit": torch.zeros(1, 1, 4, 4, requires_grad=True),
        }
        instances = torch.tensor([[12.0, 12.0, 1.2, 1.2]])

        losses = router_loss(maps, instances)
        losses["total"].backward()

        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertLess(float(objectness.grad[0, 0, 1, 1]), 0.0)


class ProposalDiagnosticsTest(unittest.TestCase):
    def test_reports_budget_dependent_target_coverage(self):
        proposals = torch.zeros(1, 24, 6)
        proposals[0, :, 0] = torch.linspace(0.9, 0.1, 24)
        proposals[0, :, 1:3] = 100
        proposals[0, :, 3:5] = 1
        proposals[0, 0, 1:3] = torch.tensor([12.0, 12.0])
        proposals[0, 23, 1:3] = torch.tensor([44.0, 44.0])
        instances = torch.tensor([
            [12.0, 12.0, 1.0, 1.0],
            [44.0, 44.0, 1.0, 1.0],
        ])

        stats = proposal_diagnostics(proposals, instances)

        self.assertEqual(stats["targets"], 2)
        self.assertEqual(stats["hits_at_8"], 1)
        self.assertEqual(stats["hits_at_16"], 1)
        self.assertEqual(stats["hits_at_24"], 2)


class FixedSubsetTest(unittest.TestCase):
    def test_seed_42_selects_a_stable_prefix(self):
        self.assertEqual(fixed_subset([str(i) for i in range(10)], 4, 42), ["7", "3", "2", "8"])

    def test_geometric_augmentation_keeps_image_and_mask_aligned(self):
        image = torch.arange(6).reshape(1, 1, 2, 3)
        expected = torch.tensor([[[[0, 3], [1, 4], [2, 5]]]])

        got_image, got_mask = augment_pair(image, image.clone(), torch.Generator().manual_seed(0))

        torch.testing.assert_close(got_image, expected)
        torch.testing.assert_close(got_mask, expected)


class RouterProbeCliTest(unittest.TestCase):
    def test_rejects_nonpositive_probe_size(self):
        result = subprocess.run(
            [sys.executable, "train_gaussamr_router_probe.py", "--train-size", "0"],
            cwd=Path(__file__).parents[1],
            text=True,
            capture_output=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be positive", result.stderr)

    def test_one_step_writes_router_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sirst4"
            (root / "images").mkdir(parents=True)
            (root / "masks").mkdir()
            (root / "img_idx").mkdir()
            for split, sid in (("train", "a"), ("test", "b")):
                image = np.zeros((32, 32), dtype=np.uint8)
                mask = np.zeros((32, 32), dtype=np.uint8)
                image[14:17, 14:17] = 255
                mask[14:17, 14:17] = 255
                Image.fromarray(image).save(root / "images" / f"{sid}.png")
                Image.fromarray(mask).save(root / "masks" / f"{sid}.png")
                (root / "img_idx" / f"{split}.txt").write_text(sid + "\n", encoding="utf-8")
            run_dir = Path(tmp) / "run"

            result = subprocess.run(
                [
                    sys.executable,
                    "train_gaussamr_router_probe.py",
                    "--data-root", str(root),
                    "--run-dir", str(run_dir),
                    "--train-size", "1",
                    "--val-size", "1",
                    "--epochs", "1",
                    "--max-steps", "1",
                    "--device", "cpu",
                    "--full-mask-loss",
                ],
                cwd=Path(__file__).parents[1],
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["config"]["seed"], 42)
            self.assertEqual(metrics["config"]["train_size"], 1)
            self.assertEqual(metrics["config"]["val_size"], 1)
            self.assertEqual(metrics["config"]["lr"], 1e-3)
            self.assertEqual(metrics["config"]["lr_schedule"], "constant")
            self.assertEqual(metrics["config"]["k1"], 24)
            self.assertIn("coverage_at_24", metrics["best"])
            self.assertIn("gaussian_n_iou", metrics["best"])
            self.assertIn("router_probability_std", metrics["best"])

            context_run_dir = Path(tmp) / "context_run"
            context_result = subprocess.run(
                [
                    sys.executable,
                    "train_gaussamr_router_probe.py",
                    "--data-root", str(root),
                    "--run-dir", str(context_run_dir),
                    "--train-size", "1",
                    "--val-size", "1",
                    "--epochs", "1",
                    "--max-steps", "1",
                    "--device", "cpu",
                    "--full-mask-loss",
                    "--context-refiner",
                    "--init-checkpoint", str(run_dir / "router_best.pt"),
                ],
                cwd=Path(__file__).parents[1],
                text=True,
                capture_output=True,
            )

            self.assertEqual(context_result.returncode, 0, context_result.stdout + context_result.stderr)
            context_metrics = json.loads((context_run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(context_metrics["config"]["context_refiner"])
            self.assertIn("context_matched_center_error_px", context_metrics["best"])
            context_checkpoint = torch.load(
                context_run_dir / "context_best.pt", map_location="cpu", weights_only=False
            )
            self.assertIn("context_refiner", context_checkpoint)

            detail_run_dir = Path(tmp) / "detail_run"
            detail_result = subprocess.run(
                [
                    sys.executable,
                    "train_gaussamr_router_probe.py",
                    "--data-root", str(root),
                    "--run-dir", str(detail_run_dir),
                    "--train-size", "1",
                    "--val-size", "1",
                    "--epochs", "1",
                    "--max-steps", "1",
                    "--device", "cpu",
                    "--full-mask-loss",
                    "--detail-refiner",
                    "--init-checkpoint", str(run_dir / "router_best.pt"),
                ],
                cwd=Path(__file__).parents[1],
                text=True,
                capture_output=True,
            )

            self.assertEqual(detail_result.returncode, 0, detail_result.stdout + detail_result.stderr)
            detail_metrics = json.loads((detail_run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("detail_n_iou", detail_metrics["best"])
            self.assertIn("passes_detail_gate", detail_metrics)
            detail_checkpoint = torch.load(
                detail_run_dir / "detail_best.pt", map_location="cpu", weights_only=False
            )
            self.assertIn("detail_refiner", detail_checkpoint)


if __name__ == "__main__":
    unittest.main()
