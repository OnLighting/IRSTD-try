import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
from irstd_gaussamr.model import GaussAMRV1
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


class GateBCliTest(unittest.TestCase):
    def test_parser_exposes_gate_b_defaults(self):
        from train_gaussamr_gate_b import build_parser

        args = build_parser().parse_args([])

        self.assertEqual(args.seed, 42)
        self.assertEqual(args.subset_size, 16)
        self.assertEqual(args.epochs, 200)
        self.assertEqual(args.max_steps, 3200)
        self.assertEqual(args.lr, 1e-3)
        self.assertEqual(args.batch_size, 1)

    def test_validation_rejects_incompatible_or_invalid_arguments(self):
        from train_gaussamr_gate_b import build_parser, validate_args

        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sirst4"
            _write_sirst4(root, ["one"])

            cases = [
                (["--data-root", str(root), "--subset-size", "0"], "positive"),
                (["--data-root", str(root), "--batch-size", "2"], "batch size 1"),
                (
                    ["--data-root", str(root), "--detail-checkpoint", "detail.pt"],
                    "requires --router-checkpoint",
                ),
                (["--data-root", str(root), "--verify-only"], "requires --checkpoint"),
            ]
            for argv, message in cases:
                with self.subTest(argv=argv):
                    with self.assertRaisesRegex(ValueError, message):
                        validate_args(parser.parse_args(argv))

            with self.assertRaisesRegex(ValueError, "dataset train split"):
                validate_args(
                    parser.parse_args(["--data-root", str(Path(tmp) / "missing")])
                )

    def test_resume_pair_requires_the_same_router_weights(self):
        from train_gaussamr_gate_b import validate_resume_compatibility

        with tempfile.TemporaryDirectory() as tmp:
            router = GaussianRouter()
            router_path = Path(tmp) / "router.pt"
            detail_path = Path(tmp) / "detail.pt"
            torch.save({"router": router.state_dict()}, router_path)
            torch.save(
                {
                    "router": router.state_dict(),
                    "detail_refiner": DetailRefiner().state_dict(),
                },
                detail_path,
            )
            validate_resume_compatibility(router_path, detail_path, "cpu")

            incompatible = GaussianRouter()
            with torch.no_grad():
                incompatible.head.bias.add_(1)
            torch.save(
                {
                    "router": incompatible.state_dict(),
                    "detail_refiner": DetailRefiner().state_dict(),
                },
                detail_path,
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                validate_resume_compatibility(router_path, detail_path, "cpu")

    def test_detail_resume_is_rechecked_against_actual_router_best(self):
        from train_gaussamr_gate_b import _run_detail_stage

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            original_router = GaussianRouter()
            changed_router = GaussianRouter()
            changed_router.load_state_dict(original_router.state_dict())
            with torch.no_grad():
                changed_router.head.bias.add_(1)
            detail_path = run_dir / "detail_resume.pt"
            torch.save({"router": changed_router.state_dict()}, run_dir / "router_best.pt")
            torch.save(
                {
                    "router": original_router.state_dict(),
                    "detail_refiner": DetailRefiner().state_dict(),
                },
                detail_path,
            )
            args = SimpleNamespace(
                device="cpu",
                detail_checkpoint=str(detail_path),
                lr=1e-3,
            )

            with self.assertRaisesRegex(ValueError, "incompatible"):
                _run_detail_stage(
                    args,
                    {},
                    [],
                    None,
                    None,
                    GaussianFeatureBank(),
                    SparseGaussianComposer(),
                    GateBThresholds(),
                    run_dir,
                )

    def test_real_all_empty_subset_cannot_pass_or_start_detail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sirst4"
            run_dir = Path(tmp) / "run"
            _write_sirst4(root, ["empty"])
            result = subprocess.run(
                [
                    sys.executable,
                    "train_gaussamr_gate_b.py",
                    "--data-root",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--subset-size",
                    "1",
                    "--epochs",
                    "1",
                    "--max-steps",
                    "1",
                    "--device",
                    "cpu",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            summary = json.loads(
                (run_dir / "gate_b_summary.json").read_text(encoding="utf-8")
            )

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertTrue(summary["finite_checks"]["router"]["passed"])
            self.assertGreaterEqual(summary["finite_checks"]["router"]["count"], 1)
            self.assertFalse(summary["target_presence_pass"])
            self.assertFalse(summary["finite_checks"]["detail"]["passed"])
            self.assertEqual(summary["finite_checks"]["detail"]["count"], 0)
            self.assertFalse(summary["overall_pass"])
            self.assertFalse((run_dir / "detail_last.pt").exists())

    def test_resume_at_ceiling_runs_finite_probes_without_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sirst4"
            first_run = Path(tmp) / "first"
            resumed_run = Path(tmp) / "resumed"
            _write_sirst4(root, ["one"])
            image = np.zeros((64, 64), dtype=np.uint8)
            image[30:34, 30:34] = 255
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[30:34, 30:34] = 255
            Image.fromarray(image).save(root / "images" / "one.png")
            Image.fromarray(mask).save(root / "masks" / "one.png")
            base_command = [
                sys.executable,
                "train_gaussamr_gate_b.py",
                "--data-root",
                str(root),
                "--subset-size",
                "1",
                "--epochs",
                "1",
                "--max-steps",
                "1",
                "--device",
                "cpu",
                "--smoke-test",
            ]
            first = subprocess.run(
                [*base_command, "--run-dir", str(first_run)],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            detail_checkpoint = torch.load(
                first_run / "detail_last.pt", map_location="cpu", weights_only=True
            )
            router_checkpoint = {
                "router": detail_checkpoint["router"],
                "epoch": 1,
                "step": 1,
                "batch_offset": 1,
                "epoch_complete": True,
            }
            router_path = Path(tmp) / "router_at_ceiling.pt"
            torch.save(router_checkpoint, router_path)

            resumed = subprocess.run(
                [
                    *base_command,
                    "--run-dir",
                    str(resumed_run),
                    "--router-checkpoint",
                    str(router_path),
                    "--detail-checkpoint",
                    str(first_run / "detail_last.pt"),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            summary = json.loads(
                (resumed_run / "gate_b_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["router_steps"], 1)
            self.assertEqual(summary["detail_steps"], 1)
            self.assertEqual(summary["finite_checks"]["router"]["count"], 1)
            self.assertEqual(summary["finite_checks"]["detail"]["count"], 1)
            self.assertEqual(
                summary["router_stopping_reason"],
                "threshold_met_initial_after_finite_probe",
            )
            self.assertEqual(
                summary["detail_stopping_reason"],
                "threshold_met_initial_after_finite_probe",
            )

    def test_router_mid_epoch_resume_matches_uninterrupted_order_and_weights(self):
        from train_gaussamr_gate_b import _run_router_stage

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sirst4"
            _write_sirst4(root, ["a", "b", "c"])
            impossible = GateBThresholds(
                n_iou=2.0, coverage_at_24=2.0, coverage_at_8=2.0
            )

            def run_stage(
                run_dir: Path, max_steps: int, checkpoint: Path | None = None
            ):
                seed_everything(73)
                train_loader, eval_loader, selected = build_overfit_loaders(
                    root, subset_size=3, seed=73
                )
                run_dir.mkdir()
                args = SimpleNamespace(
                    device="cpu",
                    router_checkpoint=str(checkpoint) if checkpoint else None,
                    detail_checkpoint=None,
                    lr=1e-3,
                    seed=73,
                    epochs=1,
                    max_steps=max_steps,
                    smoke_test=False,
                )
                result = _run_router_stage(
                    args,
                    {},
                    selected,
                    train_loader,
                    eval_loader,
                    GaussianFeatureBank(),
                    SparseGaussianComposer(),
                    impossible,
                    run_dir,
                )
                checkpoint_payload = torch.load(
                    run_dir / "router_last.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                return result, checkpoint_payload

            full_result, full = run_stage(Path(tmp) / "full", max_steps=3)
            first_result, first = run_stage(Path(tmp) / "first", max_steps=1)
            resumed_result, resumed = run_stage(
                Path(tmp) / "resumed",
                max_steps=3,
                checkpoint=Path(tmp) / "first" / "router_last.pt",
            )

            full_order = [
                sample_id
                for record in full_result[1]
                for sample_id in record.get("sample_ids", [])
            ]
            split_order = [
                sample_id
                for history in (first_result[1], resumed_result[1])
                for record in history
                for sample_id in record.get("sample_ids", [])
            ]
            self.assertEqual(split_order, full_order)
            self.assertEqual(first["epoch"], 1)
            self.assertEqual(first["batch_offset"], 1)
            self.assertFalse(first["epoch_complete"])
            for name, expected in full["router"].items():
                torch.testing.assert_close(resumed["router"][name], expected, rtol=0, atol=0)

    def test_one_step_smoke_writes_complete_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sirst4"
            run_dir = Path(tmp) / "run"
            _write_sirst4(root, ["one"])
            image = np.zeros((64, 64), dtype=np.uint8)
            image[30:34, 30:34] = 255
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[30:34, 30:34] = 255
            Image.fromarray(image).save(root / "images" / "one.png")
            Image.fromarray(mask).save(root / "masks" / "one.png")

            result = subprocess.run(
                [
                    sys.executable,
                    "train_gaussamr_gate_b.py",
                    "--data-root",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--subset-size",
                    "1",
                    "--epochs",
                    "1",
                    "--max-steps",
                    "1",
                    "--device",
                    "cpu",
                    "--smoke-test",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            expected = {
                "selected_ids.json",
                "router_best.pt",
                "router_last.pt",
                "detail_best.pt",
                "detail_last.pt",
                "gaussamr_v1_gate_b.pt",
                "gate_b_summary.json",
            }
            self.assertTrue(expected.issubset({path.name for path in run_dir.iterdir()}))
            summary = json.loads(
                (run_dir / "gate_b_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["mode"], "smoke-test")
            self.assertFalse(summary["overall_pass"])
            self.assertEqual(summary["selected_ids"], ["one"])
            self.assertTrue(summary["router_history"])
            self.assertTrue(summary["detail_history"])
            self.assertGreaterEqual(summary["router_steps"], 1)
            self.assertGreaterEqual(summary["detail_steps"], 1)
            self.assertEqual(
                summary["thresholds"],
                {"n_iou": 0.9, "coverage_at_24": 1.0, "coverage_at_8": 0.95},
            )
            self.assertEqual(summary["resume_provenance"]["router_initialization"], "fresh")
            self.assertEqual(summary["resume_provenance"]["detail_initialization"], "fresh")

            checkpoint = torch.load(
                run_dir / "gaussamr_v1_gate_b.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                set(checkpoint), {"router", "detail_refiner", "config", "best"}
            )

            model = GaussAMRV1.from_probe_checkpoint(
                run_dir / "gaussamr_v1_gate_b.pt"
            )
            output = model(torch.zeros(1, 1, 65, 70))
            self.assertEqual(output["logits"].shape, (1, 1, 65, 70))

            verification = subprocess.run(
                [
                    sys.executable,
                    "train_gaussamr_gate_b.py",
                    "--data-root",
                    str(root),
                    "--subset-size",
                    "1",
                    "--device",
                    "cpu",
                    "--verify-only",
                    "--checkpoint",
                    str(run_dir / "gaussamr_v1_gate_b.pt"),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            verification_summary = json.loads(verification.stdout)
            self.assertEqual(verification_summary["mode"], "verify-only")
            self.assertNotIn("finite_checks", verification_summary)
            self.assertEqual(
                verification.returncode,
                0 if verification_summary["overall_pass"] else 1,
            )


if __name__ == "__main__":
    unittest.main()
