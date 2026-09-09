import tempfile
import unittest
from pathlib import Path

import irstd_gaussamr
import torch
import torch.nn.functional as F

from irstd_gaussamr.refiners import (
    detail_crops,
    select_detail_proposals,
)
from irstd_gaussamr.router_probe import decode_proposals


class GaussAMRV1PublicApiTest(unittest.TestCase):
    def test_package_exposes_packaged_inference_model(self):
        self.assertTrue(hasattr(irstd_gaussamr, "GaussAMRV1"))

    def test_model_exposes_probe_checkpoint_constructor(self):
        self.assertTrue(hasattr(irstd_gaussamr.GaussAMRV1, "from_probe_checkpoint"))


class GaussAMRV1ForwardTest(unittest.TestCase):
    def test_predicted_forward_preserves_arbitrary_input_size_and_contract(self):
        model = irstd_gaussamr.GaussAMRV1()
        image = torch.randn(2, 1, 65, 70)

        output = model(image)

        self.assertEqual(
            set(output),
            {
                "logits",
                "gaussian_logits",
                "router_maps",
                "proposals_l1",
                "proposals_l2",
                "local_logits",
            },
        )
        self.assertEqual(output["logits"].shape, (2, 1, 65, 70))
        self.assertEqual(output["gaussian_logits"].shape, (2, 1, 65, 70))
        self.assertEqual(output["proposals_l1"].shape, (2, 24, 6))
        self.assertEqual(output["proposals_l2"].shape, (2, 8, 6))
        self.assertEqual(output["local_logits"].shape, (2, 8, 1, 48, 48))
        self.assertEqual(output["router_maps"]["objectness_logit"].shape, (2, 1, 9, 9))

    def test_predicted_forward_rejects_targets(self):
        model = irstd_gaussamr.GaussAMRV1()

        with self.assertRaisesRegex(ValueError, "targets"):
            model(torch.randn(1, 1, 32, 32), targets=torch.zeros(1, 1, 32, 32))

    def test_unimplemented_routing_mode_is_rejected(self):
        model = irstd_gaussamr.GaussAMRV1()

        with self.assertRaisesRegex(NotImplementedError, "oracle"):
            model(
                torch.randn(1, 1, 32, 32),
                routing_mode="oracle",
                targets=torch.zeros(1, 1, 32, 32),
            )

    def test_zero_initialized_detail_path_preserves_gaussian_logits(self):
        model = irstd_gaussamr.GaussAMRV1().eval()

        with torch.no_grad():
            output = model(torch.randn(1, 1, 33, 39))

        torch.testing.assert_close(output["logits"], output["gaussian_logits"])

    def test_wrapper_matches_explicit_passing_probe_pipeline(self):
        model = irstd_gaussamr.GaussAMRV1().eval()
        with torch.no_grad():
            model.detail_refiner.output.bias.fill_(0.25)
        image = torch.randn(1, 1, 35, 37)
        padded = F.pad(image, (0, 3, 0, 5))

        with torch.no_grad():
            features = model.feature_bank(padded)
            router_maps = model.router(features)
            proposals_l1 = decode_proposals(router_maps, k=24)
            proposals_l2, indices_l2 = select_detail_proposals(proposals_l1, k=8)
            residual = model.detail_refiner(features, proposals_l2)
            full_residual = residual.new_zeros(1, 24, 1, 48, 48).scatter(
                1,
                indices_l2[..., None, None, None].expand_as(residual),
                residual,
            )
            expected = model.composer(proposals_l1, (40, 40), full_residual)[..., :35, :37]
            actual = model(image)["logits"]

        torch.testing.assert_close(actual, expected)

    def test_end_to_end_outputs_and_gradients_are_finite(self):
        model = irstd_gaussamr.GaussAMRV1()
        image = torch.randn(1, 1, 33, 39, requires_grad=True)

        output = model(image)
        loss = output["logits"].square().mean() + output["local_logits"].square().mean()
        loss.backward()

        self.assertTrue(torch.isfinite(output["logits"]).all())
        self.assertTrue(torch.isfinite(output["gaussian_logits"]).all())
        self.assertTrue(torch.isfinite(output["local_logits"]).all())
        self.assertIsNotNone(image.grad)
        self.assertTrue(torch.isfinite(image.grad).all())
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


class GaussAMRV1CheckpointTest(unittest.TestCase):
    def test_probe_checkpoint_reproduces_router_and_detail_output(self):
        torch.manual_seed(7)
        source = irstd_gaussamr.GaussAMRV1().eval()
        with torch.no_grad():
            source.detail_refiner.output.bias.fill_(0.25)
        image = torch.randn(1, 1, 35, 37)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "detail_best.pt"
            torch.save(
                {
                    "router": source.router.state_dict(),
                    "detail_refiner": source.detail_refiner.state_dict(),
                    "config": {"k1": 24, "k2_diagnostic": 8},
                    "best": {"detail_n_iou": 0.5},
                },
                checkpoint_path,
            )

            loaded = irstd_gaussamr.GaussAMRV1.from_probe_checkpoint(checkpoint_path).eval()

        with torch.no_grad():
            expected = source(image)
            actual = loaded(image)
        self.assertFalse(torch.equal(expected["logits"], expected["gaussian_logits"]))
        torch.testing.assert_close(actual["logits"], expected["logits"])
        torch.testing.assert_close(actual["proposals_l1"], expected["proposals_l1"])


class GaussAMRV1BorderGeometryTest(unittest.TestCase):
    def test_crop_then_paste_preserves_all_four_corner_coordinates(self):
        height, width = 31, 37
        corners = (
            (0, 0),
            (width - 1, 0),
            (0, height - 1),
            (width - 1, height - 1),
        )
        source = torch.zeros(4, 1, height, width)
        proposals = torch.zeros(4, 1, 6)
        for index, (x, y) in enumerate(corners):
            source[index, 0, y, x] = 1
            proposals[index, 0] = torch.tensor([0.99, x, y, 1.0, 1.0, 0.5])

        residual = 20 * detail_crops(source, proposals)
        pasted = irstd_gaussamr.SparseGaussianComposer()(
            proposals, (height, width), residual
        )

        for index, (x, y) in enumerate(corners):
            self.assertEqual(int(pasted[index, 0].argmax()), y * width + x)


if __name__ == "__main__":
    unittest.main()
