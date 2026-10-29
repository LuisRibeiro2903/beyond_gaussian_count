import tempfile
import unittest
from pathlib import Path

import torch

from baseline_validation import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DETERMINISM_ATOL,
    FIXTURE_SCENE_ID,
    environment_report,
    fixed_validation_frames,
    render_validation,
    structural_validation,
    tiny_fixture_splats,
    write_tiny_flat_fixture,
)
from scene_runtime import GsplatRenderer, SceneLoader


class TinyFlatFixtureTest(unittest.TestCase):
    def test_fixture_is_small_deterministic_and_loads_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = write_tiny_flat_fixture(Path(tmpdir) / "tiny_flat.pt")
            payload = torch.load(fixture_path, map_location="cpu", weights_only=False)
            expected = tiny_fixture_splats()

            self.assertEqual(payload["fixture"]["scene_id"], FIXTURE_SCENE_ID)
            self.assertLess(fixture_path.stat().st_size, 64 * 1024)
            for key, expected_tensor in expected.items():
                self.assertTrue(torch.equal(payload["splats"][key], expected_tensor))

            scene = SceneLoader.load([fixture_path], torch.device("cpu"), scene_type="flat")
            active = scene.prepare_frame(
                fixed_validation_frames()[0].camera,
                fixed_validation_frames()[0].settings,
            )

            self.assertEqual(scene.get_scene_stats().scene_type, "flat")
            self.assertEqual(active.num_gaussians, expected["means"].shape[0])
            self.assertEqual(active.means.dtype, torch.float32)
            self.assertEqual(active.means.device.type, "cpu")
            self.assertEqual(tuple(active.colors.shape), (expected["means"].shape[0], 16, 3))

    def test_fixed_validation_frames_include_two_views(self):
        frames = fixed_validation_frames()

        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].camera_id, "front_identity")
        self.assertEqual(frames[1].camera_id, "offset_x")
        self.assertEqual(frames[0].settings.width, DEFAULT_WIDTH)
        self.assertEqual(frames[0].settings.height, DEFAULT_HEIGHT)
        self.assertFalse(torch.equal(frames[0].camera.c2w, frames[1].camera.c2w))


class BaselineStructuralValidationTest(unittest.TestCase):
    def test_runtime_imports_fixture_loads_and_renderer_constructs_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = write_tiny_flat_fixture(Path(tmpdir) / "tiny_flat.pt")
            report = structural_validation(fixture_path, device=torch.device("cpu"))

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["checks"]["scene_type_is_flat"])
        self.assertTrue(report["checks"]["gaussian_count"])
        self.assertTrue(report["checks"]["all_float32"])
        self.assertTrue(report["checks"]["device"])
        self.assertTrue(report["checks"]["renderer_constructed"])
        self.assertIn("simple_viewer", report["imports"])
        self.assertIn("scene_runtime.renderer", report["imports"])

    def test_environment_report_includes_core_versions(self):
        report = environment_report()

        self.assertIn("version", report["python"])
        self.assertIn("version", report["torch"])
        self.assertIn("cuda_available", report["torch"])
        self.assertIn("version", report["gsplat"])
        self.assertIn("module_path", report["gsplat"])

    def test_renderer_construction_does_not_require_block_or_lod_path(self):
        renderer = GsplatRenderer(torch.device("cpu"))

        self.assertIsInstance(renderer, GsplatRenderer)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for gsplat rasterization smoke.")
class BaselineCudaRenderValidationTest(unittest.TestCase):
    def test_tiny_flat_scene_renders_deterministically_on_cuda(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = write_tiny_flat_fixture(Path(tmpdir) / "tiny_flat.pt")
            report = render_validation(fixture_path, device=torch.device("cuda", 0))

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["frame"]["image_shape"], [DEFAULT_HEIGHT, DEFAULT_WIDTH, 3])
        self.assertEqual(report["frame"]["alpha_shape"], [DEFAULT_HEIGHT, DEFAULT_WIDTH, 1])
        self.assertTrue(report["frame"]["finite_image"])
        self.assertTrue(report["frame"]["finite_alpha"])
        self.assertLessEqual(
            report["determinism"]["image_max_abs_diff"],
            DETERMINISM_ATOL,
        )
        self.assertEqual(report["metrics"]["active_gaussian_count"], 5)
        self.assertGreaterEqual(report["metrics"]["rendered_gaussian_count"], 0.0)
        self.assertLessEqual(report["metrics"]["rendered_gaussian_count"], 5.0)
        self.assertEqual(report["result_schema"]["record_kind"], "validation_smoke")
        self.assertEqual(report["result_schema"]["scene_identifier"], FIXTURE_SCENE_ID)


if __name__ == "__main__":
    unittest.main()
