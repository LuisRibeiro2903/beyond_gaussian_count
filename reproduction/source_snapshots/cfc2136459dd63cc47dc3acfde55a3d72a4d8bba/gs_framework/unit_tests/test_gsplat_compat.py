import importlib
import unittest
from unittest.mock import patch

import torch

from scene_runtime import FrameCamera, GsplatRenderer, RenderSettings
from scene_runtime.gsplat_compat import SUPPORTED_RASTERIZE_MODES, normalize_rasterize_mode
from scene_runtime.types import ActiveSplats


def _active_splats():
    return ActiveSplats(
        means=torch.zeros((1, 3), dtype=torch.float32),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        scales=torch.ones((1, 3), dtype=torch.float32) * 0.01,
        opacities=torch.ones((1,), dtype=torch.float32),
        colors=torch.zeros((1, 1, 3), dtype=torch.float32),
    )


def _camera(width=4, height=3):
    return FrameCamera(
        c2w=torch.eye(4),
        K=torch.tensor(
            [
                [4.0, 0.0, width * 0.5],
                [0.0, 4.0, height * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        width=width,
        height=height,
    )


class GsplatCompatibilityTest(unittest.TestCase):
    def test_installed_gsplat_and_viewer_modules_import(self):
        importlib.import_module("gsplat")
        importlib.import_module("gsplat_viewer")
        importlib.import_module("simple_viewer")

    def test_supported_rasterize_modes_are_forwarded_to_gsplat(self):
        renderer = GsplatRenderer(torch.device("cpu"))
        active_splats = _active_splats()
        camera = _camera()

        for rasterize_mode in SUPPORTED_RASTERIZE_MODES:
            with self.subTest(rasterize_mode=rasterize_mode):
                calls = []

                def fake_rasterization(**kwargs):
                    calls.append(kwargs)
                    height = int(kwargs["height"])
                    width = int(kwargs["width"])
                    image = torch.zeros((1, height, width, 3), dtype=torch.float32)
                    alpha = torch.ones((1, height, width, 1), dtype=torch.float32)
                    info = {"radii": torch.ones((1, active_splats.num_gaussians))}
                    return image, alpha, info

                settings = RenderSettings(
                    width=camera.width,
                    height=camera.height,
                    near_plane=0.1,
                    far_plane=10.0,
                    rasterize_mode=rasterize_mode,
                )

                with patch("scene_runtime.renderer.rasterization", fake_rasterization):
                    result = renderer.render(active_splats, camera, settings)

                self.assertEqual(calls[0]["rasterize_mode"], rasterize_mode)
                self.assertEqual(tuple(result.image.shape), (camera.height, camera.width, 3))
                self.assertEqual(result.visible_gaussians, 1.0)

    def test_invalid_rasterize_mode_fails_before_rasterization(self):
        renderer = GsplatRenderer(torch.device("cpu"))
        settings = RenderSettings(
            width=4,
            height=3,
            near_plane=0.1,
            far_plane=10.0,
            rasterize_mode="unsupported",  # type: ignore[arg-type]
        )

        with patch("scene_runtime.renderer.rasterization") as rasterization_mock:
            with self.assertRaisesRegex(ValueError, "Unsupported gsplat rasterize_mode"):
                renderer.render(_active_splats(), _camera(), settings)

        rasterization_mock.assert_not_called()

    def test_normalize_rasterize_mode_documents_supported_values(self):
        self.assertEqual(normalize_rasterize_mode("classic"), "classic")
        self.assertEqual(normalize_rasterize_mode("antialiased"), "antialiased")
        with self.assertRaisesRegex(ValueError, "classic, antialiased"):
            normalize_rasterize_mode("edl")


if __name__ == "__main__":
    unittest.main()
