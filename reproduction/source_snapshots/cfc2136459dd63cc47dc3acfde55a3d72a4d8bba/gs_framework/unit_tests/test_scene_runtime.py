import struct
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch

from block_manager import build_partitioned_checkpoint
from scene_runtime import FrameCamera, GsplatRenderer, RenderSettings, SceneLoader
from scene_runtime.scenes import compute_scene_bounds_from_means
from scene_runtime.types import ActiveSplats


def _fake_splats(offset: float = 0.0):
    means = torch.tensor(
        [
            [-1.0 + offset, -1.0, 1.0],
            [1.0 + offset, -1.0, 1.2],
            [-1.0 + offset, 1.0, 1.4],
            [1.0 + offset, 1.0, 1.6],
            [0.0 + offset, 0.0, 1.8],
        ],
        dtype=torch.float32,
    )
    n = means.shape[0]
    return {
        "means": means,
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(n, 1),
        "scales": torch.zeros((n, 3), dtype=torch.float32),
        "opacities": torch.zeros((n,), dtype=torch.float32),
        "sh0": torch.zeros((n, 1, 3), dtype=torch.float32),
        "shN": torch.zeros((n, 15, 3), dtype=torch.float32),
    }


def _write_gaussian_ply(path: Path, rows, extra_properties=None, include_normals=True):
    extra_properties = extra_properties or []
    properties = [
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
    ]
    if include_normals:
        properties.extend([("nx", "float"), ("ny", "float"), ("nz", "float")])
    properties.extend(
        [
            ("f_dc_0", "float"),
            ("f_dc_1", "float"),
            ("f_dc_2", "float"),
        ]
    )
    properties.extend((f"f_rest_{idx}", "float") for idx in range(45))
    properties.extend(
        [
            ("opacity", "float"),
            ("scale_0", "float"),
            ("scale_1", "float"),
            ("scale_2", "float"),
            ("rot_0", "float"),
            ("rot_1", "float"),
            ("rot_2", "float"),
            ("rot_3", "float"),
        ]
    )
    properties.extend((name, "float") for name in extra_properties)

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(rows)}",
    ]
    header.extend(f"property {prop_type} {name}" for name, prop_type in properties)
    header.append("end_header")

    with path.open("wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        for row in rows:
            values = []
            for name, _ in properties:
                values.append(float(row.get(name, 0.0)))
            f.write(struct.pack("<" + "f" * len(values), *values))


def _camera(width=32, height=24):
    return FrameCamera(
        c2w=torch.eye(4),
        K=torch.tensor(
            [
                [20.0, 0.0, width * 0.5],
                [0.0, 20.0, height * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        width=width,
        height=height,
    )


class SceneLoaderTest(unittest.TestCase):
    def test_large_flat_bounds_do_not_use_torch_quantile(self):
        rows = (2**24 // 3) + 2
        means = torch.empty((rows, 3), dtype=torch.float32)
        values = torch.arange(rows, dtype=torch.float32)
        means[:, 0] = values
        means[:, 1] = -values
        means[:, 2] = 1.0

        with patch("torch.quantile", side_effect=AssertionError("quantile should not be called")):
            bounds = compute_scene_bounds_from_means(means)

        self.assertEqual(bounds["finite_count"], rows)
        self.assertEqual(bounds["nonfinite_count"], 0)
        self.assertEqual(bounds["scene_min"], (0.0, -float(rows - 1), 1.0))
        self.assertEqual(bounds["scene_max"], (float(rows - 1), -0.0, 1.0))

    def test_bounds_are_finite_ordered_and_deterministic(self):
        means = torch.tensor(
            [[-2.0, 0.0, 1.0], [4.0, 3.0, 2.0], [1.0, -1.0, 5.0]],
            dtype=torch.float32,
        )

        first = compute_scene_bounds_from_means(means)
        second = compute_scene_bounds_from_means(means)

        self.assertEqual(first, second)
        for lower, upper in zip(first["scene_min"], first["scene_max"]):
            self.assertTrue(torch.isfinite(torch.tensor(lower)))
            self.assertTrue(torch.isfinite(torch.tensor(upper)))
            self.assertLessEqual(lower, upper)
        for lower, upper in zip(first["view_min"], first["view_max"]):
            self.assertLess(lower, upper)

    def test_nonfinite_positions_are_reported_without_changing_tensors(self):
        means = torch.tensor(
            [[0.0, 0.0, 0.0], [float("nan"), 1.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )
        original = means.clone()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bounds = compute_scene_bounds_from_means(means)

        self.assertTrue(torch.allclose(means, original, equal_nan=True))
        self.assertEqual(bounds["finite_count"], 2)
        self.assertEqual(bounds["nonfinite_count"], 1)
        self.assertEqual(bounds["scene_min"], (0.0, 0.0, 0.0))
        self.assertEqual(bounds["scene_max"], (1.0, 1.0, 1.0))
        self.assertTrue(any("Non-finite Gaussian positions" in str(w.message) for w in caught))

    def test_all_nonfinite_positions_fail_bounds_explicitly(self):
        means = torch.full((4, 3), float("nan"), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "no finite Gaussian positions"):
            compute_scene_bounds_from_means(means)

    def test_flat_scene_concatenates_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p0 = Path(tmpdir) / "ckpt0.pt"
            p1 = Path(tmpdir) / "ckpt1.pt"
            torch.save({"step": 1, "splats": _fake_splats(0.0)}, p0)
            torch.save({"step": 2, "splats": _fake_splats(10.0)}, p1)

            scene = SceneLoader.load([p0, p1], torch.device("cpu"), scene_type="flat")
            stats = scene.get_scene_stats()
            active = scene.prepare_frame(
                _camera(),
                RenderSettings(width=32, height=24, near_plane=0.1, far_plane=10.0),
            )

            self.assertEqual(stats.scene_type, "flat")
            self.assertEqual(stats.total_gaussians, 10)
            self.assertEqual(active.num_gaussians, 10)
            self.assertEqual(active.scene_metrics.active_gaussians, 10)
            self.assertEqual(stats.bounds_finite_gaussian_count, 10)
            self.assertEqual(stats.bounds_nonfinite_gaussian_count, 0)
            self.assertEqual(stats.steps, [1, 2])

    def test_flat_scene_loads_gaussian_ply(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ply = Path(tmpdir) / "splats.ply"
            _write_gaussian_ply(
                ply,
                [
                    {
                        "x": 1.0,
                        "y": 2.0,
                        "z": 3.0,
                        "opacity": -2.0,
                        "scale_0": -1.0,
                        "scale_1": -2.0,
                        "scale_2": -3.0,
                        "rot_0": 1.0,
                        "f_dc_0": 0.1,
                        "f_dc_1": 0.2,
                        "f_dc_2": 0.3,
                        "f_rest_1": 0.11,
                        "f_rest_15": 0.5,
                        "f_rest_30": 0.75,
                        "f_rest_44": 0.44,
                    }
                ],
            )

            scene = SceneLoader.load([ply], torch.device("cpu"), scene_type="auto")
            stats = scene.get_scene_stats()
            active = scene.prepare_frame(
                _camera(),
                RenderSettings(width=32, height=24, near_plane=0.1, far_plane=10.0),
            )

            self.assertEqual(stats.scene_type, "flat")
            self.assertEqual(stats.total_gaussians, 1)
            self.assertEqual(stats.sh_degree, 3)
            self.assertEqual(stats.steps, [-1])
            self.assertTrue(torch.allclose(active.means[0], torch.tensor([1.0, 2.0, 3.0])))
            self.assertTrue(torch.allclose(active.quats[0], torch.tensor([1.0, 0.0, 0.0, 0.0])))
            self.assertTrue(
                torch.allclose(active.scales[0], torch.exp(torch.tensor([-1.0, -2.0, -3.0])))
            )
            self.assertAlmostEqual(float(active.opacities[0]), float(torch.sigmoid(torch.tensor(-2.0))))
            self.assertEqual(tuple(active.colors.shape), (1, 16, 3))
            self.assertAlmostEqual(float(active.colors[0, 0, 0]), 0.1, places=6)
            self.assertAlmostEqual(float(active.colors[0, 0, 1]), 0.2, places=6)
            self.assertAlmostEqual(float(active.colors[0, 0, 2]), 0.3, places=6)
            self.assertAlmostEqual(float(active.colors[0, 1, 1]), 0.5, places=6)
            self.assertAlmostEqual(float(active.colors[0, 1, 2]), 0.75, places=6)
            self.assertAlmostEqual(float(active.colors[0, 2, 0]), 0.11, places=6)
            self.assertAlmostEqual(float(active.colors[0, 15, 2]), 0.44, places=6)

            supersplat_scene = SceneLoader.load(
                [ply],
                torch.device("cpu"),
                scene_type="auto",
                ply_coordinate_mode="supersplat",
            )
            supersplat_active = supersplat_scene.prepare_frame(
                _camera(),
                RenderSettings(width=32, height=24, near_plane=0.1, far_plane=10.0),
            )
            self.assertTrue(
                torch.allclose(supersplat_active.means[0], torch.tensor([-1.0, -2.0, 3.0]))
            )
            self.assertTrue(
                torch.allclose(supersplat_active.quats[0], torch.tensor([0.0, 0.0, 0.0, 1.0]))
            )

    def test_gaussian_ply_warns_about_unexpected_properties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ply = Path(tmpdir) / "splats.ply"
            _write_gaussian_ply(
                ply,
                [{"rot_0": 1.0}],
                extra_properties=["unexpected_field"],
                include_normals=False,
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                SceneLoader.load([ply], torch.device("cpu"), scene_type="flat")

            messages = [str(warning.message) for warning in caught]
            self.assertTrue(any("unsupported PLY properties" in msg for msg in messages))
            self.assertTrue(any("no complete nx/ny/nz normal fields" in msg for msg in messages))

    def test_blocked_scene_loads_single_path_and_rejects_multiple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "ckpt.pt"
            blocked0 = Path(tmpdir) / "blocked0.pt"
            blocked1 = Path(tmpdir) / "blocked1.pt"
            torch.save({"step": 1, "splats": _fake_splats()}, ckpt)
            build_partitioned_checkpoint([ckpt], blocked0, grid=(2, 2), overwrite=False)
            build_partitioned_checkpoint([ckpt], blocked1, grid=(2, 2), overwrite=False)

            scene = SceneLoader.load([blocked0], torch.device("cpu"), scene_type="blocked")
            scene.set_culling_mode("force_all_blocks_active")
            active = scene.prepare_frame(
                _camera(),
                RenderSettings(width=32, height=24, near_plane=0.1, far_plane=10.0),
            )

            self.assertEqual(scene.get_scene_stats().scene_type, "blocked")
            self.assertEqual(active.num_gaussians, 5)
            with self.assertRaisesRegex(ValueError, "exactly one blocked scene path"):
                SceneLoader.load([blocked0, blocked1], torch.device("cpu"), scene_type="auto")

    def test_mixed_or_invalid_inputs_fail_early(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "ckpt.pt"
            blocked = Path(tmpdir) / "blocked.pt"
            invalid = Path(tmpdir) / "invalid.pt"
            torch.save({"step": 1, "splats": _fake_splats()}, ckpt)
            build_partitioned_checkpoint([ckpt], blocked, grid=(2, 2), overwrite=False)
            torch.save({"splats": {"means": torch.zeros((1, 3))}}, invalid)

            with self.assertRaisesRegex(ValueError, "Mixed scene inputs"):
                SceneLoader.load([ckpt, blocked], torch.device("cpu"), scene_type="auto")
            with self.assertRaisesRegex(ValueError, "neither a strict flat checkpoint"):
                SceneLoader.load([invalid], torch.device("cpu"), scene_type="auto")


class GsplatRendererTest(unittest.TestCase):
    def test_empty_active_splats_returns_background_without_rasterization(self):
        renderer = GsplatRenderer(torch.device("cpu"))
        active = ActiveSplats(
            means=torch.empty((0, 3)),
            quats=torch.empty((0, 4)),
            scales=torch.empty((0, 3)),
            opacities=torch.empty((0,)),
            colors=torch.empty((0, 16, 3)),
        )
        settings = RenderSettings(
            width=4,
            height=3,
            near_plane=0.1,
            far_plane=10.0,
            background=(0.25, 0.5, 0.75),
        )
        result = renderer.render(active, _camera(width=4, height=3), settings)

        self.assertEqual(tuple(result.image.shape), (3, 4, 3))
        self.assertEqual(result.visible_gaussians, 0.0)
        self.assertTrue(torch.allclose(result.image[0, 0], torch.tensor([0.25, 0.5, 0.75])))


if __name__ == "__main__":
    unittest.main()
