import json
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from scene_runtime.citygaussian import (
    CityGaussianCell,
    CityGaussianVisibilityScene,
    bounds_to_corners,
    citygaussian_in_frustum,
    citygaussian_official_camera_from_c2w,
    citygaussian_official_camera_from_frame,
)
from scene_runtime.types import ActiveSplats, FrameCamera, RenderSettings
from scripts.extract_citygaussian_cell_oracle import _camera_to_official


def _active_splats(count: int) -> ActiveSplats:
    return ActiveSplats(
        means=torch.arange(count * 3, dtype=torch.float32).reshape(count, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(count, 1),
        scales=torch.ones((count, 3), dtype=torch.float32),
        opacities=torch.ones((count,), dtype=torch.float32),
        colors=torch.arange(count * 9 * 3, dtype=torch.float32).reshape(count, 9, 3),
    )


def _corners(mn, mx) -> torch.Tensor:
    return bounds_to_corners(torch.tensor(mn, dtype=torch.float32), torch.tensor(mx, dtype=torch.float32))


def _official_camera() -> object:
    K = torch.tensor([[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return citygaussian_official_camera_from_c2w(torch.eye(4), K, width=100, height=100)


def _frame_camera() -> FrameCamera:
    return FrameCamera(
        c2w=torch.eye(4, dtype=torch.float32),
        K=torch.tensor([[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        width=100,
        height=100,
    )


def _settings() -> RenderSettings:
    return RenderSettings(width=100, height=100, near_plane=0.01, far_plane=100.0, sh_degree=2)


class CityGaussianVisibilityRuntimeTest(unittest.TestCase):
    def test_official_projected_corner_visibility_and_partial_intersection(self):
        corners = torch.stack(
            [
                _corners([-0.2, -0.2, 2.0], [0.2, 0.2, 2.2]),
                _corners([1.5, -0.2, 2.0], [3.0, 0.2, 2.2]),
            ],
            dim=0,
        )

        selection = citygaussian_in_frustum(
            _official_camera(),
            corners,
            [3, 5],
            aabb=[0, 0, 0, 1, 1, 1],
            block_dim=[2, 1, 1],
        )

        self.assertEqual(selection.selected_cell_ids, [0, 1])
        self.assertEqual(selection.selected_gaussian_count, 8)
        self.assertGreater(selection.per_visible_cell_projected_boxes[1][2], 1.0)
        self.assertFalse(selection.nearest_cell_fallback_changed_mask)

    def test_completely_offscreen_cell_is_not_selected_when_not_nearest(self):
        corners = torch.stack(
            [
                _corners([-0.2, -0.2, 2.0], [0.2, 0.2, 2.2]),
                _corners([20.0, 20.0, 8.0], [21.0, 21.0, 9.0]),
            ],
            dim=0,
        )

        selection = citygaussian_in_frustum(
            _official_camera(),
            corners,
            [3, 5],
            aabb=[0, 0, 0, 1, 1, 1],
            block_dim=[2, 1, 1],
        )

        self.assertEqual(selection.selected_cell_ids, [0])
        self.assertEqual(selection.selected_gaussian_count, 3)

    def test_front_plane_corner_handling_and_nearest_fallback(self):
        corners = torch.stack(
            [
                _corners([-0.1, -0.1, -3.0], [0.1, 0.1, -2.8]),
                _corners([-0.1, -0.1, -2.0], [0.1, 0.1, -1.8]),
            ],
            dim=0,
        )

        selection = citygaussian_in_frustum(
            _official_camera(),
            corners,
            [5, 7],
            aabb=[0, 0, 0, 1, 1, 1],
            block_dim=[2, 1, 1],
        )

        self.assertEqual(selection.selected_cell_ids, [1])
        self.assertEqual(selection.nearest_cell_id, 1)
        self.assertTrue(selection.nearest_cell_fallback_changed_mask)
        self.assertEqual(selection.per_cell_valid_front_corner_counts, [0, 0])

    def test_camera_conversion_matches_oracle_helper(self):
        source_c2w = np.eye(4, dtype=np.float32)
        source_c2w[:3, 3] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        oracle = _camera_to_official(
            trajectory_index=0,
            camera_id="fixture",
            source_c2w=source_c2w,
            width=100,
            height=80,
            fov_deg=50.0,
            scale_ratio=10.0,
            znear=0.01,
            zfar=100.0,
        )
        focal = 0.5 * 80.0 / math.tan(math.radians(50.0) * 0.5)
        runtime = citygaussian_official_camera_from_c2w(
            torch.from_numpy(oracle.converted_camera_to_world),
            torch.tensor([[focal, 0.0, 50.0], [0.0, focal, 40.0], [0.0, 0.0, 1.0]]),
            width=100,
            height=80,
            znear=0.01,
            zfar=100.0,
        )

        self.assertTrue(torch.allclose(runtime.world_view_transform, oracle.world_view_transform, atol=1e-5))
        self.assertTrue(torch.allclose(runtime.full_proj_transform, oracle.full_proj_transform, atol=1e-5))

    def test_visibility_scene_counts_residency_and_timings(self):
        active = _active_splats(3)
        scene = object.__new__(CityGaussianVisibilityScene)
        scene.path = "fixture"
        scene.device = torch.device("cpu")
        scene.block_dim = (2, 1, 1)
        scene.aabb = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        scene.lod_source_level = "fixture"
        scene.num_cells = 2
        scene.total_gaussians = 3
        scene.flat_active = active
        scene.cell_counts = [2, 1]
        scene.cell_ids = [0, 1]
        scene.cell_mad_bounds = [
            {"min": (-0.2, -0.2, 2.0), "max": (0.2, 0.2, 2.2)},
            {"min": (20.0, 20.0, 8.0), "max": (21.0, 21.0, 9.0)},
        ]
        scene.cell_corners = torch.stack(
            [
                _corners([-0.2, -0.2, 2.0], [0.2, 0.2, 2.2]),
                _corners([20.0, 20.0, 8.0], [21.0, 21.0, 9.0]),
            ],
            dim=0,
        )
        scene.cell_source_indices = [torch.tensor([0, 1], dtype=torch.long), torch.tensor([2], dtype=torch.long)]
        scene.cell_construction_ms = 0.5
        scene.visibility_znear = 0.01
        scene.visibility_zfar = 100.0

        prepared = scene.prepare_frame(_frame_camera(), _settings())
        metrics = prepared.scene_metrics

        self.assertEqual(prepared.num_gaussians, 2)
        self.assertEqual(metrics.resident_gaussians, 3)
        self.assertEqual(metrics.selected_gaussian_count, 2)
        self.assertEqual(metrics.active_gaussians, 2)
        self.assertEqual(metrics.selected_cell_ids, "0")
        self.assertEqual(metrics.selected_cell_count, 1)
        self.assertGreaterEqual(metrics.visibility_selection_ms, 0.0)
        self.assertGreaterEqual(metrics.selected_cell_assembly_ms, 0.0)
        self.assertFalse(metrics.citygaussian_no_visibility_selection)
        self.assertTrue(metrics.citygaussian_visibility_selection_used)
        self.assertTrue(metrics.citygaussian_no_adaptive_lod)
        self.assertTrue(metrics.citygaussian_no_streaming)
        json.dumps(
            {
                "selected_cell_ids": metrics.selected_cell_ids,
                "selected_gaussian_count": metrics.selected_gaussian_count,
                "per_cell_bounds": metrics.per_cell_bounds,
                "per_cell_distances": metrics.per_cell_distances,
                "nearest_cell_id": metrics.nearest_cell_id,
            },
            allow_nan=False,
        )

    def test_frame_camera_conversion_rejects_malformed_matrix(self):
        bad = _frame_camera()
        bad.c2w[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            citygaussian_official_camera_from_frame(bad, _settings())


class CityGaussianVisibilityFixtureLoadingTest(unittest.TestCase):
    def _load_fixture(self, name: str) -> dict:
        path = Path(__file__).resolve().parent / "fixtures" / "citygaussian_visibility" / name
        return json.loads(path.read_text())

    def test_finest_and_fused_oracle_fixtures_load(self):
        finest = self._load_fixture("finest_visibility_oracle_v1.json")
        fused = self._load_fixture("fused_visibility_oracle_v1.json")

        self.assertEqual(finest["asset"]["gaussian_count"], 11694810)
        self.assertEqual(fused["asset"]["gaussian_count"], 23389621)
        self.assertEqual(len(finest["visibility_records"]), 12)
        self.assertEqual(len(fused["visibility_records"]), 12)
        self.assertTrue(any(row["nearest_cell_fallback_changed_mask"] for row in finest["visibility_records"]))
        self.assertTrue(any(row["nearest_cell_fallback_changed_mask"] for row in fused["visibility_records"]))
        self.assertEqual(min(row["visible_cell_count"] for row in finest["visibility_records"]), 9)
        self.assertEqual(min(row["visible_cell_count"] for row in fused["visibility_records"]), 10)


if __name__ == "__main__":
    unittest.main()
