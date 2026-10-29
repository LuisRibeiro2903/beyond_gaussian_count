import inspect
import json
import unittest

import torch

from scene_runtime.citygaussian import (
    CityGaussianFixedAllScene,
    compare_flat_and_cell_attributes,
    construct_citygaussian_cells,
    contract_to_unisphere,
    official_cell_ids,
    official_cell_mask,
    official_mad_bounds,
    validate_citygaussian_cells,
)
from scene_runtime.scenes import _empty_scene_metrics
from scene_runtime.types import ActiveSplats, FrameCamera, RenderSettings


def _active_splats(means, scales=None):
    means = torch.tensor(means, dtype=torch.float32)
    n = means.shape[0]
    if scales is None:
        scales = torch.ones((n, 3), dtype=torch.float32)
    else:
        scales = torch.tensor(scales, dtype=torch.float32)
    return ActiveSplats(
        means=means,
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(n, 1),
        scales=scales,
        opacities=torch.ones((n,), dtype=torch.float32),
        colors=torch.arange(n * 9 * 3, dtype=torch.float32).reshape(n, 9, 3),
    )


def _camera():
    return FrameCamera(
        c2w=torch.eye(4),
        K=torch.tensor([[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]]),
        width=16,
        height=16,
    )


class CityGaussianRuntimeTest(unittest.TestCase):
    def test_flat_all_active_metrics_distinguish_resident_selected_and_active(self):
        metrics = _empty_scene_metrics(17)

        self.assertEqual(metrics.resident_gaussians, 17)
        self.assertEqual(metrics.selected_gaussian_count, 17)
        self.assertEqual(metrics.active_gaussians, 17)
        self.assertEqual(metrics.resident_gaussian_fraction, 1.0)
        self.assertEqual(metrics.active_gaussian_fraction, 1.0)

    def test_official_contraction_behavior(self):
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [4.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        contracted = contract_to_unisphere(xyz, aabb=[0, 0, 0, 2, 2, 2])

        self.assertTrue(torch.allclose(contracted[0], torch.tensor([0.25, 0.25, 0.25])))
        self.assertTrue(torch.allclose(contracted[1], torch.tensor([0.5, 0.5, 0.5])))
        self.assertTrue(torch.allclose(contracted[2], torch.tensor([0.75, 0.75, 0.75])))
        self.assertAlmostEqual(float(contracted[3, 0]), 0.9166666269302368)

    def test_boundary_points_follow_half_open_cell_order(self):
        xyz = torch.tensor(
            [
                [0.5, 0.5, 0.5],
                [1.0, 0.5, 0.5],
                [1.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        ids = official_cell_ids(xyz, block_dim=[2, 2, 1], aabb=[0, 0, 0, 2, 2, 1])

        self.assertEqual(ids.tolist(), [0, 1, 1])
        self.assertFalse(bool(official_cell_mask(0, xyz, block_dim=[2, 2, 1], aabb=[0, 0, 0, 2, 2, 1])[1]))
        self.assertTrue(bool(official_cell_mask(1, xyz, block_dim=[2, 2, 1], aabb=[0, 0, 0, 2, 2, 1])[1]))

    def test_every_gaussian_assigned_once_and_cell_order(self):
        active = _active_splats(
            [
                [0.5, 0.5, 0.5],
                [1.5, 0.5, 0.5],
                [0.5, 1.5, 0.5],
                [1.5, 1.5, 0.5],
            ]
        )
        cells = construct_citygaussian_cells(active, block_dim=[2, 2, 1], aabb=[0, 0, 0, 2, 2, 1])
        validation = validate_citygaussian_cells(active, cells)

        self.assertEqual([cell.cell_id for cell in cells], [0, 1, 2, 3])
        self.assertEqual([cell.gaussian_count for cell in cells], [1, 1, 1, 1])
        self.assertEqual(validation.assigned_total, 4)
        self.assertEqual(validation.unassigned_count, 0)
        self.assertEqual(validation.duplicated_assignment_count, 0)
        self.assertTrue(validation.every_gaussian_assigned_once)
        self.assertTrue(validation.all_cells_nonempty)

    def test_empty_cells_are_reported_without_dropping_gaussians(self):
        active = _active_splats([[0.5, 0.5, 0.5]])
        cells = construct_citygaussian_cells(active, block_dim=[2, 2, 1], aabb=[0, 0, 0, 2, 2, 1])
        validation = validate_citygaussian_cells(active, cells)

        self.assertEqual([cell.gaussian_count for cell in cells], [1, 0, 0, 0])
        self.assertEqual(validation.unassigned_count, 0)
        self.assertEqual(validation.duplicated_assignment_count, 0)
        self.assertFalse(validation.all_cells_nonempty)
        self.assertTrue(validation.every_gaussian_assigned_once)

    def test_mad_bounds_and_average_scale_match_official_formula(self):
        active = _active_splats(
            [[0.0, 0.5, 0.5], [1.0, 0.5, 0.5], [10.0, 0.5, 0.5]],
            scales=[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]],
        )
        xyz_min, xyz_max = official_mad_bounds(active.means, mad_multiplier=4.0)

        self.assertTrue(torch.allclose(xyz_min, torch.tensor([0.0, 0.5, 0.5])))
        self.assertTrue(torch.allclose(xyz_max, torch.tensor([5.0, 0.5, 0.5])))

        cells = construct_citygaussian_cells(active, block_dim=[1, 1, 1], aabb=[0, 0, 0, 2, 2, 1])
        self.assertAlmostEqual(cells[0].average_scale, 5.0)

    def test_all_cells_active_preparation_preserves_attributes_by_source_index(self):
        active = _active_splats(
            [
                [1.5, 1.5, 0.5],
                [0.5, 0.5, 0.5],
                [1.5, 0.5, 0.5],
                [0.5, 1.5, 0.5],
            ]
        )
        cells = construct_citygaussian_cells(active, block_dim=[2, 2, 1], aabb=[0, 0, 0, 2, 2, 1])
        validation = validate_citygaussian_cells(active, cells)
        source_indices = validation.concatenated_source_indices
        cell_active = ActiveSplats(
            means=active.means[source_indices],
            quats=active.quats[source_indices],
            scales=active.scales[source_indices],
            opacities=active.opacities[source_indices],
            colors=active.colors[source_indices],
        )

        self.assertTrue(compare_flat_and_cell_attributes(active, cell_active, source_indices))
        self.assertTrue(torch.equal(torch.sort(source_indices).values, torch.arange(4)))

    def test_scene_prepare_frame_disables_visibility_lod_and_streaming(self):
        active = _active_splats([[0.5, 0.5, 0.5]])
        cells = construct_citygaussian_cells(active, block_dim=[1, 1, 1], aabb=[0, 0, 0, 2, 2, 1])
        validation = validate_citygaussian_cells(active, cells)

        scene = object.__new__(CityGaussianFixedAllScene)
        scene.path = "fixture"
        scene.device = torch.device("cpu")
        scene.block_dim = (1, 1, 1)
        scene.aabb = (0.0, 0.0, 0.0, 2.0, 2.0, 1.0)
        scene.lod_source_level = "fixture_lod"
        scene.mad_multiplier = 4.0
        scene.num_cells = 1
        scene.total_gaussians = 1
        scene.flat_active = active
        scene.cells = cells
        scene.validation = validation
        scene.concatenated_source_indices = validation.concatenated_source_indices
        scene.cell_construction_ms = 0.25
        scene.cell_counts = [1]
        scene.cell_ids = [0]
        scene.cell_mad_bounds = [{"min": (0.5, 0.5, 0.5), "max": (0.5, 0.5, 0.5)}]

        prepared = scene.prepare_frame(
            _camera(),
            RenderSettings(width=16, height=16, near_plane=0.1, far_plane=10.0, sh_degree=2),
        )

        self.assertEqual(prepared.num_gaussians, 1)
        self.assertEqual(prepared.scene_metrics.active_cell_count, 1)
        self.assertEqual(prepared.scene_metrics.selected_gaussian_count, 1)
        self.assertTrue(prepared.scene_metrics.citygaussian_no_visibility_selection)
        self.assertTrue(prepared.scene_metrics.citygaussian_no_adaptive_lod)
        self.assertTrue(prepared.scene_metrics.citygaussian_no_streaming)

    def test_citygaussian_module_has_no_block_manager_dependency(self):
        import scene_runtime.citygaussian as module

        source = inspect.getsource(module)
        self.assertNotIn("from block_manager", source)
        self.assertNotIn("import block_manager", source)
        self.assertNotIn("host_residency", source)

    def test_cell_metadata_serializes(self):
        active = _active_splats([[0.5, 0.5, 0.5]])
        cells = construct_citygaussian_cells(active, block_dim=[1, 1, 1], aabb=[0, 0, 0, 2, 2, 1])
        validation = validate_citygaussian_cells(active, cells)

        payload = {
            "counts": [cell.gaussian_count for cell in cells],
            "assigned_total": validation.assigned_total,
            "unassigned_count": validation.unassigned_count,
            "duplicated_assignment_count": validation.duplicated_assignment_count,
            "average_scales": [cell.average_scale for cell in cells],
            "mad_bounds": [{"min": cell.mad_bounds_min, "max": cell.mad_bounds_max} for cell in cells],
        }

        self.assertIsInstance(json.dumps(payload, allow_nan=False), str)


if __name__ == "__main__":
    unittest.main()
