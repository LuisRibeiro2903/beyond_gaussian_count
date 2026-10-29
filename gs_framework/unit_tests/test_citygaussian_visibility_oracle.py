import math
import json
import unittest
import os
from pathlib import Path

import numpy as np
import torch

from scripts.extract_citygaussian_cell_oracle import (
    OfficialVisibilityFunction,
    OracleCamera,
    _camera_to_official,
    _load_in_frustum_source_slice,
    _projection_matrix,
    run_visibility_oracle,
)


def _visibility_function():
    configured = os.environ.get("GS_FRAMEWORK_CITYGAUSSIAN_REFERENCE_REPO")
    if not configured:
        raise unittest.SkipTest("Official-source oracle requires GS_FRAMEWORK_CITYGAUSSIAN_REFERENCE_REPO; see docs/REPRODUCING.md")
    source = Path(configured) / "utils" / "large_utils.py"
    namespace = _load_in_frustum_source_slice(source)
    return OfficialVisibilityFunction(
        fn=namespace["in_frustum"],
        method="source_slice_exec",
        source_path=str(source),
    )


def _camera(name="synthetic") -> OracleCamera:
    world_view = torch.eye(4, dtype=torch.float32)
    projection = _projection_matrix(0.01, 100.0, math.radians(90.0), math.radians(90.0)).T.contiguous()
    return OracleCamera(
        trajectory_index=0,
        camera_id=name,
        source_camera_to_world=np.eye(4, dtype=np.float32),
        converted_camera_to_world=np.eye(4, dtype=np.float32),
        world_view_transform=world_view,
        full_proj_transform=world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0),
        projection_matrix=projection,
        camera_center=torch.zeros(3, dtype=torch.float32),
        fov_x=math.radians(90.0),
        fov_y=math.radians(90.0),
        width=100,
        height=100,
        source_fov_degrees=90.0,
    )


def _bound(cell_id, mn, mx):
    return {"cell_id": cell_id, "min": list(mn), "max": list(mx)}


def _metadata(bounds, counts=None):
    if counts is None:
        counts = [10 for _ in bounds]
    return {
        "cell_count": len(bounds),
        "cell_gaussian_counts": counts,
        "mad_bounds": bounds,
    }


class CityGaussianVisibilityOracleSyntheticTest(unittest.TestCase):
    def test_all_cells_clearly_in_view_and_stable_ordering(self):
        records = run_visibility_oracle(
            visibility=_visibility_function(),
            cameras=[_camera()],
            cell_metadata=_metadata(
                [
                    _bound(0, [-0.3, -0.3, 2.0], [-0.1, -0.1, 2.2]),
                    _bound(1, [0.1, -0.3, 2.0], [0.3, -0.1, 2.2]),
                    _bound(2, [-0.3, 0.1, 2.0], [-0.1, 0.3, 2.2]),
                    _bound(3, [0.1, 0.1, 2.0], [0.3, 0.3, 2.2]),
                ],
                counts=[1, 2, 3, 4],
            ),
            block_dim=[2, 2, 1],
            aabb=[0, 0, 0, 1, 1, 1],
            precision=8,
        )

        row = records[0]
        self.assertEqual(row["visible_cell_ids"], [0, 1, 2, 3])
        self.assertEqual(row["visibility_mask"], [True, True, True, True])
        self.assertEqual(row["visible_cell_count"], 4)
        self.assertEqual(row["selected_gaussian_count"], 10)
        self.assertFalse(row["nearest_cell_fallback_changed_mask"])

    def test_no_projected_cells_uses_nearest_fallback(self):
        records = run_visibility_oracle(
            visibility=_visibility_function(),
            cameras=[_camera()],
            cell_metadata=_metadata(
                [
                    _bound(0, [-0.1, -0.1, -3.0], [0.1, 0.1, -2.8]),
                    _bound(1, [-0.1, -0.1, -2.0], [0.1, 0.1, -1.8]),
                ],
                counts=[5, 7],
            ),
            block_dim=[2, 1, 1],
            aabb=[0, 0, 0, 1, 1, 1],
            precision=8,
        )

        row = records[0]
        self.assertEqual(row["visible_cell_ids"], [1])
        self.assertEqual(row["selected_gaussian_count"], 7)
        self.assertEqual(row["nearest_cell_id"], 1)
        self.assertTrue(row["nearest_cell_fallback_changed_mask"])
        self.assertEqual(row["per_cell_valid_front_corner_count"], [0, 0])

    def test_cell_crossing_screen_boundary_is_visible(self):
        records = run_visibility_oracle(
            visibility=_visibility_function(),
            cameras=[_camera()],
            cell_metadata=_metadata([_bound(0, [1.5, -0.1, 2.0], [3.0, 0.1, 2.2])]),
            block_dim=[1, 1, 1],
            aabb=[0, 0, 0, 1, 1, 1],
            precision=8,
        )

        row = records[0]
        self.assertEqual(row["visible_cell_ids"], [0])
        self.assertGreater(row["per_visible_cell_projected_box"]["0"][2], 1.0)
        self.assertFalse(row["nearest_cell_fallback_changed_mask"])

    def test_corners_behind_front_threshold_do_not_contribute(self):
        records = run_visibility_oracle(
            visibility=_visibility_function(),
            cameras=[_camera()],
            cell_metadata=_metadata([_bound(0, [-0.1, -0.1, 0.1], [0.1, 0.1, 0.19])]),
            block_dim=[1, 1, 1],
            aabb=[0, 0, 0, 1, 1, 1],
            precision=8,
        )

        row = records[0]
        self.assertEqual(row["per_cell_valid_front_corner_count"], [0])
        self.assertEqual(row["visible_cell_ids"], [0])
        self.assertTrue(row["nearest_cell_fallback_changed_mask"])

    def test_distance_output_is_deterministic(self):
        kwargs = dict(
            visibility=_visibility_function(),
            cameras=[_camera()],
            cell_metadata=_metadata([_bound(0, [0.0, 0.0, 2.0], [1.0, 1.0, 3.0])]),
            block_dim=[1, 1, 1],
            aabb=[0, 0, 0, 1, 1, 1],
            precision=8,
        )
        first = run_visibility_oracle(**kwargs)[0]
        second = run_visibility_oracle(**kwargs)[0]

        self.assertEqual(first["per_visible_cell_camera_distance"], second["per_visible_cell_camera_distance"])
        self.assertAlmostEqual(first["per_visible_cell_camera_distance"]["0"], 2.0)

    def test_malformed_camera_matrix_fails(self):
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "non-finite"):
            _camera_to_official(
                trajectory_index=0,
                camera_id="bad",
                source_c2w=matrix,
                width=100,
                height=100,
                fov_deg=50.0,
                scale_ratio=1.0,
                znear=0.01,
                zfar=100.0,
            )

    def test_mismatched_cell_count_fails(self):
        with self.assertRaisesRegex(ValueError, "block_dim"):
            run_visibility_oracle(
                visibility=_visibility_function(),
                cameras=[_camera()],
                cell_metadata=_metadata([_bound(0, [-0.1, -0.1, 2.0], [0.1, 0.1, 2.2])]),
                block_dim=[2, 1, 1],
                aabb=[0, 0, 0, 1, 1, 1],
                precision=8,
            )


class CityGaussianVisibilityOracleFixtureTest(unittest.TestCase):
    def _load_fixture(self, name):
        path = Path(__file__).resolve().parent / "fixtures" / "citygaussian_visibility" / name
        return json.loads(path.read_text())

    def test_finest_fixture_records_expected_selected_sets(self):
        data = self._load_fixture("finest_visibility_oracle_v1.json")

        self.assertEqual(data["schema_version"], "citygaussian_visibility_oracle.v1")
        self.assertEqual(data["asset"]["gaussian_count"], 11694810)
        self.assertEqual(data["cell_metadata"]["cell_count"], 36)
        self.assertEqual(data["cell_metadata"]["total_assigned_count"], 11694810)
        self.assertEqual(data["cell_metadata"]["unassigned_count"], 0)
        self.assertEqual(data["cell_metadata"]["duplicated_assignment_count"], 0)
        self.assertEqual(
            data["camera_metadata"]["selected_indices"],
            [0, 131, 262, 392, 523, 654, 785, 916, 1047, 1177, 1308, 1439],
        )

        by_index = {record["trajectory_index"]: record for record in data["visibility_records"]}
        self.assertEqual(by_index[0]["visible_cell_ids"], [5, 24, 25, 26, 27, 30, 31, 32, 33])
        self.assertEqual(by_index[1177]["visible_cell_ids"], [0, 6, 7, 8, 9, 12, 13, 14, 15, 16, 18, 19, 24])
        self.assertTrue(by_index[1177]["nearest_cell_fallback_changed_mask"])
        self.assertEqual(by_index[1439]["visible_cell_count"], 35)
        self.assertEqual(by_index[1439]["selected_gaussian_count"], 11694702)

    def test_fused_fixture_records_expected_selected_sets(self):
        data = self._load_fixture("fused_visibility_oracle_v1.json")

        self.assertEqual(data["schema_version"], "citygaussian_visibility_oracle.v1")
        self.assertEqual(data["asset"]["gaussian_count"], 23389621)
        self.assertEqual(data["cell_metadata"]["cell_count"], 36)
        self.assertEqual(data["cell_metadata"]["total_assigned_count"], 23389621)
        self.assertEqual(data["cell_metadata"]["unassigned_count"], 0)
        self.assertEqual(data["cell_metadata"]["duplicated_assignment_count"], 0)

        by_index = {record["trajectory_index"]: record for record in data["visibility_records"]}
        self.assertEqual(by_index[0]["visible_cell_ids"], [5, 18, 24, 25, 26, 27, 30, 31, 32, 33])
        self.assertEqual(by_index[654]["visible_cell_ids"], [0, 12, 13, 18, 19, 24, 25, 26, 27, 30, 31, 32])
        self.assertTrue(by_index[654]["nearest_cell_fallback_changed_mask"])
        self.assertTrue(by_index[785]["nearest_cell_fallback_changed_mask"])
        self.assertEqual(by_index[1439]["visible_cell_count"], 35)
        self.assertEqual(by_index[1439]["selected_gaussian_count"], 23389336)


if __name__ == "__main__":
    unittest.main()
