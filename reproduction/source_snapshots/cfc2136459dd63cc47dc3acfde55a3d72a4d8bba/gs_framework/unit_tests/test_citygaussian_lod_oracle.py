import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.extract_citygaussian_lod_oracle import (
    build_lod_frames,
    focal_length_pixels,
    nyquist_scaling,
    paper_lod_index,
    release_lod_index,
)


def _corners(mn, mx):
    return {
        "cell_id": 0,
        "min": list(mn),
        "max": list(mx),
    }


def _camera():
    world_view = torch.eye(4, dtype=torch.float32)
    projection = torch.eye(4, dtype=torch.float32)
    return SimpleNamespace(
        trajectory_index=7,
        camera_id="synthetic",
        world_view_transform=world_view,
        full_proj_transform=world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0),
        projection_matrix=projection,
        camera_center=torch.zeros(3, dtype=torch.float32),
        fov_x=math.radians(90.0),
        fov_y=math.radians(90.0),
        width=100,
        height=100,
    )


def _visibility_fn(_camera, cell_corners, _aabb, _block_dim):
    mask = torch.ones(cell_corners.shape[0], dtype=torch.bool)
    distances = torch.tensor([2.0 + i for i in range(cell_corners.shape[0])], dtype=torch.float32)
    return mask, distances


def _levels():
    bounds = [
        {"cell_id": 0, "min": [-0.2, -0.2, 2.0], "max": [-0.1, -0.1, 2.2]},
        {"cell_id": 1, "min": [0.1, -0.2, 3.0], "max": [0.2, -0.1, 3.2]},
    ]
    return [
        {
            "level_index": 0,
            "label": "light_50_vq",
            "semantic": "finest",
            "gaussian_count": 100,
            "cell_ids": [0, 1],
            "cell_gaussian_counts": [40, 60],
            "average_scales": [0.09, 0.01],
            "mad_bounds": bounds,
            "total_assigned_count": 100,
            "unassigned_count": 0,
            "duplicated_assignment_count": 0,
            "nonempty_cell_count": 2,
            "all_cells_nonempty": True,
        },
        {
            "level_index": 1,
            "label": "light_66_vq",
            "semantic": "middle",
            "gaussian_count": 70,
            "cell_ids": [0, 1],
            "cell_gaussian_counts": [30, 40],
            "average_scales": [0.05, 0.13],
            "mad_bounds": bounds,
            "total_assigned_count": 70,
            "unassigned_count": 0,
            "duplicated_assignment_count": 0,
            "nonempty_cell_count": 2,
            "all_cells_nonempty": True,
        },
        {
            "level_index": 2,
            "label": "light_75_vq",
            "semantic": "coarsest",
            "gaussian_count": 50,
            "cell_ids": [0, 1],
            "cell_gaussian_counts": [20, 30],
            "average_scales": [0.01, 0.01],
            "mad_bounds": bounds,
            "total_assigned_count": 50,
            "unassigned_count": 0,
            "duplicated_assignment_count": 0,
            "nonempty_cell_count": 2,
            "all_cells_nonempty": True,
        },
    ]


class CityGaussianLodOracleFormulaTest(unittest.TestCase):
    def test_nyquist_focal_length_calculation(self):
        focal = focal_length_pixels(100, math.radians(90.0))
        self.assertAlmostEqual(focal, 50.0)
        self.assertAlmostEqual(nyquist_scaling(2.0, focal), 0.08)

    def test_release_selection_is_first_match(self):
        self.assertEqual(release_lod_index([0.09, 0.2, 0.3], 0.08), 0)
        self.assertEqual(release_lod_index([0.01, 0.09, 0.3], 0.08), 1)

    def test_release_selection_falls_back_to_last_level(self):
        self.assertEqual(release_lod_index([0.01, 0.02, 0.03], 0.08), 2)

    def test_comparison_is_strictly_greater_than(self):
        self.assertEqual(release_lod_index([0.08, 0.07, 0.03], 0.08), 2)
        self.assertEqual(release_lod_index([0.080001, 0.07, 0.03], 0.08), 0)

    def test_paper_policy_intervals_are_separate(self):
        self.assertEqual(paper_lod_index(199.999), 0)
        self.assertEqual(paper_lod_index(200.0), 1)
        self.assertEqual(paper_lod_index(400.0), 2)

    def test_release_frame_counts_match_selected_level_counts(self):
        frames = build_lod_frames(
            policy="CG-LOD-RELEASE",
            visibility_fn=_visibility_fn,
            cameras=[_camera()],
            levels=_levels(),
            visibility_level_index=2,
            block_dim=[2, 1, 1],
            aabb=[0, 0, 0, 1, 1, 1],
            precision=8,
        )

        row = frames[0]
        self.assertEqual(row["visible_cell_ids"], [0, 1])
        self.assertEqual(row["selected_lod_by_cell"], {"0": 0, "1": 1})
        self.assertEqual(row["selected_gaussian_count_by_cell"], {"0": 40, "1": 40})
        self.assertEqual(row["selected_cell_count_by_level"], [1, 1, 0])
        self.assertEqual(row["selected_gaussian_count_by_level"], [40, 40, 0])
        self.assertEqual(row["total_selected_gaussian_count"], 80)

    def test_paper_and_release_policies_diverge_on_same_distances(self):
        release = build_lod_frames(
            policy="CG-LOD-RELEASE",
            visibility_fn=_visibility_fn,
            cameras=[_camera()],
            levels=_levels(),
            visibility_level_index=2,
            block_dim=[2, 1, 1],
            aabb=[0, 0, 0, 1, 1, 1],
            precision=8,
        )[0]
        paper = build_lod_frames(
            policy="CG-LOD-PAPER",
            visibility_fn=_visibility_fn,
            cameras=[_camera()],
            levels=_levels(),
            visibility_level_index=2,
            block_dim=[2, 1, 1],
            aabb=[0, 0, 0, 1, 1, 1],
            precision=8,
        )[0]

        self.assertNotEqual(release["selected_lod_by_cell"], paper["selected_lod_by_cell"])
        self.assertEqual(paper["selected_lod_by_cell"], {"0": 0, "1": 0})

    def test_synthetic_extraction_does_not_initialize_cuda(self):
        initialized_before = torch.cuda.is_initialized()
        build_lod_frames(
            policy="CG-LOD-RELEASE",
            visibility_fn=_visibility_fn,
            cameras=[_camera()],
            levels=_levels(),
            visibility_level_index=2,
            block_dim=[2, 1, 1],
            aabb=[0, 0, 0, 1, 1, 1],
            precision=8,
        )
        self.assertEqual(torch.cuda.is_initialized(), initialized_before)

    def test_synthetic_extraction_is_deterministic(self):
        kwargs = dict(
            policy="CG-LOD-RELEASE",
            visibility_fn=_visibility_fn,
            cameras=[_camera()],
            levels=_levels(),
            visibility_level_index=2,
            block_dim=[2, 1, 1],
            aabb=[0, 0, 0, 1, 1, 1],
            precision=8,
        )
        first = json.dumps(build_lod_frames(**kwargs), sort_keys=True)
        second = json.dumps(build_lod_frames(**kwargs), sort_keys=True)
        self.assertEqual(first, second)


class CityGaussianLodOracleFixtureTest(unittest.TestCase):
    def _load_fixture(self, name):
        path = Path(__file__).resolve().parent / "fixtures" / "citygaussian_lod" / name
        return json.loads(path.read_text())

    def test_release_fixture_schema_and_level_order(self):
        data = self._load_fixture("release_lod_oracle_v1.json")

        self.assertEqual(data["schema_version"], "citygaussian_lod_oracle.v1")
        self.assertEqual(data["policy"], "CG-LOD-RELEASE")
        self.assertEqual(data["validation_status"], "authoritative")
        self.assertEqual(data["level_order"], ["light_50_vq", "light_66_vq", "light_75_vq"])
        self.assertEqual(data["level_gaussian_counts"], [11694810, 7952471, 5847405])
        self.assertEqual(data["block_dim"], [6, 6, 1])
        self.assertEqual(len(data["frames"]), 12)
        self.assertEqual(data["frames"][0]["visibility_source_level_index"], 2)

    def test_release_fixture_selected_lod_exists_and_counts_match(self):
        data = self._load_fixture("release_lod_oracle_v1.json")
        counts_by_level = [level["cell_gaussian_counts"] for level in data["level_cell_metadata"]]

        for frame in data["frames"]:
            self.assertEqual(set(map(str, frame["visible_cell_ids"])), set(frame["selected_lod_by_cell"]))
            total = 0
            per_level = [0, 0, 0]
            for cell_id in frame["visible_cell_ids"]:
                key = str(cell_id)
                level_index = frame["selected_lod_by_cell"][key]
                count = frame["selected_gaussian_count_by_cell"][key]
                self.assertEqual(count, counts_by_level[level_index][cell_id])
                total += count
                per_level[level_index] += count
            self.assertEqual(total, frame["total_selected_gaussian_count"])
            self.assertEqual(per_level, frame["selected_gaussian_count_by_level"])

    def test_release_fixture_records_expected_selected_summaries(self):
        data = self._load_fixture("release_lod_oracle_v1.json")
        by_index = {frame["trajectory_index"]: frame for frame in data["frames"]}

        self.assertEqual(by_index[0]["selected_cell_count_by_level"], [3, 0, 6])
        self.assertEqual(by_index[0]["total_selected_gaussian_count"], 1957671)
        self.assertEqual(by_index[523]["selected_cell_count_by_level"], [10, 3, 0])
        self.assertEqual(by_index[523]["total_selected_gaussian_count"], 4256076)
        self.assertEqual(by_index[1439]["selected_cell_count_by_level"], [5, 0, 30])
        self.assertEqual(by_index[1439]["total_selected_gaussian_count"], 6211124)

    def test_release_fixture_matches_previous_oracle_trajectory_contract(self):
        release = self._load_fixture("release_lod_oracle_v1.json")
        previous_path = Path(__file__).resolve().parent / "fixtures" / "citygaussian_visibility" / "finest_visibility_oracle_v1.json"
        previous = json.loads(previous_path.read_text())

        self.assertEqual(release["trajectory_indices"], previous["camera_metadata"]["selected_indices"])
        self.assertEqual(
            [frame["trajectory_index"] for frame in release["frames"]],
            [record["trajectory_index"] for record in previous["visibility_records"]],
        )
        for frame in release["frames"]:
            self.assertTrue(all(0 <= cell_id < 36 for cell_id in frame["visible_cell_ids"]))

    def test_paper_fixture_is_provisional_and_separate(self):
        release = self._load_fixture("release_lod_oracle_v1.json")
        paper = self._load_fixture("paper_lod_oracle_v1.json")

        self.assertEqual(paper["policy"], "CG-LOD-PAPER")
        self.assertEqual(paper["validation_status"], "provisional")
        self.assertTrue(paper["unresolved_semantics"])
        self.assertNotEqual(
            release["frames"][0]["selected_lod_by_cell"],
            paper["frames"][0]["selected_lod_by_cell"],
        )
        for frame in paper["frames"]:
            self.assertEqual(frame["selected_cell_count_by_level"][1:], [0, 0])


if __name__ == "__main__":
    unittest.main()
