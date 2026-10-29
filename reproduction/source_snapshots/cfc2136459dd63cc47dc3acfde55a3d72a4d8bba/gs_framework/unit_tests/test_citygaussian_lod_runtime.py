import inspect
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmarks import benchmark_real_scene as brs
from scene_runtime.citygaussian import (
    RELEASE_LOD_LEVEL_LABELS,
    RELEASE_LOD_POLICY,
    CityGaussianReleaseLodScene,
    bounds_to_corners,
    released_nyquist_lod_index,
    select_citygaussian_release_lod,
)
from scene_runtime.types import ActiveSplats, FrameCamera, RenderSettings, SceneFrameMetrics


def _active(count: int, offset: float = 0.0) -> ActiveSplats:
    return ActiveSplats(
        means=torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) + offset,
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(count, 1),
        scales=torch.ones((count, 3), dtype=torch.float32),
        opacities=torch.ones((count,), dtype=torch.float32),
        colors=torch.zeros((count, 9, 3), dtype=torch.float32),
    )


def _frame_camera() -> FrameCamera:
    return FrameCamera(
        c2w=torch.eye(4, dtype=torch.float32),
        K=torch.tensor(
            [[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        width=100,
        height=100,
    )


def _settings() -> RenderSettings:
    return RenderSettings(
        width=100,
        height=100,
        near_plane=0.01,
        far_plane=100.0,
        sh_degree=2,
    )


def _fake_level(
    *,
    count: int,
    cell_counts: list[int],
    average_scales: list[float],
    corners: torch.Tensor,
    offset: float,
):
    active = _active(count, offset)
    indices = []
    start = 0
    for cell_count in cell_counts:
        indices.append(torch.arange(start, start + cell_count, dtype=torch.long))
        start += cell_count
    return SimpleNamespace(
        sh_degree=2,
        cell_ids=[0, 1],
        validation=SimpleNamespace(every_gaussian_assigned_once=True, all_cells_nonempty=True),
        total_gaussians=count,
        cell_counts=cell_counts,
        cell_average_scales=average_scales,
        cell_construction_ms=0.25,
        cell_corners=corners,
        cell_mad_bounds=[
            {"min": (-0.2, -0.2, 2.0), "max": (0.2, 0.2, 2.2)},
            {"min": (0.1, -0.2, 3.0), "max": (0.2, -0.1, 3.2)},
        ],
        cell_source_indices=indices,
        flat_active=active,
        scene_min=(-0.2, -0.2, 2.0),
        scene_max=(0.2, 0.2, 3.2),
    )


class CityGaussianReleaseLodFormulaTest(unittest.TestCase):
    def test_strict_first_match_and_coarsest_fallback(self):
        self.assertEqual(released_nyquist_lod_index([0.09, 0.2, 0.3], 0.08), 0)
        self.assertEqual(released_nyquist_lod_index([0.08, 0.09, 0.3], 0.08), 1)
        self.assertEqual(released_nyquist_lod_index([0.08, 0.07, 0.03], 0.08), 2)

    def test_release_selection_formula_and_counts(self):
        selection = select_citygaussian_release_lod(
            visible_cell_ids=[0, 1],
            per_visible_cell_distances={0: 2.0, 1: 3.0},
            average_scales_by_level=[
                [0.09, 0.01],
                [0.05, 0.13],
                [0.01, 0.01],
            ],
            cell_counts_by_level=[[40, 60], [30, 40], [20, 30]],
            level_labels=RELEASE_LOD_LEVEL_LABELS,
            image_width=100,
            fov_x=math.radians(90.0),
        )

        self.assertAlmostEqual(selection.focal_length_pixels, 50.0)
        self.assertEqual(selection.selected_lod_by_cell, {0: 0, 1: 1})
        self.assertEqual(
            selection.selected_level_label_by_cell,
            {0: "light_50_vq", 1: "light_66_vq"},
        )
        self.assertEqual(selection.selected_gaussian_count_by_cell, {0: 40, 1: 40})
        self.assertEqual(selection.selected_cell_count_by_level, [1, 1, 0])
        self.assertEqual(selection.selected_gaussian_count_by_level, [40, 40, 0])
        self.assertEqual(selection.total_selected_gaussian_count, 80)


class CityGaussianReleaseLodRuntimeTest(unittest.TestCase):
    def _scene(self) -> CityGaussianReleaseLodScene:
        finest_corners = torch.stack(
            [
                bounds_to_corners(torch.tensor([-1.0, -1.0, 2.0]), torch.tensor([-0.8, -0.8, 2.2])),
                bounds_to_corners(torch.tensor([8.0, 8.0, 3.0]), torch.tensor([9.0, 9.0, 3.2])),
            ]
        )
        coarsest_corners = torch.stack(
            [
                bounds_to_corners(torch.tensor([-0.2, -0.2, 2.0]), torch.tensor([0.2, 0.2, 2.2])),
                bounds_to_corners(torch.tensor([0.1, -0.2, 3.0]), torch.tensor([0.2, -0.1, 3.2])),
            ]
        )
        levels = [
            _fake_level(
                count=3,
                cell_counts=[2, 1],
                average_scales=[0.09, 0.01],
                corners=finest_corners,
                offset=0.0,
            ),
            _fake_level(
                count=3,
                cell_counts=[1, 2],
                average_scales=[0.05, 0.13],
                corners=finest_corners,
                offset=100.0,
            ),
            _fake_level(
                count=2,
                cell_counts=[1, 1],
                average_scales=[0.01, 0.01],
                corners=coarsest_corners,
                offset=200.0,
            ),
        ]
        with patch(
            "scene_runtime.citygaussian.CityGaussianFixedAllScene",
            side_effect=levels,
        ):
            return CityGaussianReleaseLodScene(
                [Path("finest.ply"), Path("middle.ply"), Path("coarsest.ply")],
                device=torch.device("cpu"),
                block_dim=(2, 1, 1),
                aabb=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            )

    def test_level_metadata_and_coarsest_visibility_bounds(self):
        scene = self._scene()
        stats = scene.get_scene_stats()

        self.assertEqual(stats.scene_type, "citygaussian_lod_release")
        self.assertEqual(stats.citygaussian_lod_policy, RELEASE_LOD_POLICY)
        self.assertEqual(stats.citygaussian_lod_level_labels, list(RELEASE_LOD_LEVEL_LABELS))
        self.assertEqual(stats.citygaussian_lod_level_gaussian_counts, [3, 3, 2])
        self.assertEqual(stats.total_gaussians, 8)
        self.assertIs(scene.cell_corners, scene.level_scenes[-1].cell_corners)
        self.assertEqual(scene.visibility_bounds_level, "light_75_vq")

    def test_selected_lod_counts_residency_and_serialization(self):
        scene = self._scene()
        active = scene.prepare_frame(_frame_camera(), _settings())
        metrics = active.scene_metrics

        self.assertEqual(metrics.selected_cell_ids, "0,1")
        self.assertEqual(metrics.selected_lod_by_cell, {0: 0, 1: 1})
        self.assertEqual(metrics.selected_cell_count_by_level, [1, 1, 0])
        self.assertEqual(metrics.selected_gaussian_count_by_level, [2, 2, 0])
        self.assertEqual(metrics.selected_gaussian_count, 4)
        self.assertEqual(metrics.active_gaussians, 4)
        self.assertEqual(active.num_gaussians, 4)
        self.assertEqual(metrics.resident_gaussians, 8)
        self.assertGreaterEqual(metrics.visibility_selection_ms, 0.0)
        self.assertGreaterEqual(metrics.lod_selection_ms, 0.0)
        self.assertGreaterEqual(metrics.selected_cell_assembly_ms, 0.0)
        self.assertTrue(metrics.citygaussian_adaptive_lod_used)
        self.assertFalse(metrics.citygaussian_no_adaptive_lod)
        self.assertTrue(metrics.citygaussian_no_vq_reconstruction)
        self.assertTrue(metrics.citygaussian_no_streaming)
        json.dumps(
            {
                "selected_lod_by_cell": metrics.selected_lod_by_cell,
                "selected_cell_count_by_level": metrics.selected_cell_count_by_level,
                "selected_gaussian_count_by_level": metrics.selected_gaussian_count_by_level,
            },
            allow_nan=False,
        )

    def test_release_runtime_has_no_paper_vq_or_block_manager_path(self):
        source = inspect.getsource(CityGaussianReleaseLodScene)
        self.assertNotIn("paper_lod", source.lower())
        self.assertNotIn("extreme_saving", source)
        self.assertNotIn("BlockManager", source)


class CityGaussianReleaseLodOracleFixtureRuntimeTest(unittest.TestCase):
    def test_all_release_oracle_selected_sets_match_pure_runtime_selector(self):
        fixture_path = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "citygaussian_lod"
            / "release_lod_oracle_v1.json"
        )
        fixture = json.loads(fixture_path.read_text())
        averages = [level["average_scales"] for level in fixture["level_cell_metadata"]]
        counts = [level["cell_gaussian_counts"] for level in fixture["level_cell_metadata"]]

        for frame in fixture["frames"]:
            selection = select_citygaussian_release_lod(
                visible_cell_ids=frame["visible_cell_ids"],
                per_visible_cell_distances={
                    int(cell_id): float(distance)
                    for cell_id, distance in frame["cell_distances"].items()
                },
                average_scales_by_level=averages,
                cell_counts_by_level=counts,
                level_labels=fixture["level_order"],
                image_width=frame["image_size"][0],
                fov_x=frame["fov_x_radians"],
            )
            expected_lods = {
                int(cell_id): int(level_index)
                for cell_id, level_index in frame["selected_lod_by_cell"].items()
            }
            expected_labels = {
                int(cell_id): label
                for cell_id, label in frame["selected_level_label_by_cell"].items()
            }
            expected_counts = {
                int(cell_id): int(count)
                for cell_id, count in frame["selected_gaussian_count_by_cell"].items()
            }
            self.assertEqual(selection.visible_cell_ids, frame["visible_cell_ids"])
            self.assertEqual(selection.selected_lod_by_cell, expected_lods)
            self.assertEqual(selection.selected_level_label_by_cell, expected_labels)
            self.assertEqual(selection.selected_gaussian_count_by_cell, expected_counts)
            self.assertEqual(
                selection.selected_cell_count_by_level,
                frame["selected_cell_count_by_level"],
            )
            self.assertEqual(
                selection.selected_gaussian_count_by_level,
                frame["selected_gaussian_count_by_level"],
            )
            self.assertEqual(
                selection.total_selected_gaussian_count,
                frame["total_selected_gaussian_count"],
            )


class CityGaussianReleaseLodBenchmarkConfigTest(unittest.TestCase):
    def test_config_and_external_path_resolution(self):
        config = brs.load_config("CG-LOD-RELEASE")
        self.assertEqual(config["scene_type"], "citygaussian_lod_release")
        self.assertEqual(config["lod_policy"], RELEASE_LOD_POLICY)
        self.assertIn("paper_fixed_distance_lod", config["disabled_runtime_components"])
        with patch.dict(
            "os.environ",
            {
                config["asset_env_vars"]["finest"]: __file__,
                config["asset_env_vars"]["middle"]: __file__,
                config["asset_env_vars"]["coarsest"]: __file__,
            },
            clear=True,
        ):
            paths, supplied = brs.resolve_lod_asset_paths(
                config,
                finest_arg=None,
                middle_arg=None,
                coarsest_arg=None,
            )
        self.assertEqual(len(paths), 3)
        self.assertEqual(len(supplied), 3)

    def test_benchmark_result_serializes_release_lod_fields(self):
        config = brs.load_config("CG-LOD-RELEASE")
        camera = brs.CameraSpec(
            camera_id="fixture",
            c2w=torch.eye(4).numpy(),
            K=torch.eye(3).numpy(),
            width=4,
            height=3,
        )

        class FakeScene:
            def prepare_frame(self, _camera, _settings):
                active = _active(1)
                active.scene_metrics = SceneFrameMetrics(
                    active_blocks=1,
                    active_gaussians=1,
                    active_gaussian_fraction=1.0 / 3.0,
                    selected_cell_count=1,
                    selected_cell_ids="0",
                    selected_gaussian_count=1,
                    selected_lod_by_cell={0: 2},
                    selected_level_label_by_cell={0: "light_75_vq"},
                    selected_gaussian_count_by_cell={0: 1},
                    selected_cell_count_by_level=[0, 0, 1],
                    selected_gaussian_count_by_level=[0, 0, 1],
                    resident_gaussians=3,
                    visibility_selection_ms=0.1,
                    lod_selection_ms=0.2,
                    selected_cell_assembly_ms=0.3,
                    citygaussian_adaptive_lod_used=True,
                    citygaussian_lod_policy=RELEASE_LOD_POLICY,
                    citygaussian_visibility_bounds_level="light_75_vq",
                    citygaussian_no_vq_reconstruction=True,
                    citygaussian_no_streaming=True,
                )
                return active

        class FakeRenderer:
            def render(self, *_args, **_kwargs):
                return SimpleNamespace(
                    image=torch.zeros((3, 4, 3), dtype=torch.float32),
                    alpha=torch.ones((3, 4, 1), dtype=torch.float32),
                    render_time_ms=1.0,
                    gpu_render_time_ms=0.8,
                    visible_gaussians=1.0,
                )

        result = brs.render_one_frame(
            FakeScene(),
            FakeRenderer(),
            camera,
            config,
            scene_total_gaussians=3,
            scene_sh_degree=2,
            device=torch.device("cpu"),
            log_gpu_events=False,
        )
        self.assertEqual(result["selected_lod_by_cell"], {0: 2})
        self.assertEqual(result["selected_cell_count_by_level"], [0, 0, 1])
        self.assertEqual(result["resident_gaussian_count"], 3)
        self.assertEqual(result["active_gaussian_count"], 1)
        self.assertEqual(result["selected_gaussian_count"], 1)
        json.dumps(brs.json_safe(result), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
