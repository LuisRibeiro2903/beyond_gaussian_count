import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from alod_dhier import load_dhier
from alod_reference import HierarchyData, ReferenceCamera, ReferenceConfig, build_hspt
from alod_reference.selection import select_reference
from scene_runtime.alod import (
    AlodHierarchySource,
    AlodResidentProperties,
    AlodRuntimeConfig,
    AlodSelectionCamera,
    build_production_hspt,
    select_production,
)
from scene_runtime.alod.frustum import extract_four_planes
from scene_runtime import FrameCamera, RenderSettings, SceneLoader
from scene_runtime.renderer import _visible_count
from benchmarks import benchmark_real_scene as benchmark


def _tree() -> np.ndarray:
    return np.asarray(
        [
            [0, -1, 2, 1, 0, 0],
            [1, 0, 2, 2, 4, 0],
            [2, 1, 0, 0, 3, 0],
            [2, 1, 0, 0, 0, 0],
            [1, 0, 2, 5, 0, 0],
            [2, 4, 0, 0, 6, 0],
            [2, 4, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )


def _write_dhier(path: Path) -> dict[str, np.ndarray]:
    nodes = _tree()
    count = len(nodes)
    positions = np.asarray(
        [[0, 0, 0], [-0.3, 0, 0], [-0.5, 0, 0], [-0.1, 0, 0],
         [0.3, 0, 0], [0.1, 0, 0], [0.5, 0, 0]], dtype=np.float32
    )
    rotations = np.tile(
        np.asarray([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)
    )
    log_scales = np.full((count, 3), -2.0, dtype=np.float32)
    opacities = np.asarray(
        [[0.0], [0.25], [0.5], [0.75], [0.99999], [1.0], [0.1]],
        dtype=np.float32,
    )
    sh = np.arange(count * 12, dtype=np.float32).reshape(count, 4, 3) / 100.0
    payload = bytearray(struct.pack("<ii", count, 1))
    for array in (positions, rotations, log_scales, opacities, sh):
        payload.extend(array.astype("<f4").tobytes())
    payload.extend(struct.pack("<i", count))
    payload.extend(nodes.astype("<i4").tobytes())
    path.write_bytes(payload)
    return {
        "positions": positions,
        "rotations": rotations,
        "log_scales": log_scales,
        "opacities": opacities,
        "sh": sh,
        "nodes": nodes,
    }


def _configs() -> tuple[ReferenceConfig, AlodRuntimeConfig]:
    reference = ReferenceConfig(
        root_global_node_id=0,
        skybox_count=0,
        spt_root_volume=0.0001,
        min_spt_size=100,
        target_granularity=0.1,
        distance_multiplier=1.0,
    )
    production = AlodRuntimeConfig(
        root_global_node_id=0,
        skybox_count=0,
        spt_root_volume=0.0001,
        min_spt_size=100,
        target_granularity=0.1,
        distance_multiplier=1.0,
    )
    return reference, production


class AlodProductionRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny.dhier"
        self.raw = _write_dhier(self.path)
        self.asset = load_dhier(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_resident_properties_preserve_identity_and_activate_values(self):
        resident = AlodResidentProperties.from_dhier(self.asset, torch.device("cpu"))
        torch.testing.assert_close(resident.means, torch.from_numpy(self.raw["positions"]))
        torch.testing.assert_close(
            resident.quats,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(7, 1),
        )
        torch.testing.assert_close(
            resident.scales, torch.exp(torch.from_numpy(self.raw["log_scales"]))
        )
        torch.testing.assert_close(
            resident.opacities,
            torch.from_numpy(self.raw["opacities"].reshape(-1)).clamp(0.0, 0.99999),
        )
        torch.testing.assert_close(resident.colors, torch.from_numpy(self.raw["sh"]))
        torch.testing.assert_close(resident.global_node_ids, torch.arange(7))

    def test_selected_property_gather_keeps_global_id_order(self):
        resident = AlodResidentProperties.from_dhier(self.asset, torch.device("cpu"))
        ids = torch.tensor([6, 2, 5, 0], dtype=torch.int64)
        active = resident.gather(ids)
        torch.testing.assert_close(active.means, resident.means[ids])
        torch.testing.assert_close(active.quats, resident.quats[ids])
        torch.testing.assert_close(active.colors, resident.colors[ids])

    def test_hspt_and_selection_match_independent_reference(self):
        reference_config, production_config = _configs()
        reference_hspt = build_hspt(HierarchyData.from_dhier(self.asset), reference_config)
        hierarchy = AlodHierarchySource(
            self.asset.positions, self.asset.log_scales, self.asset.nodes
        )
        production_hspt = build_production_hspt(hierarchy, production_config)
        for name in (
            "upper_tree_nodes",
            "upper_to_global_node_id",
            "upper_tree_xyz",
            "upper_tree_log_scaling",
            "upper_min_distance_squared",
            "spt_starts",
            "spt_min",
            "spt_max",
            "spt_to_global_node_id",
            "spt_root_global_node_ids",
            "spt_sizes",
        ):
            np.testing.assert_array_equal(
                getattr(production_hspt, name), getattr(reference_hspt, name)
            )

        center = np.asarray([0.0, 0.0, 2.0], dtype=np.float32)
        transform = np.eye(4, dtype=np.float32)
        expected = select_reference(
            reference_hspt,
            ReferenceCamera("tiny", center, transform),
            reference_config,
        )
        actual = select_production(
            hierarchy,
            production_hspt,
            AlodSelectionCamera("tiny", center, transform),
            production_config,
        )
        np.testing.assert_array_equal(
            actual.upper.candidate_spt_ids, expected.upper.candidate_spt_ids
        )
        np.testing.assert_array_equal(
            actual.selected_hierarchy_global_node_ids,
            expected.selected_hierarchy_global_node_ids,
        )
        np.testing.assert_array_equal(actual.raster_global_node_ids, expected.raster_global_node_ids)

    def test_finest_cut_is_leaf_rows_not_every_hierarchy_node(self):
        _, config = _configs()
        finest = AlodRuntimeConfig(**{**config.__dict__, "selection_mode": "finest"})
        hierarchy = AlodHierarchySource(
            self.asset.positions, self.asset.log_scales, self.asset.nodes
        )
        hspt = build_production_hspt(hierarchy, finest)
        result = select_production(
            hierarchy,
            hspt,
            AlodSelectionCamera(
                "tiny", np.zeros(3, dtype=np.float32), np.eye(4, dtype=np.float32)
            ),
            finest,
        )
        np.testing.assert_array_equal(result.raster_global_node_ids, [2, 3, 5, 6])

    def test_frustum_extracts_only_four_planes(self):
        planes = extract_four_planes(np.eye(4, dtype=np.float32))
        self.assertEqual(planes.shape, (4, 4))
        np.testing.assert_array_equal(planes[:, 3], np.ones(4, dtype=np.float32))

    def test_packed_visible_count_uses_unique_projected_global_rows(self):
        self.assertEqual(
            _visible_count(
                {"gaussian_ids": torch.tensor([7, 7, 9, 12, 12])}, packed=True
            ),
            3.0,
        )

    def test_scene_loader_constructs_resident_scene_and_instruments_frame(self):
        _, config = _configs()
        scene = SceneLoader.load(
            [self.path],
            device=torch.device("cpu"),
            scene_type="alod_resident",
            alod_config=config,
        )
        stats = scene.get_scene_stats()
        self.assertEqual(stats.scene_type, "alod_resident")
        self.assertEqual(stats.total_gaussians, 7)
        self.assertTrue(stats.alod_no_streaming)
        self.assertEqual(stats.alod_hierarchy_metadata_gpu_bytes, 7 * 8)
        active = scene.prepare_frame(
            FrameCamera(
                c2w=torch.eye(4),
                K=torch.eye(3),
                width=8,
                height=6,
                camera_stable_id="tiny",
            ),
            RenderSettings(width=8, height=6, near_plane=0.01, far_plane=100.0),
        )
        self.assertEqual(
            active.num_gaussians,
            int(scene.last_selection.raster_global_node_ids.size),
        )
        self.assertEqual(active.scene_metrics.alod_total_asset_rows, 7)
        self.assertEqual(active.scene_metrics.alod_total_raster_rows, active.num_gaussians)
        self.assertTrue(active.scene_metrics.alod_no_streaming)

    def test_resident_ablation_configs_are_explicit_and_disable_streaming(self):
        expected = {
            "ALOD-HIER-FINEST-RESIDENT": ("finest", False),
            "ALOD-HIER-LOD-RESIDENT": ("lod", False),
            "ALOD-HIER-LOD-FRUSTUM": ("lod", True),
        }
        for config_id, semantics in expected.items():
            config = benchmark.load_config(config_id)
            self.assertEqual(config["scene_type"], "alod_resident")
            self.assertEqual(
                (config["alod"]["selection_mode"], config["alod"]["frustum_enabled"]),
                semantics,
            )
            self.assertIn("streaming", config["disabled_runtime_components"])
            self.assertTrue(benchmark.resolve_camera_path(config).is_file())
            camera_path = benchmark.apply_camera_config(
                benchmark.load_camera_path(
                    benchmark.resolve_camera_path(config), config["trajectory_id"]
                ),
                config,
            )
            self.assertEqual(len(camera_path.cameras), 10)
            self.assertEqual(
                (camera_path.cameras[0].width, camera_path.cameras[0].height),
                (960, 540),
            )
            self.assertEqual(
                camera_path.cameras[0].camera_id,
                "json_000000",
            )
            integrity = benchmark.benchmark_integrity_metadata(config)
            self.assertFalse(integrity["streaming_used"])
            self.assertFalse(integrity["gpu_cache_used"])

    def test_benchmark_loader_accepts_alod_progress_stages(self):
        _, runtime = _configs()
        config = {
            "scene_type": "alod_resident",
            "alod": {
                **runtime.__dict__,
            },
        }
        tracker = benchmark.FeasibilityStageTracker(torch.device("cpu"))
        scene, stats = benchmark.load_scene_for_benchmark(
            config,
            self.path,
            torch.device("cpu"),
            progress_callback=tracker.enter,
        )
        self.assertEqual(stats.scene_type, "alod_resident")
        self.assertEqual(scene.get_scene_stats().total_gaussians, 7)
        self.assertEqual(tracker.current_stage, "gpu_scene_construction")


if __name__ == "__main__":
    unittest.main()
