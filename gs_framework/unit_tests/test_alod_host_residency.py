import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from alod_dhier import load_dhier
from scene_runtime.alod import (
    AlodBoundedCache,
    AlodCacheCapacityError,
    AlodHostNoCache,
    AlodHostProperties,
    AlodHostResidencyConfig,
    AlodPreviousWorkingSetReuse,
)
from scene_runtime import FrameCamera, RenderSettings, SceneLoader
from benchmarks import benchmark_real_scene as benchmark
from unit_tests.test_alod_production_runtime import _configs, _write_dhier


def _assert_splats_equal(test, actual, expected):
    for name in ("means", "quats", "scales", "opacities", "colors"):
        test.assertTrue(torch.equal(getattr(actual, name), getattr(expected, name)), name)


class AlodHostPropertyStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny.dhier"
        self.raw = _write_dhier(self.path)
        self.asset = load_dhier(self.path)
        self.host = AlodHostProperties.from_dhier(
            self.asset, torch.device("cpu"), activation_chunk_rows=2
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_host_store_is_render_ready_and_global_id_is_host_row(self):
        self.assertEqual(self.host.row_count, 7)
        self.assertEqual(self.host.property_bytes_per_row, 92)
        self.assertEqual(self.host.host_property_bytes, 7 * 92)
        torch.testing.assert_close(self.host.means, torch.from_numpy(self.raw["positions"]))
        torch.testing.assert_close(
            self.host.quats, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(7, 1)
        )
        torch.testing.assert_close(
            self.host.scales, torch.exp(torch.from_numpy(self.raw["log_scales"]))
        )
        torch.testing.assert_close(
            self.host.opacities,
            torch.from_numpy(self.raw["opacities"].reshape(-1)).clamp(0.0, 0.99999),
        )
        ids = np.asarray([6, 0, 4], dtype=np.int64)
        gathered = self.host.gather_cpu(ids)
        self.assertTrue(torch.equal(gathered.means, self.host.means[ids]))
        self.assertTrue(torch.equal(gathered.colors, self.host.colors[ids]))

    def test_skybox_rows_can_be_permanently_transferred_separately(self):
        skybox = self.host.transfer_rows(np.asarray([0, 1]), torch.device("cpu"))
        _assert_splats_equal(self, skybox, self.host.gather_cpu([0, 1]))
        self.assertEqual(skybox.num_gaussians, 2)

    def test_out_of_range_identity_is_rejected(self):
        with self.assertRaises(IndexError):
            self.host.gather_cpu([7])


class AlodResidencyPolicyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "tiny.dhier"
        _write_dhier(path)
        self.host = AlodHostProperties.from_dhier(
            load_dhier(path), torch.device("cpu"), activation_chunk_rows=3
        )
        self.common = dict(
            host=self.host,
            device=torch.device("cpu"),
            hierarchy_start=2,
            validate_mappings=True,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_nocache_reloads_every_row_on_cold_and_repeated_frames(self):
        policy = AlodHostNoCache(**self.common)
        ids = np.asarray([2, 4, 6], dtype=np.int64)
        cold = policy.prepare(ids)
        warm = policy.prepare(ids)
        self.assertEqual((cold.cache_hits, cold.cache_misses), (0, 3))
        self.assertEqual((warm.cache_hits, warm.cache_misses), (0, 3))
        self.assertEqual(cold.h2d_bytes, 3 * 92)
        self.assertEqual(warm.retained_rows, 0)
        self.assertEqual((cold.discarded_rows, warm.discarded_rows), (0, 3))
        self.assertEqual(warm.selected_set_churn_rows, 0)
        _assert_splats_equal(self, warm.hierarchy_splats, self.host.gather_cpu(ids))

    def test_previous_working_set_reuses_only_immediate_exact_rows(self):
        policy = AlodPreviousWorkingSetReuse(**self.common)
        first = policy.prepare(np.asarray([2, 3, 4], dtype=np.int64))
        second_ids = np.asarray([4, 3, 5], dtype=np.int64)
        second = policy.prepare(second_ids)
        third = policy.prepare(np.asarray([2, 5], dtype=np.int64))
        self.assertEqual((first.cache_hits, first.h2d_rows), (0, 3))
        self.assertEqual((second.retained_rows, second.new_rows, second.discarded_rows), (2, 1, 1))
        self.assertEqual(second.h2d_bytes, 92)
        self.assertEqual(third.cache_hits, 1)
        _assert_splats_equal(self, second.hierarchy_splats, self.host.gather_cpu(second_ids))

    def test_bounded_cache_hit_miss_eviction_and_reverse_mapping(self):
        cache = AlodBoundedCache(**self.common, capacity_rows=3)
        first = cache.prepare(np.asarray([2, 3, 4], dtype=np.int64))
        second_ids = np.asarray([4, 3, 5], dtype=np.int64)
        second = cache.prepare(second_ids)
        self.assertEqual((first.cache_hits, first.cache_misses), (0, 3))
        self.assertEqual((second.cache_hits, second.cache_misses, second.evictions), (2, 1, 1))
        self.assertEqual(second.cache_occupancy_rows, 3)
        np.testing.assert_array_equal(
            cache.slot_to_global[second.ordered_gpu_slots.numpy()], second_ids
        )
        _assert_splats_equal(self, second.hierarchy_splats, self.host.gather_cpu(second_ids))

    def test_bounded_cache_can_hit_a_row_older_than_previous_frame(self):
        cache = AlodBoundedCache(**self.common, capacity_rows=3)
        cache.prepare(np.asarray([2, 3], dtype=np.int64))
        cache.prepare(np.asarray([3, 4], dtype=np.int64))
        third = cache.prepare(np.asarray([2, 4], dtype=np.int64))
        self.assertEqual(third.retained_rows, 2)
        self.assertEqual(third.selected_set_churn_rows, 2)
        self.assertEqual(third.cache_hits, 2)
        self.assertEqual(third.h2d_rows, 0)

    def test_capacity_duplicate_stale_slot_and_reset_guards(self):
        cache = AlodBoundedCache(**self.common, capacity_rows=3)
        with self.assertRaises(AlodCacheCapacityError):
            cache.prepare(np.asarray([2, 3, 4, 5], dtype=np.int64))
        with self.assertRaises(ValueError):
            cache.prepare(np.asarray([2, 2], dtype=np.int64))
        cache.prepare(np.asarray([2, 3, 4], dtype=np.int64))
        cache.prepare(np.asarray([4, 5, 6], dtype=np.int64))
        cache._validate_bijection()
        live_slots = np.flatnonzero(cache.slot_to_global >= 0)
        self.assertEqual(live_slots.size, 3)
        cache.reset()
        self.assertTrue(np.all(cache.global_to_slot == -1))
        self.assertTrue(np.all(cache.slot_to_global == -1))
        self.assertEqual(cache.frame_index, 0)

    def test_nocache_preserves_ordered_duplicate_occurrences(self):
        policy = AlodHostNoCache(**self.common)
        frame = policy.prepare(np.asarray([2, 2], dtype=np.int64))
        self.assertEqual(frame.requested_rows, 2)
        self.assertEqual(frame.h2d_rows, 2)
        np.testing.assert_array_equal(frame.ordered_global_node_ids, [2, 2])
        _assert_splats_equal(
            self, frame.hierarchy_splats, self.host.gather_cpu([2, 2])
        )


class AlodHostSceneIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny.dhier"
        _write_dhier(self.path)
        _, runtime = _configs()
        self.runtime = type(runtime)(
            **{**runtime.__dict__, "selection_mode": "finest", "frustum_enabled": False}
        )
        self.camera = FrameCamera(
            c2w=torch.eye(4),
            K=torch.eye(3),
            width=8,
            height=6,
            camera_stable_id="tiny",
        )
        self.settings = RenderSettings(width=8, height=6, near_plane=0.01, far_plane=100.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_host_modes_preserve_resident_ids_properties_and_metrics(self):
        resident = SceneLoader.load(
            [self.path],
            device=torch.device("cpu"),
            scene_type="alod_resident",
            alod_config=self.runtime,
        )
        expected = resident.prepare_frame(self.camera, self.settings)
        expected_ids = resident.last_selection.raster_global_node_ids.copy()
        modes = {
            "alod_host_nocache": AlodHostResidencyConfig("host_nocache"),
            "alod_host_previous_reuse": AlodHostResidencyConfig("host_previous_reuse"),
            "alod_host_bounded_cache": AlodHostResidencyConfig(
                "host_bounded_cache", cache_capacity_rows=4, validate_properties=True
            ),
        }
        for scene_type, residency_config in modes.items():
            with self.subTest(scene_type=scene_type):
                scene = SceneLoader.load(
                    [self.path],
                    device=torch.device("cpu"),
                    scene_type=scene_type,
                    alod_config=self.runtime,
                    alod_host_config=residency_config,
                )
                actual = scene.prepare_frame(self.camera, self.settings)
                np.testing.assert_array_equal(
                    scene.last_selection.raster_global_node_ids, expected_ids
                )
                _assert_splats_equal(self, actual, expected)
                metrics = actual.scene_metrics
                self.assertEqual(metrics.alod_requested_rows, 4)
                self.assertEqual(metrics.alod_h2d_rows, 4)
                self.assertEqual(metrics.alod_host_property_store_bytes, 7 * 92)
                self.assertEqual(metrics.alod_persistent_skybox_gpu_bytes, 0)
                self.assertEqual(metrics.alod_total_raster_rows, 4)

    def test_loader_rejects_scene_mode_configuration_mismatch(self):
        with self.assertRaises(ValueError):
            SceneLoader.load(
                [self.path],
                device=torch.device("cpu"),
                scene_type="alod_host_nocache",
                alod_config=self.runtime,
                alod_host_config=AlodHostResidencyConfig(
                    "host_bounded_cache", cache_capacity_rows=4
                ),
            )

    def test_prefilter_keeps_semantic_selection_but_reduces_materialization(self):
        filtered_runtime = type(self.runtime)(
            **{
                **self.runtime.__dict__,
                "materialization_filter": "conservative_pinhole",
            }
        )
        scene = SceneLoader.load(
            [self.path],
            device=torch.device("cpu"),
            scene_type="alod_host_nocache",
            alod_config=filtered_runtime,
            alod_host_config=AlodHostResidencyConfig("host_nocache"),
        )
        active = scene.prepare_frame(self.camera, self.settings)
        self.assertEqual(scene.last_selection.raster_global_node_ids.size, 4)
        self.assertEqual(scene.last_materialization.semantic_selected_rows, 4)
        self.assertEqual(scene.last_materialization.materialization_rows, 0)
        self.assertEqual(active.num_gaussians, 0)
        self.assertEqual(active.scene_metrics.selected_gaussian_count, 4)
        self.assertEqual(active.scene_metrics.alod_materialization_rows, 0)
        self.assertEqual(active.scene_metrics.alod_requested_rows, 0)

    def test_fast_prefilter_matches_cpu_prefilter_and_reports_compact_device_bytes(self):
        cpu_runtime = type(self.runtime)(
            **{
                **self.runtime.__dict__,
                "materialization_filter": "conservative_pinhole",
            }
        )
        fast_runtime = type(self.runtime)(
            **{
                **self.runtime.__dict__,
                "materialization_filter": "conservative_pinhole_gpu",
            }
        )
        cpu_scene = SceneLoader.load(
            [self.path],
            device=torch.device("cpu"),
            scene_type="alod_host_nocache",
            alod_config=cpu_runtime,
            alod_host_config=AlodHostResidencyConfig("host_nocache"),
        )
        fast_scene = SceneLoader.load(
            [self.path],
            device=torch.device("cpu"),
            scene_type="alod_host_nocache",
            alod_config=fast_runtime,
            alod_host_config=AlodHostResidencyConfig("host_nocache"),
        )
        cpu_active = cpu_scene.prepare_frame(self.camera, self.settings)
        fast_active = fast_scene.prepare_frame(self.camera, self.settings)
        np.testing.assert_array_equal(
            fast_scene.last_selection.raster_global_node_ids,
            cpu_scene.last_selection.raster_global_node_ids,
        )
        np.testing.assert_array_equal(
            fast_scene.last_materialization.materialization_global_node_ids,
            cpu_scene.last_materialization.materialization_global_node_ids,
        )
        _assert_splats_equal(self, fast_active, cpu_active)
        self.assertEqual(fast_scene.compact_culling_cpu_bytes, 7 * 16)
        self.assertEqual(fast_scene.compact_culling_gpu_bytes, 7 * 16)
        self.assertEqual(
            fast_active.scene_metrics.alod_materialization_selected_id_bytes,
            4 * 8,
        )

    def test_host_benchmark_configs_and_integrity_are_explicit(self):
        expected = {
            "ALOD-HOST-NOCACHE": ("alod_host_nocache", "host_nocache", False, False),
            "ALOD-HOST-PREVIOUS-REUSE": (
                "alod_host_previous_reuse",
                "host_previous_reuse",
                False,
                True,
            ),
            "ALOD-HOST-BOUNDED-CACHE": (
                "alod_host_bounded_cache",
                "host_bounded_cache",
                True,
                False,
            ),
        }
        for config_id, values in expected.items():
            config = benchmark.load_config(config_id)
            scene_type, mode, cache_used, previous_used = values
            self.assertEqual(config["scene_type"], scene_type)
            self.assertEqual(config["alod_host"]["mode"], mode)
            self.assertFalse(config["alod_host"]["validate_mappings"])
            integrity = benchmark.benchmark_integrity_metadata(config)
            self.assertTrue(integrity["host_residency_used"])
            self.assertEqual(integrity["gpu_cache_used"], cache_used)
            self.assertEqual(
                integrity["previous_working_set_reuse_used"], previous_used
            )
        bounded = benchmark.load_config("ALOD-HOST-BOUNDED-CACHE")
        derivation = bounded["alod_host"]["capacity_derivation"]
        self.assertEqual(derivation["minimum_feasible_capacity_rows"], 499030)
        self.assertEqual(bounded["alod_host"]["cache_capacity_rows"], 525000)
        reduced = benchmark.load_config("ALOD-HOST-BOUNDED-CACHE-REDUCED")
        self.assertEqual(reduced["alod_host"]["cache_capacity_rows"], 160000)
        self.assertEqual(
            reduced["alod"]["materialization_filter"],
            "conservative_pinhole_gpu",
        )
        self.assertEqual(
            reduced["alod_host"]["capacity_derivation"][
                "minimum_feasible_capacity_rows"
            ],
            50936,
        )

    def test_generic_benchmark_loader_builds_host_scene(self):
        config = {
            "scene_type": "alod_host_nocache",
            "alod": self.runtime.__dict__,
            "alod_host": {"mode": "host_nocache"},
        }
        tracker = benchmark.FeasibilityStageTracker(torch.device("cpu"))
        scene, stats = benchmark.load_scene_for_benchmark(
            config,
            self.path,
            torch.device("cpu"),
            progress_callback=tracker.enter,
        )
        self.assertEqual(stats.scene_type, "alod_host_nocache")
        self.assertEqual(tracker.current_stage, "gpu_scene_construction")
        self.assertEqual(scene.host.row_count, 7)


if __name__ == "__main__":
    unittest.main()
