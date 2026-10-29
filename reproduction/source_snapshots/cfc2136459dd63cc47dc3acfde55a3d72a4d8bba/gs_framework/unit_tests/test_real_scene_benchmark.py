import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmarks import benchmark_real_scene as brs
from block_manager import build_partitioned_checkpoint
from scene_runtime import FrameCamera, RenderSettings
from scene_runtime.types import ActiveSplats, SceneFrameMetrics


def _fake_splats():
    means = torch.tensor(
        [[0.0, 0.0, 1.5], [0.2, -0.1, 1.8], [-0.2, 0.1, 2.0]],
        dtype=torch.float32,
    )
    n = means.shape[0]
    return {
        "means": means,
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(n, 1),
        "scales": torch.full((n, 3), -3.0, dtype=torch.float32),
        "opacities": torch.full((n,), 2.0, dtype=torch.float32),
        "sh0": torch.zeros((n, 1, 3), dtype=torch.float32),
        "shN": torch.zeros((n, 15, 3), dtype=torch.float32),
    }


def _write_flat_checkpoint(path: Path):
    torch.save({"step": 7, "splats": _fake_splats()}, path)


class CameraPathParsingTest(unittest.TestCase):
    def test_viser_camera_path_parses_and_records_conversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "path.json"
            path.write_text(
                json.dumps(
                    {
                        "camera_path": [
                            {
                                "camera_to_world": [
                                    1.0, 0.0, 0.0, 0.1,
                                    0.0, 1.0, 0.0, 0.2,
                                    0.0, 0.0, 1.0, 0.3,
                                    0.0, 0.0, 0.0, 1.0,
                                ],
                                "fov": 50.0,
                            }
                        ],
                        "render_width": 320,
                        "render_height": 180,
                        "default_fov": 50.0,
                        "scale_ratio": 10.0,
                        "keyframes": [{"name": "a"}, {"name": "b"}],
                    }
                )
            )

            camera_path = brs.load_camera_path(path, trajectory_id="test_path")

        self.assertEqual(camera_path.trajectory_id, "test_path")
        self.assertEqual(len(camera_path.cameras), 1)
        self.assertEqual(camera_path.cameras[0].width, 320)
        self.assertEqual(camera_path.cameras[0].height, 180)
        self.assertAlmostEqual(float(camera_path.cameras[0].c2w[0, 3]), 1.0)
        self.assertAlmostEqual(float(camera_path.cameras[0].c2w[1, 1]), -1.0)
        self.assertAlmostEqual(float(camera_path.cameras[0].c2w[2, 2]), -1.0)
        self.assertTrue(camera_path.metadata["conversion_required"])
        self.assertEqual(camera_path.metadata["keyframe_count"], 2)

    def test_malformed_camera_data_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text(json.dumps({"camera_path": [{"fov": 50.0}]}))

            with self.assertRaisesRegex(ValueError, "no camera_to_world"):
                brs.load_camera_path(path)


class BenchmarkConfigTest(unittest.TestCase):
    def test_config_loading_and_trajectory_association(self):
        flat = brs.load_config("FLAT-REAL-FIT")
        city = brs.load_config("FLAT-CITYGS-FULL")
        finest = brs.load_config("CG-FINEST-FLAT")
        fixed = brs.load_config("CG-FIXED-FINEST-ALL")

        self.assertEqual(flat["configuration_id"], "FLAT-REAL-FIT")
        self.assertEqual(city["configuration_id"], "FLAT-CITYGS-FULL")
        self.assertEqual(finest["configuration_id"], "CG-FINEST-FLAT")
        self.assertEqual(fixed["configuration_id"], "CG-FIXED-FINEST-ALL")
        self.assertEqual(flat["trajectory_id"], "matrixcity_1m2_navigation_v1")
        self.assertEqual(city["trajectory_id"], "citygaussian_23m_navigation_v1")
        self.assertEqual(finest["trajectory_id"], "citygaussian_23m_navigation_v1")
        self.assertEqual(fixed["trajectory_id"], "citygaussian_23m_navigation_v1")
        self.assertEqual(finest["sh_degree"], 2)
        self.assertEqual(fixed["scene_type"], "citygaussian_fixed_all")
        self.assertEqual(fixed["block_dim"], [6, 6, 1])
        self.assertFalse(Path(flat["camera_path"]).is_absolute())
        self.assertIn("asset_env_var", flat)
        self.assertTrue(brs.resolve_camera_path(flat).exists())
        self.assertTrue(brs.resolve_camera_path(city).exists())
        self.assertTrue(brs.resolve_camera_path(finest).exists())
        self.assertTrue(brs.resolve_camera_path(fixed).exists())

    def test_missing_asset_path_fails_clearly(self):
        config = brs.load_config("FLAT-REAL-FIT")

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(brs.MissingAssetError):
                brs.resolve_asset_path(config, None)


class BenchmarkSchemaTest(unittest.TestCase):
    def test_memory_stage_schema_handles_unavailable_gpu_metrics(self):
        snapshot = brs.gpu_memory_snapshot(torch.device("cpu"), "process_start")

        self.assertEqual(snapshot["measurement_stage"], "process_start")
        self.assertEqual(snapshot["active_device"], "cpu")
        self.assertIsNone(snapshot["torch_current_allocated_bytes"])
        self.assertIsNone(snapshot["torch_current_reserved_bytes"])
        self.assertIsNone(snapshot["torch_peak_allocated_bytes"])
        self.assertIsNone(snapshot["torch_peak_reserved_bytes"])
        self.assertFalse(snapshot["nvml_available"])
        self.assertFalse(snapshot["nvml_process_accounting_supported"])
        self.assertFalse(snapshot["nvml_process_accounting_reliable"])
        json.dumps(brs.json_safe(snapshot), allow_nan=False)

    def test_memory_stage_rejects_unknown_stage(self):
        with self.assertRaisesRegex(ValueError, "Unknown memory measurement stage"):
            brs.gpu_memory_snapshot(torch.device("cpu"), "mystery")

    def test_peak_helpers_use_byte_values_without_stale_carryover(self):
        first = {
            "torch_peak_allocated_bytes": 1024.0,
            "torch_peak_reserved_bytes": 4096.0,
        }
        second = {
            "torch_peak_allocated_bytes": 2048.0,
            "torch_peak_reserved_bytes": 3072.0,
        }

        self.assertEqual(brs.peak_from_snapshot(first), first)
        self.assertEqual(
            brs.max_peak_dicts(first, second),
            {
                "torch_peak_allocated_bytes": 2048.0,
                "torch_peak_reserved_bytes": 4096.0,
            },
        )

    def test_sample_camera_indices_are_deterministic_and_include_ends(self):
        indices = brs.sample_camera_indices(frame_count=1440, sample_count=12)

        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 1439)
        self.assertEqual(len(indices), 12)
        self.assertEqual(indices, brs.sample_camera_indices(frame_count=1440, sample_count=12))

    def test_render_one_frame_records_finite_fields_and_resets_render_peak(self):
        config = brs.load_config("FLAT-REAL-FIT")
        camera = brs.CameraSpec(
            camera_id="cam0",
            c2w=torch.eye(4).numpy(),
            K=torch.eye(3).numpy(),
            width=4,
            height=3,
        )

        class FakeScene:
            def prepare_frame(self, frame_camera, settings):
                return ActiveSplats(
                    means=torch.zeros((1, 3)),
                    quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                    scales=torch.ones((1, 3)),
                    opacities=torch.ones((1,)),
                    colors=torch.zeros((1, 16, 3)),
                    scene_metrics=SceneFrameMetrics(
                        active_blocks=1,
                        active_gaussians=1,
                        active_gaussian_fraction=1.0,
                    ),
                )

        class FakeRenderer:
            def render(self, active_splats, frame_camera, settings, log_gpu_events=False):
                return SimpleNamespace(
                    image=torch.zeros((3, 4, 3), dtype=torch.float32),
                    alpha=torch.ones((3, 4, 1), dtype=torch.float32),
                    render_time_ms=1.5,
                    gpu_render_time_ms=float("nan"),
                    visible_gaussians=1.0,
                )

        with patch.object(brs, "reset_torch_peak_memory") as reset_peak:
            result = brs.render_one_frame(
                FakeScene(),
                FakeRenderer(),
                camera,
                config,
                scene_total_gaussians=1,
                scene_sh_degree=3,
                device=torch.device("cpu"),
                log_gpu_events=False,
            )

        reset_peak.assert_called_once()
        self.assertTrue(result["finite_image"])
        self.assertTrue(result["finite_alpha"])
        self.assertEqual(result["image_shape"], (3, 4, 3))
        self.assertEqual(result["alpha_shape"], (3, 4, 1))
        self.assertEqual(result["render_call_time_ms"], 1.5)
        self.assertGreater(result["end_to_end_frame_time_ms"], 1.5)
        self.assertNotIn("frame_time_ms", result)
        self.assertNotIn("fps", result)
        self.assertEqual(result["measurement_stage"], "result_collection")
        self.assertIn("nvml_process_accounting_supported", result)

    def test_summary_percentiles_and_fps_are_derived_consistently(self):
        config = brs.load_config("FLAT-REAL-FIT")
        camera_path = brs.CameraPath(
            path=Path("camera.json"),
            trajectory_id="traj",
            cameras=[],
            metadata={"frame_count": 4},
        )
        rows = [
            {
                "render_call_time_ms": value,
                "cuda_rasterization_time_ms": value * 0.5,
                "preparation_time_ms": 1.0,
                "end_to_end_frame_time_ms": value + 1.0,
                "torch_peak_allocated_bytes": 100.0 + value,
                "torch_peak_reserved_bytes": 200.0 + value,
                "process_ram_bytes": 300.0 + value,
                "total_gaussian_count": 3,
                "active_gaussian_count": 3,
                "resident_gaussian_count": 3,
                "all_loaded_gaussians_resident": True,
                "all_loaded_gaussians_active": True,
                "visible_gaussians": 2,
            }
            for value in [10.0, 20.0, 30.0, 40.0]
        ]

        summary = brs.summarize_rows(
            rows,
            record_kind="benchmark_validation",
            config=config,
            asset={
                "identifier": "ckpt.pt",
                "checkpoint_type": "gsplat_checkpoint",
                "path_supplied": "ckpt.pt",
            },
            camera_path=camera_path,
            environment={},
        )

        self.assertEqual(summary["record_kind"], "benchmark_validation")
        self.assertEqual(summary["frame_count"], 4)
        self.assertEqual(summary["mean_render_call_time_ms"], 25.0)
        self.assertEqual(summary["timing"]["render_call_time_ms"]["median"], 25.0)
        self.assertAlmostEqual(summary["timing"]["render_call_time_ms"]["p95"], 38.5)
        self.assertEqual(summary["mean_end_to_end_frame_time_ms"], 26.0)
        self.assertAlmostEqual(summary["end_to_end_fps"], 1000.0 / 26.0)
        self.assertNotIn("mean_frame_time_ms", summary)

    def test_citygaussian_integrity_metadata_disables_selection_lod_and_streaming(self):
        config = brs.load_config("CG-FIXED-FINEST-ALL")

        integrity = brs.benchmark_integrity_metadata(config)

        self.assertFalse(integrity["block_manager_used"])
        self.assertFalse(integrity["citygaussian_runtime_selection_used"])
        self.assertFalse(integrity["citygaussian_visibility_selection_used"])
        self.assertFalse(integrity["adaptive_lod_used"])
        self.assertFalse(integrity["streaming_used"])
        self.assertTrue(integrity["citygaussian_cell_construction_used"])
        self.assertIsNone(integrity["all_loaded_gaussians_active"])
        self.assertTrue(integrity["all_loaded_gaussians_active_expected"])

    def test_residency_and_activity_policies_are_distinct(self):
        cases = {
            "CG-FINEST-FLAT": (True, False),
            "CG-FIXED-FINEST-ALL": (True, False),
            "CG-FIXED-FINEST-VISIBILITY": (False, False),
            "CG-LOD-RELEASE": (False, True),
        }
        for config_id, (all_active, lod_used) in cases.items():
            with self.subTest(config_id=config_id):
                integrity = brs.benchmark_integrity_metadata(brs.load_config(config_id))
                self.assertTrue(integrity["all_loaded_gaussians_resident_expected"])
                self.assertEqual(integrity["all_loaded_gaussians_active_expected"], all_active)
                self.assertEqual(integrity["lod_selection_used"], lod_used)
                self.assertIsNone(integrity["all_loaded_gaussians_active"])

    def test_feasibility_result_schema(self):
        config = brs.load_config("FLAT-CITYGS-FULL")
        camera_path = brs.CameraPath(
            path=Path("camera.json"),
            trajectory_id="city",
            cameras=[],
            metadata={"frame_count": 1},
        )

        result = brs.build_feasibility_result(
            config=config,
            asset={
                "identifier": "point_cloud.ply",
                "checkpoint_type": "gaussian_ply",
                "path_supplied": "point_cloud.ply",
                "file_size_bytes": 123,
                "ply_vertex_count": 23,
            },
            camera_path=camera_path,
            status="oom",
            failure_stage="gpu_scene_construction",
            environment={},
            cpu_before={"process_ram_bytes": 1, "system_ram_used_bytes": 2},
            cpu_after={"process_ram_bytes": 3, "system_ram_used_bytes": 4},
            gpu_memory={
                "allocated_bytes": 5,
                "reserved_bytes": 6,
                "peak_allocated_bytes": 7,
                "peak_reserved_bytes": 8,
            },
            exception=torch.cuda.OutOfMemoryError("test"),
            final_successful_stage="cpu_tensor_construction",
            stage_memory={"process_ram_after_cpu_tensor_construction_bytes": 9},
        )

        self.assertEqual(result["record_kind"], "feasibility_result")
        self.assertEqual(result["configuration_id"], "FLAT-CITYGS-FULL")
        self.assertEqual(result["status"], "oom")
        self.assertEqual(result["failure_stage"], "gpu_scene_construction")
        self.assertEqual(result["final_successful_stage"], "cpu_tensor_construction")
        self.assertEqual(result["process_ram_after_cpu_tensor_construction_bytes"], 9)
        self.assertEqual(result["total_gaussians"], 23)
        self.assertFalse(result["integrity"]["scene_type_flat"])
        self.assertFalse(result["integrity"]["all_loaded_gaussians_active"])

    def test_feasibility_reports_metadata_stage_failures(self):
        config = brs.load_config("FLAT-CITYGS-FULL")
        camera_path = brs.CameraPath(
            path=Path("camera.json"),
            trajectory_id="city",
            cameras=[
                brs.CameraSpec(
                    camera_id="cam0",
                    c2w=torch.eye(4).numpy(),
                    K=torch.eye(3).numpy(),
                    width=8,
                    height=6,
                )
            ],
            metadata={"frame_count": 1},
        )

        def fail_during_metadata(asset_path, device, progress_callback=None):
            if progress_callback is not None:
                progress_callback("ply_parsing")
                progress_callback("cpu_tensor_construction")
                progress_callback("bounds_or_metadata_computation")
            raise RuntimeError("metadata boom")

        with tempfile.TemporaryDirectory() as tmpdir:
            asset = Path(tmpdir) / "point_cloud.ply"
            asset.write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
            output_dir = Path(tmpdir) / "out"

            with patch.object(brs, "load_flat_scene_for_benchmark", side_effect=fail_during_metadata):
                rc = brs.run_feasibility(
                    config=config,
                    asset_path=asset,
                    asset_supplied=str(asset),
                    camera_path=camera_path,
                    output_dir=output_dir,
                    max_frames=1,
                    device=torch.device("cpu"),
                    log_gpu_events=False,
                )

            result = json.loads((output_dir / brs.FEASIBILITY_JSON).read_text())

        self.assertEqual(rc, 0)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_stage"], "bounds_or_metadata_computation")
        self.assertEqual(result["final_successful_stage"], "cpu_tensor_construction")
        self.assertEqual(result["exception_type"], "RuntimeError")


class FlatBenchmarkIntegrityTest(unittest.TestCase):
    def test_flat_scene_enforcement_and_no_block_manager_use(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "flat.pt"
            _write_flat_checkpoint(ckpt)

            with patch("scene_runtime.scenes.BlockManager.from_file") as from_file:
                scene, stats = brs.load_flat_scene_for_benchmark(ckpt, torch.device("cpu"))

            frame_camera = FrameCamera(
                c2w=torch.eye(4),
                K=torch.tensor(
                    [[20.0, 0.0, 16.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                ),
                width=32,
                height=24,
            )
            active = scene.prepare_frame(
                frame_camera,
                RenderSettings(width=32, height=24, near_plane=0.1, far_plane=10.0),
            )

        from_file.assert_not_called()
        self.assertEqual(stats.scene_type, "flat")
        self.assertEqual(active.num_gaussians, stats.total_gaussians)
        self.assertEqual(active.scene_metrics.active_blocks, 1)

    def test_blocked_asset_is_rejected_for_flat_configuration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "flat.pt"
            blocked = Path(tmpdir) / "blocked.pt"
            _write_flat_checkpoint(ckpt)
            build_partitioned_checkpoint([ckpt], blocked, grid=(1, 1), overwrite=False)

            with self.assertRaisesRegex(ValueError, "scene_type='flat' does not match"):
                brs.load_flat_scene_for_benchmark(blocked, torch.device("cpu"))

    def test_benchmark_module_does_not_depend_on_interactive_viewer(self):
        source = inspect.getsource(brs)

        self.assertNotIn("simple_viewer", source)
        self.assertNotIn("gsplat_viewer", source)
        self.assertIn("GsplatRenderer", source)


if __name__ == "__main__":
    unittest.main()
