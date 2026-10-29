import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from block_manager import BlockManager, aabbs_intersect_frustum, build_partitioned_checkpoint


def _write_gaussian_ply(path: Path, rows):
    properties = [
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
        ("nx", "float"),
        ("ny", "float"),
        ("nz", "float"),
        ("f_dc_0", "float"),
        ("f_dc_1", "float"),
        ("f_dc_2", "float"),
    ]
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
            values = [float(row.get(name, 0.0)) for name, _ in properties]
            f.write(struct.pack("<" + "f" * len(values), *values))


def _fake_splats():
    means = torch.tensor(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.2],
            [-1.0, 1.0, 0.4],
            [1.0, 1.0, 0.6],
            [0.0, 0.0, 0.8],
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


def _test_camera_args():
    return (
        torch.eye(4),
        torch.tensor(
            [
                [100.0, 0.0, 50.0],
                [0.0, 100.0, 50.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        100,
        100,
        0.1,
        10.0,
    )


def _render_selected(manager: BlockManager, block_id: int):
    c2w, K, width, height, near_plane, far_plane = _test_camera_args()
    return manager.get_active_splats(
        c2w,
        K,
        width=width,
        height=height,
        near_plane=near_plane,
        far_plane=far_plane,
        culling_mode="selected_block_only",
        selected_block_id=block_id,
    )


class PartitionCheckpointTest(unittest.TestCase):
    def test_partition_accepts_gaussian_ply(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "splats.ply"
            out_path = Path(tmpdir) / "blocks.pt"
            _write_gaussian_ply(
                ply_path,
                [
                    {"x": 1.0, "y": 2.0, "z": 3.0, "opacity": 0.0, "rot_0": 1.0},
                    {"x": 4.0, "y": 5.0, "z": 6.0, "opacity": 0.0, "rot_0": 1.0},
                ],
            )

            build_partitioned_checkpoint(
                [ply_path],
                out_path,
                grid=(1, 1),
                overwrite=False,
                ply_coordinate_mode="supersplat",
            )
            payload = torch.load(out_path, map_location="cpu", weights_only=False)
            block = payload["blocks"][0]

            self.assertEqual(payload["source_steps"], [-1])
            self.assertEqual(payload["ply_coordinate_mode"], "supersplat")
            self.assertEqual(int(payload["total_gaussians"]), 2)
            self.assertTrue(
                torch.allclose(
                    block["tensors"]["means"],
                    torch.tensor([[-1.0, -2.0, 3.0], [-4.0, -5.0, 6.0]]),
                )
            )

    def test_partition_assigns_each_gaussian_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            out_path = Path(tmpdir) / "blocks.pt"
            torch.save({"step": 7, "splats": _fake_splats()}, ckpt_path)

            build_partitioned_checkpoint([ckpt_path], out_path, grid=(2, 2), overwrite=False)
            payload = torch.load(out_path, map_location="cpu", weights_only=False)

            counts = [int(block["gaussian_count"]) for block in payload["blocks"]]
            self.assertEqual(sum(counts), int(payload["total_gaussians"]))
            self.assertEqual(payload["total_gaussians"], 5)

            for block in payload["blocks"]:
                means = block["tensors"]["means"]
                content_min = block["content_aabb_min"]
                content_max = block["content_aabb_max"]
                cull_min = block["cull_aabb_min"]
                cull_max = block["cull_aabb_max"]
                self.assertTrue(torch.all(means >= content_min))
                self.assertTrue(torch.all(means <= content_max))
                self.assertTrue(torch.all(cull_min <= content_min))
                self.assertTrue(torch.all(cull_max >= content_max))

            manager = BlockManager.from_file(out_path, torch.device("cpu"), residency="cpu")
            active = manager.get_active_splats(
                torch.eye(4),
                torch.tensor(
                    [
                        [100.0, 0.0, 50.0],
                        [0.0, 100.0, 50.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                width=100,
                height=100,
                near_plane=0.1,
                far_plane=10.0,
                force_all=True,
            )
            self.assertEqual(active.num_gaussians, 5)
            self.assertEqual(active.stats.total_blocks, len(payload["blocks"]))
            self.assertEqual(active.stats.active_gaussians, 5)

            none_active = manager.get_active_splats(
                torch.eye(4),
                torch.tensor(
                    [
                        [100.0, 0.0, 50.0],
                        [0.0, 100.0, 50.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                width=100,
                height=100,
                near_plane=0.1,
                far_plane=10.0,
                culling_mode="force_no_blocks_active",
            )
            self.assertEqual(none_active.num_gaussians, 0)
            self.assertEqual(none_active.stats.active_blocks, 0)

            selected_id = payload["blocks"][0]["block_id"]
            selected_count = payload["blocks"][0]["gaussian_count"]
            selected_active = manager.get_active_splats(
                torch.eye(4),
                torch.tensor(
                    [
                        [100.0, 0.0, 50.0],
                        [0.0, 100.0, 50.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                width=100,
                height=100,
                near_plane=0.1,
                far_plane=10.0,
                culling_mode="selected_block_only",
                selected_block_id=selected_id,
            )
            self.assertEqual(selected_active.num_gaussians, selected_count)
            self.assertEqual(selected_active.stats.active_block_ids, str(selected_id))

            invalid_selected = manager.get_active_splats(
                torch.eye(4),
                torch.tensor(
                    [
                        [100.0, 0.0, 50.0],
                        [0.0, 100.0, 50.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                width=100,
                height=100,
                near_plane=0.1,
                far_plane=10.0,
                culling_mode="selected_block_only",
                selected_block_id=-999,
            )
            self.assertEqual(invalid_selected.num_gaussians, 0)
            self.assertEqual(invalid_selected.stats.active_blocks, 0)

    def test_cpu_residency_uploads_reuses_and_tracks_persistent_cache_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            out_path = Path(tmpdir) / "blocks.pt"
            torch.save({"step": 7, "splats": _fake_splats()}, ckpt_path)
            build_partitioned_checkpoint([ckpt_path], out_path, grid=(2, 2), overwrite=False)

            manager = BlockManager.from_file(out_path, torch.device("cpu"), residency="cpu")
            block_id = manager.blocks[0].block_id
            block_bytes = manager.blocks[0].cpu_bytes

            self.assertEqual(manager.resident_block_ids(), [])
            self.assertEqual(manager.block_residency_state(block_id), "CPU_ONLY")

            first = _render_selected(manager, block_id)
            self.assertEqual(first.stats.cache_hits, 0)
            self.assertEqual(first.stats.cache_misses, 1)
            self.assertEqual(first.stats.newly_uploaded_blocks, 1)
            self.assertEqual(first.stats.newly_uploaded_gaussians, manager.blocks[0].gaussian_count)
            self.assertEqual(first.stats.resident_blocks, 1)
            self.assertEqual(first.stats.gpu_resident_bytes, block_bytes)
            self.assertEqual(manager.block_residency_state(block_id), "GPU_RESIDENT")
            self.assertEqual(manager.blocks[0].last_used_frame, manager.frame_index)

            second = _render_selected(manager, block_id)
            self.assertEqual(second.stats.cache_hits, 1)
            self.assertEqual(second.stats.cache_misses, 0)
            self.assertEqual(second.stats.newly_uploaded_blocks, 0)
            self.assertEqual(second.stats.gpu_resident_bytes, block_bytes)
            self.assertEqual(second.stats.cache_hit_rate, 1.0)

    def test_cpu_residency_lru_evicts_only_inactive_blocks_over_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            out_path = Path(tmpdir) / "blocks.pt"
            torch.save({"step": 7, "splats": _fake_splats()}, ckpt_path)
            build_partitioned_checkpoint([ckpt_path], out_path, grid=(2, 2), overwrite=False)

            inspector = BlockManager.from_file(out_path, torch.device("cpu"), residency="cpu")
            block_ids = [block.block_id for block in inspector.blocks[:2]]
            one_block_budget_bytes = max(inspector.blocks_by_id[block_id].cpu_bytes for block_id in block_ids)
            budget_mb = (one_block_budget_bytes + 1) / (1024.0 * 1024.0)
            manager = BlockManager.from_file(
                out_path,
                torch.device("cpu"),
                residency="cpu",
                gpu_block_budget_mb=budget_mb,
            )

            first = _render_selected(manager, block_ids[0])
            self.assertEqual(first.stats.evicted_blocks, 0)
            self.assertEqual(manager.resident_block_ids(), [block_ids[0]])

            second = _render_selected(manager, block_ids[1])
            self.assertEqual(second.stats.newly_uploaded_blocks, 1)
            self.assertEqual(second.stats.evicted_blocks, 1)
            self.assertEqual(second.stats.evicted_gaussians, manager.blocks_by_id[block_ids[0]].gaussian_count)
            self.assertEqual(manager.resident_block_ids(), [block_ids[1]])
            self.assertEqual(manager.block_residency_state(block_ids[0]), "CPU_ONLY")
            self.assertEqual(manager.block_residency_state(block_ids[1]), "GPU_RESIDENT")
            self.assertFalse(bool(second.stats.active_set_exceeds_budget))

    def test_active_block_survives_budget_overflow_and_reports_exceeded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            out_path = Path(tmpdir) / "blocks.pt"
            torch.save({"step": 7, "splats": _fake_splats()}, ckpt_path)
            build_partitioned_checkpoint([ckpt_path], out_path, grid=(2, 2), overwrite=False)

            manager = BlockManager.from_file(
                out_path,
                torch.device("cpu"),
                residency="cpu",
                gpu_block_budget_mb=0.0,
            )
            block_id = manager.blocks[0].block_id
            active = _render_selected(manager, block_id)

            self.assertEqual(active.stats.active_blocks, 1)
            self.assertEqual(active.stats.resident_blocks, 1)
            self.assertEqual(active.stats.evicted_blocks, 0)
            self.assertTrue(bool(active.stats.active_set_exceeds_budget))
            self.assertTrue(bool(active.stats.gpu_budget_exceeded))
            self.assertEqual(manager.block_residency_state(block_id), "GPU_RESIDENT")

    def test_gpu_resident_mode_reports_all_blocks_without_cache_uploads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            out_path = Path(tmpdir) / "blocks.pt"
            torch.save({"step": 7, "splats": _fake_splats()}, ckpt_path)
            build_partitioned_checkpoint([ckpt_path], out_path, grid=(2, 2), overwrite=False)

            manager = BlockManager.from_file(out_path, torch.device("cpu"), residency="gpu")
            c2w, K, width, height, near_plane, far_plane = _test_camera_args()
            active = manager.get_active_splats(
                c2w,
                K,
                width=width,
                height=height,
                near_plane=near_plane,
                far_plane=far_plane,
                force_all=True,
            )

            expected_bytes = sum(block.gpu_bytes for block in manager.blocks)
            self.assertEqual(active.num_gaussians, 5)
            self.assertEqual(active.stats.resident_blocks, manager.total_blocks)
            self.assertEqual(active.stats.resident_gaussians, manager.total_gaussians)
            self.assertEqual(active.stats.cache_hits, 0)
            self.assertEqual(active.stats.cache_misses, 0)
            self.assertEqual(active.stats.gpu_resident_bytes, expected_bytes)

    def test_cleanup_metadata_filters_bounds_without_pruning_block_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            out_path = Path(tmpdir) / "blocks.pt"
            metadata_path = Path(tmpdir) / "cleanup_metadata.json"
            splats = _fake_splats()
            splats["means"] = torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.5],
                    [2.0, 2.0, 2.0],
                    [100.0, 100.0, 100.0],
                ],
                dtype=torch.float32,
            )
            n = int(splats["means"].shape[0])
            splats["quats"] = splats["quats"][:1].repeat(n, 1)
            splats["scales"] = torch.zeros((n, 3), dtype=torch.float32)
            splats["opacities"] = torch.zeros((n,), dtype=torch.float32)
            splats["sh0"] = torch.zeros((n, 1, 3), dtype=torch.float32)
            splats["shN"] = torch.zeros((n, 15, 3), dtype=torch.float32)
            torch.save({"step": 7, "splats": splats}, ckpt_path)
            metadata_path.write_text(
                json.dumps(
                    {
                        "bounds_filter_config": {
                            "opacity_min": None,
                            "opacity_percentile": None,
                            "max_scale": None,
                            "max_scale_percentile": None,
                            "roi_min": None,
                            "roi_max": [10.0, 10.0, 10.0],
                            "position_percentile_min": 0.0,
                            "position_percentile_max": 100.0,
                            "padding": 0.0,
                        }
                    }
                )
            )

            build_partitioned_checkpoint(
                [ckpt_path],
                out_path,
                grid=(1, 1),
                scale_padding_factor=0.0,
                overwrite=False,
                cleanup_metadata_path=metadata_path,
            )
            payload = torch.load(out_path, map_location="cpu", weights_only=False)
            block = payload["blocks"][0]

            self.assertEqual(int(block["tensors"]["means"].shape[0]), 4)
            self.assertEqual(int(block["raw_gaussian_count"]), 4)
            self.assertEqual(int(block["bounds_valid_gaussian_count"]), 3)
            self.assertTrue(torch.allclose(block["raw_gaussian_bounds_max"], torch.tensor([100.0, 100.0, 100.0])))
            self.assertTrue(torch.allclose(block["filtered_render_bounds_max"], torch.tensor([2.0, 2.0, 2.0])))
            self.assertTrue(torch.allclose(block["cull_aabb_max"], torch.tensor([2.0, 2.0, 2.0])))

            manager = BlockManager.from_file(out_path, torch.device("cpu"), residency="cpu")
            debug = manager.debug_blocks()[0]
            self.assertEqual(debug.raw_gaussian_count, 4)
            self.assertEqual(debug.bounds_valid_gaussian_count, 3)
            self.assertEqual(debug.filtered_render_bounds_max, (2.0, 2.0, 2.0))

    def test_metadata_bounds_default_to_exact_filtered_extent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            exact_out_path = Path(tmpdir) / "exact_blocks.pt"
            robust_out_path = Path(tmpdir) / "robust_blocks.pt"
            metadata_path = Path(tmpdir) / "cleanup_metadata.json"
            splats = _fake_splats()
            torch.save({"step": 7, "splats": splats}, ckpt_path)
            metadata_path.write_text(
                json.dumps(
                    {
                        "bounds_filter_config": {
                            "opacity_min": None,
                            "opacity_percentile": None,
                            "max_scale": None,
                            "max_scale_percentile": None,
                            "roi_min": None,
                            "roi_max": None,
                            "position_percentile_min": 50.0,
                            "position_percentile_max": 50.0,
                            "padding": 0.0,
                        }
                    }
                )
            )

            build_partitioned_checkpoint(
                [ckpt_path],
                exact_out_path,
                grid=(1, 1),
                scale_padding_factor=0.0,
                cleanup_metadata_path=metadata_path,
            )
            exact_block = torch.load(
                exact_out_path,
                map_location="cpu",
                weights_only=False,
            )["blocks"][0]

            self.assertTrue(
                torch.allclose(
                    exact_block["filtered_render_bounds_min"],
                    torch.tensor([-1.0, -1.0, 0.0]),
                )
            )
            self.assertTrue(
                torch.allclose(
                    exact_block["filtered_render_bounds_max"],
                    torch.tensor([1.0, 1.0, 0.8]),
                )
            )
            self.assertTrue(
                torch.allclose(
                    exact_block["cull_aabb_min"],
                    exact_block["raw_gaussian_bounds_min"],
                )
            )
            self.assertTrue(
                torch.allclose(
                    exact_block["cull_aabb_max"],
                    exact_block["raw_gaussian_bounds_max"],
                )
            )

            build_partitioned_checkpoint(
                [ckpt_path],
                robust_out_path,
                grid=(1, 1),
                scale_padding_factor=0.0,
                cleanup_metadata_path=metadata_path,
                metadata_bounds_mode="robust",
            )
            robust_block = torch.load(
                robust_out_path,
                map_location="cpu",
                weights_only=False,
            )["blocks"][0]

            self.assertTrue(
                torch.allclose(
                    robust_block["filtered_render_bounds_min"],
                    torch.tensor([0.0, 0.0, 0.4]),
                )
            )
            self.assertTrue(
                torch.allclose(
                    robust_block["filtered_render_bounds_max"],
                    torch.tensor([0.0, 0.0, 0.4]),
                )
            )

    def test_scale_padding_uses_percentile_and_extent_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            robust_out_path = Path(tmpdir) / "robust_blocks.pt"
            capped_out_path = Path(tmpdir) / "capped_blocks.pt"
            splats = _fake_splats()
            splats["means"] = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
            n = int(splats["means"].shape[0])
            splats["quats"] = splats["quats"][:1].repeat(n, 1)
            splats["scales"] = torch.log(
                torch.tensor(
                    [
                        [0.01, 0.01, 0.01],
                        [0.01, 0.01, 0.01],
                        [0.01, 0.01, 0.01],
                        [1.0, 1.0, 1.0],
                    ],
                    dtype=torch.float32,
                )
            )
            splats["opacities"] = torch.zeros((n,), dtype=torch.float32)
            splats["sh0"] = torch.zeros((n, 1, 3), dtype=torch.float32)
            splats["shN"] = torch.zeros((n, 15, 3), dtype=torch.float32)
            torch.save({"step": 7, "splats": splats}, ckpt_path)

            build_partitioned_checkpoint(
                [ckpt_path],
                robust_out_path,
                grid=(1, 1),
                scale_padding_factor=3.0,
                scale_padding_percentile=50.0,
                max_padding_fraction=None,
            )
            robust_block = torch.load(
                robust_out_path,
                map_location="cpu",
                weights_only=False,
            )["blocks"][0]

            self.assertTrue(
                torch.allclose(
                    robust_block["cull_aabb_min"],
                    torch.tensor([-0.03, -0.03, -0.03]),
                )
            )
            self.assertTrue(
                torch.allclose(
                    robust_block["cull_aabb_max"],
                    torch.tensor([3.03, 0.03, 0.03]),
                )
            )

            build_partitioned_checkpoint(
                [ckpt_path],
                capped_out_path,
                grid=(1, 1),
                scale_padding_factor=3.0,
                scale_padding_percentile=100.0,
                max_padding_fraction=0.1,
            )
            capped_block = torch.load(
                capped_out_path,
                map_location="cpu",
                weights_only=False,
            )["blocks"][0]

            self.assertTrue(
                torch.allclose(
                    capped_block["cull_aabb_min"],
                    torch.tensor([-0.3, -0.3, -0.3]),
                )
            )
            self.assertTrue(
                torch.allclose(
                    capped_block["cull_aabb_max"],
                    torch.tensor([3.3, 0.3, 0.3]),
                )
            )


class FrustumAabbTest(unittest.TestCase):
    def test_frustum_aabb_cases(self):
        c2w = torch.eye(4)
        K = torch.tensor(
            [
                [100.0, 0.0, 50.0],
                [0.0, 100.0, 50.0],
                [0.0, 0.0, 1.0],
            ]
        )
        mins = torch.tensor(
            [
                [-0.5, -0.5, 4.0],
                [-0.5, -0.5, -5.0],
                [10.0, -0.5, 4.0],
                [-0.5, -0.5, 20.0],
                [2.4, -0.5, 4.0],
            ],
            dtype=torch.float32,
        )
        maxs = torch.tensor(
            [
                [0.5, 0.5, 5.0],
                [0.5, 0.5, -4.0],
                [11.0, 0.5, 5.0],
                [0.5, 0.5, 21.0],
                [2.6, 0.5, 5.0],
            ],
            dtype=torch.float32,
        )
        result = aabbs_intersect_frustum(
            mins,
            maxs,
            c2w,
            K,
            width=100,
            height=100,
            near_plane=0.1,
            far_plane=10.0,
        )
        self.assertEqual(result.tolist(), [True, False, False, False, True])


if __name__ == "__main__":
    unittest.main()
