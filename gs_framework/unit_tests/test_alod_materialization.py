import unittest

import numpy as np
import torch

from scene_runtime import FrameCamera, RenderSettings
from scene_runtime.alod import (
    AlodCompactCullingMetadata,
    conservative_pinhole_materialization,
    fast_conservative_pinhole_materialization,
    select_multiset_materialization,
    stable_unique_with_inverse,
)


class AlodMaterializationTest(unittest.TestCase):
    def setUp(self):
        self.means = np.asarray(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, -2.0],
                [100.0, 0.0, 2.0],
                [100.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        )
        log_scales = np.asarray(
            [
                [-5.0, -5.0, -5.0],
                [-5.0, -5.0, -5.0],
                [-5.0, -5.0, -5.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        self.metadata = AlodCompactCullingMetadata.from_log_scales(
            self.means, log_scales
        )
        self.camera = FrameCamera(
            c2w=torch.eye(4),
            w2c=torch.eye(4),
            K=torch.tensor(
                [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
            ),
            width=100,
            height=100,
        )
        self.settings = RenderSettings(
            width=100,
            height=100,
            near_plane=0.01,
            far_plane=100.0,
            camera_model="pinhole",
        )

    def test_compact_metadata_accounts_logical_and_incremental_bytes(self):
        self.assertEqual(self.metadata.bytes_per_row, 16)
        self.assertEqual(self.metadata.logical_bytes, 4 * 16)
        self.assertEqual(self.metadata.incremental_bytes, 4 * 4)

    def test_filter_preserves_order_and_stable_global_ids(self):
        semantic = np.asarray([3, 2, 1, 0], dtype=np.int64)
        result = conservative_pinhole_materialization(
            self.metadata, semantic, self.camera, self.settings
        )
        np.testing.assert_array_equal(
            result.semantic_selected_global_node_ids, semantic
        )
        np.testing.assert_array_equal(result.materialization_global_node_ids, [3, 0])
        np.testing.assert_array_equal(result.raster_global_node_ids, [3, 0])
        np.testing.assert_array_equal(result.kept_semantic_indices, [0, 3])
        self.assertEqual(result.rejected_depth_rows, 1)
        self.assertEqual(result.rejected_screen_bound_rows, 1)

    def test_stable_unique_mapping_reconstructs_ordered_multiset(self):
        semantic = np.asarray([3, 3, 2, 0, 2], dtype=np.int64)
        unique_ids, inverse = stable_unique_with_inverse(semantic)
        np.testing.assert_array_equal(unique_ids, [3, 2, 0])
        np.testing.assert_array_equal(inverse, [0, 0, 1, 2, 1])
        np.testing.assert_array_equal(unique_ids[inverse], semantic)

    def test_multiset_filter_evaluates_unique_rows_and_restores_occurrences(self):
        semantic = np.asarray([3, 3, 2, 1, 0, 0], dtype=np.int64)
        result = select_multiset_materialization(
            "conservative_pinhole",
            self.metadata,
            semantic,
            self.camera,
            self.settings,
        )
        np.testing.assert_array_equal(
            result.unique_semantic_global_node_ids, [3, 2, 1, 0]
        )
        np.testing.assert_array_equal(
            result.semantic_to_unique_indices, [0, 0, 1, 2, 3, 3]
        )
        np.testing.assert_array_equal(
            result.unique_materialization_global_node_ids, [3, 0]
        )
        np.testing.assert_array_equal(
            result.materialization_global_node_ids, [3, 3, 0, 0]
        )
        np.testing.assert_array_equal(
            result.materialization_to_unique_indices, [0, 0, 1, 1]
        )
        np.testing.assert_array_equal(
            result.unique_materialization_global_node_ids[
                result.materialization_to_unique_indices
            ],
            result.materialization_global_node_ids,
        )
        self.assertEqual(result.unique_semantic_rows, 4)
        self.assertEqual(result.unique_materialization_rows, 2)

    def test_multiset_cpu_and_fast_paths_have_exact_occurrence_parity(self):
        semantic = np.asarray([3, 3, 2, 1, 0, 0], dtype=np.int64)
        cpu = select_multiset_materialization(
            "conservative_pinhole",
            self.metadata,
            semantic,
            self.camera,
            self.settings,
        )
        fast = select_multiset_materialization(
            "conservative_pinhole_gpu",
            self.metadata,
            semantic,
            self.camera,
            self.settings,
            device_metadata=self.metadata.to_device(torch.device("cpu")),
        )
        for name in (
            "unique_semantic_global_node_ids",
            "semantic_to_unique_indices",
            "unique_materialization_global_node_ids",
            "materialization_global_node_ids",
            "materialization_to_unique_indices",
            "kept_semantic_indices",
            "kept_unique_semantic_indices",
        ):
            np.testing.assert_array_equal(getattr(fast, name), getattr(cpu, name))
        self.assertTrue(
            torch.equal(
                fast.materialization_global_node_ids_gpu,
                torch.from_numpy(cpu.materialization_global_node_ids),
            )
        )

    def test_torch_filter_has_exact_ordered_parity_and_transfer_accounting(self):
        semantic = np.asarray([3, 2, 1, 0], dtype=np.int64)
        reference = conservative_pinhole_materialization(
            self.metadata, semantic, self.camera, self.settings
        )
        device_metadata = self.metadata.to_device(torch.device("cpu"))
        result = fast_conservative_pinhole_materialization(
            device_metadata, semantic, self.camera, self.settings
        )
        np.testing.assert_array_equal(
            result.materialization_global_node_ids,
            reference.materialization_global_node_ids,
        )
        np.testing.assert_array_equal(
            result.kept_semantic_indices, reference.kept_semantic_indices
        )
        self.assertTrue(
            torch.equal(
                result.materialization_global_node_ids_gpu,
                torch.from_numpy(reference.materialization_global_node_ids),
            )
        )
        self.assertEqual(result.timings.selected_id_bytes, semantic.nbytes)
        self.assertEqual(
            result.timings.retained_index_bytes,
            reference.kept_semantic_indices.nbytes,
        )

    def test_device_metadata_is_exact_and_contains_only_required_fields(self):
        device_metadata = self.metadata.to_device(torch.device("cpu"))
        self.assertTrue(
            torch.equal(device_metadata.means, torch.from_numpy(self.metadata.means))
        )
        self.assertTrue(
            torch.equal(
                device_metadata.max_scales,
                torch.from_numpy(self.metadata.max_scales),
            )
        )
        self.assertEqual(device_metadata.logical_bytes, self.metadata.logical_bytes)
        self.assertEqual(device_metadata.bytes_per_row, 16)

    def test_fast_filter_handles_empty_and_all_retained_materialization(self):
        device_metadata = self.metadata.to_device(torch.device("cpu"))
        empty = fast_conservative_pinhole_materialization(
            device_metadata,
            np.empty(0, dtype=np.int64),
            self.camera,
            self.settings,
        )
        self.assertEqual(empty.materialization_rows, 0)
        self.assertEqual(empty.timings.selected_id_bytes, 0)
        all_retained_ids = np.asarray([3, 0], dtype=np.int64)
        retained = fast_conservative_pinhole_materialization(
            device_metadata, all_retained_ids, self.camera, self.settings
        )
        np.testing.assert_array_equal(
            retained.materialization_global_node_ids, all_retained_ids
        )

    def test_strict_image_boundary_comparisons_match_reference(self):
        means = np.asarray(
            [
                [-1.10, 0.0, 2.0],
                [-1.06, 0.0, 2.0],
            ],
            dtype=np.float32,
        )
        metadata = AlodCompactCullingMetadata.from_log_scales(
            means, np.full((2, 3), -20.0, dtype=np.float32)
        )
        semantic = np.asarray([0, 1], dtype=np.int64)
        reference = conservative_pinhole_materialization(
            metadata, semantic, self.camera, self.settings
        )
        result = fast_conservative_pinhole_materialization(
            metadata.to_device(torch.device("cpu")),
            semantic,
            self.camera,
            self.settings,
        )
        np.testing.assert_array_equal(reference.materialization_global_node_ids, [1])
        np.testing.assert_array_equal(
            result.materialization_global_node_ids,
            reference.materialization_global_node_ids,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_metadata_and_filter_match_cpu_reference_exactly(self):
        semantic = np.asarray([3, 2, 1, 0], dtype=np.int64)
        reference = conservative_pinhole_materialization(
            self.metadata, semantic, self.camera, self.settings
        )
        device_metadata = self.metadata.to_device(torch.device("cuda"))
        self.assertTrue(
            torch.equal(
                device_metadata.means.cpu(), torch.from_numpy(self.metadata.means)
            )
        )
        self.assertTrue(
            torch.equal(
                device_metadata.max_scales.cpu(),
                torch.from_numpy(self.metadata.max_scales),
            )
        )
        result = fast_conservative_pinhole_materialization(
            device_metadata, semantic, self.camera.to(torch.device("cuda")), self.settings
        )
        np.testing.assert_array_equal(
            result.materialization_global_node_ids,
            reference.materialization_global_node_ids,
        )
        np.testing.assert_array_equal(
            result.kept_semantic_indices, reference.kept_semantic_indices
        )

    def test_unsupported_camera_model_is_identity(self):
        settings = RenderSettings(
            width=100,
            height=100,
            near_plane=0.01,
            far_plane=100.0,
            camera_model="fisheye",
        )
        semantic = np.asarray([2, 1, 0], dtype=np.int64)
        result = conservative_pinhole_materialization(
            self.metadata, semantic, self.camera, settings
        )
        np.testing.assert_array_equal(result.materialization_global_node_ids, semantic)
        self.assertEqual(result.filter_name, "none")

    def test_fast_unsupported_camera_model_is_identity(self):
        settings = RenderSettings(
            width=100,
            height=100,
            near_plane=0.01,
            far_plane=100.0,
            camera_model="fisheye",
        )
        semantic = np.asarray([2, 1, 0], dtype=np.int64)
        result = fast_conservative_pinhole_materialization(
            self.metadata.to_device(torch.device("cpu")),
            semantic,
            self.camera,
            settings,
        )
        np.testing.assert_array_equal(result.materialization_global_node_ids, semantic)
        self.assertEqual(result.filter_name, "none")

    def test_out_of_range_stable_id_is_rejected(self):
        with self.assertRaises(IndexError):
            conservative_pinhole_materialization(
                self.metadata,
                np.asarray([4], dtype=np.int64),
                self.camera,
                self.settings,
            )


if __name__ == "__main__":
    unittest.main()
