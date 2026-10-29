import unittest

import numpy as np

from alod_reference import HSPT, ReferenceCamera, ReferenceConfig
from alod_reference.selection import cut_candidate_spts, traverse_upper_hierarchy
from alod_reference.types import UpperTraversalResult


def camera(center):
    return ReferenceCamera(
        stable_id="synthetic",
        camera_center=np.asarray(center, dtype=np.float32),
        full_projection_transform=np.eye(4, dtype=np.float32),
    )


def config():
    return ReferenceConfig(10, 0, 1.0, 0, 1.0, 1.0)


def upper_hspt(node1_x=0.0, node1_threshold=4.0):
    nodes = np.asarray(
        [
            [0, -1, 2, 1, 0, 10],
            [1, 0, 2, 2, 4, 11],
            [2, 1, 0, -1, 3, 12],
            [2, 1, 0, -1, 0, 13],
            [1, 0, 0, -1, 0, 14],
        ],
        dtype=np.int32,
    )
    xyz = np.asarray(
        [[0, 0, 0], [node1_x, 0, 0], [node1_x, 0, 0], [node1_x, 0, 0], [0, 0, 0]],
        dtype=np.float32,
    )
    return HSPT(
        upper_tree_nodes=nodes,
        upper_to_global_node_id=nodes[:, 5].copy(),
        upper_tree_xyz=xyz,
        upper_tree_log_scaling=np.full((5, 3), -4, dtype=np.float32),
        upper_min_distance_squared=np.asarray(
            [1e12, node1_threshold, 4, 4, 4], dtype=np.float32
        ),
        spt_starts=np.asarray([0], dtype=np.int32),
        spt_min=np.empty(0, dtype=np.float32),
        spt_max=np.empty(0, dtype=np.float32),
        spt_to_global_node_id=np.empty(0, dtype=np.int32),
        spt_root_global_node_ids=np.empty(0, dtype=np.int64),
        spt_sizes=np.empty(0, dtype=np.int32),
    )


def dummy_upper(candidate_ids, distances):
    empty = np.empty(0, dtype=np.int32)
    return UpperTraversalResult(
        coarse_cut_upper_local_ids=empty,
        coarse_cut_global_node_ids=empty,
        cut_leaf_upper_local_ids=empty,
        cut_leaf_global_node_ids=empty,
        upper_render_global_node_ids=empty,
        candidate_spt_ids=np.asarray(candidate_ids, dtype=np.int32),
        candidate_spt_upper_local_ids=np.arange(len(candidate_ids), dtype=np.int32),
        candidate_spt_root_global_node_ids=np.asarray([100, 200][: len(candidate_ids)], dtype=np.int32),
        candidate_spt_distances=np.asarray(distances, dtype=np.float32),
        frustum_planes=np.zeros((4, 4), dtype=np.float32),
        frustum_rejected_upper_local_ids=empty,
    )


class AlodReferenceSelectionTest(unittest.TestCase):
    def test_near_camera_descends_and_far_camera_cuts_at_coarse_node(self):
        hspt = upper_hspt()
        near = traverse_upper_hierarchy(hspt, camera([0, 0, 0]), config())
        far = traverse_upper_hierarchy(hspt, camera([0, 0, 10]), config())
        np.testing.assert_array_equal(near.upper_render_global_node_ids, [14, 12, 13])
        np.testing.assert_array_equal(far.coarse_cut_global_node_ids, [14, 11])
        # Preserve the evaluator's first_child<=0 gather quirk literally.
        np.testing.assert_array_equal(far.upper_render_global_node_ids, [14])

    def test_lod_predicate_is_strict_below_equal_above(self):
        equal = traverse_upper_hierarchy(
            upper_hspt(node1_threshold=4.0), camera([0, 0, 2]), config()
        )
        below = traverse_upper_hierarchy(
            upper_hspt(node1_threshold=np.nextafter(np.float32(4), np.float32(0))),
            camera([0, 0, 2]),
            config(),
        )
        above = traverse_upper_hierarchy(
            upper_hspt(node1_threshold=np.nextafter(np.float32(4), np.float32(5))),
            camera([0, 0, 2]),
            config(),
        )
        self.assertIn(11, equal.coarse_cut_global_node_ids)
        self.assertIn(11, below.coarse_cut_global_node_ids)
        self.assertNotIn(11, above.coarse_cut_global_node_ids)
        self.assertIn(12, above.coarse_cut_global_node_ids)

    def test_frustum_rejects_entire_subtree(self):
        result = traverse_upper_hierarchy(
            upper_hspt(node1_x=-10.0), camera([0, 0, 0]), config()
        )
        np.testing.assert_array_equal(result.upper_render_global_node_ids, [14])
        np.testing.assert_array_equal(result.frustum_rejected_upper_local_ids, [1])

    def test_literal_spt_interval_and_exclusive_prefix_semantics(self):
        hspt = HSPT(
            upper_tree_nodes=np.empty((0, 6), dtype=np.int32),
            upper_to_global_node_id=np.empty(0, dtype=np.int32),
            upper_tree_xyz=np.empty((0, 3), dtype=np.float32),
            upper_tree_log_scaling=np.empty((0, 3), dtype=np.float32),
            upper_min_distance_squared=np.empty(0, dtype=np.float32),
            spt_starts=np.asarray([0, 3, 6], dtype=np.int32),
            spt_min=np.asarray([0, 0, 100, 0, 0, 100], dtype=np.float32),
            spt_max=np.asarray([1e12, 10, 0, 1e12, 10, 0], dtype=np.float32),
            spt_to_global_node_id=np.asarray([100, 101, 102, 200, 201, 202], dtype=np.int32),
            spt_root_global_node_ids=np.asarray([100, 200], dtype=np.int64),
            spt_sizes=np.asarray([3, 3], dtype=np.int32),
        )
        per_spt, combined = cut_candidate_spts(hspt, dummy_upper([0, 1], [5, 5]))
        # Offset zero is tested only for the first candidate; later SPT roots are omitted.
        np.testing.assert_array_equal(per_spt[0].emitted_global_node_ids, [100, 101])
        np.testing.assert_array_equal(per_spt[1].emitted_global_node_ids, [201])
        np.testing.assert_array_equal(combined, [100, 101, 201])

    def test_spt_near_distance_selects_finer_cut_than_far_distance(self):
        hspt = HSPT(
            upper_tree_nodes=np.empty((0, 6), dtype=np.int32),
            upper_to_global_node_id=np.empty(0, dtype=np.int32),
            upper_tree_xyz=np.empty((0, 3), dtype=np.float32),
            upper_tree_log_scaling=np.empty((0, 3), dtype=np.float32),
            upper_min_distance_squared=np.empty(0, dtype=np.float32),
            spt_starts=np.asarray([0, 5], dtype=np.int32),
            spt_min=np.asarray([100, 0, 0, 0, 0], dtype=np.float32),
            spt_max=np.asarray([1e12, 10, 6, 2, 0], dtype=np.float32),
            spt_to_global_node_id=np.asarray([100, 101, 102, 103, 104], dtype=np.int32),
            spt_root_global_node_ids=np.asarray([100], dtype=np.int64),
            spt_sizes=np.asarray([5], dtype=np.int32),
        )
        _, far = cut_candidate_spts(hspt, dummy_upper([0], [8]))
        _, near = cut_candidate_spts(hspt, dummy_upper([0], [1]))
        np.testing.assert_array_equal(far, [101])
        np.testing.assert_array_equal(near, [101, 102, 103])


if __name__ == "__main__":
    unittest.main()
