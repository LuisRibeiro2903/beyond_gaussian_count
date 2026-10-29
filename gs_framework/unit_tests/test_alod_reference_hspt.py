import unittest
from unittest import mock

import numpy as np
import torch

from alod_reference import HierarchyData, ReferenceConfig, build_hspt


def hierarchy(positions, log_scales, nodes):
    return HierarchyData(
        positions=np.asarray(positions, dtype=np.float32),
        log_scales=np.asarray(log_scales, dtype=np.float32),
        nodes=np.asarray(nodes, dtype=np.int32),
    )


def config(**overrides):
    values = dict(
        root_global_node_id=0,
        skybox_count=0,
        spt_root_volume=1.0,
        min_spt_size=0,
        target_granularity=1.0,
        distance_multiplier=1.0,
    )
    values.update(overrides)
    return ReferenceConfig(**values)


class AlodReferenceHsptTest(unittest.TestCase):
    def test_single_leaf_is_one_upper_row_without_spt(self):
        data = hierarchy([[0, 0, 0]], [[0, 0, 0]], [[0, -1, 0, 0, 0, 7]])
        result = build_hspt(data, config())
        np.testing.assert_array_equal(result.upper_to_global_node_id, [0])
        self.assertEqual(result.spt_count, 0)
        self.assertEqual(result.upper_tree_nodes[0, 3], -1)
        self.assertEqual(result.upper_tree_nodes[0, 5], 0)

    def test_root_with_two_leaves_builds_small_spt(self):
        data = hierarchy(
            [[0, 0, 0], [-1, 0, 0], [1, 0, 0]],
            [[-1, -1, -1], [-2, -2, -2], [-2, -2, -2]],
            [[0, -1, 2, 1, 0, 0], [1, 0, 0, 0, 2, 0], [1, 0, 0, 0, 0, 0]],
        )
        result = build_hspt(data, config())
        self.assertEqual(result.spt_count, 1)
        np.testing.assert_array_equal(result.upper_to_global_node_id, [0])
        np.testing.assert_array_equal(result.spt_starts, [0, 3])
        # Equal parent maxima retain breadth-first child order deterministically.
        np.testing.assert_array_equal(result.spt_to_global_node_id, [0, 1, 2])
        self.assertEqual(result.spt_max[1], result.spt_max[2])

    def test_spt_at_or_below_minimum_size_is_expanded_into_upper_tree(self):
        data = hierarchy(
            [[0, 0, 0], [-1, 0, 0], [1, 0, 0]],
            [[-1, -1, -1], [-2, -2, -2], [-2, -2, -2]],
            [[0, -1, 2, 1, 0, 0], [1, 0, 0, 0, 2, 0], [1, 0, 0, 0, 0, 0]],
        )
        result = build_hspt(data, config(min_spt_size=3))
        self.assertEqual(result.spt_count, 0)
        np.testing.assert_array_equal(result.upper_to_global_node_id, [0, 1, 2])

    def test_multiple_spts_are_derived_without_run005_counts(self):
        data = hierarchy(
            [[0, 0, 0], [-2, 0, 0], [-3, 0, 0], [-1, 0, 0], [2, 0, 0], [1, 0, 0], [3, 0, 0]],
            [[1, 1, 1], [-1, -1, -1], [-2, -2, -2], [-2, -2, -2], [-1, -1, -1], [-2, -2, -2], [-2, -2, -2]],
            [
                [0, -1, 2, 1, 0, 0],
                [1, 0, 2, 2, 4, 0],
                [2, 1, 0, 0, 3, 0],
                [2, 1, 0, 0, 0, 0],
                [1, 0, 2, 5, 0, 0],
                [2, 4, 0, 0, 6, 0],
                [2, 4, 0, 0, 0, 0],
            ],
        )
        result = build_hspt(data, config())
        self.assertEqual(result.spt_count, 2)
        np.testing.assert_array_equal(result.spt_root_global_node_ids, [1, 4])
        np.testing.assert_array_equal(result.spt_sizes, [3, 3])
        np.testing.assert_array_equal(result.upper_to_global_node_id, [0, 1, 4])

    def test_source_sensitive_float32_uses_cpu_torch_result_without_epsilon(self):
        root_log_scale = np.asarray(
            [4.3867597579956055, -0.7874048352241516, 3.5580365657806396],
            dtype=np.float32,
        )
        data = hierarchy(
            [[0, 0, 0], [-1, 0, 0], [1, 0, 0]],
            [root_log_scale, [-2, -2, -2], [-2, -2, -2]],
            [[0, -1, 2, 1, 0, 0], [1, 0, 0, 0, 2, 0], [1, 0, 0, 0, 0, 0]],
        )
        target = 0.0008629449482830516
        result = build_hspt(
            data,
            config(spt_root_volume=0.000001, target_granularity=target),
        )
        torch_scales = torch.exp(torch.from_numpy(root_log_scale.copy()))
        torch_distance = (
            torch.sqrt(
                torch_scales[0] * torch_scales[1]
                + torch_scales[0] * torch_scales[2]
                + torch_scales[1] * torch_scales[2]
            )
            / target
        )
        numpy_scales = np.exp(root_log_scale).astype(np.float32)
        numpy_distance = np.float32(
            np.sqrt(
                np.float32(
                    numpy_scales[0] * numpy_scales[1]
                    + numpy_scales[0] * numpy_scales[2]
                    + numpy_scales[1] * numpy_scales[2]
                )
            )
            / np.float32(target)
        )
        float64_distance = np.sqrt(
            np.exp(root_log_scale.astype(np.float64))[0]
            * np.exp(root_log_scale.astype(np.float64))[1]
            + np.exp(root_log_scale.astype(np.float64))[0]
            * np.exp(root_log_scale.astype(np.float64))[2]
            + np.exp(root_log_scale.astype(np.float64))[1]
            * np.exp(root_log_scale.astype(np.float64))[2]
        ) / target
        self.assertNotEqual(numpy_distance, np.float32(torch_distance.item()))
        self.assertNotEqual(float64_distance, float(torch_distance.item()))
        expected_squared = torch_distance.square().numpy()
        np.testing.assert_array_equal(result.upper_min_distance_squared[1:], expected_squared)

    def test_builder_never_initializes_or_uses_cuda(self):
        data = hierarchy([[0, 0, 0]], [[0, 0, 0]], [[0, -1, 0, 0, 0, 0]])
        with mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA used")):
            result = build_hspt(data, config())
        self.assertEqual(result.upper_row_count, 1)


if __name__ == "__main__":
    unittest.main()
