import unittest

import numpy as np

from benchmarks.alod_resident_benchmark import _hierarchy_cut_validation, _stats


class AlodResidentBenchmarkTest(unittest.TestCase):
    def setUp(self):
        # root 0 -> interior 1 and leaf 4; interior 1 -> leaves 2 and 3
        self.nodes = np.asarray(
            [
                [0, -1, 2, 1, 0, 0],
                [1, 0, 2, 2, 4, 0],
                [2, 1, 0, 0, 3, 0],
                [2, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

    def test_finest_leaf_cut_is_complete_and_has_no_ancestor_duplication(self):
        result = _hierarchy_cut_validation(
            self.nodes,
            np.asarray([2, 3, 4], dtype=np.int64),
            root_id=0,
            skybox_count=0,
            require_complete=True,
            require_no_ancestor_duplication=True,
        )
        self.assertTrue(result["complete_one_node_per_leaf_path"])
        self.assertTrue(result["no_ancestor_descendant_duplication"])
        self.assertEqual(result["selected_nodes_with_selected_ancestor"], 0)

    def test_source_style_ancestor_overlap_can_be_recorded_without_redefinition(self):
        result = _hierarchy_cut_validation(
            self.nodes,
            np.asarray([1, 2, 3, 4], dtype=np.int64),
            root_id=0,
            skybox_count=0,
            require_complete=False,
            require_no_ancestor_duplication=False,
        )
        self.assertFalse(result["no_ancestor_descendant_duplication"])
        self.assertEqual(result["selected_nodes_with_selected_ancestor"], 2)

    def test_statistics_include_requested_distribution_fields(self):
        result = _stats([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["mean"], 2.5)
        self.assertEqual(result["median"], 2.5)
        self.assertAlmostEqual(result["p10"], 1.3)
        self.assertAlmostEqual(result["p90"], 3.7)
        self.assertIsNotNone(result["std"])


if __name__ == "__main__":
    unittest.main()
