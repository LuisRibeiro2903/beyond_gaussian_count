import unittest

import numpy as np

from scene_runtime.alod import (
    characterize_request_trace,
    simulate_bounded_cache,
    simulate_previous_working_set,
)


class AlodCacheSimulationTest(unittest.TestCase):
    def setUp(self):
        self.requests = [
            np.asarray([2, 3], dtype=np.int64),
            np.asarray([3, 4], dtype=np.int64),
            np.asarray([2, 4], dtype=np.int64),
        ]

    def test_trace_records_overlap_churn_union_and_ordered_hash(self):
        trace = characterize_request_trace(self.requests)
        self.assertEqual(len(trace), 3)
        self.assertEqual(trace[0].requested_rows, 2)
        self.assertEqual(trace[1].overlap_with_previous_rows, 1)
        self.assertEqual(trace[1].new_rows, 1)
        self.assertEqual(trace[1].discarded_rows, 1)
        self.assertEqual(trace[2].cumulative_union_rows, 3)
        self.assertEqual(len(trace[2].ordered_global_node_ids_sha256), 64)
        reversed_trace = characterize_request_trace(
            [self.requests[0][::-1]]
        )
        self.assertNotEqual(
            trace[0].ordered_global_node_ids_sha256,
            reversed_trace[0].ordered_global_node_ids_sha256,
        )

    def test_bounded_replay_matches_deterministic_lru_semantics(self):
        result = simulate_bounded_cache(
            self.requests,
            capacity_rows=2,
            property_bytes_per_row=92,
            global_row_count=7,
        )
        self.assertEqual(result.requested_rows, 6)
        self.assertEqual(result.cache_hits, 2)
        self.assertEqual(result.cache_misses, 4)
        self.assertEqual(result.loads, 4)
        self.assertEqual(result.evictions, 2)
        self.assertEqual(result.total_h2d_bytes, 4 * 92)
        self.assertEqual(result.maximum_occupancy_rows, 2)
        self.assertEqual(result.zero_transfer_frame_count, 0)

    def test_larger_cache_reuses_rows_older_than_previous_frame(self):
        bounded = simulate_bounded_cache(
            self.requests,
            capacity_rows=3,
            property_bytes_per_row=92,
            global_row_count=7,
        )
        previous = simulate_previous_working_set(
            self.requests, property_bytes_per_row=92
        )
        self.assertEqual(bounded.cache_hits, 3)
        self.assertEqual(bounded.cache_misses, 3)
        self.assertEqual(bounded.evictions, 0)
        self.assertEqual(previous.cache_hits, 2)
        self.assertEqual(previous.cache_misses, 4)

    def test_capacity_below_maximum_frame_is_infeasible(self):
        with self.assertRaisesRegex(ValueError, "maximum frame request"):
            simulate_bounded_cache(
                self.requests,
                capacity_rows=1,
                property_bytes_per_row=92,
            )

    def test_duplicate_request_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            characterize_request_trace([np.asarray([2, 2], dtype=np.int64)])


if __name__ == "__main__":
    unittest.main()
