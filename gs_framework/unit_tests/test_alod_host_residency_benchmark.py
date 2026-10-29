from __future__ import annotations

import unittest

from benchmarks.alod_host_residency_benchmark import _stats
from scripts.alod.run_host_residency_benchmark_campaign import CONFIG_IDS, SCHEDULE


class TestAlodHostResidencyBenchmark(unittest.TestCase):
    def test_frozen_schedule_has_three_replicates_per_configuration(self):
        self.assertEqual(len(SCHEDULE), 12)
        for config_id in CONFIG_IDS:
            self.assertEqual(
                [replicate for replicate, candidate in SCHEDULE if candidate == config_id],
                [1, 2, 3],
            )

    def test_frozen_schedule_does_not_fix_any_mode_first_or_last(self):
        positions = {config_id: [] for config_id in CONFIG_IDS}
        for index, (_, config_id) in enumerate(SCHEDULE):
            positions[config_id].append(index % 4)
        for occupied in positions.values():
            self.assertEqual(len(set(occupied)), 3)

    def test_summary_statistics_include_requested_percentiles(self):
        summary = _stats([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["median"], 2.5)
        self.assertAlmostEqual(summary["p95"], 3.85)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 4.0)


if __name__ == "__main__":
    unittest.main()
