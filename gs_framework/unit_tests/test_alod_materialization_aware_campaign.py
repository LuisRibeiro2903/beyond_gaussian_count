from __future__ import annotations

import unittest

from scripts.alod.run_materialization_aware_residency_campaign import CONFIG_IDS, SCHEDULE


class TestMaterializationAwareCampaign(unittest.TestCase):
    def test_schedule_preserves_three_replicates(self):
        self.assertEqual(len(SCHEDULE), 12)
        for config_id in CONFIG_IDS:
            self.assertEqual(
                [replicate for replicate, candidate in SCHEDULE if candidate == config_id],
                [1, 2, 3],
            )

    def test_schedule_uses_three_distinct_positions(self):
        positions = {config_id: [] for config_id in CONFIG_IDS}
        for index, (_, config_id) in enumerate(SCHEDULE):
            positions[config_id].append(index % 4)
        for occupied in positions.values():
            self.assertEqual(len(set(occupied)), 3)


if __name__ == "__main__":
    unittest.main()
