from __future__ import annotations

from collections import Counter
import unittest

from benchmarks.alod_resident_benchmark import _runtime_config
from benchmarks.benchmark_real_scene import load_config
from scripts.alod.run_ext_formal_run009_campaign import CONFIG_IDS, SCHEDULE


class AlodExtRun009CampaignTest(unittest.TestCase):
    def test_all_modes_preserve_faithful_semantic_configuration(self) -> None:
        faithful = _runtime_config(load_config("ALOD-REPRO-HOST-NOCACHE"))
        for config_id in CONFIG_IDS:
            runtime = _runtime_config(load_config(config_id))
            self.assertEqual(
                faithful.__dict__ | {"materialization_filter": "none"},
                runtime.__dict__ | {"materialization_filter": "none"},
            )

    def test_frozen_run009_capacities_are_used(self) -> None:
        self.assertEqual(
            load_config("ALOD-EXT-PREMATERIALIZATION-CUDA-MINIMUM")["alod_host"]["cache_capacity_rows"],
            1_599_906,
        )
        self.assertEqual(
            load_config("ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL")["alod_host"]["cache_capacity_rows"],
            3_700_000,
        )
        self.assertEqual(
            load_config("ALOD-EXT-PREMATERIALIZATION-CUDA-LARGE")["alod_host"]["cache_capacity_rows"],
            5_300_000,
        )

    def test_schedule_has_three_fresh_processes_per_mode(self) -> None:
        self.assertEqual(len(SCHEDULE), 18)
        self.assertEqual(Counter(config_id for _, config_id in SCHEDULE), Counter({config_id: 3 for config_id in CONFIG_IDS}))
        self.assertEqual({replicate for replicate, _ in SCHEDULE}, {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
