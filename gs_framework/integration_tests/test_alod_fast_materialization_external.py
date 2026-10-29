"""Opt-in Run005 gate for the fast conservative materialization filter."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch

from scripts.alod.validate_fast_materialization_filter import PASS_STATUS, validate


DHIER = Path(
    "/home/luis/tese/runs/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/"
    "ALOD-SMOKE-TRAIN-005/output/final.dhier"
)
ENABLED = os.environ.get("GS_FRAMEWORK_RUN_ALOD_FAST_CULLING_INTEGRATION") == "1"
GSPLAT_EXTENSION = (
    Path(os.environ["GS_FRAMEWORK_GSPLAT_EXTENSION"])
    if os.environ.get("GS_FRAMEWORK_GSPLAT_EXTENSION")
    else None
)


@unittest.skipUnless(
    ENABLED and DHIER.is_file() and torch.cuda.is_available(),
    "set GS_FRAMEWORK_RUN_ALOD_FAST_CULLING_INTEGRATION=1 with Run005 and CUDA",
)
class AlodFastMaterializationExternalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = validate(
            DHIER,
            torch.device("cuda"),
            warmup_repetitions=1,
            timing_repetitions=2,
            gsplat_extension=GSPLAT_EXTENSION,
        )

    def test_all_ten_cameras_preserve_ids_properties_and_render(self):
        self.assertEqual(self.report["status"], PASS_STATUS)
        self.assertEqual(self.report["camera_count"], 10)
        self.assertTrue(self.report["all_cpu_fast_materialization_ids_exact"])
        self.assertTrue(self.report["all_backend_false_negatives_zero"])
        self.assertTrue(self.report["all_surviving_properties_bitwise_equal"])
        self.assertTrue(self.report["all_backend_projection_ids_exact"])
        self.assertTrue(self.report["all_rgb_bitwise_equal"])
        self.assertTrue(self.report["all_alpha_bitwise_equal"])

    def test_compact_metadata_transfer_and_reduced_cache_are_recorded(self):
        compact = self.report["compact_metadata"]
        self.assertEqual(compact["bytes_per_gaussian"], 16)
        self.assertTrue(compact["cpu_gpu_bitwise_equal"])
        self.assertGreater(compact["gpu_bytes"], 0)
        self.assertEqual(
            self.report["reduced_cache_configuration"]["capacity_rows"], 160000
        )


if __name__ == "__main__":
    unittest.main()
