"""Opt-in Run005 gate for host-resident ALoD property modes."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch

from scripts.alod.validate_host_residency import PASS_STATUS, validate


DHIER = Path(
    "/home/luis/tese/runs/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/"
    "ALOD-SMOKE-TRAIN-005/output/final.dhier"
)
MANIFEST = Path(
    "/home/luis/tese/runs/alod/oracles/"
    "ALOD-RUN005-SELECTION-ORACLE-001/manifest.json"
)
ENABLED = os.environ.get("GS_FRAMEWORK_RUN_ALOD_HOST_RESIDENCY_INTEGRATION") == "1"


@unittest.skipUnless(
    ENABLED and DHIER.is_file() and MANIFEST.is_file() and torch.cuda.is_available(),
    "set GS_FRAMEWORK_RUN_ALOD_HOST_RESIDENCY_INTEGRATION=1 with Run005 and CUDA",
)
class AlodHostResidencyExternalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = validate(DHIER, MANIFEST, device=torch.device("cuda"), render=True)

    def test_all_modes_preserve_ten_camera_ids_properties_and_render(self):
        self.assertEqual(self.report["status"], PASS_STATUS)
        self.assertEqual(self.report["resident_prerequisite"], "PASS-ALOD-PRODUCTION-RESIDENT")
        for mode in ("host_nocache", "host_previous_reuse", "host_bounded_cache"):
            cameras = self.report["modes"][mode]["cameras"]
            self.assertEqual(len(cameras), 10)
            for camera in cameras:
                self.assertTrue(camera["selection_parity"])
                self.assertTrue(camera["raster_id_parity"])
                self.assertTrue(camera["pre_raster_parity"])
                self.assertIsNotNone(camera["render"])

    def test_capacity_memory_and_sequence_gates(self):
        capacity = self.report["capacity_derivation"]
        self.assertEqual(capacity["minimum_feasible_capacity_rows"], 499030)
        self.assertEqual(capacity["chosen_first_cache_capacity_rows"], 525000)
        resident_bytes = self.report["resident"]["gaussian_property_gpu_bytes"]
        for mode, record in self.report["modes"].items():
            self.assertLess(
                record["memory"]["persistent_gpu_gaussian_property_bytes"],
                resident_bytes,
                mode,
            )
        bounded = self.report["modes"]["host_bounded_cache"]["sequences"]
        self.assertTrue(
            any(
                row["alod_cache_eviction_count"] > 0
                for row in bounded["cache_capacity_stress"][1:]
            )
        )


if __name__ == "__main__":
    unittest.main()
