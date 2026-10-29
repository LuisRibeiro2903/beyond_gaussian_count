"""Opt-in Run005 gate for the fully resident production ALoD runtime."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch

from scripts.alod.validate_production_resident import PASS_STATUS, validate


DHIER = Path(
    "/home/luis/tese/runs/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/"
    "ALOD-SMOKE-TRAIN-005/output/final.dhier"
)
MANIFEST = Path(
    "/home/luis/tese/runs/alod/oracles/"
    "ALOD-RUN005-SELECTION-ORACLE-001/manifest.json"
)
ENABLED = os.environ.get("GS_FRAMEWORK_RUN_ALOD_PRODUCTION_INTEGRATION") == "1"


@unittest.skipUnless(
    ENABLED and DHIER.is_file() and MANIFEST.is_file() and torch.cuda.is_available(),
    "set GS_FRAMEWORK_RUN_ALOD_PRODUCTION_INTEGRATION=1 with Run005 and CUDA",
)
class AlodProductionResidentExternalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = validate(
            DHIER, MANIFEST, device=torch.device("cuda"), render=True
        )

    def test_hspt_and_all_ten_camera_selection_layers_are_exact(self):
        self.assertEqual(self.report["status"], PASS_STATUS)
        self.assertTrue(self.report["hspt"]["reference_parity"])
        self.assertEqual(self.report["hspt"]["upper_rows"], 1309)
        self.assertEqual(self.report["hspt"]["spt_count"], 23)
        self.assertEqual(self.report["hspt"]["flattened_spt_rows"], 998713)
        self.assertEqual(len(self.report["cameras"]), 10)
        for camera in self.report["cameras"]:
            self.assertTrue(camera["selection_parity"])
            self.assertTrue(camera["pre_raster_parity"])

    def test_resident_memory_and_render_smoke_are_recorded(self):
        self.assertTrue(self.report["resident"])
        self.assertFalse(self.report["streaming"])
        self.assertGreater(self.report["memory"]["gaussian_properties_gpu_bytes"], 0)
        for camera in self.report["cameras"]:
            self.assertIsNotNone(camera["render"])
            self.assertGreaterEqual(camera["render"]["visible_gaussians"], 0)
            self.assertGreaterEqual(camera["render"]["max_abs_error"], 0)


if __name__ == "__main__":
    unittest.main()
