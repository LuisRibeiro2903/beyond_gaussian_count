"""Opt-in exact Run005 parity gate for the CPU ALoD reference selector."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from scripts.alod.compare_reference_selection_oracle import PASS_STATUS, compare


REPO_ROOT = Path(__file__).resolve().parents[1]
DHIER = Path(
    "/home/luis/tese/runs/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/"
    "ALOD-SMOKE-TRAIN-005/output/final.dhier"
)
MANIFEST = Path(
    "/home/luis/tese/runs/alod/oracles/"
    "ALOD-RUN005-SELECTION-ORACLE-001/manifest.json"
)
REGRESSION = (
    REPO_ROOT
    / "unit_tests/fixtures/alod_selection_oracle/"
    "run005-camera1-reference-spt-regression-v1.json"
)
ENABLED = os.environ.get("GS_FRAMEWORK_RUN_ALOD_REFERENCE_INTEGRATION") == "1"


@unittest.skipUnless(
    ENABLED and DHIER.is_file() and MANIFEST.is_file(),
    "set GS_FRAMEWORK_RUN_ALOD_REFERENCE_INTEGRATION=1 with Run005 assets",
)
class AlodReferenceSelectionExternalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = compare(dhier_path=DHIER, oracle_manifest_path=MANIFEST)

    def test_all_ten_cameras_pass_every_layer(self):
        self.assertEqual(self.report["status"], PASS_STATUS)
        self.assertTrue(self.report["hspt"]["parity"])
        self.assertEqual(len(self.report["cameras"]), 10)
        self.assertEqual(
            [camera["registered_image_id"] for camera in self.report["cameras"]],
            [1, 101, 201, 301, 351, 401, 451, 501, 601, 701],
        )
        for camera in self.report["cameras"]:
            self.assertTrue(all(camera["parity"].values()))

    def test_camera1_364158_node_ordering_regression(self):
        fixture = json.loads(REGRESSION.read_text(encoding="utf-8"))
        camera = self.report["cameras"][0]
        self.assertEqual(camera["camera_stable_id"], fixture["camera_stable_id"])
        self.assertEqual(
            [entry["spt_id"] for entry in camera["per_spt"]],
            fixture["candidate_spt_ids"],
        )
        self.assertEqual(
            [entry["root_global_node_id"] for entry in camera["per_spt"]],
            fixture["candidate_spt_root_global_node_ids"],
        )
        self.assertEqual(
            [entry["emitted_count"] for entry in camera["per_spt"]],
            fixture["per_spt_emitted_counts"],
        )
        self.assertEqual(
            sum(entry["emitted_count"] for entry in camera["per_spt"]),
            fixture["selected_spt_node_count"],
        )
        self.assertEqual(
            camera["ordered_spt_sha256"],
            fixture["ordered_spt_global_node_ids_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
