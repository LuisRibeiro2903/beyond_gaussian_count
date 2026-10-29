"""External Run005 selection-oracle validation (opt in with an environment variable)."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from alod_selection_oracle import validate_oracle_tree
from alod_selection_oracle import sha256_file


ORACLE_MANIFEST = Path(
    "/home/luis/tese/runs/alod/oracles/"
    "ALOD-RUN005-SELECTION-ORACLE-001/manifest.json"
)
LOCATOR = (
    Path(__file__).resolve().parents[1]
    / "unit_tests/fixtures/alod_selection_oracle/run005-selection-oracle-v1.json"
)


@unittest.skipUnless(
    os.environ.get("ALOD_RUN005_ORACLE_INTEGRATION") == "1",
    "set ALOD_RUN005_ORACLE_INTEGRATION=1 to validate the external Run005 oracle",
)
class Run005SelectionOracleExternalTest(unittest.TestCase):
    def test_complete_external_oracle_tree(self) -> None:
        result = validate_oracle_tree(ORACLE_MANIFEST)
        self.assertTrue(result["valid"])
        self.assertEqual(result["oracle_id"], "ALOD-RUN005-SELECTION-ORACLE-001")
        self.assertEqual(result["camera_count"], 10)
        self.assertGreater(result["validated_payload_references"], 0)

    def test_tracked_locator_matches_external_semantic_oracle(self) -> None:
        locator = json.loads(LOCATOR.read_text(encoding="utf-8"))
        root = ORACLE_MANIFEST.parent
        self.assertEqual(
            sha256_file(root / "semantic-manifest.json"),
            locator["deterministic_semantic_manifest_sha256"],
        )
        manifest = json.loads((root / "semantic-manifest.json").read_text())
        actual_cameras = {camera["camera_stable_id"]: camera for camera in manifest["cameras"]}
        for expected in locator["cameras"]:
            actual = actual_cameras[expected["camera_stable_id"]]
            for key in (
                "selected_hierarchy_count",
                "ordered_selected_global_ids_sha256",
                "sorted_selected_global_ids_sha256",
            ):
                self.assertEqual(actual[key], expected[key])
            pre = json.loads(
                (root / actual["paths"]["pre-raster.json"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                pre["pre_raster_bundle_sha256"], expected["pre_raster_bundle_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
