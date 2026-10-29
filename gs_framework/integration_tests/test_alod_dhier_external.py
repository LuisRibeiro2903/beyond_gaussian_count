"""Optional large-asset integration gates for the standalone ALoD inspector."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

from alod_dhier import build_inspection_report, load_dhier


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path("/home/luis/tese/runs/alod/ALOD-ORACLE-MC-AERIAL-TEST-01")
INITIAL = RUN_ROOT / "ALOD-SMOKE-TRAIN-004/output/scaffold/point_cloud/iteration_5000/hierarchy.dhier"
RUN004_FINAL = RUN_ROOT / "ALOD-SMOKE-TRAIN-004/output/final.dhier"
RUN005_FINAL = RUN_ROOT / "ALOD-SMOKE-TRAIN-005/output/final.dhier"
FIXTURES = REPO_ROOT / "unit_tests" / "fixtures" / "alod_dhier"
VALIDATOR = REPO_ROOT / "scripts" / "alod" / "validate_dhier_oracle.py"
ENABLED = os.environ.get("GS_FRAMEWORK_RUN_ALOD_DHIER_INTEGRATION") == "1"
ASSETS_EXIST = all(path.is_file() for path in (INITIAL, RUN004_FINAL, RUN005_FINAL))


@unittest.skipUnless(ENABLED and ASSETS_EXIST, "set GS_FRAMEWORK_RUN_ALOD_DHIER_INTEGRATION=1 with external assets")
class AlodDhierExternalIntegrationTest(unittest.TestCase):
    def test_run004_initial_degree3_creator_profile_is_valid(self):
        report = build_inspection_report(
            load_dhier(INITIAL),
            validate_tree=True,
            profile="creator-initial",
            repo_root=REPO_ROOT,
        )
        self.assertEqual(report["asset_sha256"], "a7862de91cd3620ab7be1296a50b9030aa1c2cca2b4fe15d79abaf55f4664d17")
        self.assertEqual(report["format"]["sh_degree"], 3)
        self.assertEqual(report["format"]["gaussian_count"], 887239)
        self.assertTrue(report["validation"]["overall_valid"])
        self.assertEqual(
            report["hierarchy"]["creator_initial_leaf_first_child_quirk"]["occurrence_count"],
            443620,
        )
        self.assertEqual(
            report["attribute_validation"]["activated_opacity_range"]["above_one_count"],
            92926,
        )

    def test_run005_final_degree1_strict_is_valid(self):
        report = build_inspection_report(
            load_dhier(RUN005_FINAL),
            validate_tree=True,
            skybox_count=100000,
            profile="strict",
            repo_root=REPO_ROOT,
        )
        self.assertEqual(report["asset_sha256"], "6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c")
        self.assertEqual(report["format"]["sh_degree"], 1)
        self.assertEqual(report["format"]["gaussian_count"], 1099999)
        self.assertTrue(report["validation"]["overall_valid"])
        self.assertEqual(report["hierarchy"]["root_global_id"], 100000)
        self.assertEqual(report["hierarchy"]["reachable_count"], 999999)

    def test_run004_rejected_final_rediscovers_corruption(self):
        report = build_inspection_report(
            load_dhier(RUN004_FINAL),
            validate_tree=True,
            skybox_count=100000,
            profile="strict",
            repo_root=REPO_ROOT,
        )
        self.assertFalse(report["validation"]["overall_valid"])
        self.assertEqual(report["hierarchy"]["depth_mismatches"], 4355)
        self.assertEqual(report["hierarchy"]["parent_mismatches"], 1)
        self.assertEqual(report["hierarchy"]["reachable_count"], 999999)
        self.assertEqual(
            report["hierarchy"]["issues"]["counts"]["skybox_topology_sentinel"], 1
        )

    def test_raw_and_model_layer_official_oracles(self):
        pairs = [
            (INITIAL, FIXTURES / "run004_initial_degree3_official_loader_v1.json"),
            (RUN005_FINAL, FIXTURES / "run005_final_degree1_official_loader_v1.json"),
        ]
        for asset, oracle in pairs:
            with self.subTest(asset=asset.name, oracle=oracle.name):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(VALIDATOR),
                        "--asset",
                        str(asset),
                        "--oracle",
                        str(oracle),
                    ],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
