import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from alod_dhier import (
    DhierCountMismatchError,
    DhierFormatError,
    ValidationProfile,
    build_inspection_report,
    load_dhier,
    validate_numeric,
    validate_topology,
)
from alod_dhier.report import canonical_json


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "scripts" / "alod" / "inspect_dhier.py"


def valid_nodes() -> np.ndarray:
    return np.array(
        [
            [0, -1, 2, 1, 0, 0],
            [1, 0, 0, 0, 2, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        dtype="<i4",
    )


def write_fixture(
    path: Path,
    *,
    degree: int = 1,
    nodes: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    opacity: np.ndarray | None = None,
    rotations: np.ndarray | None = None,
) -> dict[str, np.ndarray | int]:
    node_table = np.asarray(valid_nodes() if nodes is None else nodes, dtype="<i4")
    count = len(node_table)
    coeffs = (degree + 1) ** 2
    xyz = np.asarray(
        np.arange(count * 3, dtype=np.float32).reshape(count, 3) / 10
        if positions is None
        else positions,
        dtype="<f4",
    )
    rot = np.asarray(
        np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1))
        if rotations is None
        else rotations,
        dtype="<f4",
    )
    scales = np.asarray(
        np.arange(count * 3, dtype=np.float32).reshape(count, 3) / -20,
        dtype="<f4",
    )
    alpha = np.asarray(
        np.linspace(0.1, 0.9, count, dtype=np.float32).reshape(count, 1)
        if opacity is None
        else opacity,
        dtype="<f4",
    )
    sh = np.asarray(
        np.arange(count * coeffs * 3, dtype=np.float32).reshape(count, coeffs, 3) / 100,
        dtype="<f4",
    )
    payload = bytearray(struct.pack("<ii", count, degree))
    for array in (xyz, rot, scales, alpha, sh):
        payload.extend(array.tobytes(order="C"))
    node_count_offset = len(payload)
    payload.extend(struct.pack("<i", count))
    node_offset = len(payload)
    payload.extend(node_table.tobytes(order="C"))
    path.write_bytes(payload)
    return {
        "positions": xyz,
        "rotations": rot,
        "log_scales": scales,
        "activated_opacities": alpha,
        "sh_coefficients": sh,
        "nodes": node_table,
        "node_count_offset": node_count_offset,
        "node_offset": node_offset,
    }


class DhierFormatTest(unittest.TestCase):
    def test_field_major_degree_1_decode_and_offsets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "degree1.dhier"
            expected = write_fixture(path, degree=1)
            asset = load_dhier(path)

            self.assertEqual(asset.layout.sh_degree, 1)
            self.assertEqual(asset.layout.sh_coefficients_per_gaussian, 4)
            self.assertEqual(asset.layout.bytes_per_gaussian_attributes, 92)
            self.assertEqual(asset.layout.expected_file_bytes, 12 + 3 * (92 + 24))
            for name in (
                "positions",
                "rotations",
                "log_scales",
                "activated_opacities",
                "sh_coefficients",
                "nodes",
            ):
                self.assertTrue(np.array_equal(getattr(asset, name), expected[name]))
            self.assertEqual(asset.global_node_id_contract, "serialized_row_index")
            self.assertFalse(asset.positions.flags.writeable)

    def test_degree_3_sizing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "degree3.dhier"
            write_fixture(path, degree=3)
            asset = load_dhier(path)

            self.assertEqual(asset.sh_coefficients.shape, (3, 16, 3))
            self.assertEqual(asset.layout.bytes_per_gaussian_attributes, 236)
            self.assertEqual(asset.layout.expected_file_bytes, 12 + 3 * (236 + 24))

    def test_unsupported_degree_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad_degree.dhier"
            path.write_bytes(struct.pack("<ii", 1, 4) + b"\0" * 100)
            with self.assertRaises(DhierFormatError):
                load_dhier(path)

    def test_truncation_and_trailing_data_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base.dhier"
            write_fixture(base)
            payload = base.read_bytes()
            truncated = Path(tmpdir) / "truncated.dhier"
            trailing = Path(tmpdir) / "trailing.dhier"
            truncated.write_bytes(payload[:-7])
            trailing.write_bytes(payload + b"corrupt")

            with self.assertRaisesRegex(DhierFormatError, "truncated"):
                load_dhier(truncated)
            with self.assertRaisesRegex(DhierFormatError, "trailing"):
                load_dhier(trailing)

    def test_invalid_gaussian_count_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad_count.dhier"
            write_fixture(path)
            payload = bytearray(path.read_bytes())
            struct.pack_into("<i", payload, 0, 2_000_000_000)
            path.write_bytes(payload)
            with self.assertRaises(DhierFormatError):
                load_dhier(path)

    def test_gaussian_node_count_mismatch_rejected_before_length_tolerance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mismatch.dhier"
            info = write_fixture(path)
            payload = bytearray(path.read_bytes())
            struct.pack_into("<i", payload, int(info["node_count_offset"]), 2)
            path.write_bytes(payload)
            with self.assertRaises(DhierCountMismatchError):
                load_dhier(path)


class DhierNumericValidationTest(unittest.TestCase):
    def test_valid_raw_attributes_and_quaternion_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "valid.dhier"
            write_fixture(path)
            report = validate_numeric(load_dhier(path))

            self.assertTrue(report["valid"])
            self.assertEqual(report["activated_opacity_range"]["invalid_count"], 0)
            self.assertEqual(report["quaternion_norms"]["zero_count"], 0)

    def test_opacity_range_and_zero_quaternion_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "numeric_bad.dhier"
            rotations = np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (3, 1))
            rotations[1] = 0
            write_fixture(
                path,
                rotations=rotations,
                opacity=np.array([[0.2], [1.1], [0.3]], dtype=np.float32),
            )
            report = validate_numeric(load_dhier(path))

            self.assertFalse(report["valid"])
            self.assertEqual(report["activated_opacity_range"]["invalid_count"], 1)
            self.assertEqual(report["quaternion_norms"]["zero_count"], 1)

    def test_creator_profile_reports_but_accepts_positive_merged_opacity_above_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "creator_opacity.dhier"
            write_fixture(
                path,
                degree=3,
                opacity=np.array([[0.2], [1.25], [0.3]], dtype=np.float32),
            )
            asset = load_dhier(path)
            strict = validate_numeric(asset, profile="strict")
            creator = validate_numeric(asset, profile="creator-initial")

            self.assertFalse(strict["valid"])
            self.assertTrue(creator["valid"])
            self.assertEqual(creator["activated_opacity_range"]["above_one_count"], 1)
            self.assertEqual(creator["activated_opacity_range"]["invalid_count"], 0)

    def test_nonfinite_attribute_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nan.dhier"
            positions = np.arange(9, dtype=np.float32).reshape(3, 3)
            positions[2, 1] = np.nan
            write_fixture(path, positions=positions)
            report = validate_numeric(load_dhier(path))
            self.assertFalse(report["valid"])
            self.assertFalse(report["arrays"]["positions"]["finite"])


class DhierTopologyValidationTest(unittest.TestCase):
    def test_strict_valid_binary_tree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "valid.dhier"
            write_fixture(path)
            report = validate_topology(load_dhier(path), profile="strict")

            self.assertTrue(report["valid"])
            self.assertEqual(report["root_global_id"], 0)
            self.assertEqual(report["leaf_count"], 2)
            self.assertEqual(report["interior_count"], 1)
            self.assertEqual(report["edge_count"], 2)
            self.assertEqual(report["depth_histogram"], {"0": 1, "1": 2})

    def test_creator_initial_profile_is_narrow_and_never_traverses_leaf_field(self):
        nodes = valid_nodes()
        nodes[1, 3] = 3
        nodes[2, 3] = 3
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "creator.dhier"
            write_fixture(path, degree=3, nodes=nodes)
            asset = load_dhier(path)
            strict = validate_topology(asset, profile="strict")
            creator = validate_topology(asset, profile="creator-initial")

            self.assertFalse(strict["valid"])
            self.assertEqual(strict["issues"]["counts"]["leaf_first_child_nonzero"], 2)
            self.assertTrue(creator["valid"])
            quirk = creator["creator_initial_leaf_first_child_quirk"]
            self.assertEqual(quirk["occurrence_count"], 2)
            self.assertFalse(quirk["traversed"])

    def test_creator_profile_rejects_leaf_value_beyond_one_past_end(self):
        nodes = valid_nodes()
        nodes[1, 3] = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "creator_bad.dhier"
            write_fixture(path, degree=3, nodes=nodes)
            report = validate_topology(load_dhier(path), profile="creator-initial")
            self.assertFalse(report["valid"])
            self.assertEqual(
                report["issues"]["counts"]["creator_leaf_first_child_invalid"], 1
            )

    def test_explicit_skybox_sentinel_and_root_detection(self):
        nodes = np.array(
            [
                [-99, -99, -99, -99, -99, 0],
                [0, -1, 2, 2, 0, 0],
                [1, 1, 0, 0, 3, 0],
                [1, 1, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "skybox.dhier"
            write_fixture(path, nodes=nodes)
            report = validate_topology(load_dhier(path), skybox_count=1)

            self.assertTrue(report["valid"])
            self.assertEqual(report["root_global_id"], 1)
            self.assertEqual(report["reachable_count"], 3)

    def test_parent_and_depth_consistency(self):
        nodes = valid_nodes()
        nodes[1, 1] = 2
        nodes[2, 0] = 9
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reverse_bad.dhier"
            write_fixture(path, nodes=nodes)
            report = validate_topology(load_dhier(path))

            self.assertFalse(report["valid"])
            self.assertEqual(report["parent_mismatches"], 1)
            self.assertEqual(report["depth_mismatches"], 1)

    def test_cycle_detection(self):
        nodes = np.array(
            [
                [-99, -99, -99, -99, -99, 0],
                [0, -1, 2, 2, 0, 0],
                [1, 1, 2, 1, 3, 0],
                [1, 1, 0, 0, 0, 0],
                [2, 2, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        # Node 2 owns root 1 as its first child and node 4 as its second.
        nodes[1, 3] = 2
        nodes[2, 3] = 1
        nodes[1, 4] = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cycle.dhier"
            write_fixture(path, nodes=nodes)
            report = validate_topology(load_dhier(path), skybox_count=1)

            self.assertFalse(report["valid"])
            self.assertGreater(report["cycles"], 0)

    def test_duplicate_ownership_and_unreachable_detection(self):
        nodes = np.array(
            [
                [0, -1, 2, 1, 0, 0],
                [1, 0, 2, 3, 2, 0],
                [1, 0, 2, 3, 0, 0],
                [2, 1, 0, 0, 4, 0],
                [2, 1, 0, 0, 0, 0],
                [2, 2, 0, 0, 6, 0],
                [2, 2, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "duplicate.dhier"
            write_fixture(path, nodes=nodes)
            report = validate_topology(load_dhier(path))

            self.assertFalse(report["valid"])
            self.assertEqual(report["duplicate_ownership"], 2)
            self.assertEqual(report["unreachable_nodes"], 2)

    def test_invalid_parent_index_never_uses_negative_indexing(self):
        nodes = valid_nodes()
        nodes[1, 1] = -99
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "negative_parent.dhier"
            write_fixture(path, nodes=nodes)
            report = validate_topology(load_dhier(path))

            self.assertFalse(report["valid"])
            self.assertEqual(
                report["issues"]["counts"]["parent_out_of_range_or_skybox"], 1
            )


class DhierReportAndCliTest(unittest.TestCase):
    def test_compact_official_oracle_fixtures_are_versioned_and_hash_pinned(self):
        fixture_root = REPO_ROOT / "unit_tests" / "fixtures" / "alod_dhier"
        expected = {
            "run004_initial_degree3_official_loader_v1.json": (
                3,
                "a7862de91cd3620ab7be1296a50b9030aa1c2cca2b4fe15d79abaf55f4664d17",
            ),
            "run005_final_degree1_official_loader_v1.json": (
                1,
                "6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c",
            ),
        }
        for filename, (degree, digest) in expected.items():
            with self.subTest(filename=filename):
                payload = json.loads((fixture_root / filename).read_text(encoding="utf-8"))
                self.assertEqual(
                    payload["schema_version"], "alod_dhier_official_loader_oracle.v1"
                )
                self.assertEqual(payload["official_source_commit"], "73f547341547ae021ff036ad74fdf73a1240b450")
                self.assertEqual(payload["sh_degree"], degree)
                self.assertEqual(payload["asset_sha256"], digest)
                self.assertIn("layer_a_raw_native_loader", payload)
                self.assertIn("layer_b_GaussianModel_create_from_hier", payload)

    def test_report_is_deterministic_and_json_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.dhier"
            write_fixture(path)
            asset = load_dhier(path)
            first = build_inspection_report(asset, validate_tree=True, repo_root=REPO_ROOT)
            second = build_inspection_report(asset, validate_tree=True, repo_root=REPO_ROOT)

            self.assertEqual(canonical_json(first), canonical_json(second))
            self.assertIsInstance(json.dumps(first, allow_nan=False), str)
            self.assertEqual(first["schema_version"], "alod_dhier_inspection.v1")
            self.assertEqual(first["validation"]["overall_valid"], True)

    def test_cli_exit_codes_and_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            valid_path = directory / "valid.dhier"
            invalid_path = directory / "invalid.dhier"
            report_path = directory / "report.json"
            write_fixture(valid_path)
            invalid_path.write_bytes(valid_path.read_bytes()[:-1])

            valid = subprocess.run(
                [
                    sys.executable,
                    str(CLI),
                    str(valid_path),
                    "--validate-topology",
                    "--json-out",
                    str(report_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            invalid = subprocess.run(
                [sys.executable, str(CLI), str(invalid_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)
            self.assertEqual(json.loads(report_path.read_text())["validation"]["overall_valid"], True)
            self.assertEqual(invalid.returncode, 2)

    def test_cli_distinguishes_numeric_topology_and_io_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            numeric_path = directory / "numeric.dhier"
            topology_path = directory / "topology.dhier"
            rotations = np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (3, 1))
            rotations[0] = 0
            write_fixture(numeric_path, rotations=rotations)
            bad_nodes = valid_nodes()
            bad_nodes[1, 1] = 2
            write_fixture(topology_path, nodes=bad_nodes)

            numeric = subprocess.run(
                [sys.executable, str(CLI), str(numeric_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            topology = subprocess.run(
                [sys.executable, str(CLI), str(topology_path), "--validate-topology"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            missing = subprocess.run(
                [sys.executable, str(CLI), str(directory / "missing.dhier")],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(numeric.returncode, 3)
            self.assertEqual(topology.returncode, 4)
            self.assertEqual(missing.returncode, 5)

    def test_package_import_does_not_import_torch_gsplat_or_citygaussian(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, alod_dhier; print(int(any(x in sys.modules for x in "
                "('torch','gsplat','scene_runtime.citygaussian','gaussian_hierarchy'))))",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "0")


if __name__ == "__main__":
    unittest.main()
