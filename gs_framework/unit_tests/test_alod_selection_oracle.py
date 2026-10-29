import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from alod_selection_oracle import (
    CAMERA_SCHEMA_VERSION,
    HSPT_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    PRE_RASTER_SCHEMA_VERSION,
    SPT_CUT_SCHEMA_VERSION,
    OracleValidationError,
    array_descriptor,
    canonical_json,
    global_id_hash,
    load_expected_selected_global_ids,
    save_array_payload,
    validate_global_ids,
    validate_hspt_record,
    validate_manifest,
    validate_oracle_tree,
    validate_pre_raster_record,
)


def id_summary(ids: np.ndarray) -> dict:
    return {
        "count": int(ids.size),
        "ordered_sha256": global_id_hash(ids),
        "sorted_sha256": global_id_hash(ids, sorted_ids=True),
    }


class GlobalIdentityHashTest(unittest.TestCase):
    def test_ordered_and_sorted_hashes_are_separate_and_deterministic(self):
        first = np.array([7, 2, 5], dtype=np.int32)
        second = np.array([2, 5, 7], dtype=np.int64)
        self.assertNotEqual(global_id_hash(first), global_id_hash(second))
        self.assertEqual(
            global_id_hash(first, sorted_ids=True),
            global_id_hash(second, sorted_ids=True),
        )
        self.assertEqual(global_id_hash(first), global_id_hash(first.copy()))

    def test_duplicate_and_out_of_range_ids_are_rejected(self):
        with self.assertRaisesRegex(OracleValidationError, "duplicates"):
            validate_global_ids(np.array([1, 1]), row_count=4)
        with self.assertRaisesRegex(OracleValidationError, "outside"):
            validate_global_ids(np.array([0, 4]), row_count=4)
        with self.assertRaisesRegex(OracleValidationError, "outside"):
            validate_global_ids(np.array([-1, 2]), row_count=4)


class DeterministicPayloadTest(unittest.TestCase):
    def test_array_payload_regeneration_is_byte_identical(self):
        array = np.arange(24, dtype=np.float32).reshape(4, 6)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = save_array_payload(root / "first.npy", array, relative_to=root)
            second = save_array_payload(root / "second.npy", array, relative_to=root)
            self.assertEqual(first["raw_cpu_c_order_sha256"], second["raw_cpu_c_order_sha256"])
            self.assertEqual(first["payload_file_sha256"], second["payload_file_sha256"])
            self.assertEqual(first["shape"], [4, 6])
            self.assertEqual(first["dtype"], "float32")

    def test_hspt_metadata_hashing_and_count_validation(self):
        upper = np.array([[0, -1, 2, 1, 0, 100]], dtype=np.int32)
        record = {
            "schema_version": HSPT_SCHEMA_VERSION,
            "configuration": {},
            "counts": {
                "upper_hierarchy_rows": 1,
                "spt_count": 1,
                "flattened_spt_metadata_rows": 3,
                "total_hspt_metadata_rows": 4,
            },
            "arrays": {"upper_tree_nodes": array_descriptor(upper)},
            "mappings": {},
        }
        validate_hspt_record(record)
        self.assertEqual(
            record["arrays"]["upper_tree_nodes"]["raw_cpu_c_order_sha256"],
            array_descriptor(upper.copy())["raw_cpu_c_order_sha256"],
        )
        record["counts"]["total_hspt_metadata_rows"] = 5
        with self.assertRaisesRegex(OracleValidationError, "total row"):
            validate_hspt_record(record)


class CameraAndPreRasterSchemaTest(unittest.TestCase):
    def test_camera_serialization_is_canonical_and_validated(self):
        record = {
            "schema_version": CAMERA_SCHEMA_VERSION,
            "camera_stable_id": "eval-0000-img-000001-cam-0001",
            "identity": {"registered_image_id": 1, "camera_id": 1},
            "frozen_dataset_camera": {"camera_center": [1.0, 2.0, 3.0]},
            "effective_official_runtime_camera": {"image_width": 960, "image_height": 540},
            "checks": {"identity": True, "pose": True},
        }
        first = canonical_json(record)
        second = canonical_json(json.loads(first))
        self.assertEqual(first, second)
        from alod_selection_oracle.format import validate_camera_record

        validate_camera_record(record)
        record["checks"]["pose"] = False
        with self.assertRaisesRegex(OracleValidationError, "failed identity"):
            validate_camera_record(record)

    def test_pre_raster_row_count_agreement(self):
        total = 5
        descriptor = {"shape": [total, 3]}
        record = {
            "schema_version": PRE_RASTER_SCHEMA_VERSION,
            "counts": {
                "skybox_raster_rows": 2,
                "hierarchy_selected_raster_rows": 3,
                "total_raster_gaussians": total,
            },
            "raster_global_node_ids": {"count": total},
            "tensors": {
                "means": descriptor,
                "quaternions": {"shape": [total, 4]},
                "scales": descriptor,
                "opacities": {"shape": [total]},
                "sh_features": {"shape": [total, 4, 3]},
            },
            "row_correspondence": {
                "valid": True,
                "stored_tensor_exact_equality": {"means": True},
            },
        }
        validate_pre_raster_record(record)
        record["tensors"]["means"] = {"shape": [total - 1, 3]}
        with self.assertRaisesRegex(OracleValidationError, "means"):
            validate_pre_raster_record(record)


class OracleReaderTest(unittest.TestCase):
    def test_tracked_run005_locator_is_compact_and_self_consistent(self):
        locator_path = (
            Path(__file__).parent
            / "fixtures/alod_selection_oracle/run005-selection-oracle-v1.json"
        )
        locator = json.loads(locator_path.read_text(encoding="utf-8"))
        self.assertEqual(locator["schema_version"], "alod_selection_oracle_locator.v1")
        self.assertEqual(len(locator["cameras"]), 10)
        self.assertEqual(
            len({camera["camera_stable_id"] for camera in locator["cameras"]}), 10
        )
        for camera in locator["cameras"]:
            self.assertGreater(camera["selected_hierarchy_count"], 0)
            for key in (
                "ordered_selected_global_ids_sha256",
                "sorted_selected_global_ids_sha256",
                "pre_raster_bundle_sha256",
            ):
                self.assertEqual(len(camera[key]), 64)

    def test_tracked_formal_mixed_locator_is_compact_and_self_consistent(self):
        locator_path = (
            Path(__file__).parent
            / "fixtures/alod_selection_oracle/formal-mixed-run009-selection-oracle-v1.json"
        )
        locator = json.loads(locator_path.read_text(encoding="utf-8"))
        self.assertEqual(locator["schema_version"], "alod_selection_oracle_locator.v1")
        self.assertEqual(locator["oracle_id"], "ALOD-MIXED-RUN009-SELECTION-ORACLE-001")
        self.assertEqual(locator["lod_classification"]["classification"], "WEAK-LOD-ORACLE")
        self.assertEqual(locator["project_reproduction_status"], "PASS-ALOD-CPU-REFERENCE-SELECTION")
        self.assertEqual(len(locator["cameras"]), 16)
        self.assertEqual(
            {camera["modality"] for camera in locator["cameras"]},
            {"aerial", "street"},
        )
        self.assertEqual(
            len({camera["camera_stable_id"] for camera in locator["cameras"]}),
            16,
        )
        for camera in locator["cameras"]:
            self.assertGreater(camera["selected_hierarchy_count"], 0)
            self.assertGreater(camera["candidate_spt_count"], 0)
            for key in (
                "ordered_selected_global_ids_sha256",
                "sorted_selected_global_ids_sha256",
                "ordered_raster_global_ids_sha256",
                "pre_raster_bundle_sha256",
                "determinism_signature_sha256",
            ):
                self.assertEqual(len(camera[key]), 64)

    def _write_minimal_tree(self, root: Path) -> Path:
        camera_id = "eval-0000-img-000001-cam-0001"
        camera_dir = root / "cameras" / camera_id
        payload_dir = camera_dir / "payloads" / "spt-cut"
        payload_dir.mkdir(parents=True)
        selected = np.array([3, 5], dtype=np.int32)
        selected_descriptor = save_array_payload(
            payload_dir / "selected_hierarchy_global_node_ids.npy",
            selected,
            relative_to=camera_dir,
        )
        cut = {
            "schema_version": SPT_CUT_SCHEMA_VERSION,
            "identity_summaries": {
                "selected_hierarchy_global_node_ids": id_summary(selected)
            },
            "arrays": {
                "selected_hierarchy_global_node_ids": selected_descriptor
            },
        }
        (camera_dir / "spt-cut.json").write_text(canonical_json(cut))
        camera_record = {
            "schema_version": CAMERA_SCHEMA_VERSION,
            "camera_stable_id": camera_id,
            "identity": {},
            "frozen_dataset_camera": {},
            "effective_official_runtime_camera": {},
            "checks": {"identity": True},
        }
        (camera_dir / "camera.json").write_text(canonical_json(camera_record))
        raster_descriptor = save_array_payload(
            camera_dir / "payloads/pre-raster/raster_global_node_ids.npy",
            np.array([0, 1, 3, 5], dtype=np.int64),
            relative_to=camera_dir,
        )
        working = {
            "global_node_id_payload": raster_descriptor,
        }
        (camera_dir / "working-set.json").write_text(canonical_json(working))
        pre = {
            "schema_version": PRE_RASTER_SCHEMA_VERSION,
            "counts": {
                "skybox_raster_rows": 2,
                "hierarchy_selected_raster_rows": 2,
                "total_raster_gaussians": 4,
            },
            "raster_global_node_ids": {
                **id_summary(np.array([0, 1, 3, 5], dtype=np.int64)),
            },
            "tensors": {
                "means": {"shape": [4, 3]},
                "quaternions": {"shape": [4, 4]},
                "scales": {"shape": [4, 3]},
                "opacities": {"shape": [4]},
                "sh_features": {"shape": [4, 4, 3]},
            },
            "row_correspondence": {
                "valid": True,
                "stored_tensor_exact_equality": {"means": True},
            },
        }
        (camera_dir / "pre-raster.json").write_text(canonical_json(pre))
        hspt_payload_dir = camera_dir / "payloads/hspt"
        upper_nodes = np.array(
            [[0, -1, 1, 2, 0, 2], [1, 0, 0, 0, 0, 3]], dtype=np.int32
        )
        hspt_arrays = {
            "upper_tree_nodes": save_array_payload(
                hspt_payload_dir / "upper_tree_nodes.npy",
                upper_nodes,
                relative_to=camera_dir,
            ),
            "upper_to_global_node_id": save_array_payload(
                hspt_payload_dir / "upper_to_global_node_id.npy",
                np.array([2, 3], dtype=np.int32),
                relative_to=camera_dir,
            ),
            "spt_starts": save_array_payload(
                hspt_payload_dir / "spt_starts.npy",
                np.array([0, 5], dtype=np.int32),
                relative_to=camera_dir,
            ),
            "spt_to_global_node_id": save_array_payload(
                hspt_payload_dir / "spt_to_global_node_id.npy",
                np.array([3, 4, 5, 6, 7], dtype=np.int32),
                relative_to=camera_dir,
            ),
            "spt_root_global_node_ids": save_array_payload(
                hspt_payload_dir / "spt_root_global_node_ids.npy",
                np.array([3], dtype=np.int32),
                relative_to=camera_dir,
            ),
        }
        hspt = {
            "schema_version": HSPT_SCHEMA_VERSION,
            "configuration": {},
            "counts": {
                "upper_hierarchy_rows": 2,
                "spt_count": 1,
                "flattened_spt_metadata_rows": 5,
                "total_hspt_metadata_rows": 7,
            },
            "arrays": hspt_arrays,
            "mappings": {},
        }
        (camera_dir / "hspt.json").write_text(canonical_json(hspt))
        paths = {
            "camera.json": f"cameras/{camera_id}/camera.json",
            "spt-cut.json": f"cameras/{camera_id}/spt-cut.json",
            "working-set.json": f"cameras/{camera_id}/working-set.json",
            "pre-raster.json": f"cameras/{camera_id}/pre-raster.json",
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "oracle_id": "tiny",
            "global_node_id_contract": "serialized .dhier row index",
            "asset": {
                "sha256": "0" * 64,
                "rows": 8,
                "skybox_rows": 2,
                "root_global_node_id": 2,
            },
            "official_runtime": {},
            "hspt": {"path": f"cameras/{camera_id}/hspt.json"},
            "cameras": [
                {
                    "camera_stable_id": camera_id,
                    "registered_image_id": 1,
                    "camera_id": 1,
                    "stable_sorted_index": 0,
                    "selected_hierarchy_count": 2,
                    "ordered_selected_global_ids_sha256": global_id_hash(selected),
                    "sorted_selected_global_ids_sha256": global_id_hash(selected, sorted_ids=True),
                    "paths": paths,
                }
            ],
        }
        manifest_path = root / "semantic-manifest.json"
        manifest_path.write_text(canonical_json(manifest))
        return manifest_path

    def test_schema_validation_and_expected_id_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._write_minimal_tree(root)
            manifest = json.loads(manifest_path.read_text())
            validate_manifest(manifest)
            camera_dir = next((root / "cameras").iterdir())
            self.assertEqual(load_expected_selected_global_ids(camera_dir).tolist(), [3, 5])
            result = validate_oracle_tree(manifest_path)
            self.assertTrue(result["valid"])
            self.assertEqual(result["camera_count"], 1)

    def test_duplicate_selected_ids_are_rejected_by_tree_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._write_minimal_tree(root)
            camera_dir = next((root / "cameras").iterdir())
            payload = camera_dir / "payloads/spt-cut/selected_hierarchy_global_node_ids.npy"
            duplicate = np.array([3, 3], dtype=np.int32)
            descriptor = save_array_payload(payload, duplicate, relative_to=camera_dir)
            cut_path = camera_dir / "spt-cut.json"
            cut = json.loads(cut_path.read_text())
            cut["arrays"]["selected_hierarchy_global_node_ids"] = descriptor
            cut["identity_summaries"]["selected_hierarchy_global_node_ids"] = id_summary(duplicate)
            cut_path.write_text(canonical_json(cut))
            with self.assertRaisesRegex(OracleValidationError, "duplicates"):
                validate_oracle_tree(manifest_path)


if __name__ == "__main__":
    unittest.main()
