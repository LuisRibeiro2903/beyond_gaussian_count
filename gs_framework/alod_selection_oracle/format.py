"""Deterministic format helpers for official ALoD selection oracles.

The canonical ID encoding is little-endian signed int64 in C order.  Official
runtime arrays retain their actual dtype in payload descriptors, but ordered
and sorted identity hashes use this platform-independent encoding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MANIFEST_SCHEMA_VERSION = "alod_selection_oracle_manifest.v1"
HSPT_SCHEMA_VERSION = "alod_selection_oracle_hspt.v1"
CAMERA_SCHEMA_VERSION = "alod_selection_oracle_camera.v1"
UPPER_TRAVERSAL_SCHEMA_VERSION = "alod_selection_oracle_upper_traversal.v1"
SPT_CUT_SCHEMA_VERSION = "alod_selection_oracle_spt_cut.v1"
PRE_RASTER_SCHEMA_VERSION = "alod_selection_oracle_pre_raster.v1"
GLOBAL_ID_ENCODING = "little-endian signed int64, contiguous C order"


class OracleValidationError(ValueError):
    """Raised when an oracle artifact violates its stable schema contract."""


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(value)


def canonicalize_global_ids(ids: Any) -> np.ndarray:
    array = _to_numpy(ids)
    if array.ndim != 1:
        raise OracleValidationError(f"global IDs must be one-dimensional, got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise OracleValidationError(f"global IDs must be integral, got {array.dtype}")
    return np.ascontiguousarray(array, dtype="<i8")


def global_id_hash(ids: Any, *, sorted_ids: bool = False) -> str:
    array = canonicalize_global_ids(ids)
    if sorted_ids:
        array = np.sort(array, kind="stable")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def validate_global_ids(
    ids: Any,
    *,
    row_count: int,
    allow_duplicates: bool = False,
    label: str = "global_node_ids",
) -> np.ndarray:
    array = canonicalize_global_ids(ids)
    if row_count <= 0:
        raise OracleValidationError("row_count must be positive")
    if array.size:
        minimum = int(array.min())
        maximum = int(array.max())
        if minimum < 0 or maximum >= row_count:
            raise OracleValidationError(
                f"{label} outside [0, {row_count}): min={minimum}, max={maximum}"
            )
        if not allow_duplicates and np.unique(array).size != array.size:
            raise OracleValidationError(f"{label} contains unexpected duplicates")
    return array


def array_descriptor(value: Any, *, device: str | None = None) -> dict[str, Any]:
    array = _to_numpy(value)
    numeric = np.issubdtype(array.dtype, np.number)
    floating = np.issubdtype(array.dtype, np.floating)
    finite_count = int(np.count_nonzero(np.isfinite(array))) if floating else None
    return {
        "shape": [int(component) for component in array.shape],
        "dtype": str(array.dtype),
        "device_during_official_execution": device,
        "byte_count": int(array.nbytes),
        "element_count": int(array.size),
        "raw_cpu_c_order_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "finite_count": finite_count,
        "min": float(np.min(array)) if numeric and array.size else None,
        "max": float(np.max(array)) if numeric and array.size else None,
    }


def save_array_payload(
    path: str | Path,
    value: Any,
    *,
    device: str | None = None,
    relative_to: str | Path | None = None,
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = _to_numpy(value)
    np.save(destination, array, allow_pickle=False)
    descriptor = array_descriptor(array, device=device)
    descriptor["payload_path"] = (
        destination.relative_to(Path(relative_to)).as_posix()
        if relative_to is not None
        else str(destination)
    )
    descriptor["payload_file_sha256"] = sha256_file(destination)
    return descriptor


def _require_keys(payload: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    missing = sorted(set(keys) - set(payload))
    if missing:
        raise OracleValidationError(f"{label} missing required keys: {missing}")


def validate_manifest(payload: Mapping[str, Any]) -> None:
    _require_keys(
        payload,
        (
            "schema_version",
            "oracle_id",
            "asset",
            "official_runtime",
            "hspt",
            "cameras",
            "global_node_id_contract",
        ),
        "manifest",
    )
    if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise OracleValidationError(
            f"unsupported manifest schema {payload['schema_version']!r}"
        )
    if payload["global_node_id_contract"] != "serialized .dhier row index":
        raise OracleValidationError("manifest does not use serialized row global IDs")
    asset = payload["asset"]
    _require_keys(asset, ("sha256", "rows", "skybox_rows", "root_global_node_id"), "asset")
    if int(asset["rows"]) <= 0:
        raise OracleValidationError("asset row count must be positive")
    if not 0 <= int(asset["root_global_node_id"]) < int(asset["rows"]):
        raise OracleValidationError("asset root global ID is out of range")
    cameras = payload["cameras"]
    if not isinstance(cameras, list) or not cameras:
        raise OracleValidationError("manifest must contain at least one camera")
    stable_ids = []
    for camera in cameras:
        _require_keys(
            camera,
            (
                "camera_stable_id",
                "registered_image_id",
                "camera_id",
                "stable_sorted_index",
                "selected_hierarchy_count",
                "ordered_selected_global_ids_sha256",
                "sorted_selected_global_ids_sha256",
                "paths",
            ),
            "camera",
        )
        stable_ids.append(camera["camera_stable_id"])
        if int(camera["selected_hierarchy_count"]) < 0:
            raise OracleValidationError("selected hierarchy count cannot be negative")
    if len(set(stable_ids)) != len(stable_ids):
        raise OracleValidationError("duplicate camera_stable_id in manifest")


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(payload)
    return payload


def _load_payload_array(
    descriptor: Mapping[str, Any], *, root: Path, label: str
) -> np.ndarray:
    _require_keys(
        descriptor,
        ("payload_path", "payload_file_sha256", "raw_cpu_c_order_sha256", "shape", "dtype"),
        label,
    )
    path = root / str(descriptor["payload_path"])
    if not path.is_file():
        raise OracleValidationError(f"missing payload for {label}: {path}")
    if sha256_file(path) != descriptor["payload_file_sha256"]:
        raise OracleValidationError(f"payload file hash mismatch for {label}")
    array = np.load(path, allow_pickle=False)
    actual = array_descriptor(array)
    for key in ("shape", "dtype", "raw_cpu_c_order_sha256"):
        if actual[key] != descriptor[key]:
            raise OracleValidationError(f"payload descriptor mismatch for {label}.{key}")
    return array


def validate_record_payloads(record: Mapping[str, Any], *, root: str | Path) -> int:
    """Verify every recursively referenced NumPy payload in one stage record."""

    payload_root = Path(root)
    validated = 0

    def visit(value: Any, label: str) -> None:
        nonlocal validated
        if isinstance(value, Mapping):
            if "payload_path" in value:
                _load_payload_array(value, root=payload_root, label=label)
                validated += 1
            for key, child in value.items():
                visit(child, f"{label}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{label}[{index}]")

    visit(record, "record")
    return validated


def load_expected_selected_global_ids(camera_dir: str | Path) -> np.ndarray:
    root = Path(camera_dir)
    record = json.loads((root / "spt-cut.json").read_text(encoding="utf-8"))
    if record.get("schema_version") != SPT_CUT_SCHEMA_VERSION:
        raise OracleValidationError("unsupported SPT-cut schema")
    descriptor = record["arrays"]["selected_hierarchy_global_node_ids"]
    ids = canonicalize_global_ids(
        _load_payload_array(descriptor, root=root, label="selected_hierarchy_global_node_ids")
    )
    summary = record["identity_summaries"]["selected_hierarchy_global_node_ids"]
    if ids.size != int(summary["count"]):
        raise OracleValidationError("selected global-ID count disagrees with payload")
    if global_id_hash(ids) != summary["ordered_sha256"]:
        raise OracleValidationError("ordered selected global-ID hash mismatch")
    if global_id_hash(ids, sorted_ids=True) != summary["sorted_sha256"]:
        raise OracleValidationError("sorted selected global-ID hash mismatch")
    return ids


def validate_camera_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != CAMERA_SCHEMA_VERSION:
        raise OracleValidationError("unsupported camera schema")
    _require_keys(
        record,
        (
            "camera_stable_id",
            "identity",
            "frozen_dataset_camera",
            "effective_official_runtime_camera",
            "checks",
        ),
        "camera record",
    )
    if not all(bool(value) for value in record["checks"].values()):
        raise OracleValidationError("camera record contains a failed identity check")


def validate_hspt_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != HSPT_SCHEMA_VERSION:
        raise OracleValidationError("unsupported HSPT schema")
    _require_keys(record, ("configuration", "counts", "arrays", "mappings"), "HSPT")
    counts = record["counts"]
    if int(counts["spt_count"]) < 0 or int(counts["upper_hierarchy_rows"]) <= 0:
        raise OracleValidationError("invalid HSPT counts")
    if int(counts["total_hspt_metadata_rows"]) != int(
        counts["upper_hierarchy_rows"]
    ) + int(counts["flattened_spt_metadata_rows"]):
        raise OracleValidationError("HSPT total row count is inconsistent")


def validate_hspt_payload_mappings(
    record: Mapping[str, Any], *, root: str | Path, row_count: int, skybox_rows: int
) -> dict[str, int]:
    """Prove that both HSPT ID spaces map exactly onto serialized hierarchy rows."""

    arrays = record["arrays"]
    required = (
        "upper_tree_nodes",
        "upper_to_global_node_id",
        "spt_starts",
        "spt_to_global_node_id",
        "spt_root_global_node_ids",
    )
    if not all("payload_path" in arrays.get(name, {}) for name in required):
        raise OracleValidationError("canonical HSPT record lacks full mapping payloads")
    payload_root = Path(root)
    loaded = {
        name: _load_payload_array(arrays[name], root=payload_root, label=name)
        for name in required
    }
    upper_nodes = loaded["upper_tree_nodes"]
    upper = validate_global_ids(
        loaded["upper_to_global_node_id"], row_count=row_count, label="upper global IDs"
    )
    spt = validate_global_ids(
        loaded["spt_to_global_node_id"], row_count=row_count, label="SPT global IDs"
    )
    roots = validate_global_ids(
        loaded["spt_root_global_node_ids"], row_count=row_count, label="SPT roots"
    )
    starts = canonicalize_global_ids(loaded["spt_starts"])
    if upper_nodes.ndim != 2 or upper_nodes.shape[1] != 6:
        raise OracleValidationError("upper_tree_nodes must have six columns")
    if not np.array_equal(canonicalize_global_ids(upper_nodes[:, 5]), upper):
        raise OracleValidationError("upper-tree column five is not the exported global-ID map")
    if starts.size != roots.size + 1 or starts[0] != 0 or starts[-1] != spt.size:
        raise OracleValidationError("SPT starts do not delimit the flattened global-ID map")
    if np.any(np.diff(starts) <= 0):
        raise OracleValidationError("SPT intervals must be non-empty and increasing")
    if not np.array_equal(spt[starts[:-1]], roots):
        raise OracleValidationError("SPT roots do not occupy interval starts")
    overlap = np.intersect1d(upper, spt)
    if not np.array_equal(np.sort(overlap), np.sort(roots)):
        raise OracleValidationError("upper/SPT overlap is not exactly the SPT roots")
    expected = np.arange(skybox_rows, row_count, dtype="<i8")
    if not np.array_equal(np.union1d(upper, spt), expected):
        raise OracleValidationError("HSPT mappings do not cover every hierarchy row exactly")
    return {
        "upper_unique_global_ids": int(upper.size),
        "spt_unique_global_ids": int(spt.size),
        "shared_spt_roots": int(overlap.size),
        "covered_hierarchy_global_ids": int(expected.size),
    }


def validate_pre_raster_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != PRE_RASTER_SCHEMA_VERSION:
        raise OracleValidationError("unsupported pre-raster schema")
    counts = record["counts"]
    total = int(counts["total_raster_gaussians"])
    if total != int(counts["skybox_raster_rows"]) + int(
        counts["hierarchy_selected_raster_rows"]
    ):
        raise OracleValidationError("pre-raster total does not equal skybox plus hierarchy rows")
    if int(record["raster_global_node_ids"]["count"]) != total:
        raise OracleValidationError("raster global-ID count disagrees with pre-raster total")
    for name in ("means", "quaternions", "scales", "opacities", "sh_features"):
        shape = record["tensors"][name]["shape"]
        if not shape or int(shape[0]) != total:
            raise OracleValidationError(f"pre-raster tensor row mismatch for {name}")
    correspondence = record["row_correspondence"]
    if not correspondence.get("valid"):
        raise OracleValidationError("pre-raster row correspondence is not valid")
    if not all(correspondence["stored_tensor_exact_equality"].values()):
        raise OracleValidationError("stored working-set tensors do not match global-ID rows")


def validate_oracle_tree(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    root = path.parent
    manifest = load_manifest(path)
    semantic_reference = manifest.get("deterministic_semantic_manifest")
    if semantic_reference is not None:
        semantic_path = root / str(semantic_reference["path"])
        if not semantic_path.is_file():
            raise OracleValidationError("missing deterministic semantic manifest")
        if sha256_file(semantic_path) != semantic_reference["sha256"]:
            raise OracleValidationError("deterministic semantic manifest hash mismatch")
    rows = int(manifest["asset"]["rows"])
    skybox_rows = int(manifest["asset"]["skybox_rows"])
    validated_cameras = []
    payload_count = 0
    stage_json_hash_count = 0
    hspt_mapping_proof: dict[str, int] | None = None
    for camera in manifest["cameras"]:
        paths = camera["paths"]
        camera_path = root / paths["camera.json"]
        hspt_path = root / manifest["hspt"]["path"]
        cut_path = root / paths["spt-cut.json"]
        pre_path = root / paths["pre-raster.json"]
        working_path = root / paths["working-set.json"]
        for required in (camera_path, hspt_path, cut_path, working_path, pre_path):
            if not required.is_file():
                raise OracleValidationError(f"missing oracle record {required}")
        camera_record = json.loads(camera_path.read_text(encoding="utf-8"))
        hspt_record = json.loads(hspt_path.read_text(encoding="utf-8"))
        cut_record = json.loads(cut_path.read_text(encoding="utf-8"))
        working_record = json.loads(working_path.read_text(encoding="utf-8"))
        pre_record = json.loads(pre_path.read_text(encoding="utf-8"))
        validate_camera_record(camera_record)
        validate_hspt_record(hspt_record)
        if hspt_mapping_proof is None:
            hspt_mapping_proof = validate_hspt_payload_mappings(
                hspt_record,
                root=hspt_path.parent,
                row_count=rows,
                skybox_rows=skybox_rows,
            )
        validate_pre_raster_record(pre_record)
        camera_dir = cut_path.parent
        complete_path = camera_dir / "complete.json"
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            stage_names = {
                "asset": "asset.json",
                "camera": "camera.json",
                "hspt": "hspt.json",
                "pre_raster": "pre-raster.json",
                "render": "optional-image.json",
                "spt_cut": "spt-cut.json",
                "upper_traversal": "upper-traversal.json",
                "working_set": "working-set.json",
            }
            for stage, expected_hash in complete["stage_json_sha256"].items():
                stage_path = camera_dir / stage_names[stage]
                if sha256_file(stage_path) != expected_hash:
                    raise OracleValidationError(f"stage JSON hash mismatch for {stage_path}")
                stage_json_hash_count += 1
        determinism_path = camera_dir / "determinism.json"
        if determinism_path.is_file():
            determinism = json.loads(determinism_path.read_text(encoding="utf-8"))
            if not determinism.get("exact_match") or int(
                determinism.get("cold_process_repetitions", 0)
            ) < 2:
                raise OracleValidationError("camera lacks two-run cold determinism proof")
        for record_name in (
            "hspt.json",
            "upper-traversal.json",
            "spt-cut.json",
            "working-set.json",
            "pre-raster.json",
            "optional-image.json",
        ):
            record_path = camera_dir / record_name
            if record_path.is_file():
                payload_count += validate_record_payloads(
                    json.loads(record_path.read_text(encoding="utf-8")), root=camera_dir
                )
        selected = validate_global_ids(
            load_expected_selected_global_ids(camera_dir),
            row_count=rows,
            label="selected_hierarchy_global_node_ids",
        )
        if selected.size != int(camera["selected_hierarchy_count"]):
            raise OracleValidationError("manifest selected count disagrees with payload")
        if global_id_hash(selected) != camera["ordered_selected_global_ids_sha256"]:
            raise OracleValidationError("manifest ordered selected-ID hash mismatch")
        if global_id_hash(selected, sorted_ids=True) != camera["sorted_selected_global_ids_sha256"]:
            raise OracleValidationError("manifest sorted selected-ID hash mismatch")
        raster_descriptor = working_record["global_node_id_payload"]
        raster_ids = validate_global_ids(
            _load_payload_array(
                raster_descriptor, root=camera_dir, label="raster_global_node_ids"
            ),
            row_count=rows,
            label="raster_global_node_ids",
        )
        expected_raster_ids = np.concatenate(
            (np.arange(skybox_rows, dtype="<i8"), selected)
        )
        if global_id_hash(raster_ids) != pre_record["raster_global_node_ids"]["ordered_sha256"]:
            raise OracleValidationError("pre-raster ordered global-ID hash mismatch")
        if not np.array_equal(raster_ids, expected_raster_ids):
            raise OracleValidationError(
                "raster rows do not equal the skybox prefix followed by selected hierarchy IDs"
            )
        validated_cameras.append(camera["camera_stable_id"])
    return {
        "valid": True,
        "oracle_id": manifest["oracle_id"],
        "camera_count": len(validated_cameras),
        "camera_stable_ids": validated_cameras,
        "validated_payload_references": payload_count,
        "validated_stage_json_hashes": stage_json_hash_count,
        "hspt_mapping_proof": hspt_mapping_proof,
    }
