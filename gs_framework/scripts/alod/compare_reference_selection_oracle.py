#!/usr/bin/env python3
"""Compare the project CPU ALoD reference selector with a frozen oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_dhier import load_dhier
from alod_reference import (
    HierarchyData,
    ReferenceCamera,
    ReferenceConfig,
    build_hspt,
    select_reference,
)
from alod_selection_oracle import (
    global_id_hash,
    load_manifest,
    sha256_file,
    validate_oracle_tree,
)


PASS_STATUS = "PASS-ALOD-CPU-REFERENCE-SELECTION"


class ParityFailure(RuntimeError):
    def __init__(self, layer: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.layer = layer
        self.details = details or {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_described_array(record_root: Path, descriptor: dict[str, Any]) -> np.ndarray:
    return np.load(record_root / descriptor["payload_path"], allow_pickle=False)


def _first_difference(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    differences = np.argwhere(actual != expected)
    if differences.size == 0:
        return {}
    index = tuple(int(value) for value in differences[0])
    return {
        "index": list(index),
        "actual": actual[index].item(),
        "expected": expected[index].item(),
        "mismatch_count": int(np.count_nonzero(actual != expected)),
    }


def _require_array_equal(
    layer: str, label: str, actual: np.ndarray, expected: np.ndarray
) -> None:
    if actual.dtype != expected.dtype:
        raise ParityFailure(
            layer,
            f"{label} dtype differs",
            {"actual_dtype": str(actual.dtype), "expected_dtype": str(expected.dtype)},
        )
    if not np.array_equal(actual, expected):
        raise ParityFailure(
            layer, f"{label} differs", _first_difference(actual, expected)
        )


def _hspt_arrays(hspt: Any) -> dict[str, np.ndarray]:
    return {
        "upper_tree_nodes": hspt.upper_tree_nodes,
        "upper_to_global_node_id": hspt.upper_to_global_node_id,
        "upper_tree_xyz": hspt.upper_tree_xyz,
        "upper_tree_log_scaling": hspt.upper_tree_log_scaling,
        "upper_min_distance_squared": hspt.upper_min_distance_squared,
        "spt_starts": hspt.spt_starts,
        "spt_min": hspt.spt_min,
        "spt_max": hspt.spt_max,
        "spt_to_global_node_id": hspt.spt_to_global_node_id,
        "spt_root_global_node_ids": hspt.spt_root_global_node_ids,
        "spt_sizes": hspt.spt_sizes,
    }


def _compare_hspt(hspt: Any, record: dict[str, Any], record_root: Path) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for name, actual in _hspt_arrays(hspt).items():
        expected = _load_described_array(record_root, record["arrays"][name])
        _require_array_equal("A/B HSPT decomposition and mappings", name, actual, expected)
        hashes[name] = hashlib.sha256(actual.tobytes(order="C")).hexdigest()
    return hashes


def _camera_from_record(record: dict[str, Any]) -> ReferenceCamera:
    runtime = record["effective_official_runtime_camera"]
    return ReferenceCamera(
        stable_id=record["camera_stable_id"],
        camera_center=np.asarray(runtime["camera_center_float32"], dtype=np.float32),
        full_projection_transform=np.asarray(
            runtime["full_projection_transform"]["values"], dtype=np.float32
        ),
        registered_image_id=int(record["identity"]["registered_image_id"]),
    )


def _compare_camera_input(camera: ReferenceCamera, record: dict[str, Any]) -> None:
    runtime = record["effective_official_runtime_camera"]
    _require_array_equal(
        "C effective camera",
        "camera_center",
        camera.camera_center,
        np.asarray(runtime["camera_center_float32"], dtype=np.float32),
    )
    _require_array_equal(
        "C effective camera",
        "full_projection_transform",
        camera.full_projection_transform,
        np.asarray(runtime["full_projection_transform"]["values"], dtype=np.float32),
    )


def _compare_upper(result: Any, camera_dir: Path, record: dict[str, Any]) -> None:
    expected_planes = np.asarray(record["frustum"]["planes"]["values"], dtype=np.float32)
    _require_array_equal(
        "D four-plane frustum", "frustum_planes", result.upper.frustum_planes, expected_planes
    )
    arrays = record["arrays"]
    mappings = {
        "coarse_cut_upper_local_ids": result.upper.coarse_cut_upper_local_ids,
        "coarse_cut_global_node_ids": result.upper.coarse_cut_global_node_ids,
        "cut_leaf_upper_local_ids": result.upper.cut_leaf_upper_local_ids,
        "cut_leaf_global_node_ids": result.upper.cut_leaf_global_node_ids,
        "upper_render_global_node_ids": result.upper.upper_render_global_node_ids,
        "candidate_spt_ids": result.upper.candidate_spt_ids,
        "candidate_spt_upper_local_ids": result.upper.candidate_spt_upper_local_ids,
        "candidate_spt_root_global_node_ids": result.upper.candidate_spt_root_global_node_ids,
        "candidate_spt_distances": result.upper.candidate_spt_distances,
    }
    candidate_names = {
        "candidate_spt_ids",
        "candidate_spt_upper_local_ids",
        "candidate_spt_root_global_node_ids",
        "candidate_spt_distances",
    }
    for name, actual in mappings.items():
        expected = _load_described_array(camera_dir, arrays[name])
        layer = "F candidate SPT ordering" if name in candidate_names else "E upper traversal"
        _require_array_equal(layer, name, actual, expected)


def _spt_diagnostic(
    hspt: Any,
    result: Any,
    spt_index: int,
    expected: np.ndarray,
) -> dict[str, Any]:
    project = result.per_spt[spt_index]
    actual = project.emitted_global_node_ids
    details: dict[str, Any] = {
        "camera": result.camera_stable_id,
        "candidate_index": spt_index,
        "spt_id": project.spt_id,
        "spt_root_global_node_id": project.root_global_node_id,
        "camera_distance": float(project.distance),
        "interval_size": project.interval_size,
        "project_count": int(actual.size),
        "official_count": int(expected.size),
    }
    difference = _first_difference(actual, expected)
    details.update(difference)
    candidate_ids = []
    if "index" in difference and difference["index"]:
        offset = int(difference["index"][0])
        if offset < actual.size:
            candidate_ids.append(int(actual[offset]))
        if offset < expected.size:
            candidate_ids.append(int(expected[offset]))
    for global_id in candidate_ids[:2]:
        positions = np.flatnonzero(hspt.spt_to_global_node_id == global_id)
        if positions.size:
            flat = int(positions[0])
            details.setdefault("threshold_rows", []).append(
                {
                    "global_node_id": global_id,
                    "spt_flat_index": flat,
                    "spt_min": float(hspt.spt_min[flat]),
                    "spt_max": float(hspt.spt_max[flat]),
                    "official_decision_inferred_from_payload": bool(global_id in set(expected.tolist())),
                    "project_decision": bool(global_id in set(actual.tolist())),
                }
            )
    return details


def _compare_spts(hspt: Any, result: Any, camera_dir: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    raw_expected = _load_described_array(
        camera_dir, record["arrays"]["raw_cuda_spt_global_node_ids"]
    )
    prefix = _load_described_array(
        camera_dir, record["arrays"]["raw_cuda_spt_prefix_starts"]
    )
    summaries: list[dict[str, Any]] = []
    for index, project in enumerate(result.per_spt):
        start = int(prefix[index])
        end = int(prefix[index + 1]) if index + 1 < prefix.size else int(raw_expected.size)
        expected = raw_expected[start:end]
        if not np.array_equal(project.emitted_global_node_ids, expected):
            raise ParityFailure(
                "G individual SPT cut",
                f"first mismatching SPT at candidate index {index}",
                _spt_diagnostic(hspt, result, index, expected),
            )
        summaries.append(
            {
                "spt_id": project.spt_id,
                "root_global_node_id": project.root_global_node_id,
                "emitted_count": int(project.emitted_global_node_ids.size),
                "ordered_sha256": global_id_hash(project.emitted_global_node_ids),
            }
        )
    _require_array_equal(
        "G individual SPT cut",
        "combined_spt_global_node_ids",
        result.selected_spt_global_node_ids,
        raw_expected,
    )
    return summaries


def compare(
    *, dhier_path: Path, oracle_manifest_path: Path
) -> dict[str, Any]:
    manifest = load_manifest(oracle_manifest_path)
    oracle_root = oracle_manifest_path.resolve().parent
    expected_asset_hash = manifest["asset"]["sha256"]
    actual_asset_hash = sha256_file(dhier_path)
    if actual_asset_hash != expected_asset_hash:
        raise ParityFailure(
            "asset",
            "hierarchy SHA-256 differs",
            {"actual": actual_asset_hash, "expected": expected_asset_hash},
        )

    hspt_path = oracle_root / manifest["hspt"]["path"]
    hspt_record = _load_json(hspt_path)
    hspt_config = hspt_record["configuration"]
    config = ReferenceConfig(
        root_global_node_id=int(manifest["asset"]["root_global_node_id"]),
        skybox_count=int(manifest["asset"]["skybox_rows"]),
        spt_root_volume=float(hspt_config["SPT_root_volume"]),
        min_spt_size=int(hspt_config["min_SPT_size_effective_evaluation_default"]),
        target_granularity=float(hspt_config["resolved_target_granularity"]),
        distance_multiplier=1.0,
    )

    asset = load_dhier(dhier_path)
    build_start = time.perf_counter()
    hspt = build_hspt(HierarchyData.from_dhier(asset), config)
    build_seconds = time.perf_counter() - build_start

    # HSPT oracle payloads are first opened only after independent construction.
    validate_oracle_tree(oracle_manifest_path)
    hspt_hashes = _compare_hspt(hspt, hspt_record, hspt_path.parent)
    camera_results: list[dict[str, Any]] = []
    for camera_entry in manifest["cameras"]:
        camera_json_path = oracle_root / camera_entry["paths"]["camera.json"]
        camera_dir = camera_json_path.parent
        camera_record = _load_json(camera_json_path)
        camera = _camera_from_record(camera_record)
        _compare_camera_input(camera, camera_record)
        result = select_reference(hspt, camera, config)

        upper_record = _load_json(camera_dir / "upper-traversal.json")
        _compare_upper(result, camera_dir, upper_record)
        cut_record = _load_json(camera_dir / "spt-cut.json")
        per_spt = _compare_spts(hspt, result, camera_dir, cut_record)

        expected_selected = _load_described_array(
            camera_dir, cut_record["arrays"]["selected_hierarchy_global_node_ids"]
        )
        _require_array_equal(
            "H combined ordered hierarchy selection",
            "selected_hierarchy_global_node_ids",
            result.selected_hierarchy_global_node_ids,
            expected_selected,
        )
        expected_sorted = _load_described_array(
            camera_dir,
            cut_record["arrays"]["sorted_selected_hierarchy_global_node_ids"],
        )
        actual_sorted = np.sort(result.selected_hierarchy_global_node_ids, kind="stable")
        _require_array_equal(
            "I sorted hierarchy selection",
            "sorted_selected_hierarchy_global_node_ids",
            actual_sorted,
            expected_sorted,
        )
        working_record = _load_json(camera_dir / "working-set.json")
        expected_raster = _load_described_array(
            camera_dir, working_record["global_node_id_payload"]
        )
        _require_array_equal(
            "J final raster-row vector",
            "raster_global_node_ids",
            result.raster_global_node_ids,
            expected_raster,
        )
        camera_results.append(
            {
                "camera_stable_id": camera.stable_id,
                "registered_image_id": camera.registered_image_id,
                "candidate_spt_count": int(result.upper.candidate_spt_ids.size),
                "selected_hierarchy_count": int(
                    result.selected_hierarchy_global_node_ids.size
                ),
                "raster_count": int(result.raster_global_node_ids.size),
                "ordered_spt_sha256": global_id_hash(
                    result.selected_spt_global_node_ids
                ),
                "ordered_selected_sha256": global_id_hash(
                    result.selected_hierarchy_global_node_ids
                ),
                "sorted_selected_sha256": global_id_hash(
                    result.selected_hierarchy_global_node_ids, sorted_ids=True
                ),
                "upper_traversal_seconds": result.timings.upper_traversal_seconds,
                "spt_cut_seconds": result.timings.spt_cut_seconds,
                "total_selection_seconds": result.timings.total_seconds,
                "parity": {
                    "effective_camera": True,
                    "frustum": True,
                    "upper_traversal": True,
                    "candidate_spts": True,
                    "per_spt_ordered_cuts": True,
                    "combined_ordered_selection": True,
                    "sorted_selection": True,
                    "raster_rows": True,
                },
                "per_spt": per_spt,
            }
        )

    return {
        "status": PASS_STATUS,
        "asset": {
            "path": str(dhier_path.resolve()),
            "sha256": actual_asset_hash,
            "rows": int(asset.layout.node_count),
        },
        "oracle": {
            "manifest": str(oracle_manifest_path.resolve()),
            "semantic_manifest_sha256": manifest["deterministic_semantic_manifest"]["sha256"],
        },
        "configuration": {
            "root_global_node_id": config.root_global_node_id,
            "skybox_count": config.skybox_count,
            "SPT_root_volume": config.spt_root_volume,
            "min_SPT_size": config.min_spt_size,
            "target_granularity": config.target_granularity,
            "distance_multiplier": config.distance_multiplier,
        },
        "hspt": {
            "upper_rows": hspt.upper_row_count,
            "spt_count": hspt.spt_count,
            "flattened_spt_rows": hspt.flattened_spt_row_count,
            "build_seconds": build_seconds,
            "raw_hashes": hspt_hashes,
            "parity": True,
        },
        "cameras": camera_results,
        "performance": {
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dhier", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    try:
        report = compare(
            dhier_path=args.dhier.resolve(),
            oracle_manifest_path=args.oracle_manifest.resolve(),
        )
    except ParityFailure as exc:
        failure = {
            "status": "FAIL-ALOD-CPU-REFERENCE-SELECTION",
            "first_divergent_layer": exc.layer,
            "message": str(exc),
            "details": exc.details,
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
