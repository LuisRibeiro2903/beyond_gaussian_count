#!/usr/bin/env python3
"""Apply the synthetic-control hierarchy and selector diagnostics to an oracle.

This is a read-only diagnostic.  It reconstructs the HSPT from the serialized
hierarchy and the configuration frozen in an official selection oracle, while
using the official selected-ID payloads for selected-state change counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_dhier import load_dhier  # noqa: E402
from alod_reference import HierarchyData, ReferenceConfig, build_hspt  # noqa: E402


CHILD_COUNT = 2
PARENT = 1


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _described_array(base: Path, description: dict[str, Any]) -> np.ndarray:
    return np.load(base / description["payload_path"], allow_pickle=False)


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _structure(asset: Any, hierarchy_start: int) -> dict[str, Any]:
    nodes = np.asarray(asset.nodes)
    depths = nodes[hierarchy_start:, 0].astype(np.int16, copy=False)
    extents = np.float32(6.0) * np.max(
        np.exp(np.asarray(asset.log_scales[hierarchy_start:], dtype=np.float32)), axis=1
    )
    parent = nodes[hierarchy_start:, PARENT].astype(np.int64, copy=False)
    valid_parent = parent >= hierarchy_start
    ratio = np.full(depths.size, np.nan, dtype=np.float32)
    ratio[valid_parent] = extents[parent[valid_parent] - hierarchy_start] / np.maximum(
        extents[valid_parent], np.float32(1.0e-30)
    )
    output: dict[str, Any] = {}
    for depth in np.unique(depths):
        mask = depths == depth
        output[str(int(depth))] = {
            "node_count": int(np.count_nonzero(mask)),
            "interior_count": int(np.count_nonzero(nodes[hierarchy_start:, CHILD_COUNT][mask] > 0)),
            "six_sigma_max_axis_extent": _summary(extents[mask]),
            "parent_to_child_extent_ratio": _summary(ratio[mask]),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = _json(args.oracle)
    oracle_root = args.oracle.resolve().parent
    asset_path = Path(manifest["asset"]["path"])
    if _sha256(asset_path) != manifest["asset"]["sha256"]:
        raise RuntimeError("asset hash differs from the frozen oracle")
    hspt_record_path = oracle_root / manifest["hspt"]["path"]
    hspt_record = _json(hspt_record_path)
    frozen = hspt_record["configuration"]
    config = ReferenceConfig(
        root_global_node_id=int(manifest["asset"]["root_global_node_id"]),
        skybox_count=int(manifest["asset"]["skybox_rows"]),
        spt_root_volume=float(frozen["SPT_root_volume"]),
        min_spt_size=int(frozen["min_SPT_size_effective_evaluation_default"]),
        target_granularity=float(frozen["resolved_target_granularity"]),
        distance_multiplier=1.0,
    )

    asset = load_dhier(asset_path)
    hspt = build_hspt(HierarchyData.from_dhier(asset), config)
    hierarchy_start = config.skybox_count
    hierarchy_rows = asset.layout.gaussian_count - hierarchy_start
    hierarchy_depths = np.asarray(asset.nodes[hierarchy_start:, 0], dtype=np.int16)

    ever_selected = np.zeros(hierarchy_rows, dtype=np.bool_)
    always_selected = np.ones(hierarchy_rows, dtype=np.bool_)
    camera_centers: list[np.ndarray] = []
    selection_counts: list[int] = []
    for entry in manifest["cameras"]:
        camera_path = oracle_root / entry["paths"]["camera.json"]
        camera_record = _json(camera_path)
        camera_centers.append(np.asarray(
            camera_record["effective_official_runtime_camera"]["camera_center_float32"],
            dtype=np.float32,
        ))
        cut_path = camera_path.parent / "spt-cut.json"
        cut = _json(cut_path)
        ids = _described_array(
            cut_path.parent, cut["arrays"]["selected_hierarchy_global_node_ids"]
        ).astype(np.int64, copy=False)
        present = np.zeros(hierarchy_rows, dtype=np.bool_)
        valid = ids[(ids >= hierarchy_start) & (ids < asset.layout.gaussian_count)]
        present[valid - hierarchy_start] = True
        ever_selected |= present
        always_selected &= present
        selection_counts.append(int(ids.size))

    centers = np.stack(camera_centers)
    spt_roots = np.asarray(asset.positions[hspt.spt_root_global_node_ids], dtype=np.float32)
    flattened_depths = np.asarray(asset.nodes[hspt.spt_to_global_node_id, 0], dtype=np.int16)
    spt_ids = np.repeat(np.arange(hspt.spt_count, dtype=np.int32), np.diff(hspt.spt_starts))
    distances = np.empty((centers.shape[0], hspt.spt_count), dtype=np.float32)
    for camera_index, center in enumerate(centers):
        delta = np.asarray(spt_roots - center, dtype=np.float32)
        distances[camera_index] = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float32))

    per_depth: dict[str, Any] = {}
    for depth in np.unique(flattened_depths):
        mask = flattened_depths == depth
        d = distances[:, spt_ids[mask]]
        minimum = hspt.spt_min[mask][None, :]
        maximum = hspt.spt_max[mask][None, :]
        min_norm = (d - minimum) / np.maximum(
            np.maximum(np.abs(d), np.abs(minimum)), np.float32(1.0e-12)
        )
        max_norm = (maximum - d) / np.maximum(
            np.maximum(np.abs(d), np.abs(maximum)), np.float32(1.0e-12)
        )
        boundary = np.logical_or(np.abs(min_norm) <= 0.05, np.abs(max_norm) <= 0.05)
        per_depth[str(int(depth))] = {
            "minimum_predicate_normalized_margin": _summary(min_norm.reshape(-1)),
            "maximum_prefix_normalized_margin": _summary(max_norm.reshape(-1)),
            "near_either_boundary_abs_margin_le_0_05_count": int(np.count_nonzero(boundary)),
            "candidate_decision_count": int(boundary.size),
            "near_either_boundary_fraction": float(np.mean(boundary)),
        }

    changing = ever_selected & ~always_selected
    output = {
        "schema_version": 1,
        "diagnostic": "natural hierarchy structure and unchanged-selector decision margins",
        "label": args.label,
        "scope": "official oracle cameras only; hierarchy selection excludes the always-on skybox prefix",
        "oracle_manifest": str(args.oracle.resolve()),
        "oracle_manifest_sha256": _sha256(args.oracle),
        "asset": {"path": str(asset_path), "sha256": _sha256(asset_path)},
        "configuration": {
            "spt_root_volume": config.spt_root_volume,
            "min_spt_size": config.min_spt_size,
            "target_granularity": config.target_granularity,
            "distance_multiplier": config.distance_multiplier,
        },
        "counts": {
            "camera_count": int(centers.shape[0]),
            "hierarchy_rows": int(hierarchy_rows),
            "spt_count": int(hspt.spt_count),
            "flattened_spt_rows": int(hspt.flattened_spt_row_count),
            "candidate_spt_row_decisions": int(centers.shape[0] * hspt.flattened_spt_row_count),
        },
        "selected_count": _summary(np.asarray(selection_counts, dtype=np.float64)),
        "decision_changes": {
            "hierarchy_nodes_changing_selected_state": int(np.count_nonzero(changing)),
            "hierarchy_nodes_ever_selected": int(np.count_nonzero(ever_selected)),
            "fraction_all_hierarchy_nodes": float(np.mean(changing)),
            "fraction_ever_selected_nodes": float(np.count_nonzero(changing) / max(1, np.count_nonzero(ever_selected))),
        },
        "hierarchy_structure_by_depth": _structure(asset, hierarchy_start),
        "decision_margin_by_depth": per_depth,
        "status": "PASS-NATURAL-SELECTOR-MARGIN-INSTRUMENTATION",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(output["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
