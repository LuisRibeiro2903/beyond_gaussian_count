#!/usr/bin/env python3
"""Instrument the unchanged ALoD predicate for the synthetic control.

The compact NPZ stores static HSPT thresholds and a bit-packed camera-by-node
decision matrix.  Together with the per-camera SPT-root distances it is
sufficient to reconstruct both source predicates for every candidate row:
``spt_max > distance`` (prefix eligibility) and ``spt_min < distance``
(selection).  JSON contains aggregate distributions only.
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
from alod_reference import HierarchyData, build_hspt  # noqa: E402
from scripts.alod.synthetic_strong_lod_control import (  # noqa: E402
    EXPERIMENT_ID,
    _reference_cameras,
    reference_config,
    sha256_file,
)


def summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(finite.max()),
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()

    asset = load_dhier(args.asset)
    hierarchy = HierarchyData.from_dhier(asset)
    config = reference_config()
    hspt = build_hspt(hierarchy, config)
    cameras = _reference_cameras(args.cameras)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    hierarchy_start = config.skybox_count
    hierarchy_rows = asset.layout.gaussian_count - hierarchy_start
    depths = np.asarray(asset.nodes[hierarchy_start:, 0], dtype=np.int16)

    selected = np.zeros((len(cameras), hierarchy_rows), dtype=np.bool_)
    for index, record in enumerate(selection["cameras"]):
        ids = np.load(record["selected_ids_path"], allow_pickle=False).astype(np.int64)
        selected[index, ids - hierarchy_start] = True

    spt_root_positions = np.asarray(
        asset.positions[hspt.spt_root_global_node_ids], dtype=np.float32
    )
    root_distances = np.empty((len(cameras), hspt.spt_count), dtype=np.float32)
    for index, camera in enumerate(cameras):
        delta = spt_root_positions - camera.camera_center
        root_distances[index] = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float32))

    flattened_depths = np.asarray(asset.nodes[hspt.spt_to_global_node_id, 0], dtype=np.int16)
    spt_ids = np.repeat(
        np.arange(hspt.spt_count, dtype=np.int32), np.diff(hspt.spt_starts)
    )
    minimum_margin_summaries: dict[str, list[np.ndarray]] = {}
    maximum_margin_summaries: dict[str, list[np.ndarray]] = {}
    boundary_counts: dict[str, list[int]] = {}
    total_counts: dict[str, list[int]] = {}
    for depth in np.unique(flattened_depths):
        mask = flattened_depths == depth
        d = root_distances[:, spt_ids[mask]]
        minimum = hspt.spt_min[mask][None, :]
        maximum = hspt.spt_max[mask][None, :]
        min_margin = d - minimum
        max_margin = maximum - d
        min_norm = min_margin / np.maximum(
            np.maximum(np.abs(d), np.abs(minimum)), np.float32(1.0e-12)
        )
        max_norm = max_margin / np.maximum(
            np.maximum(np.abs(d), np.abs(maximum)), np.float32(1.0e-12)
        )
        key = str(int(depth))
        minimum_margin_summaries[key] = min_norm.reshape(-1)
        maximum_margin_summaries[key] = max_norm.reshape(-1)
        boundary = np.logical_or(np.abs(min_norm) <= 0.05, np.abs(max_norm) <= 0.05)
        boundary_counts[key] = [int(np.count_nonzero(boundary))]
        total_counts[key] = [int(boundary.size)]

    per_depth = {}
    for key in minimum_margin_summaries:
        near_count = boundary_counts[key][0]
        total = total_counts[key][0]
        per_depth[key] = {
            "minimum_predicate_normalized_margin": summary(minimum_margin_summaries[key]),
            "maximum_prefix_normalized_margin": summary(maximum_margin_summaries[key]),
            "near_either_boundary_abs_margin_le_0_05_count": near_count,
            "candidate_decision_count": total,
            "near_either_boundary_fraction": near_count / total,
        }

    changes = np.any(selected != selected[:1], axis=0)
    ever_selected = np.any(selected, axis=0)
    packed = np.packbits(selected, axis=0, bitorder="little")
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        schema_version=np.asarray([1], dtype=np.int32),
        hierarchy_global_ids=np.arange(
            hierarchy_start, asset.layout.gaussian_count, dtype=np.int32
        ),
        hierarchy_depths=depths,
        selected_by_camera_packbits=packed,
        selected_camera_count=np.asarray([len(cameras)], dtype=np.int32),
        spt_to_global_node_id=hspt.spt_to_global_node_id.astype(np.int32),
        spt_row_depth=flattened_depths,
        spt_id_per_flattened_row=spt_ids,
        spt_min=hspt.spt_min.astype(np.float32),
        spt_max=hspt.spt_max.astype(np.float32),
        spt_starts=hspt.spt_starts.astype(np.int32),
        spt_root_global_node_ids=hspt.spt_root_global_node_ids.astype(np.int32),
        camera_spt_root_distances=root_distances,
        camera_centers=np.stack([camera.camera_center for camera in cameras]),
    )
    output = {
        "experiment_id": EXPERIMENT_ID,
        "asset_sha256": sha256_file(args.asset),
        "camera_path_sha256": sha256_file(args.cameras),
        "selection_sha256": sha256_file(args.selection),
        "reconstruction_payload": {
            "path": str(args.output_npz.resolve()),
            "sha256": sha256_file(args.output_npz),
            "selected_decision_encoding": (
                "np.unpackbits(selected_by_camera_packbits, axis=0, bitorder='little')"
                "[:selected_camera_count] aligned to hierarchy_global_ids"
            ),
            "spt_predicates": {
                "prefix_eligible": "spt_max > camera_spt_root_distances[camera,spt_id]",
                "selected": "prefix_eligible and spt_min < camera_spt_root_distances[camera,spt_id]",
            },
        },
        "counts": {
            "camera_count": len(cameras),
            "hierarchy_rows": hierarchy_rows,
            "spt_count": hspt.spt_count,
            "flattened_spt_rows": hspt.flattened_spt_row_count,
            "candidate_spt_row_decisions": int(len(cameras) * hspt.flattened_spt_row_count),
        },
        "decision_changes": {
            "hierarchy_nodes_changing_selected_state": int(np.count_nonzero(changes)),
            "hierarchy_nodes_ever_selected": int(np.count_nonzero(ever_selected)),
            "fraction_all_hierarchy_nodes": float(np.mean(changes)),
            "fraction_ever_selected_nodes": float(
                np.count_nonzero(changes) / max(1, np.count_nonzero(ever_selected))
            ),
        },
        "decision_margin_by_depth": per_depth,
        "status": "PASS-SYNTH-SELECTOR-MARGIN-INSTRUMENTATION",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(canonical_json(output), encoding="utf-8")
    print(output["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
