#!/usr/bin/env python3
"""Freeze Run009 ALOD-EXT working-set statistics before timed benchmarking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_dhier import load_dhier  # noqa: E402
from benchmarks.alod_resident_benchmark import _runtime_config  # noqa: E402
from benchmarks.benchmark_real_scene import (  # noqa: E402
    apply_camera_config,
    load_camera_path,
    load_config,
    make_frame_camera,
    make_render_settings,
    resolve_camera_path,
    sha256_file,
)
from scene_runtime.alod import (  # noqa: E402
    AlodCompactCullingMetadata,
    AlodHierarchySource,
    build_production_hspt,
    characterize_request_trace,
    conservative_pinhole_materialization,
    select_production,
    selection_camera_from_frame,
    simulate_bounded_cache,
    simulate_previous_working_set,
)


EXPECTED_ASSET_SHA256 = "38f4abdc0d47f2d20316de33f4fff87086e038a57daac0f2feaf98fa9178eb7b"
STATUS = "PASS-ALOD-EXT-RUN009-CACHE-SIZING"


def _round_up(value: int, quantum: int = 50_000) -> int:
    return int(math.ceil(value / quantum) * quantum)


def _trace_hash(requests: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for ids in requests:
        digest.update(int(ids.size).to_bytes(8, "little"))
        digest.update(np.ascontiguousarray(ids, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _distribution(values: list[int] | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def _overlaps(requests: list[np.ndarray]) -> list[float]:
    values = []
    for previous, current in zip(requests, requests[1:]):
        intersection = np.intersect1d(previous, current, assume_unique=True).size
        values.append(intersection / current.size if current.size else 1.0)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-npz", type=Path, required=True)
    args = parser.parse_args()

    asset_path = args.asset.resolve()
    asset_hash = sha256_file(asset_path)
    if asset_hash != EXPECTED_ASSET_SHA256:
        raise RuntimeError(f"unexpected Run009 hierarchy SHA-256 {asset_hash}")

    config = load_config("ALOD-REPRO-HOST-NOCACHE")
    runtime = _runtime_config(config)
    cameras = apply_camera_config(
        load_camera_path(resolve_camera_path(config), config["trajectory_id"]), config
    )
    if len(cameras.cameras) != 71:
        raise RuntimeError("formal Run009 workload must contain exactly 71 cameras")

    asset = load_dhier(asset_path)
    hierarchy = AlodHierarchySource(asset.positions, asset.log_scales, asset.nodes)
    hspt = build_production_hspt(hierarchy, runtime)
    compact = AlodCompactCullingMetadata.from_log_scales(
        asset.positions, asset.log_scales
    )

    semantic: list[np.ndarray] = []
    materialized: list[np.ndarray] = []
    frame_rows = []
    for index, spec in enumerate(cameras.cameras):
        camera = make_frame_camera(spec, torch.device("cpu"))
        settings = make_render_settings(config, spec, sh_degree=int(config["sh_degree"]))
        selection = select_production(
            hierarchy,
            hspt,
            selection_camera_from_frame(camera, settings, stable_id=spec.camera_id),
            runtime,
        )
        filtered = conservative_pinhole_materialization(
            compact, selection.raster_global_node_ids, camera, settings
        )
        semantic_ids = np.ascontiguousarray(
            selection.selected_hierarchy_global_node_ids, dtype=np.int64
        )
        materialized_ids = np.ascontiguousarray(
            filtered.materialization_global_node_ids[
                filtered.materialization_global_node_ids >= runtime.skybox_count
            ],
            dtype=np.int64,
        )
        semantic.append(semantic_ids)
        materialized.append(materialized_ids)
        frame_rows.append(
            {
                "camera_index": index,
                "camera_id": spec.camera_id,
                "semantic_hierarchy_rows": int(semantic_ids.size),
                "semantic_skybox_rows": runtime.skybox_count,
                "prematerialization_candidate_rows": int(
                    selection.raster_global_node_ids.size
                ),
                "materialized_hierarchy_rows": int(materialized_ids.size),
                "materialized_total_rows": int(filtered.materialization_rows),
                "filtered_skybox_rows": int(
                    filtered.materialization_rows - materialized_ids.size
                ),
                "cpu_filter_ms": filtered.filter_seconds * 1000.0,
            }
        )

    semantic_union = len(set().union(*(set(ids.tolist()) for ids in semantic)))
    materialized_union = len(set().union(*(set(ids.tolist()) for ids in materialized)))
    maximum = max(ids.size for ids in materialized)
    minimum_capacity = int(maximum)
    practical_capacity = _round_up(max(maximum, math.ceil(materialized_union * 1.05)))
    sensitivity_capacity = _round_up(max(practical_capacity + 50_000, materialized_union * 1.5))
    capacities = (minimum_capacity, practical_capacity, sensitivity_capacity)
    property_bytes_per_row = int(
        (3 + 4 + 3 + 1 + int(np.prod(asset.sh_coefficients.shape[1:]))) * 4
    )
    simulations = [
        simulate_bounded_cache(
            materialized,
            capacity_rows=capacity,
            property_bytes_per_row=property_bytes_per_row,
            global_row_count=asset.layout.gaussian_count,
        ).to_dict()
        for capacity in capacities
    ]

    result = {
        "schema_version": 1,
        "status": STATUS,
        "asset": {"path": str(asset_path), "sha256": asset_hash},
        "configuration_source": "ALOD-REPRO-HOST-NOCACHE",
        "trajectory_id": config["trajectory_id"],
        "camera_path": str(resolve_camera_path(config)),
        "camera_count": len(cameras.cameras),
        "camera_order_sha256": hashlib.sha256(
            "\n".join(spec.camera_id for spec in cameras.cameras).encode()
        ).hexdigest(),
        "property_bytes_per_row": property_bytes_per_row,
        "semantic_working_set": {
            "rows": _distribution([ids.size for ids in semantic]),
            "trajectory_union_rows": semantic_union,
            "consecutive_required_row_overlap": _distribution(_overlaps(semantic)),
            "ordered_trace_sha256": _trace_hash(semantic),
        },
        "post_filter_working_set": {
            "rows": _distribution([ids.size for ids in materialized]),
            "trajectory_union_rows": materialized_union,
            "consecutive_required_row_overlap": _distribution(_overlaps(materialized)),
            "ordered_trace_sha256": _trace_hash(materialized),
        },
        "frozen_capacities": {
            "minimum_feasible_rows": minimum_capacity,
            "practical_bounded_rows": practical_capacity,
            "larger_sensitivity_rows": sensitivity_capacity,
            "policy": (
                "minimum=max simultaneous post-filter hierarchy rows; practical="
                "ceil(max(maximum, 1.05*union), 50k); sensitivity="
                "ceil(max(practical+50k, 1.5*union), 50k)"
            ),
            "chosen_without_timed_fps": True,
        },
        "bounded_cache_one_pass_simulations": simulations,
        "previous_working_set_one_pass_simulation": simulate_previous_working_set(
            materialized, property_bytes_per_row=property_bytes_per_row
        ).to_dict(),
        "frames": frame_rows,
        "semantic_selection_unchanged": True,
        "materialization_predicate_unchanged": True,
        "weak_lod_context": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.trace_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    arrays = {}
    for index, ids in enumerate(semantic):
        arrays[f"frame_{index:03d}_semantic_hierarchy_ids"] = ids
    for index, ids in enumerate(materialized):
        arrays[f"frame_{index:03d}_materialized_hierarchy_ids"] = ids
    np.savez_compressed(args.trace_npz, **arrays)
    print(json.dumps({"status": STATUS, "capacities": result["frozen_capacities"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
