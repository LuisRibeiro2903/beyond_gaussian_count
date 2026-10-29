#!/usr/bin/env python3
"""Export the deterministic post-filter trace and replay residency policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

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


EXPECTED_ASSET_SHA256 = (
    "6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c"
)
DEFAULT_CAPACITIES = (64_000, 96_000, 128_000, 160_000, 525_000)
RECOMMENDED_CAPACITY = 160_000


def _trace_sha256(requests: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for ids in requests:
        digest.update(int(ids.size).to_bytes(8, "little"))
        digest.update(np.ascontiguousarray(ids, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _property_bytes_per_row(asset) -> int:
    color_values = int(np.prod(asset.sh_coefficients.shape[1:]))
    return int((3 + 4 + 3 + 1 + color_values) * np.dtype("float32").itemsize)


@torch.no_grad()
def analyze(
    asset_path: Path,
    *,
    repetitions: int = 10,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    asset_path = asset_path.resolve()
    asset_sha256 = sha256_file(asset_path)
    if asset_sha256 != EXPECTED_ASSET_SHA256:
        raise RuntimeError(f"unexpected hierarchy SHA-256 {asset_sha256}")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    config = load_config("ALOD-HIER-LOD-FRUSTUM")
    runtime = _runtime_config(config)
    cameras = apply_camera_config(
        load_camera_path(
            resolve_camera_path(config), config.get("trajectory_id")
        ),
        config,
    )
    asset = load_dhier(asset_path)
    hierarchy = AlodHierarchySource(
        asset.positions, asset.log_scales, asset.nodes
    )
    hspt = build_production_hspt(hierarchy, runtime)
    compact = AlodCompactCullingMetadata.from_log_scales(
        asset.positions, asset.log_scales
    )

    dynamic_requests: list[np.ndarray] = []
    total_requests: list[np.ndarray] = []
    semantic_hashes: list[str] = []
    camera_ids: list[str] = []
    for camera_spec in cameras.cameras:
        camera = make_frame_camera(camera_spec, torch.device("cpu"))
        settings = make_render_settings(
            config, camera_spec, sh_degree=int(config["sh_degree"])
        )
        selection = select_production(
            hierarchy,
            hspt,
            selection_camera_from_frame(
                camera, settings, stable_id=camera_spec.camera_id
            ),
            runtime,
        )
        materialization = conservative_pinhole_materialization(
            compact,
            selection.raster_global_node_ids,
            camera,
            settings,
        )
        total_ids = materialization.materialization_global_node_ids
        dynamic_ids = np.ascontiguousarray(
            total_ids[total_ids >= runtime.skybox_count], dtype=np.int64
        )
        total_requests.append(total_ids)
        dynamic_requests.append(dynamic_ids)
        semantic_hashes.append(
            hashlib.sha256(
                selection.raster_global_node_ids.tobytes()
            ).hexdigest()
        )
        camera_ids.append(camera_spec.camera_id)

    trace_rows = characterize_request_trace(dynamic_requests)
    maximum_request = max(ids.size for ids in dynamic_requests)
    union_rows = trace_rows[-1].cumulative_union_rows if trace_rows else 0
    capacities = sorted(
        set((int(maximum_request), int(union_rows), *DEFAULT_CAPACITIES))
    )
    property_bytes = _property_bytes_per_row(asset)
    replay_requests = dynamic_requests * repetitions
    simulations = [
        simulate_bounded_cache(
            replay_requests,
            capacity_rows=capacity,
            property_bytes_per_row=property_bytes,
            global_row_count=asset.layout.gaussian_count,
        ).to_dict()
        for capacity in capacities
        if capacity >= maximum_request
    ]
    previous = simulate_previous_working_set(
        replay_requests, property_bytes_per_row=property_bytes
    )
    recommended = next(
        row for row in simulations if row["capacity_rows"] == RECOMMENDED_CAPACITY
    )
    historical = next(
        row for row in simulations if row["capacity_rows"] == 525_000
    )
    if recommended["total_h2d_rows"] != historical["total_h2d_rows"]:
        raise RuntimeError("recommended cache does not preserve historical trace loads")
    if recommended["evictions"] != 0:
        raise RuntimeError("recommended cache unexpectedly evicts on the frozen trace")

    rows = []
    for camera_index, (trace, total_ids, semantic_hash) in enumerate(
        zip(trace_rows, total_requests, semantic_hashes)
    ):
        row = trace.to_dict()
        row.update(
            {
                "camera_index": camera_index,
                "camera_id": camera_ids[camera_index],
                "semantic_selected_global_node_ids_sha256": semantic_hash,
                "total_materialization_rows_including_filtered_skybox": int(
                    total_ids.size
                ),
                "filtered_skybox_rows": int(
                    total_ids.size - trace.requested_rows
                ),
            }
        )
        rows.append(row)

    arrays = {
        f"frame_{index:03d}_dynamic_global_node_ids": ids
        for index, ids in enumerate(dynamic_requests)
    }
    return (
        {
            "schema_version": 1,
            "status": "PASS-ALOD-CACHE-TRACE",
            "asset": {"path": str(asset_path), "sha256": asset_sha256},
            "trajectory_id": config["trajectory_id"],
            "trajectory_frames": len(dynamic_requests),
            "formal_repetitions": repetitions,
            "formal_replay_frames": len(replay_requests),
            "property_bytes_per_row": property_bytes,
            "maximum_frame_request_rows": int(maximum_request),
            "absolute_minimum_feasible_capacity_rows": int(maximum_request),
            "trajectory_union_rows": int(union_rows),
            "ordered_trace_sha256": _trace_sha256(dynamic_requests),
            "frames": rows,
            "bounded_cache_simulations": simulations,
            "previous_working_set_simulation": previous.to_dict(),
            "recommended_capacity": {
                "configuration_id": "ALOD-HOST-BOUNDED-CACHE-REDUCED",
                "capacity_rows": RECOMMENDED_CAPACITY,
                "capacity_bytes": RECOMMENDED_CAPACITY * property_bytes,
                "headroom_above_union_rows": RECOMMENDED_CAPACITY - union_rows,
                "headroom_above_union_fraction": (
                    (RECOMMENDED_CAPACITY - union_rows) / union_rows
                    if union_rows
                    else 0.0
                ),
                "reason": (
                    "smallest evaluated round capacity with the same zero-eviction "
                    "and transfer behavior as 525k, plus measured-union headroom"
                ),
            },
            "semantic_selector_unchanged": True,
            "historical_525k_configuration_unchanged": True,
        },
        arrays,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-npz", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    result, arrays = analyze(args.asset, repetitions=args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.trace_npz, **arrays)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "trace_npz": str(args.trace_npz.resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
