#!/usr/bin/env python3
"""Read-only stage and Python profile of unchanged production ALoD selection."""

from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import sys
import time
import tracemalloc
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
from scene_runtime.alod.camera import selection_camera_from_frame  # noqa: E402
from scene_runtime.alod.frustum import (  # noqa: E402
    extract_four_planes,
    scale_bounds,
    visible_spheres,
)
from scene_runtime.alod.hspt import build_production_hspt  # noqa: E402
from scene_runtime.alod.selection import (  # noqa: E402
    CHILD_COUNT,
    FIRST_CHILD,
    GLOBAL_ID,
    NEXT_SIBLING,
    _distances,
    _interval_size,
    _squared_distances,
    select_production,
)
from scene_runtime.alod.types import (  # noqa: E402
    AlodHierarchySource,
    AlodSelection,
    AlodSelectionTimings,
    AlodSptCut,
    AlodUpperSelection,
)


def _timed_upper(hspt, camera, config):
    timings: dict[str, float] = {}
    start = time.perf_counter()
    planes = extract_four_planes(camera.full_projection_transform)
    timings["frustum_plane_construction"] = time.perf_counter() - start
    start = time.perf_counter()
    bounds = scale_bounds(hspt.upper_tree_log_scaling)
    timings["upper_bound_preparation"] = time.perf_counter() - start
    stack = np.asarray([0], dtype=np.int32)
    cut_parts: list[np.ndarray] = []
    rejected_parts: list[np.ndarray] = []
    nodes = hspt.upper_tree_nodes
    loop = {
        "upper_frustum_tests": 0.0,
        "upper_node_classification": 0.0,
        "upper_distance_threshold": 0.0,
        "upper_stack_concatenation": 0.0,
    }
    while stack.size:
        if config.frustum_enabled:
            start = time.perf_counter()
            visible = visible_spheres(hspt.upper_tree_xyz[stack], bounds[stack], planes)
            if np.any(~visible):
                rejected_parts.append(stack[~visible].copy())
            stack = stack[visible]
            loop["upper_frustum_tests"] += time.perf_counter() - start
            if stack.size == 0:
                break
        start = time.perf_counter()
        leaves = nodes[stack, CHILD_COUNT] == 0
        if np.any(leaves):
            cut_parts.append(stack[leaves].copy())
        interior = stack[~leaves]
        loop["upper_node_classification"] += time.perf_counter() - start
        if interior.size == 0:
            break
        start = time.perf_counter()
        distance_squared = _squared_distances(
            hspt.upper_tree_xyz[interior], camera.camera_center
        )
        rhs = np.asarray(
            distance_squared * np.float32(config.distance_multiplier), dtype=np.float32
        )
        descend = hspt.upper_min_distance_squared[interior] > rhs
        if np.any(~descend):
            cut_parts.append(interior[~descend].copy())
        interior = interior[descend]
        loop["upper_distance_threshold"] += time.perf_counter() - start
        start = time.perf_counter()
        first = nodes[interior, FIRST_CHILD]
        second = nodes[first, NEXT_SIBLING]
        stack = np.concatenate((first, second)).astype(np.int32, copy=False)
        loop["upper_stack_concatenation"] += time.perf_counter() - start
    timings.update(loop)

    start = time.perf_counter()
    coarse_local = (
        np.concatenate(cut_parts).astype(np.int32, copy=False)
        if cut_parts
        else np.empty(0, dtype=np.int32)
    )
    rejected_local = (
        np.concatenate(rejected_parts).astype(np.int32, copy=False)
        if rejected_parts
        else np.empty(0, dtype=np.int32)
    )
    leaf_local = coarse_local[nodes[coarse_local, CHILD_COUNT] == 0]
    timings["upper_cut_assembly"] = time.perf_counter() - start
    start = time.perf_counter()
    candidate_local = leaf_local[nodes[leaf_local, FIRST_CHILD] >= 0]
    candidate_spt_ids = nodes[candidate_local, FIRST_CHILD].astype(np.int32, copy=True)
    candidate_distances = np.asarray(
        _distances(hspt.upper_tree_xyz[candidate_local], camera.camera_center)
        * np.float32(config.distance_multiplier),
        dtype=np.float32,
    )
    timings["candidate_spt_construction"] = time.perf_counter() - start
    start = time.perf_counter()
    upper_render = nodes[coarse_local][
        nodes[coarse_local, FIRST_CHILD] <= 0, GLOBAL_ID
    ].astype(np.int32, copy=True)
    result = AlodUpperSelection(
        coarse_cut_upper_local_ids=np.ascontiguousarray(coarse_local, dtype=np.int32),
        coarse_cut_global_node_ids=np.ascontiguousarray(
            nodes[coarse_local, GLOBAL_ID], dtype=np.int32
        ),
        cut_leaf_upper_local_ids=np.ascontiguousarray(leaf_local, dtype=np.int32),
        cut_leaf_global_node_ids=np.ascontiguousarray(
            nodes[leaf_local, GLOBAL_ID], dtype=np.int32
        ),
        upper_render_global_node_ids=np.ascontiguousarray(upper_render, dtype=np.int32),
        candidate_spt_ids=np.ascontiguousarray(candidate_spt_ids, dtype=np.int32),
        candidate_spt_upper_local_ids=np.ascontiguousarray(candidate_local, dtype=np.int32),
        candidate_spt_root_global_node_ids=np.ascontiguousarray(
            nodes[candidate_local, GLOBAL_ID], dtype=np.int32
        ),
        candidate_spt_distances=np.ascontiguousarray(candidate_distances, dtype=np.float32),
        frustum_planes=np.ascontiguousarray(planes, dtype=np.float32),
        frustum_rejected_upper_local_ids=np.ascontiguousarray(
            rejected_local, dtype=np.int32
        ),
    )
    timings["upper_output_array_assembly"] = time.perf_counter() - start
    return result, timings


def _timed_cut(hspt, upper):
    timings = {
        "spt_interval_search": 0.0,
        "spt_index_and_predicate": 0.0,
        "spt_python_result_assembly": 0.0,
        "spt_combined_concatenation": 0.0,
    }
    results: list[AlodSptCut] = []
    combined: list[np.ndarray] = []
    for candidate_index, (raw_spt_id, raw_distance) in enumerate(
        zip(upper.candidate_spt_ids, upper.candidate_spt_distances)
    ):
        spt_id = int(raw_spt_id)
        distance = np.float32(raw_distance)
        start = time.perf_counter()
        interval_size = _interval_size(hspt, spt_id, distance)
        timings["spt_interval_search"] += time.perf_counter() - start
        start = time.perf_counter()
        spt_start = int(hspt.spt_starts[spt_id])
        first_offset = 0 if candidate_index == 0 else 1
        flat_indices = spt_start + np.arange(
            first_offset, interval_size + 1, dtype=np.int64
        )
        emitted = hspt.spt_to_global_node_id[flat_indices][
            hspt.spt_min[flat_indices] < distance
        ]
        emitted = np.ascontiguousarray(emitted[emitted != 0], dtype=np.int64)
        timings["spt_index_and_predicate"] += time.perf_counter() - start
        start = time.perf_counter()
        results.append(
            AlodSptCut(
                spt_id=spt_id,
                root_global_node_id=int(hspt.spt_root_global_node_ids[spt_id]),
                distance=distance,
                interval_size=interval_size,
                emitted_global_node_ids=emitted,
            )
        )
        combined.append(emitted)
        timings["spt_python_result_assembly"] += time.perf_counter() - start
    start = time.perf_counter()
    selected = (
        np.concatenate(combined).astype(np.int64, copy=False)
        if combined
        else np.empty(0, dtype=np.int64)
    )
    selected = np.ascontiguousarray(selected, dtype=np.int64)
    timings["spt_combined_concatenation"] = time.perf_counter() - start
    return tuple(results), selected, timings


def _timed_replica(hierarchy, hspt, camera, config):
    total_start = time.perf_counter()
    upper, timings = _timed_upper(hspt, camera, config)
    per_spt, selected_spt, cut_timings = _timed_cut(hspt, upper)
    timings.update(cut_timings)
    start = time.perf_counter()
    selected_hierarchy = np.ascontiguousarray(
        np.concatenate(
            (upper.upper_render_global_node_ids.astype(np.int64), selected_spt)
        ),
        dtype=np.int64,
    )
    raster = np.ascontiguousarray(
        np.concatenate(
            (np.arange(config.skybox_count, dtype=np.int64), selected_hierarchy)
        ),
        dtype=np.int64,
    )
    timings["final_global_id_assembly"] = time.perf_counter() - start
    timings["instrumented_total"] = time.perf_counter() - total_start
    result = AlodSelection(
        camera_stable_id=camera.stable_id,
        mode="lod",
        frustum_enabled=config.frustum_enabled,
        upper=upper,
        per_spt=per_spt,
        selected_spt_global_node_ids=selected_spt,
        selected_hierarchy_global_node_ids=selected_hierarchy,
        raster_global_node_ids=raster,
        timings=AlodSelectionTimings(),
    )
    return result, timings


def _compare(actual, expected):
    np.testing.assert_array_equal(
        actual.selected_hierarchy_global_node_ids,
        expected.selected_hierarchy_global_node_ids,
    )
    np.testing.assert_array_equal(actual.raster_global_node_ids, expected.raster_global_node_ids)
    for name in actual.upper.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(actual.upper, name), getattr(expected.upper, name))


def _stats(values):
    array = np.asarray(values, dtype=np.float64) * 1000.0
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def _profile_rows(profile: cProfile.Profile) -> list[dict[str, Any]]:
    stats = pstats.Stats(profile)
    rows = []
    for (filename, line, function), (cc, nc, tt, ct, _callers) in stats.stats.items():
        if "scene_runtime/alod" not in filename and function not in {
            "concatenate",
            "ascontiguousarray",
            "arange",
            "any",
        }:
            continue
        rows.append(
            {
                "file": filename,
                "line": line,
                "function": function,
                "primitive_calls": cc,
                "total_calls": nc,
                "total_seconds": tt,
                "cumulative_seconds": ct,
            }
        )
    rows.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
    return rows[:40]


def profile_selection(asset: Path, repeats: int) -> dict[str, Any]:
    config_dict = load_config("ALOD-HIER-LOD-FRUSTUM")
    config = _runtime_config(config_dict)
    asset_data = load_dhier(asset)
    hierarchy = AlodHierarchySource(
        asset_data.positions, asset_data.log_scales, asset_data.nodes
    )
    hspt = build_production_hspt(hierarchy, config)
    camera_path = apply_camera_config(
        load_camera_path(
            resolve_camera_path(config_dict), config_dict.get("trajectory_id")
        ),
        config_dict,
    )
    prepared = []
    camera_prep_samples = []
    for camera_spec in camera_path.cameras:
        frame = make_frame_camera(camera_spec, torch.device("cpu"))
        settings = make_render_settings(config_dict, camera_spec, sh_degree=1)
        start = time.perf_counter()
        camera = selection_camera_from_frame(
            frame, settings, stable_id=camera_spec.camera_id
        )
        camera_prep_samples.append(time.perf_counter() - start)
        prepared.append(camera)

    for camera in prepared:
        select_production(hierarchy, hspt, camera, config)

    stage_samples: dict[str, list[float]] = {
        "camera_preparation": camera_prep_samples * repeats
    }
    production_samples = []
    for _ in range(repeats):
        for camera in prepared:
            start = time.perf_counter()
            production = select_production(hierarchy, hspt, camera, config)
            production_samples.append(time.perf_counter() - start)
            replica, timings = _timed_replica(hierarchy, hspt, camera, config)
            _compare(replica, production)
            for name, value in timings.items():
                stage_samples.setdefault(name, []).append(value)
    stage_samples["production_select_production_wall"] = production_samples

    tracemalloc.start()
    for camera in prepared:
        select_production(hierarchy, hspt, camera, config)
    _current, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    profile = cProfile.Profile()
    profile.enable()
    for _ in range(repeats):
        for camera in prepared:
            select_production(hierarchy, hspt, camera, config)
    profile.disable()

    example = select_production(hierarchy, hspt, prepared[0], config)
    result_array_bytes = int(
        example.selected_spt_global_node_ids.nbytes
        + example.selected_hierarchy_global_node_ids.nbytes
        + example.raster_global_node_ids.nbytes
        + sum(cut.emitted_global_node_ids.nbytes for cut in example.per_spt)
        + sum(
            getattr(example.upper, name).nbytes
            for name in example.upper.__dataclass_fields__
        )
    )
    return {
        "schema_version": 1,
        "status": "PASS-ALOD-CPU-SELECTION-PROFILE",
        "asset": {"path": str(asset.resolve()), "sha256": sha256_file(asset)},
        "camera_count": len(prepared),
        "repeats": repeats,
        "profiled_calls": len(prepared) * repeats,
        "production_selector_modified": False,
        "replica_output_parity_exact": True,
        "timing_note": (
            "instrumented replica stages include perf_counter overhead; production wall "
            "timing is reported separately"
        ),
        "stage_timings": {
            name: _stats(values) for name, values in stage_samples.items()
        },
        "allocation_profile": {
            "tracemalloc_peak_bytes": int(peak_tracemalloc),
            "example_returned_numpy_array_bytes_with_per_spt_duplication": result_array_bytes,
            "temporary_allocation_patterns": [
                "per-SPT np.arange index arrays",
                "two advanced-index arrays for every emitted SPT cut",
                "per-SPT emitted arrays plus final concatenated duplicate",
                "upper traversal stack concatenation at every descended level",
                "final selected-hierarchy and skybox+raster concatenations",
                "torch-to-NumPy camera copies in selection camera preparation",
            ],
        },
        "cprofile_top_relevant": _profile_rows(profile),
        "future_optimization_ranking": [
            {
                "rank": 1,
                "candidate": "fuse per-SPT interval, predicate, and output compaction",
                "category": "vectorization / persistent buffers / GPU migration",
            },
            {
                "rank": 2,
                "candidate": "avoid duplicated per-SPT emitted arrays and final concatenate",
                "category": "persistent buffers / allocation removal",
            },
            {
                "rank": 3,
                "candidate": "reuse upper traversal stacks and cut buffers",
                "category": "persistent buffers / remove Python overhead",
            },
            {
                "rank": 4,
                "candidate": "move HSPT/SPT cut selection to GPU",
                "category": "GPU migration / custom kernel",
            },
            {
                "rank": 5,
                "candidate": "stronger hierarchy-level pruning",
                "category": "algorithmic pruning",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    result = profile_selection(args.asset.resolve(), args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
