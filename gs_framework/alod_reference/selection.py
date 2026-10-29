"""CPU reference upper traversal and SPT cut for pinned ALoD evaluation."""

from __future__ import annotations

import time

import numpy as np

from .frustum import (
    extract_four_frustum_planes,
    frustum_visible_spheres,
    scale_derived_bounds,
)
from .types import (
    HSPT,
    ReferenceCamera,
    ReferenceConfig,
    SelectionResult,
    SelectionTimings,
    SPTCutResult,
    UpperTraversalResult,
)


CHILD_COUNT = 2
FIRST_CHILD = 3
NEXT_SIBLING = 4
GLOBAL_ID = 5


def _sum3_cuda_order(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.asarray(values[..., 0] + values[..., 2], dtype=np.float32)
        + values[..., 1],
        dtype=np.float32,
    )


def _squared_distances(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    delta = np.asarray(points - center, dtype=np.float32)
    squares = np.asarray(delta * delta, dtype=np.float32)
    return _sum3_cuda_order(squares)


def _distances(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.asarray(np.sqrt(_squared_distances(points, center)), dtype=np.float32)


def traverse_upper_hierarchy(
    hspt: HSPT,
    camera: ReferenceCamera,
    config: ReferenceConfig,
) -> UpperTraversalResult:
    planes = extract_four_frustum_planes(camera.full_projection_transform)
    bounds = scale_derived_bounds(hspt.upper_tree_log_scaling)

    stack = np.asarray([0], dtype=np.int32)
    cut_parts: list[np.ndarray] = []
    rejected_parts: list[np.ndarray] = []
    nodes = hspt.upper_tree_nodes

    while stack.size:
        visible = frustum_visible_spheres(
            hspt.upper_tree_xyz[stack], bounds[stack], planes
        )
        if np.any(~visible):
            rejected_parts.append(stack[~visible].copy())
        stack = stack[visible]
        if stack.size == 0:
            break

        leaves = nodes[stack, CHILD_COUNT] == 0
        if np.any(leaves):
            cut_parts.append(stack[leaves].copy())
        interior = stack[~leaves]
        if interior.size == 0:
            break

        camera_distance_squared = _squared_distances(
            hspt.upper_tree_xyz[interior], camera.camera_center
        )
        rhs = np.asarray(
            camera_distance_squared * np.float32(config.distance_multiplier),
            dtype=np.float32,
        )
        descend = hspt.upper_min_distance_squared[interior] > rhs
        if np.any(~descend):
            cut_parts.append(interior[~descend].copy())
        interior = interior[descend]
        first = nodes[interior, FIRST_CHILD]
        second = nodes[first, NEXT_SIBLING]
        stack = np.concatenate((first, second)).astype(np.int32, copy=False)

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
    candidate_mask = nodes[leaf_local, FIRST_CHILD] >= 0
    candidate_local = leaf_local[candidate_mask]
    candidate_spt_ids = nodes[candidate_local, FIRST_CHILD].astype(np.int32, copy=True)
    candidate_roots = nodes[candidate_local, GLOBAL_ID].astype(np.int32, copy=True)

    candidate_distances = _distances(
        hspt.upper_tree_xyz[candidate_local], camera.camera_center
    )
    candidate_distances = np.asarray(
        candidate_distances * np.float32(config.distance_multiplier),
        dtype=np.float32,
    )
    upper_render = nodes[coarse_local][
        nodes[coarse_local, FIRST_CHILD] <= 0, GLOBAL_ID
    ].astype(np.int32, copy=True)

    return UpperTraversalResult(
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
        candidate_spt_upper_local_ids=np.ascontiguousarray(
            candidate_local, dtype=np.int32
        ),
        candidate_spt_root_global_node_ids=np.ascontiguousarray(
            candidate_roots, dtype=np.int32
        ),
        candidate_spt_distances=np.ascontiguousarray(
            candidate_distances, dtype=np.float32
        ),
        frustum_planes=planes,
        frustum_rejected_upper_local_ids=np.ascontiguousarray(
            rejected_local, dtype=np.int32
        ),
    )


def _official_interval_size(
    hspt: HSPT, spt_id: int, distance: np.float32
) -> int:
    low = int(hspt.spt_starts[spt_id])
    start = low
    high = int(hspt.spt_starts[spt_id + 1])
    while high - low > 1:
        pivot = (low + high) // 2
        if hspt.spt_max[pivot] > distance:
            low = pivot
        else:
            high = pivot
    return max(high - start - 1, 1)


def cut_candidate_spts(
    hspt: HSPT, upper: UpperTraversalResult
) -> tuple[tuple[SPTCutResult, ...], np.ndarray]:
    """Port the native interval/prefix/population semantics literally.

    The exclusive-prefix population kernel tests offset zero only for the
    first requested SPT; every interval then covers offsets 1 through its
    computed interval size.  The root test normally fails its minimum-distance
    predicate, but preserving it matters for synthetic and regression cases.
    Selected zero IDs are compacted away exactly like ``IsNonZero``.
    """

    results: list[SPTCutResult] = []
    combined: list[np.ndarray] = []
    for candidate_index, (spt_id_value, distance_value) in enumerate(
        zip(upper.candidate_spt_ids, upper.candidate_spt_distances)
    ):
        spt_id = int(spt_id_value)
        distance = np.float32(distance_value)
        if not 0 <= spt_id < hspt.spt_count:
            raise ValueError(f"candidate SPT ID {spt_id} is out of range")
        interval_size = _official_interval_size(hspt, spt_id, distance)
        start = int(hspt.spt_starts[spt_id])
        first_offset = 0 if candidate_index == 0 else 1
        local_offsets = np.arange(
            first_offset, interval_size + 1, dtype=np.int64
        )
        flat_indices = start + local_offsets
        keep = hspt.spt_min[flat_indices] < distance
        emitted = hspt.spt_to_global_node_id[flat_indices][keep]
        emitted = emitted[emitted != 0].astype(np.int64, copy=False)
        emitted = np.ascontiguousarray(emitted, dtype=np.int64)
        root_global_id = int(hspt.spt_root_global_node_ids[spt_id])
        result = SPTCutResult(
            spt_id=spt_id,
            root_global_node_id=root_global_id,
            distance=distance,
            interval_size=interval_size,
            emitted_global_node_ids=emitted,
        )
        results.append(result)
        combined.append(emitted)

    selected = (
        np.concatenate(combined).astype(np.int64, copy=False)
        if combined
        else np.empty(0, dtype=np.int64)
    )
    return tuple(results), np.ascontiguousarray(selected, dtype=np.int64)


def select_reference(
    hspt: HSPT,
    camera: ReferenceCamera,
    config: ReferenceConfig,
) -> SelectionResult:
    total_start = time.perf_counter()
    upper_start = total_start
    upper = traverse_upper_hierarchy(hspt, camera, config)
    upper_seconds = time.perf_counter() - upper_start

    cut_start = time.perf_counter()
    per_spt, selected_spt = cut_candidate_spts(hspt, upper)
    cut_seconds = time.perf_counter() - cut_start

    selected_hierarchy = np.concatenate(
        (
            upper.upper_render_global_node_ids.astype(np.int64, copy=False),
            selected_spt,
        )
    ).astype(np.int64, copy=False)
    skybox = np.arange(config.skybox_count, dtype=np.int64)
    raster = np.concatenate((skybox, selected_hierarchy)).astype(
        np.int64, copy=False
    )
    total_seconds = time.perf_counter() - total_start
    return SelectionResult(
        camera_stable_id=camera.stable_id,
        upper=upper,
        per_spt=per_spt,
        selected_spt_global_node_ids=selected_spt,
        selected_hierarchy_global_node_ids=np.ascontiguousarray(
            selected_hierarchy, dtype=np.int64
        ),
        raster_global_node_ids=np.ascontiguousarray(raster, dtype=np.int64),
        timings=SelectionTimings(
            upper_traversal_seconds=upper_seconds,
            spt_cut_seconds=cut_seconds,
            total_seconds=total_seconds,
        ),
    )


__all__ = [
    "cut_candidate_spts",
    "select_reference",
    "traverse_upper_hierarchy",
]
