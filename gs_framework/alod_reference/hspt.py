"""Independent CPU construction of ALoD hierarchical SPT metadata.

Topology, ordering, and identity are represented with NumPy.  The pinned
official implementation constructs its distance thresholds with PyTorch
float32 operations on CPU.  NumPy's vector math differs by one or more ULPs
for some rows, so only those source-sensitive expressions use CPU PyTorch.
Every tensor is checked to remain on CPU; this module never calls CUDA.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from .types import HSPT, HierarchyData, ReferenceConfig


DEPTH = 0
PARENT = 1
CHILD_COUNT = 2
FIRST_CHILD = 3
NEXT_SIBLING = 4
GLOBAL_ID = 5

_ROOT_MAX_DISTANCE = np.float32(1_000_000_000_000.0)
_LEAF_MIN_DISTANCE = np.float32(-1_000_000_000.0)


def _require_cpu(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if tensor.device.type != "cpu":
            raise RuntimeError("ALoD reference tensors must remain on CPU")


def _rows_float32(array: np.ndarray, indices: np.ndarray) -> torch.Tensor:
    values = np.array(array[np.asarray(indices, dtype=np.int64)], dtype=np.float32, copy=True)
    tensor = torch.from_numpy(values)
    _require_cpu(tensor)
    return tensor


def _source_min_distances(
    hierarchy: HierarchyData,
    global_ids: np.ndarray,
    target_granularity: float,
    *,
    leaves_are_negative_infinity: bool = True,
) -> torch.Tensor:
    """Match ``GaussianModel.get_min_distance`` float32 operation order."""

    ids = np.asarray(global_ids, dtype=np.int32).reshape(-1)
    if ids.size == 0:
        return torch.empty(0, dtype=torch.float32, device="cpu")
    scales = torch.exp(_rows_float32(hierarchy.log_scales, ids))
    _require_cpu(scales)
    area_proxy = (
        scales[:, 0] * scales[:, 1]
        + scales[:, 0] * scales[:, 2]
        + scales[:, 1] * scales[:, 2]
    )
    result = torch.sqrt(area_proxy) / target_granularity
    if leaves_are_negative_infinity:
        leaves = np.asarray(hierarchy.nodes[ids, CHILD_COUNT] == 0)
        if np.any(leaves):
            result[torch.from_numpy(leaves)] = float(_LEAF_MIN_DISTANCE)
    _require_cpu(result)
    return result


def _volume_above_threshold(
    hierarchy: HierarchyData, global_ids: np.ndarray, threshold: float
) -> np.ndarray:
    ids = np.asarray(global_ids, dtype=np.int32).reshape(-1)
    if ids.size == 0:
        return np.empty(0, dtype=bool)
    scales = torch.exp(_rows_float32(hierarchy.log_scales, ids))
    mask = torch.prod(scales, dim=-1) > threshold
    _require_cpu(scales, mask)
    return mask.numpy().copy()


def _children(nodes: np.ndarray, stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(nodes[stack, FIRST_CHILD], dtype=np.int32)
    second = np.asarray(nodes[first, NEXT_SIBLING], dtype=np.int32)
    combined = np.concatenate((first, second)).astype(np.int32, copy=False)
    return first, second, combined[combined > 0]


def _initial_upper_cut(
    hierarchy: HierarchyData, config: ReferenceConfig
) -> tuple[np.ndarray, np.ndarray]:
    stack = np.asarray([config.root_global_node_id], dtype=np.int32)
    upper_levels: list[np.ndarray] = []
    cut_levels: list[np.ndarray] = []
    while stack.size:
        upper_levels.append(stack.copy())
        leaves = hierarchy.nodes[stack, CHILD_COUNT] == 0
        if np.any(leaves):
            cut_levels.append(stack[leaves].copy())
        interior = stack[~leaves]
        descend = _volume_above_threshold(
            hierarchy, interior, config.spt_root_volume
        )
        if np.any(~descend):
            cut_levels.append(interior[~descend].copy())
        _, _, stack = _children(hierarchy.nodes, interior[descend])
    upper = np.concatenate(upper_levels).astype(np.int32, copy=False)
    cut = (
        np.concatenate(cut_levels).astype(np.int32, copy=False)
        if cut_levels
        else np.empty(0, dtype=np.int32)
    )
    return upper, cut


def _build_one_spt(
    hierarchy: HierarchyData,
    root_global_id: int,
    target_granularity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = np.asarray([root_global_id], dtype=np.int32)
    root_center = _rows_float32(hierarchy.positions, root)[0]
    root_min = _source_min_distances(
        hierarchy, root, target_granularity
    )[0]
    _require_cpu(root_center, root_min)

    id_levels: list[np.ndarray] = [root]
    min_levels: list[np.ndarray] = [np.asarray([root_min.item()], dtype=np.float32)]
    max_levels: list[np.ndarray] = [np.asarray([_ROOT_MAX_DISTANCE], dtype=np.float32)]

    stack = root
    max_distances = root_min.reshape(1).clone()
    while stack.size:
        first, _, next_stack = _children(hierarchy.nodes, stack)
        if next_stack.size == 0:
            break

        positions = _rows_float32(hierarchy.positions, next_stack)
        center_distances = torch.sqrt(
            torch.sum((positions - root_center) ** 2, dim=1)
        )
        min_distances = _source_min_distances(
            hierarchy, next_stack, target_granularity
        ) + center_distances

        first_exists = torch.from_numpy(np.asarray(first > 0))
        max_distances = max_distances[first_exists]
        max_distances = torch.cat((max_distances, max_distances))
        effective_min = torch.where(
            min_distances < max_distances, min_distances, max_distances
        )
        _require_cpu(positions, center_distances, min_distances, max_distances, effective_min)

        id_levels.append(next_stack.copy())
        min_levels.append(effective_min.numpy().copy())
        max_levels.append(max_distances.numpy().copy())
        stack = next_stack
        max_distances = effective_min.clone()

    ids = np.concatenate(id_levels).astype(np.int32, copy=False)
    mins = np.concatenate(min_levels).astype(np.float32, copy=False)
    maxs = np.concatenate(max_levels).astype(np.float32, copy=False)

    # CUDA argsort in the pinned run preserved input order for equal maxima.
    # Make that behavior explicit rather than depending on an unspecified tie rule.
    order = np.argsort(-maxs, kind="stable")
    return ids[order], mins[order], maxs[order]


def _concatenate(parts: Iterable[np.ndarray], dtype: np.dtype) -> np.ndarray:
    materialized = list(parts)
    if not materialized:
        return np.empty(0, dtype=dtype)
    return np.concatenate(materialized).astype(dtype, copy=False)


def build_hspt(hierarchy: HierarchyData, config: ReferenceConfig) -> HSPT:
    """Build the reference HSPT using only hierarchy rows and configuration."""

    if config.root_global_node_id >= hierarchy.row_count:
        raise ValueError("root_global_node_id lies outside the hierarchy")
    if config.root_global_node_id < config.skybox_count:
        raise ValueError("hierarchy root must not lie inside the skybox prefix")

    source_nodes = np.array(hierarchy.nodes, dtype=np.int32, copy=True)
    # ``create_from_hier`` zeros the serialized scratch/index column before HSPT use.
    source_nodes[:, GLOBAL_ID] = 0
    working = HierarchyData(hierarchy.positions, hierarchy.log_scales, source_nodes)

    upper_ids, cut_ids = _initial_upper_cut(working, config)
    spt_id_parts: list[np.ndarray] = []
    spt_min_parts: list[np.ndarray] = []
    spt_max_parts: list[np.ndarray] = []
    spt_roots: list[int] = []

    for cut_global_id in cut_ids:
        cut_id = int(cut_global_id)
        if source_nodes[cut_id, CHILD_COUNT] == 0:
            continue
        ids, mins, maxs = _build_one_spt(
            working, cut_id, config.target_granularity
        )
        if ids.size > config.min_spt_size:
            spt_roots.append(cut_id)
            spt_id_parts.append(ids)
            spt_min_parts.append(mins)
            spt_max_parts.append(maxs)
        else:
            upper_ids = np.concatenate((upper_ids, ids[1:]))

    upper_ids = np.sort(upper_ids.astype(np.int32, copy=False), kind="stable")
    if np.unique(upper_ids).size != upper_ids.size:
        raise ValueError("derived upper hierarchy contains duplicate global IDs")

    spt_to_global = _concatenate(spt_id_parts, np.dtype("int32"))
    spt_min = _concatenate(spt_min_parts, np.dtype("float32"))
    spt_max = _concatenate(spt_max_parts, np.dtype("float32"))
    spt_sizes = np.asarray([part.size for part in spt_id_parts], dtype=np.int32)
    spt_starts = np.empty(spt_sizes.size + 1, dtype=np.int32)
    spt_starts[0] = 0
    np.cumsum(spt_sizes, dtype=np.int32, out=spt_starts[1:])
    spt_roots_array = np.asarray(spt_roots, dtype=np.int64)

    upper_nodes = np.array(source_nodes[upper_ids], dtype=np.int32, copy=True)
    cut_spt_local = np.searchsorted(
        upper_ids, np.asarray(spt_roots, dtype=np.int32)
    ).astype(np.int32)
    if cut_spt_local.size:
        upper_nodes[cut_spt_local, CHILD_COUNT] = 0
        upper_nodes[cut_spt_local, FIRST_CHILD] = np.arange(
            cut_spt_local.size, dtype=np.int32
        )

    upper_nodes[:, GLOBAL_ID] = upper_ids
    upper_nodes[:, PARENT] = np.searchsorted(
        upper_ids, upper_nodes[:, PARENT]
    ).astype(np.int32)
    upper_nodes[0, PARENT] = -1

    is_spt_leaf = np.zeros(upper_ids.size, dtype=bool)
    is_spt_leaf[cut_spt_local] = True
    normal = ~is_spt_leaf
    original_children = upper_nodes[normal, FIRST_CHILD]
    upper_nodes[normal, FIRST_CHILD] = np.where(
        original_children == 0,
        -1,
        np.searchsorted(upper_ids, original_children),
    ).astype(np.int32)
    has_sibling = upper_nodes[:, NEXT_SIBLING] > 0
    upper_nodes[has_sibling, NEXT_SIBLING] = np.searchsorted(
        upper_ids, upper_nodes[has_sibling, NEXT_SIBLING]
    ).astype(np.int32)

    parent_global_ids = upper_nodes[upper_nodes[:, PARENT], GLOBAL_ID]
    upper_min = _source_min_distances(
        working,
        parent_global_ids.astype(np.int32, copy=False),
        config.target_granularity,
    ).square()
    upper_min[0] = float(_ROOT_MAX_DISTANCE)
    _require_cpu(upper_min)

    return HSPT(
        upper_tree_nodes=np.ascontiguousarray(upper_nodes, dtype=np.int32),
        upper_to_global_node_id=np.ascontiguousarray(upper_ids, dtype=np.int32),
        upper_tree_xyz=np.ascontiguousarray(hierarchy.positions[upper_ids], dtype=np.float32),
        upper_tree_log_scaling=np.ascontiguousarray(
            hierarchy.log_scales[upper_ids], dtype=np.float32
        ),
        upper_min_distance_squared=np.ascontiguousarray(
            upper_min.numpy(), dtype=np.float32
        ),
        spt_starts=np.ascontiguousarray(spt_starts, dtype=np.int32),
        spt_min=np.ascontiguousarray(spt_min, dtype=np.float32),
        spt_max=np.ascontiguousarray(spt_max, dtype=np.float32),
        spt_to_global_node_id=np.ascontiguousarray(spt_to_global, dtype=np.int32),
        spt_root_global_node_ids=np.ascontiguousarray(spt_roots_array, dtype=np.int64),
        spt_sizes=np.ascontiguousarray(spt_sizes, dtype=np.int32),
    )
