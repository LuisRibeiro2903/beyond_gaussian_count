"""Production HSPT construction for fully resident ALoD scenes.

This implementation intentionally does not import ``alod_reference``.  It
retains the pinned source's CPU float32 operation order so the production and
reference representations can be compared independently.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from .types import AlodHierarchySource, AlodHspt, AlodRuntimeConfig


DEPTH = 0
PARENT = 1
CHILD_COUNT = 2
FIRST_CHILD = 3
NEXT_SIBLING = 4
GLOBAL_ID = 5

_ROOT_MAX_DISTANCE = np.float32(1_000_000_000_000.0)
_LEAF_MIN_DISTANCE = np.float32(-1_000_000_000.0)


def _cpu_rows(array: np.ndarray, indices: np.ndarray) -> torch.Tensor:
    values = np.array(array[np.asarray(indices, dtype=np.int64)], dtype=np.float32, copy=True)
    result = torch.from_numpy(values)
    if result.device.type != "cpu":
        raise RuntimeError("production HSPT construction must remain on CPU")
    return result


def _minimum_distances(
    hierarchy: AlodHierarchySource,
    global_ids: np.ndarray,
    target_granularity: float,
    *,
    leaves_are_negative_infinity: bool = True,
) -> torch.Tensor:
    ids = np.asarray(global_ids, dtype=np.int32).reshape(-1)
    if ids.size == 0:
        return torch.empty(0, dtype=torch.float32)
    scales = torch.exp(_cpu_rows(hierarchy.log_scales, ids))
    area_proxy = (
        scales[:, 0] * scales[:, 1]
        + scales[:, 0] * scales[:, 2]
        + scales[:, 1] * scales[:, 2]
    )
    result = torch.sqrt(area_proxy) / target_granularity
    if leaves_are_negative_infinity:
        leaf_mask = np.asarray(hierarchy.nodes[ids, CHILD_COUNT] == 0)
        if np.any(leaf_mask):
            result[torch.from_numpy(leaf_mask)] = float(_LEAF_MIN_DISTANCE)
    return result


def _volume_mask(
    hierarchy: AlodHierarchySource, global_ids: np.ndarray, threshold: float
) -> np.ndarray:
    ids = np.asarray(global_ids, dtype=np.int32).reshape(-1)
    if ids.size == 0:
        return np.empty(0, dtype=bool)
    scales = torch.exp(_cpu_rows(hierarchy.log_scales, ids))
    return (torch.prod(scales, dim=-1) > threshold).numpy().copy()


def _children(nodes: np.ndarray, parents: np.ndarray) -> np.ndarray:
    if parents.size == 0:
        return np.empty(0, dtype=np.int32)
    first = np.asarray(nodes[parents, FIRST_CHILD], dtype=np.int32)
    second = np.asarray(nodes[first, NEXT_SIBLING], dtype=np.int32)
    both = np.concatenate((first, second)).astype(np.int32, copy=False)
    return both[both > 0]


def _upper_partition(
    hierarchy: AlodHierarchySource, config: AlodRuntimeConfig
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
        descend = _volume_mask(hierarchy, interior, config.spt_root_volume)
        if np.any(~descend):
            cut_levels.append(interior[~descend].copy())
        stack = _children(hierarchy.nodes, interior[descend])
    return (
        np.concatenate(upper_levels).astype(np.int32, copy=False),
        np.concatenate(cut_levels).astype(np.int32, copy=False)
        if cut_levels
        else np.empty(0, dtype=np.int32),
    )


def _one_spt(
    hierarchy: AlodHierarchySource,
    root_global_id: int,
    target_granularity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = np.asarray([root_global_id], dtype=np.int32)
    root_center = _cpu_rows(hierarchy.positions, root)[0]
    root_min = _minimum_distances(hierarchy, root, target_granularity)[0]
    id_levels = [root]
    min_levels = [np.asarray([root_min.item()], dtype=np.float32)]
    max_levels = [np.asarray([_ROOT_MAX_DISTANCE], dtype=np.float32)]

    stack = root
    parent_max = root_min.reshape(1).clone()
    while stack.size:
        first = np.asarray(hierarchy.nodes[stack, FIRST_CHILD], dtype=np.int32)
        next_stack = _children(hierarchy.nodes, stack)
        if next_stack.size == 0:
            break
        positions = _cpu_rows(hierarchy.positions, next_stack)
        center_distances = torch.sqrt(torch.sum((positions - root_center) ** 2, dim=1))
        minimum = _minimum_distances(
            hierarchy, next_stack, target_granularity
        ) + center_distances
        parent_max = parent_max[torch.from_numpy(np.asarray(first > 0))]
        parent_max = torch.cat((parent_max, parent_max))
        effective_min = torch.where(minimum < parent_max, minimum, parent_max)
        id_levels.append(next_stack.copy())
        min_levels.append(effective_min.numpy().copy())
        max_levels.append(parent_max.numpy().copy())
        stack = next_stack
        parent_max = effective_min.clone()

    ids = np.concatenate(id_levels).astype(np.int32, copy=False)
    mins = np.concatenate(min_levels).astype(np.float32, copy=False)
    maxs = np.concatenate(max_levels).astype(np.float32, copy=False)
    order = np.argsort(-maxs, kind="stable")
    return ids[order], mins[order], maxs[order]


def _concat(parts: Iterable[np.ndarray], dtype: np.dtype) -> np.ndarray:
    values = list(parts)
    return (
        np.concatenate(values).astype(dtype, copy=False)
        if values
        else np.empty(0, dtype=dtype)
    )


def build_production_hspt(
    hierarchy: AlodHierarchySource, config: AlodRuntimeConfig
) -> AlodHspt:
    if config.root_global_node_id >= hierarchy.row_count:
        raise ValueError("root_global_node_id lies outside the hierarchy")
    if config.root_global_node_id < config.skybox_count:
        raise ValueError("hierarchy root must not lie inside the skybox prefix")

    source_nodes = np.array(hierarchy.nodes, dtype=np.int32, copy=True)
    source_nodes[:, GLOBAL_ID] = 0
    working = AlodHierarchySource(hierarchy.positions, hierarchy.log_scales, source_nodes)
    upper_ids, cut_ids = _upper_partition(working, config)

    spt_ids: list[np.ndarray] = []
    spt_mins: list[np.ndarray] = []
    spt_maxs: list[np.ndarray] = []
    roots: list[int] = []
    for raw_cut_id in cut_ids:
        cut_id = int(raw_cut_id)
        if source_nodes[cut_id, CHILD_COUNT] == 0:
            continue
        ids, mins, maxs = _one_spt(working, cut_id, config.target_granularity)
        if ids.size > config.min_spt_size:
            roots.append(cut_id)
            spt_ids.append(ids)
            spt_mins.append(mins)
            spt_maxs.append(maxs)
        else:
            upper_ids = np.concatenate((upper_ids, ids[1:]))

    upper_ids = np.sort(upper_ids.astype(np.int32, copy=False), kind="stable")
    if np.unique(upper_ids).size != upper_ids.size:
        raise ValueError("derived upper hierarchy contains duplicate global IDs")

    spt_to_global = _concat(spt_ids, np.dtype("int32"))
    spt_min = _concat(spt_mins, np.dtype("float32"))
    spt_max = _concat(spt_maxs, np.dtype("float32"))
    spt_sizes = np.asarray([part.size for part in spt_ids], dtype=np.int32)
    spt_starts = np.empty(spt_sizes.size + 1, dtype=np.int32)
    spt_starts[0] = 0
    np.cumsum(spt_sizes, dtype=np.int32, out=spt_starts[1:])
    roots_array = np.asarray(roots, dtype=np.int64)

    upper_nodes = np.array(source_nodes[upper_ids], dtype=np.int32, copy=True)
    spt_upper_local = np.searchsorted(
        upper_ids, np.asarray(roots, dtype=np.int32)
    ).astype(np.int32)
    if spt_upper_local.size:
        upper_nodes[spt_upper_local, CHILD_COUNT] = 0
        upper_nodes[spt_upper_local, FIRST_CHILD] = np.arange(
            spt_upper_local.size, dtype=np.int32
        )
    upper_nodes[:, GLOBAL_ID] = upper_ids
    upper_nodes[:, PARENT] = np.searchsorted(
        upper_ids, upper_nodes[:, PARENT]
    ).astype(np.int32)
    upper_nodes[0, PARENT] = -1

    is_spt_leaf = np.zeros(upper_ids.size, dtype=bool)
    is_spt_leaf[spt_upper_local] = True
    ordinary = ~is_spt_leaf
    original_children = upper_nodes[ordinary, FIRST_CHILD]
    upper_nodes[ordinary, FIRST_CHILD] = np.where(
        original_children == 0,
        -1,
        np.searchsorted(upper_ids, original_children),
    ).astype(np.int32)
    has_sibling = upper_nodes[:, NEXT_SIBLING] > 0
    upper_nodes[has_sibling, NEXT_SIBLING] = np.searchsorted(
        upper_ids, upper_nodes[has_sibling, NEXT_SIBLING]
    ).astype(np.int32)

    parent_global_ids = upper_nodes[upper_nodes[:, PARENT], GLOBAL_ID]
    upper_min = _minimum_distances(
        working,
        parent_global_ids.astype(np.int32, copy=False),
        config.target_granularity,
    ).square()
    upper_min[0] = float(_ROOT_MAX_DISTANCE)

    return AlodHspt(
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
        spt_root_global_node_ids=np.ascontiguousarray(roots_array, dtype=np.int64),
        spt_sizes=np.ascontiguousarray(spt_sizes, dtype=np.int32),
    )


__all__ = ["build_production_hspt"]
