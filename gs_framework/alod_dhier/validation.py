"""Numeric and topology validation for read-only ALoD hierarchies."""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Any

import numpy as np

from .format import DhierAsset, DhierError


DEPTH = 0
PARENT = 1
CHILD_COUNT = 2
FIRST_CHILD = 3
NEXT_SIBLING = 4
MAX_SIDE_OR_INDEX = 5
SKYBOX_TOPOLOGY_SENTINEL = -99
NULL_LINK = 0
ROOT_PARENT = -1


class ValidationProfile(str, Enum):
    STRICT = "strict"
    CREATOR_INITIAL = "creator-initial"


class DhierValidationError(DhierError):
    report: dict[str, Any]

    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message, details={"validation": report})
        self.report = report


class DhierNumericError(DhierValidationError):
    category = "numeric"


class DhierTopologyError(DhierValidationError):
    category = "topology"


def _json_float(value: Any) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def _numeric_array_summary(array: np.ndarray) -> dict[str, Any]:
    finite_mask = np.isfinite(array)
    finite_count = int(np.count_nonzero(finite_mask))
    total = int(array.size)
    if finite_count == total:
        minimum = _json_float(np.min(array)) if total else None
        maximum = _json_float(np.max(array)) if total else None
    elif finite_count:
        finite_values = np.asarray(array)[finite_mask]
        minimum = _json_float(np.min(finite_values))
        maximum = _json_float(np.max(finite_values))
    else:
        minimum = None
        maximum = None
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "byte_size": int(array.nbytes),
        "finite": finite_count == total,
        "finite_count": finite_count,
        "element_count": total,
        "min": minimum,
        "max": maximum,
    }


def validate_numeric(
    asset: DhierAsset,
    *,
    profile: ValidationProfile | str = ValidationProfile.STRICT,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Validate serialized numeric attributes without applying activations."""
    selected_profile = ValidationProfile(profile)
    arrays = {
        "positions": asset.positions,
        "rotations": asset.rotations,
        "log_scales": asset.log_scales,
        "activated_opacities": asset.activated_opacities,
        "sh_coefficients": asset.sh_coefficients,
    }
    summaries = {name: _numeric_array_summary(array) for name, array in arrays.items()}

    opacity = asset.activated_opacities
    opacity_below_zero = opacity < 0.0
    opacity_above_one = opacity > 1.0
    opacity_invalid = (
        opacity_below_zero
        if selected_profile is ValidationProfile.CREATOR_INITIAL
        else np.logical_or(opacity_below_zero, opacity_above_one)
    )
    opacity_invalid_count = int(np.count_nonzero(opacity_invalid))
    opacity_bad_rows = np.flatnonzero(opacity_invalid.reshape(-1))[:16]

    rotations = np.asarray(asset.rotations)
    quaternion_norms = np.linalg.norm(rotations, axis=1)
    nonzero = quaternion_norms > 0.0
    quaternion_nonzero_count = int(np.count_nonzero(nonzero))
    quaternion_summary = {
        "count": int(quaternion_norms.size),
        "nonzero_count": quaternion_nonzero_count,
        "zero_count": int(quaternion_norms.size - quaternion_nonzero_count),
        "min": _json_float(np.min(quaternion_norms)),
        "max": _json_float(np.max(quaternion_norms)),
        "mean": _json_float(np.mean(quaternion_norms, dtype=np.float64)),
        "std": _json_float(np.std(quaternion_norms, dtype=np.float64)),
        "p50": _json_float(np.percentile(quaternion_norms, 50)),
        "p95": _json_float(np.percentile(quaternion_norms, 95)),
        "p99": _json_float(np.percentile(quaternion_norms, 99)),
    }

    all_finite = all(summary["finite"] for summary in summaries.values())
    valid = all_finite and opacity_invalid_count == 0 and quaternion_summary["zero_count"] == 0
    report = {
        "valid": valid,
        "profile": selected_profile.value,
        "serialized_representation": {
            "positions": "raw world-space float32",
            "rotations": "raw float32; not normalized by parser",
            "log_scales": "raw float32; exp is not applied by parser",
            "activated_opacities": (
                "serialized activated/pre-clamp float32; creator interior merging may exceed 1; "
                "sigmoid is not applied by parser"
            ),
            "sh_coefficients": "raw float32",
        },
        "arrays": summaries,
        "activated_opacity_range": {
            "expected": (
                "non-negative; creator-merged interior values above 1 are permitted and later clamped"
                if selected_profile is ValidationProfile.CREATOR_INITIAL
                else "inclusive [0, 1]"
            ),
            "below_zero_count": int(np.count_nonzero(opacity_below_zero)),
            "above_one_count": int(np.count_nonzero(opacity_above_one)),
            "invalid_count": opacity_invalid_count,
            "invalid_global_node_id_examples": [int(value) for value in opacity_bad_rows],
        },
        "quaternion_norms": quaternion_summary,
    }
    if raise_on_error and not valid:
        raise DhierNumericError("numeric/attribute validation failed", report)
    return report


class _Issues:
    def __init__(self, max_examples: int):
        self.max_examples = max(0, int(max_examples))
        self.counts: Counter[str] = Counter()
        self.examples: list[dict[str, Any]] = []

    def add(self, kind: str, count: int = 1, **details: Any) -> None:
        count = int(count)
        if count <= 0:
            return
        self.counts[kind] += count
        if len(self.examples) < self.max_examples:
            self.examples.append({"kind": kind, **details})

    def add_rows(
        self,
        kind: str,
        global_ids: np.ndarray,
        *,
        detail_builder: Any | None = None,
    ) -> None:
        ids = np.asarray(global_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return
        self.counts[kind] += int(ids.size)
        remaining = self.max_examples - len(self.examples)
        for global_id in ids[: max(0, remaining)]:
            details = {"global_node_id": int(global_id)}
            if detail_builder is not None:
                details.update(detail_builder(int(global_id)))
            self.examples.append({"kind": kind, **details})

    def as_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(sorted(self.counts.items())),
            "total": int(sum(self.counts.values())),
            "examples": self.examples,
            "examples_truncated": sum(self.counts.values()) > len(self.examples),
        }


def _histogram(values: np.ndarray) -> dict[str, int]:
    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(unique, counts)}


def validate_topology(
    asset: DhierAsset,
    *,
    skybox_count: int = 0,
    profile: ValidationProfile | str = ValidationProfile.STRICT,
    max_examples: int = 32,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Validate source-defined binary-tree topology.

    ``creator-initial`` differs from ``strict`` only for the pinned creator's
    unused leaf ``first_child`` field.  That writer assigns the current
    flattened-vector size even when ``child_count == 0``.  The compatibility
    profile permits only positive values no greater than one-past-last row,
    reports their count, and never indexes or traverses them.
    """
    selected_profile = ValidationProfile(profile)
    nodes = asset.nodes
    node_count = asset.layout.node_count
    skybox = int(skybox_count)
    if not 0 <= skybox < node_count:
        report = {
            "valid": False,
            "profile": selected_profile.value,
            "skybox_count": skybox,
            "issues": {
                "counts": {"invalid_skybox_count": 1},
                "total": 1,
                "examples": [{"kind": "invalid_skybox_count", "node_count": node_count}],
                "examples_truncated": False,
            },
        }
        if raise_on_error:
            raise DhierTopologyError("invalid skybox count", report)
        return report

    issues = _Issues(max_examples)
    hierarchy_ids = np.arange(skybox, node_count, dtype=np.int64)
    hierarchy = nodes[skybox:node_count]
    hierarchy_count = int(hierarchy.shape[0])

    skybox_report: dict[str, Any] = {
        "count": skybox,
        "topology_sentinel": SKYBOX_TOPOLOGY_SENTINEL,
        "topology_fields_valid": True,
        "reserved_field_zero": True,
    }
    if skybox:
        bad_topology = np.flatnonzero(np.any(nodes[:skybox, :5] != SKYBOX_TOPOLOGY_SENTINEL, axis=1))
        bad_reserved = np.flatnonzero(nodes[:skybox, MAX_SIDE_OR_INDEX] != 0)
        issues.add_rows("skybox_topology_sentinel", bad_topology)
        issues.add_rows("skybox_reserved_field", bad_reserved)
        skybox_report["topology_fields_valid"] = bad_topology.size == 0
        skybox_report["reserved_field_zero"] = bad_reserved.size == 0

    child_counts = np.asarray(hierarchy[:, CHILD_COUNT])
    leaf_mask = child_counts == 0
    interior_mask = child_counts == 2
    invalid_child_count = hierarchy_ids[np.logical_not(np.logical_or(leaf_mask, interior_mask))]
    issues.add_rows(
        "invalid_binary_child_count",
        invalid_child_count,
        detail_builder=lambda gid: {"child_count": int(nodes[gid, CHILD_COUNT])},
    )

    parents = np.asarray(hierarchy[:, PARENT])
    roots = hierarchy_ids[parents == ROOT_PARENT]
    if roots.size != 1:
        issues.add(
            "root_count",
            roots=int(roots.size),
            expected=1,
            root_examples=[int(value) for value in roots[:16]],
        )
    root = int(roots[0]) if roots.size == 1 else None
    invalid_parent_mask = np.logical_and(
        parents != ROOT_PARENT,
        np.logical_or(parents < skybox, parents >= node_count),
    )
    issues.add_rows(
        "parent_out_of_range_or_skybox",
        hierarchy_ids[invalid_parent_mask],
        detail_builder=lambda gid: {"parent": int(nodes[gid, PARENT])},
    )

    next_siblings = np.asarray(hierarchy[:, NEXT_SIBLING])
    invalid_sibling_mask = np.logical_and(
        next_siblings != NULL_LINK,
        np.logical_or(next_siblings < skybox, next_siblings >= node_count),
    )
    issues.add_rows(
        "next_sibling_out_of_range_or_skybox",
        hierarchy_ids[invalid_sibling_mask],
        detail_builder=lambda gid: {"next_sibling": int(nodes[gid, NEXT_SIBLING])},
    )

    leaf_ids = hierarchy_ids[leaf_mask]
    leaf_first_children = np.asarray(hierarchy[leaf_mask, FIRST_CHILD])
    leaf_quirk_count = 0
    if selected_profile is ValidationProfile.STRICT:
        bad_leaf_ids = leaf_ids[leaf_first_children != NULL_LINK]
        issues.add_rows(
            "leaf_first_child_nonzero",
            bad_leaf_ids,
            detail_builder=lambda gid: {"first_child": int(nodes[gid, FIRST_CHILD])},
        )
    else:
        quirk_mask = leaf_first_children != NULL_LINK
        leaf_quirk_count = int(np.count_nonzero(quirk_mask))
        invalid_quirk = np.logical_and(
            quirk_mask,
            np.logical_or(leaf_first_children <= 0, leaf_first_children > node_count),
        )
        issues.add_rows(
            "creator_leaf_first_child_invalid",
            leaf_ids[invalid_quirk],
            detail_builder=lambda gid: {"first_child": int(nodes[gid, FIRST_CHILD])},
        )

    interior_ids = hierarchy_ids[interior_mask]
    interior_first = np.asarray(hierarchy[interior_mask, FIRST_CHILD], dtype=np.int64)
    first_valid = np.logical_and.reduce(
        (interior_first != NULL_LINK, interior_first >= skybox, interior_first < node_count)
    )
    issues.add_rows(
        "interior_first_child_out_of_range_or_skybox",
        interior_ids[~first_valid],
        detail_builder=lambda gid: {"first_child": int(nodes[gid, FIRST_CHILD])},
    )

    valid_parent_ids = interior_ids[first_valid]
    valid_first = interior_first[first_valid]
    second_values = np.asarray(nodes[valid_first, NEXT_SIBLING], dtype=np.int64)
    second_valid = np.logical_and(second_values >= skybox, second_values < node_count)
    short_chain_ids = valid_parent_ids[second_values == NULL_LINK]
    issues.add_rows("short_sibling_chain", short_chain_ids)
    invalid_second = np.logical_and(second_values != NULL_LINK, ~second_valid)
    issues.add_rows(
        "second_child_out_of_range_or_skybox",
        valid_parent_ids[invalid_second],
        detail_builder=lambda gid: {
            "first_child": int(nodes[gid, FIRST_CHILD]),
            "second_child": int(nodes[int(nodes[gid, FIRST_CHILD]), NEXT_SIBLING]),
        },
    )

    complete_parent_ids = valid_parent_ids[second_valid]
    complete_first = valid_first[second_valid]
    complete_second = second_values[second_valid]
    unterminated = np.asarray(nodes[complete_second, NEXT_SIBLING]) != NULL_LINK
    issues.add_rows(
        "unterminated_sibling_chain",
        complete_parent_ids[unterminated],
        detail_builder=lambda gid: {
            "second_child": int(nodes[int(nodes[int(nodes[gid, FIRST_CHILD]), NEXT_SIBLING]), NEXT_SIBLING])
        },
    )

    edge_parents = np.concatenate((complete_parent_ids, complete_parent_ids))
    edge_children = np.concatenate((complete_first, complete_second))
    if edge_children.size:
        stored_parents = np.asarray(nodes[edge_children, PARENT], dtype=np.int64)
        parent_mismatch = stored_parents != edge_parents
        mismatch_indices = np.flatnonzero(parent_mismatch)
        issues.counts["parent_mismatch"] += int(mismatch_indices.size)
        for edge_index in mismatch_indices[: max(0, max_examples - len(issues.examples))]:
            child = int(edge_children[edge_index])
            issues.examples.append(
                {
                    "kind": "parent_mismatch",
                    "parent": int(edge_parents[edge_index]),
                    "child": child,
                    "stored_parent": int(nodes[child, PARENT]),
                }
            )

        expected_depths = np.asarray(nodes[edge_parents, DEPTH], dtype=np.int64) + 1
        stored_depths = np.asarray(nodes[edge_children, DEPTH], dtype=np.int64)
        depth_mismatch = stored_depths != expected_depths
        mismatch_indices = np.flatnonzero(depth_mismatch)
        issues.counts["depth_mismatch"] += int(mismatch_indices.size)
        for edge_index in mismatch_indices[: max(0, max_examples - len(issues.examples))]:
            child = int(edge_children[edge_index])
            parent = int(edge_parents[edge_index])
            issues.examples.append(
                {
                    "kind": "depth_mismatch",
                    "parent": parent,
                    "parent_depth": int(nodes[parent, DEPTH]),
                    "child": child,
                    "stored_child_depth": int(nodes[child, DEPTH]),
                    "expected_child_depth": int(nodes[parent, DEPTH]) + 1,
                }
            )

        local_children = edge_children - skybox
        ownership = np.bincount(local_children, minlength=hierarchy_count)
    else:
        ownership = np.zeros(hierarchy_count, dtype=np.int64)

    duplicate_local = np.flatnonzero(ownership > 1)
    issues.add_rows("duplicate_child_ownership", duplicate_local + skybox)
    if root is not None and ownership[root - skybox] != 0:
        issues.add(
            "root_has_incoming_edge",
            global_node_id=root,
            incoming_edges=int(ownership[root - skybox]),
        )
    expected_owned = np.ones(hierarchy_count, dtype=bool)
    if root is not None:
        expected_owned[root - skybox] = False
    missing_owner = np.flatnonzero(np.logical_and(expected_owned, ownership == 0))
    issues.add_rows("nonroot_without_forward_owner", missing_owner + skybox)

    first_by_local = np.full(hierarchy_count, -1, dtype=np.int64)
    second_by_local = np.full(hierarchy_count, -1, dtype=np.int64)
    if complete_parent_ids.size:
        parent_local = complete_parent_ids - skybox
        first_by_local[parent_local] = complete_first - skybox
        second_by_local[parent_local] = complete_second - skybox

    state = np.zeros(hierarchy_count, dtype=np.uint8)
    traversed_depth = np.full(hierarchy_count, -1, dtype=np.int32)
    reachable_count = 0
    cycle_count = 0
    duplicate_reach_count = 0
    if root is not None:
        capacity = max(8, hierarchy_count * 2 + 3)
        stack_nodes = np.empty(capacity, dtype=np.int64)
        stack_depths = np.empty(capacity, dtype=np.int32)
        stack_phases = np.empty(capacity, dtype=np.uint8)
        top = 1
        stack_nodes[0] = root - skybox
        stack_depths[0] = 0
        stack_phases[0] = 0
        while top:
            top -= 1
            local = int(stack_nodes[top])
            depth = int(stack_depths[top])
            phase = int(stack_phases[top])
            if phase:
                state[local] = 2
                continue
            if state[local] == 1:
                cycle_count += 1
                issues.add("cycle", global_node_id=local + skybox)
                continue
            if state[local] == 2:
                duplicate_reach_count += 1
                issues.add("duplicate_reachability", global_node_id=local + skybox)
                continue
            state[local] = 1
            traversed_depth[local] = depth
            reachable_count += 1
            if top + 3 > capacity:
                issues.add("traversal_stack_overflow")
                break
            stack_nodes[top] = local
            stack_depths[top] = depth
            stack_phases[top] = 1
            top += 1
            second = int(second_by_local[local])
            first = int(first_by_local[local])
            if second >= 0:
                stack_nodes[top] = second
                stack_depths[top] = depth + 1
                stack_phases[top] = 0
                top += 1
            if first >= 0:
                stack_nodes[top] = first
                stack_depths[top] = depth + 1
                stack_phases[top] = 0
                top += 1

    unreachable_local = np.flatnonzero(state == 0)
    issues.add_rows("unreachable_node", unreachable_local + skybox)
    if root is not None:
        if int(nodes[root, DEPTH]) != 0:
            issues.add("root_depth", global_node_id=root, stored_depth=int(nodes[root, DEPTH]))
        if int(nodes[root, NEXT_SIBLING]) != NULL_LINK:
            issues.add(
                "root_next_sibling_nonzero",
                global_node_id=root,
                next_sibling=int(nodes[root, NEXT_SIBLING]),
            )

    reached_depths = traversed_depth[state != 0]
    stored_hierarchy_depths = np.asarray(hierarchy[:, DEPTH])
    issue_report = issues.as_dict()
    report = {
        "valid": issue_report["total"] == 0,
        "profile": selected_profile.value,
        "binary_tree_child_counts": [0, 2],
        "global_node_id_contract": asset.global_node_id_contract,
        "skybox": skybox_report,
        "root_global_id": root,
        "root_count": int(roots.size),
        "root_examples": [int(value) for value in roots[:16]],
        "hierarchy_node_count": hierarchy_count,
        "leaf_count": int(np.count_nonzero(leaf_mask)),
        "interior_count": int(np.count_nonzero(interior_mask)),
        "edge_count": int(edge_children.size),
        "reachable_count": reachable_count,
        "unreachable_nodes": int(unreachable_local.size),
        "duplicate_ownership": int(duplicate_local.size),
        "duplicate_reachability": duplicate_reach_count,
        "cycles": cycle_count,
        "parent_mismatches": int(issues.counts.get("parent_mismatch", 0)),
        "depth_mismatches": int(issues.counts.get("depth_mismatch", 0)),
        "minimum_depth": int(np.min(reached_depths)) if reached_depths.size else None,
        "max_depth": int(np.max(reached_depths)) if reached_depths.size else None,
        "depth_histogram": _histogram(reached_depths),
        "stored_depth_histogram": _histogram(stored_hierarchy_depths),
        "creator_initial_leaf_first_child_quirk": {
            "enabled": selected_profile is ValidationProfile.CREATOR_INITIAL,
            "occurrence_count": leaf_quirk_count,
            "traversed": False,
            "allowed_range_when_nonzero": [1, node_count],
        },
        "max_side_or_index": {
            "semantic_status": "decoded raw; not used for topology",
            "min": int(np.min(hierarchy[:, MAX_SIDE_OR_INDEX])),
            "max": int(np.max(hierarchy[:, MAX_SIDE_OR_INDEX])),
        },
        "issues": issue_report,
    }
    if raise_on_error and not report["valid"]:
        raise DhierTopologyError("hierarchy topology validation failed", report)
    return report
