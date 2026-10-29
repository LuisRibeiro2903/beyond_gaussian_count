#!/usr/bin/env python3
"""Read-only hierarchy-topology invariants for A LoD of Gaussians node tables.

The ALoD node layout is ``[depth, parent, child_count, first_child,
next_sibling, reserved]``.  Index zero is also used as the null child/sibling
sentinel, while fine-training hierarchies prepend skybox rows.  This checker
never follows or repairs skybox rows.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEPTH = 0
PARENT = 1
CHILD_COUNT = 2
FIRST_CHILD = 3
NEXT_SIBLING = 4


@dataclass
class TopologyReport:
    valid: bool
    active_size: int
    node_rows: int
    skybox_rows: int
    root: int | None
    roots: list[int]
    expected_nodes: int
    reachable_nodes: int
    checked_edges: int
    maximum_traversed_depth: int
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    mismatch_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "active_size": self.active_size,
            "node_rows": self.node_rows,
            "skybox_rows": self.skybox_rows,
            "root": self.root,
            "roots": self.roots,
            "expected_nodes": self.expected_nodes,
            "reachable_nodes": self.reachable_nodes,
            "checked_edges": self.checked_edges,
            "maximum_traversed_depth": self.maximum_traversed_depth,
            "mismatch_counts": self.mismatch_counts,
            "mismatches": self.mismatches,
        }


def _as_numpy(nodes: Any) -> np.ndarray:
    """Return a detached CPU view/copy without mutating the input."""
    if hasattr(nodes, "detach"):
        nodes = nodes.detach().cpu().numpy()
    result = np.asarray(nodes)
    if result.ndim != 2 or result.shape[1] < 5:
        raise ValueError(f"expected node table shaped [N, >=5], got {result.shape}")
    return result


def validate_topology(
    nodes: Any,
    *,
    active_size: int | None = None,
    skybox_rows: int = 0,
    root: int | None = None,
    subtree_roots: Sequence[int] | None = None,
    expected_subtree_nodes: Iterable[int] | None = None,
    require_leaf_first_child_zero: bool = True,
    max_mismatches: int = 1_000,
) -> TopologyReport:
    """Validate a complete active hierarchy or selected affected subtrees.

    ``subtree_roots`` limits reachability/edge checks to those roots.  When
    supplied, ``expected_subtree_nodes`` can require an exact affected-node
    set.  Global root cardinality is still checked only for a full validation.
    """
    table = _as_numpy(nodes)
    row_count = int(table.shape[0])
    size = row_count if active_size is None else int(active_size)
    if not 0 <= skybox_rows <= size <= row_count:
        raise ValueError(
            f"invalid sizes: skybox={skybox_rows}, active={size}, rows={row_count}"
        )

    mismatches: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    def add(kind: str, **details: Any) -> None:
        counts[kind] += 1
        if len(mismatches) < max_mismatches:
            mismatches.append({"kind": kind, **details})

    hierarchy_ids = range(skybox_rows, size)
    roots = [index for index in hierarchy_ids if int(table[index, PARENT]) == -1]
    full_check = subtree_roots is None
    if full_check:
        if len(roots) != 1:
            add("root_count", roots=roots, expected=1)
        selected_root = roots[0] if len(roots) == 1 else root
        if root is not None and roots != [int(root)]:
            add("root_identity", roots=roots, expected=int(root))
            selected_root = int(root)
        start_nodes = [] if selected_root is None else [selected_root]
        expected = set(hierarchy_ids)
    else:
        selected_root = int(root) if root is not None else None
        requested_starts = [int(index) for index in subtree_roots]
        requested_set = set(requested_starts)
        start_nodes = []
        for candidate in requested_starts:
            ancestor = int(table[candidate, PARENT]) if 0 <= candidate < size else -1
            seen_ancestors: set[int] = set()
            nested = False
            while skybox_rows <= ancestor < size and ancestor not in seen_ancestors:
                if ancestor in requested_set:
                    nested = True
                    break
                seen_ancestors.add(ancestor)
                ancestor = int(table[ancestor, PARENT])
            if not nested and candidate not in start_nodes:
                start_nodes.append(candidate)
        expected = (
            {int(index) for index in expected_subtree_nodes}
            if expected_subtree_nodes is not None
            else None
        )

    for start in start_nodes:
        if not skybox_rows <= start < size:
            add("subtree_root_out_of_range", node=start, lower=skybox_rows, upper=size)

    visited: set[int] = set()
    owner: dict[int, int] = {}
    stack = list(reversed(start_nodes))
    checked_edges = 0
    maximum_depth = 0

    while stack:
        index = stack.pop()
        if not skybox_rows <= index < size:
            add("forward_index_out_of_range", node=index, lower=skybox_rows, upper=size)
            continue
        if index in visited:
            add("cycle_or_duplicate_reachability", node=index)
            continue
        visited.add(index)
        stored_depth = int(table[index, DEPTH])
        maximum_depth = max(maximum_depth, stored_depth)

        child_count = int(table[index, CHILD_COUNT])
        first_child = int(table[index, FIRST_CHILD])
        if child_count < 0:
            add("negative_child_count", node=index, child_count=child_count)
            continue
        if child_count == 0:
            if require_leaf_first_child_zero and first_child != 0:
                add("leaf_first_child_nonzero", node=index, first_child=first_child)
            continue
        if first_child == 0:
            add("interior_first_child_null", node=index, child_count=child_count)
            continue

        children: list[int] = []
        child = first_child
        chain_seen: set[int] = set()
        for ordinal in range(child_count):
            if child in chain_seen:
                add("sibling_cycle", node=index, child=child, ordinal=ordinal)
                break
            chain_seen.add(child)
            if not skybox_rows <= child < size:
                add(
                    "child_out_of_range_or_skybox",
                    node=index,
                    child=child,
                    ordinal=ordinal,
                    lower=skybox_rows,
                    upper=size,
                )
                break
            children.append(child)
            checked_edges += 1
            previous_owner = owner.get(child)
            if previous_owner is not None:
                add("duplicate_child_ownership", child=child, parents=[previous_owner, index])
            else:
                owner[child] = index

            stored_parent = int(table[child, PARENT])
            if stored_parent != index:
                add(
                    "parent_mismatch",
                    parent=index,
                    child=child,
                    stored_parent=stored_parent,
                )
            child_depth = int(table[child, DEPTH])
            if child_depth != stored_depth + 1:
                add(
                    "depth_mismatch",
                    parent=index,
                    parent_depth=stored_depth,
                    child=child,
                    stored_child_depth=child_depth,
                    expected_child_depth=stored_depth + 1,
                )

            sibling = int(table[child, NEXT_SIBLING])
            if ordinal + 1 < child_count:
                if sibling == 0:
                    add(
                        "short_sibling_chain",
                        node=index,
                        child=child,
                        ordinal=ordinal,
                        child_count=child_count,
                    )
                    break
                child = sibling
            elif sibling != 0:
                add(
                    "unterminated_sibling_chain",
                    node=index,
                    child=child,
                    next_sibling=sibling,
                )

        if len(children) != child_count:
            add(
                "child_count_mismatch",
                node=index,
                child_count=child_count,
                traversed_children=children,
            )
        stack.extend(reversed(children))

    if full_check and selected_root is not None:
        root_depth = int(table[selected_root, DEPTH])
        if root_depth != 0:
            add("root_depth", node=selected_root, stored_depth=root_depth, expected=0)
        if int(table[selected_root, PARENT]) != -1:
            add(
                "root_parent",
                node=selected_root,
                stored_parent=int(table[selected_root, PARENT]),
                expected=-1,
            )
        missing = sorted(expected - visited)
        extra = sorted(visited - expected)
        if missing or extra:
            add(
                "reachability",
                missing_count=len(missing),
                missing=missing[:100],
                extra_count=len(extra),
                extra=extra[:100],
            )
        for index in expected:
            if index == selected_root:
                if index in owner:
                    add("root_has_incoming_edge", node=index, owner=owner[index])
            elif index not in owner:
                add("nonroot_without_forward_parent", node=index)
    elif expected is not None:
        missing = sorted(expected - visited)
        extra = sorted(visited - expected)
        if missing or extra:
            add(
                "subtree_reachability",
                missing_count=len(missing),
                missing=missing[:100],
                extra_count=len(extra),
                extra=extra[:100],
            )

    return TopologyReport(
        valid=not counts,
        active_size=size,
        node_rows=row_count,
        skybox_rows=skybox_rows,
        root=selected_root,
        roots=roots,
        expected_nodes=len(expected) if expected is not None else len(visited),
        reachable_nodes=len(visited),
        checked_edges=checked_edges,
        maximum_traversed_depth=maximum_depth,
        mismatches=mismatches,
        mismatch_counts=dict(sorted(counts.items())),
    )


def changed_rows(before: Any, after: Any, *, active_size: int | None = None) -> list[dict[str, Any]]:
    """Return exact row-level node metadata changes."""
    before_np = _as_numpy(before)
    after_np = _as_numpy(after)
    if before_np.shape != after_np.shape:
        raise ValueError(f"node table shapes differ: {before_np.shape} != {after_np.shape}")
    size = before_np.shape[0] if active_size is None else int(active_size)
    indices = np.flatnonzero(np.any(before_np[:size] != after_np[:size], axis=1))
    return [
        {
            "node": int(index),
            "before": [int(value) for value in before_np[index]],
            "after": [int(value) for value in after_np[index]],
        }
        for index in indices
    ]


def load_dhier_nodes(path: Path) -> tuple[np.memmap, dict[str, int]]:
    """Memory-map the node payload of an uncompressed dynamic hierarchy."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        gaussian_count, degree = struct.unpack("<ii", handle.read(8))
    sh_floats = 3 * (degree + 1) ** 2
    property_floats = 3 + 4 + 3 + 1 + sh_floats
    node_count_offset = 8 + gaussian_count * property_floats * 4
    with path.open("rb") as handle:
        handle.seek(node_count_offset)
        node_count = struct.unpack("<i", handle.read(4))[0]
    expected_bytes = node_count_offset + 4 + node_count * 6 * 4
    if expected_bytes != size:
        raise ValueError(f".dhier byte length mismatch: {expected_bytes} != {size}")
    nodes = np.memmap(
        path,
        dtype="<i4",
        mode="r",
        offset=node_count_offset + 4,
        shape=(node_count, 6),
    )
    return nodes, {
        "gaussians": gaussian_count,
        "nodes": node_count,
        "sh_degree": degree,
        "bytes": size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dhier", type=Path, required=True)
    parser.add_argument("--skybox-rows", type=int, default=0)
    parser.add_argument("--allow-leaf-first-child", action="store_true")
    parser.add_argument("--max-mismatches", type=int, default=100)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    nodes, header = load_dhier_nodes(args.dhier)
    report = validate_topology(
        nodes,
        active_size=header["nodes"],
        skybox_rows=args.skybox_rows,
        root=args.skybox_rows,
        require_leaf_first_child_zero=not args.allow_leaf_first_child,
        max_mismatches=args.max_mismatches,
    )
    output = {"path": str(args.dhier.resolve()), **header, **report.as_dict()}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if report.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
