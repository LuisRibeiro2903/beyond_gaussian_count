#!/usr/bin/env python3
"""Exercise patched ALoD topology mutations against a real loaded hierarchy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

from topology_invariants import validate_topology


DEPTH = 0
PARENT = 1
CHILD_COUNT = 2
FIRST_CHILD = 3
NEXT_SIBLING = 4


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def checked_report(model: object, roots: torch.Tensor | None = None) -> dict[str, object]:
    report = validate_topology(
        model.nodes,
        active_size=model.size,
        skybox_rows=model.skybox_points,
        root=model.skybox_points if roots is None else None,
        subtree_roots=None if roots is None else roots.detach().cpu().tolist(),
        max_mismatches=100,
    )
    if not report.valid:
        raise RuntimeError(json.dumps(report.as_dict(), sort_keys=True))
    return report.as_dict()


def changed_rows(before: torch.Tensor, after: torch.Tensor) -> list[int]:
    common = min(len(before), len(after))
    changed = torch.where(torch.any(before[:common] != after[:common], dim=1))[0].tolist()
    changed.extend(range(common, len(after)))
    return changed


def subtree_max_relative_depth(nodes: torch.Tensor, root: int, size: int) -> int:
    base = int(nodes[root, DEPTH])
    maximum = base
    stack = [root]
    visited: set[int] = set()
    while stack:
        parent = stack.pop()
        if parent in visited:
            raise RuntimeError(f"cycle while selecting integration case at node {parent}")
        visited.add(parent)
        maximum = max(maximum, int(nodes[parent, DEPTH]))
        count = int(nodes[parent, CHILD_COUNT])
        if count == 0:
            continue
        child = int(nodes[parent, FIRST_CHILD])
        for child_number in range(count):
            if child < 0 or child >= size:
                raise RuntimeError(f"invalid integration selection child {child}")
            stack.append(child)
            sibling = int(nodes[child, NEXT_SIBLING])
            if child_number + 1 < count:
                child = sibling
    return maximum - base


def find_leaf_sibling_pair(model: object) -> tuple[int, int]:
    nodes = model.nodes
    parents = torch.where(nodes[: model.size, CHILD_COUNT] == 2)[0].tolist()
    for parent in parents:
        first = int(nodes[parent, FIRST_CHILD])
        second = int(nodes[first, NEXT_SIBLING])
        if first < model.skybox_points or second < model.skybox_points:
            continue
        if int(nodes[first, CHILD_COUNT]) == 0 and int(nodes[second, CHILD_COUNT]) == 0:
            return first, second
    raise RuntimeError("could not find a real leaf sibling pair")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sh-degree", type=int, default=1)
    parser.add_argument("--cap-max", type=int, default=1_100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = args.source.resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))
    from scene.gaussian_model import GaussianModel
    from train_hierarchy import validate_training_topology
    from utils.general_utils import safe_state

    safe_state(False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = GaussianModel(args.sh_degree)
    model.create_from_hier(str(args.hierarchy), 1.0, str(args.scaffold))
    model.compact_gaussians(
        "cpu", args.cap_max, "classic", prune_unused_gaussians=False
    )
    initial = checked_report(model)
    validate_training_topology(model, 0, "integration-initial")
    initial_size = model.size
    initial_skybox = model.nodes[: model.skybox_points].clone()

    deep_dead, selected_leaf = find_leaf_sibling_pair(model)
    add_results = []
    for level in range(1, 4):
        size_before_add = model.size
        model._densification_criterium.zero_()
        model._densification_criterium[selected_leaf] = 1.0
        before_add = model.nodes[: model.size].clone()
        started = time.perf_counter()
        add_return_value = model.add_new_gs(
            cap_max=args.cap_max,
            size=model.size,
            densification="classic",
            densify_percent=1.028,
            densify_threshold=0.0000075,
        )
        add_seconds = time.perf_counter() - started
        added_nodes = model.size - size_before_add
        if added_nodes != 2:
            raise RuntimeError(f"expected exactly two added nodes, got {added_nodes}")
        add_results.append(
            {
                "level": level,
                "selected_leaf": selected_leaf,
                "official_return_value": add_return_value,
                "added_nodes": added_nodes,
                "changed_rows": changed_rows(before_add, model.nodes[: model.size]),
                "seconds": add_seconds,
                "topology": checked_report(model),
            }
        )
        selected_leaf = size_before_add

    deep_sibling = int(model.nodes[deep_dead, NEXT_SIBLING])
    if deep_sibling == 0:
        parent = int(model.nodes[deep_dead, PARENT])
        deep_sibling = int(model.nodes[parent, FIRST_CHILD])
    descendant_levels = subtree_max_relative_depth(
        model.nodes, deep_sibling, model.size
    )
    if descendant_levels < 3:
        raise RuntimeError(f"constructed only {descendant_levels} descendant levels")
    dead_mask = torch.zeros(model.size, dtype=torch.bool, device=model.nodes.device)
    dead_mask[deep_dead] = True
    before_relocate = model.nodes[: model.size].clone()
    properties_before = tensor_sha256(model.properties[: model.size])
    started = time.perf_counter()
    affected_roots = model.relocate_gs(
        dead_mask,
        model.size,
        storage_device="cpu",
        densification="classic",
    )
    relocate_seconds = time.perf_counter() - started
    local_after_relocate = checked_report(model, affected_roots)
    validate_training_topology(
        model,
        1,
        "integration-deep-relocate-local",
        roots=affected_roots,
        before_nodes=before_relocate,
    )
    started = time.perf_counter()
    global_after_relocate = checked_report(model)
    validate_training_topology(
        model,
        1,
        "integration-deep-relocate-global",
        before_nodes=before_relocate,
    )
    global_validation_seconds = time.perf_counter() - started
    relocate_changed = changed_rows(before_relocate, model.nodes[: model.size])
    properties_after_relocate = tensor_sha256(model.properties[: model.size])

    sentinel_dead, sentinel_sibling = find_leaf_sibling_pair(model)
    sentinel_mask = torch.zeros(model.size, dtype=torch.bool, device=model.nodes.device)
    sentinel_mask[sentinel_dead] = True
    before_sentinel = model.nodes[: model.size].clone()
    started = time.perf_counter()
    sentinel_roots = model.relocate_gs(
        sentinel_mask,
        model.size,
        storage_device="cpu",
        densification="classic",
    )
    sentinel_seconds = time.perf_counter() - started
    sentinel_local = checked_report(model, sentinel_roots)
    started = time.perf_counter()
    sentinel_global = checked_report(model)
    sentinel_global_validation_seconds = time.perf_counter() - started
    sentinel_changed = changed_rows(before_sentinel, model.nodes[: model.size])
    skybox_unchanged = torch.equal(initial_skybox, model.nodes[: model.skybox_points])
    if not skybox_unchanged:
        raise RuntimeError("real mutation integration changed a skybox row")

    output = {
        "gate": "PASS",
        "source": str(source),
        "hierarchy": str(args.hierarchy.resolve()),
        "scaffold": str(args.scaffold.resolve()),
        "seed": args.seed,
        "cuda_device": torch.cuda.get_device_name(0),
        "initial_topology": initial,
        "initial_size": initial_size,
        "add_new_gs": add_results,
        "relocate_gs": {
            "dead_nodes": [deep_dead],
            "deep_interior_sibling": deep_sibling,
            "deep_interior_descendant_levels": descendant_levels,
            "affected_roots": affected_roots.detach().cpu().tolist(),
            "changed_rows": relocate_changed,
            "seconds": relocate_seconds,
            "local_topology": local_after_relocate,
            "global_topology": global_after_relocate,
            "global_validation_seconds": global_validation_seconds,
            "properties_before_sha256": properties_before,
            "properties_after_sha256": properties_after_relocate,
        },
        "sentinel_relocate_gs": {
            "dead_node": sentinel_dead,
            "surviving_leaf_sibling": sentinel_sibling,
            "affected_roots": sentinel_roots.detach().cpu().tolist(),
            "changed_rows": sentinel_changed,
            "seconds": sentinel_seconds,
            "local_topology": sentinel_local,
            "global_topology": sentinel_global,
            "global_validation_seconds": sentinel_global_validation_seconds,
        },
        "skybox_rows_unchanged": skybox_unchanged,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
