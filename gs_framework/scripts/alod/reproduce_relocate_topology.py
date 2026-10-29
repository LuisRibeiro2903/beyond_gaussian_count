#!/usr/bin/env python3
"""Deterministic synthetic reproducer for official ALoD ``relocate_gs``.

The official method is loaded directly from a requested source checkout.  Its
CUDA-only property relocation helper and stochastic sampler are replaced with
deterministic CPU test doubles; the topology-mutating method body is not
copied or altered.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from topology_invariants import changed_rows, validate_topology


SKYBOX_ROWS = 1
PROPERTY_COLUMNS = 30
NUMBER_PROPERTIES = 23


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def install_import_stubs() -> None:
    def package(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
        return module

    def module(name: str, **attributes: Any) -> types.ModuleType:
        result = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(result, key, value)
        sys.modules[name] = result
        return result

    noop: Callable[..., Any] = lambda *args, **kwargs: None
    package("utils")
    module(
        "utils.general_utils",
        inverse_sigmoid=noop,
        get_expon_lr_func=noop,
        build_rotation=noop,
        strip_symmetric=noop,
        build_scaling_rotation=noop,
    )
    module("utils.system_utils", mkdir_p=noop)
    module("utils.sh_utils", RGB2SH=noop)
    module("utils.graphics_utils", BasicPointCloud=object)
    module("utils.reloc_utils", compute_relocation_cuda=noop)

    package("simple_knn")
    module("simple_knn._C", distCUDA2=noop)
    package("gaussian_hierarchy")
    module(
        "gaussian_hierarchy._C",
        load_hierarchy=noop,
        write_hierarchy=noop,
        load_dynamic_hierarchy=noop,
        write_dynamic_hierarchy=noop,
        get_morton_indices=noop,
        get_spt_cut_cuda=noop,
    )
    package("scene")
    module("scene.OurAdam", Adam=object)
    module("gaussian_renderer", occlusion_cull=noop)


def load_gaussian_model(source: Path) -> tuple[type, types.ModuleType]:
    install_import_stubs()
    path = source / "scene" / "gaussian_model.py"
    name = f"alod_gaussian_model_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded.GaussianModel, loaded


def build_nodes(size: int, children: dict[int, list[int]]) -> torch.Tensor:
    nodes = torch.zeros((size, 6), dtype=torch.int32)
    nodes[:SKYBOX_ROWS] = -99
    nodes[:SKYBOX_ROWS, 5] = 0
    hierarchy = set(range(SKYBOX_ROWS, size))
    owned: set[int] = set()
    for parent, child_ids in children.items():
        if len(child_ids) != 2:
            raise ValueError("synthetic cases are binary")
        if parent not in hierarchy or any(child not in hierarchy for child in child_ids):
            raise ValueError("synthetic child index outside hierarchy")
        nodes[parent, 2] = len(child_ids)
        nodes[parent, 3] = child_ids[0]
        for ordinal, child in enumerate(child_ids):
            if child in owned:
                raise ValueError(f"duplicate synthetic child {child}")
            owned.add(child)
            nodes[child, 1] = parent
            nodes[child, 4] = child_ids[ordinal + 1] if ordinal + 1 < len(child_ids) else 0

    roots = sorted(hierarchy - owned)
    if len(roots) != 1:
        raise ValueError(f"synthetic hierarchy has roots {roots}")
    root = roots[0]
    nodes[root, 1] = -1
    stack = [(root, 0)]
    visited: set[int] = set()
    while stack:
        index, depth = stack.pop()
        if index in visited:
            raise ValueError("synthetic hierarchy cycle")
        visited.add(index)
        nodes[index, 0] = depth
        child = int(nodes[index, 3])
        child_ids: list[int] = []
        for _ in range(int(nodes[index, 2])):
            child_ids.append(child)
            child = int(nodes[child, 4])
        stack.extend((child_index, depth + 1) for child_index in reversed(child_ids))
    if visited != hierarchy:
        raise ValueError(f"synthetic hierarchy misses {sorted(hierarchy - visited)}")
    return nodes


def synthetic_cases() -> dict[str, dict[str, Any]]:
    heap_children = {index: [index * 2, index * 2 + 1] for index in range(1, 50)}
    return {
        "A_leaf_sibling": {
            "size": 6,
            "children": {1: [2, 3], 2: [4, 5]},
            "dead": [4],
            "reinit": [3],
        },
        "A2_leaf_sibling_real_capacity": {
            "size": 100,
            "capacity_padding": 1,
            "children": heap_children,
            "dead": [98],
            "reinit": [97],
        },
        "B_one_level_interior_sibling": {
            "size": 8,
            "children": {1: [2, 3], 2: [4, 5], 5: [6, 7]},
            "dead": [4],
            "reinit": [3],
        },
        "C_deep_interior_sibling": {
            "size": 12,
            "children": {1: [2, 3], 2: [4, 5], 5: [6, 7], 6: [8, 9], 8: [10, 11]},
            "dead": [4],
            "reinit": [3],
        },
        "D_multiple_independent": {
            "size": 12,
            "children": {1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9], 6: [10, 11]},
            "dead": [8, 10],
            "reinit": [5, 7],
        },
        "E_interacting_ancestor_descendant": {
            "size": 10,
            "children": {1: [2, 3], 2: [4, 5], 5: [6, 7], 6: [8, 9]},
            "dead": [4, 8],
            "reinit": [3, 7],
        },
        "F_respawn_target": {
            "size": 6,
            "children": {1: [2, 3], 2: [4, 5]},
            "dead": [4],
            "reinit": [3],
            "verify_respawn": True,
        },
    }


def make_model(
    model_class: type,
    nodes: torch.Tensor,
    reinit: list[int],
    capacity_padding: int = 100,
) -> Any:
    model = model_class.__new__(model_class)
    # The real fine model keeps a cap-sized backing tensor.  Extra capacity is
    # essential to reproduce Python's valid negative-index write for the
    # official ``-99`` skybox sentinel on small synthetic hierarchies.
    capacity = len(nodes) + capacity_padding
    model.nodes = torch.zeros((capacity, nodes.shape[1]), dtype=nodes.dtype)
    model.nodes[: len(nodes)] = nodes
    model.size = len(nodes)
    model.skybox_points = SKYBOX_ROWS
    base = torch.arange(capacity * PROPERTY_COLUMNS, dtype=torch.float32)
    model.properties = base.reshape(capacity, PROPERTY_COLUMNS) / 1000.0
    model._densification_criterium = torch.zeros(capacity, dtype=torch.float32)
    model.opacity_activation = lambda values: torch.ones_like(values)
    selected = torch.tensor(reinit, dtype=torch.int64)
    model._synthetic_selected_reinit = selected.clone()

    def sample_alives(
        self: Any,
        probs: torch.Tensor,
        num: int,
        alive_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if alive_indices is None:
            raise AssertionError("official relocate_gs did not pass alive_indices")
        if not torch.isin(selected, alive_indices).all():
            raise AssertionError(
                f"requested reinit {selected.tolist()} not in alive leaves {alive_indices.tolist()}"
            )
        ratio = torch.zeros(self.size, dtype=torch.int64)
        ratio[selected] = 1
        return selected.clone(), ratio

    def update_params(
        self: Any, idxs: torch.Tensor, ratio: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        del ratio
        return (
            self.properties[idxs, 0:3].clone(),
            self.properties[idxs, 10:13].clone(),
            self.properties[idxs, 14:NUMBER_PROPERTIES].clone(),
            self.properties[idxs, 13:14].clone(),
            self.properties[idxs, 3:6].clone(),
            self.properties[idxs, 6:10].clone(),
        )

    model._sample_alives = types.MethodType(sample_alives, model)
    model._update_params = types.MethodType(update_params, model)
    return model


def sibling_ids(nodes: torch.Tensor, dead: list[int]) -> list[int]:
    result: list[int] = []
    for dead_index in dead:
        next_sibling = int(nodes[dead_index, 4])
        if next_sibling > 0:
            result.append(next_sibling)
        else:
            parent = int(nodes[dead_index, 1])
            result.append(int(nodes[parent, 3]))
    return result


def run_case(model_class: type, name: str, case: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(0)
    nodes = build_nodes(case["size"], case["children"])
    before_report = validate_topology(
        nodes, active_size=len(nodes), skybox_rows=SKYBOX_ROWS, root=SKYBOX_ROWS
    )
    if not before_report.valid:
        raise AssertionError(f"{name} invalid before mutation: {before_report.as_dict()}")
    model = make_model(
        model_class,
        nodes,
        case["reinit"],
        capacity_padding=case.get("capacity_padding", 100),
    )
    before_nodes = model.nodes.clone()
    before_properties = model.properties.clone()
    siblings = sibling_ids(before_nodes, case["dead"])
    affected_roots = sorted(
        set(int(before_nodes[index, 1]) for index in case["dead"])
        | set(case["reinit"])
    )
    dead_mask = torch.zeros(len(nodes), dtype=torch.bool)
    dead_mask[case["dead"]] = True

    model.relocate_gs(
        dead_mask,
        len(nodes),
        storage_device="cpu",
        densification="classic",
    )

    after_report = validate_topology(
        model.nodes,
        active_size=len(nodes),
        skybox_rows=SKYBOX_ROWS,
        root=SKYBOX_ROWS,
    )
    affected_report = validate_topology(
        model.nodes,
        active_size=len(nodes),
        skybox_rows=SKYBOX_ROWS,
        subtree_roots=affected_roots,
    )
    skybox_unchanged = torch.equal(
        before_nodes[:SKYBOX_ROWS], model.nodes[:SKYBOX_ROWS]
    )
    property_change_mask = torch.any(before_properties != model.properties, dim=1)
    property_changed_rows = torch.nonzero(property_change_mask, as_tuple=True)[0].tolist()
    forward_columns = model.nodes[: len(nodes), 2:5]

    respawn_checks: dict[str, Any] | None = None
    if case.get("verify_respawn"):
        target = case["reinit"][0]
        first = int(model.nodes[target, 3])
        second = int(model.nodes[first, 4]) if first else 0
        respawn_checks = {
            "target": target,
            "target_became_binary_interior": int(model.nodes[target, 2]) == 2,
            "children": [first, second],
            "expected_children": [case["dead"][0], siblings[0]],
            "children_point_back_to_target": (
                int(model.nodes[first, 1]) == target
                and int(model.nodes[second, 1]) == target
            ),
        }

    return {
        "case": name,
        "dead_nodes": case["dead"],
        "sibling_nodes": siblings,
        "requested_respawn_targets": case["reinit"],
        "before": before_report.as_dict(),
        "after": after_report.as_dict(),
        "affected_subtrees_after": affected_report.as_dict(),
        "affected_roots": affected_roots,
        "skybox_unchanged": skybox_unchanged,
        "before_node_table": before_nodes.tolist(),
        "after_node_table": model.nodes.tolist(),
        "changed_node_rows": changed_rows(before_nodes, model.nodes),
        "changed_active_node_rows": changed_rows(
            before_nodes, model.nodes, active_size=len(nodes)
        ),
        "property_changed_rows": property_changed_rows,
        "properties_after_sha256": tensor_sha256(model.properties),
        "forward_topology_after_sha256": tensor_sha256(forward_columns),
        "respawn_checks": respawn_checks,
    }


def run_add_new_case(model_class: type) -> dict[str, Any]:
    nodes = build_nodes(6, {1: [2, 3], 2: [4, 5]})
    model = make_model(model_class, nodes, [3], capacity_padding=10)
    model._densification_criterium[:] = 0
    model._densification_criterium[3] = 1
    model.densification_postfix_with_storage = types.MethodType(
        lambda self, *args, **kwargs: None, model
    )
    before = model.nodes.clone()
    returned = model.add_new_gs(
        cap_max=8,
        size=6,
        densification="classic",
        densify_percent=2.0,
        densify_threshold=0.5,
    )
    report = validate_topology(
        model.nodes,
        active_size=model.size,
        skybox_rows=SKYBOX_ROWS,
        root=SKYBOX_ROWS,
    )
    return {
        "valid": report.valid,
        "report": report.as_dict(),
        "returned_num_gs": int(returned),
        "active_size": int(model.size),
        "changed_node_rows": changed_rows(before, model.nodes),
        "forward_topology_after_sha256": tensor_sha256(model.nodes[: model.size, 2:5]),
        "nodes_after_sha256": tensor_sha256(model.nodes),
    }


def run_serialization_passthrough(
    model_class: type, loaded: types.ModuleType
) -> dict[str, Any]:
    nodes = build_nodes(8, {1: [2, 3], 2: [4, 5], 3: [6, 7]})
    model = make_model(model_class, nodes, [4], capacity_padding=4)
    model.max_sh_degree = 1
    captured: dict[str, Any] = {}

    def capture_writer(
        path: str,
        positions: torch.Tensor,
        shs: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        serialized_nodes: torch.Tensor,
        degree: int,
    ) -> None:
        del positions, shs, opacities, scales, rotations
        captured["path"] = path
        captured["nodes"] = serialized_nodes.clone()
        captured["degree"] = degree

    loaded.write_dynamic_hierarchy = capture_writer
    model.save_hierarchy("/synthetic-not-written", file_name="capture.dhier")
    passed_nodes = captured["nodes"]
    expected_nodes = model.nodes[: model.size]
    return {
        "nodes_passed_without_transform": torch.equal(passed_nodes, expected_nodes),
        "input_nodes_sha256": tensor_sha256(expected_nodes),
        "writer_nodes_sha256": tensor_sha256(passed_nodes),
        "writer_degree": captured["degree"],
        "writer_path": captured["path"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--expect",
        choices=("official-failure", "patched-pass"),
        required=True,
    )
    args = parser.parse_args()

    source = args.source.resolve()
    model_class, loaded = load_gaussian_model(source)
    results = [
        run_case(model_class, name, case)
        for name, case in synthetic_cases().items()
    ]
    failing = [
        result["case"]
        for result in results
        if not result["after"]["valid"] or not result["skybox_unchanged"]
    ]
    correct = [result["case"] for result in results if result["after"]["valid"]]
    respawn = next(result for result in results if result["case"] == "F_respawn_target")
    respawn_valid = bool(
        respawn["respawn_checks"]["target_became_binary_interior"]
        and respawn["respawn_checks"]["children"]
        == respawn["respawn_checks"]["expected_children"]
        and respawn["respawn_checks"]["children_point_back_to_target"]
    )
    add_new_result = run_add_new_case(model_class)
    serialization_result = run_serialization_passthrough(model_class, loaded)

    if args.expect == "official-failure":
        expectation_met = (
            bool(failing)
            and "A2_leaf_sibling_real_capacity" in failing
            and "C_deep_interior_sibling" in failing
            and add_new_result["valid"]
            and serialization_result["nodes_passed_without_transform"]
        )
    else:
        expectation_met = (
            not failing
            and respawn_valid
            and add_new_result["valid"]
            and serialization_result["nodes_passed_without_transform"]
        )
    report = {
        "gate": "PASS" if expectation_met else "FAIL",
        "expectation": args.expect,
        "source": str(source),
        "source_commit": _git_head(source),
        "loaded_method_file": str(Path(loaded.__file__).resolve()),
        "torch_version": torch.__version__,
        "cuda_initialized": torch.cuda.is_initialized(),
        "cases": results,
        "failing_cases": failing,
        "topologically_correct_cases": correct,
        "respawn_target_behavior_valid": respawn_valid,
        "add_new_gs": add_new_result,
        "serialization_passthrough": serialization_result,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    for result in results:
        print(
            f"{result['case']}: topology={result['after']['valid']} "
            f"skybox_unchanged={result['skybox_unchanged']} "
            f"mismatches={result['after']['mismatch_counts']}"
        )
    return 0 if expectation_met else 2


def _git_head(source: Path) -> str | None:
    head_file = source / ".git"
    if head_file.is_dir():
        head_file = head_file / "HEAD"
    if head_file.is_file():
        text = head_file.read_text().strip()
        if text.startswith("gitdir: "):
            gitdir = Path(text.removeprefix("gitdir: "))
            if not gitdir.is_absolute():
                gitdir = (source / gitdir).resolve()
            head_file = gitdir / "HEAD"
    if not head_file.exists():
        return None
    head = head_file.read_text().strip()
    if head.startswith("ref: "):
        ref = head_file.parent / head.removeprefix("ref: ")
        return ref.read_text().strip() if ref.exists() else head
    return head


if __name__ == "__main__":
    raise SystemExit(main())
