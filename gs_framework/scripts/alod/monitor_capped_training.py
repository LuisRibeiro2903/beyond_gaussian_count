#!/usr/bin/env python3
"""Narrow validator and process-group monitor for ALOD-SMOKE-TRAIN-001.

Validation modes are CPU-only unless ``preflight-cache`` is explicitly selected.
The latter is a future execution-stage CUDA preflight and is never used by this
planning task. This file does not patch or replace official training functions.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import runpy
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


GIB = 1024**3
MIB = 1024**2


def parse_log_progress(
    log_text: str,
    latest_iteration: int | None,
    latest_node_count: int | None,
) -> tuple[int | None, int | None, bool]:
    """Extract monotonic training/creator progress from newly appended output.

    The official coarse trainer reports progress through tqdm using carriage
    returns (for example ``Training progress: ... | 27070/30000 [...]``).
    Treat a changed iteration or node count as progress; repeated status lines
    must not keep the no-progress watchdog alive on their own.
    """
    progress_observed = False
    iteration_patterns = (
        r"(?:\[ITER\s+|iteration[_ :=]|Fine Training[^\r\n]*?)([0-9][0-9_]*)",
        r"Training progress:[^\r\n]*?\|\s*([0-9][0-9_]*)/([0-9][0-9_]*)\s*\[",
    )
    for pattern in iteration_patterns:
        for match in re.finditer(pattern, log_text):
            candidate = int(match.group(1).replace("_", ""))
            if len(match.groups()) > 1 and match.group(2) is not None:
                total = int(match.group(2).replace("_", ""))
                if not 0 <= candidate <= total:
                    continue
            if candidate != latest_iteration:
                latest_iteration = candidate
                progress_observed = True

    for match in re.finditer(
        r"Size(?:[\"']|\x1b\[[0-9;]*m|[ :=])*([0-9][0-9_]*)/([0-9][0-9_]*)",
        log_text,
    ):
        candidate = int(match.group(1).replace("_", ""))
        if candidate != latest_node_count:
            latest_node_count = candidate
            progress_observed = True

    return latest_iteration, latest_node_count, progress_observed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_strict(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"top-level JSON value must be an object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def optimization_keys(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "OptimizationParams":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                    keys: set[str] = set()
                    for item in ast.walk(child):
                        targets: list[ast.expr] = []
                        if isinstance(item, ast.Assign):
                            targets.extend(item.targets)
                        elif isinstance(item, ast.AnnAssign):
                            targets.append(item.target)
                        for target in targets:
                            if (
                                isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                            ):
                                keys.add(target.attr)
                    return keys
    raise ValueError(f"OptimizationParams.__init__ not found in {source}")


def required_violation_samples(duration_seconds: int | float, sampling_seconds: int | float) -> int:
    return max(1, math.ceil(duration_seconds / sampling_seconds))


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    sampling = policy.get("sampling_interval_seconds")
    if sampling != 5 or policy.get("sampling_interval_unit") != "seconds":
        raise ValueError("canonical policy requires a five-second sampling interval")
    semantics = policy.get("duration_semantics", {})
    if semantics.get("mode") != "consecutive-violating-samples":
        raise ValueError("policy must use consecutive-violating-samples duration semantics")
    if semantics.get("first_violating_sample_counts_as") != 1:
        raise ValueError("first violating sample must count as sample one")
    if semantics.get("rounding") != "ceiling":
        raise ValueError("duration-to-sample conversion must use ceiling")

    expected = {
        ("process_group_rss", "warning"): 20 * GIB,
        ("process_group_rss", "hard"): 24 * GIB,
        ("available_system_ram", "warning"): 8 * GIB,
        ("available_system_ram", "hard"): 5 * GIB,
        ("swap_growth_from_stage_baseline", "warning"): 128 * MIB,
        ("swap_growth_from_stage_baseline", "hard"): 256 * MIB,
        ("whole_device_vram", "warning"): 12 * GIB,
        ("whole_device_vram", "hard"): int(13.5 * GIB),
        ("training_process_vram", "warning"): int(11.5 * GIB),
        ("training_process_vram", "hard"): 13 * GIB,
        ("output_directory", "warning"): 2 * GIB,
        ("output_directory", "hard"): 5 * GIB,
    }
    sample_counts: dict[str, int | None] = {}
    for threshold_name, threshold in policy.get("thresholds", {}).items():
        if threshold_name not in {"pytorch_allocated", "pytorch_reserved"}:
            if threshold.get("comparison") not in {
                "greater-than", "less-than", "greater-than-or-equal"
            }:
                raise ValueError(f"missing/invalid comparison: {threshold_name}")
        for level in ("warning", "hard"):
            record = threshold.get(level, {})
            if not record.get("unit"):
                raise ValueError(f"missing unit: {threshold_name}.{level}")
            if not record.get("enforcement"):
                raise ValueError(f"missing enforcement: {threshold_name}.{level}")
            duration = record.get("duration_seconds")
            value_keys = [key for key in ("value_bytes", "value_rows", "value_seconds") if key in record]
            if len(value_keys) != 1:
                raise ValueError(f"exactly one typed value is required: {threshold_name}.{level}")
            value = record[value_keys[0]]
            if value is None:
                if duration is not None or record["enforcement"] != "diagnostic-if-emitted":
                    raise ValueError(f"nullable threshold must be diagnostic-if-emitted: {threshold_name}.{level}")
                sample_counts[f"{threshold_name}.{level}"] = None
            else:
                if duration is None or duration < 0:
                    raise ValueError(f"invalid duration: {threshold_name}.{level}")
                sample_counts[f"{threshold_name}.{level}"] = required_violation_samples(duration, sampling)
    for (name, level), value in expected.items():
        if policy["thresholds"][name][level]["value_bytes"] != value:
            raise ValueError(f"stale canonical value: {name}.{level}")
    duration_expected = {
        ("process_group_rss", "hard"): 15,
        ("available_system_ram", "hard"): 15,
        ("swap_growth_from_stage_baseline", "hard"): 15,
        ("whole_device_vram", "hard"): 10,
        ("training_process_vram", "hard"): 10,
    }
    for (name, level), duration in duration_expected.items():
        if policy["thresholds"][name][level]["duration_seconds"] != duration:
            raise ValueError(f"stale canonical duration: {name}.{level}")
    scalar_expected = {
        ("fine_node_count", "warning", "value_rows"): 1050000,
        ("fine_node_count", "hard", "value_rows"): 1100000,
        ("no_progress", "warning", "value_seconds"): 900,
        ("no_progress", "hard", "value_seconds"): 1800,
    }
    for (name, level, field), value in scalar_expected.items():
        if policy["thresholds"][name][level][field] != value:
            raise ValueError(f"stale canonical value: {name}.{level}.{field}")
    stage_expected = {
        "coarse": (5400, 10800),
        "creator": (1200, 3600),
        "fine": (14400, 36000),
    }
    for stage, (warning, hard) in stage_expected.items():
        record = policy["stage_wall_time"][stage]
        if (record["warning_seconds"], record["hard_seconds"]) != (warning, hard):
            raise ValueError(f"stale canonical wall time: {stage}")
    if policy["stage_wall_time"]["whole_pipeline"]["hard_seconds"] != 43200:
        raise ValueError("stale canonical wall time: whole_pipeline")
    enforcement_expected = {
        ("process_group_rss", "hard"): "hard-enforced",
        ("available_system_ram", "hard"): "hard-enforced",
        ("swap_growth_from_stage_baseline", "hard"): "hard-enforced",
        ("whole_device_vram", "hard"): "hard-enforced",
        ("training_process_vram", "hard"): "hard-enforced-when-attribution-reliable",
        ("output_directory", "hard"): "hard-enforced",
        ("fine_node_count", "hard"): "configuration-enforced",
    }
    for (name, level), enforcement in enforcement_expected.items():
        if policy["thresholds"][name][level]["enforcement"] != enforcement:
            raise ValueError(f"stale enforcement classification: {name}.{level}")
    for name in ("pytorch_allocated", "pytorch_reserved"):
        for level in ("warning", "hard"):
            if policy["thresholds"][name][level]["value_bytes"] is not None:
                raise ValueError(f"PyTorch threshold must remain null: {name}.{level}")
    process_policy = policy["thresholds"]["training_process_vram"]
    if process_policy.get("unavailable_behavior") != "record null; never substitute whole-device memory":
        raise ValueError("process VRAM unavailable behavior must prohibit substitution")
    monotonic = policy["thresholds"]["swap_growth_from_stage_baseline"].get(
        "persistent_monotonic_growth", {}
    )
    if monotonic.get("consecutive_increase_samples") != 6:
        raise ValueError("persistent swap rule must use six increasing samples")
    if monotonic.get("minimum_total_growth_bytes") != 128 * MIB:
        raise ValueError("persistent swap rule must start at the warning growth threshold")
    if not any(
        item.get("status") == "BLOCKED-ALOD-PREFLIGHT"
        for item in policy.get("planning_history", [])
    ):
        raise ValueError("policy lost prior BLOCKED-ALOD-PREFLIGHT history")
    return {
        "sampling_interval_seconds": sampling,
        "duration_semantics": "first violating sample counts as one; ceiling conversion",
        "required_samples": sample_counts,
        "canonical_thresholds": {
            "process_group_rss_bytes": {"warning": 20 * GIB, "hard": 24 * GIB},
            "available_system_ram_bytes": {"warning_below": 8 * GIB, "hard_below": 5 * GIB},
            "swap_growth_bytes": {"warning": 128 * MIB, "hard": 256 * MIB},
            "whole_device_vram_bytes": {"warning": 12 * GIB, "hard": int(13.5 * GIB)},
            "training_process_vram_bytes": {"warning": int(11.5 * GIB), "hard": 13 * GIB},
            "pytorch_allocated_bytes": {"warning": None, "hard": None},
            "pytorch_reserved_bytes": {"warning": None, "hard": None},
            "fine_node_count_rows": {"warning": 1050000, "hard_configuration_cap": 1100000},
            "output_directory_bytes": {"warning": 2 * GIB, "hard": 5 * GIB},
            "no_progress_seconds": {"warning": 900, "hard": 1800},
        },
        "persistent_swap_rule": {
            "consecutive_increase_samples": 6,
            "minimum_total_growth_bytes": 128 * MIB,
        },
        "stage_wall_time_seconds": {
            stage: {"warning": warning, "hard": hard}
            for stage, (warning, hard) in stage_expected.items()
        } | {"whole_pipeline": {"warning": None, "hard": 43200}},
    }


def validate_plan(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    config_path = args.config.resolve()
    resolved_path = args.resolved.resolve()
    policy_path = args.policy.resolve()
    accepted = optimization_keys(source / "arguments" / "__init__.py")
    capped = load_json_strict(config_path)
    resolved = load_json_strict(resolved_path)
    policy = load_json_strict(policy_path)
    policy_validation = validate_policy(policy)

    unknown = sorted(set(capped) - accepted)
    if unknown:
        raise ValueError(f"unknown optimization JSON keys: {unknown}")
    resolved_opt = resolved["effective_parameters"]["optimization"]
    missing_resolved = sorted(accepted - set(resolved_opt))
    extra_resolved = sorted(set(resolved_opt) - accepted)
    if missing_resolved or extra_resolved:
        raise ValueError(
            f"resolved optimization coverage mismatch; missing={missing_resolved}, extra={extra_resolved}"
        )
    mismatches = {
        key: {"json": value, "resolved": resolved_opt[key]["value"]}
        for key, value in capped.items()
        if resolved_opt[key]["value"] != value
    }
    if mismatches:
        raise ValueError(f"capped/resolved value mismatch: {mismatches}")

    expected = {
        "iterations": 12000,
        "coarse_iterations": 5000,
        "SH_degree": 1,
        "cap_max": 1100000,
        "cache_size": 300000,
        "cache_size_after_reduction": 240000,
        "llff_hold": 50,
        "SPT_root_volume": 150,
        "target_granularity_pixels": 2,
        "min_SPT_size": 256,
    }
    for key, value in expected.items():
        if capped.get(key) != value:
            raise ValueError(f"required capped value {key}={value!r}, got {capped.get(key)!r}")
    model = resolved["effective_parameters"]["model"]
    if model["skybox_num"]["value"] != 100000:
        raise ValueError("source-faithful run requires exactly 100000 skybox rows")
    if model["resolution"]["value"] != 2:
        raise ValueError("resolution factor must be 2")
    if resolved["count_model"]["fine_initial_total_rows_max"] != 987239:
        raise ValueError("revised count model is not resolved")
    if resolved["cache_feasibility"]["classification"] != "CONDITIONAL-EXECUTION-PREFLIGHT":
        raise ValueError("cache feasibility must remain conditional")

    omitted = set(resolved["json_validation"]["omitted_parsed_unused_keys"])
    accidentally_present = sorted(omitted & set(capped))
    if accidentally_present:
        raise ValueError(f"parsed-unused/overwritten keys must be omitted: {accidentally_present}")

    policy_hash = sha256_file(policy_path)
    if resolved["paths"].get("resource_policy_sha256") != policy_hash:
        raise ValueError("resolved manifest resource-policy hash is stale")
    summary = resolved.get("resource_policy", {})
    summary_expected = {
        "sampling_interval_seconds": 5,
        "hard_duration_sample_counts": {"15_seconds": 3, "10_seconds": 2},
        "process_group_rss_bytes": {
            "warning": 20 * GIB, "hard": 24 * GIB, "hard_duration_seconds": 15,
        },
        "available_system_ram_bytes": {
            "warning_below": 8 * GIB, "hard_below": 5 * GIB, "hard_duration_seconds": 15,
        },
        "whole_device_vram_bytes": {
            "warning": 12 * GIB, "hard": int(13.5 * GIB), "hard_duration_seconds": 10,
        },
        "fine_node_count_rows": {
            "warning": 1050000, "hard_configuration_cap": 1100000,
        },
        "output_directory_bytes": {"warning": 2 * GIB, "hard": 5 * GIB},
        "no_progress_seconds": {"warning": 900, "hard": 1800},
        "execution_authorized": False,
    }
    for key, expected_value in summary_expected.items():
        if summary.get(key) != expected_value:
            raise ValueError(f"resolved resource-policy summary is stale: {key}")
    swap_summary = summary.get("swap_growth_bytes", {})
    if (swap_summary.get("warning"), swap_summary.get("hard"),
            swap_summary.get("hard_duration_seconds")) != (128 * MIB, 256 * MIB, 15):
        raise ValueError("resolved resource-policy summary is stale: swap_growth_bytes")
    if "six strictly increasing samples" not in swap_summary.get("persistent_monotonic_abort", ""):
        raise ValueError("resolved persistent swap rule is stale")
    process_summary = summary.get("training_process_vram_bytes", {})
    if (process_summary.get("warning"), process_summary.get("hard"),
            process_summary.get("hard_duration_seconds")) != (
                int(11.5 * GIB), 13 * GIB, 10
            ):
        raise ValueError("resolved resource-policy summary is stale: training_process_vram_bytes")
    if process_summary.get("unavailable_behavior") != (
        "record null; never substitute whole-device memory"
    ):
        raise ValueError("resolved process-VRAM unavailable behavior is stale")
    for name in ("pytorch_allocated_bytes", "pytorch_reserved_bytes"):
        if summary.get(name) != {
            "warning": None, "hard": None, "enforcement": "diagnostic-if-emitted",
        }:
            raise ValueError(f"resolved resource-policy summary is stale: {name}")
    if summary.get("wall_time_seconds") != {
        "coarse": {"warning": 5400, "hard": 10800},
        "creator": {"warning": 1200, "hard": 3600},
        "fine": {"warning": 14400, "hard": 36000},
        "whole_pipeline_hard": 43200,
    }:
        raise ValueError("resolved resource-policy summary is stale: wall_time_seconds")
    if not any(
        item.get("status") == "BLOCKED-ALOD-PREFLIGHT"
        for item in resolved.get("planning_history", [])
    ):
        raise ValueError("resolved manifest lost prior BLOCKED-ALOD-PREFLIGHT history")
    if args.provenance:
        provenance = load_json_strict(args.provenance.resolve())
        if provenance["configuration"].get("resource_policy_sha256") != policy_hash:
            raise ValueError("provenance resource-policy hash is stale")
        if provenance["configuration"].get("resolved_manifest_sha256") != sha256_file(resolved_path):
            raise ValueError("provenance resolved-manifest hash is stale")
        if args.command and provenance["command"].get("script_sha256") != sha256_file(args.command.resolve()):
            raise ValueError("provenance command hash is stale")
        if args.monitor and provenance["monitor"].get("sha256") != sha256_file(args.monitor.resolve()):
            raise ValueError("provenance monitor hash is stale")
        if not any(
            item.get("status") == "BLOCKED-ALOD-PREFLIGHT"
            and item.get("training_started") is False
            for item in provenance.get("failure_and_restart_history", [])
        ):
            raise ValueError("provenance lost prior BLOCKED-ALOD-PREFLIGHT history")
    if args.command and policy_hash not in args.command.read_text(encoding="utf-8"):
        raise ValueError("command does not pin the current resource-policy hash")

    result = {
        "gate": "PASS",
        "source": str(source),
        "accepted_optimization_keys": sorted(accepted),
        "accepted_optimization_key_count": len(accepted),
        "capped_keys": sorted(capped),
        "capped_key_count": len(capped),
        "unknown_keys": [],
        "duplicate_keys": [],
        "resolved_field_coverage": "complete",
        "capped_sha256": sha256_file(config_path),
        "resolved_sha256": sha256_file(resolved_path),
        "policy_sha256": policy_hash,
        "policy_validation": policy_validation,
        "training_started": False,
        "cuda_initialized": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def validate_launcher(_: argparse.Namespace) -> int:
    """Exercise the launcher's pop-and-runpy argv forwarding without safe_state/CUDA."""
    payload = "import sys\nFORWARDED_ARGV = list(sys.argv)\n"
    original = list(sys.argv)
    with tempfile.TemporaryDirectory(prefix="alod-launcher-check-") as directory:
        target = Path(directory) / "payload.py"
        target.write_text(payload, encoding="utf-8")
        try:
            sys.argv = ["-c", str(target), "--alpha", "value with spaces", "--flag"]
            entry = sys.argv.pop(1)
            result = runpy.run_path(entry, run_name="__main__")
        finally:
            sys.argv = original
    forwarded = result["FORWARDED_ARGV"]
    expected = [str(target), "--alpha", "value with spaces", "--flag"]
    if forwarded != expected:
        raise ValueError(f"runpy argv forwarding failed: {forwarded!r} != {expected!r}")
    print(json.dumps({"gate": "PASS", "forwarded_argv": forwarded, "safe_state_called": False,
                      "cuda_initialized": False}, indent=2))
    return 0


def inspect_ply(args: argparse.Namespace) -> int:
    try:
        from plyfile import PlyData
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("inspect-ply requires plyfile and numpy from alod_official") from exc
    ply = PlyData.read(str(args.path))
    if len(ply.elements) != 1 or ply.elements[0].name != "vertex":
        raise ValueError("creator input must contain exactly one vertex element")
    vertex = ply.elements[0].data
    required = [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
        *[f"f_rest_{index}" for index in range(9)], "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    names = list(vertex.dtype.names or ())
    if names != required:
        raise ValueError(f"coarse PLY property layout mismatch: {names}")
    rows = len(vertex)
    if not (args.min_rows <= rows <= args.max_rows):
        raise ValueError(f"coarse PLY rows {rows} outside [{args.min_rows}, {args.max_rows}]")
    for name in required:
        if not np.isfinite(vertex[name]).all():
            raise ValueError(f"non-finite PLY property: {name}")
    pc_info = args.path.parent / "pc_info.txt"
    skybox = int(pc_info.read_text(encoding="utf-8").strip())
    if skybox != args.skybox_rows:
        raise ValueError(f"pc_info skybox rows {skybox} != {args.skybox_rows}")
    report = {
        "gate": "PASS", "path": str(args.path), "sha256": sha256_file(args.path),
        "rows": rows, "skybox_rows": skybox, "properties": names, "all_finite": True,
    }
    if args.report:
        atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def inspect_dhier(args: argparse.Namespace) -> int:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("inspect-dhier requires numpy") from exc
    path = args.path
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError("truncated .dhier header")
        gaussians, degree = struct.unpack("<ii", header)
    if degree != args.expected_degree or degree not in (0, 1, 2, 3):
        raise ValueError(f"unexpected SH degree {degree}")
    sh_floats = 3 * (degree + 1) ** 2
    property_floats = 3 + 4 + 3 + 1 + sh_floats
    node_count_offset = 8 + gaussians * property_floats * 4
    if node_count_offset + 4 > size:
        raise ValueError("truncated .dhier property payload")
    with path.open("rb") as handle:
        handle.seek(node_count_offset)
        nodes_count = struct.unpack("<i", handle.read(4))[0]
    expected_size = node_count_offset + 4 + nodes_count * 6 * 4
    if expected_size != size:
        raise ValueError(f".dhier byte length mismatch: expected {expected_size}, got {size}")
    if gaussians != nodes_count:
        raise ValueError(f"Gaussian/node mismatch: {gaussians} != {nodes_count}")
    if not (args.min_rows <= gaussians <= args.max_rows):
        raise ValueError(f"rows {gaussians} outside [{args.min_rows}, {args.max_rows}]")
    properties = np.memmap(path, dtype="<f4", mode="r", offset=8,
                           shape=(gaussians, property_floats))
    if not np.isfinite(properties).all():
        raise ValueError("non-finite serialized hierarchy attribute")
    nodes = np.memmap(path, dtype="<i4", mode="r", offset=node_count_offset + 4,
                      shape=(nodes_count, 6))
    skybox = args.skybox_rows
    if skybox:
        # Fine loading prepends six -99 fields, then source-faithfully normalizes
        # the reserved sixth field to zero for every active row.  The five
        # topology fields must remain sentinel values; the reserved field must
        # remain zero.  Checking all six for -99 would reject an unmutated load.
        if not np.all(nodes[:skybox, :5] == -99):
            raise ValueError("prepended skybox topology metadata is not the expected -99 sentinel")
        if not np.all(nodes[:skybox, 5] == 0):
            raise ValueError("prepended skybox reserved metadata is not the expected zero value")
    root = skybox
    if root >= nodes_count or int(nodes[root, 1]) != -1:
        raise ValueError(f"expected hierarchy root at row {root}")
    visited: set[int] = set()
    stack = [root]
    while stack:
        index = stack.pop()
        if index in visited:
            raise ValueError(f"cycle or duplicate reachability at node {index}")
        if index < root or index >= nodes_count:
            raise ValueError(f"node index out of range: {index}")
        visited.add(index)
        child_count = int(nodes[index, 2])
        if child_count < 0:
            raise ValueError(f"negative child count at node {index}")
        if child_count == 0:
            continue
        child = int(nodes[index, 3])
        children: list[int] = []
        for child_index in range(child_count):
            if child < root or child >= nodes_count:
                raise ValueError(f"child index out of range at node {index}: {child}")
            children.append(child)
            if int(nodes[child, 1]) != index:
                raise ValueError(f"parent mismatch for child {child}")
            if int(nodes[child, 0]) != int(nodes[index, 0]) + 1:
                raise ValueError(f"depth mismatch for child {child}")
            sibling = int(nodes[child, 4])
            if child_index + 1 < child_count:
                if sibling <= 0:
                    raise ValueError(f"short sibling chain at node {index}")
                child = sibling
            elif sibling != 0:
                raise ValueError(f"unterminated sibling chain at node {index}")
        stack.extend(reversed(children))
    expected_reachable = nodes_count - skybox
    if len(visited) != expected_reachable:
        raise ValueError(f"unreachable hierarchy nodes: reached {len(visited)} of {expected_reachable}")
    total_rows_for_cap = gaussians + args.additional_rows
    headroom = args.cap_max - total_rows_for_cap
    ratio = headroom / args.cap_max
    if headroom < args.min_growth_rows or ratio < args.min_headroom_ratio:
        raise ValueError(f"fine headroom too small: {headroom} rows, {ratio:.6f}")
    report = {
        "gate": "PASS", "path": str(path), "sha256": sha256_file(path),
        "bytes": size, "gaussians": gaussians, "nodes": nodes_count,
        "sh_degree": degree, "skybox_rows": skybox, "reachable_non_skybox_nodes": len(visited),
        "all_attributes_finite": True, "tree_links_valid": True,
        "skybox_topology_sentinel_valid": True, "skybox_reserved_field_valid": True,
        "additional_rows_for_cap": args.additional_rows,
        "total_rows_for_cap": total_rows_for_cap,
        "growth_headroom_rows": headroom, "headroom_ratio_of_cap": ratio,
    }
    if args.report:
        atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def preflight_cache(args: argparse.Namespace) -> int:
    """Future CUDA preflight; rebuild the exact official HSPT but perform no training."""
    source = args.source.resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))
    from utils.general_utils import safe_state
    from scene.dataset_readers import readColmapSceneInfo
    from scene.gaussian_model import GaussianModel

    safe_state(False)
    info = readColmapSceneInfo(str(args.dataset), str(args.dataset / "images"), "", "",
                               True, False, args.llff_hold)
    model = GaussianModel(args.sh_degree)
    model.create_from_hier(str(args.hierarchy), info.nerf_normalization["radius"],
                           str(args.scaffold))
    initial_rows = len(model.nodes)
    model.compact_gaussians("cpu", args.cap_max, "classic", prune_unused_gaussians=False)
    focal = info.train_cameras[0].focal_length
    model.build_hierarchical_SPT(args.spt_root_volume, args.target_pixels / focal,
                                 args.min_spt_size, use_bounding_spheres=False)
    upper_rows = len(model.upper_tree_nodes)
    spt_sizes = (model.SPT_starts[1:] - model.SPT_starts[:-1]).detach().cpu()
    minimum_spt_rows = int(spt_sizes.min().item()) if len(spt_sizes) else 0
    maximum_spt_rows = int(spt_sizes.max().item()) if len(spt_sizes) else 0
    conservative_rows = args.skybox_rows + upper_rows + (maximum_spt_rows or args.min_spt_size + 1) + args.reserve_rows
    passed = conservative_rows < args.reduced_cache_rows
    report = {
        "gate": "PASS" if passed else "BLOCKED-ALOD-RESOURCE-BUDGET",
        "initial_rows": initial_rows, "skybox_rows": args.skybox_rows,
        "rebuilt_upper_rows": upper_rows, "SPT_count": len(spt_sizes),
        "minimum_qualifying_SPT_rows": minimum_spt_rows,
        "maximum_complete_SPT_rows": maximum_spt_rows,
        "reserve_rows": args.reserve_rows, "conservative_complete_rows": conservative_rows,
        "reduced_cache_target_rows": args.reduced_cache_rows,
        "strict_inequality_passed": passed,
    }
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


def proc_snapshot() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    page_size = os.sysconf("SC_PAGE_SIZE")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8").split()
            status = (entry / "status").read_text(encoding="utf-8")
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            rss_pages = int(stat[23])
            result[int(entry.name)] = {
                "ppid": int(stat[3]), "pgrp": int(stat[4]), "rss": rss_pages * page_size,
                "cmdline": cmdline, "name": next((line.split(":", 1)[1].strip() for line in status.splitlines()
                                                    if line.startswith("Name:")), ""),
            }
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
            continue
    return result


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    return values


def directory_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def gpu_snapshot(group_pids: set[int]) -> tuple[int | None, int | None, bool]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None, None, False
    try:
        total = subprocess.check_output(
            [executable, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
        )
        device = subprocess.check_output(
            [executable, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
        )
        process_mib = 0
        attributed = False
        for line in total.splitlines():
            pid_text, memory_text = [part.strip() for part in line.split(",", 1)]
            if int(pid_text) in group_pids:
                process_mib += int(memory_text)
                attributed = True
        device_mib = max(int(line.strip()) for line in device.splitlines() if line.strip())
        process_bytes = process_mib * MIB if attributed else None
        return device_mib * MIB, process_bytes, attributed
    except (subprocess.SubprocessError, OSError, ValueError):
        return None, None, False


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def terminate_group(process: subprocess.Popen[Any], pgid: int, events: Path, reason: str) -> None:
    for signal_value, wait_seconds in ((signal.SIGINT, 30), (signal.SIGTERM, 30), (signal.SIGKILL, 0)):
        if process.poll() is not None:
            return
        timestamp = time.time()
        append_jsonl(events, {"event": "signal", "timestamp": timestamp, "signal": signal_value.name,
                              "process_group_id": pgid, "reason": reason})
        try:
            os.killpg(pgid, signal_value)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + wait_seconds
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.25)


def threshold_active(state: dict[str, int], name: str, violating: bool,
                     required_samples: int) -> bool:
    if not violating:
        state.pop(name, None)
        return False
    state[name] = state.get(name, 0) + 1
    return state[name] >= required_samples


def comparison_violated(value: int, limit: int, comparison: str) -> bool:
    if comparison == "greater-than":
        return value > limit
    if comparison == "less-than":
        return value < limit
    if comparison == "greater-than-or-equal":
        return value >= limit
    raise ValueError(f"unsupported comparison: {comparison}")


def monitor(args: argparse.Namespace) -> int:
    policy = load_json_strict(args.policy)
    policy_validation = validate_policy(policy)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("monitor requires a command after --")
    stage_policy = policy["stage_wall_time"][args.stage]
    dry_result = {
        "stage": args.stage, "cwd": str(args.cwd.resolve()), "command": command,
        "leader_session": "new", "policy": str(args.policy.resolve()), "would_launch": not args.dry_run,
        "policy_validation": policy_validation,
        "process_vram_unavailable_behavior":
            policy["thresholds"]["training_process_vram"]["unavailable_behavior"],
        "pytorch_memory_enforcement": "diagnostic-if-emitted",
    }
    if args.dry_run:
        print(json.dumps(dry_result, indent=2))
        return 0

    args.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.events.parent.mkdir(parents=True, exist_ok=True)
    launch = time.time()
    baseline = meminfo()
    swap_baseline = baseline["SwapTotal"] - baseline["SwapFree"]
    with args.stdout_log.open("ab", buffering=0) as output:
        process = subprocess.Popen(command, cwd=args.cwd, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    pgid = os.getpgid(process.pid)
    append_jsonl(args.events, {"event": "launch", "timestamp": launch, "stage": args.stage,
                              "leader_pid": process.pid, "process_group_id": pgid,
                              "command": command, "swap_baseline_bytes": swap_baseline})
    log_position = 0
    latest_iteration: int | None = None
    latest_node_count: int | None = None
    latest_pytorch_memory_log_observation: str | None = None
    last_progress = time.monotonic()
    threshold_state: dict[str, int] = {}
    warned: set[str] = set()
    abort_reason: str | None = None
    sample_interval = policy["sampling_interval_seconds"]
    previous_swap_growth = 0
    monotonic_swap_samples = 0
    hard_patterns = [pattern.lower() for pattern in policy["failure_patterns"]["hard"]]
    while process.poll() is None:
        now_wall = time.time()
        now = time.monotonic()
        snapshot = proc_snapshot()
        group = {pid: row for pid, row in snapshot.items() if row["pgrp"] == pgid}
        pids = set(group)
        rss = sum(row["rss"] for row in group.values())
        memory = meminfo()
        swap = memory["SwapTotal"] - memory["SwapFree"]
        swap_growth = max(0, swap - swap_baseline)
        output_bytes = directory_bytes(args.output_dir)
        device_vram, process_vram, process_vram_attribution_reliable = gpu_snapshot(pids)
        try:
            with args.stdout_log.open("rb") as handle:
                handle.seek(log_position)
                new_log = handle.read().decode("utf-8", "replace")
                log_position = handle.tell()
        except FileNotFoundError:
            new_log = ""
        if new_log:
            latest_iteration, latest_node_count, progress_observed = parse_log_progress(
                new_log, latest_iteration, latest_node_count
            )
            for match in re.finditer(r"(?:Peak memory|peak memory:)[^\r\n]*", new_log):
                latest_pytorch_memory_log_observation = match.group(0).strip()
            if progress_observed:
                last_progress = now
            lowered = new_log.lower()
            if any(pattern in lowered for pattern in hard_patterns):
                abort_reason = "hard failure pattern in official output"
            if re.search(r"loss[^\n]{0,40}\b(?:nan|inf)\b", lowered):
                abort_reason = "non-finite loss"
        forbidden = policy["failure_patterns"]["forbidden_stage3_process_basename"]
        if args.stage == "fine" and any(forbidden in row["cmdline"] for row in group.values()):
            abort_reason = "orchestration failure: creator launched during fine stage"

        row = {
            "timestamp": now_wall, "stage": args.stage, "leader_pid": process.pid,
            "process_group_id": pgid, "child_pids": sorted(pids - {process.pid}),
            "process_group_rss_bytes": rss, "system_available_ram_bytes": memory["MemAvailable"],
            "swap_used_bytes": swap, "swap_growth_bytes": swap_growth,
            "whole_device_vram_bytes": device_vram, "process_vram_bytes": process_vram,
            "process_vram_attribution_reliable": process_vram_attribution_reliable,
            "pytorch_allocated_bytes": None, "pytorch_reserved_bytes": None,
            "pytorch_memory_log_observation": latest_pytorch_memory_log_observation,
            "output_directory_bytes": output_bytes, "latest_iteration": latest_iteration,
            "node_count": latest_node_count, "elapsed_seconds": now_wall - launch,
        }
        append_jsonl(args.metrics, row)

        node_warning = policy["thresholds"]["fine_node_count"]["warning"]["value_rows"]
        if (latest_node_count is not None and latest_node_count >= node_warning
                and "node_count" not in warned):
            warned.add("node_count")
            append_jsonl(args.events, {"event": "diagnostic", "timestamp": now_wall,
                                      "threshold": "fine_node_count", "value": latest_node_count,
                                      "enforcement": "log-derived-diagnostic"})

        checks = {
            "rss": (rss, policy["thresholds"]["process_group_rss"]),
            "available_ram": (memory["MemAvailable"], policy["thresholds"]["available_system_ram"]),
            "swap": (swap_growth, policy["thresholds"]["swap_growth_from_stage_baseline"]),
            "output": (output_bytes, policy["thresholds"]["output_directory"]),
        }
        if device_vram is not None:
            checks["device_vram"] = (device_vram, policy["thresholds"]["whole_device_vram"])
        if process_vram is not None:
            checks["process_vram"] = (process_vram, policy["thresholds"]["training_process_vram"])
        for name, (value, levels) in checks.items():
            for level in ("warning", "hard"):
                limit = levels[level].get("value_bytes")
                duration = levels[level].get("duration_seconds") or 0
                required_samples = required_violation_samples(duration, sample_interval)
                violating = limit is not None and comparison_violated(
                    value, limit, levels["comparison"]
                )
                active = limit is not None and threshold_active(
                    threshold_state, f"{name}.{level}", violating, required_samples
                )
                if active and level == "warning" and f"{name}.{level}" not in warned:
                    warned.add(f"{name}.{level}")
                    append_jsonl(args.events, {"event": "warning", "timestamp": now_wall,
                                              "threshold": name, "value": value, "limit": limit})
                if active and level == "hard":
                    abort_reason = f"hard {name} threshold exceeded"
        monotonic = policy["thresholds"]["swap_growth_from_stage_baseline"][
            "persistent_monotonic_growth"
        ]
        if swap_growth > previous_swap_growth:
            monotonic_swap_samples += 1
        else:
            monotonic_swap_samples = 0
        previous_swap_growth = swap_growth
        if (monotonic_swap_samples >= monotonic["consecutive_increase_samples"]
                and swap_growth >= monotonic["minimum_total_growth_bytes"]):
            append_jsonl(args.events, {
                "event": "hard", "timestamp": now_wall,
                "threshold": "persistent_monotonic_swap_growth",
                "consecutive_increase_samples": monotonic_swap_samples,
                "swap_growth_bytes": swap_growth,
                "minimum_total_growth_bytes": monotonic["minimum_total_growth_bytes"],
            })
            abort_reason = "persistent monotonic swap growth"
        if (now_wall - launch > stage_policy["warning_seconds"]
                and "stage_wall_time" not in warned):
            warned.add("stage_wall_time")
            append_jsonl(args.events, {"event": "warning", "timestamp": now_wall,
                                      "threshold": f"{args.stage}_wall_time",
                                      "seconds": now_wall - launch})
        if now_wall - launch > stage_policy["hard_seconds"]:
            abort_reason = f"hard {args.stage} wall-time exceeded"
        if args.pipeline_start_epoch and now_wall - args.pipeline_start_epoch > policy["stage_wall_time"]["whole_pipeline"]["hard_seconds"]:
            abort_reason = "hard whole-pipeline wall-time exceeded"
        no_progress = policy["thresholds"]["no_progress"]
        if now - last_progress > no_progress["hard"]["value_seconds"]:
            abort_reason = "hard no-progress threshold exceeded"
        elif now - last_progress > no_progress["warning"]["value_seconds"] and "no_progress" not in warned:
            warned.add("no_progress")
            append_jsonl(args.events, {"event": "warning", "timestamp": now_wall,
                                      "threshold": "no_progress", "seconds": now - last_progress})
        if abort_reason:
            terminate_group(process, pgid, args.events, abort_reason)
            break
        time.sleep(sample_interval)

    exit_status = process.wait()
    append_jsonl(args.events, {"event": "exit", "timestamp": time.time(), "stage": args.stage,
                              "leader_pid": process.pid, "process_group_id": pgid,
                              "exit_status": exit_status, "abort_reason": abort_reason})
    if abort_reason or exit_status != 0:
        return 2 if abort_reason else exit_status
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="subcommand", required=True)

    check = commands.add_parser("validate-plan")
    check.add_argument("--source", type=Path, required=True)
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--resolved", type=Path, required=True)
    check.add_argument("--policy", type=Path, required=True)
    check.add_argument("--provenance", type=Path)
    check.add_argument("--command", type=Path)
    check.add_argument("--monitor", type=Path)
    check.set_defaults(function=validate_plan)

    launcher = commands.add_parser("validate-launcher")
    launcher.set_defaults(function=validate_launcher)

    ply = commands.add_parser("inspect-ply")
    ply.add_argument("--path", type=Path, required=True)
    ply.add_argument("--skybox-rows", type=int, default=100000)
    ply.add_argument("--min-rows", type=int, default=100001)
    ply.add_argument("--max-rows", type=int, default=1100000)
    ply.add_argument("--report", type=Path)
    ply.set_defaults(function=inspect_ply)

    hierarchy = commands.add_parser("inspect-dhier")
    hierarchy.add_argument("--path", type=Path, required=True)
    hierarchy.add_argument("--expected-degree", type=int, required=True)
    hierarchy.add_argument("--skybox-rows", type=int, default=0)
    hierarchy.add_argument("--min-rows", type=int, default=1)
    hierarchy.add_argument("--max-rows", type=int, default=1100000)
    hierarchy.add_argument("--cap-max", type=int, default=1100000)
    hierarchy.add_argument("--additional-rows", type=int, default=0)
    hierarchy.add_argument("--min-growth-rows", type=int, default=0)
    hierarchy.add_argument("--min-headroom-ratio", type=float, default=0.0)
    hierarchy.add_argument("--report", type=Path)
    hierarchy.set_defaults(function=inspect_dhier)

    cache = commands.add_parser("preflight-cache")
    cache.add_argument("--source", type=Path, required=True)
    cache.add_argument("--dataset", type=Path, required=True)
    cache.add_argument("--hierarchy", type=Path, required=True)
    cache.add_argument("--scaffold", type=Path, required=True)
    cache.add_argument("--report", type=Path, required=True)
    cache.add_argument("--llff-hold", type=int, default=50)
    cache.add_argument("--sh-degree", type=int, default=1)
    cache.add_argument("--cap-max", type=int, default=1100000)
    cache.add_argument("--spt-root-volume", type=float, default=150)
    cache.add_argument("--target-pixels", type=float, default=2)
    cache.add_argument("--min-spt-size", type=int, default=256)
    cache.add_argument("--skybox-rows", type=int, default=100000)
    cache.add_argument("--reserve-rows", type=int, default=20000)
    cache.add_argument("--reduced-cache-rows", type=int, default=240000)
    cache.set_defaults(function=preflight_cache)

    watch = commands.add_parser("monitor")
    watch.add_argument("--policy", type=Path, required=True)
    watch.add_argument("--stage", choices=["coarse", "creator", "cache_preflight", "fine"], required=True)
    watch.add_argument("--cwd", type=Path, required=True)
    watch.add_argument("--output-dir", type=Path, required=True)
    watch.add_argument("--metrics", type=Path, required=True)
    watch.add_argument("--events", type=Path, required=True)
    watch.add_argument("--stdout-log", type=Path, required=True)
    watch.add_argument("--pipeline-start-epoch", type=float)
    watch.add_argument("--dry-run", action="store_true")
    watch.add_argument("command", nargs=argparse.REMAINDER)
    watch.set_defaults(function=monitor)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.function(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
