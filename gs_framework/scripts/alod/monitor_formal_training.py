#!/usr/bin/env python3
"""Formal-envelope adapter for the frozen Run005 process monitor.

The historical monitor remains byte-for-byte unchanged for Run005 provenance.
This adapter reuses its process-tree sampling, abort sequence, artifact
inspectors, cache preflight, and CLI while permitting only formal-run node and
wall-time values. Machine RSS/RAM/swap/VRAM/disk/no-progress thresholds are
still validated by the historical canonical validator.
"""

from __future__ import annotations

import copy
import importlib.util
import shutil
import subprocess
from pathlib import Path


LEGACY_PATH = Path(__file__).with_name("monitor_capped_training.py")
SPEC = importlib.util.spec_from_file_location("alod_legacy_monitor", LEGACY_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load historical monitor: {LEGACY_PATH}")
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)
_legacy_validate_policy = legacy.validate_policy


def validate_formal_policy(policy):
    """Validate unchanged safety limits, then the formal cap/time dimensions."""
    probe = copy.deepcopy(policy)
    probe["thresholds"]["fine_node_count"]["warning"]["value_rows"] = 1_050_000
    probe["thresholds"]["fine_node_count"]["hard"]["value_rows"] = 1_100_000
    probe["configuration_limits"]["fine_cap_max"] = 1_100_000
    probe["stage_wall_time"]["coarse"].update(
        warning_seconds=5_400, hard_seconds=10_800
    )
    probe["stage_wall_time"]["creator"].update(
        warning_seconds=1_200, hard_seconds=3_600
    )
    probe["stage_wall_time"]["fine"].update(
        warning_seconds=14_400, hard_seconds=36_000
    )
    probe["stage_wall_time"]["whole_pipeline"].update(
        warning_seconds=None, hard_seconds=43_200
    )
    result = _legacy_validate_policy(probe)

    limits = policy.get("configuration_limits", {})
    cap = limits.get("fine_cap_max")
    warning = policy["thresholds"]["fine_node_count"]["warning"]["value_rows"]
    hard = policy["thresholds"]["fine_node_count"]["hard"]["value_rows"]
    if not isinstance(cap, int) or cap <= 0 or hard != cap:
        raise ValueError("formal fine-node hard threshold must equal positive cap_max")
    if not 0 < warning < hard:
        raise ValueError("formal fine-node warning must be positive and below cap_max")

    stage_values = {}
    for stage in ("coarse", "creator", "cache_preflight", "fine"):
        record = policy["stage_wall_time"][stage]
        stage_warning = record["warning_seconds"]
        stage_hard = record["hard_seconds"]
        if not 0 < stage_warning < stage_hard:
            raise ValueError(f"invalid formal wall-time ordering: {stage}")
        stage_values[stage] = {"warning": stage_warning, "hard": stage_hard}
    pipeline = policy["stage_wall_time"]["whole_pipeline"]
    if not 0 < pipeline["warning_seconds"] < pipeline["hard_seconds"]:
        raise ValueError("invalid formal whole-pipeline wall-time ordering")

    result["canonical_thresholds"]["fine_node_count_rows"] = {
        "warning": warning,
        "hard_configuration_cap": hard,
    }
    result["stage_wall_time_seconds"] = stage_values | {
        "whole_pipeline": {
            "warning": pipeline["warning_seconds"],
            "hard": pipeline["hard_seconds"],
        }
    }
    result["adapter"] = "formal node-cap and wall-time dimensions only"
    return result


legacy.validate_policy = validate_formal_policy


def gpu_snapshot(group_pids: set[int]) -> tuple[int | None, int | None, bool]:
    """Keep device enforcement when WSL cannot attribute process VRAM.

    WSL commonly returns ``[N/A]`` for ``used_memory`` in the compute-process
    table.  The frozen policy explicitly requires null process attribution in
    that case, but whole-device memory remains independently measurable and
    hard-enforced.
    """
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None, None, False

    device_bytes = None
    try:
        device = subprocess.check_output(
            [executable, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        device_mib = max(int(line.strip()) for line in device.splitlines() if line.strip())
        device_bytes = device_mib * legacy.MIB
    except (subprocess.SubprocessError, OSError, ValueError):
        device_bytes = None

    process_mib = 0
    attributed = False
    try:
        processes = subprocess.check_output(
            [
                executable,
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        for line in processes.splitlines():
            pid_text, memory_text = [part.strip() for part in line.split(",", 1)]
            if int(pid_text) not in group_pids:
                continue
            try:
                process_mib += int(memory_text)
            except ValueError:
                continue
            attributed = True
    except (subprocess.SubprocessError, OSError, ValueError):
        attributed = False

    return device_bytes, process_mib * legacy.MIB if attributed else None, attributed


legacy.gpu_snapshot = gpu_snapshot


if __name__ == "__main__":
    raise SystemExit(legacy.main())
