#!/usr/bin/env python3
"""Apply the frozen Campus Stage-A fine-training admission gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET_ID = "ALOD-PREV-H3DGS-CAMPUS-CHUNK-01"
TRAINING_ID = f"{DATASET_ID}-TRAIN-001"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    dataset = load(run / "dataset-manifest.json")
    policy = load(run / "resource-policy.json")
    coarse = load(run / "validation/coarse.json")
    initial = load(run / "validation/initial-hierarchy.json")
    independent = load(run / "validation/initial-hierarchy-independent.json")
    structural = load(run / "validation/spt-upward-sweep.json")
    official = load(run / "validation/official-hspt-9600.json")
    cache = load(run / "validation/cache-preflight-9600.json")
    config = load(Path(__file__).resolve().parents[2] / "configs/alod/ALOD-PREV-H3DGS-CAMPUS-CHUNK-01-TRAIN-001.json")

    checks = {
        "dataset_provenance_and_split": dataset["dataset_id"] == DATASET_ID
        and dataset["status"] == "PASS-FROZEN-DATASET-PREFLIGHT"
        and all(dataset["validation"].values())
        and dataset["counts"]["train_images"] == 1166
        and dataset["counts"]["evaluation_images"] == 24,
        "coarse_complete": coarse["gate"] == "PASS"
        and coarse["rows"] == 459475
        and coarse["all_finite"],
        "creator_complete": initial["gate"] == "PASS"
        and initial["sha256"] == independent["asset_sha256"],
        "strict_hierarchy_valid": independent["validation"]["overall_valid"]
        and independent["hierarchy"]["valid"]
        and independent["hierarchy"]["cycles"] == 0
        and independent["hierarchy"]["unreachable_nodes"] == 0
        and independent["hierarchy"]["parent_mismatches"] == 0
        and independent["hierarchy"]["depth_mismatches"] == 0,
        "initial_below_cap": initial["total_rows_for_cap"] < 12000000,
        "absolute_growth_headroom": initial["growth_headroom_rows"] >= 1000000,
        "relative_growth_headroom": initial["headroom_ratio_of_cap"] >= 0.20,
        "binding_initial_limit": initial["total_rows_for_cap"] <= 9600000,
        "official_hspt_matches_structure": official["gate"] == "PASS-OFFICIAL-HSPT-STRUCTURE"
        and official["SPT_root_volume"] == 9600
        and official["hierarchy_sha256"] == initial["sha256"],
        "hspt_coverage": official["SPT_coverage"] >= 0.95,
        "nondegenerate_spts": official["SPT_count"] >= 2
        and official["SPT_size"]["minimum"] > 256,
        "cache_feasible": cache["gate"] == "PASS"
        and cache["strict_inequality_passed"]
        and cache["conservative_complete_rows"]
        == 100000 + official["upper_hierarchy_rows"] + official["SPT_size"]["maximum"] + 100000
        and cache["conservative_complete_rows"] <= 5000000,
        "fixed_method_configuration": config["iterations"] == 100000
        and config["coarse_iterations"] == 30000
        and config["SH_degree"] == 1
        and config["cap_max"] == 12000000
        and config["min_SPT_size"] == 256
        and config["SPT_root_volume"] == 9600
        and config["cache_size"] == 6250000
        and config["cache_size_after_reduction"] == 5000000
        and config["llff_hold"] == 50,
    }
    candidates = {row["SPT_root_volume"]: row for row in structural["candidates"]}
    checks["stable_calibration_neighborhood"] = (
        all(candidates[value]["passes_all_nonstability_criteria"] for value in (4800.0, 9600.0, 19200.0))
        and not candidates[38400.0]["passes_all_nonstability_criteria"]
        and all(
            0.5 <= candidates[left]["SPT_count"] / candidates[right]["SPT_count"] <= 2.0
            and 0.5 <= candidates[left]["SPT_size"]["maximum"] / candidates[right]["SPT_size"]["maximum"] <= 2.0
            for left, right in ((4800.0, 9600.0), (9600.0, 19200.0))
        )
    )
    checks["official_hspt_exact_prediction"] = all(
        official[field] == candidates[9600.0][field]
        for field in ("upper_hierarchy_rows", "SPT_count", "flattened_SPT_rows", "SPT_coverage")
    ) and official["SPT_size"]["maximum"] == candidates[9600.0]["SPT_size"]["maximum"]

    events = [json.loads(line) for line in (run / "logs/events.jsonl").read_text().splitlines()]
    required = ("coarse", "creator", "cache_preflight")
    checks["previous_stage_exit_statuses"] = (
        any(row.get("event") == "exit" and row.get("stage") == "coarse" and row.get("exit_status") == 0 for row in events)
        and any(row.get("event") == "exit" and row.get("stage") == "creator" and row.get("exit_status") == 0 for row in events)
        and not any(row.get("abort_reason") for row in events)
    )
    peaks = {}
    thresholds = policy["thresholds"]
    for stage in ("coarse", "creator", "initial-validation", "spt-coarse-sweep", "spt-upward-sweep-corrected", "official-hspt-9600", "cache-preflight-9600"):
        path = run / "metrics" / f"{stage}.jsonl"
        samples = [json.loads(line) for line in path.read_text().splitlines()]
        peaks[stage] = {
            "elapsed_seconds": max(row["elapsed_seconds"] for row in samples),
            "process_group_RSS_bytes": max(row["process_group_rss_bytes"] for row in samples),
            "minimum_available_RAM_bytes": min(row["system_available_ram_bytes"] for row in samples),
            "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in samples),
            "whole_device_VRAM_bytes": max(row["whole_device_vram_bytes"] or 0 for row in samples),
            "output_directory_bytes": max(row["output_directory_bytes"] for row in samples),
        }
    stage_limits = {
        "coarse": "coarse",
        "creator": "creator",
        "initial-validation": "cache_preflight",
        "spt-coarse-sweep": "cache_preflight",
        "spt-upward-sweep-corrected": "cache_preflight",
        "official-hspt-9600": "cache_preflight",
        "cache-preflight-9600": "cache_preflight",
    }
    checks["previous_stage_resources"] = all(
        value["process_group_RSS_bytes"] <= thresholds["process_group_rss"]["hard"]["value_bytes"]
        and value["minimum_available_RAM_bytes"] >= thresholds["available_system_ram"]["hard"]["value_bytes"]
        and value["maximum_swap_growth_bytes"] <= thresholds["swap_growth_from_stage_baseline"]["hard"]["value_bytes"]
        and value["whole_device_VRAM_bytes"] <= thresholds["whole_device_vram"]["hard"]["value_bytes"]
        and value["output_directory_bytes"] <= thresholds["output_directory"]["hard"]["value_bytes"]
        and value["elapsed_seconds"] < policy["stage_wall_time"][stage_limits[stage]]["hard_seconds"]
        for stage, value in peaks.items()
    )

    admitted = all(checks.values())
    result = {
        "schema_version": 1,
        "protocol_id": "ALOD-LOD-PREVALENCE-DESIGN-001",
        "dataset_id": DATASET_ID,
        "training_id": TRAINING_ID,
        "gate": "PASS-FROZEN-FINE-ADMISSION" if admitted else "BLOCKED-FROZEN-FINE-ADMISSION",
        "checks": checks,
        "selected_SPT_root_volume": 9600,
        "calibration_neighbors": [4800, 19200],
        "initial_total_rows": initial["total_rows_for_cap"],
        "initial_hierarchy_rows": initial["gaussians"],
        "initial_maximum_depth": independent["hierarchy"]["max_depth"],
        "initial_hierarchy_sha256": initial["sha256"],
        "initial_hierarchy_bytes": independent["asset_size"],
        "HSPT": {
            "upper_hierarchy_rows": official["upper_hierarchy_rows"],
            "SPT_count": official["SPT_count"],
            "flattened_SPT_rows": official["flattened_SPT_rows"],
            "maximum_SPT_rows": official["SPT_size"]["maximum"],
            "coverage": official["SPT_coverage"],
            "conservative_cache_requirement_rows": cache["conservative_complete_rows"],
        },
        "resource_peaks": peaks,
        "artifact_sha256": {
            name: sha256(run / name)
            for name in (
                "dataset-manifest.json",
                "resource-policy.json",
                "validation/initial-hierarchy-independent.json",
                "validation/spt-upward-sweep.json",
                "validation/official-hspt-9600.json",
                "validation/cache-preflight-9600.json",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": result["gate"], "failed_checks": [key for key, value in checks.items() if not value]}))
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
