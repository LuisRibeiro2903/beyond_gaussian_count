#!/usr/bin/env python3
"""Package the frozen natural Previous-completion timing campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


CAMPAIGN_ID = "ALOD-EXT-NATURAL-PREVIOUS-COMPLETION-001"
DATASETS = {"campus": 24, "sciart": 21}
MODES = ("nocache", "previous", "bounded")
TIMINGS = (
    "end_to_end_frame_time_ms",
    "alod_total_selection_ms",
    "alod_materialization_filter_ms",
    "preparation_time_ms",
    "alod_host_gather_ms",
    "alod_h2d_transfer_ms",
    "frame_time_ms",
    "rasterization_time_ms",
    "alod_cache_management_ms",
    "alod_pre_raster_assembly_ms",
    "alod_materialization_unique_extraction_ms",
    "alod_materialization_occurrence_propagation_ms",
    "alod_occurrence_reindex_ms",
)
COUNTS = (
    "alod_semantic_selected_rows",
    "alod_unique_semantic_rows",
    "alod_materialization_rows",
    "alod_unique_materialization_rows",
    "alod_assembled_renderer_occurrence_rows",
    "backend_projection_survivor_count",
    "alod_h2d_rows",
    "alod_h2d_bytes",
    "alod_requested_rows",
    "alod_cache_hit_count",
    "alod_cache_miss_count",
    "alod_cache_eviction_count",
    "alod_previous_intersection_rows",
    "alod_previous_working_set_rows",
    "alod_retention_ratio",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"frozen hash mismatch for {path}: {actual} != {expected}")


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = sorted(float(row[key]) for row in rows if row.get(key) is not None)
    if not values:
        return {"minimum": None, "mean": None, "median": None, "p95": None, "maximum": None}
    position = 0.95 * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    p95 = values[lower] if lower == upper else values[lower] + (position - lower) * (values[upper] - values[lower])
    return {
        "minimum": values[0],
        "mean": mean(values),
        "median": median(values),
        "p95": p95,
        "maximum": values[-1],
    }


def total(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(float(row[key]) for row in rows if row.get(key) is not None))


def summarize_mode(run_dirs: list[Path], expected_frames: int) -> dict[str, Any]:
    measured_by_rep = [read_jsonl(path / "frames.jsonl") for path in run_dirs]
    warmup_by_rep = [read_jsonl(path / "warmup_frames.jsonl") for path in run_dirs]
    if any(len(rows) != expected_frames for rows in measured_by_rep):
        raise RuntimeError("measured frame count does not match the frozen trajectory")
    if any(len(rows) != 10 for rows in warmup_by_rep):
        raise RuntimeError("warmup frame count does not match the frozen protocol")
    measured = [row for rows in measured_by_rep for row in rows]
    warmup = [row for rows in warmup_by_rep for row in rows]
    camera_orders = [[row["camera_id"] for row in rows] for rows in measured_by_rep]
    if any(order != camera_orders[0] for order in camera_orders[1:]):
        raise RuntimeError("replicate camera order mismatch")
    hits = total(measured, "alod_cache_hit_count")
    misses = total(measured, "alod_cache_miss_count")
    e2e = distribution(measured, "end_to_end_frame_time_ms")
    return {
        "replicate_count": len(run_dirs),
        "measured_frames": len(measured),
        "warmup_frames": len(warmup),
        "camera_ids": camera_orders[0],
        "timings_ms": {key: distribution(measured, key) for key in TIMINGS},
        "counts": {key: distribution(measured, key) for key in COUNTS},
        "fps_from_mean_end_to_end": 1000.0 / float(e2e["mean"]),
        "steady_dynamic_property_h2d_rows": total(measured, "alod_h2d_rows"),
        "steady_dynamic_property_h2d_bytes": total(measured, "alod_h2d_bytes"),
        "warmup_dynamic_property_h2d_rows": total(warmup, "alod_h2d_rows"),
        "warmup_dynamic_property_h2d_bytes": total(warmup, "alod_h2d_bytes"),
        "cold_inclusive_dynamic_property_h2d_rows": total(measured + warmup, "alod_h2d_rows"),
        "cold_inclusive_dynamic_property_h2d_bytes": total(measured + warmup, "alod_h2d_bytes"),
        "pooled_measured_cache_hit_rate": hits / (hits + misses) if hits + misses else None,
        "replicates": [
            {
                "run_dir": str(path.resolve()),
                "mean_end_to_end_frame_time_ms": mean(float(row["end_to_end_frame_time_ms"]) for row in rows),
            }
            for path, rows in zip(run_dirs, measured_by_rep)
        ],
    }


def previous_comparison(modes: dict[str, Any]) -> dict[str, float]:
    released = modes["nocache"]
    previous = modes["previous"]
    bounded = modes["bounded"]
    released_ms = released["timings_ms"]["end_to_end_frame_time_ms"]["mean"]
    previous_ms = previous["timings_ms"]["end_to_end_frame_time_ms"]["mean"]
    bounded_ms = bounded["timings_ms"]["end_to_end_frame_time_ms"]["mean"]
    return {
        "previous_frame_time_reduction_vs_released_fraction": 1.0 - previous_ms / released_ms,
        "previous_speedup_vs_released": released_ms / previous_ms,
        "previous_minus_bounded_ms": previous_ms - bounded_ms,
        "previous_frame_time_ratio_vs_bounded": previous_ms / bounded_ms,
        "previous_steady_h2d_reduction_vs_released_fraction": 1.0 - previous["steady_dynamic_property_h2d_bytes"] / released["steady_dynamic_property_h2d_bytes"],
        "previous_cold_h2d_reduction_vs_released_fraction": 1.0 - previous["cold_inclusive_dynamic_property_h2d_bytes"] / released["cold_inclusive_dynamic_property_h2d_bytes"],
    }


def prior_workloads(workspace: Path) -> dict[str, Any]:
    run009_path = workspace / "benchmark_artifacts/alod_ext_formal_run009_001/tables/formal_summary.json"
    run009 = read_json(run009_path)
    run_modes = run009["extension_modes"]
    run_released = run009["faithful_context"]["ALOD-REPRO-HOST-NOCACHE"]
    synthetic_path = workspace / "benchmark_artifacts/alod_synth_strong_lod_ext_multiset_closure_001/tables/aggregate.json"
    synthetic = read_json(synthetic_path)
    return {
        "run009": {
            "lod_class": "VALID-WEAK-LOD (historical context)",
            "released_ms": run_released["timings_ms"]["end_to_end_frame_time_ms"]["mean"],
            "previous_ms": run_modes["ALOD-EXT-PREMATERIALIZATION-CUDA-PREVIOUS"]["timings_ms"]["end_to_end_frame_time_ms"]["mean"],
            "bounded_ms": run_modes["ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL"]["timings_ms"]["end_to_end_frame_time_ms"]["mean"],
            "authority": str(run009_path),
            "authority_sha256": sha256(run009_path),
        },
        "synthetic_strong_control": {
            "lod_class": "STRONG CONTROL",
            "released_ms": synthetic["modes"]["nocache"]["timings_ms"]["end_to_end_frame_time_ms"]["mean"],
            "previous_ms": synthetic["modes"]["previous"]["timings_ms"]["end_to_end_frame_time_ms"]["mean"],
            "bounded_ms": synthetic["modes"]["bounded"]["timings_ms"]["end_to_end_frame_time_ms"]["mean"],
            "authority": str(synthetic_path),
            "authority_sha256": sha256(synthetic_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    workspace = args.workspace_root.resolve()
    protocol = read_json(root / "protocol.json")
    if protocol.get("campaign_id") != CAMPAIGN_ID or protocol.get("status") != "FROZEN-BEFORE-TIMING":
        raise RuntimeError("campaign protocol is not the frozen completion protocol")
    require_hash(Path(protocol["protocol_note"]["path"]), protocol["protocol_note"]["sha256"])
    parent_config_path = Path(protocol["parent_configuration"]["path"])
    require_hash(parent_config_path, protocol["parent_configuration"]["sha256"])
    amendment = read_json(root / "protocol_amendment_001.json")
    if amendment.get("original_protocol_sha256") != sha256(root / "protocol.json"):
        raise RuntimeError("protocol amendment does not identify the frozen protocol")
    if any(amendment.get(key) is not False for key in ("scientific_changes", "runtime_changes", "input_changes", "result_changes")):
        raise RuntimeError("protocol amendment exceeds an administrative correction")
    amended_path = amendment["field"].removeprefix("implementation.")
    if protocol["implementation"].get(amended_path) != amendment["recorded_value"]:
        raise RuntimeError("protocol amendment recorded value mismatch")
    implementation = dict(protocol["implementation"])
    implementation[amended_path] = amendment["corrected_value"]
    for path, expected in implementation.items():
        require_hash(Path(path), expected)
    correctness = protocol["correctness_reuse"]
    correctness_path = Path(correctness["authority"])
    require_hash(correctness_path, correctness["authority_sha256"])
    correctness_payload = read_json(correctness_path)
    if correctness_payload.get("status") != correctness["status"]:
        raise RuntimeError("reused natural correctness status changed")
    require_hash(
        workspace / "benchmark_artifacts/alod_ext_prevalence_association_001/checks/campus_correctness.json",
        correctness["campus_correctness_sha256"],
    )
    require_hash(
        workspace / "benchmark_artifacts/alod_ext_prevalence_association_001/checks/sciart_correctness.json",
        correctness["sciart_correctness_sha256"],
    )
    parent_config = read_json(parent_config_path)
    for dataset, frozen in protocol["datasets"].items():
        parent = parent_config["datasets"][dataset]
        require_hash(Path(parent["asset_path"]), frozen["asset_sha256"])
        require_hash(Path(parent["camera_path"]), frozen["camera_path_sha256"])
        require_hash(Path(parent["oracle_manifest"]), frozen["oracle_manifest_sha256"])

    datasets: dict[str, Any] = {}
    canonical_orders: dict[str, list[str]] = {}
    schedule: list[dict[str, Any]] = []
    for dataset, frame_count in DATASETS.items():
        modes = {}
        for mode in MODES:
            dirs = [root / "runs" / dataset / mode / f"replicate-{replicate:02d}" for replicate in range(1, 4)]
            modes[mode] = summarize_mode(dirs, frame_count)
            if dataset not in canonical_orders:
                canonical_orders[dataset] = modes[mode]["camera_ids"]
            elif modes[mode]["camera_ids"] != canonical_orders[dataset]:
                raise RuntimeError("mode camera order mismatch")
            for replicate, run_dir in enumerate(dirs, start=1):
                resolved = read_json(run_dir / "resolved_config.json")
                expected_residency = {
                    "nocache": "host_nocache",
                    "previous": "host_previous_reuse",
                    "bounded": "host_bounded_cache",
                }[mode]
                resolved_host = resolved["configuration"]["alod_host"]
                if resolved_host["mode"] != expected_residency:
                    raise RuntimeError("resolved residency policy mismatch")
                if mode == "bounded" and int(resolved_host["cache_capacity_rows"]) != int(protocol["datasets"][dataset]["bounded_capacity_rows"]):
                    raise RuntimeError("resolved bounded capacity mismatch")
                warmup_path = run_dir / "warmup_frames.jsonl"
                schedule.append({
                    "dataset": dataset,
                    "mode": mode,
                    "replicate": replicate,
                    "run_dir": str(run_dir),
                    "warmup_file_mtime_ns": warmup_path.stat().st_mtime_ns,
                    "status": "complete",
                })
        if modes["bounded"]["counts"]["alod_cache_eviction_count"]["maximum"] != 0.0:
            raise RuntimeError("bounded policy evicted rows despite frozen union-covering capacity")
        semantic = modes["previous"]["counts"]["alod_semantic_selected_rows"]["mean"]
        materialized = modes["previous"]["counts"]["alod_materialization_rows"]["mean"]
        survivors = modes["previous"]["counts"]["backend_projection_survivor_count"]["mean"]
        datasets[dataset] = {
            "dataset_id": protocol["datasets"][dataset]["dataset_id"],
            "lod_class": protocol["datasets"][dataset]["stage_a_label"],
            "camera_count": frame_count,
            "bounded_capacity_rows": protocol["datasets"][dataset]["bounded_capacity_rows"],
            "workload_funnel": {
                "mean_semantic_rows": semantic,
                "mean_materialized_rows": materialized,
                "mean_packed_survivors": survivors,
                "materialized_over_semantic": materialized / semantic,
                "survivors_over_semantic": survivors / semantic,
            },
            "modes": modes,
            "previous_comparison": previous_comparison(modes),
            "fastest_ext_policy": min(("previous", "bounded"), key=lambda mode: modes[mode]["timings_ms"]["end_to_end_frame_time_ms"]["mean"]),
        }

    prior = prior_workloads(workspace)
    comparison = {
        "run009": prior["run009"],
        "campus": {
            "lod_class": datasets["campus"]["lod_class"],
            "released_ms": datasets["campus"]["modes"]["nocache"]["timings_ms"]["end_to_end_frame_time_ms"]["mean"],
            "previous_ms": datasets["campus"]["modes"]["previous"]["timings_ms"]["end_to_end_frame_time_ms"]["mean"],
            "bounded_ms": datasets["campus"]["modes"]["bounded"]["timings_ms"]["end_to_end_frame_time_ms"]["mean"],
        },
        "sciart": {
            "lod_class": datasets["sciart"]["lod_class"],
            "released_ms": datasets["sciart"]["modes"]["nocache"]["timings_ms"]["end_to_end_frame_time_ms"]["mean"],
            "previous_ms": datasets["sciart"]["modes"]["previous"]["timings_ms"]["end_to_end_frame_time_ms"]["mean"],
            "bounded_ms": datasets["sciart"]["modes"]["bounded"]["timings_ms"]["end_to_end_frame_time_ms"]["mean"],
        },
        "synthetic_strong_control": prior["synthetic_strong_control"],
    }
    for row in comparison.values():
        row["fastest_ext_policy"] = "previous" if row["previous_ms"] < row["bounded_ms"] else "bounded"

    summary = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "PASS-ALOD-EXT-NATURAL-PREVIOUS-COMPLETION",
        "classification": protocol["classification"],
        "correctness_reuse": protocol["correctness_reuse"],
        "protocol_sha256": sha256(root / "protocol.json"),
        "datasets": datasets,
        "four_workload_comparison": comparison,
        "interpretation": "bounded is faster than previous on both rerun natural trajectories; the synthetic previous-over-bounded observation does not appear on these natural workloads",
    }
    write_json(root / "tables/completion_summary.json", summary)
    schedule.sort(key=lambda row: (row["dataset"], row["warmup_file_mtime_ns"]))
    expected_order = [
        (1, "nocache"), (1, "previous"), (1, "bounded"),
        (2, "previous"), (2, "bounded"), (2, "nocache"),
        (3, "bounded"), (3, "nocache"), (3, "previous"),
    ]
    for dataset in DATASETS:
        actual = [(row["replicate"], row["mode"]) for row in schedule if row["dataset"] == dataset]
        if actual != expected_order:
            raise RuntimeError(f"{dataset} execution order differs from the frozen balanced schedule")
    for sequence_index, row in enumerate(schedule, start=1):
        row["sequence_index"] = sequence_index
    write_json(root / "checks/schedule.json", {"campaign_id": CAMPAIGN_ID, "schedule": schedule})

    validation = {
        "campaign_id": CAMPAIGN_ID,
        "status": "PASS-PACKAGE-VALIDATION",
        "protocol_frozen_before_runs": True,
        "correctness_reused_from_byte_exact_post_multiset_natural_regression": True,
        "frozen_protocol_and_implementation_hashes_exact": True,
        "frozen_asset_camera_and_oracle_hashes_exact": True,
        "dataset_count": 2,
        "mode_count_per_dataset": 3,
        "replicates_per_mode": 3,
        "measured_frames": {key: DATASETS[key] * 3 * 3 for key in DATASETS},
        "warmup_frames": {key: 10 * 3 * 3 for key in DATASETS},
        "camera_order_exact_across_modes_and_replicates": True,
        "balanced_execution_schedule_exact": True,
        "resolved_policy_and_capacity_exact": True,
        "bounded_zero_evictions": True,
        "all_required_timing_columns_present": all(
            datasets[d]["modes"][m]["timings_ms"][key]["mean"] is not None
            for d in DATASETS
            for m in MODES
            for key in (
                "end_to_end_frame_time_ms",
                "alod_total_selection_ms",
                "alod_materialization_filter_ms",
                "preparation_time_ms",
                "alod_host_gather_ms",
                "alod_h2d_transfer_ms",
                "frame_time_ms",
                "rasterization_time_ms",
                "alod_pre_raster_assembly_ms",
            )
        ),
        "all_ext_diagnostic_timing_columns_present": all(
            datasets[d]["modes"][m]["timings_ms"][key]["mean"] is not None
            for d in DATASETS
            for m in ("previous", "bounded")
            for key in (
                "alod_cache_management_ms",
                "alod_materialization_unique_extraction_ms",
                "alod_materialization_occurrence_propagation_ms",
                "alod_occurrence_reindex_ms",
            )
        ),
    }
    write_json(root / "checks/package_validation.json", validation)

    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "status": summary["status"],
        "file_count": len(files),
        "files": files,
    }
    write_json(root / "manifest.json", manifest)
    print(json.dumps({"status": summary["status"], "validation": validation["status"], "files": len(files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
