#!/usr/bin/env python3
"""Validate and package the ordered-multiset synthetic closure evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


CAMPAIGN_ID = "ALOD-SYNTH-STRONG-LOD-EXT-MULTISET-CLOSURE-001"
MODES = ("nocache", "previous", "bounded")
REPRESENTATIVES = {"near": 0, "middle": 12, "far": 24}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(statistics.mean(values)) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = args.repo_root.resolve()

    correctness = read_json(root / "checks/all_cameras_correctness.json")
    duplicate = read_json(root / "checks/duplicate_cameras_correctness.json")
    regression = read_json(root / "checks/natural_regression_summary.json")
    schedule = read_json(root / "checks/schedule.json")
    aggregate = read_json(root / "tables/aggregate.json")
    execution_protocol = read_json(root / "protocol_execution_002.json")

    run_root = root / "runs" / CAMPAIGN_ID
    per_mode: dict[str, list[dict[str, Any]]] = {}
    for mode in MODES:
        per_mode[mode] = [
            frame
            for replicate in sorted((run_root / mode).glob("replicate-*"))
            for frame in read_jsonl(replicate / "frames.jsonl")
        ]

    timing_keys = (
        "end_to_end_frame_time_ms",
        "frame_preparation_ms",
        "frame_time_ms",
        "alod_total_selection_ms",
        "alod_materialization_filter_ms",
        "alod_host_gather_ms",
        "alod_cache_management_ms",
        "alod_h2d_transfer_ms",
        "alod_materialization_unique_extraction_ms",
        "alod_materialization_occurrence_propagation_ms",
        "alod_occurrence_reindex_ms",
        "alod_pre_raster_assembly_ms",
        "rasterization_time_ms",
        "alod_h2d_bytes",
    )
    per_camera: dict[str, Any] = {}
    for camera_index in range(25):
        correctness_row = correctness["cameras"][camera_index]
        per_camera[str(camera_index)] = {
            "funnel": {
                key: correctness_row[key]
                for key in (
                    "semantic_rows",
                    "unique_semantic_rows",
                    "duplicate_occurrence_count",
                    "unique_materialized_rows",
                    "assembled_renderer_occurrence_rows",
                    "packed_projection_survivors",
                )
            },
            "modes": {
                mode: {
                    key: mean(
                        [row for row in rows if int(row["camera_index"]) == camera_index],
                        key,
                    )
                    for key in timing_keys
                }
                for mode, rows in per_mode.items()
            },
        }
    write_json(
        root / "tables/per_camera_summary.json",
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "replicate_mean_per_camera": True,
            "cameras": per_camera,
            "representatives": {
                name: per_camera[str(index)] for name, index in REPRESENTATIVES.items()
            },
        },
    )

    rgb_max_mean = max(row["rgb"]["mean_absolute_error"] for row in correctness["cameras"])
    rgb_max_abs = max(row["rgb"]["maximum_absolute_error"] for row in correctness["cameras"])
    alpha_max_mean = max(row["alpha"]["mean_absolute_error"] for row in correctness["cameras"])
    alpha_max_abs = max(row["alpha"]["maximum_absolute_error"] for row in correctness["cameras"])
    write_json(
        root / "checks/image_agreement.json",
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "status": "PASS-IMAGE-AGREEMENT",
            "camera_count": 25,
            "rgb": {
                "maximum_camera_mean_absolute_error": rgb_max_mean,
                "maximum_absolute_error": rgb_max_abs,
                "all_bitwise_equal": all(
                    row["rgb"]["bitwise_equal"] for row in correctness["cameras"]
                ),
            },
            "alpha": {
                "maximum_camera_mean_absolute_error": alpha_max_mean,
                "maximum_absolute_error": alpha_max_abs,
                "all_bitwise_equal": all(
                    row["alpha"]["bitwise_equal"] for row in correctness["cameras"]
                ),
            },
            "bounds": correctness["criteria"]["image_agreement_bounds"],
        },
    )

    required_frame_keys = (
        "alod_unique_semantic_rows",
        "alod_duplicate_semantic_occurrences",
        "alod_unique_materialization_rows",
        "alod_assembled_renderer_occurrence_rows",
        "alod_materialization_unique_extraction_ms",
        "alod_materialization_occurrence_propagation_ms",
        "alod_occurrence_reindex_ms",
    )
    raw_frames = [row for rows in per_mode.values() for row in rows]
    implementation = execution_protocol["implementation"]
    implementation_hashes_match = all(
        sha256(Path(path)) == expected for path, expected in implementation.items()
    )
    checks = {
        "campaign_id_exact": aggregate["campaign_id"] == CAMPAIGN_ID,
        "duplicate_gate_passed": duplicate["camera_count"] == 7
        and duplicate["status"].startswith("PASS-"),
        "all_camera_gate_passed": correctness["camera_count"] == 25
        and correctness["status"].startswith("PASS-"),
        "natural_regression_passed": regression["status"].startswith("PASS-"),
        "schedule_complete": len(schedule["schedule"]) == 9
        and all(row["status"] == "complete" for row in schedule["schedule"]),
        "raw_frame_count_exact": len(raw_frames) == 225,
        "required_frame_metrics_present": all(
            all(row.get(key) is not None for key in required_frame_keys)
            for row in raw_frames
        ),
        "bounded_capacity_matches_correctness": (
            aggregate["bounded_capacity_rows"]
            == correctness["capacity"]["bounded_capacity_rows"]
        ),
        "zero_bounded_evictions": (
            aggregate["modes"]["bounded"]["counts"]["alod_cache_eviction_count"][
                "maximum"
            ]
            == 0.0
        ),
        "execution_code_hashes_match": implementation_hashes_match,
    }
    validation = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "PASS-PACKAGE-VALIDATION" if all(checks.values()) else "FAIL-PACKAGE-VALIDATION",
        "checks": checks,
        "first_timing_attempt_excluded": True,
        "excluded_attempt_path": "attempts/instrumentation_missing_001",
    }
    write_json(root / "checks/package_validation.json", validation)
    if not all(checks.values()):
        raise RuntimeError(validation)

    evidence_paths = [
        root / "protocol.json",
        root / "protocol_amendment_001.json",
        root / "protocol_execution_002.json",
        root / "checks/duplicate_cameras_correctness.json",
        root / "checks/all_cameras_correctness.json",
        root / "checks/natural_regression_summary.json",
        root / "checks/campus_regression.json",
        root / "checks/sciart_regression.json",
        root / "checks/image_agreement.json",
        root / "checks/schedule.json",
        root / "checks/package_validation.json",
        root / "tables/aggregate.json",
        root / "tables/per_camera_summary.json",
        root / "traces/duplicate_cameras_trace.npz",
        root / "traces/all_cameras_trace.npz",
        root / "traces/campus_regression_trace.npz",
        root / "traces/sciart_regression_trace.npz",
    ]
    evidence_paths.extend(
        path
        for mode in MODES
        for replicate in sorted((run_root / mode).glob("replicate-*"))
        for path in (
            replicate / "frames.jsonl",
            replicate / "warmup_frames.jsonl",
            replicate / "summary.json",
        )
    )
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "PASS-ALOD-SYNTH-STRONG-LOD-EXT-MULTISET-CLOSURE",
        "classification": "project-specific ALoD-EXT ordered-multiset closure",
        "controlled_parent_id": "ALOD-SYNTH-STRONG-LOD-CONTROL-001",
        "parent_result_unchanged": "BLOCKED-SYNTH-ALOD-EXT-OFFICIAL-DUPLICATE-SEMANTIC-IDS",
        "final_execution": "execution-002 after instrumentation-only amendment",
        "aggregate_status": aggregate["status"],
        "artifacts": {
            str(path.relative_to(root)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in evidence_paths
        },
        "postprocessor": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "repository": str(repo),
    }
    write_json(root / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
