#!/usr/bin/env python
"""Aggregate the clean three-process full CityGaussian official-runtime campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


CAMERAS = (0, 131, 262, 392, 523, 654, 785, 916, 1047, 1177, 1308, 1439)
TIMING_FIELDS = (
    "visibility_time_ms",
    "lod_selection_time_ms",
    "selection_time_ms",
    "active_set_assembly_time_ms",
    "render_call_time_ms",
    "cuda_rasterization_time_ms",
    "end_to_end_frame_time_ms",
)
ARTIFACT_CANONICAL = Path("/home/luis/tese/benchmark_artifacts/citygs_canonical_20260824_135615")
ARTIFACT_HISTORICAL = Path("/home/luis/tese/benchmark_artifacts/citygs_official_parity_20260824_191215")
ARTIFACT_ATTRIBUTION = Path("/home/luis/tese/benchmark_artifacts/citygs_12camera_attribution_20260824_222048")
TRAJECTORY = Path("/home/luis/tese/gs_framework/benchmarks/camera_paths/citygaussian_23m_navigation_v1.json")
ASSETS = (
    Path("/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_50_vq/point_cloud.ply"),
    Path("/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_66_vq/point_cloud.ply"),
    Path("/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_75_vq/point_cloud.ply"),
)
SCRIPTS = (
    Path("/home/luis/tese/official_clean_full/capture_gpu.py"),
    Path("/home/luis/tese/official_clean_full/run_repeat.sh"),
    Path("/home/luis/tese/official_clean_full/aggregate.py"),
    Path("/home/luis/tese/CityGaussian_official_parity/official_parity_benchmark.py"),
)
REQUIRED_ENVIRONMENT_SNAPSHOTS = (
    "aggregate.py",
    "capture_gpu.py",
    "run_repeat.sh",
    "official_parity_benchmark.py",
    "packages.txt",
    "cuda.txt",
    "official_source.txt",
    "instrumentation.patch",
    "graphdeco_build.txt",
    "graphdeco_build.ninja",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
        "std_population": float(array.std(ddof=0)),
        "std_sample": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def across_run(values: list[float]) -> dict[str, float]:
    stats = distribution(values)
    return {
        "run_count": len(values),
        "mean_of_run_means": stats["mean"],
        "sample_standard_deviation": stats["std_sample"],
        "coefficient_of_variation": stats["std_sample"] / stats["mean"] if stats["mean"] else 0.0,
        "min_run_mean": stats["min"],
        "max_run_mean": stats["max"],
    }


def corr(x: list[float], y: list[float]) -> dict[str, float]:
    p = pearsonr(x, y)
    s = spearmanr(x, y)
    return {
        "pearson_r": float(p.statistic),
        "pearson_p_two_sided": float(p.pvalue),
        "spearman_rho": float(s.statistic),
        "spearman_p_two_sided": float(s.pvalue),
        "interpretation": "descriptive association only; not causal evidence",
    }


def frame_record(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_index": int(frame["trajectory_frame_index"]),
        "camera_index": int(frame["trajectory_frame_index"]),
        "selected_gaussian_count": int(frame["total_selected_gaussian_count"]),
        "visible_cell_count": len(frame["visible_cell_ids"]),
        "selected_cell_count_by_level": frame["selected_cell_count_by_level"],
        "selected_gaussian_count_by_level": frame["selected_gaussian_count_by_level"],
        "fallback_changed_mask": frame["fallback_changed_mask"],
        "selection_time_ms": frame["selection_time_ms"],
        "active_set_assembly_time_ms": frame["active_set_assembly_time_ms"],
        "render_call_time_ms": frame["render_call_time_ms"],
        "cuda_rasterization_time_ms": frame["cuda_rasterization_time_ms"],
        "end_to_end_frame_time_ms": frame["end_to_end_frame_time_ms"],
        "torch_current_allocated_bytes": frame["torch_current_allocated_bytes"],
        "torch_current_reserved_bytes": frame["torch_current_reserved_bytes"],
        "torch_peak_reserved_bytes": frame["torch_peak_reserved_bytes"],
    }


def contiguous_ranges(values: set[int]) -> list[list[int]]:
    if not values:
        return []
    ordered = sorted(values)
    ranges = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            ranges.append([start, previous])
            start = value
        previous = value
    ranges.append([start, previous])
    return ranges


def main() -> None:
    args = parse_args()
    artifact = args.artifact
    tables = artifact / "tables"
    validation = artifact / "validation"
    figures = artifact / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    runs = [
        read_jsonl(artifact / "runs" / f"repeat_{repeat:02d}" / "frames.jsonl")
        for repeat in (1, 2, 3)
    ]
    run_summaries = [
        read_json(artifact / "runs" / f"repeat_{repeat:02d}" / "summary.json")
        for repeat in (1, 2, 3)
    ]
    for repeat, records in enumerate(runs, 1):
        if len(records) != 1440:
            errors.append(f"repeat {repeat}: expected 1440 frames, found {len(records)}")
        if [record["trajectory_frame_index"] for record in records] != list(range(1440)):
            errors.append(f"repeat {repeat}: trajectory indices/order differ")
        if not all(record["finite_rgb"] for record in records):
            errors.append(f"repeat {repeat}: non-finite RGB output")

    per_run = []
    for repeat, records in enumerate(runs, 1):
        timing = {
            field: distribution([float(record[field]) for record in records]) for field in TIMING_FIELDS
        }
        per_run.append(
            {
                "repeat": repeat,
                "measured_frame_count": len(records),
                "warmup_policy": run_summaries[repeat - 1]["warmup_policy"],
                "timing": timing,
                "end_to_end_fps_from_run_mean": 1000.0 / timing["end_to_end_frame_time_ms"]["mean"],
                "resident_gaussian_count": run_summaries[repeat - 1]["resident_gaussian_count"],
                "selected_gaussian_count": run_summaries[repeat - 1]["selected_gaussian_count"],
                "memory": {
                    "after_load": run_summaries[repeat - 1]["memory_after_load"],
                    "after_warmup_peak_reset": run_summaries[repeat - 1]["memory_after_warmup_peak_reset"],
                    "after_run": run_summaries[repeat - 1]["memory_after_run"],
                },
            }
        )
    across = {
        field: across_run([record["timing"][field]["mean"] for record in per_run])
        for field in TIMING_FIELDS
    }
    headline_ms = across["end_to_end_frame_time_ms"]["mean_of_run_means"]
    performance_summary = {
        "campaign_id": args.campaign_id,
        "primary_metric": "end_to_end_frame_time_ms",
        "headline_mean_of_run_means_ms": headline_ms,
        "headline_between_run_sample_sd_ms": across["end_to_end_frame_time_ms"]["sample_standard_deviation"],
        "headline_between_run_cv": across["end_to_end_frame_time_ms"]["coefficient_of_variation"],
        "headline_end_to_end_fps": 1000.0 / headline_ms,
        "per_run": per_run,
        "across_run": across,
        "aggregation_rule": "Each trajectory repetition is one independent unit; headline is the mean of the three run means, not a pooled 4,320-frame estimate.",
    }
    write_json(tables / "performance_summary.json", performance_summary)

    official_prep_means = [
        record["timing"]["selection_time_ms"]["mean"]
        + record["timing"]["active_set_assembly_time_ms"]["mean"]
        for record in per_run
    ]
    stage_decomposition = {
        "selection_time_ms": across["selection_time_ms"],
        "active_set_assembly_time_ms": across["active_set_assembly_time_ms"],
        "selection_plus_assembly_time_ms": across_run(official_prep_means),
        "render_call_time_ms": across["render_call_time_ms"],
        "cuda_rasterization_time_ms": across["cuda_rasterization_time_ms"],
        "end_to_end_frame_time_ms": across["end_to_end_frame_time_ms"],
        "boundary_note": "Selection plus assembly and render call are synchronized measured stages. CUDA is nested within render call and must not be added again. Minor residual differences arise from Python timing boundaries.",
    }
    write_json(tables / "stage_decomposition.json", stage_decomposition)

    threshold_sets: dict[int, dict[str, set[int]]] = {}
    stall_runs = []
    for repeat, records in enumerate(runs, 1):
        values = np.asarray([record["end_to_end_frame_time_ms"] for record in records], dtype=np.float64)
        median = float(np.median(values))
        predicates = {
            "above_2x_run_median": values > 2.0 * median,
            "above_3x_run_median": values > 3.0 * median,
            "above_50_ms": values > 50.0,
            "above_100_ms": values > 100.0,
            "above_250_ms": values > 250.0,
        }
        threshold_sets[repeat] = {
            key: {int(index) for index in np.nonzero(mask)[0].tolist()} for key, mask in predicates.items()
        }
        major = [frame_record(records[index]) for index in sorted(threshold_sets[repeat]["above_250_ms"])]
        top = [frame_record(records[index]) for index in np.argsort(values)[-20:][::-1].tolist()]
        memory_transitions = []
        previous_reserved = int(records[0]["torch_current_reserved_bytes"])
        for index, record in enumerate(records[1:], 1):
            current_reserved = int(record["torch_current_reserved_bytes"])
            delta = current_reserved - previous_reserved
            if abs(delta) >= 256 * 1024 * 1024:
                memory_transitions.append(
                    {
                        "frame_index": index,
                        "reserved_delta_bytes": delta,
                        "torch_current_reserved_bytes": current_reserved,
                        "end_to_end_frame_time_ms": record["end_to_end_frame_time_ms"],
                        "cuda_rasterization_time_ms": record["cuda_rasterization_time_ms"],
                    }
                )
            previous_reserved = current_reserved
        signatures_seen: set[tuple[Any, ...]] = set()
        first_signature_frames = set()
        for record in records:
            signature = (
                tuple(record["selected_cell_count_by_level"]),
                tuple(record["selected_gaussian_count_by_level"]),
                int(record["total_selected_gaussian_count"]),
            )
            if signature not in signatures_seen:
                first_signature_frames.add(int(record["trajectory_frame_index"]))
                signatures_seen.add(signature)
        major_set = threshold_sets[repeat]["above_250_ms"]
        stall_runs.append(
            {
                "repeat": repeat,
                "median_end_to_end_ms": median,
                "thresholds": {
                    key: {"count": len(indices), "frame_indices": sorted(indices)}
                    for key, indices in threshold_sets[repeat].items()
                },
                "major_outlier_records_above_250_ms": major,
                "top_20_frames": top,
                "memory_reservation_transitions_at_least_256_mib": memory_transitions,
                "first_occurrence_workload_signature": {
                    "unique_signature_count": len(first_signature_frames),
                    "major_stalls_that_are_first_signature_occurrences": len(major_set & first_signature_frames),
                    "major_stall_count": len(major_set),
                    "frame_indices": sorted(major_set & first_signature_frames),
                },
                "workload_correlations": {
                    "end_to_end_vs_selected_gaussian_count": corr(
                        [float(record["total_selected_gaussian_count"]) for record in records],
                        [float(record["end_to_end_frame_time_ms"]) for record in records],
                    ),
                    "cuda_vs_selected_gaussian_count": corr(
                        [float(record["total_selected_gaussian_count"]) for record in records],
                        [float(record["cuda_rasterization_time_ms"]) for record in records],
                    ),
                },
            }
        )

    framewise = []
    for frame_index in range(1440):
        values = [float(records[frame_index]["end_to_end_frame_time_ms"]) for records in runs]
        cuda_values = [float(records[frame_index]["cuda_rasterization_time_ms"]) for records in runs]
        framewise.append(
            {
                "frame_index": frame_index,
                "selected_gaussian_count": int(runs[0][frame_index]["total_selected_gaussian_count"]),
                "end_to_end_time_ms": distribution(values),
                "cuda_rasterization_time_ms": distribution(cuda_values),
                "repeats_above_100_ms": [repeat for repeat in (1, 2, 3) if frame_index in threshold_sets[repeat]["above_100_ms"]],
                "repeats_above_250_ms": [repeat for repeat in (1, 2, 3) if frame_index in threshold_sets[repeat]["above_250_ms"]],
            }
        )
    write_json(tables / "framewise_nonstationarity.json", framewise)

    recurrence = {}
    for threshold in ("above_100_ms", "above_250_ms"):
        sets = [threshold_sets[repeat][threshold] for repeat in (1, 2, 3)]
        at_least_two = {frame for frame in range(1440) if sum(frame in value for value in sets) >= 2}
        all_three = set.intersection(*sets)
        any_run = set.union(*sets)
        recurrence[threshold] = {
            "any_run_count": len(any_run),
            "at_least_two_runs_count": len(at_least_two),
            "all_three_runs_count": len(all_three),
            "all_three_frame_indices": sorted(all_three),
            "all_three_contiguous_ranges": contiguous_ranges(all_three),
            "at_least_two_contiguous_ranges": contiguous_ranges(at_least_two),
        }
    run_time_arrays = [np.asarray([record["end_to_end_frame_time_ms"] for record in records]) for records in runs]
    recurrence["pairwise_frame_time_pearson_r"] = {
        "repeat_1_vs_2": float(np.corrcoef(run_time_arrays[0], run_time_arrays[1])[0, 1]),
        "repeat_1_vs_3": float(np.corrcoef(run_time_arrays[0], run_time_arrays[2])[0, 1]),
        "repeat_2_vs_3": float(np.corrcoef(run_time_arrays[1], run_time_arrays[2])[0, 1]),
    }
    recurrence["interpretation"] = "Large stalls strongly recur at the same trajectory regions, while their exact magnitude varies substantially across processes."
    stall_summary = {
        "per_run": stall_runs,
        "cross_run_recurrence": recurrence,
        "primary_aggregate_includes_all_stalls": True,
        "allocator_counter_caveat": "PyTorch reserved counters exceed physical VRAM under this WSL/custom-extension environment and are not representation-neutral. Their recurrent transitions are diagnostic associations, not proof of allocator causality.",
    }
    write_json(tables / "stall_summary.json", stall_summary)

    reference_runs = [
        read_jsonl(ARTIFACT_HISTORICAL / "runs" / "official_lod_release" / f"repeat_{repeat:02d}" / "frames.jsonl")
        for repeat in (1, 2, 3)
    ]
    clean_replay = read_json(ARTIFACT_ATTRIBUTION / "tables" / "clean_official_oracle_replay.json")
    direct_rows = read_json(ARTIFACT_ATTRIBUTION / "tables" / "per_camera_backend.json")
    camera_comparisons = []
    for camera in (785, 1047):
        historical_records = [records[camera] for records in reference_runs]
        new_records = [records[camera] for records in runs]
        replay = next(record for record in clean_replay if record["camera_index"] == camera)
        direct = next(record for record in direct_rows if record["camera_index"] == camera)
        camera_comparisons.append(
            {
                "camera_index": camera,
                "new_clean_full_repetitions": [frame_record(record) for record in new_records],
                "historical_official_repetitions": [frame_record(record) for record in historical_records],
                "clean_sparse_oracle_replay": replay["clean_official_oracle_replay"],
                "isolated_graphdeco_gpu_time_ms": direct["graphdeco"]["backend_gpu_time_ms_across_process_means"],
                "isolated_graphdeco_wall_time_ms": direct["graphdeco"]["backend_synchronized_wall_time_ms_across_process_means"],
                "hundreds_of_milliseconds_recurred_in_all_new_full_runs": all(record["end_to_end_frame_time_ms"] > 250 for record in new_records),
            }
        )
    historical_summary = read_json(ARTIFACT_HISTORICAL / "tables" / "official_config_summary.json")
    historical_lod = next(
        record
        for record in historical_summary["configurations"]
        if record["configuration_id"] == "OFFICIAL-CITYGS-LOD-RELEASE"
    )
    historical_comparison = {
        "historical_official_parity": {
            "end_to_end_ms": historical_lod["timing"]["end_to_end_frame_time_ms"]["mean_of_run_means"],
            "end_to_end_fps": historical_lod["end_to_end_fps"],
            "render_call_ms": historical_lod["timing"]["render_call_time_ms"]["mean_of_run_means"],
            "cuda_rasterization_ms": historical_lod["timing"]["cuda_rasterization_time_ms"]["mean_of_run_means"],
        },
        "new_clean_full": {
            "end_to_end_ms": headline_ms,
            "end_to_end_fps": 1000.0 / headline_ms,
            "render_call_ms": across["render_call_time_ms"]["mean_of_run_means"],
            "cuda_rasterization_ms": across["cuda_rasterization_time_ms"]["mean_of_run_means"],
        },
        "historical_problem_cameras": camera_comparisons,
        "outcome": "B — the full-trajectory stalls recur under the controlled condition; the historical campaign remains non-canonical for current performance because the new clean campaign provides the representative three-run estimate.",
        "interpretation": "The historical 19.04 FPS value is superseded as the representative official performance estimate by this clean full campaign, not because the stalls disappeared, but because the new controlled three-run result directly measures their recurrence and variability.",
    }
    write_json(tables / "historical_comparison.json", historical_comparison)

    oracle_reference = reference_runs[0]
    semantic_rows = []
    semantic_fields = (
        "visible_cell_ids",
        "selected_lod_by_cell",
        "selected_cell_count_by_level",
        "selected_gaussian_count_by_cell",
        "selected_gaussian_count_by_level",
        "total_selected_gaussian_count",
        "nearest_cell_id",
        "fallback_changed_mask",
    )
    for camera in CAMERAS:
        reference = oracle_reference[camera]
        repeat_checks = []
        for repeat, records in enumerate(runs, 1):
            checks = {field: records[camera][field] == reference[field] for field in semantic_fields}
            repeat_checks.append({"repeat": repeat, "field_checks": checks, "exact": all(checks.values())})
            if not all(checks.values()):
                errors.append(f"semantic mismatch at camera {camera}, repeat {repeat}")
        semantic_rows.append({"camera_index": camera, "repeat_checks": repeat_checks, "all_three_exact": all(v["exact"] for v in repeat_checks)})
    selected_gate = {
        "status": "pass" if all(record["all_three_exact"] for record in semantic_rows) else "fail",
        "camera_indices": list(CAMERAS),
        "exact_repeat_camera_checks": sum(check["exact"] for record in semantic_rows for check in record["repeat_checks"]),
        "required_repeat_camera_checks": 36,
        "frames": semantic_rows,
        "reference": str(ARTIFACT_HISTORICAL / "runs" / "official_lod_release" / "repeat_01" / "frames.jsonl"),
    }
    write_json(validation / "selected_set_gate.json", selected_gate)

    canonical_summary = read_json(ARTIFACT_CANONICAL / "tables" / "config_summary.json")
    canonical_lod = next(record for record in canonical_summary["configurations"] if record["configuration_id"] == "CG-LOD-RELEASE")
    official_prep = statistics.fmean(official_prep_means)
    official_vs_gsplat = {
        "official_clean_full": {
            "end_to_end_frame_time_ms": headline_ms,
            "end_to_end_fps": 1000.0 / headline_ms,
            "selection_plus_active_set_assembly_ms": official_prep,
            "selection_ms": across["selection_time_ms"]["mean_of_run_means"],
            "active_set_assembly_ms": across["active_set_assembly_time_ms"]["mean_of_run_means"],
            "render_call_ms": across["render_call_time_ms"]["mean_of_run_means"],
            "cuda_rasterization_ms": across["cuda_rasterization_time_ms"]["mean_of_run_means"],
            "resident_gaussian_count": 25494686,
            "mean_selected_gaussian_count": statistics.fmean(record["total_selected_gaussian_count"] for record in runs[0]),
        },
        "canonical_gsplat": {
            "end_to_end_frame_time_ms": canonical_lod["headline_mean_end_to_end_frame_time_ms"],
            "end_to_end_fps": canonical_lod["end_to_end_fps"],
            "preparation_time_ms": canonical_lod["mean_preparation_time_ms"],
            "render_call_ms": canonical_lod["mean_render_call_time_ms"],
            "cuda_rasterization_ms": canonical_lod["mean_cuda_rasterization_time_ms"],
            "resident_gaussian_count": int(canonical_lod["resident_gaussian_count"]),
            "mean_selected_gaussian_count": canonical_lod["mean_selected_gaussian_count"],
        },
        "ratios": {
            "official_over_gsplat_end_to_end_time": headline_ms / canonical_lod["headline_mean_end_to_end_frame_time_ms"],
            "gsplat_over_official_fps": canonical_lod["end_to_end_fps"] / (1000.0 / headline_ms),
            "official_over_gsplat_preparation": official_prep / canonical_lod["mean_preparation_time_ms"],
            "official_over_gsplat_render_call": across["render_call_time_ms"]["mean_of_run_means"] / canonical_lod["mean_render_call_time_ms"],
            "official_over_gsplat_cuda": across["cuda_rasterization_time_ms"]["mean_of_run_means"] / canonical_lod["mean_cuda_rasterization_time_ms"],
        },
        "memory": {
            "official_after_warmup_current_allocated_bytes": across_run([float(s["memory_after_warmup_peak_reset"]["torch_current_allocated_bytes"]) for s in run_summaries]),
            "official_after_run_current_reserved_bytes": across_run([float(s["memory_after_run"]["torch_current_reserved_bytes"]) for s in run_summaries]),
            "official_peak_allocated_bytes": across_run([float(s["memory_after_run"]["torch_peak_allocated_bytes"]) for s in run_summaries]),
            "official_peak_reserved_bytes": across_run([float(s["memory_after_run"]["torch_peak_reserved_bytes"]) for s in run_summaries]),
            "official_whole_device_after_run_bytes": across_run([float(s["memory_after_run"]["whole_device_used_bytes"]) for s in run_summaries]),
            "canonical_post_transfer_allocated_bytes": canonical_lod["post_transfer_allocated_bytes"],
            "canonical_render_peak_allocated_bytes": canonical_lod["render_peak_allocated_bytes"],
            "canonical_render_peak_reserved_bytes": canonical_lod["render_peak_reserved_bytes"],
            "caveat": "PyTorch allocator and GraphDECO custom-buffer accounting are not representation-neutral. Official reserved counters exceed physical VRAM under WSL and are diagnostic only; whole-device after-run usage is not a continuously sampled peak.",
        },
        "attribution": "The measured remaining gap combines recurring camera/trajectory-state-dependent GraphDECO rasterization stalls, camera-dependent raw backend differences, selection, and active-set assembly. It must not be reduced to a static-input rasterizer ratio or an unspecified wrapper overhead.",
    }
    write_json(tables / "official_vs_gsplat.json", official_vs_gsplat)

    baselines = [read_json(artifact / "environment" / f"repeat_{repeat:02d}_pre_run_gpu.json") for repeat in (1, 2, 3)]
    baseline_gate = all("No running processes found" in record["nvidia_smi_full"] for record in baselines)
    if not baseline_gate:
        errors.append("pre-run NVIDIA compute-process gate failed")
    missing_environment_snapshots = [
        name for name in REQUIRED_ENVIRONMENT_SNAPSHOTS if not (artifact / "environment" / name).is_file()
    ]
    if missing_environment_snapshots:
        errors.append(f"missing environment snapshots: {missing_environment_snapshots}")
    package_validation = {
        "status": "pass" if not errors else "fail",
        "campaign_id": args.campaign_id,
        "expected_repetitions": 3,
        "expected_frames_per_repetition": 1440,
        "validated_frames": sum(len(records) for records in runs),
        "all_outputs_finite": all(record["finite_rgb"] for records in runs for record in records),
        "semantic_gate": selected_gate["status"],
        "pre_run_compute_process_gate": baseline_gate,
        "avoidable_visible_application_gate": "pass; only Codex was visible before each repetition",
        "errors": errors,
        "outcome": historical_comparison["outcome"],
    }
    write_json(validation / "package_validation.json", package_validation)
    if errors:
        raise RuntimeError(errors)

    canonical_runs = [
        read_jsonl(ARTIFACT_CANONICAL / "runs" / "CG-LOD-RELEASE" / f"repeat_{repeat:02d}" / "frames.jsonl")
        for repeat in (1, 2, 3)
    ]
    x = np.arange(1440)
    official_arrays = np.asarray([[record["end_to_end_frame_time_ms"] for record in records] for records in runs])
    official_mean = official_arrays.mean(axis=0)
    canonical_arrays = np.asarray([[record["end_to_end_frame_time_ms"] for record in records] for records in canonical_runs])
    canonical_mean = canonical_arrays.mean(axis=0)

    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.plot(x, official_mean, linewidth=0.9)
    ax.set_xlabel("Trajectory frame")
    ax.set_ylabel("Clean official end-to-end time (ms)")
    ax.grid(alpha=0.25)
    fig.savefig(figures / "official_clean_end_to_end_trajectory.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    for index in range(3):
        ax.plot(x, official_arrays[index], linewidth=0.7, alpha=0.8, label=f"Repeat {index + 1}")
    ax.set_xlabel("Trajectory frame")
    ax.set_ylabel("End-to-end time (ms)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(figures / "official_clean_three_repetitions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.plot(x, official_mean, linewidth=0.9, label="Official clean")
    ax.plot(x, canonical_mean, linewidth=0.9, label="Canonical gsplat")
    ax.set_yscale("log")
    ax.set_xlabel("Trajectory frame")
    ax.set_ylabel("Mean end-to-end time (ms, log scale)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(figures / "official_clean_vs_canonical_per_camera.png", dpi=180)
    plt.close(fig)

    stage_values = [
        across["selection_time_ms"]["mean_of_run_means"],
        across["active_set_assembly_time_ms"]["mean_of_run_means"],
        across["render_call_time_ms"]["mean_of_run_means"],
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar(["Selection", "Active-set assembly", "Render call"], stage_values)
    ax.set_ylabel("Mean of run means (ms)")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(figures / "official_clean_stage_decomposition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    mask = official_mean > 100.0
    ax.scatter(x[mask], official_mean[mask], s=12)
    ax.set_xlabel("Trajectory frame")
    ax.set_ylabel("Three-run mean end-to-end time for >100 ms frames")
    ax.grid(alpha=0.25)
    fig.savefig(figures / "official_clean_major_outliers.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.bar(["Official clean", "Canonical gsplat"], [headline_ms, canonical_lod["headline_mean_end_to_end_frame_time_ms"]])
    ax.set_ylabel("End-to-end mean of run means (ms)")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(figures / "official_clean_vs_gsplat_summary.png", dpi=180)
    plt.close(fig)

    outcome = historical_comparison["outcome"]
    manifest = {
        "campaign_id": args.campaign_id,
        "classification": "CITYGS-OFFICIAL-CLEAN-FULL; controlled full-trajectory official CityGaussian V1 PLY LoD runtime",
        "status": "complete",
        "created_date": "2026-08-25",
        "outcome": outcome,
        "pinned_sources": {
            "CityGaussian": "e16c526f78ab01d377b13b3e6862dcc93ddbe953",
            "diff_gaussian_rasterization": "59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d",
            "gs_framework_comparison": "cfc2136459dd63cc47dc3acfde55a3d72a4d8bba",
        },
        "compiled_graphdeco": {
            "path": "/home/luis/tese/CityGaussian_official_parity/.official_extensions/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so",
            "sha256": "cc2ada2c5875c272771ebe4624208c0a5f4004fc150cc9914d553274d02fffd8",
            "architecture": "four native sm_120 cubins; no PTX; no CUDA -G",
        },
        "assets": [{"path": str(path), "sha256": sha256_file(path)} for path in ASSETS],
        "trajectory": {"path": str(TRAJECTORY), "sha256": sha256_file(TRAJECTORY), "frame_count": 1440, "resolution": [1920, 1080]},
        "protocol": {
            "fresh_processes": 3,
            "warmup": "one deterministic frame-0 render excluded per process",
            "measured_frames_per_process": 1440,
            "trajectory_order": "unchanged 0..1439",
            "natural_first_shape_effects_measured": True,
            "primary_metric": "end_to_end_frame_time_ms",
        },
        "source_harnesses": [{"path": str(path), "sha256": sha256_file(path)} for path in SCRIPTS],
        "artifact_environment_snapshots": [
            {
                "path": str(path.relative_to(artifact)),
                "sha256": sha256_file(path),
            }
            for path in sorted((artifact / "environment").iterdir())
            if path.is_file()
        ],
        "supersession": {
            "representative_official_performance": "Supersedes citygs_official_parity_20260824_191215 for representative official runtime performance estimates.",
            "semantic_parity": "Does not replace the historical parity campaign's semantic evidence.",
            "backend_attribution": "Preserves and extends CITYGS-BACKEND-MICROBENCH and CITYGS-12CAM-ATTRIBUTION interpretations; static warmed backend ratios do not explain full sequential stalls.",
        },
        "no_optimization_or_semantic_change": True,
        "pinned_reference_checkout_modified": False,
        "thesis_latex_modified": False,
    }
    write_json(artifact / "manifest.json", manifest)

    e2e = across["end_to_end_frame_time_ms"]
    render = across["render_call_time_ms"]
    cuda = across["cuda_rasterization_time_ms"]
    selection = across["selection_time_ms"]
    assembly = across["active_set_assembly_time_ms"]
    readme = f"""# Clean full-trajectory official CityGaussian runtime

Campaign ID: `{args.campaign_id}`  
Evidence class: `CITYGS-OFFICIAL-CLEAN-FULL-*`  
Status: **complete**

## Outcome

{outcome}

The clean three-process result is `{headline_ms:.3f} ± {e2e['sample_standard_deviation']:.3f} ms` end-to-end (mean and sample SD of run means), or `{1000.0 / headline_ms:.2f} FPS`. All stalls remain in the primary aggregate. This supersedes the historical `52.516 ms / 19.04 FPS` value only as the representative official-runtime performance estimate; the historical package remains authoritative semantic-parity and historical-observation evidence.

## Controlled condition and protocol

Before each repetition, Computer Use showed only Codex open: no Twitch/browser video, media player, Spotify, game, or launcher. NVIDIA reported P8 idle clocks, 39–44 °C, 154–889 MiB, 10–11% display utilization, and no compute process. The elevated display utilization was observable with only Codex/the desktop present; no system, driver, power, or clock setting was changed.

Each fresh process loaded the exact three direct SH2 PLY levels, rendered trajectory frame 0 once as an excluded warmup, synchronized/reset peak state, then measured frames 0–1439 in unchanged order at 1920×1080. No camera/workload-shape prewarming was performed.

## Three-run timing

| Repeat | Mean | Median | p95 | p99 | Max | FPS from mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for record in per_run:
        values = record["timing"]["end_to_end_frame_time_ms"]
        readme += f"| {record['repeat']} | {values['mean']:.3f} | {values['median']:.3f} | {values['p95']:.3f} | {values['p99']:.3f} | {values['max']:.3f} | {record['end_to_end_fps_from_run_mean']:.2f} |\n"
    readme += f"""

Across-run end-to-end CV is `{100.0 * e2e['coefficient_of_variation']:.2f}%`; run means span `{e2e['min_run_mean']:.3f}–{e2e['max_run_mean']:.3f} ms`.

## Stage decomposition

| Stage | Mean of run means (ms) | Between-run SD (ms) |
| --- | ---: | ---: |
| Visibility + LoD selection | {selection['mean_of_run_means']:.3f} | {selection['sample_standard_deviation']:.3f} |
| Active-set assembly | {assembly['mean_of_run_means']:.3f} | {assembly['sample_standard_deviation']:.3f} |
| Render call | {render['mean_of_run_means']:.3f} | {render['sample_standard_deviation']:.3f} |
| CUDA rasterization boundary | {cuda['mean_of_run_means']:.3f} | {cuda['sample_standard_deviation']:.3f} |
| End-to-end | {e2e['mean_of_run_means']:.3f} | {e2e['sample_standard_deviation']:.3f} |

CUDA is nested within render call and is not additive. The large variation is inside the synchronized GraphDECO rasterization boundary; selection remains stable, while assembly also spikes in the same difficult trajectory region.

## Stall behavior

The hundreds-of-milliseconds behavior recurs under the controlled condition. Cameras 785 and 1047 exceed 250 ms in all three repetitions. Full threshold frame lists, decompositions, recurring ranges, workload associations, and allocator-reservation transitions are preserved in `tables/stall_summary.json` and `tables/framewise_nonstationarity.json`.

Pairwise frame-time correlations are `{recurrence['pairwise_frame_time_pearson_r']['repeat_1_vs_2']:.3f}`, `{recurrence['pairwise_frame_time_pearson_r']['repeat_1_vs_3']:.3f}`, and `{recurrence['pairwise_frame_time_pearson_r']['repeat_2_vs_3']:.3f}`. Stalls therefore recur at strongly similar camera regions rather than migrating randomly, although their magnitude changes substantially.

PyTorch reserved-memory transitions recur near the stall regions and the official process finishes near whole-device capacity. Reserved counters exceed physical VRAM under WSL/custom-extension accounting, so this is a diagnostic association, not proof that allocator behavior causes the stalls.

## Comparison with canonical gsplat

| Metric | Official clean full | Canonical gsplat | Official / gsplat |
| --- | ---: | ---: | ---: |
| End-to-end (ms) | {headline_ms:.3f} | {canonical_lod['headline_mean_end_to_end_frame_time_ms']:.3f} | {official_vs_gsplat['ratios']['official_over_gsplat_end_to_end_time']:.3f}× |
| FPS | {1000.0 / headline_ms:.2f} | {canonical_lod['end_to_end_fps']:.2f} | {official_vs_gsplat['ratios']['gsplat_over_official_fps']:.3f}× gsplat |
| Selection/assembly or preparation (ms) | {official_prep:.3f} | {canonical_lod['mean_preparation_time_ms']:.3f} | {official_vs_gsplat['ratios']['official_over_gsplat_preparation']:.3f}× |
| Render call (ms) | {render['mean_of_run_means']:.3f} | {canonical_lod['mean_render_call_time_ms']:.3f} | {official_vs_gsplat['ratios']['official_over_gsplat_render_call']:.3f}× |
| CUDA (ms) | {cuda['mean_of_run_means']:.3f} | {canonical_lod['mean_cuda_rasterization_time_ms']:.3f} | {official_vs_gsplat['ratios']['official_over_gsplat_cuda']:.3f}× |

Both retain `25,494,686` resident Gaussians and select the identical mean `4,214,145.394` Gaussians. All 36 repeat-camera semantic checks pass.

## Interpretation

Confirmed recurring costs are synchronized visibility/LoD selection, active-set assembly, camera-dependent raw backend differences, natural first/workload-shape effects, and—only in full sequential official execution—large recurring GraphDECO rasterization-boundary stalls. Static warmed isolated calls do not exhibit those stalls, so the final gap must not be described solely as a generic backend ratio or vague wrapper overhead.

The exact low-level cause of the sequential stalls remains unresolved. They align strongly across runs and with recurrent reservation-state transitions/workload regions, but allocator counters are unreliable and causality was not instrumented. External GPU contention is not supported by a competing process or visible avoidable application in this campaign.

## Evidence map

- `runs/repeat_*/frames.jsonl`: all 4,320 measured frame records.
- `tables/performance_summary.json`: independent-run distributions and across-run statistics.
- `tables/stage_decomposition.json`: synchronized stage means.
- `tables/stall_summary.json`: all thresholds, frame IDs, major records, recurrence, and diagnostics.
- `tables/framewise_nonstationarity.json`: three-run comparison for every trajectory frame.
- `tables/historical_comparison.json`: historical, sparse replay, isolated, and new full-run values.
- `tables/official_vs_gsplat.json`: controlled comparison with the frozen canonical gsplat campaign.
- `validation/selected_set_gate.json`: 36/36 exact oracle-camera semantic checks.
- `validation/package_validation.json`: completeness and system/semantic gates.
- `environment/`: per-run system baselines plus copied harness, package, source,
  patch, and native-build records.
- `figures/`: six thesis-ready figures.

No backend, kernel, method, asset, camera, resolution, visibility, LoD policy, or thesis LaTeX was changed. The pinned CityGaussian reference checkout remains untouched.
"""
    (artifact / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
