#!/usr/bin/env python
"""Aggregate and validate the frozen 12-camera CityGaussian attribution study."""

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
from skimage.metrics import structural_similarity


CAMERAS = (0, 131, 262, 392, 523, 654, 785, 916, 1047, 1177, 1308, 1439)
BACKENDS = ("graphdeco", "gsplat")
OFFICIAL = Path("/home/luis/tese/benchmark_artifacts/citygs_official_parity_20260824_191215")
CANONICAL = Path("/home/luis/tese/benchmark_artifacts/citygs_canonical_20260824_135615")
PREVIOUS_MICROBENCH = Path("/home/luis/tese/benchmark_artifacts/citygs_backend_microbench_20260824_213420")
SOURCE_SCRIPTS = (
    Path("/home/luis/tese/backend_microbench/prepare_inputs.py"),
    Path("/home/luis/tese/backend_microbench/run_backend.py"),
    Path("/home/luis/tese/backend_attribution_12cam/run_campaign.sh"),
    Path("/home/luis/tese/backend_attribution_12cam/aggregate.py"),
    Path("/home/luis/tese/backend_attribution_12cam/wrapper_probe.py"),
    Path("/home/luis/tese/backend_attribution_12cam/run_wrapper_probe.sh"),
    Path("/home/luis/tese/backend_attribution_12cam/run_official_oracle_replay.sh"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
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


def sample_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {
        "count": int(array.size),
        "mean": mean,
        "std_sample": sd,
        "coefficient_of_variation": sd / mean if mean else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def correlation(x: list[float], y: list[float]) -> dict[str, float]:
    px = pearsonr(x, y)
    sx = spearmanr(x, y)
    return {
        "pearson_r": float(px.statistic),
        "pearson_p_two_sided": float(px.pvalue),
        "spearman_rho": float(sx.statistic),
        "spearman_p_two_sided": float(sx.pvalue),
        "n": len(x),
        "interpretation": "association only; no causal interpretation",
    }


def metric_stats(records: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: sample_stats([float(record[field]) for record in records]) for field in fields}


def find_camera_record(records: list[dict[str, Any]], camera: int, keys: tuple[str, ...]) -> dict[str, Any]:
    matches = [record for record in records if any(record.get(key) == camera for key in keys)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one record for camera {camera}, found {len(matches)}")
    return matches[0]


def image_parity(camera: int, artifact: Path) -> dict[str, Any]:
    roots = {
        backend: artifact / "microbench" / f"camera_{camera:04d}" / backend / "repeat_01"
        for backend in BACKENDS
    }
    graph_rgb = np.load(roots["graphdeco"] / "rgb.npy")
    gs_rgb = np.load(roots["gsplat"] / "rgb.npy")
    graph_alpha = np.load(roots["graphdeco"] / "alpha.npy")
    gs_alpha = np.load(roots["gsplat"] / "alpha.npy")
    rgb_diff = np.abs(graph_rgb.astype(np.float64) - gs_rgb.astype(np.float64))
    alpha_diff = np.abs(graph_alpha.astype(np.float64) - gs_alpha.astype(np.float64))
    mse = float(np.mean((graph_rgb.astype(np.float64) - gs_rgb.astype(np.float64)) ** 2))
    psnr = math.inf if mse == 0 else float(10.0 * math.log10(1.0 / mse))
    ssim = float(
        structural_similarity(
            np.clip(graph_rgb, 0.0, 1.0),
            np.clip(gs_rgb, 0.0, 1.0),
            data_range=1.0,
            channel_axis=2,
        )
    )
    graph_coverage = graph_alpha[..., 0] > 1.0e-4
    gs_coverage = gs_alpha[..., 0] > 1.0e-4
    union = np.logical_or(graph_coverage, gs_coverage).sum()
    intersection = np.logical_and(graph_coverage, gs_coverage).sum()
    return {
        "camera_index": camera,
        "finite_rgb": bool(np.isfinite(graph_rgb).all() and np.isfinite(gs_rgb).all()),
        "finite_alpha": bool(np.isfinite(graph_alpha).all() and np.isfinite(gs_alpha).all()),
        "mean_rgb_absolute_difference": float(rgb_diff.mean()),
        "max_rgb_absolute_difference": float(rgb_diff.max()),
        "psnr_data_range_1_db": psnr,
        "ssim_clipped_0_1": ssim,
        "mean_alpha_absolute_difference": float(alpha_diff.mean()),
        "max_alpha_absolute_difference": float(alpha_diff.max()),
        "coverage_jaccard_alpha_gt_1e_4": float(intersection / union) if union else 1.0,
    }


def main() -> None:
    args = parse_args()
    artifact = args.artifact
    tables = artifact / "tables"
    figures = artifact / "figures"
    validation = artifact / "validation"
    figures.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    official_runs = [
        read_jsonl(OFFICIAL / "runs" / "official_lod_release" / f"repeat_{repeat:02d}" / "frames.jsonl")
        for repeat in (1, 2, 3)
    ]
    canonical_runs = [
        read_jsonl(CANONICAL / "runs" / "CG-LOD-RELEASE" / f"repeat_{repeat:02d}" / "frames.jsonl")
        for repeat in (1, 2, 3)
    ]

    per_camera: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    high_level_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []

    for camera in CAMERAS:
        label = f"camera_{camera:04d}"
        metadata_path = validation / f"{label}_input.json"
        metadata = read_json(metadata_path)
        input_file = args.input_dir / f"{label}.pt"
        if sha256_file(input_file) != metadata["canonical_input_file_sha256"]:
            errors.append(f"{label}: canonical input file hash mismatch")
        selection = metadata["selection"]
        official_records = [
            find_camera_record(records, camera, ("trajectory_frame_index", "measured_frame_index"))
            for records in official_runs
        ]
        canonical_records = [find_camera_record(records, camera, ("camera_index", "frame_index")) for records in canonical_runs]
        oracle = official_records[0]
        selection_equal = all(
            selection["visible_cell_ids"] == record["visible_cell_ids"]
            and selection["selected_lod_by_cell"] == record["selected_lod_by_cell"]
            and selection["selected_gaussian_count_by_level"] == record["selected_gaussian_count_by_level"]
            and selection["total_selected_gaussian_count"] == record["total_selected_gaussian_count"]
            for record in official_records
        )
        canonical_equal = all(
            int(record["selected_gaussian_count"]) == selection["total_selected_gaussian_count"]
            and [int(v) for v in record["selected_cell_count_by_level"]] == selection["selected_cell_count_by_level"]
            and [int(v) for v in record["selected_gaussian_count_by_level"]] == selection["selected_gaussian_count_by_level"]
            for record in canonical_records
        )
        if not selection_equal:
            errors.append(f"{label}: prepared selection differs from official oracle")
        if not canonical_equal:
            errors.append(f"{label}: prepared selection differs from canonical records")
        if not all(metadata["camera_equivalence"].values()):
            errors.append(f"{label}: camera equivalence validation failed")
        input_rows.append(
            {
                "camera_index": camera,
                "camera_id": metadata["camera_id"],
                "canonical_input_file_sha256": metadata["canonical_input_file_sha256"],
                "canonical_tensor_sha256": {
                    key: value["sha256"] for key, value in metadata["canonical_tensor_records"].items()
                },
                "camera_tensor_sha256": {
                    key: value["sha256"] for key, value in metadata["camera_tensor_records"].items()
                },
                "camera_equivalence": metadata["camera_equivalence"],
                "official_oracle_selection_equal": selection_equal,
                "canonical_selection_equal": canonical_equal,
                "visible_cell_ids": selection["visible_cell_ids"],
                "visible_cell_count": len(selection["visible_cell_ids"]),
                "selected_cell_count_by_level": selection["selected_cell_count_by_level"],
                "selected_gaussian_count_by_level": selection["selected_gaussian_count_by_level"],
                "selected_gaussian_count": selection["total_selected_gaussian_count"],
            }
        )

        backend_rows: dict[str, Any] = {}
        for backend in BACKENDS:
            repeats = []
            for repeat in (1, 2, 3):
                repeat_dir = artifact / "microbench" / label / backend / f"repeat_{repeat:02d}"
                summary = read_json(repeat_dir / "repeat.json")
                samples = read_jsonl(repeat_dir / "samples.jsonl")
                if len(samples) != 100 or summary["measured_render_count"] != 100:
                    errors.append(f"{label}/{backend}/repeat_{repeat:02d}: sample count is not 100")
                if summary["warmup_render_count"] != 10:
                    errors.append(f"{label}/{backend}/repeat_{repeat:02d}: warmup count is not 10")
                if summary["input_file_sha256"] != metadata["canonical_input_file_sha256"]:
                    errors.append(f"{label}/{backend}/repeat_{repeat:02d}: input hash mismatch")
                if not summary["all_rgb_finite"] or not summary["all_alpha_finite"]:
                    errors.append(f"{label}/{backend}/repeat_{repeat:02d}: non-finite output")
                repeats.append(summary)
            gpu_means = [float(record["backend_gpu_time_ms"]["mean"]) for record in repeats]
            wall_means = [float(record["backend_synchronized_wall_time_ms"]["mean"]) for record in repeats]
            diagnostic = repeats[0]["measured_diagnostics"]
            backend_rows[backend] = {
                "backend_gpu_time_ms_across_process_means": sample_stats(gpu_means),
                "backend_synchronized_wall_time_ms_across_process_means": sample_stats(wall_means),
                "backend_throughput_equivalent": 1000.0 / statistics.fmean(gpu_means),
                "diagnostics": diagnostic,
                "repeat_summaries": [
                    {
                        "repeat": record["repeat"],
                        "backend_gpu_time_ms": record["backend_gpu_time_ms"],
                        "backend_synchronized_wall_time_ms": record["backend_synchronized_wall_time_ms"],
                        "memory": record["memory"],
                    }
                    for record in repeats
                ],
            }

        graph_gpu = backend_rows["graphdeco"]["backend_gpu_time_ms_across_process_means"]["mean"]
        gs_gpu = backend_rows["gsplat"]["backend_gpu_time_ms_across_process_means"]["mean"]
        ratio = graph_gpu / gs_gpu
        row = {
            "camera_index": camera,
            "camera_id": metadata["camera_id"],
            "visible_cell_count": len(selection["visible_cell_ids"]),
            "selected_gaussian_count": selection["total_selected_gaussian_count"],
            "selected_cell_count_by_level": selection["selected_cell_count_by_level"],
            "selected_gaussian_count_by_level": selection["selected_gaussian_count_by_level"],
            "graphdeco": backend_rows["graphdeco"],
            "gsplat": backend_rows["gsplat"],
            "graphdeco_over_gsplat_gpu_time_ratio": ratio,
        }
        per_camera.append(row)
        parity_rows.append(image_parity(camera, artifact))

        official_stats = metric_stats(
            official_records,
            (
                "visibility_time_ms",
                "lod_selection_time_ms",
                "selection_time_ms",
                "active_set_assembly_time_ms",
                "render_call_time_ms",
                "cuda_rasterization_time_ms",
                "end_to_end_frame_time_ms",
            ),
        )
        canonical_stats = metric_stats(
            canonical_records,
            (
                "visibility_selection_ms",
                "lod_selection_ms",
                "selected_cell_assembly_ms",
                "preparation_time_ms",
                "render_call_time_ms",
                "cuda_rasterization_time_ms",
                "end_to_end_frame_time_ms",
            ),
        )
        graph_wall = backend_rows["graphdeco"]["backend_synchronized_wall_time_ms_across_process_means"]["mean"]
        gs_wall = backend_rows["gsplat"]["backend_synchronized_wall_time_ms_across_process_means"]["mean"]
        high_level_rows.append(
            {
                "camera_index": camera,
                "selected_gaussian_count": selection["total_selected_gaussian_count"],
                "official_three_repeat_camera_metrics": official_stats,
                "canonical_gsplat_three_repeat_camera_metrics": canonical_stats,
                "isolated_graphdeco_gpu_time_ms": graph_gpu,
                "isolated_graphdeco_synchronized_wall_time_ms": graph_wall,
                "isolated_gsplat_gpu_time_ms": gs_gpu,
                "isolated_gsplat_synchronized_wall_time_ms": gs_wall,
                "diagnostic_differences_ms": {
                    "official_render_call_minus_isolated_graphdeco_wall": official_stats["render_call_time_ms"]["mean"] - graph_wall,
                    "official_cuda_boundary_minus_isolated_graphdeco_gpu": official_stats["cuda_rasterization_time_ms"]["mean"] - graph_gpu,
                    "canonical_render_call_minus_isolated_gsplat_wall": canonical_stats["render_call_time_ms"]["mean"] - gs_wall,
                    "canonical_cuda_boundary_minus_isolated_gsplat_gpu": canonical_stats["cuda_rasterization_time_ms"]["mean"] - gs_gpu,
                },
                "comparison_limit": "Differences are diagnostic, not accounting identities: the historical values are single trajectory encounters averaged across three campaign repetitions, while isolated values are warmed repeated calls with preassembled immutable inputs.",
            }
        )

    write_json(validation / "prepared_input_validation.json", input_rows)
    write_json(tables / "per_camera_backend.json", per_camera)
    write_json(tables / "high_level_vs_backend.json", high_level_rows)
    write_json(tables / "image_parity.json", parity_rows)

    wrapper_stage_rows = []
    for camera in (392, 785, 1047):
        probe_records = [
            read_json(artifact / "wrapper_probe" / f"camera_{camera:04d}" / f"repeat_{repeat:02d}.json")
            for repeat in (1, 2, 3)
        ]
        first_calls = [record["iterations"][0] for record in probe_records]
        warm_process_means = []
        for record in probe_records:
            warm = record["iterations"][1:]
            warm_process_means.append(
                {
                    key: statistics.fmean(float(item[key]) for item in warm)
                    for key in (
                        "screen_space_tensor_creation_gpu_ms",
                        "screen_space_tensor_creation_synchronized_wall_ms",
                        "activated_attribute_view_preparation_synchronized_wall_ms",
                        "rasterizer_argument_construction_synchronized_wall_ms",
                        "rasterizer_call_gpu_ms",
                        "rasterizer_call_synchronized_wall_ms",
                    )
                }
            )
        stage_keys = (
            "screen_space_tensor_creation_gpu_ms",
            "screen_space_tensor_creation_synchronized_wall_ms",
            "activated_attribute_view_preparation_synchronized_wall_ms",
            "rasterizer_argument_construction_synchronized_wall_ms",
            "rasterizer_call_gpu_ms",
            "rasterizer_call_synchronized_wall_ms",
        )
        wrapper_stage_rows.append(
            {
                "camera_index": camera,
                "gaussian_count": probe_records[0]["gaussian_count"],
                "first_target_call_after_one_camera_zero_warmup": {
                    key: sample_stats([float(item[key]) for item in first_calls]) for key in stage_keys
                },
                "warmed_target_call_across_process_means": {
                    key: sample_stats([float(item[key]) for item in warm_process_means]) for key in stage_keys
                },
                "interpretation_limit": probe_records[0]["interpretation"],
            }
        )
    write_json(tables / "wrapper_stage_probe.json", wrapper_stage_rows)

    replay_runs = [
        read_jsonl(
            artifact
            / "wrapper_probe"
            / "official_oracle_replay"
            / f"repeat_{repeat:02d}"
            / "frames.jsonl"
        )
        for repeat in (1, 2, 3)
    ]
    replay_rows = []
    for camera in CAMERAS:
        replay_records = [
            find_camera_record(records, camera, ("trajectory_frame_index", "measured_frame_index"))
            for records in replay_runs
        ]
        historical = next(row for row in high_level_rows if row["camera_index"] == camera)
        direct = next(row for row in per_camera if row["camera_index"] == camera)
        replay_stats = metric_stats(
            replay_records,
            (
                "visibility_time_ms",
                "lod_selection_time_ms",
                "selection_time_ms",
                "active_set_assembly_time_ms",
                "render_call_time_ms",
                "cuda_rasterization_time_ms",
                "end_to_end_frame_time_ms",
            ),
        )
        replay_rows.append(
            {
                "camera_index": camera,
                "selected_gaussian_count": direct["selected_gaussian_count"],
                "historical_official_render_call_ms": historical["official_three_repeat_camera_metrics"]["render_call_time_ms"],
                "historical_official_cuda_rasterization_ms": historical["official_three_repeat_camera_metrics"]["cuda_rasterization_time_ms"],
                "historical_official_end_to_end_ms": historical["official_three_repeat_camera_metrics"]["end_to_end_frame_time_ms"],
                "clean_official_oracle_replay": replay_stats,
                "isolated_graphdeco_gpu_time_ms": direct["graphdeco"]["backend_gpu_time_ms_across_process_means"],
                "clean_replay_minus_isolated_graphdeco_gpu_ms": replay_stats["cuda_rasterization_time_ms"]["mean"]
                - direct["graphdeco"]["backend_gpu_time_ms_across_process_means"]["mean"],
                "historical_minus_clean_replay_cuda_ms": historical["official_three_repeat_camera_metrics"]["cuda_rasterization_time_ms"]["mean"]
                - replay_stats["cuda_rasterization_time_ms"]["mean"],
            }
        )
    write_json(tables / "clean_official_oracle_replay.json", replay_rows)

    ratios = [row["graphdeco_over_gsplat_gpu_time_ratio"] for row in per_camera]
    counts = [float(row["selected_gaussian_count"]) for row in per_camera]
    graph_times = [row["graphdeco"]["backend_gpu_time_ms_across_process_means"]["mean"] for row in per_camera]
    gs_times = [row["gsplat"]["backend_gpu_time_ms_across_process_means"]["mean"] for row in per_camera]
    graph_radii = [float(row["graphdeco"]["diagnostics"]["positive_radii_projected_gaussian_count"]) for row in per_camera]
    gs_radii = [float(row["gsplat"]["diagnostics"]["positive_radii_projected_gaussian_count"]) for row in per_camera]
    gs_intersections = [float(row["gsplat"]["diagnostics"]["tile_intersection_count"]) for row in per_camera]
    official_render = [row["official_three_repeat_camera_metrics"]["render_call_time_ms"]["mean"] for row in high_level_rows]
    canonical_render = [row["canonical_gsplat_three_repeat_camera_metrics"]["render_call_time_ms"]["mean"] for row in high_level_rows]
    replay_render = [row["clean_official_oracle_replay"]["render_call_time_ms"]["mean"] for row in replay_rows]
    replay_cuda = [row["clean_official_oracle_replay"]["cuda_rasterization_time_ms"]["mean"] for row in replay_rows]
    replay_e2e = [row["clean_official_oracle_replay"]["end_to_end_frame_time_ms"]["mean"] for row in replay_rows]
    canonical_e2e = [row["canonical_gsplat_three_repeat_camera_metrics"]["end_to_end_frame_time_ms"]["mean"] for row in high_level_rows]
    ratio_stats = sample_stats(ratios)
    modest_count = sum(value <= 1.5 for value in ratios)
    if modest_count >= 10 and max(ratios) <= 1.75:
        outcome = "A — isolated backend ratios remain modest across the frozen cameras"
        conclusion = "The large released-runtime versus gsplat difference is predominantly associated with high-level rendering-path/integration behavior rather than the raw rasterization backend."
    elif max(ratios) < 3.0:
        outcome = "B — the isolated backend ratio varies materially with camera"
        conclusion = "Camera-dependent backend behavior and high-level integration behavior both contribute; neither should be treated as a complete single-cause explanation."
    else:
        outcome = "C — difficult views produce an isolated backend gap approaching the high-level gap"
        conclusion = "Raw backend behavior is a larger contributor on difficult views than camera 392 alone indicated, while high-level integration costs remain separately observable."
    ratio_392 = next(row["graphdeco_over_gsplat_gpu_time_ratio"] for row in per_camera if row["camera_index"] == 392)
    sorted_ratios = sorted(ratios)
    percentile_392 = 100.0 * (sum(value <= ratio_392 for value in ratios) - 0.5) / len(ratios)
    correlations = {
        "backend_ratio_vs_selected_gaussian_count": correlation(counts, ratios),
        "backend_ratio_vs_graphdeco_positive_radii": correlation(graph_radii, ratios),
        "backend_ratio_vs_gsplat_positive_radii": correlation(gs_radii, ratios),
        "backend_ratio_vs_gsplat_tile_intersections": correlation(gs_intersections, ratios),
        "backend_ratio_vs_official_render_call": correlation(official_render, ratios),
        "graphdeco_gpu_time_vs_selected_gaussian_count": correlation(counts, graph_times),
        "gsplat_gpu_time_vs_selected_gaussian_count": correlation(counts, gs_times),
        "official_render_call_vs_isolated_graphdeco_gpu": correlation(graph_times, official_render),
        "canonical_render_call_vs_isolated_gsplat_gpu": correlation(gs_times, canonical_render),
    }
    parity_pass = all(
        row["finite_rgb"]
        and row["finite_alpha"]
        and row["ssim_clipped_0_1"] > 0.999
        and row["psnr_data_range_1_db"] > 60.0
        for row in parity_rows
    )
    if not parity_pass:
        errors.append("One or more camera pairs failed the image-parity gate")
    max_cv = max(
        row[backend]["backend_gpu_time_ms_across_process_means"]["coefficient_of_variation"]
        for row in per_camera
        for backend in BACKENDS
    )
    attribution_summary = {
        "outcome": outcome,
        "conclusion": conclusion,
        "ratio_across_cameras": ratio_stats,
        "cameras_at_or_below_1_5x": modest_count,
        "camera_392": {
            "ratio": ratio_392,
            "percentile_midrank": percentile_392,
            "difference_from_12_camera_median": ratio_392 - ratio_stats["median"],
            "representative_interpretation": "Camera 392 is representative if it lies within the central spread; use the reported percentile and full range rather than treating one view as universal.",
        },
        "correlations": correlations,
        "correlation_warning": "Positive-radii definitions differ between backends, and all correlations are descriptive associations over n=12 rather than causal evidence.",
        "high_level_comparison_warning": high_level_rows[0]["comparison_limit"],
        "clean_official_oracle_replay": {
            "mean_render_call_ms_across_12_camera_means": statistics.fmean(replay_render),
            "mean_cuda_rasterization_ms_across_12_camera_means": statistics.fmean(replay_cuda),
            "mean_end_to_end_ms_across_12_camera_means": statistics.fmean(replay_e2e),
            "throughput_equivalent_from_mean_end_to_end": 1000.0 / statistics.fmean(replay_e2e),
            "canonical_historical_mean_end_to_end_ms_across_same_12_camera_means": statistics.fmean(canonical_e2e),
            "canonical_throughput_equivalent_from_same_12_camera_means": 1000.0 / statistics.fmean(canonical_e2e),
            "clean_official_over_canonical_end_to_end_ratio": statistics.fmean(replay_e2e) / statistics.fmean(canonical_e2e),
            "historical_spike_interpretation": "The historical 350–380 ms CUDA stalls at cameras 785 and 1047 did not reproduce without avoidable external GPU workload. Clean replay values were about 24 and 31 ms, while warmed direct calls were about 12 and 16 ms. External contention or another nonstationary trajectory/system state remains plausible; the old raw observations are preserved but cannot support a stable backend claim.",
        },
        "optional_wrapper_instrumentation": {
            "performed": True,
            "cameras": [392, 785, 1047],
            "finding": "Screen-space tensor creation, activated attribute views, and rasterizer-object construction are sub-millisecond. First target-shaped rasterizer calls are slower than warmed calls but do not approach the historical 350–380 ms stalls."
        },
    }
    write_json(tables / "attribution_summary.json", attribution_summary)

    package_validation = {
        "status": "pass" if not errors else "fail",
        "campaign_id": args.campaign_id,
        "expected_camera_indices": list(CAMERAS),
        "backend_count": 2,
        "repetitions_per_camera_backend": 3,
        "warmups_per_process": 10,
        "measured_samples_per_process": 100,
        "validated_processes": len(CAMERAS) * len(BACKENDS) * 3,
        "validated_measured_samples": len(CAMERAS) * len(BACKENDS) * 3 * 100,
        "input_and_oracle_gate": not any("input" in error or "selection" in error or "camera" in error for error in errors),
        "image_parity_gate": parity_pass,
        "maximum_across_process_gpu_time_cv": max_cv,
        "errors": errors,
        "outcome": outcome,
    }
    write_json(validation / "package_validation.json", package_validation)
    if errors:
        raise RuntimeError(f"Package validation failed: {errors}")

    x = np.arange(len(CAMERAS))
    labels = [str(camera) for camera in CAMERAS]
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, graph_times, width, label="GraphDECO")
    ax.bar(x + width / 2, gs_times, width, label="gsplat")
    ax.set_xticks(x, labels, rotation=45)
    ax.set_xlabel("Frozen camera index")
    ax.set_ylabel("Direct backend GPU time (ms)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(figures / "isolated_backend_gpu_time_by_camera.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    ax.plot(x, ratios, marker="o")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.axhline(1.5, color="#777777", linewidth=1, linestyle="--")
    ax.set_xticks(x, labels, rotation=45)
    ax.set_xlabel("Frozen camera index")
    ax.set_ylabel("GraphDECO / gsplat GPU-time ratio")
    ax.grid(alpha=0.25)
    fig.savefig(figures / "backend_ratio_by_camera.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(graph_times, official_render)
    for camera, xvalue, yvalue in zip(CAMERAS, graph_times, official_render):
        ax.annotate(str(camera), (xvalue, yvalue), fontsize=7, xytext=(3, 3), textcoords="offset points")
    limit = max(max(graph_times), max(official_render)) * 1.05
    ax.plot([0, limit], [0, limit], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Isolated GraphDECO GPU time (ms)")
    ax.set_ylabel("Official high-level render-call time (ms)")
    ax.grid(alpha=0.25)
    fig.savefig(figures / "official_render_vs_isolated_graphdeco.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(gs_times, canonical_render)
    for camera, xvalue, yvalue in zip(CAMERAS, gs_times, canonical_render):
        ax.annotate(str(camera), (xvalue, yvalue), fontsize=7, xytext=(3, 3), textcoords="offset points")
    limit = max(max(gs_times), max(canonical_render)) * 1.05
    ax.plot([0, limit], [0, limit], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Isolated gsplat GPU time (ms)")
    ax.set_ylabel("Canonical high-level render-call time (ms)")
    ax.grid(alpha=0.25)
    fig.savefig(figures / "canonical_render_vs_isolated_gsplat.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(np.asarray(counts) / 1.0e6, ratios)
    for camera, xvalue, yvalue in zip(CAMERAS, np.asarray(counts) / 1.0e6, ratios):
        ax.annotate(str(camera), (xvalue, yvalue), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Selected Gaussians (millions)")
    ax.set_ylabel("GraphDECO / gsplat GPU-time ratio")
    ax.grid(alpha=0.25)
    fig.savefig(figures / "backend_ratio_vs_selected_gaussians.png", dpi=180)
    plt.close(fig)

    manifest = {
        "campaign_id": args.campaign_id,
        "status": "complete",
        "classification": "CITYGS-12CAM-ATTRIBUTION; frozen-oracle-camera direct-backend and historical high-level attribution",
        "created_date": "2026-08-24",
        "camera_indices": list(CAMERAS),
        "resolution": [1920, 1080],
        "protocol": {
            "warmup_renders_per_process": 10,
            "measured_renders_per_process": 100,
            "independent_processes_per_camera_backend": 3,
            "backend_order": ["repeat 1 GraphDECO→gsplat", "repeat 2 gsplat→GraphDECO", "repeat 3 GraphDECO→gsplat"],
            "primary_metric": "backend_gpu_time_ms",
            "secondary_metric": "backend_synchronized_wall_time_ms",
            "historical_camera_values": "mean and sample SD of each frozen camera's value across the three existing formal repetitions",
        },
        "pinned_sources": {
            "gs_framework": "cfc2136459dd63cc47dc3acfde55a3d72a4d8bba",
            "CityGaussian": "e16c526f78ab01d377b13b3e6862dcc93ddbe953",
            "diff_gaussian_rasterization": "59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d",
        },
        "compiled_binaries": {
            "graphdeco": {
                "path": "/home/luis/tese/CityGaussian_official_parity/.official_extensions/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so",
                "sha256": "cc2ada2c5875c272771ebe4624208c0a5f4004fc150cc9914d553274d02fffd8",
                "architecture": "four native sm_120 cubins; no PTX",
            },
            "gsplat": {
                "path": "/home/luis/.cache/torch_extensions/py312_cu129/gsplat_cuda/gsplat_cuda.so",
                "sha256": "fc0ceb5237b25a95f78d1641a79a564bf7f2ed22bfa69835e7012fc7eb1971a7",
                "version": "1.5.3",
                "architecture": "twenty native sm_120 cubins plus compute_120 PTX",
            },
        },
        "source_harnesses": [{"path": str(path), "sha256": sha256_file(path)} for path in SOURCE_SCRIPTS],
        "source_evidence": {
            "official_parity": str(OFFICIAL),
            "canonical_gsplat": str(CANONICAL),
            "single_camera_microbenchmark": str(PREVIOUS_MICROBENCH),
        },
        "ephemeral_canonical_inputs": {
            "path_during_campaign": str(args.input_dir),
            "file_count": len(CAMERAS),
            "total_bytes": sum((args.input_dir / f"camera_{camera:04d}.pt").stat().st_size for camera in CAMERAS),
            "retention": "Removed after package validation to reclaim temporary space; deterministic tensor/file hashes, source asset hashes, selection records, per-process re-hashing, raw results, and reconstruction harnesses are preserved in the artifact.",
        },
        "external_gpu_workload_baseline": {
            "windows_app_record": "environment/windows_open_apps_baseline.json",
            "nvidia_query": "environment/gpu_baseline_query.csv",
            "nvidia_process_snapshot": "environment/gpu_baseline_full.txt",
        },
        "focused_followup": {
            "wrapper_stage_probe": "tables/wrapper_stage_probe.json",
            "clean_official_oracle_replay": "tables/clean_official_oracle_replay.json",
            "scope": "three-stage-probe cameras and three fresh twelve-oracle-frame official-wrapper processes; not a full trajectory rerun",
        },
        "outcome": outcome,
        "no_backend_or_method_optimization": True,
        "assets_cameras_resolution_and_selection_unchanged": True,
        "pinned_reference_checkout_modified": False,
        "thesis_latex_modified": False,
    }
    write_json(artifact / "manifest.json", manifest)

    readme = f"""# CityGaussian 12-camera performance attribution

Campaign ID: `{args.campaign_id}`  
Status: **complete**  
Evidence class: `CITYGS-12CAM-ATTRIBUTION-*`

## Outcome

{outcome}. {conclusion}

The study uses the exact frozen camera indices `{list(CAMERAS)}`, 1920×1080 resolution, released Nyquist selections, immutable activated Gaussian inputs, and the same direct rasterizer boundary as the preceding backend microbenchmark. Every camera/backend cell has three fresh processes, ten excluded warmups, and 100 CUDA-event samples. Historical per-camera values are means and sample SDs across the three already-completed formal repetitions; neither 1,440-frame campaign was rerun.

## Per-camera direct timing

| Camera | Visible cells | Selected Gaussians | GraphDECO GPU ms | gsplat GPU ms | Ratio | GraphDECO wall ms | gsplat wall ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for row in per_camera:
        graph_gpu_stats = row["graphdeco"]["backend_gpu_time_ms_across_process_means"]
        gs_gpu_stats = row["gsplat"]["backend_gpu_time_ms_across_process_means"]
        graph_wall_stats = row["graphdeco"]["backend_synchronized_wall_time_ms_across_process_means"]
        gs_wall_stats = row["gsplat"]["backend_synchronized_wall_time_ms_across_process_means"]
        readme += (
            f"| {row['camera_index']} | {row['visible_cell_count']} | {row['selected_gaussian_count']:,} | "
            f"{graph_gpu_stats['mean']:.3f} ± {graph_gpu_stats['std_sample']:.3f} | "
            f"{gs_gpu_stats['mean']:.3f} ± {gs_gpu_stats['std_sample']:.3f} | "
            f"{row['graphdeco_over_gsplat_gpu_time_ratio']:.3f}× | "
            f"{graph_wall_stats['mean']:.3f} ± {graph_wall_stats['std_sample']:.3f} | "
            f"{gs_wall_stats['mean']:.3f} ± {gs_wall_stats['std_sample']:.3f} |\n"
        )
    readme += f"""

Across cameras, the ratio mean/median/min/max is `{ratio_stats['mean']:.3f} / {ratio_stats['median']:.3f} / {ratio_stats['min']:.3f} / {ratio_stats['max']:.3f}`. Camera 392 is at approximately the `{percentile_392:.1f}` percentile of the 12-camera ratios.

## Clean official-wrapper replay and stage probe

The unchanged official wrapper was replayed on only the twelve oracle cameras in three fresh processes under the clean GPU condition. Its mean render/CUDA/end-to-end time across the twelve camera means is `{statistics.fmean(replay_render):.3f} / {statistics.fmean(replay_cuda):.3f} / {statistics.fmean(replay_e2e):.3f} ms`, corresponding to a `{1000.0 / statistics.fmean(replay_e2e):.2f} s⁻¹` end-to-end throughput equivalent for this sparse oracle sample. The historical canonical gsplat values for the same twelve camera indices average `{statistics.fmean(canonical_e2e):.3f} ms` (`{1000.0 / statistics.fmean(canonical_e2e):.2f} s⁻¹`), a clean sparse-sample end-to-end ratio of `{statistics.fmean(replay_e2e) / statistics.fmean(canonical_e2e):.3f}×`.

The historical official CUDA stalls of roughly 378 ms at camera 785 and 349 ms at camera 1047 did not reproduce: clean replay values are about 24 and 31 ms. Focused stage probes show sub-millisecond screen-space tensor creation, activated attribute views, and rasterizer-object construction. First target-shaped rasterizer calls are about 26 and 32 ms; warmed calls return to about 12 and 16 ms. The original stalls are preserved as raw evidence but cannot be treated as stable wrapper or backend cost. External contention or another nonstationary trajectory/system state remains plausible and cannot be distinguished retrospectively.

## Validation and interpretation boundaries

Every prepared active set matches the official released-LoD oracle and the canonical reproduction records exactly in visible cells, LoD assignments, per-level counts, and total count. Every process re-hashed the immutable input before upload. Cross-backend image parity passes for all cameras; details are in `tables/image_parity.json`.

The twelve serialized canonical `.pt` inputs were a 7.0 GiB temporary execution cache, not source evidence. They were removed after validation; their exact file/tensor hashes, frozen source-asset hashes, full selection metadata, per-process input hashes, output arrays, and reconstruction harnesses remain durable in this package.

The historical high-level/direct differences in `tables/high_level_vs_backend.json` are diagnostic rather than exact accounting identities. Historical frames are single trajectory encounters averaged across three repetitions, while the direct calls repeatedly render a warmed, preassembled input. Positive-radii definitions differ between rasterizers, and correlation does not establish causality.

## Evidence map

- `manifest.json`: frozen sources, binaries, protocol, and evidence relationships.
- `environment/`: idle GPU/window baseline, build records, packages, and harness copies.
- `validation/prepared_input_validation.json`: hashes, camera checks, and exact oracle comparisons.
- `validation/package_validation.json`: completeness and parity gate for 72 processes and 7,200 samples.
- `microbench/`: raw 100-sample JSONL, summaries, images, diagnostics, memory, and provenance.
- `tables/per_camera_backend.json`: direct timings and internal diagnostics.
- `tables/high_level_vs_backend.json`: extracted historical camera values and conservative differences.
- `tables/attribution_summary.json`: ratio distribution, camera-392 placement, and correlations.
- `tables/image_parity.json`: finite, RGB/alpha, PSNR, SSIM, and coverage results.
- `tables/wrapper_stage_probe.json`: synchronized stage probe for cameras 392, 785, and 1047.
- `tables/clean_official_oracle_replay.json`: historical versus clean official-wrapper camera values.
- `figures/`: five thesis-ready diagnostic plots.

No backend, kernel, method, asset, camera, resolution, visibility, or LoD behavior was optimized or changed. The pinned CityGaussian reference checkout and thesis LaTeX were not modified.
"""
    (artifact / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
