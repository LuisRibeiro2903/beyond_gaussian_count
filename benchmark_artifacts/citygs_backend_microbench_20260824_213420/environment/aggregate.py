#!/usr/bin/env python
"""Validate, aggregate, plot, and document the backend microbenchmark package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity


CASES = ("finest_flat", "representative_lod")
BACKENDS = ("graphdeco", "gsplat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def across(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std_sample": sample_sd,
        "coefficient_of_variation": sample_sd / mean if mean else 0.0,
        "min": min(values),
        "max": max(values),
    }


def comparison_metrics(graph_dir: Path, gsplat_dir: Path) -> dict[str, Any]:
    graph_rgb = np.load(graph_dir / "rgb.npy")
    gsplat_rgb = np.load(gsplat_dir / "rgb.npy")
    graph_alpha = np.load(graph_dir / "alpha.npy")
    gsplat_alpha = np.load(gsplat_dir / "alpha.npy")
    if graph_rgb.shape != gsplat_rgb.shape or graph_alpha.shape != gsplat_alpha.shape:
        raise RuntimeError("Cross-backend image or alpha shapes differ")
    raw_diff = np.abs(graph_rgb - gsplat_rgb)
    alpha_diff = np.abs(graph_alpha - gsplat_alpha)
    clipped_graph = np.clip(graph_rgb, 0.0, 1.0)
    clipped_gsplat = np.clip(gsplat_rgb, 0.0, 1.0)
    graph_coverage = graph_alpha[..., 0] > 1.0e-4
    gsplat_coverage = gsplat_alpha[..., 0] > 1.0e-4
    union = np.logical_or(graph_coverage, gsplat_coverage)
    intersection = np.logical_and(graph_coverage, gsplat_coverage)
    return {
        "finite_rgb": bool(np.isfinite(graph_rgb).all() and np.isfinite(gsplat_rgb).all()),
        "finite_alpha": bool(np.isfinite(graph_alpha).all() and np.isfinite(gsplat_alpha).all()),
        "graphdeco_rgb_range": [float(graph_rgb.min()), float(graph_rgb.max())],
        "gsplat_rgb_range": [float(gsplat_rgb.min()), float(gsplat_rgb.max())],
        "rgb_mean_absolute_difference_raw": float(raw_diff.mean()),
        "rgb_max_absolute_difference_raw": float(raw_diff.max()),
        "rgb_psnr_raw_data_range_1": psnr(graph_rgb, gsplat_rgb),
        "rgb_psnr_clipped_0_1": psnr(clipped_graph, clipped_gsplat),
        "rgb_ssim_clipped_0_1": float(
            structural_similarity(clipped_graph, clipped_gsplat, channel_axis=2, data_range=1.0)
        ),
        "alpha_mean_absolute_difference": float(alpha_diff.mean()),
        "alpha_max_absolute_difference": float(alpha_diff.max()),
        "alpha_psnr_data_range_1": psnr(graph_alpha, gsplat_alpha),
        "graphdeco_coverage_fraction_alpha_gt_1e_4": float(graph_coverage.mean()),
        "gsplat_coverage_fraction_alpha_gt_1e_4": float(gsplat_coverage.mean()),
        "coverage_jaccard": float(intersection.sum() / max(int(union.sum()), 1)),
        "graphdeco_only_coverage_fraction": float(
            np.logical_and(graph_coverage, ~gsplat_coverage).mean()
        ),
        "gsplat_only_coverage_fraction": float(
            np.logical_and(gsplat_coverage, ~graph_coverage).mean()
        ),
    }


def prior_camera_gpu_times(root: Path, camera_id: str) -> list[float]:
    values = []
    for repeat_dir in sorted(root.glob("repeat_*")):
        matches = [
            json.loads(line)
            for line in (repeat_dir / "frames.jsonl").read_text().splitlines()
            if json.loads(line).get("camera_id") == camera_id
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {camera_id} record in {repeat_dir}, found {len(matches)}")
        values.append(float(matches[0]["cuda_rasterization_time_ms"]))
    if len(values) != 3:
        raise RuntimeError(f"Expected three prior repeat values under {root}, found {len(values)}")
    return values


def main() -> None:
    args = parse_args()
    artifact = args.artifact
    tables = artifact / "tables"
    figures = artifact / "figures"
    validation = artifact / "validation"
    images = artifact / "images"
    for path in (tables, figures, validation, images):
        path.mkdir(parents=True, exist_ok=True)

    inputs = {
        case: load_json(validation / f"{case}_input.json")
        for case in CASES
    }
    repeats: dict[str, dict[str, list[dict[str, Any]]]] = {}
    validation_errors: list[str] = []
    for case in CASES:
        repeats[case] = {}
        for backend in BACKENDS:
            rows = []
            for repeat in range(1, 4):
                run_dir = artifact / "cases" / case / backend / f"repeat_{repeat:02d}"
                row = load_json(run_dir / "repeat.json")
                sample_count = sum(1 for _ in (run_dir / "samples.jsonl").open())
                if sample_count != row["measured_render_count"]:
                    validation_errors.append(
                        f"{case}/{backend}/repeat_{repeat:02d}: sample count mismatch"
                    )
                if row["gaussian_count"] != inputs[case]["gaussian_count"]:
                    validation_errors.append(
                        f"{case}/{backend}/repeat_{repeat:02d}: Gaussian count mismatch"
                    )
                expected_hashes = {
                    key: value["sha256"]
                    for key, value in inputs[case]["canonical_tensor_records"].items()
                }
                if row["input_tensor_sha256"] != expected_hashes:
                    validation_errors.append(
                        f"{case}/{backend}/repeat_{repeat:02d}: tensor hash mismatch"
                    )
                if not row["all_rgb_finite"] or not row["all_alpha_finite"]:
                    validation_errors.append(
                        f"{case}/{backend}/repeat_{repeat:02d}: non-finite output"
                    )
                rows.append(row)
            repeats[case][backend] = rows

    timing_rows = []
    relative_rows = []
    for case in CASES:
        case_means: dict[str, dict[str, float]] = {}
        for backend in BACKENDS:
            rows = repeats[case][backend]
            gpu = across([float(row["backend_gpu_time_ms"]["mean"]) for row in rows])
            wall = across(
                [float(row["backend_synchronized_wall_time_ms"]["mean"]) for row in rows]
            )
            case_means[backend] = {"gpu": gpu["mean"], "wall": wall["mean"]}
            timing_rows.append(
                {
                    "case": case,
                    "backend": backend,
                    "gaussian_count": inputs[case]["gaussian_count"],
                    "camera_index": inputs[case]["camera_index"],
                    "backend_gpu_time_ms_across_process_means": gpu,
                    "backend_synchronized_wall_time_ms_across_process_means": wall,
                    "backend_throughput_equivalent": 1000.0 / gpu["mean"],
                    "repeat_process_summaries": [
                        {
                            "repeat": row["repeat"],
                            "backend_gpu_time_ms": row["backend_gpu_time_ms"],
                            "backend_synchronized_wall_time_ms": row[
                                "backend_synchronized_wall_time_ms"
                            ],
                        }
                        for row in rows
                    ],
                }
            )
        ratio = case_means["graphdeco"]["gpu"] / case_means["gsplat"]["gpu"]
        relative_rows.append(
            {
                "case": case,
                "gaussian_count": inputs[case]["gaussian_count"],
                "graphdeco_backend_gpu_time_ms": case_means["graphdeco"]["gpu"],
                "gsplat_backend_gpu_time_ms": case_means["gsplat"]["gpu"],
                "graphdeco_over_gsplat_gpu_time_ratio": ratio,
                "graphdeco_backend_synchronized_wall_time_ms": case_means["graphdeco"]["wall"],
                "gsplat_backend_synchronized_wall_time_ms": case_means["gsplat"]["wall"],
            }
        )

    write_json(tables / "backend_timing.json", timing_rows)
    write_json(tables / "relative_speed.json", relative_rows)

    prior_roots = {
        ("finest_flat", "graphdeco"): Path(
            "/home/luis/tese/benchmark_artifacts/citygs_official_parity_20260824_191215/runs/official_finest_flat"
        ),
        ("representative_lod", "graphdeco"): Path(
            "/home/luis/tese/benchmark_artifacts/citygs_official_parity_20260824_191215/runs/official_lod_release"
        ),
        ("finest_flat", "gsplat"): Path(
            "/home/luis/tese/benchmark_artifacts/citygs_canonical_20260824_135615/runs/CG-FINEST-FLAT"
        ),
        ("representative_lod", "gsplat"): Path(
            "/home/luis/tese/benchmark_artifacts/citygs_canonical_20260824_135615/runs/CG-LOD-RELEASE"
        ),
    }
    prior_boundary_rows = []
    for case in CASES:
        for backend in BACKENDS:
            values = prior_camera_gpu_times(prior_roots[(case, backend)], "viser_000392")
            prior_stats = across(values)
            direct_row = next(
                row for row in timing_rows if row["case"] == case and row["backend"] == backend
            )
            direct_mean = direct_row["backend_gpu_time_ms_across_process_means"]["mean"]
            prior_boundary_rows.append(
                {
                    "case": case,
                    "backend": backend,
                    "camera_id": "viser_000392",
                    "prior_campaign_gpu_boundary_ms": prior_stats,
                    "prior_repeat_values_ms": values,
                    "direct_backend_gpu_time_ms": direct_mean,
                    "prior_boundary_minus_direct_ms": prior_stats["mean"] - direct_mean,
                    "comparison_limit": "Prior values are camera-392 samples encountered within full trajectory replay; direct values repeatedly render a warmed fixed camera. The difference characterizes timing-boundary/context contamination and is not a standalone component-cost measurement.",
                }
            )
    write_json(tables / "prior_boundary_comparison.json", prior_boundary_rows)
    with (tables / "backend_timing.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case",
                "backend",
                "gaussian_count",
                "camera_index",
                "gpu_mean_ms",
                "gpu_sd_ms",
                "gpu_cv",
                "wall_mean_ms",
                "wall_sd_ms",
                "backend_throughput_equivalent",
            ),
        )
        writer.writeheader()
        for row in timing_rows:
            writer.writerow(
                {
                    "case": row["case"],
                    "backend": row["backend"],
                    "gaussian_count": row["gaussian_count"],
                    "camera_index": row["camera_index"],
                    "gpu_mean_ms": row["backend_gpu_time_ms_across_process_means"]["mean"],
                    "gpu_sd_ms": row["backend_gpu_time_ms_across_process_means"]["std_sample"],
                    "gpu_cv": row["backend_gpu_time_ms_across_process_means"][
                        "coefficient_of_variation"
                    ],
                    "wall_mean_ms": row[
                        "backend_synchronized_wall_time_ms_across_process_means"
                    ]["mean"],
                    "wall_sd_ms": row[
                        "backend_synchronized_wall_time_ms_across_process_means"
                    ]["std_sample"],
                    "backend_throughput_equivalent": row["backend_throughput_equivalent"],
                }
            )

    parity_rows = []
    for case in CASES:
        graph_dir = artifact / "cases" / case / "graphdeco" / "repeat_01"
        gsplat_dir = artifact / "cases" / case / "gsplat" / "repeat_01"
        metrics = comparison_metrics(graph_dir, gsplat_dir)
        graph_summary = repeats[case]["graphdeco"][0]
        gsplat_summary = repeats[case]["gsplat"][0]
        expected_hashes = {
            key: value["sha256"]
            for key, value in inputs[case]["canonical_tensor_records"].items()
        }
        parity = {
            "case": case,
            "gaussian_count": inputs[case]["gaussian_count"],
            "gaussian_count_equal": graph_summary["gaussian_count"]
            == gsplat_summary["gaussian_count"],
            "canonical_tensor_hashes_equal": graph_summary["input_tensor_sha256"]
            == gsplat_summary["input_tensor_sha256"]
            == expected_hashes,
            "camera_equal": all(inputs[case]["camera_equivalence"].values()),
            **metrics,
        }
        parity_rows.append(parity)
        np.save(images / f"{case}_graphdeco_rgb.npy", np.load(graph_dir / "rgb.npy"))
        np.save(images / f"{case}_graphdeco_alpha.npy", np.load(graph_dir / "alpha.npy"))
        np.save(images / f"{case}_gsplat_rgb.npy", np.load(gsplat_dir / "rgb.npy"))
        np.save(images / f"{case}_gsplat_alpha.npy", np.load(gsplat_dir / "alpha.npy"))
    write_json(tables / "input_output_parity.json", parity_rows)

    diagnostics_rows = []
    memory_rows = []
    for case in CASES:
        for backend in BACKENDS:
            rows = repeats[case][backend]
            diagnostics_rows.append(
                {
                    "case": case,
                    "backend": backend,
                    "diagnostics": [row["measured_diagnostics"] for row in rows],
                    "limitation": (
                        "Pinned GraphDECO public API exposes radii only; tile/sort counts require invasive kernel/API changes and were not added."
                        if backend == "graphdeco"
                        else None
                    ),
                }
            )
            memory_rows.append(
                {
                    "case": case,
                    "backend": backend,
                    "repeats": [row["memory"] for row in rows],
                    "interpretation_limit": "Allocator counters are backend-specific and GraphDECO custom-extension allocations may not be representation-neutral; memory is secondary evidence.",
                }
            )
    write_json(tables / "internal_workload_diagnostics.json", diagnostics_rows)
    write_json(tables / "memory_observations.json", memory_rows)

    labels = ["Representative LoD\n3.82M", "Finest flat\n11.69M"]
    ordered_cases = ["representative_lod", "finest_flat"]
    x = np.arange(len(ordered_cases))
    width = 0.34
    graph_times = [
        next(row for row in relative_rows if row["case"] == case)[
            "graphdeco_backend_gpu_time_ms"
        ]
        for case in ordered_cases
    ]
    gsplat_times = [
        next(row for row in relative_rows if row["case"] == case)["gsplat_backend_gpu_time_ms"]
        for case in ordered_cases
    ]
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.bar(x - width / 2, graph_times, width, label="GraphDECO")
    ax.bar(x + width / 2, gsplat_times, width, label="gsplat")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Direct backend GPU time (ms)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(figures / "direct_backend_gpu_time.png", dpi=180)
    plt.close(fig)

    ratios = [
        next(row for row in relative_rows if row["case"] == case)[
            "graphdeco_over_gsplat_gpu_time_ratio"
        ]
        for case in ordered_cases
    ]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.bar(x, ratios, width=0.5, color="#6f4e7c")
    ax.axhline(1.0, color="black", linewidth=1.0)
    ax.set_xticks(x, labels)
    ax.set_ylabel("GraphDECO / gsplat GPU-time ratio")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(figures / "backend_speed_ratio.png", dpi=180)
    plt.close(fig)

    max_cv = max(
        row["backend_gpu_time_ms_across_process_means"]["coefficient_of_variation"]
        for row in timing_rows
    )
    min_ratio = min(row["graphdeco_over_gsplat_gpu_time_ratio"] for row in relative_rows)
    parity_pass = all(
        row["finite_rgb"]
        and row["finite_alpha"]
        and row["gaussian_count_equal"]
        and row["canonical_tensor_hashes_equal"]
        and row["camera_equal"]
        and row["rgb_ssim_clipped_0_1"] > 0.999
        for row in parity_rows
    )
    if min_ratio >= 3.0 and parity_pass:
        outcome = "A — a large direct backend gap remains with equivalent logical input and images"
        attribution = (
            "The majority of the official/runtime performance gap is directly attributable "
            "to the pinned rasterization backend on the target RTX 5080."
        )
    elif min_ratio < 2.0:
        outcome = "B — the direct gap is materially smaller than the prior flat runtime gap"
        attribution = (
            "The previous high-level runtime gap cannot yet be attributed primarily to the rasterizer."
        )
    else:
        outcome = "Intermediate — a direct gap remains, but it does not meet the prespecified large-gap example"
        attribution = "Backend contribution is supported, while dominance requires qualified wording."
    if max_cv > 0.05:
        outcome += "; across-process GPU-time CV exceeds 5% for at least one cell"

    package_validation = {
        "status": "pass" if not validation_errors and parity_pass else "fail",
        "campaign_id": args.campaign_id,
        "expected_cases": list(CASES),
        "expected_backends": list(BACKENDS),
        "repetitions_per_backend_case": 3,
        "measured_samples_per_repetition": 100,
        "validated_measured_samples": 3 * 100 * len(CASES) * len(BACKENDS),
        "input_hash_parity": not any("hash" in error for error in validation_errors),
        "image_sanity_gate": parity_pass,
        "errors": validation_errors,
        "outcome_classification": outcome,
        "attribution": attribution,
        "max_across_process_gpu_time_cv": max_cv,
    }
    write_json(validation / "package_validation.json", package_validation)
    if package_validation["status"] != "pass":
        raise RuntimeError(f"Package validation failed: {package_validation}")

    source_files = [
        Path("/home/luis/tese/backend_microbench/prepare_inputs.py"),
        Path("/home/luis/tese/backend_microbench/run_backend.py"),
        Path("/home/luis/tese/backend_microbench/aggregate.py"),
        Path("/home/luis/tese/backend_microbench/run_campaign.sh"),
    ]
    manifest = {
        "campaign_id": args.campaign_id,
        "status": "complete",
        "classification": "CITYGS-BACKEND-MICROBENCH; isolated same-input rasterizer/backend comparison",
        "created_date": "2026-08-24",
        "cases": [
            {
                "case": case,
                "gaussian_count": inputs[case]["gaussian_count"],
                "camera_index": inputs[case]["camera_index"],
                "camera_id": inputs[case]["camera_id"],
                "resolution": inputs[case]["resolution"],
                "input_file_sha256": inputs[case]["canonical_input_file_sha256"],
                "input_tensor_sha256": {
                    key: value["sha256"]
                    for key, value in inputs[case]["canonical_tensor_records"].items()
                },
                "assets": inputs[case]["assets"],
            }
            for case in CASES
        ],
        "pinned_sources": {
            "gs_framework": "cfc2136459dd63cc47dc3acfde55a3d72a4d8bba",
            "CityGaussian": "e16c526f78ab01d377b13b3e6862dcc93ddbe953",
            "diff_gaussian_rasterization": "59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d",
        },
        "protocol": {
            "warmup_renders_per_process": 10,
            "measured_renders_per_process": 100,
            "independent_processes_per_backend_case": 3,
            "primary_metric": "backend_gpu_time_ms",
            "secondary_metric": "backend_synchronized_wall_time_ms",
            "ordering": [
                "repeat 1: GraphDECO then gsplat",
                "repeat 2: gsplat then GraphDECO",
                "repeat 3: GraphDECO then gsplat",
            ],
            "timed_boundary": "one direct rasterization invocation after immutable input/camera/static-object setup and warmup",
        },
        "trajectory": inputs["finest_flat"]["trajectory"],
        "output_interpretation": {
            "backend_throughput_equivalent": "1000 divided by mean direct backend GPU time; not end-to-end viewer FPS",
            "outcome": outcome,
            "attribution": attribution,
            "prior_boundary_reinterpretation": "The official-parity CUDA field remains valid for its recorded high-level call boundary, but it is not an isolated GraphDECO-rasterizer measurement. Strong rasterizer-dominance attribution is superseded by this microbenchmark.",
        },
        "source_harnesses": [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_files
        ],
        "compiled_binaries": {
            "graphdeco": {
                "path": "/home/luis/tese/CityGaussian_official_parity/.official_extensions/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so",
                "sha256": "cc2ada2c5875c272771ebe4624208c0a5f4004fc150cc9914d553274d02fffd8",
                "architecture": "four native sm_120 cubins; no PTX",
            },
            "gsplat": {
                "path": "/home/luis/.cache/torch_extensions/py312_cu129/gsplat_cuda/gsplat_cuda.so",
                "sha256": "fc0ceb5237b25a95f78d1641a79a564bf7f2ed22bfa69835e7012fc7eb1971a7",
                "architecture": "twenty native sm_120 cubins plus compute_120 PTX",
            },
        },
        "excluded_setup_run": {
            "path": "validation/superseded_setup_runs/representative_lod_gsplat_repeat_01_pre_rebuild",
            "reason": "Preserved for audit but excluded from aggregation because it preceded the unchanged-source gsplat CUDA 12.9 sm_120 rebuild and its exact loaded binary hash was not preserved.",
        },
        "no_optimization_or_semantic_change": True,
        "pinned_reference_checkout_modified": False,
        "dissertation_files_modified": False,
    }
    write_json(artifact / "manifest.json", manifest)

    timing_lookup = {(row["case"], row["backend"]): row for row in timing_rows}
    relative_lookup = {row["case"]: row for row in relative_rows}
    parity_lookup = {row["case"]: row for row in parity_rows}
    readme = f"""# CityGaussian backend microbenchmark

Campaign ID: `{args.campaign_id}`  
Status: **complete**  
Evidence class: `CITYGS-BACKEND-MICROBENCH-*`

## Outcome

{outcome}. {attribution}

This is a direct same-input rasterizer comparison, separate from both the canonical gsplat CityGaussian campaign and the official-runtime parity campaign. It is not viewer FPS and does not reproduce paper numbers.

## Test cases and protocol

- `finest_flat`: {inputs['finest_flat']['gaussian_count']:,} Gaussians from the exact finest released SH2 PLY, camera index {inputs['finest_flat']['camera_index']}.
- `representative_lod`: {inputs['representative_lod']['gaussian_count']:,} preassembled Gaussians from the validated released visibility/Nyquist selector, camera index {inputs['representative_lod']['camera_index']}.
- Resolution: 1920×1080; black background; scale modifier 1; SH degree 2.
- Three fresh processes per backend/case, ten excluded warmup renders, 100 measured renders per process.
- Repeat ordering: GraphDECO→gsplat, gsplat→GraphDECO, GraphDECO→gsplat.
- Primary timing is CUDA-event `backend_gpu_time_ms`; secondary timing is synchronized `backend_synchronized_wall_time_ms`.

## Direct timing

| Case | Backend | Gaussian count | GPU mean ± process SD (ms) | Wall mean ± process SD (ms) | Backend throughput equivalent (s⁻¹) |
| --- | --- | ---: | ---: | ---: | ---: |
"""
    for case in ("representative_lod", "finest_flat"):
        for backend in BACKENDS:
            row = timing_lookup[(case, backend)]
            gpu = row["backend_gpu_time_ms_across_process_means"]
            wall = row["backend_synchronized_wall_time_ms_across_process_means"]
            readme += (
                f"| `{case}` | {backend} | {row['gaussian_count']:,} | "
                f"{gpu['mean']:.3f} ± {gpu['std_sample']:.3f} | "
                f"{wall['mean']:.3f} ± {wall['std_sample']:.3f} | "
                f"{row['backend_throughput_equivalent']:.2f} |\n"
            )
    readme += "\n## Relative speed\n\n"
    readme += "| Case | GraphDECO GPU ms | gsplat GPU ms | GraphDECO / gsplat |\n| --- | ---: | ---: | ---: |\n"
    for case in ("representative_lod", "finest_flat"):
        row = relative_lookup[case]
        readme += (
            f"| `{case}` | {row['graphdeco_backend_gpu_time_ms']:.3f} | "
            f"{row['gsplat_backend_gpu_time_ms']:.3f} | "
            f"{row['graphdeco_over_gsplat_gpu_time_ratio']:.2f}× |\n"
        )
    readme += "\n## Input and image gate\n\n"
    readme += "| Case | Count/hash/camera equal | Raw RGB PSNR (dB) | Clipped SSIM | Mean raw RGB diff | Coverage Jaccard |\n| --- | --- | ---: | ---: | ---: | ---: |\n"
    for case in ("representative_lod", "finest_flat"):
        row = parity_lookup[case]
        equal = row["gaussian_count_equal"] and row["canonical_tensor_hashes_equal"] and row["camera_equal"]
        readme += (
            f"| `{case}` | {'yes' if equal else 'no'} | {row['rgb_psnr_raw_data_range_1']:.3f} | "
            f"{row['rgb_ssim_clipped_0_1']:.9f} | {row['rgb_mean_absolute_difference_raw']:.8g} | "
            f"{row['coverage_jaccard']:.9f} |\n"
        )
    readme += "\n## Reconciliation with the earlier timing boundary\n\n"
    readme += "| Case | Backend | Earlier camera-392 CUDA boundary (ms) | Direct backend GPU time (ms) | Boundary minus direct (ms) |\n| --- | --- | ---: | ---: | ---: |\n"
    for case in ("representative_lod", "finest_flat"):
        for backend in BACKENDS:
            row = next(
                value
                for value in prior_boundary_rows
                if value["case"] == case and value["backend"] == backend
            )
            readme += (
                f"| `{case}` | {backend} | {row['prior_campaign_gpu_boundary_ms']['mean']:.3f} | "
                f"{row['direct_backend_gpu_time_ms']:.3f} | {row['prior_boundary_minus_direct_ms']:.3f} |\n"
            )
    readme += """

The earlier official flat CUDA event began before the high-level `render()` function. Source inspection shows that boundary included creation/zeroing of the full N×3 screen-space tensor and per-frame `get_opacity` sigmoid, `get_scaling` exponentiation, `get_rotation` normalization, and `get_features` concatenation before `GaussianRasterizer.__call__`. The 11.69M camera-392 boundary is therefore 23.751 ms versus 10.769 ms for the direct preactivated/preconstructed call. By contrast, official adaptive-LoD instrumentation placed its CUDA event after active-set assembly and attribute extraction, immediately around `GaussianRasterizer`; its earlier 10.507 ms camera-392 value is correspondingly close to the direct 8.989 ms result. Canonical gsplat already held activated scene tensors and shows much smaller boundary-to-direct differences.

These paired values are not an exact subtraction-based component profile: the earlier samples were encountered during a moving-camera trajectory, whereas the microbenchmark repeatedly renders one warmed camera. They do, however, identify timing-boundary contamination in the flat official control and disprove treating its old CUDA field as a pure GraphDECO-kernel measurement. Camera-dependent tile/sort workload across the full trajectory is a remaining explanation for the old 40.968 ms official-LoD average and was not re-benchmarked here.
"""
    readme += f"""

Every backend repeat re-hashed the same serialized activated tensors before upload. Camera validation confirms that the official transposed world-view matrix is exactly the transpose of the gsplat world-to-camera matrix and that both derive from the already validated parity conversion. Output RGB and alpha are finite. GraphDECO alpha is obtained by one extra untimed constant-one render because the pinned API does not return alpha.

## Exact logical-input mapping

| Logical attribute | gsplat input | GraphDECO input | Outside-timing conversion |
| --- | --- | --- | --- |
| Means / xyz | `means` N×3 | `means3D` N×3 | none after shared float32 tensor load |
| Scale | activated positive `scales` N×3 | activated positive `scales` N×3 | PLY log-scale exponentiation before serialization |
| Rotation | normalized wxyz `quats` N×4 | normalized wxyz `rotations` N×4 | normalization before serialization |
| Opacity | activated vector N | activated column N×1 | PLY sigmoid before serialization; GraphDECO column view only |
| SH | `colors` N×9×3, degree 2 | `shs` N×9×3, degree 2 | channel-major PLY rest coefficients mapped once before serialization |
| Camera | conventional w2c plus K | official transposed world-view and full projection | validated parity conversion before serialization |
| Image / background | 1920×1080, black | 1920×1080, black | static setup before timing |
| Scale modifier | implicit 1 | explicit 1 | static setup before timing |

Both backends internally perform covariance construction, SH evaluation, camera-space projection, frustum rejection, tile overlap generation, duplicated-instance sorting, and alpha compositing. The gsplat public result exposes positive radii, per-Gaussian tile intersections, and sorted intersection arrays. The pinned GraphDECO public result exposes radii but not tile/sort counts; obtaining those would require invasive API/kernel instrumentation, which this task forbids. Therefore this experiment isolates implementation-level rasterizer cost but cannot reduce the difference to identical internal work or kernel efficiency alone.

## Build verification

`environment/graphdeco_build.txt` records native `sm_120` cubins, no PTX fallback, `-DNDEBUG`, host `-O2`, CUDA `-gencode=arch=compute_120,code=sm_120`, and no `-G`. `environment/gsplat_build.txt` records native `sm_120` cubins plus compute_120 PTX, host/CUDA `-O3`, and no `-G`. GraphDECO remains pinned at `{manifest['pinned_sources']['diff_gaussian_rasterization']}` with only the previously recorded `<cstdint>` include compatibility change. Neither backend or kernel was optimized for this campaign.

One initial gsplat setup result is retained under `validation/superseded_setup_runs/` but excluded from every table and validation total because it preceded the unchanged-source sm_120 rebuild and its exact loaded binary hash was not preserved. The formal replacement repeat and all other gsplat measurements loaded the binary hash recorded in `manifest.json`.

## Memory and thermal limits

Allocator and whole-device observations are in `tables/memory_observations.json`. PyTorch allocator values are secondary because custom-extension allocation semantics are not representation-neutral. NVML temperature/clock fields are retained when WSL exposed them; missing fields remain an environment limitation rather than being inferred.

## Evidence map

- `manifest.json`: identities, hashes, protocol, and interpretation.
- `cases/`: raw 100-sample JSONL, repeat summaries, provenance, and output arrays for all 12 fresh processes.
- `validation/*_input.json`: deterministic activated-tensor statistics/hashes, exact selection record, camera checks, and asset hashes.
- `validation/package_validation.json`: completeness and sanity gate.
- `tables/backend_timing.json`: within-process distributions and across-process mean/SD/CV.
- `tables/relative_speed.json`: direct GraphDECO/gsplat ratios.
- `tables/input_output_parity.json`: RGB, alpha, PSNR, SSIM, and coverage comparisons.
- `tables/internal_workload_diagnostics.json`: accessible backend diagnostic counts and limitations.
- `figures/`: direct GPU time and speed-ratio plots.

## Boundaries

Loading, selection, active-set assembly, concatenation, activation, camera construction, host-to-device transfer, rasterizer/static setup, warmup, diagnostics, alpha sanity rendering, image transfer, and serialization are excluded. The gsplat far-plane argument remains the canonical reproduction value (1e10), while the pinned GraphDECO projection helper stores 100; this is the already validated parity setup and the images pass the sanity gate. The direct operation still includes each public backend's Python/CUDA call chain and required output allocation. Internal rasterization workloads are implementation-dependent and only partially observable.

The pinned reference checkout `/home/luis/tese/CityGaussian`, Gaussian assets, trajectory, selection semantics, and thesis LaTeX were not modified.
"""
    (artifact / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
