#!/usr/bin/env python3
"""Aggregate the frozen ALOD faithful-reproduction campaign.

This script is intentionally limited to ALOD-REPRO evidence.  It validates the
camera/replicate structure, computes pooled and paired distributions, and emits
the thesis-facing JSON and Markdown tables.  It does not run or inspect any
ALOD-EXT configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


CONFIGS = (
    "ALOD-REPRO-HOST-NOCACHE",
    "ALOD-REPRO-NOFRUSTUM",
    "ALOD-REPRO-PREVIOUS-REUSE",
)
REPLICATES = (1, 2, 3)
EXPECTED_CAMERAS = 71
EXPECTED_HIERARCHY_SHA256 = (
    "38f4abdc0d47f2d20316de33f4fff87086e038a57daac0f2feaf98fa9178eb7b"
)
TIMING_FIELDS = (
    "alod_camera_preparation_ms",
    "alod_upper_traversal_ms",
    "alod_spt_cut_ms",
    "alod_total_selection_ms",
    "alod_host_gather_ms",
    "alod_h2d_transfer_ms",
    "alod_pre_raster_assembly_ms",
    "frame_preparation_ms",
    "rasterization_time_ms",
    "end_to_end_frame_time_ms",
)
COUNT_FIELDS = (
    "alod_hierarchy_rows",
    "alod_selected_hierarchy_rows",
    "alod_skybox_rows",
    "alod_total_raster_rows",
    "backend_projection_survivor_count",
    "alod_candidate_spt_count",
    "alod_upper_rows",
)
MEMORY_FIELDS = (
    "alod_host_property_store_bytes",
    "alod_hierarchy_metadata_cpu_bytes",
    "alod_hierarchy_metadata_gpu_bytes",
    "alod_gaussian_properties_gpu_bytes",
    "alod_frame_local_property_bytes",
    "torch_current_allocated_bytes",
    "torch_current_reserved_bytes",
    "torch_peak_allocated_bytes",
    "torch_peak_reserved_bytes",
    "frame_total_peak_allocated_bytes",
    "frame_total_peak_reserved_bytes",
    "nvml_device_used_bytes",
    "process_peak_rss_bytes",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Iterable[float | int | None]) -> dict[str, float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        raise ValueError("no finite values")
    return {
        "minimum": min(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p95": percentile(finite, 0.95),
        "maximum": max(finite),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    return distribution(row.get(field) for row in rows)


def paired_distribution(
    left: list[dict[str, Any]], right: list[dict[str, Any]], field: str
) -> dict[str, float]:
    return distribution(r[field] - l[field] for l, r in zip(left, right, strict=True))


def pct_change(child: float, parent: float) -> float:
    return 100.0 * (child - parent) / parent


def gib(value: float) -> float:
    return value / (1024.0**3)


def ms(value: float) -> str:
    return f"{value:.2f}"


def million(value: float) -> str:
    return f"{value / 1_000_000.0:.3f}M"


def load_runs(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    pooled: dict[str, list[dict[str, Any]]] = {}
    by_replicate: dict[str, dict[int, list[dict[str, Any]]]] = {}
    reference_order: list[tuple[int, str]] | None = None
    for config in CONFIGS:
        pooled[config] = []
        by_replicate[config] = {}
        for replicate in REPLICATES:
            run = root / "runs" / config / f"replicate-{replicate:02d}"
            rows = read_jsonl(run / "frames.jsonl")
            summary = read_json(run / "summary.json")
            if len(rows) != EXPECTED_CAMERAS:
                raise AssertionError(f"{run}: expected 71 frames, found {len(rows)}")
            if summary.get("record_kind") != "formal_benchmark":
                raise AssertionError(f"{run}: not marked as a formal result")
            order = [(int(row["camera_index"]), str(row["camera_id"])) for row in rows]
            if reference_order is None:
                reference_order = order
            elif order != reference_order:
                raise AssertionError(f"{run}: frozen camera order differs")
            if any(row["configuration_id"] != config for row in rows):
                raise AssertionError(f"{run}: configuration ID mismatch")
            if any(row.get("record_kind") != "frame_sample" for row in rows):
                raise AssertionError(f"{run}: unexpected frame record kind")
            pooled[config].extend(rows)
            by_replicate[config][replicate] = rows
    assert reference_order is not None
    return pooled, {"by_replicate": by_replicate, "camera_order": reference_order}


def aggregate_rendering(
    pooled: dict[str, list[dict[str, Any]]], by_replicate: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "scope": {
            "camera_count": EXPECTED_CAMERAS,
            "replicate_count": len(REPLICATES),
            "pooled_samples_per_configuration": EXPECTED_CAMERAS * len(REPLICATES),
            "warmup_frames_per_fresh_process": 10,
            "timing_policy": "CUDA-synchronized; three fresh processes; pooled measured-frame distributions",
            "fps_policy": "1000 / mean end-to-end frame time; frame_time_ms is raster-only and is not used as total FPS",
        },
        "configurations": {},
    }
    for config, rows in pooled.items():
        timings = {field: metric(rows, field) for field in TIMING_FIELDS}
        counts = {field: metric(rows, field) for field in COUNT_FIELDS}
        memory = {field: metric(rows, field) for field in MEMORY_FIELDS}
        replicate_means = {
            str(replicate): statistics.fmean(
                row["end_to_end_frame_time_ms"]
                for row in by_replicate[config][replicate]
            )
            for replicate in REPLICATES
        }
        end_to_end_mean = timings["end_to_end_frame_time_ms"]["mean"]
        output["configurations"][config] = {
            "timings_ms": timings,
            "counts": counts,
            "memory_bytes": memory,
            "h2d": {
                "rows": metric(rows, "alod_h2d_rows"),
                "bytes": metric(rows, "alod_h2d_bytes"),
            },
            "end_to_end_fps_from_mean_time": 1000.0 / end_to_end_mean,
            "replicate_mean_end_to_end_ms": replicate_means,
            "replicate_mean_distribution_ms": distribution(replicate_means.values()),
        }
    return output


def aggregate_frustum(by_replicate: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    on_rows: list[dict[str, Any]] = []
    off_rows: list[dict[str, Any]] = []
    for replicate in REPLICATES:
        on_rows.extend(by_replicate["ALOD-REPRO-HOST-NOCACHE"][replicate])
        off_rows.extend(by_replicate["ALOD-REPRO-NOFRUSTUM"][replicate])
    for on, off in zip(on_rows, off_rows, strict=True):
        if (on["camera_index"], on["camera_id"]) != (off["camera_index"], off["camera_id"]):
            raise AssertionError("frustum paired-camera order mismatch")
    fields = {
        "semantic_selected_hierarchy_rows_avoided": "alod_selected_hierarchy_rows",
        "h2d_rows_avoided": "alod_h2d_rows",
        "h2d_bytes_avoided": "alod_h2d_bytes",
        "packed_projection_survivors_avoided": "backend_projection_survivor_count",
        "candidate_spts_avoided": "alod_candidate_spt_count",
        "host_gather_time_saved_ms": "alod_host_gather_ms",
        "h2d_time_saved_ms": "alod_h2d_transfer_ms",
        "pre_raster_time_saved_ms": "alod_pre_raster_assembly_ms",
        "raster_time_saved_ms": "rasterization_time_ms",
        "end_to_end_time_saved_ms": "end_to_end_frame_time_ms",
    }
    # For workload and time-saved values, positive means frustum-on avoided work.
    result = {
        name: paired_distribution(on_rows, off_rows, field)
        for name, field in fields.items()
    }
    result["frustum_traversal_increment_ms_on_minus_off"] = distribution(
        on["alod_upper_traversal_ms"] - off["alod_upper_traversal_ms"]
        for on, off in zip(on_rows, off_rows, strict=True)
    )
    on_end = statistics.fmean(row["end_to_end_frame_time_ms"] for row in on_rows)
    off_end = statistics.fmean(row["end_to_end_frame_time_ms"] for row in off_rows)
    result["paired_samples"] = len(on_rows)
    result["mean_end_to_end_change_frustum_on_vs_off_percent"] = pct_change(on_end, off_end)
    result["mean_selected_hierarchy_reduction_percent"] = 100.0 * (
        statistics.fmean(off["alod_selected_hierarchy_rows"] - on["alod_selected_hierarchy_rows"]
                         for on, off in zip(on_rows, off_rows, strict=True))
        / statistics.fmean(off["alod_selected_hierarchy_rows"] for off in off_rows)
    )
    result["image_equivalence"] = quality["frustum_rendering_equivalence"]
    result["interpretation"] = (
        "Frustum-on avoided a small semantic/H2D workload on most cameras, but the "
        "total frame-time effect is near measurement variability and does not establish "
        "the paper's cached-renderer frustum timing trend."
    )
    return result


def aggregate_reuse(by_replicate: dict[str, Any]) -> dict[str, Any]:
    no_cache: list[dict[str, Any]] = []
    reuse: list[dict[str, Any]] = []
    for replicate in REPLICATES:
        no_cache.extend(by_replicate["ALOD-REPRO-HOST-NOCACHE"][replicate])
        reuse.extend(by_replicate["ALOD-REPRO-PREVIOUS-REUSE"][replicate])
    fields = {
        "h2d_rows_saved": "alod_h2d_rows",
        "h2d_bytes_saved": "alod_h2d_bytes",
        "host_gather_time_saved_ms": "alod_host_gather_ms",
        "h2d_time_saved_ms": "alod_h2d_transfer_ms",
        "end_to_end_time_saved_ms": "end_to_end_frame_time_ms",
    }
    result = {
        name: paired_distribution(reuse, no_cache, field)
        for name, field in fields.items()
    }
    # Above computes parent - child because paired_distribution is right-left.
    no_cache_end = statistics.fmean(row["end_to_end_frame_time_ms"] for row in no_cache)
    reuse_end = statistics.fmean(row["end_to_end_frame_time_ms"] for row in reuse)
    result.update(
        {
            "paired_samples": len(reuse),
            "retained_rows": metric(reuse, "alod_retained_rows"),
            "newly_loaded_rows": metric(reuse, "alod_rows_loaded"),
            "retention_ratio": metric(reuse, "alod_retention_ratio"),
            "previous_working_set_rows": metric(reuse, "alod_previous_working_set_rows"),
            "end_to_end_change_vs_no_cache_percent": pct_change(reuse_end, no_cache_end),
            "classification": (
                "exact released viewer preceding-working-set reuse semantics: strict "
                "0.3 < previous/current camera-distance ratio < 1.5 per SPT; whole prior "
                "SPT cuts retained; newly required properties loaded; no LRU or bounded cache"
            ),
        }
    )
    return result


def markdown_rendering(rendering: dict[str, Any]) -> str:
    lines = [
        "# ALoD Faithful Rendering Summary",
        "",
        "Three fresh processes per configuration, each with 10 ordered warmups and all 71 frozen measured cameras. Times are pooled CUDA-synchronized measurements.",
        "",
        "| Configuration | End-to-end mean / median / p95 (ms) | FPS from mean | Prep mean (ms) | Raster mean (ms) | Selected hierarchy mean | H2D mean | Peak torch allocated | Whole-device VRAM max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config in CONFIGS:
        entry = rendering["configurations"][config]
        timing = entry["timings_ms"]
        memory = entry["memory_bytes"]
        lines.append(
            f"| `{config}` | {ms(timing['end_to_end_frame_time_ms']['mean'])} / "
            f"{ms(timing['end_to_end_frame_time_ms']['median'])} / "
            f"{ms(timing['end_to_end_frame_time_ms']['p95'])} | "
            f"{entry['end_to_end_fps_from_mean_time']:.2f} | "
            f"{ms(timing['frame_preparation_ms']['mean'])} | "
            f"{ms(timing['rasterization_time_ms']['mean'])} | "
            f"{million(entry['counts']['alod_selected_hierarchy_rows']['mean'])} | "
            f"{million(entry['h2d']['rows']['mean'])} rows | "
            f"{gib(memory['frame_total_peak_allocated_bytes']['maximum']):.2f} GiB | "
            f"{gib(memory['nvml_device_used_bytes']['maximum']):.2f} GiB |"
        )
    lines.extend(
        [
            "",
            "`frame_time_ms` in the raw logs is raster-only. The table uses `end_to_end_frame_time_ms` for total performance and FPS.",
            "",
        ]
    )
    return "\n".join(lines)


def markdown_frustum(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ALoD Frustum Ablation",
            "",
            "Positive avoided/saved values mean frustum-on performed less work than frustum-off. The 213 samples are camera-paired across three fresh-process replicates.",
            "",
            "| Measure | Mean | Median | p95 | Maximum |",
            "| --- | ---: | ---: | ---: | ---: |",
            *[
                f"| {name.replace('_', ' ')} | {values['mean']:.4f} | {values['median']:.4f} | {values['p95']:.4f} | {values['maximum']:.4f} |"
                for name, values in result.items()
                if isinstance(values, dict) and "mean" in values and name != "image_equivalence"
            ],
            "",
            f"Mean selected-hierarchy reduction: {result['mean_selected_hierarchy_reduction_percent']:.4f}%.",
            f"Mean end-to-end change, frustum-on versus off: {result['mean_end_to_end_change_frustum_on_vs_off_percent']:+.3f}%.",
            f"Image check: {result['image_equivalence']['exact_image_count']}/71 bit-identical; maximum RGB absolute difference {result['image_equivalence']['rgb_max_abs']['maximum']:.8f}.",
            "",
            result["interpretation"],
            "",
        ]
    )


def markdown_reuse(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ALoD Released Viewer Previous-Working-Set Reuse",
            "",
            "| Measure | Mean | Median | p95 |",
            "| --- | ---: | ---: | ---: |",
            f"| Retained rows | {result['retained_rows']['mean']:.2f} | {result['retained_rows']['median']:.2f} | {result['retained_rows']['p95']:.2f} |",
            f"| Retention ratio | {result['retention_ratio']['mean']:.4f} | {result['retention_ratio']['median']:.4f} | {result['retention_ratio']['p95']:.4f} |",
            f"| Newly loaded rows | {result['newly_loaded_rows']['mean']:.2f} | {result['newly_loaded_rows']['median']:.2f} | {result['newly_loaded_rows']['p95']:.2f} |",
            f"| H2D rows saved vs no-cache | {result['h2d_rows_saved']['mean']:.2f} | {result['h2d_rows_saved']['median']:.2f} | {result['h2d_rows_saved']['p95']:.2f} |",
            f"| End-to-end time saved (ms) | {result['end_to_end_time_saved_ms']['mean']:.2f} | {result['end_to_end_time_saved_ms']['median']:.2f} | {result['end_to_end_time_saved_ms']['p95']:.2f} |",
            "",
            f"Mean end-to-end change versus no-cache: {result['end_to_end_change_vs_no_cache_percent']:+.2f}%.",
            "",
            result["classification"] + ".",
            "",
        ]
    )


def markdown_paper(rendering: dict[str, Any], hspt_bfs: dict[str, Any]) -> str:
    no_cache = rendering["configurations"]["ALOD-REPRO-HOST-NOCACHE"]["timings_ms"]["end_to_end_frame_time_ms"]
    no_frustum = rendering["configurations"]["ALOD-REPRO-NOFRUSTUM"]["timings_ms"]["end_to_end_frame_time_ms"]
    return "\n".join(
        [
            "# ALoD Paper/Public-Source Comparison",
            "",
            "PAPER-SOURCE-DIVERGENCE: no released rendering entry point exactly implements the paper-reported full bounded rendering-cache configuration.",
            "",
            "| Mode | Paper Table 3, four reported scenes (ms) | This campaign | Fidelity / comparability |",
            "| --- | --- | ---: | --- |",
            "| Render Full | 48.1 / 68.3 / 47.1 / 83.2 | unavailable | No exact released rendering entry point; not fabricated. |",
            f"| w/o cache | 119.4 / 127.4 / 92.6 / 222.3 | {no_cache['mean']:.2f} ms end-to-end | Closest public evaluator-style no-cache proxy; different hierarchy, cameras, training envelope, and GPU. |",
            f"| w/o frustum | 52.3 / 89.3 / 38.3 / 110.2 | {no_frustum['mean']:.2f} ms end-to-end | Source component ablation on no-cache proxy; paper row retained its unreleased cache, so numerical comparison is invalid. |",
            f"| HSPT cut | 31.9 / 45.7 / 31.3 / 36.5 | {hspt_bfs['timing']['hspt_per_camera_median_ms']['mean']:.2f} ms | Exact released evaluator cut, diagnostic cameras; trend-only. |",
            f"| BFS cut | 47.8 / OOM / 40.0 / 53.7 | {hspt_bfs['timing']['bfs_per_camera_median_ms']['mean']:.2f} ms | Minimal source-faithful reconstruction; no released BFS executable; trend-only. |",
            "",
            "The local results must not be used as direct paper-number reproduction: assets, dataset scale, trained hierarchy, cameras, rendering harness, and GPU differ, and some paper configurations were not released.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    repo = args.repo_root.resolve()
    pooled, structure = load_runs(root)
    by_replicate = structure["by_replicate"]
    quality = read_json(root / "quality/results/summary.json")
    hspt_bfs = read_json(root / "hspt_bfs/summary.json")
    lod = read_json(root / "lod_coverage_71/summary.json")
    parity = read_json(root / "checks/selection_oracle_parity.json")
    pre_raster = read_json(root / "checks/formal_pre_raster_parity.json")
    if parity.get("status") != "PASS-ALOD-CPU-REFERENCE-SELECTION":
        raise AssertionError("selection parity gate is not PASS")
    if pre_raster.get("status") != "PASS-ALOD-FORMAL-FAITHFUL-PRE-RASTER":
        raise AssertionError("pre-raster parity gate is not PASS")
    if quality.get("hierarchy_sha256") != EXPECTED_HIERARCHY_SHA256:
        raise AssertionError("quality hierarchy hash mismatch")
    rendering = aggregate_rendering(pooled, by_replicate)
    frustum = aggregate_frustum(by_replicate, quality)
    reuse = aggregate_reuse(by_replicate)
    tables = root / "tables"
    write_json(tables / "rendering_summary.json", rendering)
    write_json(tables / "frustum_ablation.json", frustum)
    write_json(tables / "viewer_reuse.json", reuse)
    write_json(
        root / "memory_summary.json",
        {
            config: entry["memory_bytes"]
            for config, entry in rendering["configurations"].items()
        },
    )
    write_json(
        root / "h2d_summary.json",
        {
            config: entry["h2d"]
            for config, entry in rendering["configurations"].items()
        },
    )
    (tables / "rendering_summary.md").write_text(markdown_rendering(rendering), encoding="utf-8")
    (tables / "frustum_ablation.md").write_text(markdown_frustum(frustum), encoding="utf-8")
    (tables / "viewer_reuse.md").write_text(markdown_reuse(reuse), encoding="utf-8")
    (tables / "paper_comparison.md").write_text(
        markdown_paper(rendering, hspt_bfs), encoding="utf-8"
    )
    provenance_files = [
        repo / "benchmarks/configs/alod_repro_host_nocache.json",
        repo / "benchmarks/configs/alod_repro_nofrustum.json",
        repo / "benchmarks/configs/alod_repro_previous_reuse.json",
        repo / "datasets/alod/ALOD-REPRO-MC-MIXED-01/formal_evaluation_cameras.json",
        repo / "scripts/alod/run_faithful_reproduction_campaign.py",
        repo / "scripts/alod/validate_formal_faithful_render.py",
        repo / "scripts/alod/benchmark_official_hspt_bfs.py",
        repo / "scripts/alod/evaluate_official_faithful_quality.py",
        repo / "scripts/alod/aggregate_faithful_reproduction.py",
        repo / "docs/results/alod_faithful_reproduction.md",
        Path("/home/luis/tese/data/alod/ALOD-REPRO-MC-MIXED-01/manifest.json"),
        Path("/home/luis/tese/data/alod/ALOD-REPRO-MC-MIXED-01/diagnostic_cameras.json"),
        Path("/home/luis/tese/runs/alod/oracles/ALOD-MIXED-RUN009-SELECTION-ORACLE-001/manifest.json"),
    ]
    provenance = {
        "campaign_id": "ALOD-FAITHFUL-REPRODUCTION-001",
        "hierarchy_sha256": EXPECTED_HIERARCHY_SHA256,
        "official_source_commit": "73f547341547ae021ff036ad74fdf73a1240b450",
        "frozen_camera_order": [
            {"camera_index": index, "camera_id": camera_id}
            for index, camera_id in structure["camera_order"]
        ],
        "file_sha256": {str(path): sha256(path) for path in provenance_files},
        "quality_weight_sha256": {
            str(path): sha256(path)
            for path in sorted((root / "quality/torch_cache/hub/checkpoints").glob("*.pth"))
        },
        "project_extensions_enabled": False,
        "forbidden_components": {
            "prematerialization_culling": False,
            "cuda_materialization_filter": False,
            "project_525k_cache": False,
            "project_160k_cache": False,
            "optimized_project_selector": False,
        },
        "component_gates": {
            "selection": parity["status"],
            "pre_raster": pre_raster["status"],
            "lod_coverage": lod["status"],
            "hspt_bfs": hspt_bfs["status"],
            "quality": quality["status"],
        },
    }
    write_json(root / "provenance.json", provenance)
    print(json.dumps({"status": "PASS-ALOD-REPRO-AGGREGATION", "root": str(root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
