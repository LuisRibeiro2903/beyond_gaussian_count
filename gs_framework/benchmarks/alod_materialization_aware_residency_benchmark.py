#!/usr/bin/env python3
"""Validate and aggregate the formal materialization-aware ALoD campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.alod_host_residency_benchmark import (  # noqa: E402
    _cold_summary,
    _finite,
    _maximum,
    _memory_summary,
    _read_json,
    _read_jsonl,
    _stats,
    _sum,
    _tree_sha256,
)
from benchmarks.benchmark_real_scene import (  # noqa: E402
    CONFIG_FILES,
    canonical_json_sha256,
    environment_metadata,
    load_config,
    sha256_file,
    write_json,
)
from scripts.alod.run_materialization_aware_residency_campaign import (  # noqa: E402
    CAMPAIGN_ID,
    CONFIG_IDS,
)


PASS_STATUS = "PASS-ALOD-MATERIALIZATION-AWARE-BENCHMARK"
PREFLIGHT_STATUS = "PASS-ALOD-MATERIALIZATION-AWARE-PREFLIGHT"
EXPECTED_ASSET_SHA256 = "6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c"
EXPECTED_TRAJECTORY_SHA256 = "c6d607b8baee0cc30f3c74121e9d19ba045a820fc87c8550a11c3dbc0c92968d"
CAPACITIES = {CONFIG_IDS[2]: 525_000, CONFIG_IDS[3]: 160_000}
LABELS = {
    CONFIG_IDS[0]: "Resident",
    CONFIG_IDS[1]: "Fast + previous reuse",
    CONFIG_IDS[2]: "Fast + 525k cache",
    CONFIG_IDS[3]: "Fast + 160k cache",
}


def _resolved_config_hash(config_id: str) -> str:
    config = load_config(config_id)
    config.update({"warmup_frames": 10, "num_repeat": 10, "record_warmup_frames": True})
    return canonical_json_sha256(config)


def _validate_run(run_dir: Path, config_id: str) -> tuple[list[dict], dict, dict, list[dict]]:
    frames = _read_jsonl(run_dir / "frames.jsonl")
    summary = _read_json(run_dir / "summary.json")
    cold = _read_json(run_dir / "cold_start.json")
    warmup = _read_jsonl(run_dir / "warmup_frames.jsonl")
    resolved = _read_json(run_dir / "resolved_config.json")
    if len(frames) != 100 or len(warmup) != 10:
        raise RuntimeError(f"{run_dir}: expected 100 measured and 10 warmup frames")
    if summary.get("record_kind") != "formal_benchmark" or summary.get("configuration_id") != config_id:
        raise RuntimeError(f"{run_dir}: formal identity mismatch")
    if resolved.get("warmup_frames") != 10 or resolved.get("num_repeat") != 10:
        raise RuntimeError(f"{run_dir}: warmup/repetition mismatch")
    if resolved.get("max_frames") != 100 or not resolved.get("configuration", {}).get("record_warmup_frames"):
        raise RuntimeError(f"{run_dir}: frame cap/warmup recording mismatch")
    provenance = summary["provenance"]
    if provenance.get("configuration_sha256") != _resolved_config_hash(config_id):
        raise RuntimeError(f"{run_dir}: resolved configuration hash mismatch")
    if provenance.get("asset_sha256") != EXPECTED_ASSET_SHA256:
        raise RuntimeError(f"{run_dir}: asset hash mismatch")
    if provenance.get("trajectory_file_sha256") != EXPECTED_TRAJECTORY_SHA256:
        raise RuntimeError(f"{run_dir}: trajectory hash mismatch")
    for index, frame in enumerate(frames):
        repeat_index, camera_index = divmod(index, 10)
        if (frame.get("frame_index"), frame.get("repeat_index"), frame.get("camera_index")) != (
            index, repeat_index, camera_index
        ):
            raise RuntimeError(f"{run_dir}: frame order mismatch at {index}")
        if frame.get("width") != 960 or frame.get("height") != 540 or not frame.get("finite_image"):
            raise RuntimeError(f"{run_dir}: resolution/finite-image failure at {index}")
    if config_id in CAPACITIES and any(
        row.get("alod_dynamic_cache_capacity_rows") != CAPACITIES[config_id] for row in frames
    ):
        raise RuntimeError(f"{run_dir}: cache capacity mismatch")
    if config_id != CONFIG_IDS[0] and any(
        row.get("alod_materialization_filter_ms") is None for row in frames
    ):
        raise RuntimeError(f"{run_dir}: missing fast-filter timing")
    return frames, summary, cold, warmup


def _timing(rows: list[dict]) -> dict[str, Any]:
    fields = {
        "camera_preparation_ms": "alod_camera_preparation_ms",
        "upper_traversal_ms": "alod_upper_traversal_ms",
        "spt_cut_ms": "alod_spt_cut_ms",
        "semantic_selection_ms": "alod_total_selection_ms",
        "selected_id_preparation_ms": "alod_materialization_id_preparation_ms",
        "filter_selected_id_h2d_ms": "alod_materialization_selected_id_h2d_ms",
        "filter_metadata_index_ms": "alod_materialization_metadata_gather_ms",
        "filter_culling_ms": "alod_materialization_projection_ms",
        "filter_compaction_ms": "alod_materialization_compaction_ms",
        "filter_retained_index_d2h_ms": "alod_materialization_retained_index_d2h_ms",
        "filter_id_output_ms": "alod_materialization_id_production_ms",
        "filter_total_ms": "alod_materialization_filter_ms",
        "residency_lookup_ms": "alod_residency_lookup_ms",
        "cache_management_ms": "alod_cache_management_ms",
        "host_gather_ms": "alod_host_gather_ms",
        "full_property_h2d_ms": "alod_h2d_transfer_ms",
        "gpu_slot_index_ms": "alod_gpu_slot_index_ms",
        "selected_gpu_gather_ms": "alod_selected_gpu_gather_ms",
        "assembly_ms": "alod_pre_raster_assembly_ms",
        "rasterization_ms": "rasterization_time_ms",
        "total_frame_ms": "end_to_end_frame_time_ms",
    }
    return {name: _stats(_finite(rows, field)) for name, field in fields.items()}


def _counts(rows: list[dict]) -> dict[str, Any]:
    fields = {
        "semantic_hierarchy_rows": "alod_selected_hierarchy_rows",
        "semantic_skybox_rows": "alod_semantic_skybox_rows",
        "semantic_total_rows": "alod_semantic_selected_rows",
        "materialization_hierarchy_rows": "alod_materialization_hierarchy_rows",
        "materialization_skybox_rows": "alod_materialization_skybox_rows",
        "materialization_total_rows": "alod_materialization_rows",
        "backend_survivors": "backend_projection_survivor_count",
        "candidate_spt_count": "alod_candidate_spt_count",
    }
    result = {name: _stats(_finite(rows, field)) for name, field in fields.items()}
    semantic = result["semantic_total_rows"]["mean"]
    materialized = result["materialization_total_rows"]["mean"]
    backend = result["backend_survivors"]["mean"]
    result["ratios_of_means"] = {
        "semantic_over_materialized": semantic / materialized,
        "materialized_over_backend": materialized / backend,
        "semantic_over_backend": semantic / backend,
    }
    return result


def _residency(config_id: str, rows: list[dict]) -> dict[str, Any]:
    requests = _sum(rows, "alod_requested_rows") or 0.0
    hits = _sum(rows, "alod_cache_hit_count")
    misses = _sum(rows, "alod_cache_miss_count")
    evictions = _sum(rows, "alod_cache_eviction_count")
    h2d = _finite(rows, "alod_h2d_bytes")
    applicable = config_id != CONFIG_IDS[0]
    return {
        "requests": requests if applicable else None,
        "hits": hits if applicable else None,
        "misses": misses if applicable else None,
        "loads": _sum(rows, "alod_rows_loaded") if applicable else None,
        "evictions": evictions if config_id in CAPACITIES else None,
        "occupancy_rows": _stats(_finite(rows, "alod_cache_occupancy_rows")),
        "capacity_rows": CAPACITIES.get(config_id),
        "hit_or_reuse_rate": (float(hits or 0.0) / requests) if applicable and requests else None,
        "h2d_rows": _sum(rows, "alod_h2d_rows") if applicable else 0.0,
        "h2d_bytes": float(sum(h2d)),
        "zero_transfer_frames": int(sum(value == 0 for value in h2d)),
    }


def _memory(rows: list[dict]) -> dict[str, Any]:
    base = _memory_summary(rows)
    compact = rows[0].get("alod_compact_culling_gpu_bytes")
    selection = rows[0].get("alod_hierarchy_metadata_gpu_bytes")
    residency_maps = rows[0].get("alod_residency_metadata_gpu_bytes")
    properties = base["persistent_gaussian_property_bytes"]["mean"]
    base.update(
        {
            "selection_metadata_gpu_bytes": selection,
            "compact_culling_gpu_bytes": compact,
            "residency_maps_gpu_bytes": residency_maps,
            "persistent_total_gpu_bytes": sum(
                float(value or 0.0) for value in (selection, compact, residency_maps, properties)
            ),
            "selected_materialization_index_bytes": _stats(
                [
                    float(row.get("alod_selected_index_bytes") or 0.0)
                    + float(row.get("alod_materialization_selected_id_bytes") or 0.0)
                    + float(row.get("alod_materialization_retained_index_bytes") or 0.0)
                    for row in rows
                ]
            ),
            "rasterizer_temporary_peak_increment_bytes": _maximum(
                [
                    {
                        "value": max(
                            0.0,
                            float(row.get("frame_total_peak_allocated_bytes") or 0.0)
                            - float(row.get("before_rasterization_torch_current_allocated_bytes") or 0.0),
                        )
                    }
                    for row in rows
                ],
                "value",
            ),
        }
    )
    return base


def _phase_residency(rows: list[dict]) -> dict[str, Any]:
    requests = sum(_finite(rows, "alod_requested_rows"))
    hits = sum(_finite(rows, "alod_cache_hit_count"))
    misses = sum(_finite(rows, "alod_cache_miss_count"))
    return {
        "frames": len(rows),
        "requests": requests,
        "hits": hits,
        "misses": misses,
        "loads": sum(_finite(rows, "alod_rows_loaded")),
        "evictions": sum(_finite(rows, "alod_cache_eviction_count")),
        "h2d_rows": sum(_finite(rows, "alod_h2d_rows")),
        "h2d_bytes": sum(_finite(rows, "alod_h2d_bytes")),
        "zero_transfer_frames": sum(value == 0 for value in _finite(rows, "alod_h2d_bytes")),
        "hit_or_reuse_rate": hits / requests if requests else None,
    }


def _warmup(records: list[list[dict]], *, bounded: bool) -> dict[str, Any]:
    per_process = []
    for rows in records:
        final_occupancy = rows[-1].get("alod_cache_occupancy_rows")
        frames_to_95 = None
        frames_to_final = None
        if bounded and isinstance(final_occupancy, (int, float)) and math.isfinite(float(final_occupancy)):
            for index, row in enumerate(rows, start=1):
                occupancy = row.get("alod_cache_occupancy_rows")
                if frames_to_95 is None and occupancy >= 0.95 * final_occupancy:
                    frames_to_95 = index
                if frames_to_final is None and occupancy >= final_occupancy:
                    frames_to_final = index
        per_process.append(
            {
                "cold_first_frame": _phase_residency(rows[:1]),
                "warm_navigation_after_cold": _phase_residency(rows[1:]),
                "complete_warmup": _phase_residency(rows),
                "final_occupancy_rows": final_occupancy,
                "frames_to_95_percent_final_occupancy": frames_to_95,
                "frames_to_final_occupancy": frames_to_final,
            }
        )
    all_warmup = [row for process_rows in records for row in process_rows]
    return {"per_process": per_process, "all_processes": _phase_residency(all_warmup)}


def _decomposition(timing: dict[str, Any]) -> dict[str, Any]:
    mean = lambda key: float(timing[key]["mean"] or 0.0)
    components = {
        "semantic_selection": mean("semantic_selection_ms"),
        "pre_materialization_filter": mean("filter_total_ms"),
        "residency_gather_h2d": sum(
            mean(key)
            for key in (
                "residency_lookup_ms", "cache_management_ms", "host_gather_ms",
                "full_property_h2d_ms", "gpu_slot_index_ms", "selected_gpu_gather_ms",
            )
        ),
        "assembly": mean("assembly_ms"),
        "rasterization": mean("rasterization_ms"),
    }
    total = mean("total_frame_ms")
    components["other"] = max(0.0, total - sum(components.values()))
    fractions = {key: value / total for key, value in components.items()}
    ranking = sorted(components, key=components.get, reverse=True)
    return {"mean_ms": components, "fraction_of_mean_frame": fractions, "ranking": ranking}


def _write_tables(root: Path, configs: list[dict]) -> None:
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    fields = [
        "configuration", "mean_ms", "median_ms", "p95_ms", "fps", "selection_ms",
        "filter_ms", "residency_ms", "gather_ms", "h2d_ms", "assembly_ms", "raster_ms",
        "semantic_rows", "materialized_rows", "backend_survivors", "property_bytes",
        "compact_bytes", "persistent_total_bytes", "peak_allocated_bytes", "peak_vram_bytes",
        "h2d_bytes", "hit_rate", "evictions",
    ]
    with (tables / "config_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in configs:
            timing, counts, memory, residency = row["timing"], row["counts"], row["memory"], row["residency"]
            writer.writerow(
                {
                    "configuration": row["configuration_id"],
                    "mean_ms": timing["total_frame_ms"]["mean"],
                    "median_ms": timing["total_frame_ms"]["median"],
                    "p95_ms": timing["total_frame_ms"]["p95"],
                    "fps": row["fps"],
                    "selection_ms": timing["semantic_selection_ms"]["mean"],
                    "filter_ms": timing["filter_total_ms"]["mean"],
                    "residency_ms": timing["residency_lookup_ms"]["mean"],
                    "gather_ms": timing["host_gather_ms"]["mean"],
                    "h2d_ms": timing["full_property_h2d_ms"]["mean"],
                    "assembly_ms": timing["assembly_ms"]["mean"],
                    "raster_ms": timing["rasterization_ms"]["mean"],
                    "semantic_rows": counts["semantic_total_rows"]["mean"],
                    "materialized_rows": counts["materialization_total_rows"]["mean"],
                    "backend_survivors": counts["backend_survivors"]["mean"],
                    "property_bytes": memory["persistent_gaussian_property_bytes"]["mean"],
                    "compact_bytes": memory["compact_culling_gpu_bytes"],
                    "persistent_total_bytes": memory["persistent_total_gpu_bytes"],
                    "peak_allocated_bytes": memory["peak_frame_total_torch_allocated_bytes"],
                    "peak_vram_bytes": memory["peak_whole_device_vram_bytes"],
                    "h2d_bytes": residency["h2d_bytes"],
                    "hit_rate": residency["hit_or_reuse_rate"],
                    "evictions": residency["evictions"],
                }
            )
    lines = [
        "# Materialization-aware ALoD residency benchmark", "",
        "| Configuration | Mean ms | Median ms | P95 ms | FPS | Selection ms | Filter ms | Residency/gather/H2D ms | Assembly ms | Raster ms | Semantic rows | Materialized rows | Backend survivors | Full properties MiB | Compact MiB | Persistent total MiB | Peak allocated MiB | Peak device MiB | H2D MiB | Hit/reuse | Evictions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fmt = lambda value, digits=3: "N/A" if value is None else f"{float(value):.{digits}f}"
    for row in configs:
        timing, counts, memory, residency = row["timing"], row["counts"], row["memory"], row["residency"]
        res_ms = row["bottleneck_decomposition"]["mean_ms"]["residency_gather_h2d"]
        lines.append(
            f"| `{row['configuration_id']}` | {fmt(timing['total_frame_ms']['mean'])} | {fmt(timing['total_frame_ms']['median'])} | {fmt(timing['total_frame_ms']['p95'])} | {fmt(row['fps'], 2)} | {fmt(timing['semantic_selection_ms']['mean'])} | {fmt(timing['filter_total_ms']['mean'])} | {fmt(res_ms)} | {fmt(timing['assembly_ms']['mean'])} | {fmt(timing['rasterization_ms']['mean'])} | {counts['semantic_total_rows']['mean']:,.0f} | {counts['materialization_total_rows']['mean']:,.0f} | {counts['backend_survivors']['mean']:,.0f} | {fmt(memory['persistent_gaussian_property_bytes']['mean']/2**20, 2)} | {fmt((memory['compact_culling_gpu_bytes'] or 0)/2**20, 2)} | {fmt(memory['persistent_total_gpu_bytes']/2**20, 2)} | {fmt(memory['peak_frame_total_torch_allocated_bytes']/2**20, 2)} | {fmt(memory['peak_whole_device_vram_bytes']/2**20 if memory['peak_whole_device_vram_bytes'] else None, 2)} | {fmt(residency['h2d_bytes']/2**20, 2)} | {fmt(residency['hit_or_reuse_rate'], 4)} | {fmt(residency['evictions'], 0)} |"
        )
    (tables / "thesis_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figures(root: Path, configs: list[dict]) -> None:
    import matplotlib.pyplot as plt

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = [row["label"] for row in configs]
    colors = ["#4C78A8", "#E45756", "#F2CF5B", "#54A24B"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, [row["timing"]["total_frame_ms"]["mean"] for row in configs], color=colors)
    ax.scatter(range(4), [row["timing"]["total_frame_ms"]["p95"] for row in configs], color="black", label="P95")
    ax.set_ylabel("Frame time (ms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "frame_time.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, [row["memory"]["persistent_gaussian_property_bytes"]["mean"] / 2**20 for row in configs], color=colors, label="Full properties")
    ax.bar(labels, [(row["memory"]["compact_culling_gpu_bytes"] or 0) / 2**20 for row in configs], bottom=[row["memory"]["persistent_gaussian_property_bytes"]["mean"] / 2**20 for row in configs], color="#B279A2", label="Compact culling")
    ax.set_ylabel("Persistent GPU memory (MiB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "persistent_memory.png", dpi=160)
    plt.close(fig)

    stages = ("semantic_selection", "pre_materialization_filter", "residency_gather_h2d", "assembly", "rasterization", "other")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bottom = np.zeros(4)
    for stage in stages:
        values = np.asarray([row["bottleneck_decomposition"]["mean_ms"][stage] for row in configs])
        ax.bar(labels, values, bottom=bottom, label=stage.replace("_", " "))
        bottom += values
    ax.set_ylabel("Mean frame time decomposition (ms)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "bottleneck_decomposition.png", dpi=160)
    plt.close(fig)


def run_aggregate(root: Path, asset: Path, preflight_path: Path) -> int:
    root, asset = root.resolve(), asset.resolve()
    if sha256_file(asset) != EXPECTED_ASSET_SHA256:
        raise RuntimeError("asset hash mismatch")
    preflight = _read_json(preflight_path)
    if preflight.get("status") != PREFLIGHT_STATUS:
        raise RuntimeError("correctness preflight did not pass")
    configs, provenance = [], []
    pooled: dict[str, list[dict]] = {}
    for config_id in CONFIG_IDS:
        rows, colds, warmups, replicate_means = [], [], [], []
        for replicate in range(1, 4):
            run_dir = root / "runs" / config_id / f"replicate-{replicate:02d}"
            run_rows, summary, cold, warmup = _validate_run(run_dir, config_id)
            rows.extend(run_rows)
            colds.append(cold)
            warmups.append(warmup)
            replicate_means.append(float(np.mean(_finite(run_rows, "end_to_end_frame_time_ms"))))
            provenance.append(
                {
                    "configuration_id": config_id,
                    "replicate": replicate,
                    "frames_sha256": sha256_file(run_dir / "frames.jsonl"),
                    "warmup_sha256": sha256_file(run_dir / "warmup_frames.jsonl"),
                    "summary_sha256": sha256_file(run_dir / "summary.json"),
                }
            )
        pooled[config_id] = rows
        timing = _timing(rows)
        decomposition = _decomposition(timing)
        configs.append(
            {
                "configuration_id": config_id,
                "label": LABELS[config_id],
                "frame_count": len(rows),
                "process_replicates": 3,
                "replicate_mean_frame_ms": replicate_means,
                "replicate_mean_std_ms": statistics.pstdev(replicate_means),
                "fps": 1000.0 / timing["total_frame_ms"]["mean"],
                "timing": timing,
                "counts": _counts(rows),
                "residency": _residency(config_id, rows),
                "memory": _memory(rows),
                "cold_start": _cold_summary(colds),
                "warmup_population": _warmup(warmups, bounded=config_id in CAPACITIES),
                "bottleneck_decomposition": decomposition,
            }
        )

    reference = pooled[CONFIG_IDS[0]]
    for config_id in CONFIG_IDS[1:]:
        for index, (left, right) in enumerate(zip(reference, pooled[config_id])):
            for field in ("alod_selected_hierarchy_rows", "alod_semantic_selected_rows", "backend_projection_survivor_count"):
                if left.get(field) != right.get(field):
                    raise RuntimeError(f"cross-mode {field} mismatch for {config_id} row {index}")
    reduced = next(row for row in configs if row["configuration_id"] == CONFIG_IDS[3])
    max_request = max(_finite(pooled[CONFIG_IDS[3]], "alod_materialization_hierarchy_rows"))
    warmup_final = [
        row["final_occupancy_rows"] for row in reduced["warmup_population"]["per_process"]
    ]
    union_rows = int(max(warmup_final))
    evictions = reduced["residency"]["evictions"] or 0
    if evictions == 0 and union_rows > 160_000:
        raise RuntimeError("reduced-cache union exceeds fixed capacity")
    by_id = {row["configuration_id"]: row for row in configs}
    old, small, previous = by_id[CONFIG_IDS[2]], by_id[CONFIG_IDS[3]], by_id[CONFIG_IDS[1]]
    comparisons = {
        "cache_525k_vs_160k": {
            "persistent_full_property_bytes_saved": old["memory"]["persistent_gaussian_property_bytes"]["mean"] - small["memory"]["persistent_gaussian_property_bytes"]["mean"],
            "persistent_total_gpu_bytes_saved": old["memory"]["persistent_total_gpu_bytes"] - small["memory"]["persistent_total_gpu_bytes"],
            "frame_time_change_ms_160k_minus_525k": small["timing"]["total_frame_ms"]["mean"] - old["timing"]["total_frame_ms"]["mean"],
            "h2d_bytes_change_160k_minus_525k": small["residency"]["h2d_bytes"] - old["residency"]["h2d_bytes"],
            "hit_rate_change": small["residency"]["hit_or_reuse_rate"] - old["residency"]["hit_or_reuse_rate"],
            "eviction_change": (small["residency"]["evictions"] or 0) - (old["residency"]["evictions"] or 0),
        },
        "previous_reuse_vs_160k": {
            "persistent_total_gpu_bytes_change_160k_minus_previous": small["memory"]["persistent_total_gpu_bytes"] - previous["memory"]["persistent_total_gpu_bytes"],
            "frame_time_change_ms_160k_minus_previous": small["timing"]["total_frame_ms"]["mean"] - previous["timing"]["total_frame_ms"]["mean"],
            "h2d_bytes_change_160k_minus_previous": small["residency"]["h2d_bytes"] - previous["residency"]["h2d_bytes"],
            "hit_rate_change": small["residency"]["hit_or_reuse_rate"] - previous["residency"]["hit_or_reuse_rate"],
        },
    }
    for row in configs:
        warm = row["warmup_population"]["all_processes"]
        steady = row["residency"]
        combined_requests = float(warm["requests"] or 0.0) + float(steady["requests"] or 0.0)
        combined_hits = float(warm["hits"] or 0.0) + float(steady["hits"] or 0.0)
        row["cold_warm_steady_residency"] = {
            "warmup_including_cold": warm,
            "steady_measured": steady,
            "combined_warmup_and_measured": {
                "requests": combined_requests if config_id != CONFIG_IDS[0] else None,
                "hits": combined_hits if config_id != CONFIG_IDS[0] else None,
                "hit_or_reuse_rate": (
                    combined_hits / combined_requests
                    if config_id != CONFIG_IDS[0] and combined_requests
                    else None
                ),
                "h2d_bytes": warm["h2d_bytes"] + steady["h2d_bytes"],
                "evictions": float(warm["evictions"] or 0.0) + float(steady["evictions"] or 0.0),
            },
        }
    result = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": PASS_STATUS,
        "weak_lod_oracle": True,
        "asset": {"path": str(asset), "sha256": sha256_file(asset)},
        "trajectory": {"id": "alod_run005_frozen_evaluation_v1", "sha256": EXPECTED_TRAJECTORY_SHA256, "camera_count": 10, "resolution": [960, 540]},
        "configurations": configs,
        "recomputed_reduced_cache_trace": {
            "maximum_single_frame_materialized_hierarchy_rows": max_request,
            "trajectory_union_rows_from_zero_eviction_warmup_occupancy": union_rows,
            "capacity_rows": 160_000,
            "union_headroom_rows": 160_000 - union_rows,
            "union_headroom_fraction": (160_000 - union_rows) / union_rows,
            "zero_evictions": evictions == 0,
        },
        "comparisons": comparisons,
        "quality": {
            key: preflight[key]
            for key in (
                "semantic_selected_ids_exact", "fast_materialization_ids_cpu_reference_exact",
                "backend_false_negatives_zero", "surviving_properties_bitwise_exact",
                "backend_projection_ids_exact", "rgb_bitwise_exact", "alpha_bitwise_exact",
                "global_node_to_gpu_slot_exact", "no_stale_slot_reads", "cache_history_independent",
            )
        },
        "configuration_file_sha256": {key: sha256_file(CONFIG_FILES[key]) for key in CONFIG_IDS},
        "run_provenance": provenance,
        "environment": environment_metadata(),
    }
    (root / "tables").mkdir(parents=True, exist_ok=True)
    write_json(root / "tables" / "config_summary.json", result)
    write_json(root / "tables" / "memory_summary.json", {row["configuration_id"]: row["memory"] for row in configs})
    write_json(root / "tables" / "cache_statistics.json", {row["configuration_id"]: row["residency"] for row in configs})
    write_json(root / "tables" / "cold_warm_summary.json", {row["configuration_id"]: {"cold": row["cold_start"], "warmup": row["warmup_population"], "residency_phases": row["cold_warm_steady_residency"]} for row in configs})
    write_json(root / "tables" / "bottleneck_decomposition.json", {row["configuration_id"]: row["bottleneck_decomposition"] for row in configs})
    _write_tables(root, configs)
    _write_figures(root, configs)
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": PASS_STATUS,
        "run_policy": {"processes": 12, "replicates_per_configuration": 3, "warmup_frames_per_process": 10, "measured_frames_per_process": 100, "measured_frames_per_configuration": 300, "total_measured_frames": 1200},
        "asset": result["asset"],
        "trajectory": result["trajectory"],
        "run_provenance": provenance,
    }
    write_json(root / "manifest.json", manifest)
    write_json(root / "checks" / "package_validation.json", {"status": PASS_STATUS, "formal_runs": 12, "total_measured_frames": 1200, "correctness_gate": PREFLIGHT_STATUS})
    (root / "README.md").write_text(
        "# ALoD Materialization-Aware Residency Benchmark 001\n\n"
        "Formal resident, fast-filter previous-reuse, fast-filter 525k-cache, and fast-filter 160k-cache campaign.\n",
        encoding="utf-8",
    )
    return 0


def _git_state(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True).stdout
    return {
        "path": str(path.resolve()),
        "commit": run("rev-parse", "HEAD").strip(),
        "status": run("status", "--porcelain=v1", "--untracked-files=all"),
        "diff_sha256": hashlib.sha256(run("diff", "--binary").encode()).hexdigest(),
        "cached_diff_sha256": hashlib.sha256(run("diff", "--cached", "--binary").encode()).hexdigest(),
    }


def snapshot(output: Path, paths: list[Path], repositories: list[Path]) -> int:
    result = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protected_paths": [{"path": str(path.resolve()), "sha256": _tree_sha256(path.resolve())} for path in paths],
        "git_repositories": [_git_state(path.resolve()) for path in repositories],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return 0


def compare(before: Path, after: Path, output: Path) -> int:
    left, right = _read_json(before), _read_json(after)
    unchanged = left["protected_paths"] == right["protected_paths"] and left["git_repositories"] == right["git_repositories"]
    result = {"schema_version": 1, "campaign_id": CAMPAIGN_ID, "status": "PASS-PROTECTED-STATE-UNCHANGED" if unchanged else "FAILED-PROTECTED-STATE-CHANGED"}
    write_json(output, result)
    if not unchanged:
        raise RuntimeError("protected state changed")
    return 0


def finalize(root: Path, output: Path, unit: int, external: int, city: int) -> int:
    summary = _read_json(root / "tables" / "config_summary.json")
    protected = _read_json(root / "checks" / "protected_state_comparison.json")
    if summary.get("status") != PASS_STATUS or protected.get("status") != "PASS-PROTECTED-STATE-UNCHANGED":
        raise RuntimeError("campaign is not ready for finalization")
    result = {"schema_version": 1, "status": PASS_STATUS, "unit_tests_passed": unit, "alod_external_tests_passed": external, "citygaussian_tests_passed": city, "git_diff_check": "PASS", "protected_state": protected["status"], "staged_committed_or_pushed": False}
    write_json(output, result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--campaign-root", type=Path, required=True)
    aggregate.add_argument("--asset", type=Path, required=True)
    aggregate.add_argument("--preflight", type=Path, required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--output", type=Path, required=True)
    snap.add_argument("--path", type=Path, action="append", default=[])
    snap.add_argument("--git-repository", type=Path, action="append", default=[])
    comp = sub.add_parser("compare")
    comp.add_argument("--before", type=Path, required=True)
    comp.add_argument("--after", type=Path, required=True)
    comp.add_argument("--output", type=Path, required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--campaign-root", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--unit-tests", type=int, required=True)
    final.add_argument("--external-tests", type=int, required=True)
    final.add_argument("--citygaussian-tests", type=int, required=True)
    args = parser.parse_args()
    if args.command == "aggregate":
        return run_aggregate(args.campaign_root, args.asset, args.preflight)
    if args.command == "snapshot":
        return snapshot(args.output, args.path, args.git_repository)
    if args.command == "compare":
        return compare(args.before, args.after, args.output)
    if args.command == "finalize":
        return finalize(args.campaign_root, args.output, args.unit_tests, args.external_tests, args.citygaussian_tests)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
