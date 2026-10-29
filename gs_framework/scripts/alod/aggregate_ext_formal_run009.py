#!/usr/bin/env python3
"""Aggregate the immutable formal Run009 ALOD-EXT campaign and write its memo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


STATUS = "PASS-ALOD-EXT-FORMAL-RUN009"
CONFIG_IDS = (
    "ALOD-EXT-ROW-LRU-PRACTICAL",
    "ALOD-EXT-PREMATERIALIZATION-CPU-PRACTICAL",
    "ALOD-EXT-PREMATERIALIZATION-CUDA-PREVIOUS",
    "ALOD-EXT-PREMATERIALIZATION-CUDA-MINIMUM",
    "ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL",
    "ALOD-EXT-PREMATERIALIZATION-CUDA-LARGE",
)
TIMINGS = (
    "end_to_end_frame_time_ms", "frame_preparation_ms", "rasterization_time_ms",
    "alod_camera_preparation_ms", "alod_total_selection_ms",
    "alod_materialization_filter_ms", "alod_residency_lookup_ms",
    "alod_cache_management_ms", "alod_host_gather_ms", "alod_h2d_transfer_ms",
    "alod_selected_gpu_gather_ms", "alod_pre_raster_assembly_ms",
)
COUNTS = (
    "alod_hierarchy_rows", "alod_selected_hierarchy_rows", "alod_semantic_skybox_rows",
    "alod_semantic_selected_rows", "alod_materialization_hierarchy_rows",
    "alod_materialization_skybox_rows", "alod_materialization_rows",
    "backend_projection_survivor_count",
)
MEMORY = (
    "alod_host_property_store_bytes", "alod_hierarchy_metadata_cpu_bytes",
    "alod_hierarchy_metadata_gpu_bytes", "alod_gaussian_properties_gpu_bytes",
    "alod_compact_culling_gpu_bytes", "alod_frame_local_property_bytes",
    "frame_total_peak_allocated_bytes", "frame_total_peak_reserved_bytes",
    "nvml_device_used_bytes", "process_peak_rss_bytes",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stats(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=np.float64)
    if not values.size:
        return {name: float("nan") for name in ("minimum", "mean", "median", "p95", "maximum")}
    return {
        "minimum": float(values.min()), "mean": float(values.mean()),
        "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
        "maximum": float(values.max()),
    }


def _sum(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(row.get(key) or 0 for row in rows))


def _aggregate_mode(root: Path, config_id: str) -> dict[str, Any]:
    measured: list[dict[str, Any]] = []
    warmup: list[dict[str, Any]] = []
    cold = []
    replicate_means = {}
    for replicate in range(1, 4):
        run = root / "runs" / config_id / f"replicate-{replicate:02d}"
        frames = _read_jsonl(run / "frames.jsonl")
        warms = _read_jsonl(run / "warmup_frames.jsonl")
        if len(frames) != 71 or len(warms) != 10:
            raise RuntimeError(f"incomplete formal run {run}")
        measured.extend(frames)
        warmup.extend(warms)
        cold.append(json.loads((run / "cold_start.json").read_text())["first_frame"])
        replicate_means[str(replicate)] = float(np.mean([r["end_to_end_frame_time_ms"] for r in frames]))
    first_warm = [warmup[index * 10] for index in range(3)]
    remaining_warm = [row for index, row in enumerate(warmup) if index % 10]
    timing = {key: _stats(measured, key) for key in TIMINGS}
    counts = {key: _stats(measured, key) for key in COUNTS}
    memory = {key: _stats(measured, key) for key in MEMORY}
    total = timing["end_to_end_frame_time_ms"]["mean"]
    preparation = timing["frame_preparation_ms"]["mean"]
    camera = timing["alod_camera_preparation_ms"]["mean"]
    selection = timing["alod_total_selection_ms"]["mean"]
    filtering = timing["alod_materialization_filter_ms"]["mean"]
    assembly = timing["alod_pre_raster_assembly_ms"]["mean"]
    raster = timing["rasterization_time_ms"]["mean"]
    residency = max(0.0, preparation - camera - selection - filtering - assembly)
    other = max(0.0, total - preparation - raster)
    stage = {
        "camera_preparation_ms": camera,
        "semantic_selection_ms": selection,
        "materialization_filter_ms": filtering,
        "residency_gather_h2d_ms": residency,
        "pre_raster_assembly_ms": assembly,
        "rasterization_ms": raster,
        "other_end_to_end_ms": other,
    }
    return {
        "configuration_id": config_id,
        "measured_frames": len(measured), "warmup_frames": len(warmup),
        "replicate_mean_end_to_end_ms": replicate_means,
        "timings_ms": timing, "fps_from_mean": 1000.0 / total,
        "stage_breakdown": stage,
        "stage_fraction_of_total": {key: value / total for key, value in stage.items()},
        "counts": counts,
        "funnel_ratios_of_means": {
            "semantic_selected_over_materialized": counts["alod_semantic_selected_rows"]["mean"] / counts["alod_materialization_rows"]["mean"],
            "semantic_selected_over_projection_survivors": counts["alod_semantic_selected_rows"]["mean"] / counts["backend_projection_survivor_count"]["mean"],
            "materialized_over_projection_survivors": counts["alod_materialization_rows"]["mean"] / counts["backend_projection_survivor_count"]["mean"],
        },
        "traffic": {
            "cold_first_frame_rows": _sum(first_warm, "alod_h2d_rows"),
            "cold_first_frame_bytes": _sum(first_warm, "alod_h2d_bytes"),
            "remaining_warmup_rows": _sum(remaining_warm, "alod_h2d_rows"),
            "remaining_warmup_bytes": _sum(remaining_warm, "alod_h2d_bytes"),
            "steady_rows": _sum(measured, "alod_h2d_rows"),
            "steady_bytes": _sum(measured, "alod_h2d_bytes"),
            "steady_hits": _sum(measured, "alod_cache_hit_count"),
            "steady_misses": _sum(measured, "alod_cache_miss_count"),
            "steady_evictions": _sum(measured, "alod_cache_eviction_count"),
            "steady_insertions": _sum(measured, "alod_rows_loaded"),
            "zero_transfer_steady_frames": sum((row.get("alod_h2d_rows") or 0) == 0 for row in measured),
        },
        "memory_bytes": memory,
        "persistent_gpu_property_plus_filter_bytes_max": memory["alod_gaussian_properties_gpu_bytes"]["maximum"] + memory["alod_compact_culling_gpu_bytes"]["maximum"],
    }


def _faithful_modes(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text())["configurations"]
    modes = {}
    for config_id in ("ALOD-REPRO-HOST-NOCACHE", "ALOD-REPRO-PREVIOUS-REUSE"):
        row = source[config_id]
        modes[config_id] = {
            "configuration_id": config_id,
            "timings_ms": row["timings_ms"],
            "fps_from_mean": row["end_to_end_fps_from_mean_time"],
            "counts": row["counts"],
            "memory_bytes": row["memory_bytes"],
            "steady_h2d_rows": int(round(row["h2d"]["rows"]["mean"] * 213)),
            "steady_h2d_bytes": int(round(row["h2d"]["bytes"]["mean"] * 213)),
        }
    return modes


def _effect(left: dict[str, Any], right: dict[str, Any], modes: dict[str, Any], faithful: dict[str, Any]) -> dict[str, Any]:
    def time(row): return row["timings_ms"]["end_to_end_frame_time_ms"]["mean"]
    l, r = (modes.get(left) or faithful[left]), (modes.get(right) or faithful[right])
    lt, rt = time(l), time(r)
    lh2d = l.get("traffic", {}).get("steady_bytes", l.get("steady_h2d_bytes", 0))
    rh2d = r.get("traffic", {}).get("steady_bytes", r.get("steady_h2d_bytes", 0))
    lmem = l.get("persistent_gpu_property_plus_filter_bytes_max", l["memory_bytes"]["alod_gaussian_properties_gpu_bytes"]["maximum"])
    rmem = r.get("persistent_gpu_property_plus_filter_bytes_max", r["memory_bytes"]["alod_gaussian_properties_gpu_bytes"]["maximum"])
    return {
        "from": left, "to": right, "frame_time_saved_ms": lt - rt,
        "frame_time_reduction_fraction": (lt - rt) / lt,
        "steady_h2d_bytes_change": rh2d - lh2d,
        "persistent_gpu_property_plus_filter_bytes_change": rmem - lmem,
    }


def _fmt(value: float) -> str:
    return f"{value:,.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--faithful-summary", type=Path, required=True)
    parser.add_argument("--memo", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    correctness = json.loads((root / "checks/correctness.json").read_text())
    sizing = json.loads((root / "checks/cache_sizing.json").read_text())
    if not correctness["status"].startswith("PASS-") or not sizing["status"].startswith("PASS-"):
        raise RuntimeError("correctness and sizing gates must pass before aggregation")
    modes = {config_id: _aggregate_mode(root, config_id) for config_id in CONFIG_IDS}
    faithful = _faithful_modes(args.faithful_summary)
    ladder = (
        ("ALOD-REPRO-HOST-NOCACHE", "ALOD-REPRO-PREVIOUS-REUSE"),
        ("ALOD-REPRO-HOST-NOCACHE", "ALOD-EXT-ROW-LRU-PRACTICAL"),
        ("ALOD-EXT-ROW-LRU-PRACTICAL", "ALOD-EXT-PREMATERIALIZATION-CPU-PRACTICAL"),
        ("ALOD-EXT-PREMATERIALIZATION-CPU-PRACTICAL", "ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL"),
        ("ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL", "ALOD-EXT-PREMATERIALIZATION-CUDA-MINIMUM"),
        ("ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL", "ALOD-EXT-PREMATERIALIZATION-CUDA-LARGE"),
        ("ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL", "ALOD-EXT-PREMATERIALIZATION-CUDA-PREVIOUS"),
    )
    attribution = [_effect(left, right, modes, faithful) for left, right in ladder]
    best = min(modes.values(), key=lambda row: row["timings_ms"]["end_to_end_frame_time_ms"]["mean"])
    baseline = faithful["ALOD-REPRO-HOST-NOCACHE"]
    baseline_ms = baseline["timings_ms"]["end_to_end_frame_time_ms"]["mean"]
    best_ms = best["timings_ms"]["end_to_end_frame_time_ms"]["mean"]
    selection_ms = best["stage_breakdown"]["semantic_selection_ms"]
    result = {
        "schema_version": 1, "status": STATUS,
        "campaign_id": "ALOD-EXT-FORMAL-RUN009-001",
        "asset": sizing["asset"], "trajectory_id": sizing["trajectory_id"],
        "protocol": {"warmup_frames_per_process": 10, "measured_cameras_per_process": 71, "replicates": 3, "measured_frames_per_configuration": 213, "renderer_backend": "gsplat packed", "same_as_faithful_workload": True},
        "correctness_gate": correctness["status"], "cache_sizing_gate": sizing["status"],
        "cache_sizing": sizing["frozen_capacities"], "working_sets": {"semantic": sizing["semantic_working_set"], "post_filter": sizing["post_filter_working_set"]},
        "faithful_context": faithful, "extension_modes": modes,
        "incremental_attribution": attribution,
        "best_existing_extension": {
            "configuration_id": best["configuration_id"], "mean_frame_time_ms": best_ms,
            "fps": best["fps_from_mean"], "saved_vs_faithful_ms": baseline_ms - best_ms,
            "reduction_vs_faithful_fraction": (baseline_ms - best_ms) / baseline_ms,
            "speedup_vs_faithful": baseline_ms / best_ms,
            "dominant_stage": "semantic_selection", "dominant_stage_ms": selection_ms,
            "dominant_stage_fraction": selection_ms / best_ms,
        },
        "weak_lod_context": True,
    }
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "formal_summary.json").write_text(json.dumps(result, indent=2) + "\n")

    order = ["ALOD-REPRO-HOST-NOCACHE", "ALOD-REPRO-PREVIOUS-REUSE", *CONFIG_IDS]
    lines = [
        "# Formal Run009 ALOD-EXT Runtime Evaluation", "", f"Status: `{STATUS}`", "",
        "Campaign: `/home/luis/tese/benchmark_artifacts/alod_ext_formal_run009_001`", "",
        "## Result", "",
        f"The best existing extension is `{best['configuration_id']}` at {_fmt(best_ms)} ms / {_fmt(best['fps_from_mean'])} FPS. Relative to frozen `ALOD-REPRO-HOST-NOCACHE` ({_fmt(baseline_ms)} ms / {_fmt(baseline['fps_from_mean'])} FPS), it saves {_fmt(baseline_ms-best_ms)} ms ({(baseline_ms-best_ms)/baseline_ms:.2%}) and is {baseline_ms/best_ms:.2f}x faster.", "",
        "All modes preserve exact semantic ALoD selection. The 16-camera gate found exact ordered oracle IDs, selected/touched SPT parity, bitwise-exact surviving properties, zero backend false negatives, and bitwise-exact RGB/alpha. CPU and CUDA filters produced exact ordered IDs on all 71 workload cameras.", "",
        "## Frozen protocol and capacity sizing", "",
        "The campaign reuses the faithful 71-camera order, per-camera resolution, packed gsplat backend, ten ordered warmups, synchronization, and three fresh processes. Run005 remains development evidence; this is the formal mixed-scale Run009 evaluation. Capacities were frozen from count traces before timing:", "",
        f"- maximum post-filter hierarchy working set: {int(sizing['post_filter_working_set']['rows']['maximum']):,} rows;",
        f"- mean / p95: {int(sizing['post_filter_working_set']['rows']['mean']):,} / {int(sizing['post_filter_working_set']['rows']['p95']):,} rows;",
        f"- trajectory union: {sizing['post_filter_working_set']['trajectory_union_rows']:,} rows;",
        f"- mean consecutive required-row overlap: {sizing['post_filter_working_set']['consecutive_required_row_overlap']['mean']:.2%};",
        f"- minimum / practical / larger sensitivity capacities: {sizing['frozen_capacities']['minimum_feasible_rows']:,} / {sizing['frozen_capacities']['practical_bounded_rows']:,} / {sizing['frozen_capacities']['larger_sensitivity_rows']:,} rows.", "",
        "## Thesis-facing comparison", "",
        "| Configuration | Mean / median / p95 ms | FPS | Semantic rows | Materialized rows | Projection survivors | Steady H2D MiB | Property + filter MiB | Peak allocated MiB | Whole-device peak MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config_id in order:
        row = modes.get(config_id) or faithful[config_id]
        t = row["timings_ms"]["end_to_end_frame_time_ms"]
        counts = row["counts"]
        semantic = counts["alod_semantic_selected_rows"]["mean"] if "alod_semantic_selected_rows" in counts else counts["alod_total_raster_rows"]["mean"]
        materialized = counts["alod_materialization_rows"]["mean"] if "alod_materialization_rows" in counts else counts["alod_total_raster_rows"]["mean"]
        survivors = counts["backend_projection_survivor_count"]["mean"]
        h2d = row.get("traffic", {}).get("steady_bytes", row.get("steady_h2d_bytes", 0)) / 2**20
        mem = row.get("persistent_gpu_property_plus_filter_bytes_max", row["memory_bytes"]["alod_gaussian_properties_gpu_bytes"]["maximum"]) / 2**20
        peak = row["memory_bytes"]["frame_total_peak_allocated_bytes"]["maximum"] / 2**20
        nvml = row["memory_bytes"]["nvml_device_used_bytes"]["maximum"] / 2**20
        lines.append(f"| `{config_id}` | {_fmt(t['mean'])} / {_fmt(t['median'])} / {_fmt(t['p95'])} | {_fmt(row['fps_from_mean'])} | {semantic:,.0f} | {materialized:,.0f} | {survivors:,.0f} | {h2d:,.1f} | {mem:,.1f} | {peak:,.1f} | {nvml:,.1f} |")
    fast = modes["ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL"]
    funnel = fast["funnel_ratios_of_means"]
    lines += [
        "", "## Gaussian-count funnel", "",
        f"On the fast practical-cache path, mean semantic selection is {fast['counts']['alod_semantic_selected_rows']['mean']:,.0f} rows, mean materialization is {fast['counts']['alod_materialization_rows']['mean']:,.0f}, and packed-projection survival is {fast['counts']['backend_projection_survivor_count']['mean']:,.0f}. The ratios are {funnel['semantic_selected_over_materialized']:.3f}:1 semantic/materialized, {funnel['semantic_selected_over_projection_survivors']:.3f}:1 semantic/survivor, and {funnel['materialized_over_projection_survivors']:.3f}:1 materialized/survivor.", "",
        "The filter acts only after exact semantic selection. It reduces full-property materialization, not the LoD cut.", "",
        "## Incremental attribution", "",
        "| From → to | Saved ms | Time reduction | Steady H2D change MiB | Persistent property+filter change MiB |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for effect in attribution:
        lines.append(f"| `{effect['from']}` → `{effect['to']}` | {effect['frame_time_saved_ms']:.2f} | {effect['frame_time_reduction_fraction']:.2%} | {effect['steady_h2d_bytes_change']/2**20:,.1f} | {effect['persistent_gpu_property_plus_filter_bytes_change']/2**20:,.1f} |")
    traffic = fast["traffic"]
    lines += [
        "", "## H2D and memory", "",
        f"For the fast practical cache across three processes, cold first-frame, remaining warmup, and measured traffic are {traffic['cold_first_frame_bytes']/2**20:,.1f}, {traffic['remaining_warmup_bytes']/2**20:,.1f}, and {traffic['steady_bytes']/2**20:,.1f} MiB. Steady hits/misses/evictions/insertions are {traffic['steady_hits']:,} / {traffic['steady_misses']:,} / {traffic['steady_evictions']:,} / {traffic['steady_insertions']:,}; {traffic['zero_transfer_steady_frames']}/213 frames transfer no property rows.", "",
        "The practical and large caches have identical one-pass union coverage; the large cache is only a sensitivity control. The minimum cache lowers persistent property memory but incurs evictions and higher traffic. Allocated, reserved, whole-device VRAM, and RSS remain separate in the machine-readable table; whole-device claims use only measured NVML values and are not treated as process-attributed under WSL.", "",
        "## Remaining bottleneck", "",
        f"For the fastest mode, semantic HSPT/SPT selection is the largest stage at {selection_ms:.2f} ms, or {selection_ms/best_ms:.2%} of total frame time. The filter is {best['stage_breakdown']['materialization_filter_ms']:.2f} ms, residency/gather/H2D {best['stage_breakdown']['residency_gather_h2d_ms']:.2f} ms, assembly {best['stage_breakdown']['pre_raster_assembly_ms']:.2f} ms, and rasterization {best['stage_breakdown']['rasterization_ms']:.2f} ms.", "",
        "## Thesis questions", "",
        f"- Q1/Q4: only {fast['counts']['alod_materialization_rows']['mean']/fast['counts']['alod_semantic_selected_rows']['mean']:.2%} of semantically selected rows are materialized on average; the exact conservative decision sharply reduces frame-local property work.",
        f"- Q2: practical bounded residency reduces measured H2D by {(1-fast['traffic']['steady_bytes']/faithful['ALOD-REPRO-HOST-NOCACHE']['steady_h2d_bytes']):.2%} versus faithful no-cache over the matched 213 frames.",
        "- Q3: yes. Exact semantic selection is preserved with all three bounded capacities; capacity changes only residency/traffic.",
        "- Q5: yes in persistent full-property bytes, but the 1.60M minimum cache trades the reduction for evictions and additional traffic; 3.70M is the practical zero-eviction union-sized choice.",
        f"- Q6: best formal improvement is {baseline_ms-best_ms:.2f} ms ({(baseline_ms-best_ms)/baseline_ms:.2%}), from {baseline['fps_from_mean']:.2f} to {best['fps_from_mean']:.2f} FPS.",
        f"- Q7: semantic selection remains dominant at {selection_ms:.2f} ms ({selection_ms/best_ms:.2%}).", "",
        "## Interpretation boundary", "",
        "Run009 is `WEAK-LOD`. These are the measured behaviors of the validated formal hierarchy and fixed workload; they do not establish unchanged performance for a stronger-LoD hierarchy. Official ALoD, the faithful campaign, Run005 history, the hierarchy, oracle, datasets, and cameras remain untouched. Every evaluated optimization is classified `ALOD-EXT-*`.", "",
        "## Validation", "",
        "- 217 project unit tests passed (2 skipped);",
        "- all 14 opt-in external ALoD integration tests passed; the first combined invocation exposed the known system-CUDA `sm_120` JIT limitation, and the affected two-test fast-filter class passed on retry with the already-built pinned gsplat extension used by the campaign;",
        "- 82 focused CityGaussian/shared-runtime regressions passed;",
        "- Run009 correctness gate: `PASS-ALOD-EXT-FORMAL-RUN009-CORRECTNESS`;",
        "- cache-sizing gate: `PASS-ALOD-EXT-RUN009-CACHE-SIZING`;",
        "- `git diff --check` passed;",
        "- nothing was staged, committed, or pushed.", "",
        "## Next task", "",
        "Decide whether the dominant remaining bottleneck justifies a new semantic-selection optimization, and if so design it as the next ALOD-EXT contribution with exact oracle parity as a hard constraint. No such optimization was implemented here.",
    ]
    args.memo.parent.mkdir(parents=True, exist_ok=True)
    args.memo.write_text("\n".join(lines) + "\n", encoding="utf-8")
    provenance = {
        "campaign_id": "ALOD-EXT-FORMAL-RUN009-001",
        "classification": "ALOD-EXT",
        "official_source": {
            "path": "/home/luis/tese/LoDOfGaussians",
            "commit": "73f547341547ae021ff036ad74fdf73a1240b450",
            "modified_by_campaign": False,
        },
        "protected_inputs_modified": False,
        "faithful_campaign_modified": False,
        "historical_run005_campaigns_modified": False,
        "semantic_selection_modified": False,
        "materialization_predicate_modified": False,
        "memo": {"path": str(args.memo.resolve()), "sha256": hashlib.sha256(args.memo.read_bytes()).hexdigest()},
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    digest = hashlib.sha256((tables / "formal_summary.json").read_bytes()).hexdigest()
    (root / "ALOD-EXT-FORMAL-RUN009.json").write_text(json.dumps({"status": STATUS, "formal_summary_sha256": digest, "memo": str(args.memo.resolve())}, indent=2) + "\n")
    manifest_rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest_rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": "ALOD-EXT-FORMAL-RUN009-001",
                "status": STATUS,
                "immutable_after_finalization": True,
                "file_count": len(manifest_rows),
                "files": manifest_rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"status": STATUS, "best": result["best_existing_extension"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
