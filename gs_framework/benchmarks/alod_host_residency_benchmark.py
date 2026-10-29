#!/usr/bin/env python3
"""Validate and aggregate the formal four-mode ALoD residency campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.benchmark_real_scene import (  # noqa: E402
    CONFIG_FILES,
    canonical_json_sha256,
    environment_metadata,
    load_config,
    sha256_file,
    write_json,
)


CAMPAIGN_ID = "ALOD-HOST-RESIDENCY-BENCHMARK-001"
PASS_STATUS = "PASS-ALOD-HOST-RESIDENCY-BENCHMARK"
CONFIG_IDS = (
    "ALOD-HIER-LOD-FRUSTUM",
    "ALOD-HOST-NOCACHE",
    "ALOD-HOST-PREVIOUS-REUSE",
    "ALOD-HOST-BOUNDED-CACHE",
)
MODE_LABELS = {
    CONFIG_IDS[0]: "Resident",
    CONFIG_IDS[1]: "Host no-cache",
    CONFIG_IDS[2]: "Previous reuse",
    CONFIG_IDS[3]: "Bounded cache",
}
IMPLEMENTATION_FILES = (
    REPO_ROOT / "scene_runtime" / "alod" / "host.py",
    REPO_ROOT / "scene_runtime" / "alod" / "residency.py",
    REPO_ROOT / "scene_runtime" / "alod" / "host_scene.py",
    REPO_ROOT / "scene_runtime" / "alod" / "selection.py",
    REPO_ROOT / "scene_runtime" / "alod" / "hspt.py",
    REPO_ROOT / "benchmarks" / "benchmark_real_scene.py",
    REPO_ROOT / "scripts" / "alod" / "run_host_residency_benchmark_campaign.py",
)
EXPECTED_ASSET_SHA256 = "6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c"
EXPECTED_TRAJECTORY_SHA256 = "c6d607b8baee0cc30f3c74121e9d19ba045a820fc87c8550a11c3dbc0c92968d"
EXPECTED_CACHE_CAPACITY = 525_000
EXPECTED_MAX_SELECTED = 499_030
EXPECTED_SKYBOX = 100_000


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _finite(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    result = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result.append(float(value))
    return result


def _stats(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            key: None
            for key in ("mean", "median", "p10", "p90", "p95", "std", "min", "max")
        }
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _sum(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = _finite(rows, key)
    return float(sum(values)) if values else None


def _maximum(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = _finite(rows, key)
    return max(values) if values else None


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _percent_change(value: float, baseline: float) -> float | None:
    return 100.0 * (value - baseline) / baseline if baseline else None


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        return sha256_file(path)
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def _git_state(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=path, check=True, capture_output=True, text=True
        ).stdout

    return {
        "path": str(path.resolve()),
        "commit": run("rev-parse", "HEAD").strip(),
        "status_porcelain": run("status", "--porcelain=v1", "--untracked-files=all"),
        "diff_sha256": hashlib.sha256(run("diff", "--binary").encode("utf-8")).hexdigest(),
        "cached_diff_sha256": hashlib.sha256(
            run("diff", "--cached", "--binary").encode("utf-8")
        ).hexdigest(),
    }


def run_snapshot(output: Path, paths: list[Path], git_repositories: list[Path]) -> int:
    records = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        records.append(
            {
                "path": str(resolved),
                "kind": "file" if resolved.is_file() else "directory",
                "sha256": _tree_sha256(resolved),
            }
        )
    result = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protected_paths": records,
        "git_repositories": [_git_state(path.resolve()) for path in git_repositories],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return 0


def run_compare_snapshots(before_path: Path, after_path: Path, output: Path) -> int:
    before = _read_json(before_path)
    after = _read_json(after_path)
    before_paths = {row["path"]: row for row in before["protected_paths"]}
    after_paths = {row["path"]: row for row in after["protected_paths"]}
    before_git = {row["path"]: row for row in before["git_repositories"]}
    after_git = {row["path"]: row for row in after["git_repositories"]}
    path_changes = {
        path: {"before": before_paths.get(path), "after": after_paths.get(path)}
        for path in sorted(set(before_paths) | set(after_paths))
        if before_paths.get(path) != after_paths.get(path)
    }
    git_changes = {
        path: {"before": before_git.get(path), "after": after_git.get(path)}
        for path in sorted(set(before_git) | set(after_git))
        if before_git.get(path) != after_git.get(path)
    }
    unchanged = not path_changes and not git_changes
    result = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "PASS-PROTECTED-STATE-UNCHANGED" if unchanged else "FAILED-PROTECTED-STATE-CHANGED",
        "protected_path_count": len(before_paths),
        "protected_git_repository_count": len(before_git),
        "path_changes": path_changes,
        "git_changes": git_changes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    if not unchanged:
        raise RuntimeError("protected state changed during the benchmark campaign")
    return 0


def run_finalize(
    campaign_root: Path,
    output: Path,
    *,
    unit_tests: int,
    external_tests: int,
    citygaussian_tests: int,
) -> int:
    root = campaign_root.resolve()
    summary = _read_json(root / "tables" / "config_summary.json")
    package = _read_json(root / "checks" / "package_validation.json")
    correctness = _read_json(root / "checks" / "preflight_correctness.json")
    protected = _read_json(root / "checks" / "protected_state_comparison.json")
    schedule = _read_json(root / "checks" / "formal_run_commands.json")["schedule"]
    if summary.get("status") != PASS_STATUS or package.get("status") != PASS_STATUS:
        raise RuntimeError("campaign aggregation/package validation is not passing")
    if correctness.get("status") != "PASS-ALOD-HOST-RESIDENCY":
        raise RuntimeError("correctness prerequisite is not passing")
    if protected.get("status") != "PASS-PROTECTED-STATE-UNCHANGED":
        raise RuntimeError("protected state is not unchanged")
    if len(schedule) != 12 or any(row.get("status") != "complete" for row in schedule):
        raise RuntimeError("formal process schedule is incomplete")
    diff_check = subprocess.run(
        ["git", "diff", "--check"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if diff_check.returncode != 0:
        raise RuntimeError(f"git diff --check failed:\n{diff_check.stdout}{diff_check.stderr}")
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout
    result = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": PASS_STATUS,
        "formal_processes": 12,
        "accepted_measured_frames": 1200,
        "correctness_status": correctness["status"],
        "protected_state_status": protected["status"],
        "quality_exact": package["quality_exact"],
        "history_independent": package["history_independent"],
        "unit_tests_passed": unit_tests,
        "external_alod_integration_tests_passed": external_tests,
        "citygaussian_regression_tests_passed": citygaussian_tests,
        "git_diff_check": "PASS",
        "git_status_short": git_status,
        "excluded_attempts": [
            "rejected/diagnostic_validation_enabled",
            "rejected/missing_cold_raster_event",
        ],
        "staged_committed_or_pushed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return 0


def _validate_run(
    run_dir: Path,
    config_id: str,
    expected_config_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    frames = _read_jsonl(run_dir / "frames.jsonl")
    summary = _read_json(run_dir / "summary.json")
    cold = _read_json(run_dir / "cold_start.json")
    resolved = _read_json(run_dir / "resolved_config.json")
    if len(frames) != 100:
        raise RuntimeError(f"{run_dir}: expected 100 measured frames, got {len(frames)}")
    if summary.get("record_kind") != "formal_benchmark":
        raise RuntimeError(f"{run_dir}: record_kind is not formal_benchmark")
    if summary.get("configuration_id") != config_id:
        raise RuntimeError(f"{run_dir}: configuration ID mismatch")
    if resolved.get("warmup_frames") != 10 or resolved.get("num_repeat") != 10:
        raise RuntimeError(f"{run_dir}: warmup/repeat protocol mismatch")
    if resolved.get("max_frames") != 100:
        raise RuntimeError(f"{run_dir}: measured-frame cap mismatch")
    provenance = summary.get("provenance", {})
    if provenance.get("configuration_sha256") != expected_config_hash:
        raise RuntimeError(f"{run_dir}: resolved configuration hash mismatch")
    if provenance.get("asset_sha256") != EXPECTED_ASSET_SHA256:
        raise RuntimeError(f"{run_dir}: asset hash mismatch")
    if provenance.get("trajectory_file_sha256") != EXPECTED_TRAJECTORY_SHA256:
        raise RuntimeError(f"{run_dir}: trajectory hash mismatch")
    for index, frame in enumerate(frames):
        expected_repeat, expected_camera = divmod(index, 10)
        if (
            frame.get("frame_index") != index
            or frame.get("repeat_index") != expected_repeat
            or frame.get("camera_index") != expected_camera
            or frame.get("width") != 960
            or frame.get("height") != 540
        ):
            raise RuntimeError(f"{run_dir}: frame-order/protocol mismatch at row {index}")
        if not frame.get("finite_image"):
            raise RuntimeError(f"{run_dir}: non-finite image at row {index}")
    if cold.get("first_frame") is None or cold.get("first_frame", {}).get("camera_index") != 0:
        raise RuntimeError(f"{run_dir}: missing cold first-frame characterization")
    cold_raster = cold["first_frame"].get("rasterization_time_ms")
    if not isinstance(cold_raster, (int, float)) or not math.isfinite(float(cold_raster)):
        raise RuntimeError(f"{run_dir}: missing finite cold first-frame raster timing")
    if config_id == CONFIG_IDS[1] and any(
        frame.get("alod_cache_hit_count") is not None
        or frame.get("alod_cache_occupancy_rows") is not None
        or frame.get("alod_cache_management_ms") is not None
        for frame in frames
    ):
        raise RuntimeError(f"{run_dir}: no-cache metrics incorrectly use cache schema")
    if config_id == CONFIG_IDS[2] and any(
        frame.get("alod_previous_working_set_rows") is None
        or frame.get("alod_previous_intersection_rows") is None
        for frame in frames
    ):
        raise RuntimeError(f"{run_dir}: previous-working-set metrics are missing")
    if config_id == CONFIG_IDS[3] and any(
        frame.get("alod_dynamic_cache_capacity_rows") != EXPECTED_CACHE_CAPACITY
        for frame in frames
    ):
        raise RuntimeError(f"{run_dir}: bounded-cache capacity differs from 525k")
    return frames, summary, cold


def _stage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "camera_preparation_ms": "alod_camera_preparation_ms",
        "upper_traversal_ms": "alod_upper_traversal_ms",
        "spt_cut_ms": "alod_spt_cut_ms",
        "total_selection_ms": "alod_total_selection_ms",
        "residency_lookup_ms": "alod_residency_lookup_ms",
        "host_gather_ms": "alod_host_gather_ms",
        "cache_management_ms": "alod_cache_management_ms",
        "h2d_transfer_ms": "alod_h2d_transfer_ms",
        "gpu_slot_index_ms": "alod_gpu_slot_index_ms",
        "selected_gpu_gather_ms": "alod_selected_gpu_gather_ms",
        "pre_raster_assembly_ms": "alod_pre_raster_assembly_ms",
        "preparation_ms": "preparation_time_ms",
        "rasterization_ms": "rasterization_time_ms",
        "total_frame_time_ms": "end_to_end_frame_time_ms",
    }
    return {output: _stats(_finite(rows, source)) for output, source in fields.items()}


def _residency_summary(config_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    requests = _sum(rows, "alod_requested_rows") or 0.0
    hits = _sum(rows, "alod_cache_hit_count")
    misses = _sum(rows, "alod_cache_miss_count")
    evictions = _sum(rows, "alod_cache_eviction_count")
    h2d = _finite(rows, "alod_h2d_bytes")
    cache_applicable = config_id in {CONFIG_IDS[2], CONFIG_IDS[3]}
    return {
        "requested_rows": _stats(_finite(rows, "alod_requested_rows")),
        "rows_loaded": _stats(_finite(rows, "alod_rows_loaded")),
        "retained_rows": _stats(_finite(rows, "alod_retained_rows")),
        "evicted_rows": _stats(_finite(rows, "alod_cache_eviction_count")),
        "cache_occupancy_rows": _stats(_finite(rows, "alod_cache_occupancy_rows")),
        "cache_capacity_rows": (
            EXPECTED_CACHE_CAPACITY if config_id == CONFIG_IDS[3] else None
        ),
        "cache_occupancy_ratio": _stats(_finite(rows, "alod_cache_occupancy_ratio")),
        "overall_cache_or_reuse_hit_rate": (
            _ratio(float(hits or 0.0), requests) if cache_applicable else None
        ),
        "overall_miss_rate": (
            _ratio(float(misses or 0.0), requests) if cache_applicable else None
        ),
        "overall_eviction_rate_per_request": (
            _ratio(float(evictions or 0.0), requests) if cache_applicable else None
        ),
        "total_hits": hits if cache_applicable else None,
        "total_misses": misses if cache_applicable else None,
        "total_evictions": evictions if cache_applicable else None,
        "previous_working_set_rows": _stats(
            _finite(rows, "alod_previous_working_set_rows")
        ),
        "previous_intersection_rows": _stats(
            _finite(rows, "alod_previous_intersection_rows")
        ),
        "retention_ratio": _stats(_finite(rows, "alod_retention_ratio")),
        "discarded_rows": _stats(_finite(rows, "alod_discarded_rows")),
        "h2d": {
            "total_bytes": float(sum(h2d)),
            "bytes_per_frame": _stats(h2d),
            "zero_transfer_frames": int(sum(value == 0.0 for value in h2d)),
        },
    }


def _memory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "host_property_store_bytes": rows[0].get("alod_host_property_store_bytes"),
        "host_pinned_bytes": rows[0].get("alod_host_pinned_bytes"),
        "hierarchy_metadata_cpu_bytes": rows[0].get("alod_hierarchy_metadata_cpu_bytes"),
        "residency_metadata_cpu_bytes": _stats(
            _finite(rows, "alod_residency_metadata_cpu_bytes")
        ),
        "persistent_skybox_gpu_bytes": rows[0].get("alod_persistent_skybox_gpu_bytes"),
        "persistent_dynamic_property_bytes": _stats(
            _finite(rows, "alod_dynamic_cache_gpu_bytes")
        ),
        "persistent_gaussian_property_bytes": _stats(
            _finite(rows, "alod_gaussian_properties_gpu_bytes")
        ),
        "hierarchy_metadata_gpu_bytes": rows[0].get("alod_hierarchy_metadata_gpu_bytes"),
        "frame_local_property_bytes": _stats(
            _finite(rows, "alod_frame_local_property_bytes")
        ),
        "frame_local_index_bytes": _stats(_finite(rows, "alod_frame_local_index_bytes")),
        "steady_torch_allocated_bytes": _stats(
            _finite(rows, "torch_current_allocated_bytes")
        ),
        "peak_frame_total_torch_allocated_bytes": _maximum(
            rows, "frame_total_peak_allocated_bytes"
        ),
        "peak_frame_total_torch_reserved_bytes": _maximum(
            rows, "frame_total_peak_reserved_bytes"
        ),
        "peak_whole_device_vram_bytes": _maximum(rows, "nvml_device_used_bytes"),
        "peak_process_rss_bytes": _maximum(rows, "process_peak_rss_bytes"),
        "process_rss_bytes": _stats(_finite(rows, "process_ram_bytes")),
    }


def _cold_summary(cold_records: list[dict[str, Any]]) -> dict[str, Any]:
    first = [record["first_frame"] for record in cold_records]
    return {
        "definition": cold_records[0]["cold_definition"],
        "process_replicates": len(cold_records),
        "host_store_initialization_ms": _stats(
            record["host_store_initialization_ms"]
            for record in cold_records
            if record.get("host_store_initialization_ms") is not None
        ),
        "gpu_cache_allocation_ms": _stats(
            record["gpu_cache_allocation_ms"]
            for record in cold_records
            if record.get("gpu_cache_allocation_ms") is not None
        ),
        "skybox_transfer_ms": _stats(
            record["skybox_transfer_ms"]
            for record in cold_records
            if record.get("skybox_transfer_ms") is not None
        ),
        "scene_load_total_ms": _stats(record["scene_load_total_ms"] for record in cold_records),
        "cold_initialization_total_ms": _stats(
            record["cold_initialization_total_ms"] for record in cold_records
        ),
        "first_frame_total_ms": _stats(_finite(first, "end_to_end_frame_time_ms")),
        "first_frame_h2d_bytes": _stats(_finite(first, "alod_h2d_bytes")),
        "first_frame_h2d_ms": _stats(_finite(first, "alod_h2d_transfer_ms")),
        "first_frame_raster_ms": _stats(_finite(first, "rasterization_time_ms")),
    }


def _validate_cross_mode_counts(pooled: dict[str, list[dict[str, Any]]]) -> None:
    fields = (
        "alod_selected_hierarchy_rows",
        "alod_total_raster_rows",
        "active_gaussians",
        "visible_gaussians",
    )
    reference = pooled[CONFIG_IDS[0]]
    for config_id in CONFIG_IDS[1:]:
        rows = pooled[config_id]
        for index, (left, right) in enumerate(zip(reference, rows)):
            for field in fields:
                if left.get(field) != right.get(field):
                    raise RuntimeError(
                        f"cross-mode {field} mismatch for {config_id} at pooled row {index}"
                    )


def _quality_summary(validation: dict[str, Any]) -> dict[str, Any]:
    camera_records = []
    history_records = []
    for mode, content in validation["modes"].items():
        for camera in content["cameras"]:
            if camera.get("render") is not None:
                camera_records.append({"mode": mode, **camera["render"]})
        for sequence, rows in content["sequences"].items():
            for row in rows:
                if row.get("history_render") is not None:
                    history_records.append(
                        {"mode": mode, "sequence": sequence, **row["history_render"]}
                    )
    if not camera_records or not all(row["exact"] for row in camera_records):
        raise RuntimeError("resident/host quality parity was not bitwise exact")
    if not history_records or not all(
        row["exact"] and row["visible_count_exact"] for row in history_records
    ):
        raise RuntimeError("history-independence render parity was not exact")
    return {
        "criterion": "bitwise RGB equality and exact backend-visible count",
        "representative_render_count": len(camera_records),
        "representative_all_exact": True,
        "representative_psnr_db": "infinite",
        "representative_ssim": 1.0,
        "representative_max_abs_error": max(row["max_abs_error"] for row in camera_records),
        "history_render_count": len(history_records),
        "history_all_exact": True,
        "history_visible_counts_all_exact": True,
        "history_psnr_db": "infinite",
        "history_ssim": 1.0,
        "history_max_abs_error": max(row["max_abs_error"] for row in history_records),
    }


def _write_tables(output_dir: Path, configs: list[dict[str, Any]]) -> None:
    csv_path = output_dir / "config_summary.csv"
    fields = (
        "configuration_id", "mean_frame_ms", "median_frame_ms", "p95_frame_ms", "fps",
        "selection_ms", "residency_lookup_ms", "cache_management_ms", "host_gather_ms",
        "h2d_ms", "assembly_ms", "raster_ms", "selected_rows", "visible_rows",
        "persistent_property_bytes", "peak_vram_bytes", "peak_rss_bytes", "total_h2d_bytes",
        "mean_h2d_bytes", "hit_rate", "evictions",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in configs:
            timing, residency, memory = row["timing"], row["residency"], row["memory"]
            writer.writerow({
                "configuration_id": row["configuration_id"],
                "mean_frame_ms": timing["total_frame_time_ms"]["mean"],
                "median_frame_ms": timing["total_frame_time_ms"]["median"],
                "p95_frame_ms": timing["total_frame_time_ms"]["p95"],
                "fps": row["fps_from_mean_frame_time"],
                "selection_ms": timing["total_selection_ms"]["mean"],
                "residency_lookup_ms": timing["residency_lookup_ms"]["mean"],
                "cache_management_ms": timing["cache_management_ms"]["mean"],
                "host_gather_ms": timing["host_gather_ms"]["mean"],
                "h2d_ms": timing["h2d_transfer_ms"]["mean"],
                "assembly_ms": timing["pre_raster_assembly_ms"]["mean"],
                "raster_ms": timing["rasterization_ms"]["mean"],
                "selected_rows": row["counts"]["selected_hierarchy_rows"]["mean"],
                "visible_rows": row["counts"]["backend_visible_rows"]["mean"],
                "persistent_property_bytes": memory["persistent_gaussian_property_bytes"]["mean"],
                "peak_vram_bytes": memory["peak_whole_device_vram_bytes"],
                "peak_rss_bytes": memory["peak_process_rss_bytes"],
                "total_h2d_bytes": residency["h2d"]["total_bytes"],
                "mean_h2d_bytes": residency["h2d"]["bytes_per_frame"]["mean"],
                "hit_rate": residency["overall_cache_or_reuse_hit_rate"],
                "evictions": residency["total_evictions"],
            })

    def f(value: Any, digits: int = 3) -> str:
        return "N/A" if value is None else f"{float(value):.{digits}f}"

    lines = [
        "# ALoD Host Residency Benchmark Tables", "",
        "| Configuration | Mean ms | Median ms | P95 ms | FPS | Selection ms | Residency ms | Cache mgmt ms | Gather ms | H2D ms | Assembly ms | Raster ms | Selected rows | Visible rows | Persistent properties MiB | Peak VRAM MiB | Peak RSS MiB | Total H2D GiB | H2D MiB/frame | Hit rate | Evictions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in configs:
        timing, residency, memory = row["timing"], row["residency"], row["memory"]
        lines.append(
            "| {name} | {mean} | {median} | {p95} | {fps} | {selection} | {lookup} | "
            "{management} | {gather} | {h2d_ms} | {assembly} | {raster} | {selected:,.0f} | "
            "{visible:,.0f} | {properties} | {vram} | {rss} | {h2d_total} | {h2d_frame} | {hit} | {evictions} |".format(
                name=row["configuration_id"], mean=f(timing["total_frame_time_ms"]["mean"]),
                median=f(timing["total_frame_time_ms"]["median"]), p95=f(timing["total_frame_time_ms"]["p95"]),
                fps=f(row["fps_from_mean_frame_time"], 2), selection=f(timing["total_selection_ms"]["mean"]),
                lookup=f(timing["residency_lookup_ms"]["mean"]), management=f(timing["cache_management_ms"]["mean"]),
                gather=f(timing["host_gather_ms"]["mean"]), h2d_ms=f(timing["h2d_transfer_ms"]["mean"]),
                assembly=f(timing["pre_raster_assembly_ms"]["mean"]), raster=f(timing["rasterization_ms"]["mean"]),
                selected=row["counts"]["selected_hierarchy_rows"]["mean"], visible=row["counts"]["backend_visible_rows"]["mean"],
                properties=f(memory["persistent_gaussian_property_bytes"]["mean"] / 2**20, 2),
                vram=f(memory["peak_whole_device_vram_bytes"] / 2**20 if memory["peak_whole_device_vram_bytes"] else None, 2),
                rss=f(memory["peak_process_rss_bytes"] / 2**20 if memory["peak_process_rss_bytes"] else None, 2),
                h2d_total=f(residency["h2d"]["total_bytes"] / 2**30, 3),
                h2d_frame=f(
                    residency["h2d"]["bytes_per_frame"]["mean"] / 2**20
                    if residency["h2d"]["bytes_per_frame"]["mean"] is not None
                    else None,
                    3,
                ),
                hit=f(residency["overall_cache_or_reuse_hit_rate"], 4),
                evictions="N/A" if residency["total_evictions"] is None else f"{residency['total_evictions']:,.0f}",
            )
        )
    (output_dir / "thesis_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figures(campaign_root: Path, configs: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    figures = campaign_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = [MODE_LABELS[row["configuration_id"]] for row in configs]
    colors = ["#4C78A8", "#E45756", "#F2CF5B", "#54A24B"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    means = [row["timing"]["total_frame_time_ms"]["mean"] for row in configs]
    p95 = [row["timing"]["total_frame_time_ms"]["p95"] for row in configs]
    ax.bar(labels, means, color=colors)
    ax.scatter(range(4), p95, color="black", label="P95", zorder=3)
    ax.set_ylabel("End-to-end frame time (ms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "frame_time_by_residency_mode.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, [row["memory"]["persistent_gaussian_property_bytes"]["mean"] / 2**20 for row in configs], color=colors)
    ax.set_ylabel("Persistent Gaussian properties (MiB)")
    fig.tight_layout()
    fig.savefig(figures / "persistent_property_memory.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, [row["residency"]["h2d"]["total_bytes"] / 2**30 for row in configs], color=colors)
    ax.set_ylabel("Measured H2D traffic (GiB, 300 frames)")
    fig.tight_layout()
    fig.savefig(figures / "total_h2d_traffic.png", dpi=160)
    plt.close(fig)

    stages = (
        ("total_selection_ms", "Selection"), ("residency_lookup_ms", "Residency lookup"),
        ("cache_management_ms", "Cache management"), ("host_gather_ms", "Host gather"),
        ("h2d_transfer_ms", "H2D"), ("selected_gpu_gather_ms", "GPU gather"),
        ("pre_raster_assembly_ms", "Assembly"), ("rasterization_ms", "Raster"),
    )
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bottom = np.zeros(4)
    for key, label in stages:
        values = np.asarray([row["timing"][key]["mean"] or 0.0 for row in configs])
        ax.bar(labels, values, bottom=bottom, label=label)
        bottom += values
    ax.set_ylabel("Mean measured stage time (ms)")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "stage_breakdown.png", dpi=160)
    plt.close(fig)


def run_aggregate(campaign_root: Path, asset: Path, oracle_manifest: Path) -> int:
    root = campaign_root.resolve()
    if sha256_file(asset) != EXPECTED_ASSET_SHA256:
        raise RuntimeError("formal hierarchy hash differs from the frozen Run005 asset")
    validation = _read_json(root / "checks" / "preflight_correctness.json")
    if validation.get("status") != "PASS-ALOD-HOST-RESIDENCY":
        raise RuntimeError("host residency correctness prerequisite did not pass")
    quality = _quality_summary(validation)

    pooled: dict[str, list[dict[str, Any]]] = {}
    config_rows = []
    run_provenance = []
    for config_id in CONFIG_IDS:
        config = load_config(config_id)
        expected_config_hash = canonical_json_sha256({**config, "warmup_frames": 10, "num_repeat": 10})
        rows: list[dict[str, Any]] = []
        cold_records = []
        replicate_means = []
        for replicate in range(1, 4):
            run_dir = root / "runs" / config_id / f"replicate-{replicate:02d}"
            run_rows, summary, cold = _validate_run(run_dir, config_id, expected_config_hash)
            rows.extend(run_rows)
            cold_records.append(cold)
            replicate_means.append(float(np.mean(_finite(run_rows, "end_to_end_frame_time_ms"))))
            run_provenance.append({
                "configuration_id": config_id, "replicate": replicate,
                "summary_sha256": sha256_file(run_dir / "summary.json"),
                "frames_sha256": sha256_file(run_dir / "frames.jsonl"),
                "resolved_config_sha256": sha256_file(run_dir / "resolved_config.json"),
                "cold_start_sha256": sha256_file(run_dir / "cold_start.json"),
                "environment": summary.get("environment"),
            })
        pooled[config_id] = rows
        timing = _stage_summary(rows)
        config_rows.append({
            "configuration_id": config_id,
            "label": MODE_LABELS[config_id],
            "frame_count": len(rows),
            "process_replicates": 3,
            "trajectory_repetitions_per_process": 10,
            "replicate_mean_frame_time_ms": replicate_means,
            "replicate_mean_frame_time_std_ms": float(statistics.pstdev(replicate_means)),
            "fps_from_mean_frame_time": 1000.0 / float(timing["total_frame_time_ms"]["mean"]),
            "timing": timing,
            "counts": {
                "selected_hierarchy_rows": _stats(_finite(rows, "alod_selected_hierarchy_rows")),
                "raster_rows": _stats(_finite(rows, "alod_total_raster_rows")),
                "backend_visible_rows": _stats(_finite(rows, "visible_gaussians")),
            },
            "residency": _residency_summary(config_id, rows),
            "memory": _memory_summary(rows),
            "cold_start": _cold_summary(cold_records),
        })

    _validate_cross_mode_counts(pooled)
    maximum_selected = max(_finite(pooled[CONFIG_IDS[0]], "alod_selected_hierarchy_rows"))
    if maximum_selected != EXPECTED_MAX_SELECTED:
        raise RuntimeError(f"selected maximum changed: {maximum_selected} != {EXPECTED_MAX_SELECTED}")
    bounded_capacity = config_rows[3]["residency"]["cache_capacity_rows"]
    if bounded_capacity != EXPECTED_CACHE_CAPACITY:
        raise RuntimeError("bounded cache capacity changed")

    by_id = {row["configuration_id"]: row for row in config_rows}
    no_cache = by_id[CONFIG_IDS[1]]
    comparisons = []
    resident = by_id[CONFIG_IDS[0]]
    resident_memory = resident["memory"]["persistent_gaussian_property_bytes"]["mean"]
    for config_id in CONFIG_IDS[1:]:
        child = by_id[config_id]
        child_time = child["timing"]["total_frame_time_ms"]["mean"]
        resident_time = resident["timing"]["total_frame_time_ms"]["mean"]
        child_memory = child["memory"]["persistent_gaussian_property_bytes"]["mean"]
        comparisons.append({
            "configuration_id": config_id,
            "frame_time_overhead_vs_resident_ms": child_time - resident_time,
            "frame_time_overhead_vs_resident_percent": _percent_change(child_time, resident_time),
            "fps_change_vs_resident_percent": _percent_change(child["fps_from_mean_frame_time"], resident["fps_from_mean_frame_time"]),
            "persistent_property_bytes_saved_vs_resident": resident_memory - child_memory,
            "persistent_property_memory_reduction_percent": 100.0 * (resident_memory - child_memory) / resident_memory,
            "h2d_reduction_vs_nocache_bytes": no_cache["residency"]["h2d"]["total_bytes"] - child["residency"]["h2d"]["total_bytes"],
            "h2d_reduction_vs_nocache_percent": 100.0 * (no_cache["residency"]["h2d"]["total_bytes"] - child["residency"]["h2d"]["total_bytes"]) / no_cache["residency"]["h2d"]["total_bytes"],
        })

    previous = by_id[CONFIG_IDS[2]]
    bounded = by_id[CONFIG_IDS[3]]
    bounded_vs_previous = {
        "additional_h2d_bytes_eliminated": previous["residency"]["h2d"]["total_bytes"] - bounded["residency"]["h2d"]["total_bytes"],
        "additional_h2d_reduction_percent_of_previous": 100.0 * (previous["residency"]["h2d"]["total_bytes"] - bounded["residency"]["h2d"]["total_bytes"]) / previous["residency"]["h2d"]["total_bytes"],
        "mean_frame_time_change_ms": bounded["timing"]["total_frame_time_ms"]["mean"] - previous["timing"]["total_frame_time_ms"]["mean"],
        "mean_frame_time_change_percent": _percent_change(bounded["timing"]["total_frame_time_ms"]["mean"], previous["timing"]["total_frame_time_ms"]["mean"]),
        "persistent_property_memory_change_bytes": bounded["memory"]["persistent_gaussian_property_bytes"]["mean"] - previous["memory"]["persistent_gaussian_property_bytes"]["mean"],
        "hit_rate_change": bounded["residency"]["overall_cache_or_reuse_hit_rate"] - previous["residency"]["overall_cache_or_reuse_hit_rate"],
        "mean_management_time_change_ms": (bounded["timing"]["residency_lookup_ms"]["mean"] + bounded["timing"]["cache_management_ms"]["mean"]) - (previous["timing"]["residency_lookup_ms"]["mean"] + previous["timing"]["cache_management_ms"]["mean"]),
    }

    component_keys = (
        "total_selection_ms", "residency_lookup_ms", "cache_management_ms", "host_gather_ms",
        "h2d_transfer_ms", "selected_gpu_gather_ms", "pre_raster_assembly_ms", "rasterization_ms",
    )
    bottlenecks = {}
    for row in config_rows:
        measured = {key: row["timing"][key]["mean"] for key in component_keys if row["timing"][key]["mean"] is not None}
        bottlenecks[row["configuration_id"]] = max(measured, key=measured.get)

    result = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": PASS_STATUS,
        "weak_lod_oracle": True,
        "asset": {"path": str(asset.resolve()), "sha256": sha256_file(asset)},
        "oracle": {"path": str(oracle_manifest.resolve()), "sha256": sha256_file(oracle_manifest)},
        "trajectory": {"id": "alod_run005_frozen_evaluation_v1", "sha256": EXPECTED_TRAJECTORY_SHA256, "camera_count": 10, "resolution": [960, 540]},
        "capacity": {"minimum_feasible_rows": EXPECTED_MAX_SELECTED, "chosen_rows": EXPECTED_CACHE_CAPACITY, "headroom_rows": EXPECTED_CACHE_CAPACITY - EXPECTED_MAX_SELECTED, "skybox_rows_separate": EXPECTED_SKYBOX},
        "configurations": config_rows,
        "comparisons_vs_resident": comparisons,
        "bounded_vs_previous": bounded_vs_previous,
        "measured_dominant_stage_by_configuration": bottlenecks,
        "quality_and_history_independence": quality,
        "cross_mode_selected_raster_visible_counts_exact": True,
        "configuration_file_sha256": {config_id: sha256_file(CONFIG_FILES[config_id]) for config_id in CONFIG_IDS},
        "implementation_file_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in IMPLEMENTATION_FILES
        },
        "run_provenance": run_provenance,
        "environment": environment_metadata(),
    }
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_json(tables / "config_summary.json", result)
    write_json(tables / "memory_summary.json", {row["configuration_id"]: row["memory"] for row in config_rows})
    write_json(tables / "transfer_summary.json", {row["configuration_id"]: row["residency"]["h2d"] for row in config_rows})
    write_json(tables / "cache_statistics.json", {row["configuration_id"]: row["residency"] for row in config_rows})
    write_json(tables / "cold_start_summary.json", {row["configuration_id"]: row["cold_start"] for row in config_rows})
    (root / "quality").mkdir(parents=True, exist_ok=True)
    write_json(root / "quality" / "quality_history_summary.json", quality)
    _write_tables(tables, config_rows)
    _write_figures(root, config_rows)

    manifest = {
        "schema_version": 1, "campaign_id": CAMPAIGN_ID, "status": PASS_STATUS,
        "asset": result["asset"], "oracle": result["oracle"], "trajectory": result["trajectory"],
        "capacity": result["capacity"], "configuration_file_sha256": result["configuration_file_sha256"],
        "implementation_file_sha256": result["implementation_file_sha256"],
        "run_policy": {
            "process_replicates": 3,
            "warmup_frames_per_process": 10,
            "trajectory_repetitions_per_process": 10,
            "measured_frames_per_process": 100,
            "measured_frames_per_configuration": 300,
            "total_measured_frames": 1200,
            "rejected_accepted_runs": [],
            "excluded_preliminary_attempts": [
                "rejected/diagnostic_validation_enabled",
                "rejected/missing_cold_raster_event",
            ],
        },
        "run_provenance": run_provenance,
    }
    write_json(root / "manifest.json", manifest)
    write_json(root / "checks" / "package_validation.json", {
        "status": PASS_STATUS, "formal_runs": 12, "frames_per_run": 100,
        "total_measured_frames": 1200, "correctness_gate": validation["status"],
        "quality_exact": True, "history_independent": True,
        "cross_mode_counts_exact": True,
    })
    (root / "README.md").write_text(
        "# ALoD Host Residency Benchmark 001\n\n"
        "Formal resident versus host no-cache, previous-working-set reuse, and "
        "525k-row bounded-cache campaign on the frozen Run005 hierarchy and trajectory.\n\n"
        "See `tables/config_summary.json`, `tables/thesis_tables.md`, "
        "`quality/quality_history_summary.json`, and `checks/package_validation.json`.\n",
        encoding="utf-8",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--campaign-root", type=Path, required=True)
    aggregate.add_argument("--asset", type=Path, required=True)
    aggregate.add_argument("--oracle-manifest", type=Path, required=True)
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--path", action="append", type=Path, default=[])
    snapshot.add_argument("--git-repository", action="append", type=Path, default=[])
    compare = sub.add_parser("compare-snapshots")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--campaign-root", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--unit-tests", type=int, required=True)
    finalize.add_argument("--external-tests", type=int, required=True)
    finalize.add_argument("--citygaussian-tests", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "aggregate":
        return run_aggregate(args.campaign_root, args.asset, args.oracle_manifest)
    if args.command == "snapshot":
        return run_snapshot(args.output, args.path, args.git_repository)
    if args.command == "compare-snapshots":
        return run_compare_snapshots(args.before, args.after, args.output)
    if args.command == "finalize":
        return run_finalize(
            args.campaign_root,
            args.output,
            unit_tests=args.unit_tests,
            external_tests=args.external_tests,
            citygaussian_tests=args.citygaussian_tests,
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
