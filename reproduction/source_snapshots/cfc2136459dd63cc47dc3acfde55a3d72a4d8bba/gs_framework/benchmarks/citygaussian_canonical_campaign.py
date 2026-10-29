#!/usr/bin/env python
"""Run, aggregate, plot, and validate the canonical CityGaussian V1 campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "benchmarks" / "configs"
TRAJECTORY = REPO_ROOT / "benchmarks" / "camera_paths" / "citygaussian_23m_navigation_v1.json"
FUSED = Path("/home/luis/tese/mc_aerial_c36/point_cloud/iteration_30000/point_cloud.ply")
FINEST = Path("/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_50_vq/point_cloud.ply")
MIDDLE = Path("/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_66_vq/point_cloud.ply")
COARSEST = Path("/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_75_vq/point_cloud.ply")

CONFIG_IDS = [
    "CG-FINEST-FLAT", "CG-FIXED-FINEST-ALL", "CG-FIXED-FINEST-VISIBILITY",
    "CG-LOD-RELEASE", "FLAT-CITYGS-FULL", "CG-FUSED-CELLS-ALL",
    "CG-FUSED-VISIBILITY",
]
CONFIG_PATHS = {
    config_id: CONFIG_DIR / {
        "CG-FINEST-FLAT": "cg_finest_flat.json",
        "CG-FIXED-FINEST-ALL": "cg_fixed_finest_all.json",
        "CG-FIXED-FINEST-VISIBILITY": "cg_fixed_finest_visibility.json",
        "CG-LOD-RELEASE": "cg_lod_release.json",
        "FLAT-CITYGS-FULL": "flat_citygs_full.json",
        "CG-FUSED-CELLS-ALL": "cg_fused_cells_all.json",
        "CG-FUSED-VISIBILITY": "cg_fused_visibility.json",
    }[config_id]
    for config_id in CONFIG_IDS
}
EXECUTION_ORDER = {
    1: CONFIG_IDS,
    2: ["FLAT-CITYGS-FULL", "CG-FUSED-CELLS-ALL", "CG-FUSED-VISIBILITY",
        "CG-FINEST-FLAT", "CG-FIXED-FINEST-ALL", "CG-FIXED-FINEST-VISIBILITY",
        "CG-LOD-RELEASE"],
    3: list(reversed(CONFIG_IDS)),
}
TRANSITIONS = [
    ("CG-FINEST-FLAT", "CG-FIXED-FINEST-ALL", "official cell organization, finest"),
    ("CG-FIXED-FINEST-ALL", "CG-FIXED-FINEST-VISIBILITY", "official visibility, finest"),
    ("CG-FIXED-FINEST-VISIBILITY", "CG-LOD-RELEASE", "released LoD after visibility"),
    ("FLAT-CITYGS-FULL", "CG-FUSED-CELLS-ALL", "official cell organization, fused"),
    ("CG-FUSED-CELLS-ALL", "CG-FUSED-VISIBILITY", "official visibility, fused"),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def git_is_dirty() -> bool:
    return bool(git_value("status", "--porcelain"))


def command_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CUDA_HOME": "/home/luis/miniconda3/envs/nerfstudio2",
        "CPATH": "/home/luis/miniconda3/envs/nerfstudio2/targets/x86_64-linux/include",
        "TORCH_EXTENSIONS_DIR": "/tmp/torch_ext_citygs_canonical",
    })
    return env


def conda_python_command(*args: str) -> list[str]:
    return ["/home/luis/miniconda3/bin/conda", "run", "-n", "nerfstudio2", "python", *args]


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"started_utc={started}\ncommand={json.dumps(command)}\n")
        handle.flush()
        result = subprocess.run(
            command, cwd=REPO_ROOT, env=command_env(), stdout=handle,
            stderr=subprocess.STDOUT, text=True, check=False,
        )
        handle.write(f"\nreturncode={result.returncode}\nfinished_utc={datetime.now(timezone.utc).isoformat()}\n")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log_path}")


def asset_args(config_id: str) -> list[str]:
    if config_id == "CG-LOD-RELEASE":
        return [
            "--lod-finest-asset", str(FINEST), "--lod-middle-asset", str(MIDDLE),
            "--lod-coarsest-asset", str(COARSEST),
        ]
    asset = FUSED if config_id in {
        "FLAT-CITYGS-FULL", "CG-FUSED-CELLS-ALL", "CG-FUSED-VISIBILITY"
    } else FINEST
    return ["--asset", str(asset)]


def input_manifest() -> dict[str, Any]:
    inputs = {
        "fused_asset": FUSED, "lod_finest_asset": FINEST,
        "lod_middle_asset": MIDDLE, "lod_coarsest_asset": COARSEST,
        "trajectory": TRAJECTORY,
    }
    absent = [str(path) for path in inputs.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Missing canonical input(s): {', '.join(absent)}")
    return {
        role: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for role, path in inputs.items()
    }


def initialize_manifest(root: Path) -> dict[str, Any]:
    if git_is_dirty():
        raise RuntimeError("Canonical campaign requires a clean Git worktree.")
    config_hashes = {
        config_id: {"path": str(CONFIG_PATHS[config_id].relative_to(REPO_ROOT)),
                    "sha256": sha256_file(CONFIG_PATHS[config_id])}
        for config_id in CONFIG_IDS
    }
    manifest = {
        "schema_version": 2,
        "campaign_id": root.name,
        "campaign_kind": "canonical_citygaussian_v1_repeated_benchmark",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"git_commit": git_value("rev-parse", "HEAD"), "git_dirty": False},
        "inputs": input_manifest(),
        "configuration_files": config_hashes,
        "warmup_policy": {
            "frame_count": 1,
            "camera": "trajectory frame 0",
            "included_in_aggregates": False,
            "memory_peaks_reset_after_warmup": True,
            "rationale": "One deterministic frame excludes the observed lazy first-use outlier; extensions are precompiled and the same policy is used for all configurations.",
        },
        "repetition_policy": {
            "independent_processes_per_configuration": 3,
            "measured_frames_per_process": 1440,
            "trajectory_order": "unchanged 0..1439",
            "execution_order": {str(key): value for key, value in EXECUTION_ORDER.items()},
        },
        "timing_semantics": {
            "primary": "end_to_end_frame_time_ms",
            "end_to_end_boundary": "synchronized scene.prepare_frame through completion of synchronized GsplatRenderer.render",
            "formula": "preparation_time_ms + render_call_time_ms",
            "excluded": ["scene/asset loading", "warmup", "post-render finite checks", "metric serialization", "image writing"],
            "components": ["preparation_time_ms", "render_call_time_ms", "cuda_rasterization_time_ms"],
        },
        "environment_policy": {
            "CUDA_HOME": command_env()["CUDA_HOME"], "CPATH": command_env()["CPATH"],
            "TORCH_EXTENSIONS_DIR": command_env()["TORCH_EXTENSIONS_DIR"],
        },
        "status": "initialized",
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def run_validation(root: Path) -> None:
    validation = root / "validation"
    commands = [
        ("unit_tests.log", conda_python_command("-m", "unittest", "discover", "-s", "unit_tests")),
        ("five_gaussian_smoke.log", conda_python_command(
            "scripts/smoke_baseline.py", "--device", "cuda", "--output-dir", str(validation / "five_gaussian_smoke"))),
        ("finest_visibility_oracle.log", conda_python_command(
            "scripts/validate_citygaussian_visibility_oracle.py", "--asset", str(FINEST),
            "--fixture", "unit_tests/fixtures/citygaussian_visibility/finest_visibility_oracle_v1.json",
            "--output", str(validation / "finest_visibility_oracle.json"), "--tolerance", "3e-5")),
        ("fused_visibility_oracle.log", conda_python_command(
            "scripts/validate_citygaussian_visibility_oracle.py", "--asset", str(FUSED),
            "--fixture", "unit_tests/fixtures/citygaussian_visibility/fused_visibility_oracle_v1.json",
            "--output", str(validation / "fused_visibility_oracle.json"), "--tolerance", "1.2e-4")),
        ("release_lod_oracle.log", conda_python_command(
            "scripts/validate_citygaussian_lod_oracle.py", "--lod-finest-asset", str(FINEST),
            "--lod-middle-asset", str(MIDDLE), "--lod-coarsest-asset", str(COARSEST),
            "--output", str(validation / "release_lod_oracle.json"))),
    ]
    for name, command in commands:
        run_logged(command, validation / name)
    frames = ["0", "131", "262", "392", "523", "654", "785", "916", "1047", "1177", "1308", "1439"]
    for label, asset, left, right in [
        ("finest_flat_vs_all_cells", FINEST, "CG-FINEST-FLAT", "CG-FIXED-FINEST-ALL"),
        ("fused_flat_vs_all_cells", FUSED, "FLAT-CITYGS-FULL", "CG-FUSED-CELLS-ALL"),
    ]:
        command = conda_python_command(
            "benchmarks/compare_citygaussian_same_asset.py", "--asset", str(asset),
            "--output-dir", str(validation / label), "--left-config-id", left,
            "--right-config-id", right, "--sequential-load",
        )
        for frame in frames:
            command.extend(["--frame-index", frame])
        run_logged(command, validation / f"{label}.log")


def run_formal_processes(root: Path) -> None:
    for repeat, order in EXECUTION_ORDER.items():
        for config_id in order:
            run_dir = root / "runs" / config_id / f"repeat_{repeat:02d}"
            command = conda_python_command(
                "benchmarks/benchmark_real_scene.py", "--config-id", config_id,
                *asset_args(config_id), "--output-dir", str(run_dir), "--formal",
                "--max-frames", "1440", "--num-warmup", "1", "--num-repeat", "1",
                "--repetition-index", str(repeat), "--log-gpu-events",
            )
            run_logged(command, run_dir / "process.log")


def numeric(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "p99": float(np.percentile(values, 99)) if values else None,
        "min": min(values) if values else None, "max": max(values) if values else None,
    }


def across_run_stats(values: list[float]) -> dict[str, float | None]:
    mean = float(np.mean(values)) if values else None
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
    return {
        "mean_of_run_means": mean, "standard_deviation": std,
        "min": min(values) if values else None, "max": max(values) if values else None,
        "coefficient_of_variation_percent": 100.0 * std / mean if mean and std is not None else None,
    }


def percent_change(child: float, parent: float) -> float:
    return 100.0 * (child - parent) / parent


def validate_run(run_dir: Path, config_id: str, repeat: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for name in ["frames.jsonl", "summary.json", "resolved_config.json"]:
        if not (run_dir / name).is_file():
            raise ValueError(f"Missing {run_dir / name}")
    rows = read_jsonl(run_dir / "frames.jsonl")
    summary = read_json(run_dir / "summary.json")
    resolved = read_json(run_dir / "resolved_config.json")
    if len(rows) != 1440 or summary.get("frame_count") != 1440:
        raise ValueError(f"{run_dir}: expected 1440 measured frames")
    if summary.get("configuration_id") != config_id or summary.get("repetition_index") != repeat:
        raise ValueError(f"{run_dir}: configuration/repetition mismatch")
    if resolved.get("warmup_frames") != 1 or resolved.get("repetition_index") != repeat:
        raise ValueError(f"{run_dir}: warmup/repetition metadata mismatch")
    required = {
        "preparation_time_ms", "render_call_time_ms", "cuda_rasterization_time_ms",
        "end_to_end_frame_time_ms", "resident_gaussian_count",
        "selected_gaussian_count", "active_gaussian_count",
    }
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{run_dir} row {index}: missing {sorted(missing)}")
        for key in ("resident_gaussian_count", "selected_gaussian_count", "active_gaussian_count"):
            value = row[key]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{run_dir} row {index}: non-finite {key}")
        expected = float(row["preparation_time_ms"]) + float(row["render_call_time_ms"])
        if not math.isclose(float(row["end_to_end_frame_time_ms"]), expected, abs_tol=1e-6):
            raise ValueError(f"{run_dir} row {index}: invalid end-to-end decomposition")
        if "frame_time_ms" in row or "fps" in row:
            raise ValueError(f"{run_dir} row {index}: ambiguous legacy timing field")
    return rows, summary


def aggregate(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    configs = []
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for config_id in CONFIG_IDS:
        run_records, pooled = [], []
        for repeat in range(1, 4):
            run_dir = root / "runs" / config_id / f"repeat_{repeat:02d}"
            rows, summary = validate_run(run_dir, config_id, repeat)
            pooled.extend(rows)
            timing = {
                key: stats(numeric(rows, key)) for key in [
                    "preparation_time_ms", "render_call_time_ms",
                    "cuda_rasterization_time_ms", "end_to_end_frame_time_ms",
                ]
            }
            run_records.append({
                "repetition_index": repeat, "timing": timing,
                "end_to_end_fps": 1000.0 / timing["end_to_end_frame_time_ms"]["mean"],
                "summary_sha256": sha256_file(run_dir / "summary.json"),
            })
        run_means = {
            key: [record["timing"][key]["mean"] for record in run_records]
            for key in run_records[0]["timing"]
        }
        row = {
            "configuration_id": config_id,
            "repetitions": run_records,
            "across_run": {key: across_run_stats(values) for key, values in run_means.items()},
            "headline_mean_end_to_end_frame_time_ms": across_run_stats(run_means["end_to_end_frame_time_ms"])["mean_of_run_means"],
            "headline_between_run_std_ms": across_run_stats(run_means["end_to_end_frame_time_ms"])["standard_deviation"],
            "end_to_end_fps": 1000.0 / across_run_stats(run_means["end_to_end_frame_time_ms"])["mean_of_run_means"],
            "mean_preparation_time_ms": across_run_stats(run_means["preparation_time_ms"])["mean_of_run_means"],
            "mean_render_call_time_ms": across_run_stats(run_means["render_call_time_ms"])["mean_of_run_means"],
            "mean_cuda_rasterization_time_ms": across_run_stats(run_means["cuda_rasterization_time_ms"])["mean_of_run_means"],
            "resident_gaussian_count": float(np.mean(numeric(pooled, "resident_gaussian_count"))),
            "mean_selected_gaussian_count": float(np.mean(numeric(pooled, "selected_gaussian_count"))),
            "mean_active_gaussian_count": float(np.mean(numeric(pooled, "active_gaussian_count"))),
            "mean_radii_visible_gaussian_count": float(np.mean(numeric(pooled, "visible_gaussians"))),
            "post_transfer_allocated_bytes": max(numeric(pooled, "post_transfer_torch_allocated_bytes")),
            "render_peak_allocated_bytes": max(numeric(pooled, "torch_peak_allocated_bytes")),
            "render_peak_reserved_bytes": max(numeric(pooled, "torch_peak_reserved_bytes")),
        }
        configs.append(row)
        all_rows[config_id] = pooled

    by_id = {row["configuration_id"]: row for row in configs}
    transitions = []
    for parent_id, child_id, label in TRANSITIONS:
        parent, child = by_id[parent_id], by_id[child_id]
        transitions.append({
            "label": label, "parent": parent_id, "child": child_id,
            "end_to_end_frame_time_change_percent": percent_change(child["headline_mean_end_to_end_frame_time_ms"], parent["headline_mean_end_to_end_frame_time_ms"]),
            "end_to_end_fps_change_percent": percent_change(child["end_to_end_fps"], parent["end_to_end_fps"]),
            "preparation_time_change_percent": percent_change(child["mean_preparation_time_ms"], parent["mean_preparation_time_ms"]),
            "render_call_time_change_percent": percent_change(child["mean_render_call_time_ms"], parent["mean_render_call_time_ms"]),
            "cuda_rasterization_time_change_percent": percent_change(child["mean_cuda_rasterization_time_ms"], parent["mean_cuda_rasterization_time_ms"]),
            "selected_gaussian_change_percent": percent_change(child["mean_selected_gaussian_count"], parent["mean_selected_gaussian_count"]),
            "render_peak_allocated_memory_change_bytes": child["render_peak_allocated_bytes"] - parent["render_peak_allocated_bytes"],
            "post_transfer_resident_memory_change_bytes": child["post_transfer_allocated_bytes"] - parent["post_transfer_allocated_bytes"],
        })

    tables = root / "tables"
    write_json(tables / "config_summary.json", {"schema_version": 2, "configurations": configs})
    write_json(tables / "controlled_transitions.json", transitions)
    write_json(tables / "memory_summary.json", [
        {key: row[key] for key in ["configuration_id", "resident_gaussian_count",
            "post_transfer_allocated_bytes", "render_peak_allocated_bytes", "render_peak_reserved_bytes"]}
        for row in configs
    ])
    write_json(tables / "selected_workload_summary.json", [
        {key: row[key] for key in ["configuration_id", "resident_gaussian_count",
            "mean_selected_gaussian_count", "mean_active_gaussian_count", "mean_radii_visible_gaussian_count"]}
        for row in configs
    ])
    write_csv(tables / "config_summary.csv", configs)
    write_csv(tables / "controlled_transitions.csv", transitions)
    write_thesis_table(tables / "thesis_tables.md", configs, transitions)
    make_figures(root, configs, transitions, all_rows)
    write_json(root / "validation" / "package_validation.json", {
        "status": "pass", "validated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_count": len(configs), "repetition_count": 3,
        "measured_frame_count": sum(len(rows) for rows in all_rows.values()),
        "ambiguous_timing_fields_absent": True,
    })
    manifest = read_json(root / "manifest.json")
    manifest.update({"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat()})
    manifest["generated_tables"] = sorted(str(path.relative_to(root)) for path in tables.iterdir())
    manifest["generated_figures"] = sorted(str(path.relative_to(root)) for path in (root / "figures").iterdir())
    write_json(root / "manifest.json", manifest)
    return configs, transitions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for row in rows:
        flat.append({key: value for key, value in row.items() if isinstance(value, (str, int, float)) or value is None})
    keys = list(dict.fromkeys(key for row in flat for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(flat)


def write_thesis_table(path: Path, configs: list[dict[str, Any]], transitions: list[dict[str, Any]]) -> None:
    lines = ["# Canonical CityGaussian V1 benchmark tables", "", "| Configuration | End-to-end ms | Between-run SD | FPS | Preparation ms | Render call ms | CUDA raster ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in configs:
        lines.append(
            f"| `{row['configuration_id']}` | {row['headline_mean_end_to_end_frame_time_ms']:.3f} | "
            f"{row['headline_between_run_std_ms']:.3f} | {row['end_to_end_fps']:.2f} | "
            f"{row['mean_preparation_time_ms']:.3f} | {row['mean_render_call_time_ms']:.3f} | "
            f"{row['mean_cuda_rasterization_time_ms']:.3f} |"
        )
    lines.extend(["", "Controlled transitions are in `controlled_transitions.json`.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figures(root: Path, configs: list[dict[str, Any]], transitions: list[dict[str, Any]], all_rows: dict[str, list[dict[str, Any]]]) -> None:
    import matplotlib.pyplot as plt
    figures = root / "figures"; figures.mkdir(parents=True, exist_ok=True)
    labels = [row["configuration_id"] for row in configs]; x = np.arange(len(labels))
    prep = [row["mean_preparation_time_ms"] for row in configs]
    render = [row["mean_render_call_time_ms"] for row in configs]
    plt.figure(figsize=(11, 5)); plt.bar(x, prep, label="Preparation"); plt.bar(x, render, bottom=prep, label="Render call")
    plt.xticks(x, labels, rotation=30, ha="right"); plt.ylabel("Time (ms)"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "timing_decomposition.png", dpi=180); plt.close()
    plt.figure(figsize=(11, 5)); plt.bar(x, [row["end_to_end_fps"] for row in configs]); plt.xticks(x, labels, rotation=30, ha="right"); plt.ylabel("End-to-end FPS"); plt.tight_layout(); plt.savefig(figures / "end_to_end_fps.png", dpi=180); plt.close()
    width = 0.38; plt.figure(figsize=(11, 5)); plt.bar(x-width/2, [row["resident_gaussian_count"]/1e6 for row in configs], width, label="Resident"); plt.bar(x+width/2, [row["mean_selected_gaussian_count"]/1e6 for row in configs], width, label="Selected"); plt.xticks(x, labels, rotation=30, ha="right"); plt.ylabel("Gaussians (millions)"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "resident_vs_selected.png", dpi=180); plt.close()
    tx = np.arange(len(transitions)); plt.figure(figsize=(11, 5)); plt.bar(tx, [row["end_to_end_frame_time_change_percent"] for row in transitions]); plt.axhline(0, color="black", linewidth=.8); plt.xticks(tx, [row["label"] for row in transitions], rotation=25, ha="right"); plt.ylabel("End-to-end time change (%)"); plt.tight_layout(); plt.savefig(figures / "controlled_end_to_end_transitions.png", dpi=180); plt.close()
    main = read_jsonl(root / "runs" / "CG-FIXED-FINEST-VISIBILITY" / "repeat_01" / "frames.jsonl")
    plt.figure(figsize=(11, 4)); plt.plot(numeric(main, "end_to_end_frame_time_ms"), linewidth=.8); plt.xlabel("Trajectory frame"); plt.ylabel("End-to-end time (ms)"); plt.tight_layout(); plt.savefig(figures / "finest_visibility_end_to_end_along_trajectory.png", dpi=180); plt.close()
    plt.figure(figsize=(11, 4)); plt.plot(np.asarray(numeric(main, "selected_gaussian_count"))/1e6, linewidth=.8); plt.xlabel("Trajectory frame"); plt.ylabel("Selected Gaussians (millions)"); plt.tight_layout(); plt.savefig(figures / "finest_visibility_selected_along_trajectory.png", dpi=180); plt.close()
    lod = all_rows["CG-LOD-RELEASE"]; level_totals = np.sum(np.asarray([row["selected_cell_count_by_level"] for row in lod]), axis=0); plt.figure(figsize=(6, 4)); plt.bar(["Finest", "Middle", "Coarsest"], level_totals); plt.ylabel("Selected cell instances"); plt.tight_layout(); plt.savefig(figures / "lod_selection_distribution.png", dpi=180); plt.close()
    plt.figure(figsize=(11, 5)); plt.bar(x-width/2, [row["post_transfer_allocated_bytes"]/2**30 for row in configs], width, label="Post-transfer allocated"); plt.bar(x+width/2, [row["render_peak_allocated_bytes"]/2**30 for row in configs], width, label="Render peak allocated"); plt.xticks(x, labels, rotation=30, ha="right"); plt.ylabel("PyTorch memory (GiB)"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "allocated_memory_stages.png", dpi=180); plt.close()


def write_readme(root: Path) -> None:
    (root / "README.md").write_text(
        "# Canonical CityGaussian V1 repeated benchmark\n\n"
        "Primary performance is synchronized end-to-end frame time: preparation plus the renderer call. "
        "Warmup is excluded. Each configuration has three fresh-process 1,440-frame repetitions. "
        "See `manifest.json`, `validation/`, `runs/`, `tables/`, and `figures/`.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--run-formal", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); root = args.output_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.run_all or args.initialize:
        initialize_manifest(root); write_readme(root)
    if args.run_all or args.validate:
        run_validation(root)
    if args.run_all or args.run_formal:
        run_formal_processes(root)
    if args.run_all or args.aggregate:
        aggregate(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
