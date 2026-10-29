#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_real_scene import (  # noqa: E402
    load_camera_path,
    make_frame_camera,
    make_render_settings,
    require_cuda,
    write_json,
)
from scene_runtime import SceneLoader  # noqa: E402


DEFAULT_CAMERA_PATH = ROOT / "benchmarks" / "camera_paths" / "citygaussian_23m_navigation_v1.json"
DEFAULT_FIXTURE = (
    ROOT / "unit_tests" / "fixtures" / "citygaussian_lod" / "release_lod_oracle_v1.json"
)


def _parse_cell_ids(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _int_key_dict(value: dict[Any, Any]) -> dict[int, Any]:
    return {int(key): item for key, item in value.items()}


def validate(
    *,
    assets: list[Path],
    fixture_path: Path,
    output: Path,
    camera_path_json: Path,
    device: torch.device,
) -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text())
    if fixture.get("policy") != "CG-LOD-RELEASE":
        raise ValueError("Runtime validation requires the authoritative CG-LOD-RELEASE fixture.")
    camera_path = load_camera_path(camera_path_json, trajectory_id="citygaussian_23m_navigation_v1")
    scene = SceneLoader.load(
        assets,
        device=device,
        scene_type="citygaussian_lod_release",
        citygaussian_block_dim=fixture["block_dim"],
        citygaussian_aabb=fixture["aabb"],
    )
    stats = scene.get_scene_stats()

    records = []
    failures = []
    aggregate_cells_by_level = [0 for _ in fixture["level_order"]]
    aggregate_gaussians_by_level = [0 for _ in fixture["level_order"]]
    for expected in fixture["frames"]:
        trajectory_index = int(expected["trajectory_index"])
        camera_spec = camera_path.cameras[trajectory_index]
        frame_camera = make_frame_camera(camera_spec, device)
        settings = make_render_settings(
            {
                "width": camera_spec.width,
                "height": camera_spec.height,
                "near_plane": 0.01,
                "far_plane": 10000000000.0,
                "render_mode": "RGB",
                "background": [0.0, 0.0, 0.0],
                "packed": False,
                "rasterize_mode": "classic",
                "camera_model": "pinhole",
                "radius_clip": 0.0,
                "eps2d": 0.3,
                "sh_degree": stats.sh_degree,
            },
            camera_spec,
            sh_degree=stats.sh_degree,
        )
        active = scene.prepare_frame(frame_camera, settings)
        metrics = active.scene_metrics

        actual_ids = _parse_cell_ids(metrics.selected_cell_ids)
        actual_lods = dict(metrics.selected_lod_by_cell or {})
        actual_labels = dict(metrics.selected_level_label_by_cell or {})
        actual_counts = dict(metrics.selected_gaussian_count_by_cell or {})
        actual_cells_by_level = list(metrics.selected_cell_count_by_level or [])
        actual_gaussians_by_level = list(metrics.selected_gaussian_count_by_level or [])
        expected_lods = _int_key_dict(expected["selected_lod_by_cell"])
        expected_labels = _int_key_dict(expected["selected_level_label_by_cell"])
        expected_counts = _int_key_dict(expected["selected_gaussian_count_by_cell"])

        checks = {
            "visible_cell_ids_match": actual_ids == expected["visible_cell_ids"],
            "selected_lod_by_cell_match": actual_lods == expected_lods,
            "selected_level_label_by_cell_match": actual_labels == expected_labels,
            "selected_gaussian_count_by_cell_match": actual_counts == expected_counts,
            "selected_cell_count_by_level_match": actual_cells_by_level
            == expected["selected_cell_count_by_level"],
            "selected_gaussian_count_by_level_match": actual_gaussians_by_level
            == expected["selected_gaussian_count_by_level"],
            "total_selected_gaussian_count_match": int(metrics.selected_gaussian_count)
            == int(expected["total_selected_gaussian_count"]),
            "nearest_cell_id_match": int(metrics.nearest_cell_id)
            == int(expected["nearest_cell_id"]),
            "fallback_changed_mask_match": bool(metrics.nearest_cell_fallback_changed_mask)
            == bool(expected["fallback_changed_mask"]),
            "active_equals_selected": int(active.num_gaussians)
            == int(metrics.selected_gaussian_count),
            "resident_count_match": int(metrics.resident_gaussians)
            == sum(int(value) for value in fixture["level_gaussian_counts"]),
            "release_policy_used": metrics.citygaussian_lod_policy == "CG-LOD-RELEASE",
            "coarsest_visibility_bounds_used": metrics.citygaussian_visibility_bounds_level
            == fixture["level_order"][-1],
            "no_vq_reconstruction": bool(metrics.citygaussian_no_vq_reconstruction),
            "no_streaming": bool(metrics.citygaussian_no_streaming),
        }
        passed = all(checks.values())
        row = {
            "trajectory_index": trajectory_index,
            "camera_id": camera_spec.camera_id,
            **checks,
            "passed": passed,
            "visible_cell_ids": actual_ids,
            "selected_lod_by_cell": actual_lods,
            "selected_level_label_by_cell": actual_labels,
            "selected_gaussian_count_by_cell": actual_counts,
            "selected_cell_count_by_level": actual_cells_by_level,
            "selected_gaussian_count_by_level": actual_gaussians_by_level,
            "selected_gaussian_count": int(metrics.selected_gaussian_count),
            "active_gaussian_count": int(active.num_gaussians),
            "resident_gaussian_count": int(metrics.resident_gaussians),
            "nearest_cell_id": int(metrics.nearest_cell_id),
            "fallback_changed_mask": bool(metrics.nearest_cell_fallback_changed_mask),
            "visibility_selection_ms": float(metrics.visibility_selection_ms),
            "lod_selection_ms": float(metrics.lod_selection_ms),
            "selected_cell_assembly_ms": float(metrics.selected_cell_assembly_ms),
        }
        for level_index in range(len(aggregate_cells_by_level)):
            aggregate_cells_by_level[level_index] += actual_cells_by_level[level_index]
            aggregate_gaussians_by_level[level_index] += actual_gaussians_by_level[level_index]
        if not passed:
            failures.append(row)
        records.append(row)

    selected_counts = [row["selected_gaussian_count"] for row in records]
    summary = {
        "schema_version": 1,
        "record_kind": "citygaussian_lod_release_oracle_validation",
        "policy": fixture["policy"],
        "assets": [str(path) for path in assets],
        "fixture": str(fixture_path),
        "scene_type": stats.scene_type,
        "level_order": fixture["level_order"],
        "level_gaussian_counts": fixture["level_gaussian_counts"],
        "resident_gaussian_count": int(stats.total_gaussians),
        "frame_count": len(records),
        "passed": not failures,
        "failure_count": len(failures),
        "selected_cell_count_by_level_across_frames": aggregate_cells_by_level,
        "selected_gaussian_count_by_level_across_frames": aggregate_gaussians_by_level,
        "min_selected_gaussian_count": min(selected_counts),
        "max_selected_gaussian_count": max(selected_counts),
        "fallback_cases_validated": [
            row["trajectory_index"] for row in records if row["fallback_changed_mask"]
        ],
        "mean_visibility_selection_ms": sum(
            row["visibility_selection_ms"] for row in records
        )
        / len(records),
        "mean_lod_selection_ms": sum(row["lod_selection_ms"] for row in records)
        / len(records),
        "mean_selected_cell_assembly_ms": sum(
            row["selected_cell_assembly_ms"] for row in records
        )
        / len(records),
        "records": records,
        "failures": failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate CG-LOD-RELEASE selected sets against the authoritative oracle."
    )
    parser.add_argument("--lod-finest-asset", type=Path, required=True)
    parser.add_argument("--lod-middle-asset", type=Path, required=True)
    parser.add_argument("--lod-coarsest-asset", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--camera-path", type=Path, default=DEFAULT_CAMERA_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate(
        assets=[
            args.lod_finest_asset,
            args.lod_middle_asset,
            args.lod_coarsest_asset,
        ],
        fixture_path=args.fixture,
        output=args.output,
        camera_path_json=args.camera_path,
        device=require_cuda(args.device_index),
    )
    if not summary["passed"]:
        print(f"Release-LoD oracle validation failed; see {args.output}", file=sys.stderr)
        return 1
    print(f"Release-LoD oracle validation passed; summary written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
