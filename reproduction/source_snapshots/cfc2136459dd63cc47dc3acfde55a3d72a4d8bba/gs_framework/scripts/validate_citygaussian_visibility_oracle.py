#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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
from scene_runtime.citygaussian import (  # noqa: E402
    OFFICIAL_MATRIXCITY_AABB,
    OFFICIAL_MATRIXCITY_BLOCK_DIM,
    citygaussian_in_frustum,
    citygaussian_official_camera_from_frame,
)


DEFAULT_CAMERA_PATH = ROOT / "benchmarks" / "camera_paths" / "citygaussian_23m_navigation_v1.json"


def _parse_ids(value: str) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.split(",") if part != ""]


def _max_abs_box_diff(left: list[float], right: list[float]) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def _compare_float(left: float, right: float, tolerance: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(float(left) - float(right)) <= tolerance


def validate(
    *,
    asset: Path,
    fixture_path: Path,
    output: Path,
    camera_path_json: Path,
    device: torch.device,
    tolerance: float,
) -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text())
    camera_path = load_camera_path(camera_path_json, trajectory_id="citygaussian_23m_navigation_v1")
    scene = SceneLoader.load(
        [asset],
        device=device,
        scene_type="citygaussian_visibility",
        citygaussian_block_dim=fixture["cell_metadata"].get("block_dim", OFFICIAL_MATRIXCITY_BLOCK_DIM),
        citygaussian_aabb=fixture["cell_metadata"].get("aabb", OFFICIAL_MATRIXCITY_AABB),
        citygaussian_lod_source_level=fixture["asset"].get("identifier", "citygaussian_asset"),
    )
    stats = scene.get_scene_stats()

    records = []
    failures = []
    for expected in fixture["visibility_records"]:
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
        official_camera = citygaussian_official_camera_from_frame(frame_camera, settings)
        selection = citygaussian_in_frustum(
            official_camera,
            scene.cell_corners,
            scene.cell_counts,
            aabb=scene.aabb,
            block_dim=scene.block_dim,
        )
        actual_ids = _parse_ids(metrics.selected_cell_ids)
        expected_ids = [int(v) for v in expected["visible_cell_ids"]]
        distance_ok = True
        max_distance_abs_diff = 0.0
        for cell_id_text, expected_distance in expected["per_visible_cell_camera_distance"].items():
            cell_id = int(cell_id_text)
            actual_distance = float(selection.per_visible_cell_distances[cell_id])
            diff = abs(actual_distance - float(expected_distance))
            max_distance_abs_diff = max(max_distance_abs_diff, diff)
            distance_ok = distance_ok and diff <= tolerance
        boxes_ok = True
        max_box_abs_diff = 0.0
        for cell_id_text, expected_box in expected["per_visible_cell_projected_box"].items():
            cell_id = int(cell_id_text)
            diff = _max_abs_box_diff(selection.per_visible_cell_projected_boxes[cell_id], expected_box)
            max_box_abs_diff = max(max_box_abs_diff, diff)
            boxes_ok = boxes_ok and diff <= tolerance

        row = {
            "trajectory_index": trajectory_index,
            "camera_id": camera_spec.camera_id,
            "expected_visible_cell_ids": expected_ids,
            "actual_visible_cell_ids": actual_ids,
            "visible_cell_ids_match": actual_ids == expected_ids,
            "expected_visible_cell_count": int(expected["visible_cell_count"]),
            "actual_visible_cell_count": int(metrics.selected_cell_count),
            "visible_cell_count_match": int(metrics.selected_cell_count) == int(expected["visible_cell_count"]),
            "expected_nearest_cell_id": int(expected["nearest_cell_id"]),
            "actual_nearest_cell_id": int(metrics.nearest_cell_id),
            "nearest_cell_id_match": int(metrics.nearest_cell_id) == int(expected["nearest_cell_id"]),
            "expected_fallback_changed_mask": bool(expected["nearest_cell_fallback_changed_mask"]),
            "actual_fallback_changed_mask": bool(metrics.nearest_cell_fallback_changed_mask),
            "fallback_changed_mask_match": bool(metrics.nearest_cell_fallback_changed_mask)
            == bool(expected["nearest_cell_fallback_changed_mask"]),
            "expected_selected_gaussian_count": int(expected["selected_gaussian_count"]),
            "actual_selected_gaussian_count": int(metrics.selected_gaussian_count),
            "selected_gaussian_count_match": int(metrics.selected_gaussian_count)
            == int(expected["selected_gaussian_count"]),
            "active_gaussian_count": int(active.num_gaussians),
            "active_equals_selected": int(active.num_gaussians) == int(metrics.selected_gaussian_count),
            "resident_gaussian_count": int(metrics.resident_gaussians),
            "resident_equals_total": int(metrics.resident_gaussians) == int(stats.total_gaussians),
            "max_distance_abs_diff": max_distance_abs_diff,
            "distances_within_tolerance": distance_ok,
            "max_projected_box_abs_diff": max_box_abs_diff,
            "projected_boxes_within_tolerance": boxes_ok,
            "visibility_selection_ms": float(metrics.visibility_selection_ms),
            "selected_cell_assembly_ms": float(metrics.selected_cell_assembly_ms),
            "no_adaptive_lod": bool(metrics.citygaussian_no_adaptive_lod),
            "no_streaming": bool(metrics.citygaussian_no_streaming),
        }
        row_ok = all(
            bool(row[key])
            for key in [
                "visible_cell_ids_match",
                "visible_cell_count_match",
                "nearest_cell_id_match",
                "fallback_changed_mask_match",
                "selected_gaussian_count_match",
                "active_equals_selected",
                "resident_equals_total",
                "distances_within_tolerance",
                "projected_boxes_within_tolerance",
                "no_adaptive_lod",
                "no_streaming",
            ]
        )
        row["passed"] = row_ok
        if not row_ok:
            failures.append(row)
        records.append(row)

    summary = {
        "schema_version": 1,
        "record_kind": "citygaussian_visibility_oracle_validation",
        "asset": str(asset),
        "fixture": str(fixture_path),
        "scene_type": stats.scene_type,
        "total_gaussians": int(stats.total_gaussians),
        "total_cell_count": int(stats.citygaussian_cell_count or 0),
        "frame_count": len(records),
        "passed": not failures,
        "failure_count": len(failures),
        "fallback_cases_validated": [
            row["trajectory_index"] for row in records if row["expected_fallback_changed_mask"]
        ],
        "min_selected_cell_count": min(row["actual_visible_cell_count"] for row in records),
        "max_selected_cell_count": max(row["actual_visible_cell_count"] for row in records),
        "min_selected_gaussian_count": min(row["actual_selected_gaussian_count"] for row in records),
        "max_selected_gaussian_count": max(row["actual_selected_gaussian_count"] for row in records),
        "max_distance_abs_diff": max(row["max_distance_abs_diff"] for row in records),
        "max_projected_box_abs_diff": max(row["max_projected_box_abs_diff"] for row in records),
        "mean_visibility_selection_ms": sum(row["visibility_selection_ms"] for row in records) / len(records),
        "mean_selected_cell_assembly_ms": sum(row["selected_cell_assembly_ms"] for row in records) / len(records),
        "records": records,
        "failures": failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CityGaussian V1 runtime visibility against oracle fixture.")
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-path", type=Path, default=DEFAULT_CAMERA_PATH)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = require_cuda(args.device_index)
    summary = validate(
        asset=args.asset,
        fixture_path=args.fixture,
        output=args.output,
        camera_path_json=args.camera_path,
        device=device,
        tolerance=float(args.tolerance),
    )
    if not summary["passed"]:
        print(f"Visibility oracle validation failed; see {args.output}", file=sys.stderr)
        return 1
    print(f"Visibility oracle validation passed; summary written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
