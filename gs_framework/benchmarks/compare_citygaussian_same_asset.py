#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.benchmark_real_scene import (  # noqa: E402
    CONFIG_FILES,
    FRAME_JSONL,
    RESOLVED_CONFIG_JSON,
    environment_metadata,
    gpu_memory_snapshot,
    load_camera_path,
    load_config,
    load_scene_for_benchmark,
    make_frame_camera,
    make_render_settings,
    require_cuda,
    resolve_camera_path,
    json_safe,
    scene_stats_metadata,
    write_json,
)
from scene_runtime import GsplatRenderer  # noqa: E402


SUMMARY_JSON = "image_equivalence_summary.json"


def _finite_bool(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _psnr_from_mse(mse: float) -> float:
    if mse == 0.0:
        return float("inf")
    return float(-10.0 * math.log10(mse))


@torch.no_grad()
def render_for_comparison(scene, renderer, camera_spec, config, scene_stats, device: torch.device) -> dict[str, Any]:
    frame_camera = make_frame_camera(camera_spec, device)
    sh_degree = min(int(config.get("sh_degree", 3)), int(scene_stats.sh_degree))
    settings = make_render_settings(config, camera_spec, sh_degree=sh_degree)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prep_start = time.perf_counter()
    active = scene.prepare_frame(frame_camera, settings)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prep_ms = (time.perf_counter() - prep_start) * 1000.0
    result = renderer.render(active, frame_camera, settings)
    return {
        "image": result.image.detach(),
        "alpha": result.alpha.detach() if result.alpha is not None else None,
        "finite_image": _finite_bool(result.image),
        "finite_alpha": _finite_bool(result.alpha) if result.alpha is not None else None,
        "active_gaussians": int(active.num_gaussians),
        "visible_gaussians": float(result.visible_gaussians),
        "preparation_time_ms": float(prep_ms),
        "render_time_ms": float(result.render_time_ms),
        "scene_metrics": active.scene_metrics,
    }


def compare_images(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    diff = (left.float() - right.float()).abs()
    mse = float(torch.mean((left.float() - right.float()) ** 2).item())
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "mse": mse,
        "psnr": _psnr_from_mse(mse),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render two CityGaussian-related configurations on the same frames and compare images."
    )
    parser.add_argument("--asset", type=Path, required=True, help="Gaussian PLY used by both configurations.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left-config-id", default="CG-FINEST-FLAT", choices=sorted(CONFIG_FILES))
    parser.add_argument("--right-config-id", default="CG-FIXED-FINEST-ALL", choices=sorted(CONFIG_FILES))
    parser.add_argument("--camera-path", type=Path, default=None)
    parser.add_argument("--frame-index", action="append", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--log-gpu-events", action="store_true")
    parser.add_argument(
        "--sequential-load",
        action="store_true",
        help="Render the left configuration, release it, then load and render the right configuration.",
    )
    return parser


def _row_from_pair(
    *,
    frame_index,
    camera,
    left,
    right,
    left_config_id: str,
    right_config_id: str,
    device: torch.device,
) -> dict[str, Any]:
    image_diff = compare_images(left["image"], right["image"])
    alpha_diff = (
        compare_images(left["alpha"], right["alpha"])
        if left["alpha"] is not None and right["alpha"] is not None
        else {}
    )
    return {
        "schema_version": 1,
        "record_kind": "citygaussian_same_asset_image_comparison_frame",
        "frame_index": int(frame_index),
        "camera_id": camera.camera_id,
        "left_configuration_id": left_config_id,
        "right_configuration_id": right_config_id,
        "left_finite_image": left["finite_image"],
        "left_finite_alpha": left["finite_alpha"],
        "right_finite_image": right["finite_image"],
        "right_finite_alpha": right["finite_alpha"],
        "left_active_gaussians": left["active_gaussians"],
        "right_active_gaussians": right["active_gaussians"],
        "left_visible_gaussians": left["visible_gaussians"],
        "right_visible_gaussians": right["visible_gaussians"],
        "left_preparation_time_ms": left["preparation_time_ms"],
        "right_preparation_time_ms": right["preparation_time_ms"],
        "left_render_time_ms": left["render_time_ms"],
        "right_render_time_ms": right["render_time_ms"],
        "left_active_cell_count": float(left["scene_metrics"].active_cell_count),
        "right_active_cell_count": float(right["scene_metrics"].active_cell_count),
        "left_selected_cell_ids": left["scene_metrics"].selected_cell_ids,
        "right_selected_cell_ids": right["scene_metrics"].selected_cell_ids,
        "left_selected_gaussian_count": float(left["scene_metrics"].selected_gaussian_count),
        "right_selected_gaussian_count": float(right["scene_metrics"].selected_gaussian_count),
        "right_visibility_selection_ms": float(right["scene_metrics"].visibility_selection_ms),
        "right_selected_cell_assembly_ms": float(right["scene_metrics"].selected_cell_assembly_ms),
        "image_difference": image_diff,
        "alpha_difference": alpha_diff,
        "memory": gpu_memory_snapshot(device, "result_collection"),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    left_config = load_config(args.left_config_id)
    right_config = load_config(args.right_config_id)
    camera_json = resolve_camera_path(left_config, args.camera_path)
    camera_path = load_camera_path(camera_json, trajectory_id=left_config.get("trajectory_id"))
    device = require_cuda(args.device_index)

    indices = args.frame_index
    if not indices:
        indices = list(range(min(int(args.max_frames), len(camera_path.cameras))))
    for index in indices:
        if index < 0 or index >= len(camera_path.cameras):
            raise ValueError(f"Frame index {index} is outside camera path length {len(camera_path.cameras)}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / RESOLVED_CONFIG_JSON,
        {
            "schema_version": 1,
            "left_configuration": left_config,
            "right_configuration": right_config,
            "asset": str(args.asset),
            "frame_indices": indices,
            "camera_path": camera_path.metadata,
        },
    )

    rows = []
    if args.sequential_load:
        left_scene, left_stats = load_scene_for_benchmark(left_config, args.asset, device)
        left_renderer = GsplatRenderer(device)
        left_outputs = {}
        for frame_index in indices:
            camera = camera_path.cameras[frame_index]
            rendered = render_for_comparison(left_scene, left_renderer, camera, left_config, left_stats, device)
            rendered["image"] = rendered["image"].detach().cpu()
            rendered["alpha"] = rendered["alpha"].detach().cpu() if rendered["alpha"] is not None else None
            left_outputs[frame_index] = rendered
        left_scene_metadata = scene_stats_metadata(left_scene, left_stats)
        del left_scene, left_renderer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        right_scene, right_stats = load_scene_for_benchmark(right_config, args.asset, device)
        right_renderer = GsplatRenderer(device)
        for frame_index in indices:
            camera = camera_path.cameras[frame_index]
            right = render_for_comparison(right_scene, right_renderer, camera, right_config, right_stats, device)
            right["image"] = right["image"].detach().cpu()
            right["alpha"] = right["alpha"].detach().cpu() if right["alpha"] is not None else None
            rows.append(
                _row_from_pair(
                    frame_index=frame_index,
                    camera=camera,
                    left=left_outputs[frame_index],
                    right=right,
                    left_config_id=args.left_config_id,
                    right_config_id=args.right_config_id,
                    device=device,
                )
            )
        right_scene_metadata = scene_stats_metadata(right_scene, right_stats)
    else:
        left_scene, left_stats = load_scene_for_benchmark(left_config, args.asset, device)
        right_scene, right_stats = load_scene_for_benchmark(right_config, args.asset, device)
        left_renderer = GsplatRenderer(device)
        right_renderer = GsplatRenderer(device)
        for frame_index in indices:
            camera = camera_path.cameras[frame_index]
            left = render_for_comparison(left_scene, left_renderer, camera, left_config, left_stats, device)
            right = render_for_comparison(right_scene, right_renderer, camera, right_config, right_stats, device)
            rows.append(
                _row_from_pair(
                    frame_index=frame_index,
                    camera=camera,
                    left=left,
                    right=right,
                    left_config_id=args.left_config_id,
                    right_config_id=args.right_config_id,
                    device=device,
                )
            )
        left_scene_metadata = scene_stats_metadata(left_scene, left_stats)
        right_scene_metadata = scene_stats_metadata(right_scene, right_stats)

    with (args.output_dir / FRAME_JSONL).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n")

    summary = {
        "schema_version": 1,
        "record_kind": "citygaussian_same_asset_image_comparison_summary",
        "asset": str(args.asset),
        "frame_indices": indices,
        "frame_count": len(rows),
        "left_configuration_id": args.left_config_id,
        "right_configuration_id": args.right_config_id,
        "left_scene": left_scene_metadata,
        "right_scene": right_scene_metadata,
        "max_image_max_abs_diff": max(row["image_difference"]["max_abs_diff"] for row in rows),
        "max_image_mean_abs_diff": max(row["image_difference"]["mean_abs_diff"] for row in rows),
        "min_image_psnr": min(row["image_difference"]["psnr"] for row in rows),
        "all_finite": all(
            row["left_finite_image"]
            and row["right_finite_image"]
            and row["left_finite_alpha"]
            and row["right_finite_alpha"]
            for row in rows
        ),
        "environment": environment_metadata(),
        "tolerance_policy": {
            "bitwise_identity_required": False,
            "reason": "Cell concatenation reorders the same Gaussian set before gsplat rasterization.",
            "investigate_if_max_abs_diff_exceeds": 0.02,
            "investigate_if_mean_abs_diff_exceeds": 0.001,
        },
    }
    write_json(args.output_dir / SUMMARY_JSON, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
