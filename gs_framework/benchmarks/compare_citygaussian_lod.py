#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_real_scene import (  # noqa: E402
    FRAME_JSONL,
    RESOLVED_CONFIG_JSON,
    environment_metadata,
    gpu_memory_snapshot,
    json_safe,
    load_camera_path,
    load_config,
    load_scene_for_benchmark,
    require_cuda,
    resolve_camera_path,
    sample_camera_indices,
    scene_stats_metadata,
    write_json,
)
from benchmarks.compare_citygaussian_same_asset import (  # noqa: E402
    compare_images,
    render_for_comparison,
)
from scene_runtime import GsplatRenderer  # noqa: E402


SUMMARY_JSON = "image_comparison_summary.json"


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _percent_change(child: float, parent: float) -> float | None:
    if parent == 0.0:
        return None
    return 100.0 * (child - parent) / parent


def _write_comparison_png(
    path: Path,
    parent: torch.Tensor,
    child: torch.Tensor,
) -> None:
    try:
        import imageio.v3 as imageio
    except Exception as exc:
        raise RuntimeError(f"imageio is required for --write-images: {exc}") from exc
    difference = (parent.float() - child.float()).abs()
    composite = torch.cat([parent.float(), child.float(), difference * 4.0], dim=1)
    image = (composite.clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, image)


def _render_frames(
    *,
    config: dict[str, Any],
    assets: Path | list[Path],
    cameras,
    indices: list[int],
    device: torch.device,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    scene, stats = load_scene_for_benchmark(config, assets, device)
    post_transfer = gpu_memory_snapshot(device, "after_gpu_transfer")
    renderer = GsplatRenderer(device)
    outputs = {}
    for frame_index in indices:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rendered = render_for_comparison(
            scene,
            renderer,
            cameras[frame_index],
            config,
            stats,
            device,
        )
        rendered["memory"] = gpu_memory_snapshot(device, "result_collection")
        rendered["image"] = rendered["image"].detach().cpu()
        rendered["alpha"] = (
            rendered["alpha"].detach().cpu() if rendered["alpha"] is not None else None
        )
        outputs[frame_index] = rendered
    metadata = scene_stats_metadata(scene, stats)
    del scene, renderer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return outputs, metadata, post_transfer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare CG-FIXED-FINEST-VISIBILITY and CG-LOD-RELEASE on frozen cameras."
    )
    parser.add_argument("--lod-finest-asset", type=Path, required=True)
    parser.add_argument("--lod-middle-asset", type=Path, required=True)
    parser.add_argument("--lod-coarsest-asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-path", type=Path)
    parser.add_argument("--frame-index", action="append", type=int)
    parser.add_argument("--sample-frames", type=int, default=12)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--write-images",
        action="store_true",
        help="Write parent | child | 4x absolute-difference RGB composites.",
    )
    args = parser.parse_args(argv)

    parent_config = load_config("CG-FIXED-FINEST-VISIBILITY")
    child_config = load_config("CG-LOD-RELEASE")
    camera_path = load_camera_path(
        resolve_camera_path(parent_config, args.camera_path),
        trajectory_id=parent_config["trajectory_id"],
    )
    indices = args.frame_index or sample_camera_indices(
        len(camera_path.cameras), int(args.sample_frames)
    )
    for frame_index in indices:
        if frame_index < 0 or frame_index >= len(camera_path.cameras):
            raise ValueError(
                f"Frame index {frame_index} is outside camera path length {len(camera_path.cameras)}."
            )
    device = require_cuda(args.device_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        args.output_dir / RESOLVED_CONFIG_JSON,
        {
            "schema_version": 1,
            "parent_configuration": parent_config,
            "child_configuration": child_config,
            "lod_assets": [
                str(args.lod_finest_asset),
                str(args.lod_middle_asset),
                str(args.lod_coarsest_asset),
            ],
            "frame_indices": indices,
            "camera_path": camera_path.metadata,
            "load_policy": "sequential_parent_then_child",
        },
    )

    parent_outputs, parent_scene, parent_post_transfer = _render_frames(
        config=parent_config,
        assets=args.lod_finest_asset,
        cameras=camera_path.cameras,
        indices=indices,
        device=device,
    )
    child_outputs, child_scene, child_post_transfer = _render_frames(
        config=child_config,
        assets=[
            args.lod_finest_asset,
            args.lod_middle_asset,
            args.lod_coarsest_asset,
        ],
        cameras=camera_path.cameras,
        indices=indices,
        device=device,
    )

    rows = []
    for frame_index in indices:
        parent = parent_outputs[frame_index]
        child = child_outputs[frame_index]
        image_difference = compare_images(parent["image"], child["image"])
        alpha_difference = compare_images(parent["alpha"], child["alpha"])
        if args.write_images:
            _write_comparison_png(
                args.output_dir / "images" / f"frame_{frame_index:04d}.png",
                parent["image"],
                child["image"],
            )
        parent_end_to_end = parent["preparation_time_ms"] + parent["render_time_ms"]
        child_end_to_end = child["preparation_time_ms"] + child["render_time_ms"]
        metrics = child["scene_metrics"]
        rows.append(
            {
                "schema_version": 1,
                "record_kind": "citygaussian_release_lod_image_comparison_frame",
                "trajectory_index": frame_index,
                "camera_id": camera_path.cameras[frame_index].camera_id,
                "parent_configuration_id": parent_config["configuration_id"],
                "child_configuration_id": child_config["configuration_id"],
                "parent_finite_rgb": parent["finite_image"],
                "parent_finite_alpha": parent["finite_alpha"],
                "child_finite_rgb": child["finite_image"],
                "child_finite_alpha": child["finite_alpha"],
                "parent_selected_gaussian_count": int(
                    parent["scene_metrics"].selected_gaussian_count
                ),
                "child_selected_gaussian_count": int(metrics.selected_gaussian_count),
                "child_selected_cell_ids": metrics.selected_cell_ids,
                "child_selected_lod_by_cell": metrics.selected_lod_by_cell,
                "child_selected_level_label_by_cell": metrics.selected_level_label_by_cell,
                "child_selected_cell_count_by_level": metrics.selected_cell_count_by_level,
                "child_selected_gaussian_count_by_level": metrics.selected_gaussian_count_by_level,
                "parent_radii_visible_gaussian_count": parent["visible_gaussians"],
                "child_radii_visible_gaussian_count": child["visible_gaussians"],
                "parent_visibility_selection_ms": parent[
                    "scene_metrics"
                ].visibility_selection_ms,
                "child_visibility_selection_ms": metrics.visibility_selection_ms,
                "child_lod_selection_ms": metrics.lod_selection_ms,
                "parent_selected_cell_assembly_ms": parent[
                    "scene_metrics"
                ].selected_cell_assembly_ms,
                "child_selected_cell_assembly_ms": metrics.selected_cell_assembly_ms,
                "parent_rasterization_time_ms": parent["render_time_ms"],
                "child_rasterization_time_ms": child["render_time_ms"],
                "parent_end_to_end_frame_time_ms": parent_end_to_end,
                "child_end_to_end_frame_time_ms": child_end_to_end,
                "parent_end_to_end_fps": 1000.0 / parent_end_to_end,
                "child_end_to_end_fps": 1000.0 / child_end_to_end,
                "rgb_difference": image_difference,
                "alpha_difference": alpha_difference,
                "parent_render_peak_allocated_bytes": parent["memory"][
                    "torch_peak_allocated_bytes"
                ],
                "parent_render_peak_reserved_bytes": parent["memory"][
                    "torch_peak_reserved_bytes"
                ],
                "child_render_peak_allocated_bytes": child["memory"][
                    "torch_peak_allocated_bytes"
                ],
                "child_render_peak_reserved_bytes": child["memory"][
                    "torch_peak_reserved_bytes"
                ],
            }
        )

    with (args.output_dir / FRAME_JSONL).open("w") as handle:
        for row in rows:
            handle.write(
                json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n"
            )

    parent_frame_time = _mean(rows, "parent_end_to_end_frame_time_ms")
    child_frame_time = _mean(rows, "child_end_to_end_frame_time_ms")
    parent_fps = 1000.0 / parent_frame_time
    child_fps = 1000.0 / child_frame_time
    parent_peak_allocated = max(
        int(row["parent_render_peak_allocated_bytes"]) for row in rows
    )
    child_peak_allocated = max(
        int(row["child_render_peak_allocated_bytes"]) for row in rows
    )
    parent_peak_reserved = max(
        int(row["parent_render_peak_reserved_bytes"]) for row in rows
    )
    child_peak_reserved = max(
        int(row["child_render_peak_reserved_bytes"]) for row in rows
    )
    summary = {
        "schema_version": 1,
        "record_kind": "citygaussian_release_lod_image_comparison_summary",
        "parent_configuration_id": parent_config["configuration_id"],
        "child_configuration_id": child_config["configuration_id"],
        "frame_indices": indices,
        "frame_count": len(rows),
        "parent_scene": parent_scene,
        "child_scene": child_scene,
        "all_finite": all(
            row["parent_finite_rgb"]
            and row["parent_finite_alpha"]
            and row["child_finite_rgb"]
            and row["child_finite_alpha"]
            for row in rows
        ),
        "max_rgb_absolute_difference": max(
            row["rgb_difference"]["max_abs_diff"] for row in rows
        ),
        "mean_rgb_absolute_difference": float(
            np.mean([row["rgb_difference"]["mean_abs_diff"] for row in rows])
        ),
        "max_alpha_absolute_difference": max(
            row["alpha_difference"]["max_abs_diff"] for row in rows
        ),
        "mean_alpha_absolute_difference": float(
            np.mean([row["alpha_difference"]["mean_abs_diff"] for row in rows])
        ),
        "minimum_psnr": min(row["rgb_difference"]["psnr"] for row in rows),
        "mean_psnr": float(np.mean([row["rgb_difference"]["psnr"] for row in rows])),
        "parent_mean_end_to_end_frame_time_ms": parent_frame_time,
        "child_mean_end_to_end_frame_time_ms": child_frame_time,
        "end_to_end_frame_time_change_percent": _percent_change(
            child_frame_time, parent_frame_time
        ),
        "parent_end_to_end_fps": parent_fps,
        "child_end_to_end_fps": child_fps,
        "end_to_end_fps_change_percent": _percent_change(child_fps, parent_fps),
        "parent_post_transfer_allocated_bytes": parent_post_transfer[
            "torch_current_allocated_bytes"
        ],
        "child_post_transfer_allocated_bytes": child_post_transfer[
            "torch_current_allocated_bytes"
        ],
        "post_transfer_allocated_change_bytes": int(
            child_post_transfer["torch_current_allocated_bytes"]
            - parent_post_transfer["torch_current_allocated_bytes"]
        ),
        "parent_render_peak_allocated_bytes": parent_peak_allocated,
        "child_render_peak_allocated_bytes": child_peak_allocated,
        "render_peak_allocated_change_bytes": child_peak_allocated
        - parent_peak_allocated,
        "parent_render_peak_reserved_bytes": parent_peak_reserved,
        "child_render_peak_reserved_bytes": child_peak_reserved,
        "render_peak_reserved_change_bytes": child_peak_reserved - parent_peak_reserved,
        "environment": environment_metadata(),
        "interpretation": {
            "image_equivalence_required": False,
            "reason": "Adaptive LoD intentionally selects different released representations.",
            "official_runtime_streaming": False,
            "all_three_child_levels_resident": True,
        },
    }
    write_json(args.output_dir / SUMMARY_JSON, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
