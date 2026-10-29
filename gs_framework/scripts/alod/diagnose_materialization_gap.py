#!/usr/bin/env python3
"""Characterize ALoD semantic selection through gsplat preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from gsplat.cuda._wrapper import fully_fused_projection


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.alod_resident_benchmark import _runtime_config  # noqa: E402
from benchmarks.benchmark_real_scene import (  # noqa: E402
    apply_camera_config,
    load_camera_path,
    load_config,
    make_frame_camera,
    make_render_settings,
    resolve_camera_path,
    sha256_file,
)
from scene_runtime.alod.materialization import (  # noqa: E402
    AlodCompactCullingMetadata,
    conservative_pinhole_materialization,
)
from scene_runtime.alod.scene import AlodResidentScene  # noqa: E402
from scene_runtime.renderer import GsplatRenderer  # noqa: E402


def _ids(values: torch.Tensor, semantic_ids: np.ndarray) -> np.ndarray:
    local = values.detach().to(device="cpu", dtype=torch.int64).numpy()
    return np.ascontiguousarray(semantic_ids[local], dtype=np.int64)


def _projection_without_opacity(active, camera, settings) -> torch.Tensor:
    result = fully_fused_projection(
        active.means,
        None,
        active.quats,
        active.scales,
        camera.w2c[None] if camera.w2c is not None else torch.linalg.inv(camera.c2w)[None],
        camera.K[None],
        int(settings.width),
        int(settings.height),
        eps2d=float(settings.eps2d),
        packed=True,
        near_plane=float(settings.near_plane),
        far_plane=float(settings.far_plane),
        radius_clip=float(settings.radius_clip),
        calc_compensations=settings.rasterize_mode == "antialiased",
        camera_model=settings.camera_model,
        opacities=None,
    )
    return result[2]


def _stats(values: list[int | float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


@torch.no_grad()
def diagnose(asset: Path, config_id: str, device: torch.device) -> dict[str, Any]:
    config = load_config(config_id)
    cameras = apply_camera_config(
        load_camera_path(resolve_camera_path(config), config.get("trajectory_id")),
        config,
    )
    scene = AlodResidentScene(asset, device, _runtime_config(config))
    renderer = GsplatRenderer(device)
    compact = AlodCompactCullingMetadata.from_log_scales(
        scene.asset.positions, scene.asset.log_scales
    )

    rows: list[dict[str, Any]] = []
    for camera_index, camera_spec in enumerate(cameras.cameras):
        frame_camera = make_frame_camera(camera_spec, device)
        settings = make_render_settings(config, camera_spec, sh_degree=int(config["sh_degree"]))
        active = scene.prepare_frame(frame_camera, settings)
        selection = scene.last_selection
        assert selection is not None
        semantic_ids = np.ascontiguousarray(selection.raster_global_node_ids, dtype=np.int64)
        materialization = conservative_pinhole_materialization(
            compact, semantic_ids, frame_camera, settings
        )

        no_opacity_local = _projection_without_opacity(active, frame_camera, settings)
        no_opacity_ids = _ids(no_opacity_local, semantic_ids)
        rendered = renderer.render(active, frame_camera, settings)
        actual_local = rendered.info["gaussian_ids"]
        backend_ids = _ids(actual_local, semantic_ids)
        false_negative_ids = np.setdiff1d(
            backend_ids,
            materialization.materialization_global_node_ids,
            assume_unique=False,
        )

        w2c = frame_camera.w2c if frame_camera.w2c is not None else torch.linalg.inv(frame_camera.c2w)
        camera_depths = active.means @ w2c[2, :3] + w2c[2, 3]
        depth_accept = (camera_depths >= settings.near_plane) & (
            camera_depths <= settings.far_plane
        )
        depth_ids = semantic_ids[
            depth_accept.detach().to(device="cpu").numpy().astype(bool, copy=False)
        ]
        no_opacity_rejected = np.setdiff1d(semantic_ids, no_opacity_ids)
        depth_rejected = np.setdiff1d(semantic_ids, depth_ids)
        projection_or_image_rejected = np.setdiff1d(no_opacity_rejected, depth_rejected)
        opacity_aware_rejected = np.setdiff1d(no_opacity_ids, backend_ids)

        tiles_per_gauss = rendered.info.get("tiles_per_gauss")
        if tiles_per_gauss is None:
            tile_ids = np.empty(0, dtype=np.int64)
        else:
            tile_mask = tiles_per_gauss > 0
            tile_ids = _ids(actual_local[tile_mask], semantic_ids)

        rows.append(
            {
                "camera_index": camera_index,
                "camera_id": camera_spec.camera_id,
                "semantic_selected_hierarchy_rows": int(
                    selection.selected_hierarchy_global_node_ids.size
                ),
                "semantic_selected_rows_including_skybox": int(semantic_ids.size),
                "depth_gate_survivors": int(depth_ids.size),
                "projection_survivors_without_opacity": int(no_opacity_ids.size),
                "backend_projection_survivors": int(backend_ids.size),
                "tile_intersection_survivors": int(tile_ids.size),
                "rejected_by_depth_gate": int(depth_rejected.size),
                "rejected_by_projection_or_image_after_depth": int(
                    projection_or_image_rejected.size
                ),
                "rejected_by_opacity_aware_projection": int(opacity_aware_rejected.size),
                "conservative_materialization_rows": materialization.materialization_rows,
                "conservative_rejected_depth_rows": materialization.rejected_depth_rows,
                "conservative_rejected_screen_bound_rows": (
                    materialization.rejected_screen_bound_rows
                ),
                "backend_false_negative_count": int(false_negative_ids.size),
                "backend_false_negative_global_node_ids": false_negative_ids.tolist(),
                "filter_ms": materialization.filter_seconds * 1000.0,
                "rendered_visible_metric": int(rendered.visible_gaussians),
                "tile_survivor_matches_backend": bool(
                    np.array_equal(np.sort(tile_ids), np.sort(backend_ids))
                ),
            }
        )

    aggregate_keys = (
        "semantic_selected_hierarchy_rows",
        "semantic_selected_rows_including_skybox",
        "depth_gate_survivors",
        "projection_survivors_without_opacity",
        "backend_projection_survivors",
        "tile_intersection_survivors",
        "conservative_materialization_rows",
        "filter_ms",
    )
    aggregate = {key: _stats([row[key] for row in rows]) for key in aggregate_keys}
    return {
        "schema_version": 1,
        "status": (
            "PASS-DIAGNOSTIC-ZERO-BACKEND-FALSE-NEGATIVES"
            if all(row["backend_false_negative_count"] == 0 for row in rows)
            else "FAILED-DIAGNOSTIC-BACKEND-FALSE-NEGATIVES"
        ),
        "asset": {"path": str(asset.resolve()), "sha256": sha256_file(asset)},
        "config_id": config_id,
        "gsplat_metric_semantics": (
            "unique packed fully_fused_projection gaussian_ids; after projection culling, "
            "before tile intersection and pixel rasterization"
        ),
        "compact_metadata": {
            "fields": ["mean_float32[3]", "padded_max_scale_float32"],
            "bytes_per_row": compact.bytes_per_row,
            "logical_bytes": compact.logical_bytes,
            "incremental_bytes": compact.incremental_bytes,
            "gpu_bytes": 0,
        },
        "rows": rows,
        "aggregate": aggregate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-id", default="ALOD-HIER-LOD-FRUSTUM")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = diagnose(args.asset.resolve(), args.config_id, torch.device(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
