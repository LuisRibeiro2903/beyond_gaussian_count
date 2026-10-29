#!/usr/bin/env python3
"""Validate conservative ALoD pre-materialization filtering end to end."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


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
from scene_runtime.alod.host_scene import AlodHostScene  # noqa: E402
from scene_runtime.alod.scene import AlodResidentScene  # noqa: E402
from scene_runtime.alod.types import AlodHostResidencyConfig  # noqa: E402
from scene_runtime.renderer import GsplatRenderer  # noqa: E402


def _assert_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual, expected):
        difference = torch.max(torch.abs(actual.float() - expected.float())).item()
        raise RuntimeError(f"{name} differs; maximum absolute difference {difference}")


def _projection_ids(rendered) -> np.ndarray:
    values = rendered.info.get("backend_projection_survivor_global_node_ids")
    if values is None:
        raise RuntimeError("renderer did not expose backend projection survivor IDs")
    return np.ascontiguousarray(
        values.detach().to(device="cpu", dtype=torch.int64).numpy(), dtype=np.int64
    )


@torch.no_grad()
def validate(asset: Path, device: torch.device) -> dict[str, Any]:
    base_config = load_config("ALOD-HIER-LOD-FRUSTUM")
    filtered_config = load_config("ALOD-HOST-BOUNDED-CACHE-PREMATERIALIZATION")
    cameras = apply_camera_config(
        load_camera_path(
            resolve_camera_path(base_config), base_config.get("trajectory_id")
        ),
        base_config,
    )
    base_runtime = _runtime_config(base_config)
    filtered_runtime = replace(
        base_runtime, materialization_filter="conservative_pinhole"
    )
    baseline = AlodResidentScene(asset, device, base_runtime)
    filtered_resident = AlodResidentScene(asset, device, filtered_runtime)
    filtered_host = AlodHostScene(
        asset,
        device,
        filtered_runtime,
        AlodHostResidencyConfig(
            mode="host_bounded_cache",
            cache_capacity_rows=int(
                filtered_config["alod_host"]["cache_capacity_rows"]
            ),
            validate_mappings=True,
            validate_properties=True,
        ),
    )
    renderer = GsplatRenderer(device)

    rows = []
    semantic_hash_baseline: list[bytes] = []
    for camera_index, camera_spec in enumerate(cameras.cameras):
        camera = make_frame_camera(camera_spec, device)
        settings = make_render_settings(
            base_config, camera_spec, sh_degree=int(base_config["sh_degree"])
        )

        baseline_active = baseline.prepare_frame(camera, settings)
        baseline_render = renderer.render(baseline_active, camera, settings)
        filtered_active = filtered_resident.prepare_frame(camera, settings)
        filtered_render = renderer.render(filtered_active, camera, settings)
        host_active = filtered_host.prepare_frame(camera, settings)
        host_render = renderer.render(host_active, camera, settings)

        baseline_selection = baseline.last_selection
        filtered_selection = filtered_resident.last_selection
        host_selection = filtered_host.last_selection
        materialization = filtered_resident.last_materialization
        host_materialization = filtered_host.last_materialization
        assert baseline_selection is not None
        assert filtered_selection is not None
        assert host_selection is not None
        assert materialization is not None
        assert host_materialization is not None
        np.testing.assert_array_equal(
            baseline_selection.selected_hierarchy_global_node_ids,
            filtered_selection.selected_hierarchy_global_node_ids,
        )
        np.testing.assert_array_equal(
            baseline_selection.raster_global_node_ids,
            filtered_selection.raster_global_node_ids,
        )
        np.testing.assert_array_equal(
            filtered_selection.raster_global_node_ids,
            host_selection.raster_global_node_ids,
        )
        np.testing.assert_array_equal(
            materialization.materialization_global_node_ids,
            host_materialization.materialization_global_node_ids,
        )

        kept = torch.from_numpy(materialization.kept_semantic_indices).to(device)
        for field in ("means", "quats", "scales", "opacities", "colors"):
            _assert_equal(
                f"resident surviving {field}",
                getattr(filtered_active, field),
                getattr(baseline_active, field).index_select(0, kept),
            )
            _assert_equal(
                f"host surviving {field}",
                getattr(host_active, field),
                getattr(filtered_active, field),
            )
        _assert_equal("filtered image", filtered_render.image, baseline_render.image)
        _assert_equal("filtered alpha", filtered_render.alpha, baseline_render.alpha)
        _assert_equal("host image", host_render.image, baseline_render.image)
        _assert_equal("host alpha", host_render.alpha, baseline_render.alpha)

        baseline_backend_ids = _projection_ids(baseline_render)
        filtered_backend_ids = _projection_ids(filtered_render)
        host_backend_ids = _projection_ids(host_render)
        np.testing.assert_array_equal(filtered_backend_ids, baseline_backend_ids)
        np.testing.assert_array_equal(host_backend_ids, baseline_backend_ids)
        false_negatives = np.setdiff1d(
            baseline_backend_ids,
            materialization.materialization_global_node_ids,
        )
        if false_negatives.size:
            raise RuntimeError("materialization filter removed backend projection survivors")

        semantic_hash_baseline.append(
            baseline_selection.raster_global_node_ids.tobytes()
        )
        rows.append(
            {
                "camera_index": camera_index,
                "camera_id": camera_spec.camera_id,
                "semantic_selected_hierarchy_rows": int(
                    baseline_selection.selected_hierarchy_global_node_ids.size
                ),
                "semantic_selected_rows_including_skybox": int(
                    baseline_selection.raster_global_node_ids.size
                ),
                "materialization_rows": materialization.materialization_rows,
                "backend_projection_survivors": int(baseline_backend_ids.size),
                "backend_false_negatives": int(false_negatives.size),
                "resident_image_bitwise_equal": True,
                "host_image_bitwise_equal": True,
                "surviving_properties_bitwise_equal": True,
                "semantic_selected_ids_unchanged": True,
                "filter_ms": materialization.filter_seconds * 1000.0,
                "host_requested_hierarchy_rows": int(
                    filtered_host.last_residency_frame.requested_rows
                ),
                "host_h2d_rows": int(filtered_host.last_residency_frame.h2d_rows),
                "host_h2d_bytes": int(filtered_host.last_residency_frame.h2d_bytes),
                "host_frame_local_property_bytes": int(
                    host_active.scene_metrics.alod_frame_local_property_bytes
                ),
            }
        )

    return {
        "schema_version": 1,
        "status": "PASS-ALOD-PREMATERIALIZATION-CULLING",
        "asset": {"path": str(asset.resolve()), "sha256": sha256_file(asset)},
        "camera_count": len(rows),
        "semantic_selection_implementation_unchanged": True,
        "cache_capacity_rows_unchanged": int(
            filtered_config["alod_host"]["cache_capacity_rows"]
        ),
        "all_semantic_selected_ids_unchanged": True,
        "all_backend_false_negatives_zero": True,
        "all_images_bitwise_equal": True,
        "all_surviving_properties_bitwise_equal": True,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = validate(args.asset.resolve(), torch.device(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
