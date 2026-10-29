#!/usr/bin/env python3
"""Exact preflight for the formal materialization-aware residency campaign."""

from __future__ import annotations

import argparse
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
from scene_runtime.alod import AlodHostResidencyConfig  # noqa: E402
from scene_runtime.alod.host_scene import AlodHostScene  # noqa: E402
from scene_runtime.alod.scene import AlodResidentScene  # noqa: E402
from scene_runtime.renderer import GsplatRenderer  # noqa: E402
from scripts.alod.validate_fast_materialization_filter import (  # noqa: E402
    EXPECTED_ASSET_SHA256,
    _preload_gsplat_extension,
    _projection_ids,
    validate as validate_fast_filter,
)
from scripts.alod.validate_production_resident import _pre_raster_hashes  # noqa: E402


PASS_STATUS = "PASS-ALOD-MATERIALIZATION-AWARE-PREFLIGHT"
MODE_CONFIGS = {
    "previous_reuse": "ALOD-HOST-PREVIOUS-REUSE-FAST-PREMATERIALIZATION",
    "bounded_525k": "ALOD-HOST-BOUNDED-CACHE-FAST-PREMATERIALIZATION",
    "bounded_160k": "ALOD-HOST-BOUNDED-CACHE-REDUCED",
}


def _sequences(selected: list[np.ndarray]) -> dict[str, list[int]]:
    sets = [set(values.tolist()) for values in selected]
    best_pair = (0, len(sets) - 1)
    best_jaccard = 1.0
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            union = len(sets[left] | sets[right])
            score = len(sets[left] & sets[right]) / union if union else 1.0
            if score < best_jaccard:
                best_jaccard = score
                best_pair = (left, right)
    maximum = int(np.argmax([values.size for values in selected]))
    minimum = int(np.argmin([values.size for values in selected]))
    return {
        "same_camera_repeated": [0, 0, 0],
        "slow_adjacent_movement": list(range(len(selected))),
        "stationary_camera": [len(selected) // 2] * 4,
        "alternating_endpoints": [0, len(selected) - 1] * 3,
        "teleport_low_overlap": [best_pair[0], best_pair[1]] * 3,
        "cache_capacity_stress": [maximum, minimum, *range(len(selected) - 1, -1, -1)],
    }


def _host_config(config: dict[str, Any]) -> AlodHostResidencyConfig:
    host = config["alod_host"]
    return AlodHostResidencyConfig(
        mode=str(host["mode"]),
        cache_capacity_rows=host.get("cache_capacity_rows"),
        activation_chunk_rows=int(host.get("activation_chunk_rows", 131072)),
        pin_memory=bool(host.get("pin_memory", False)),
        validate_mappings=True,
        validate_properties=True,
    )


@torch.no_grad()
def validate(
    asset: Path,
    device: torch.device,
    *,
    gsplat_extension: Path | None = None,
) -> dict[str, Any]:
    asset = asset.resolve()
    if sha256_file(asset) != EXPECTED_ASSET_SHA256:
        raise RuntimeError("Run005 hierarchy hash differs from the frozen asset")
    extension = _preload_gsplat_extension(gsplat_extension)

    # Re-run the validated CPU-reference/fast-filter/backend/image gate first.
    fast_gate = validate_fast_filter(
        asset,
        device,
        warmup_repetitions=1,
        timing_repetitions=2,
        gsplat_extension=None,
    )

    base_config = load_config("ALOD-HIER-LOD-FRUSTUM")
    filtered_config = load_config("ALOD-HOST-BOUNDED-CACHE-REDUCED")
    cameras_spec = apply_camera_config(
        load_camera_path(resolve_camera_path(base_config), base_config["trajectory_id"]),
        base_config,
    )
    cameras = [make_frame_camera(spec, device) for spec in cameras_spec.cameras]
    settings = [
        make_render_settings(base_config, spec, sh_degree=int(base_config["sh_degree"]))
        for spec in cameras_spec.cameras
    ]
    renderer = GsplatRenderer(device)
    reference = AlodResidentScene(asset, device, _runtime_config(filtered_config))
    reference_rows: list[dict[str, Any]] = []
    reference_selected: list[np.ndarray] = []
    for index, (camera, frame_settings) in enumerate(zip(cameras, settings)):
        active = reference.prepare_frame(camera, frame_settings)
        rendered = renderer.render(active, camera, frame_settings)
        selection = reference.last_selection
        materialization = reference.last_materialization
        assert selection is not None and materialization is not None
        reference_selected.append(materialization.materialization_global_node_ids.copy())
        reference_rows.append(
            {
                "camera_index": index,
                "semantic_ids": selection.raster_global_node_ids.copy(),
                "materialization_ids": materialization.materialization_global_node_ids.copy(),
                "hashes": _pre_raster_hashes(active, camera, frame_settings),
                "backend_ids": _projection_ids(rendered),
                "image": rendered.image.detach().clone(),
                "alpha": rendered.alpha.detach().clone(),
            }
        )
    histories = _sequences(reference_selected)

    modes: dict[str, Any] = {}
    for label, config_id in MODE_CONFIGS.items():
        config = load_config(config_id)
        runtime = _runtime_config(config)
        if runtime.__dict__ | {"materialization_filter": "none"} != (
            _runtime_config(base_config).__dict__ | {"materialization_filter": "none"}
        ):
            raise RuntimeError(f"{config_id} changed semantic selection settings")
        scene = AlodHostScene(asset, device, runtime, _host_config(config))
        camera_checks = []
        for index, (camera, frame_settings) in enumerate(zip(cameras, settings)):
            active = scene.prepare_frame(camera, frame_settings)
            rendered = renderer.render(active, camera, frame_settings)
            selection = scene.last_selection
            materialization = scene.last_materialization
            assert selection is not None and materialization is not None
            expected = reference_rows[index]
            np.testing.assert_array_equal(selection.raster_global_node_ids, expected["semantic_ids"])
            np.testing.assert_array_equal(
                materialization.materialization_global_node_ids,
                expected["materialization_ids"],
            )
            if _pre_raster_hashes(active, camera, frame_settings) != expected["hashes"]:
                raise RuntimeError(f"{config_id} surviving property/order hashes differ")
            np.testing.assert_array_equal(_projection_ids(rendered), expected["backend_ids"])
            if not torch.equal(rendered.image, expected["image"]):
                raise RuntimeError(f"{config_id} RGB differs at camera {index}")
            if not torch.equal(rendered.alpha, expected["alpha"]):
                raise RuntimeError(f"{config_id} alpha differs at camera {index}")
            camera_checks.append(
                {
                    "camera_index": index,
                    "semantic_rows": int(expected["semantic_ids"].size),
                    "materialization_rows": int(expected["materialization_ids"].size),
                    "backend_survivors": int(expected["backend_ids"].size),
                    "global_node_to_slot_exact": True,
                    "properties_exact": True,
                    "backend_ids_exact": True,
                    "rgb_exact": True,
                    "alpha_exact": True,
                }
            )

        history_checks = []
        for history_name, sequence in histories.items():
            for step, index in enumerate(sequence):
                active = scene.prepare_frame(cameras[index], settings[index])
                rendered = renderer.render(active, cameras[index], settings[index])
                expected = reference_rows[index]
                if _pre_raster_hashes(active, cameras[index], settings[index]) != expected["hashes"]:
                    raise RuntimeError(f"{config_id} stale-slot/history property failure")
                np.testing.assert_array_equal(_projection_ids(rendered), expected["backend_ids"])
                if not torch.equal(rendered.image, expected["image"]) or not torch.equal(
                    rendered.alpha, expected["alpha"]
                ):
                    raise RuntimeError(f"{config_id} cache history changed rendering")
                history_checks.append(
                    {"sequence": history_name, "step": step, "camera_index": index, "exact": True}
                )
        modes[label] = {
            "configuration_id": config_id,
            "camera_checks": camera_checks,
            "history_checks": history_checks,
            "capacity_rows": config["alod_host"].get("cache_capacity_rows"),
        }

    return {
        "schema_version": 1,
        "status": PASS_STATUS,
        "asset": {"path": str(asset), "sha256": sha256_file(asset)},
        "trajectory_id": base_config["trajectory_id"],
        "camera_count": len(cameras),
        "fast_filter_gate": fast_gate,
        "semantic_selected_ids_exact": True,
        "fast_materialization_ids_cpu_reference_exact": True,
        "backend_false_negatives_zero": True,
        "surviving_properties_bitwise_exact": True,
        "backend_projection_ids_exact": True,
        "rgb_bitwise_exact": True,
        "alpha_bitwise_exact": True,
        "global_node_to_gpu_slot_exact": True,
        "no_stale_slot_reads": True,
        "cache_history_independent": True,
        "modes": modes,
        "gsplat_extension": extension,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gsplat-extension", type=Path)
    args = parser.parse_args()
    result = validate(
        args.asset,
        torch.device(args.device),
        gsplat_extension=args.gsplat_extension,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
