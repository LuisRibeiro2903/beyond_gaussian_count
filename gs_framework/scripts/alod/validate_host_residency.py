#!/usr/bin/env python3
"""Validate ALoD host-property residency against the resident production path."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_selection_oracle import load_manifest, sha256_file, validate_oracle_tree  # noqa: E402
from scene_runtime import GsplatRenderer  # noqa: E402
from scene_runtime.alod.host_scene import AlodHostScene  # noqa: E402
from scene_runtime.alod.residency import AlodBoundedCache  # noqa: E402
from scene_runtime.alod.scene import AlodResidentScene  # noqa: E402
from scene_runtime.alod.types import AlodHostResidencyConfig  # noqa: E402
from benchmarks.benchmark_real_scene import gpu_memory_snapshot  # noqa: E402
from scripts.alod.validate_production_resident import (  # noqa: E402
    _config,
    _frame_camera,
    _json,
    _pre_raster_hashes,
    _raw_hash,
    _settings,
    validate as validate_resident,
)


PASS_STATUS = "PASS-ALOD-HOST-RESIDENCY"
FIRST_CACHE_CAPACITY_ROWS = 525_000
EXPECTED_TRAJECTORY_MAX_ROWS = 499_030
RESIDENT_OFFICIAL_MAX_ABS_LIMIT = 9.7885728e-4
RESIDENT_OFFICIAL_MEAN_ABS_LIMIT = 1.3340947e-8


def _render_numpy(renderer, active, camera, settings) -> tuple[np.ndarray, float, dict[str, int]]:
    before = int(torch.cuda.memory_allocated(active.means.device))
    torch.cuda.reset_peak_memory_stats(active.means.device)
    result = renderer.render(active, camera, settings, log_gpu_events=True)
    peak = int(torch.cuda.max_memory_allocated(active.means.device))
    after = int(torch.cuda.memory_allocated(active.means.device))
    image = np.ascontiguousarray(result.image.detach().cpu().numpy(), dtype=np.float32)
    return image, float(result.visible_gaussians), {
        "torch_before_render_bytes": before,
        "torch_after_render_bytes": after,
        "torch_render_peak_bytes": peak,
        "rasterizer_temporary_peak_increment_bytes": max(0, peak - before),
    }


def _mode_config(mode: str) -> AlodHostResidencyConfig:
    return AlodHostResidencyConfig(
        mode=mode,
        cache_capacity_rows=(
            FIRST_CACHE_CAPACITY_ROWS if mode == "host_bounded_cache" else None
        ),
        activation_chunk_rows=131_072,
        pin_memory=False,
        validate_mappings=True,
        validate_properties=False,
    )


def _metric_record(scene: AlodHostScene, phase: str) -> dict[str, Any]:
    metrics = asdict(scene.get_metrics())
    names = (
        "alod_requested_rows",
        "alod_cache_hit_count",
        "alod_cache_miss_count",
        "alod_cache_eviction_count",
        "alod_cache_occupancy_rows",
        "alod_retained_rows",
        "alod_new_rows",
        "alod_discarded_rows",
        "alod_retention_ratio",
        "alod_selected_set_churn",
        "alod_h2d_rows",
        "alod_h2d_bytes",
        "alod_residency_lookup_ms",
        "alod_host_gather_ms",
        "alod_h2d_transfer_ms",
        "alod_gpu_slot_index_ms",
        "alod_selected_gpu_gather_ms",
        "alod_pre_raster_assembly_ms",
        "alod_frame_local_property_bytes",
        "alod_frame_local_index_bytes",
        "alod_gaussian_properties_gpu_bytes",
        "alod_persistent_skybox_gpu_bytes",
        "alod_dynamic_cache_gpu_bytes",
        "alod_residency_metadata_cpu_bytes",
    )
    return {"phase": phase, **{name: metrics[name] for name in names}}


def _sequence_definitions(records: list[dict], selected: list[np.ndarray]) -> dict[str, list[int]]:
    centers = np.asarray(
        [record["effective_official_runtime_camera"]["camera_center_float32"] for record in records],
        dtype=np.float32,
    )
    scene_center = centers.mean(axis=0)
    distances = np.linalg.norm(centers - scene_center[None], axis=1)
    near_index = int(np.argmin(distances))
    far_index = int(np.argmax(distances))
    best_pair = (0, len(records) - 1)
    best_jaccard = 1.0
    sets = [set(values.tolist()) for values in selected]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            union = len(sets[left] | sets[right])
            jaccard = len(sets[left] & sets[right]) / union if union else 1.0
            if jaccard < best_jaccard:
                best_jaccard = jaccard
                best_pair = (left, right)
    maximum = int(np.argmax([values.size for values in selected]))
    minimum = int(np.argmin([values.size for values in selected]))
    return {
        "same_camera_repeated": [0, 0, 0],
        "slow_adjacent_movement": list(range(len(records))),
        "stationary_camera": [len(records) // 2] * 4,
        "alternating_two_cameras": [0, len(records) - 1] * 3,
        "near_far_alternation": [near_index, far_index] * 3,
        "teleport_low_overlap": [best_pair[0], best_pair[1]] * 3,
        "cache_capacity_stress": [maximum, minimum, *range(len(records) - 1, -1, -1)],
    }


def validate(
    dhier_path: Path,
    oracle_manifest: Path,
    *,
    device: torch.device,
    render: bool,
) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("external host-residency parity requires CUDA")
    manifest = load_manifest(oracle_manifest)
    validate_oracle_tree(oracle_manifest)
    if sha256_file(dhier_path) != manifest["asset"]["sha256"]:
        raise RuntimeError("Run005 asset hash does not match the frozen oracle")

    resident_gate = validate_resident(
        dhier_path, oracle_manifest, device=device, render=False
    )
    if resident_gate["status"] != "PASS-ALOD-PRODUCTION-RESIDENT":
        raise RuntimeError("resident production prerequisite did not pass")
    runtime_config, _ = _config(manifest, oracle_manifest.resolve().parent)
    oracle_root = oracle_manifest.resolve().parent
    records = [
        _json(oracle_root / entry["paths"]["camera.json"])
        for entry in manifest["cameras"]
    ]
    cameras = [_frame_camera(record, device) for record in records]
    settings = [_settings(record) for record in records]
    renderer = GsplatRenderer(device) if render else None

    resident = AlodResidentScene(dhier_path, device, runtime_config)
    resident_rows: list[dict[str, Any]] = []
    resident_selected: list[np.ndarray] = []
    resident_raster: list[np.ndarray] = []
    resident_hashes: list[dict[str, str]] = []
    resident_images: list[np.ndarray | None] = []
    for index, (camera, frame_settings) in enumerate(zip(cameras, settings)):
        active = resident.prepare_frame(camera, frame_settings)
        selection = resident.last_selection
        assert selection is not None
        resident_selected.append(selection.selected_hierarchy_global_node_ids.copy())
        resident_raster.append(selection.raster_global_node_ids.copy())
        hashes = _pre_raster_hashes(active, camera, frame_settings)
        resident_hashes.append(hashes)
        image = None
        visible = None
        if renderer is not None:
            image, visible, _ = _render_numpy(renderer, active, camera, frame_settings)
        resident_images.append(image)
        resident_rows.append(
            {
                "camera_index": index,
                "camera_stable_id": records[index]["camera_stable_id"],
                "selected_hierarchy_rows": int(resident_selected[-1].size),
                "raster_rows": int(resident_raster[-1].size),
                "pre_raster_hashes": hashes,
                "backend_visible_rows": visible,
            }
        )
    maximum_selected = max(int(values.size) for values in resident_selected)
    if maximum_selected != EXPECTED_TRAJECTORY_MAX_ROWS:
        raise RuntimeError(
            f"capacity derivation changed: {maximum_selected} != {EXPECTED_TRAJECTORY_MAX_ROWS}"
        )
    if FIRST_CACHE_CAPACITY_ROWS < maximum_selected:
        raise RuntimeError("chosen bounded-cache capacity cannot hold the largest frame")
    resident_property_bytes = resident.resident.resident_property_bytes
    del active, resident
    torch.cuda.empty_cache()

    sequences = _sequence_definitions(records, resident_selected)
    modes: dict[str, Any] = {}
    for mode in ("host_nocache", "host_previous_reuse", "host_bounded_cache"):
        scene = AlodHostScene(
            dhier_path,
            device,
            runtime_config,
            _mode_config(mode),
        )
        camera_rows = []
        for index, (camera, frame_settings) in enumerate(zip(cameras, settings)):
            active = scene.prepare_frame(camera, frame_settings)
            selection = scene.last_selection
            assert selection is not None
            if not np.array_equal(
                selection.selected_hierarchy_global_node_ids, resident_selected[index]
            ):
                raise RuntimeError(f"{mode} selected global IDs differ on camera {index}")
            if not np.array_equal(selection.raster_global_node_ids, resident_raster[index]):
                raise RuntimeError(f"{mode} raster global IDs differ on camera {index}")
            hashes = _pre_raster_hashes(active, camera, frame_settings)
            if hashes != resident_hashes[index]:
                mismatches = {
                    key: {"host": hashes[key], "resident": resident_hashes[index][key]}
                    for key in hashes
                    if hashes[key] != resident_hashes[index][key]
                }
                raise RuntimeError(f"{mode} pre-raster parity failed: {mismatches}")
            render_record = None
            if renderer is not None:
                host_image, visible, render_memory = _render_numpy(
                    renderer, active, camera, frame_settings
                )
                assert resident_images[index] is not None
                difference = np.abs(host_image - resident_images[index])
                mse = float(np.mean(np.square(difference), dtype=np.float64))
                maximum_error = float(difference.max())
                mean_error = float(difference.mean(dtype=np.float64))
                if (
                    maximum_error > RESIDENT_OFFICIAL_MAX_ABS_LIMIT
                    or mean_error > RESIDENT_OFFICIAL_MEAN_ABS_LIMIT
                ):
                    raise RuntimeError(
                        f"{mode} render parity failed on camera {index}: "
                        f"max={maximum_error}, mean={mean_error}"
                    )
                render_record = {
                    "resident_sha256": _raw_hash(resident_images[index]),
                    "host_sha256": _raw_hash(host_image),
                    "exact": bool(np.array_equal(host_image, resident_images[index])),
                    "psnr_db": None if mse == 0.0 else float(-10.0 * np.log10(mse)),
                    "ssim": 1.0 if mse == 0.0 else None,
                    "mse": mse,
                    "max_abs_error": maximum_error,
                    "mean_abs_error": mean_error,
                    "backend_visible_rows": visible,
                    **render_memory,
                }
            camera_rows.append(
                {
                    "camera_index": index,
                    "camera_stable_id": records[index]["camera_stable_id"],
                    "selection_parity": True,
                    "raster_id_parity": True,
                    "pre_raster_parity": True,
                    "metrics": _metric_record(scene, "frozen_trajectory"),
                    "render": render_record,
                }
            )

        sequence_rows: dict[str, list[dict[str, Any]]] = {}
        modes_to_sequence = mode != "host_nocache"
        selected_sequences = sequences if modes_to_sequence else {
            "same_camera_repeated": sequences["same_camera_repeated"]
        }
        for sequence_name, indices in selected_sequences.items():
            production_residency = None
            if mode == "host_bounded_cache" and sequence_name == "cache_capacity_stress":
                production_residency = scene.residency
                scene.residency = AlodBoundedCache(
                    host=scene.host,
                    device=device,
                    hierarchy_start=runtime_config.skybox_count,
                    validate_mappings=True,
                    capacity_rows=maximum_selected,
                )
            scene.reset_residency()
            rows = []
            for frame_index, camera_index in enumerate(indices):
                active = scene.prepare_frame(cameras[camera_index], settings[camera_index])
                if not np.array_equal(
                    scene.last_selection.raster_global_node_ids,
                    resident_raster[camera_index],
                ):
                    raise RuntimeError(
                        f"{mode}/{sequence_name} changed raster IDs at frame {frame_index}"
                    )
                phase = "cold" if frame_index == 0 else "warm" if frame_index == 1 else "steady"
                history_render = None
                if renderer is not None:
                    history_image, history_visible, _ = _render_numpy(
                        renderer,
                        active,
                        cameras[camera_index],
                        settings[camera_index],
                    )
                    resident_image = resident_images[camera_index]
                    assert resident_image is not None
                    difference = np.abs(history_image - resident_image)
                    mse = float(np.mean(np.square(difference), dtype=np.float64))
                    exact = bool(np.array_equal(history_image, resident_image))
                    resident_visible = resident_rows[camera_index]["backend_visible_rows"]
                    visible_exact = float(history_visible) == float(resident_visible)
                    if not exact or not visible_exact:
                        raise RuntimeError(
                            f"{mode}/{sequence_name} render history dependence at "
                            f"frame {frame_index}: exact={exact}, visible_exact={visible_exact}"
                        )
                    history_render = {
                        "exact": exact,
                        "resident_sha256": _raw_hash(resident_image),
                        "history_sha256": _raw_hash(history_image),
                        "psnr_db": None if mse == 0.0 else float(-10.0 * np.log10(mse)),
                        "ssim": 1.0 if mse == 0.0 else None,
                        "mse": mse,
                        "max_abs_error": float(difference.max()),
                        "mean_abs_error": float(difference.mean(dtype=np.float64)),
                        "backend_visible_rows": float(history_visible),
                        "resident_backend_visible_rows": float(resident_visible),
                        "visible_count_exact": visible_exact,
                    }
                rows.append(
                    {
                        "frame_index": frame_index,
                        "camera_index": camera_index,
                        "history_render": history_render,
                        **_metric_record(scene, phase),
                    }
                )
            sequence_rows[sequence_name] = rows
            if production_residency is not None:
                stress_residency = scene.residency
                scene.residency = production_residency
                del stress_residency
                torch.cuda.empty_cache()
        if mode in {"host_previous_reuse", "host_bounded_cache"}:
            repeat = sequence_rows["same_camera_repeated"]
            if int(repeat[0]["alod_h2d_rows"]) == 0 or int(repeat[1]["alod_h2d_rows"]) != 0:
                raise RuntimeError(f"{mode} cold/warm repeated-camera semantics failed")
        if mode == "host_nocache":
            repeat = sequence_rows["same_camera_repeated"]
            if any(int(row["alod_h2d_rows"]) != int(row["alod_requested_rows"]) for row in repeat):
                raise RuntimeError("host-no-cache unexpectedly reused hierarchy properties")
        if mode == "host_bounded_cache":
            stress = sequence_rows["cache_capacity_stress"]
            if not any(int(row["alod_cache_eviction_count"]) > 0 for row in stress[1:]):
                raise RuntimeError("bounded-cache stress sequence did not exercise eviction")

        stats = scene.get_scene_stats()
        persistent_gpu_properties = int(stats.alod_gaussian_properties_gpu_bytes or 0)
        if not persistent_gpu_properties < resident_property_bytes:
            raise RuntimeError(f"{mode} persistent GPU properties are not budget-controlled")
        gpu_snapshot = gpu_memory_snapshot(device, "result_collection")
        render_temporaries = [
            row["render"]["rasterizer_temporary_peak_increment_bytes"]
            for row in camera_rows
            if row["render"] is not None
        ]
        memory = {
            "host_property_store_bytes": scene.host.host_property_bytes,
            "hierarchy_metadata_cpu_bytes": scene.hierarchy_metadata_cpu_bytes,
            "persistent_skybox_gpu_bytes": scene.skybox_gpu_bytes,
            "dynamic_cache_gpu_bytes": scene.persistent_dynamic_gpu_bytes,
            "persistent_gpu_gaussian_property_bytes": persistent_gpu_properties,
            "residency_metadata_cpu_bytes": scene.residency_metadata_cpu_bytes,
            "residency_metadata_gpu_bytes": 0,
            "torch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "torch_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "rasterizer_temporary_peak_increment_bytes": (
                max(render_temporaries) if render_temporaries else None
            ),
            "whole_device_vram_used_bytes": gpu_snapshot["nvml_device_used_bytes"],
            "nvml_process_used_bytes": gpu_snapshot["nvml_process_used_bytes"],
            "nvml_process_accounting_reliable": gpu_snapshot[
                "nvml_process_accounting_reliable"
            ],
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
        modes[mode] = {
            "configuration": asdict(_mode_config(mode)),
            "cameras": camera_rows,
            "sequences": sequence_rows,
            "memory": memory,
        }
        del active, scene
        torch.cuda.empty_cache()

    return {
        "status": PASS_STATUS,
        "asset": {"path": str(dhier_path.resolve()), "sha256": sha256_file(dhier_path)},
        "oracle_manifest": str(oracle_manifest.resolve()),
        "resident_prerequisite": resident_gate["status"],
        "capacity_derivation": {
            "backend_visible_not_used_for_capacity": True,
            "minimum_feasible_capacity_rows": maximum_selected,
            "chosen_first_cache_capacity_rows": FIRST_CACHE_CAPACITY_ROWS,
            "headroom_rows": FIRST_CACHE_CAPACITY_ROWS - maximum_selected,
            "headroom_fraction": (FIRST_CACHE_CAPACITY_ROWS - maximum_selected)
            / maximum_selected,
            "skybox_rows_separate": runtime_config.skybox_count,
        },
        "resident": {
            "gaussian_property_gpu_bytes": resident_property_bytes,
            "cameras": resident_rows,
        },
        "modes": modes,
        "rendered": render,
        "formal_900_frame_campaign_run": False,
        "future_issue": "selected hierarchy rows greatly exceed backend-visible rows",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dhier", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    result = validate(
        args.dhier,
        args.oracle_manifest,
        device=torch.device(args.device),
        render=not args.skip_render,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
