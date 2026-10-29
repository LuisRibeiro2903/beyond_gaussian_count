#!/usr/bin/env python3
"""Validate the fully resident production ALoD path against frozen authorities."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_dhier import load_dhier
from alod_reference import HierarchyData, ReferenceCamera, ReferenceConfig, build_hspt
from alod_reference.selection import select_reference
from alod_selection_oracle import load_manifest, sha256_file, validate_oracle_tree
from scene_runtime import FrameCamera, GsplatRenderer, RenderSettings
from scene_runtime.alod.scene import AlodResidentScene
from scene_runtime.alod.types import AlodRuntimeConfig


PASS_STATUS = "PASS-ALOD-PRODUCTION-RESIDENT"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _payload(root: Path, descriptor: dict) -> np.ndarray:
    return np.load(root / descriptor["payload_path"], allow_pickle=False)


def _raw_hash(value: np.ndarray | torch.Tensor) -> str:
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _require_equal(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
        raise RuntimeError(
            f"{label} differs: actual={actual.dtype}{actual.shape}, "
            f"expected={expected.dtype}{expected.shape}, "
            f"actual_hash={_raw_hash(actual)}, expected_hash={_raw_hash(expected)}"
        )


def _config(manifest: dict, oracle_root: Path) -> tuple[AlodRuntimeConfig, ReferenceConfig]:
    hspt_record = _json(oracle_root / manifest["hspt"]["path"])
    raw = hspt_record["configuration"]
    kwargs = {
        "root_global_node_id": int(manifest["asset"]["root_global_node_id"]),
        "skybox_count": int(manifest["asset"]["skybox_rows"]),
        "spt_root_volume": float(raw["SPT_root_volume"]),
        "min_spt_size": int(raw["min_SPT_size_effective_evaluation_default"]),
        "target_granularity": float(raw["resolved_target_granularity"]),
        "distance_multiplier": 1.0,
    }
    return AlodRuntimeConfig(**kwargs), ReferenceConfig(**kwargs)


def _reference_camera(record: dict) -> ReferenceCamera:
    runtime = record["effective_official_runtime_camera"]
    return ReferenceCamera(
        stable_id=record["camera_stable_id"],
        camera_center=np.asarray(runtime["camera_center_float32"], dtype=np.float32),
        full_projection_transform=np.asarray(
            runtime["full_projection_transform"]["values"], dtype=np.float32
        ),
        registered_image_id=int(record["identity"]["registered_image_id"]),
    )


def _frame_camera(record: dict, device: torch.device) -> FrameCamera:
    runtime = record["effective_official_runtime_camera"]
    world_view = np.asarray(runtime["world_view_transform"]["values"], dtype=np.float32)
    w2c_np = np.ascontiguousarray(world_view.T, dtype=np.float32)
    c2w_np = np.linalg.inv(w2c_np.astype(np.float64)).astype(np.float32)
    intrinsics = runtime["intrinsics"]
    K = np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return FrameCamera(
        c2w=torch.from_numpy(c2w_np).to(device),
        K=torch.from_numpy(K).to(device),
        width=int(runtime["image_width"]),
        height=int(runtime["image_height"]),
        w2c=torch.from_numpy(w2c_np).to(device),
        selection_camera_center=torch.tensor(
            runtime["camera_center_float32"], dtype=torch.float32, device=device
        ),
        selection_full_projection_transform=torch.tensor(
            runtime["full_projection_transform"]["values"],
            dtype=torch.float32,
            device=device,
        ),
        camera_stable_id=record["camera_stable_id"],
    )


def _settings(record: dict) -> RenderSettings:
    runtime = record["effective_official_runtime_camera"]
    return RenderSettings(
        width=int(runtime["image_width"]),
        height=int(runtime["image_height"]),
        near_plane=0.01,
        far_plane=1.0e10,
        render_mode="RGB",
        sh_degree=1,
        background=(0.0, 0.0, 0.0),
        packed=True,
        radius_clip=0.0,
        eps2d=0.3,
    )


def _compare_hspt(production, reference) -> dict[str, str]:
    hashes = {}
    for name in (
        "upper_tree_nodes",
        "upper_to_global_node_id",
        "upper_tree_xyz",
        "upper_tree_log_scaling",
        "upper_min_distance_squared",
        "spt_starts",
        "spt_min",
        "spt_max",
        "spt_to_global_node_id",
        "spt_root_global_node_ids",
        "spt_sizes",
    ):
        actual = getattr(production, name)
        _require_equal(f"HSPT {name}", actual, getattr(reference, name))
        hashes[name] = _raw_hash(actual)
    return hashes


def _compare_selection(production, reference, camera_dir: Path) -> None:
    if production.upper is None:
        raise RuntimeError("LoD production selection unexpectedly has no upper result")
    for name in (
        "coarse_cut_upper_local_ids",
        "coarse_cut_global_node_ids",
        "cut_leaf_upper_local_ids",
        "cut_leaf_global_node_ids",
        "upper_render_global_node_ids",
        "candidate_spt_ids",
        "candidate_spt_upper_local_ids",
        "candidate_spt_root_global_node_ids",
        "candidate_spt_distances",
        "frustum_planes",
        "frustum_rejected_upper_local_ids",
    ):
        _require_equal(
            f"selection {name}",
            getattr(production.upper, name),
            getattr(reference.upper, name),
        )
    if len(production.per_spt) != len(reference.per_spt):
        raise RuntimeError("per-SPT result count differs")
    for index, (actual, expected) in enumerate(zip(production.per_spt, reference.per_spt)):
        _require_equal(
            f"SPT {index} ordered cut",
            actual.emitted_global_node_ids,
            expected.emitted_global_node_ids,
        )
    _require_equal(
        "combined ordered selection",
        production.selected_hierarchy_global_node_ids,
        reference.selected_hierarchy_global_node_ids,
    )
    _require_equal(
        "sorted selection",
        np.sort(production.selected_hierarchy_global_node_ids, kind="stable"),
        np.sort(reference.selected_hierarchy_global_node_ids, kind="stable"),
    )
    expected_working = _json(camera_dir / "working-set.json")
    expected_raster = _payload(camera_dir, expected_working["global_node_id_payload"])
    _require_equal("official raster global IDs", production.raster_global_node_ids, expected_raster)


def _pre_raster_hashes(active, camera: FrameCamera, settings: RenderSettings) -> dict[str, str]:
    background = torch.tensor(settings.background, dtype=torch.float32, device=active.means.device)
    return {
        "background": _raw_hash(background),
        "intrinsics": _raw_hash(camera.K[None]),
        "means": _raw_hash(active.means),
        "opacities": _raw_hash(active.opacities),
        "quaternions": _raw_hash(active.quats),
        "scales": _raw_hash(active.scales),
        "sh_features": _raw_hash(active.colors),
        "view_matrices": _raw_hash(camera.w2c[None]),
    }


def validate(
    dhier_path: Path,
    oracle_manifest: Path,
    *,
    device: torch.device,
    render: bool,
) -> dict:
    manifest = load_manifest(oracle_manifest)
    oracle_root = oracle_manifest.resolve().parent
    validate_oracle_tree(oracle_manifest)
    actual_asset_hash = sha256_file(dhier_path)
    if actual_asset_hash != manifest["asset"]["sha256"]:
        raise RuntimeError("Run005 asset hash does not match oracle manifest")
    production_config, reference_config = _config(manifest, oracle_root)

    scene = AlodResidentScene(dhier_path, device, production_config)
    reference_asset = load_dhier(dhier_path)
    reference_build_start = time.perf_counter()
    reference_hspt = build_hspt(
        HierarchyData.from_dhier(reference_asset), reference_config
    )
    reference_build_seconds = time.perf_counter() - reference_build_start
    hspt_hashes = _compare_hspt(scene.hspt, reference_hspt)
    if (scene.hspt.upper_row_count, scene.hspt.spt_count, scene.hspt.flattened_spt_row_count) != (
        1309,
        23,
        998713,
    ):
        raise RuntimeError("Run005 HSPT structural counts differ from validated structure")

    renderer = GsplatRenderer(device) if render else None
    camera_rows = []
    for entry in manifest["cameras"]:
        camera_path = oracle_root / entry["paths"]["camera.json"]
        camera_dir = camera_path.parent
        record = _json(camera_path)
        ref_camera = _reference_camera(record)
        ref_start = time.perf_counter()
        expected = select_reference(reference_hspt, ref_camera, reference_config)
        reference_seconds = time.perf_counter() - ref_start

        frame_camera = _frame_camera(record, device)
        settings = _settings(record)
        active = scene.prepare_frame(frame_camera, settings)
        actual = scene.last_selection
        assert actual is not None
        _compare_selection(actual, expected, camera_dir)

        hashes = _pre_raster_hashes(active, frame_camera, settings)
        expected_hashes = entry["pre_raster_tensor_hashes"]
        mismatches = {
            key: {"actual": value, "expected": expected_hashes[key]}
            for key, value in hashes.items()
            if value != expected_hashes[key]
        }
        if mismatches:
            raise RuntimeError(
                f"pre-raster hashes differ for {entry['camera_stable_id']}: {mismatches}"
            )

        render_record = None
        if renderer is not None:
            result = renderer.render(active, frame_camera, settings, log_gpu_events=True)
            official_record = _json(camera_dir / "optional-image.json")
            official = _payload(camera_dir, official_record["rendered_rgb_chw"])
            actual_image = np.ascontiguousarray(
                result.image.detach().cpu().permute(2, 0, 1).numpy(), dtype=np.float32
            )
            difference = np.abs(actual_image - official)
            render_record = {
                "production_sha256": _raw_hash(actual_image),
                "official_sha256": official_record["rendered_rgb_chw"][
                    "raw_cpu_c_order_sha256"
                ],
                "exact": bool(np.array_equal(actual_image, official)),
                "max_abs_error": float(difference.max()),
                "mean_abs_error": float(difference.mean(dtype=np.float64)),
                "raster_time_ms": result.render_time_ms,
                "gpu_raster_time_ms": result.gpu_render_time_ms,
                "visible_gaussians": result.visible_gaussians,
            }

        metrics = asdict(scene.get_metrics())
        camera_rows.append(
            {
                "camera_stable_id": entry["camera_stable_id"],
                "candidate_spt_count": int(actual.upper.candidate_spt_ids.size),
                "selected_hierarchy_count": int(
                    actual.selected_hierarchy_global_node_ids.size
                ),
                "raster_count": int(actual.raster_global_node_ids.size),
                "reference_selection_ms": reference_seconds * 1000.0,
                "production_selection_ms": actual.timings.total_selection_seconds * 1000.0,
                "gather_ms": metrics["alod_selected_gather_ms"],
                "pre_raster_assembly_ms": metrics["alod_pre_raster_assembly_ms"],
                "pre_raster_hashes": hashes,
                "selection_parity": True,
                "pre_raster_parity": True,
                "render": render_record,
            }
        )

    memory = {
        "hierarchy_metadata_cpu_bytes": scene.hierarchy_metadata_cpu_bytes,
        "hierarchy_metadata_gpu_bytes": scene.resident.global_id_bytes,
        "gaussian_properties_cpu_bytes": scene.resident.serialized_cpu_bytes,
        "gaussian_properties_gpu_bytes": scene.resident.resident_property_bytes,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    if device.type == "cuda":
        memory.update(
            {
                "torch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "torch_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return {
        "status": PASS_STATUS,
        "asset": {"path": str(dhier_path.resolve()), "sha256": actual_asset_hash},
        "oracle_manifest": str(oracle_manifest.resolve()),
        "configuration": asdict(production_config),
        "hspt": {
            "upper_rows": scene.hspt.upper_row_count,
            "spt_count": scene.hspt.spt_count,
            "flattened_spt_rows": scene.hspt.flattened_spt_row_count,
            "production_build_ms": scene.hspt_build_seconds * 1000.0,
            "reference_build_ms": reference_build_seconds * 1000.0,
            "hashes": hspt_hashes,
            "reference_parity": True,
        },
        "resident": True,
        "streaming": False,
        "memory": memory,
        "cameras": camera_rows,
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
