#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_runtime import FrameCamera, GsplatRenderer, RenderSettings, SceneLoader


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
CONFIG_FILES = {
    "FLAT-REAL-FIT": CONFIG_DIR / "flat_real_fit.json",
    "FLAT-CITYGS-FULL": CONFIG_DIR / "flat_citygs_full.json",
    "CG-FINEST-FLAT": CONFIG_DIR / "cg_finest_flat.json",
    "CG-FIXED-FINEST-ALL": CONFIG_DIR / "cg_fixed_finest_all.json",
    "CG-FIXED-FINEST-VISIBILITY": CONFIG_DIR / "cg_fixed_finest_visibility.json",
    "CG-LOD-RELEASE": CONFIG_DIR / "cg_lod_release.json",
    "CG-FUSED-CELLS-ALL": CONFIG_DIR / "cg_fused_cells_all.json",
    "CG-FUSED-VISIBILITY": CONFIG_DIR / "cg_fused_visibility.json",
}
FRAME_JSONL = "frames.jsonl"
SUMMARY_JSON = "summary.json"
RESOLVED_CONFIG_JSON = "resolved_config.json"
FEASIBILITY_JSON = "feasibility_result.json"
CAMERA_SAMPLE_JSON = "camera_sample_summary.json"
FEASIBILITY_STAGES = (
    "asset_validation",
    "ply_parsing",
    "cpu_tensor_construction",
    "bounds_or_metadata_computation",
    "gpu_transfer",
    "gpu_scene_construction",
    "rasterizer_initialization",
    "active_set_preparation",
    "rasterization",
    "result_collection",
)
MEMORY_STAGES = (
    "process_start",
    "after_cuda_initialization",
    "before_asset_load",
    "after_cpu_asset_load",
    "before_gpu_transfer",
    "after_gpu_transfer",
    "before_rasterization",
    "after_rasterization",
    "result_collection",
)


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    c2w: np.ndarray
    K: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class CameraPath:
    path: Path
    trajectory_id: str
    cameras: list[CameraSpec]
    metadata: dict[str, Any]


class MissingAssetError(ValueError):
    pass


def load_config(config_id: str) -> dict[str, Any]:
    if config_id not in CONFIG_FILES:
        supported = ", ".join(sorted(CONFIG_FILES))
        raise ValueError(f"Unknown configuration ID {config_id!r}; expected one of: {supported}")
    config = json.loads(CONFIG_FILES[config_id].read_text())
    if config.get("configuration_id") != config_id:
        raise ValueError(f"Config file for {config_id} has mismatched configuration_id.")
    return config


def resolve_asset_path(config: dict[str, Any], asset_arg: str | None) -> tuple[Path, str]:
    supplied = asset_arg
    if supplied is None:
        env_var = config.get("asset_env_var")
        supplied = os.environ.get(env_var, "") if env_var else ""
    if not supplied:
        raise MissingAssetError(
            f"Pass --asset or set {config.get('asset_env_var')} for {config['configuration_id']}."
        )
    path = Path(supplied).expanduser()
    if not path.exists():
        raise MissingAssetError(f"Asset path does not exist: {supplied}")
    return path, supplied


def resolve_lod_asset_paths(
    config: dict[str, Any],
    finest_arg: str | None,
    middle_arg: str | None,
    coarsest_arg: str | None,
) -> tuple[list[Path], list[str]]:
    args = [finest_arg, middle_arg, coarsest_arg]
    roles = ["finest", "middle", "coarsest"]
    env_vars = config.get("asset_env_vars", {})
    supplied = [value or os.environ.get(str(env_vars.get(role, "")), "") for role, value in zip(roles, args)]
    missing = [role for role, value in zip(roles, supplied) if not value]
    if missing:
        raise MissingAssetError(
            "Pass all three --lod-*-asset paths or set the configured environment variables; "
            f"missing: {', '.join(missing)}."
        )
    paths = [Path(value).expanduser() for value in supplied]
    absent = [str(path) for path in paths if not path.exists()]
    if absent:
        raise MissingAssetError(f"LoD asset path(s) do not exist: {', '.join(absent)}")
    return paths, [str(value) for value in supplied]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_camera_path(config: dict[str, Any], override: Path | None = None) -> Path:
    if override is not None:
        return override
    config_path = CONFIG_FILES[config["configuration_id"]]
    return (config_path.parent / config["camera_path"]).resolve()


def _parse_matrix(record: dict[str, Any], keys: Iterable[str], shape: tuple[int, int]) -> np.ndarray | None:
    for key in keys:
        if key in record:
            return np.asarray(record[key], dtype=np.float32).reshape(shape)
    return None


def _vertical_fov_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = 0.5 * float(height) / math.tan(math.radians(fov_deg) * 0.5)
    return np.asarray(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _horizontal_fov_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = 0.5 * float(width) / math.tan(math.radians(fov_deg) * 0.5)
    return np.asarray(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _viser_export_to_render_c2w(c2w: np.ndarray, scale_ratio: float) -> np.ndarray:
    converted = c2w.copy()
    converted[:3, 1:3] *= -1.0
    converted[:3, 3] *= float(scale_ratio)
    return converted.astype(np.float32)


def load_camera_path(path: Path, trajectory_id: str | None = None) -> CameraPath:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "camera_path" in data:
        return _load_viser_camera_path(Path(path), data, trajectory_id)
    return _load_generic_camera_path(Path(path), data, trajectory_id)


def _load_viser_camera_path(path: Path, data: dict[str, Any], trajectory_id: str | None) -> CameraPath:
    records = data.get("camera_path")
    if not isinstance(records, list) or not records:
        raise ValueError("Viser camera path must contain a non-empty 'camera_path' list.")

    width = int(data.get("render_width", data.get("width", 1920)))
    height = int(data.get("render_height", data.get("height", 1080)))
    default_fov = float(data.get("default_fov", data.get("fov_deg", 50.0)))
    scale_ratio = float(data.get("scale_ratio", 10.0))

    cameras = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Camera path record {idx} must be an object.")
        c2w = _parse_matrix(record, ["camera_to_world", "camtoworld", "c2w"], (4, 4))
        if c2w is None:
            raise ValueError(f"Camera path record {idx} has no camera_to_world matrix.")
        frame_width = int(record.get("width", width))
        frame_height = int(record.get("height", height))
        fov_deg = float(record.get("fov", default_fov))
        cameras.append(
            CameraSpec(
                camera_id=f"viser_{idx:06d}",
                c2w=_viser_export_to_render_c2w(c2w, scale_ratio),
                K=_vertical_fov_intrinsics(frame_width, frame_height, fov_deg),
                width=frame_width,
                height=frame_height,
            )
        )

    return CameraPath(
        path=path,
        trajectory_id=trajectory_id or path.stem,
        cameras=cameras,
        metadata={
            "source_file": str(path),
            "source_format": "viser_camera_path",
            "frame_count": len(cameras),
            "camera_matrix_convention": (
                "source camera_to_world is a Viser/nerfview export; runtime conversion "
                "flips camera Y/Z axes and multiplies translation by scale_ratio"
            ),
            "conversion_required": True,
            "scale_ratio": scale_ratio,
            "projection": "vertical_fov_to_pinhole_intrinsics",
            "default_fov_degrees": default_fov,
            "resolution": [width, height],
            "coordinate_system": "scene-compatible after deterministic Viser export conversion",
            "interpolation_already_exists": True,
            "keyframe_count": len(data.get("keyframes", [])) if isinstance(data.get("keyframes"), list) else 0,
            "is_cycle": bool(data.get("is_cycle", False)),
            "seconds": float(data.get("seconds", 0.0)),
            "fps": float(data.get("fps", 0.0)),
        },
    )


def _load_generic_camera_path(path: Path, data: Any, trajectory_id: str | None) -> CameraPath:
    if isinstance(data, list):
        records = data
        width = 1920
        height = 1080
        global_K = None
        fov_deg = 60.0
    elif isinstance(data, dict):
        records = data.get("cameras", data.get("frames"))
        width = int(data.get("width", 1920))
        height = int(data.get("height", 1080))
        global_K = _parse_matrix(data, ["K", "intrinsics"], (3, 3))
        fov_deg = float(data.get("fov_deg", 60.0))
    else:
        records = None
    if not isinstance(records, list) or not records:
        raise ValueError("Camera path JSON must be a non-empty list or contain cameras/frames.")

    cameras = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Camera path record {idx} must be an object.")
        c2w = _parse_matrix(record, ["camtoworld", "camera_to_world", "c2w", "transform_matrix"], (4, 4))
        if c2w is None:
            w2c = _parse_matrix(record, ["world_to_camera", "w2c", "viewmat", "view_matrix"], (4, 4))
            if w2c is None:
                raise ValueError(f"Camera path record {idx} has no c2w/camtoworld or w2c matrix.")
            c2w = np.linalg.inv(w2c).astype(np.float32)
        frame_width = int(record.get("width", width))
        frame_height = int(record.get("height", height))
        K = _parse_matrix(record, ["K", "intrinsics"], (3, 3))
        if K is None:
            K = global_K.copy() if global_K is not None else _horizontal_fov_intrinsics(
                frame_width, frame_height, fov_deg
            )
        cameras.append(
            CameraSpec(
                camera_id=str(record.get("name", f"json_{idx:06d}")),
                c2w=c2w.astype(np.float32),
                K=K.astype(np.float32),
                width=frame_width,
                height=frame_height,
            )
        )
    return CameraPath(
        path=path,
        trajectory_id=trajectory_id or path.stem,
        cameras=cameras,
        metadata={
            "source_file": str(path),
            "source_format": "generic_json_camera_path",
            "frame_count": len(cameras),
            "camera_matrix_convention": "c2w/camtoworld or inverted w2c matrix",
            "conversion_required": False,
            "projection": "record_K_or_global_K_or_horizontal_fov",
            "resolution": [width, height],
            "coordinate_system": "as supplied by camera path",
            "interpolation_already_exists": False,
        },
    )


def inspect_asset(path: Path, supplied_path: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "path_supplied": supplied_path,
        "path_resolved": str(path.resolve()),
        "identifier": path.name,
        "checkpoint_type": config.get("checkpoint_type"),
        "asset_role": config.get("asset_role"),
        "training_source": config.get("training_source"),
        "file_size_bytes": path.stat().st_size if path.exists() else None,
        "ply_vertex_count": read_ply_vertex_count(path) if path.suffix.lower() == ".ply" else None,
    }


def inspect_benchmark_asset(
    path: Path | list[Path],
    supplied_path: str | list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(path, list):
        if not isinstance(supplied_path, list) or len(path) != len(supplied_path):
            raise ValueError("LoD supplied and resolved asset lists must have equal lengths.")
        labels = list(config.get("lod_level_order", []))
        assets = []
        for index, (item, supplied) in enumerate(zip(path, supplied_path)):
            row = inspect_asset(item, supplied, {**config, "checkpoint_type": "gaussian_ply"})
            row["level_index"] = index
            row["level_label"] = labels[index] if index < len(labels) else str(index)
            assets.append(row)
        return {
            "path_supplied": supplied_path,
            "path_resolved": [str(item.resolve()) for item in path],
            "identifier": "+".join(str(item["identifier"]) for item in assets),
            "checkpoint_type": config.get("checkpoint_type"),
            "asset_role": config.get("asset_role"),
            "training_source": config.get("training_source"),
            "file_size_bytes": sum(int(item["file_size_bytes"] or 0) for item in assets),
            "ply_vertex_count": sum(int(item["ply_vertex_count"] or 0) for item in assets),
            "assets": assets,
        }
    if isinstance(supplied_path, list):
        raise ValueError("Single asset path cannot have a supplied path list.")
    return inspect_asset(path, supplied_path, config)


def read_ply_vertex_count(path: Path) -> int | None:
    try:
        with Path(path).open("rb") as handle:
            for raw_line in handle:
                line = raw_line.decode("ascii", errors="replace").strip()
                if line.startswith("element vertex "):
                    return int(line.split()[-1])
                if line == "end_header":
                    return None
    except UnicodeDecodeError:
        return None
    return None


def require_cuda(device_index: int = 0) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real-scene gsplat benchmarks.")
    if torch.cuda.device_count() <= device_index:
        raise RuntimeError(f"CUDA device index {device_index} is unavailable.")
    return torch.device("cuda", device_index)


def load_flat_scene_for_benchmark(asset_path: Path, device: torch.device, progress_callback=None):
    scene = SceneLoader.load(
        [asset_path],
        device=device,
        scene_type="flat",
        progress_callback=progress_callback,
    )
    stats = scene.get_scene_stats()
    if stats.scene_type != "flat":
        raise ValueError(f"Expected a flat scene, got {stats.scene_type!r}.")
    return scene, stats


def load_scene_for_benchmark(
    config: dict[str, Any],
    asset_path: Path | list[Path],
    device: torch.device,
    progress_callback=None,
):
    scene_type = str(config.get("scene_type", "flat"))
    if scene_type == "flat":
        if isinstance(asset_path, list):
            raise ValueError("Flat benchmark requires one asset path.")
        return load_flat_scene_for_benchmark(asset_path, device, progress_callback=progress_callback)
    if scene_type == "citygaussian_fixed_all":
        if isinstance(asset_path, list):
            raise ValueError("CityGaussian fixed-all benchmark requires one asset path.")
        scene = SceneLoader.load(
            [asset_path],
            device=device,
            scene_type="citygaussian_fixed_all",
            citygaussian_block_dim=config.get("block_dim", [6, 6, 1]),
            citygaussian_aabb=config.get("aabb", [-3.5, -4.0, -10.0, 4.5, 2.0, 10.0]),
            citygaussian_lod_source_level=str(config.get("lod_source_level", "light_50_vq")),
            progress_callback=progress_callback,
        )
        stats = scene.get_scene_stats()
        if stats.scene_type != "citygaussian_fixed_all":
            raise ValueError(f"Expected a CityGaussian fixed-all scene, got {stats.scene_type!r}.")
        return scene, stats
    if scene_type == "citygaussian_visibility":
        if isinstance(asset_path, list):
            raise ValueError("CityGaussian visibility benchmark requires one asset path.")
        scene = SceneLoader.load(
            [asset_path],
            device=device,
            scene_type="citygaussian_visibility",
            citygaussian_block_dim=config.get("block_dim", [6, 6, 1]),
            citygaussian_aabb=config.get("aabb", [-3.5, -4.0, -10.0, 4.5, 2.0, 10.0]),
            citygaussian_lod_source_level=str(config.get("lod_source_level", "light_50_vq")),
            progress_callback=progress_callback,
        )
        stats = scene.get_scene_stats()
        if stats.scene_type != "citygaussian_visibility":
            raise ValueError(f"Expected a CityGaussian visibility scene, got {stats.scene_type!r}.")
        return scene, stats
    if scene_type == "citygaussian_lod_release":
        if not isinstance(asset_path, list):
            raise ValueError("CityGaussian released-LoD benchmark requires three asset paths.")
        scene = SceneLoader.load(
            asset_path,
            device=device,
            scene_type="citygaussian_lod_release",
            citygaussian_block_dim=config.get("block_dim", [6, 6, 1]),
            citygaussian_aabb=config.get("aabb", [-3.5, -4.0, -10.0, 4.5, 2.0, 10.0]),
            citygaussian_lod_level_labels=config.get("lod_level_order", []),
            citygaussian_lod_level_semantics=config.get("lod_level_semantics", []),
            progress_callback=progress_callback,
        )
        stats = scene.get_scene_stats()
        if stats.scene_type != "citygaussian_lod_release":
            raise ValueError(f"Expected a CityGaussian released-LoD scene, got {stats.scene_type!r}.")
        return scene, stats
    raise ValueError(f"Unsupported benchmark scene_type {scene_type!r}.")


def requires_all_loaded_gaussians_active(config: dict[str, Any]) -> bool:
    return str(config.get("scene_type", "flat")) in {"flat", "citygaussian_fixed_all"}


def make_frame_camera(camera: CameraSpec, device: torch.device) -> FrameCamera:
    return FrameCamera(
        c2w=torch.from_numpy(camera.c2w).to(device=device, dtype=torch.float32),
        K=torch.from_numpy(camera.K).to(device=device, dtype=torch.float32),
        width=camera.width,
        height=camera.height,
    )


def make_render_settings(config: dict[str, Any], camera: CameraSpec, sh_degree: int | None = None) -> RenderSettings:
    return RenderSettings(
        width=camera.width,
        height=camera.height,
        near_plane=float(config["near_plane"]),
        far_plane=float(config["far_plane"]),
        render_mode=str(config.get("render_mode", "RGB")),
        sh_degree=sh_degree,
        background=tuple(float(v) for v in config.get("background", [0.0, 0.0, 0.0])),
        packed=bool(config.get("packed", False)),
        rasterize_mode=str(config.get("rasterize_mode", "classic")),
        camera_model=str(config.get("camera_model", "pinhole")),
        radius_clip=float(config.get("radius_clip", 0.0)),
        eps2d=float(config.get("eps2d", 0.3)),
    )


@torch.no_grad()
def render_one_frame(
    scene,
    renderer: GsplatRenderer,
    camera: CameraSpec,
    config: dict[str, Any],
    scene_total_gaussians: int,
    scene_sh_degree: int | None,
    device: torch.device,
    log_gpu_events: bool,
    stage_callback=None,
    reset_render_peak: bool = True,
) -> dict[str, Any]:
    frame_camera = make_frame_camera(camera, device)
    sh_degree = min(int(config.get("sh_degree", 3)), scene_sh_degree) if scene_sh_degree is not None else None
    settings = make_render_settings(config, camera, sh_degree=sh_degree)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if stage_callback is not None:
        stage_callback("active_set_preparation")
    prep_start = time.perf_counter()
    active_splats = scene.prepare_frame(frame_camera, settings)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preparation_time_ms = (time.perf_counter() - prep_start) * 1000.0

    scene_metrics = active_splats.scene_metrics
    if requires_all_loaded_gaussians_active(config) and active_splats.num_gaussians != int(scene_total_gaussians):
        raise RuntimeError(
            "Benchmark integrity violation: active Gaussian count does not equal total "
            f"({active_splats.num_gaussians} != {scene_total_gaussians})."
        )
    if not requires_all_loaded_gaussians_active(config):
        selected_count = int(scene_metrics.selected_gaussian_count)
        if selected_count != int(active_splats.num_gaussians):
            raise RuntimeError(
                "Benchmark integrity violation: selected Gaussian count does not equal active "
                f"({selected_count} != {active_splats.num_gaussians})."
            )
    if stage_callback is not None:
        stage_callback("rasterization")
    before_render_memory = gpu_memory_snapshot(device, "before_rasterization")
    if reset_render_peak:
        reset_torch_peak_memory(device)
    result = renderer.render(
        active_splats,
        frame_camera,
        settings,
        log_gpu_events=log_gpu_events,
    )
    if stage_callback is not None:
        stage_callback("result_collection")
    after_render_memory = gpu_memory_snapshot(device, "after_rasterization")
    finite_image = bool(torch.isfinite(result.image).all().item())
    finite_alpha = bool(torch.isfinite(result.alpha).all().item()) if result.alpha is not None else None
    memory = gpu_memory_snapshot(device, "result_collection")
    cpu = cpu_memory_snapshot()
    render_call_time_ms = float(result.render_time_ms)
    cuda_rasterization_time_ms = float(result.gpu_render_time_ms)
    end_to_end_frame_time_ms = preparation_time_ms + render_call_time_ms
    resident_gaussian_count = (
        int(scene_metrics.resident_gaussians)
        if math.isfinite(float(scene_metrics.resident_gaussians))
        else int(scene_total_gaussians)
    )
    return {
        "finite_image": finite_image,
        "finite_alpha": finite_alpha,
        "image_shape": tuple(int(v) for v in result.image.shape),
        "alpha_shape": tuple(int(v) for v in result.alpha.shape) if result.alpha is not None else None,
        "preparation_time_ms": preparation_time_ms,
        "render_call_time_ms": render_call_time_ms,
        "render_call_fps_equivalent": 1000.0 / max(render_call_time_ms, 1e-9),
        "cuda_rasterization_time_ms": cuda_rasterization_time_ms,
        "end_to_end_frame_time_ms": end_to_end_frame_time_ms,
        "end_to_end_fps": 1000.0 / max(end_to_end_frame_time_ms, 1e-9),
        "total_gaussian_count": int(scene_total_gaussians),
        "active_gaussian_count": int(active_splats.num_gaussians),
        "active_gaussian_fraction": float(scene_metrics.active_gaussian_fraction),
        "active_blocks": float(scene_metrics.active_blocks),
        "active_block_ids": scene_metrics.active_block_ids,
        "total_cell_count": float(scene_metrics.total_cell_count),
        "active_cell_count": float(scene_metrics.active_cell_count),
        "selected_cell_count": float(scene_metrics.selected_cell_count),
        "selected_cell_ids": scene_metrics.selected_cell_ids,
        "selected_gaussian_count": float(scene_metrics.selected_gaussian_count),
        "selected_lod_by_cell": scene_metrics.selected_lod_by_cell,
        "selected_level_label_by_cell": scene_metrics.selected_level_label_by_cell,
        "selected_gaussian_count_by_cell": scene_metrics.selected_gaussian_count_by_cell,
        "selected_cell_count_by_level": scene_metrics.selected_cell_count_by_level,
        "selected_gaussian_count_by_level": scene_metrics.selected_gaussian_count_by_level,
        "resident_gaussian_count": resident_gaussian_count,
        "all_loaded_gaussians_resident": resident_gaussian_count == int(scene_total_gaussians),
        "all_loaded_gaussians_active": int(active_splats.num_gaussians) == int(scene_total_gaussians),
        "visibility_selection_ms": float(scene_metrics.visibility_selection_ms),
        "lod_selection_ms": float(scene_metrics.lod_selection_ms),
        "selected_cell_assembly_ms": float(scene_metrics.selected_cell_assembly_ms),
        "nearest_cell_id": float(scene_metrics.nearest_cell_id),
        "nearest_cell_fallback_changed_mask": scene_metrics.nearest_cell_fallback_changed_mask,
        "cell_construction_ms": float(scene_metrics.cell_construction_ms),
        "frame_preparation_ms": float(scene_metrics.frame_preparation_ms),
        "block_concat_ms": float(scene_metrics.block_concat_ms),
        "citygaussian_no_visibility_selection": scene_metrics.citygaussian_no_visibility_selection,
        "citygaussian_visibility_selection_used": scene_metrics.citygaussian_visibility_selection_used,
        "citygaussian_no_adaptive_lod": scene_metrics.citygaussian_no_adaptive_lod,
        "citygaussian_adaptive_lod_used": scene_metrics.citygaussian_adaptive_lod_used,
        "citygaussian_lod_policy": scene_metrics.citygaussian_lod_policy,
        "citygaussian_visibility_bounds_level": scene_metrics.citygaussian_visibility_bounds_level,
        "citygaussian_no_vq_reconstruction": scene_metrics.citygaussian_no_vq_reconstruction,
        "citygaussian_no_streaming": scene_metrics.citygaussian_no_streaming,
        "visible_gaussians": float(result.visible_gaussians),
        "rendered_gaussian_count": float(result.visible_gaussians),
        "measurement_stage": memory["measurement_stage"],
        "active_device": memory["active_device"],
        "active_device_index": memory["active_device_index"],
        "allocator_backend": memory["allocator_backend"],
        "torch_current_allocated_bytes": memory["torch_current_allocated_bytes"],
        "torch_current_reserved_bytes": memory["torch_current_reserved_bytes"],
        "torch_peak_allocated_bytes": memory["torch_peak_allocated_bytes"],
        "torch_peak_reserved_bytes": memory["torch_peak_reserved_bytes"],
        "torch_allocated_bytes": memory["torch_current_allocated_bytes"],
        "torch_reserved_bytes": memory["torch_current_reserved_bytes"],
        "before_rasterization_torch_current_allocated_bytes": before_render_memory[
            "torch_current_allocated_bytes"
        ],
        "before_rasterization_torch_current_reserved_bytes": before_render_memory[
            "torch_current_reserved_bytes"
        ],
        "before_rasterization_nvml_device_used_bytes": before_render_memory[
            "nvml_device_used_bytes"
        ],
        "before_rasterization_nvml_process_used_bytes": before_render_memory[
            "nvml_process_used_bytes"
        ],
        "after_rasterization_torch_current_allocated_bytes": after_render_memory[
            "torch_current_allocated_bytes"
        ],
        "after_rasterization_torch_current_reserved_bytes": after_render_memory[
            "torch_current_reserved_bytes"
        ],
        "after_rasterization_nvml_device_used_bytes": after_render_memory[
            "nvml_device_used_bytes"
        ],
        "after_rasterization_nvml_process_used_bytes": after_render_memory[
            "nvml_process_used_bytes"
        ],
        "nvml_available": memory["nvml_available"],
        "nvml_device_total_bytes": memory["nvml_device_total_bytes"],
        "nvml_device_used_bytes": memory["nvml_device_used_bytes"],
        "nvml_device_free_bytes": memory["nvml_device_free_bytes"],
        "nvml_process_used_bytes": memory["nvml_process_used_bytes"],
        "nvml_other_processes_used_bytes": memory["nvml_other_processes_used_bytes"],
        "nvml_process_accounting_supported": memory["nvml_process_accounting_supported"],
        "nvml_process_accounting_reliable": memory["nvml_process_accounting_reliable"],
        "nvml_process_accounting_note": memory["nvml_process_accounting_note"],
        "nvml_error": memory["nvml_error"],
        "process_ram_bytes": cpu["process_ram_bytes"],
        "system_ram_used_bytes": cpu["system_ram_used_bytes"],
    }


def device_index(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    if device.index is not None:
        return int(device.index)
    if torch.cuda.is_available():
        return int(torch.cuda.current_device())
    return 0


def allocator_backend() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        return str(torch.cuda.get_allocator_backend())
    except Exception:
        return None


def reset_torch_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def cuda_memory_snapshot(device: torch.device) -> dict[str, float | str | int | None]:
    """Return PyTorch allocator metrics using byte units.

    The legacy aliases are kept for existing readers; the
    torch_current_* fields are the canonical schema names.
    """
    active_index = device_index(device)
    backend = allocator_backend()
    if device.type != "cuda":
        return {
            "active_device": str(device),
            "active_device_index": active_index,
            "allocator_backend": backend,
            "torch_current_allocated_bytes": None,
            "torch_current_reserved_bytes": None,
            "torch_peak_allocated_bytes": None,
            "torch_peak_reserved_bytes": None,
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    peak_reserved = torch.cuda.max_memory_reserved(device) if hasattr(torch.cuda, "max_memory_reserved") else 0
    current_allocated = float(torch.cuda.memory_allocated(device))
    current_reserved = float(torch.cuda.memory_reserved(device))
    peak_allocated = float(torch.cuda.max_memory_allocated(device))
    peak_reserved = float(peak_reserved)
    return {
        "active_device": str(device),
        "active_device_index": active_index,
        "allocator_backend": backend,
        "torch_current_allocated_bytes": current_allocated,
        "torch_current_reserved_bytes": current_reserved,
        "torch_peak_allocated_bytes": peak_allocated,
        "torch_peak_reserved_bytes": peak_reserved,
        "allocated_bytes": current_allocated,
        "reserved_bytes": current_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def nvml_memory_snapshot(device: torch.device) -> dict[str, float | bool | str | None]:
    base: dict[str, float | bool | str | None] = {
        "nvml_available": False,
        "nvml_device_total_bytes": None,
        "nvml_device_used_bytes": None,
        "nvml_device_free_bytes": None,
        "nvml_process_used_bytes": None,
        "nvml_other_processes_used_bytes": None,
        "nvml_process_accounting_supported": False,
        "nvml_error": None,
    }
    if device.type != "cuda":
        base["nvml_error"] = "device is not cuda"
        return base
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index(device) or 0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        base.update(
            {
                "nvml_available": True,
                "nvml_device_total_bytes": float(mem.total),
                "nvml_device_used_bytes": float(mem.used),
                "nvml_device_free_bytes": float(mem.free),
            }
        )
        try:
            process_fn = getattr(
                pynvml,
                "nvmlDeviceGetComputeRunningProcesses_v2",
                pynvml.nvmlDeviceGetComputeRunningProcesses,
            )
            processes = process_fn(handle)
            own_pid = os.getpid()
            own_used = 0.0
            other_used = 0.0
            for proc in processes:
                used = getattr(proc, "usedGpuMemory", None)
                if used is None:
                    continue
                if int(getattr(proc, "pid", -1)) == own_pid:
                    own_used += float(used)
                else:
                    other_used += float(used)
            base.update(
                {
                    "nvml_process_accounting_supported": True,
                    "nvml_process_used_bytes": own_used,
                    "nvml_other_processes_used_bytes": other_used,
                }
            )
        except Exception as exc:
            base["nvml_process_accounting_supported"] = False
            base["nvml_error"] = f"process accounting unavailable: {type(exc).__name__}: {exc}"
    except Exception as exc:
        base["nvml_error"] = f"{type(exc).__name__}: {exc}"
    return base


def gpu_memory_snapshot(device: torch.device, measurement_stage: str) -> dict[str, Any]:
    if measurement_stage not in MEMORY_STAGES:
        raise ValueError(f"Unknown memory measurement stage {measurement_stage!r}.")
    snapshot: dict[str, Any] = {"measurement_stage": measurement_stage}
    snapshot.update(cuda_memory_snapshot(device))
    snapshot.update(nvml_memory_snapshot(device))
    process_supported = bool(snapshot.get("nvml_process_accounting_supported"))
    torch_allocated = snapshot.get("torch_current_allocated_bytes")
    process_used = snapshot.get("nvml_process_used_bytes")
    process_reliable = process_supported
    process_note = None
    if (
        process_supported
        and isinstance(torch_allocated, (int, float))
        and float(torch_allocated) > 0.0
        and isinstance(process_used, (int, float))
        and float(process_used) == 0.0
    ):
        process_reliable = False
        process_note = (
            "NVML process accounting reported zero bytes while PyTorch had active "
            "allocations; treat process-level NVML as unreliable in this environment."
        )
    snapshot["nvml_process_accounting_reliable"] = process_reliable
    snapshot["nvml_process_accounting_note"] = process_note
    return snapshot


def cpu_memory_snapshot() -> dict[str, float | None]:
    try:
        import psutil
    except Exception:
        return {"process_ram_bytes": None, "system_ram_used_bytes": None}
    process = psutil.Process(os.getpid())
    return {
        "process_ram_bytes": float(process.memory_info().rss),
        "system_ram_used_bytes": float(psutil.virtual_memory().used),
    }


class FeasibilityStageTracker:
    def __init__(self, device: torch.device):
        self.device = device
        self.current_stage: str | None = None
        self.final_successful_stage: str | None = None
        self.memory: dict[str, float | None] = {}
        self.memory_stages: list[dict[str, Any]] = []

    def enter(self, stage: str) -> None:
        if stage not in FEASIBILITY_STAGES:
            raise ValueError(f"Unknown feasibility stage {stage!r}.")
        if stage == self.current_stage:
            return
        if self.current_stage is not None:
            self.final_successful_stage = self.current_stage
        self.current_stage = stage
        self._record_stage_boundary(stage)

    def complete_current(self) -> None:
        if self.current_stage is not None:
            self.final_successful_stage = self.current_stage

    def _record_stage_boundary(self, stage: str) -> None:
        if stage == "asset_validation":
            self._record_cpu("process_ram_before_loading_bytes", "system_ram_before_loading_bytes")
        elif stage == "bounds_or_metadata_computation":
            self.record_memory_stage("after_cpu_asset_load")
            self._record_cpu(
                "process_ram_after_cpu_tensor_construction_bytes",
                "system_ram_after_cpu_tensor_construction_bytes",
            )
        elif stage == "gpu_transfer":
            self.record_memory_stage("before_gpu_transfer")
            self._record_gpu("gpu_before_transfer", "before_gpu_transfer")
        elif stage == "gpu_scene_construction":
            self.record_memory_stage("after_gpu_transfer")
            self._record_gpu("gpu_after_transfer", "after_gpu_transfer")
        elif stage == "rasterization":
            self.record_memory_stage("before_rasterization")
            self._record_gpu("gpu_before_rasterization", "before_rasterization")

    def record_memory_stage(self, measurement_stage: str) -> dict[str, Any]:
        snapshot = gpu_memory_snapshot(self.device, measurement_stage)
        self.memory_stages.append(snapshot)
        self.memory[f"{measurement_stage}_torch_current_allocated_bytes"] = snapshot[
            "torch_current_allocated_bytes"
        ]
        self.memory[f"{measurement_stage}_torch_current_reserved_bytes"] = snapshot[
            "torch_current_reserved_bytes"
        ]
        self.memory[f"{measurement_stage}_torch_peak_allocated_bytes"] = snapshot[
            "torch_peak_allocated_bytes"
        ]
        self.memory[f"{measurement_stage}_torch_peak_reserved_bytes"] = snapshot[
            "torch_peak_reserved_bytes"
        ]
        self.memory[f"{measurement_stage}_nvml_device_used_bytes"] = snapshot[
            "nvml_device_used_bytes"
        ]
        self.memory[f"{measurement_stage}_nvml_device_free_bytes"] = snapshot[
            "nvml_device_free_bytes"
        ]
        self.memory[f"{measurement_stage}_nvml_process_used_bytes"] = snapshot[
            "nvml_process_used_bytes"
        ]
        return snapshot

    def _record_cpu(self, process_key: str, system_key: str) -> None:
        snapshot = cpu_memory_snapshot()
        self.memory.setdefault(process_key, snapshot["process_ram_bytes"])
        self.memory.setdefault(system_key, snapshot["system_ram_used_bytes"])

    def _record_gpu(self, prefix: str, measurement_stage: str) -> None:
        snapshot = gpu_memory_snapshot(self.device, measurement_stage)
        self.memory.setdefault(f"{prefix}_allocated_bytes", snapshot["torch_current_allocated_bytes"])
        self.memory.setdefault(f"{prefix}_reserved_bytes", snapshot["torch_current_reserved_bytes"])
        self.memory.setdefault(f"{prefix}_peak_allocated_bytes", snapshot["torch_peak_allocated_bytes"])
        self.memory.setdefault(f"{prefix}_peak_reserved_bytes", snapshot["torch_peak_reserved_bytes"])


def environment_metadata() -> dict[str, Any]:
    gpu = {"available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu.update(
            {
                "name": torch.cuda.get_device_name(0),
                "total_memory_bytes": int(props.total_memory),
                "capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gsplat_version": package_version("gsplat"),
        "gpu": gpu,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def select_measured_cameras(camera_path: CameraPath, max_frames: int | None, repeat: int) -> list[tuple[int, int, CameraSpec]]:
    frames = []
    limit = max_frames if max_frames is not None and max_frames > 0 else len(camera_path.cameras) * repeat
    for repeat_index in range(repeat):
        for camera_index, camera in enumerate(camera_path.cameras):
            if len(frames) >= limit:
                return frames
            frames.append((repeat_index, camera_index, camera))
    return frames


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def timing_statistics(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": float(np.mean(values)) if values else None,
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    record_kind: str,
    config: dict[str, Any],
    asset: dict[str, Any],
    camera_path: CameraPath,
    environment: dict[str, Any],
) -> dict[str, Any]:
    render_call_times = finite_values(rows, "render_call_time_ms")
    cuda_rasterization_times = finite_values(rows, "cuda_rasterization_time_ms")
    prep_times = finite_values(rows, "preparation_time_ms")
    end_to_end_times = finite_values(rows, "end_to_end_frame_time_ms")
    peak_allocated = finite_values(rows, "torch_peak_allocated_bytes")
    peak_reserved = finite_values(rows, "torch_peak_reserved_bytes")
    current_allocated = finite_values(rows, "torch_current_allocated_bytes")
    current_reserved = finite_values(rows, "torch_current_reserved_bytes")
    nvml_device_used = finite_values(rows, "nvml_device_used_bytes")
    nvml_process_used = finite_values(rows, "nvml_process_used_bytes")
    process_ram = finite_values(rows, "process_ram_bytes")
    selected_counts = finite_values(rows, "selected_gaussian_count")
    active_cell_counts = finite_values(rows, "active_cell_count")
    selected_cell_counts = finite_values(rows, "selected_cell_count")
    total_cell_counts = finite_values(rows, "total_cell_count")
    visibility_times = finite_values(rows, "visibility_selection_ms")
    lod_selection_times = finite_values(rows, "lod_selection_ms")
    assembly_times = finite_values(rows, "selected_cell_assembly_ms")
    resident_counts = finite_values(rows, "resident_gaussian_count")
    active_counts = finite_values(rows, "active_gaussian_count")
    mean_end_to_end = float(np.mean(end_to_end_times)) if end_to_end_times else None
    return {
        "schema_version": 2,
        "record_kind": record_kind,
        "configuration_id": config["configuration_id"],
        "checkpoint_identifier": asset["identifier"],
        "checkpoint_type": asset["checkpoint_type"],
        "checkpoint_path_supplied": asset["path_supplied"],
        "trajectory_identifier": camera_path.trajectory_id,
        "frame_count": len(rows),
        "timing": {
            "preparation_time_ms": timing_statistics(prep_times),
            "render_call_time_ms": timing_statistics(render_call_times),
            "cuda_rasterization_time_ms": timing_statistics(cuda_rasterization_times),
            "end_to_end_frame_time_ms": timing_statistics(end_to_end_times),
        },
        "mean_preparation_time_ms": timing_statistics(prep_times)["mean"],
        "mean_render_call_time_ms": timing_statistics(render_call_times)["mean"],
        "mean_cuda_rasterization_time_ms": timing_statistics(cuda_rasterization_times)["mean"],
        "mean_end_to_end_frame_time_ms": mean_end_to_end,
        "end_to_end_fps": 1000.0 / mean_end_to_end
        if mean_end_to_end and mean_end_to_end > 0
        else None,
        "render_call_fps_equivalent": 1000.0 / float(np.mean(render_call_times))
        if render_call_times and float(np.mean(render_call_times)) > 0
        else None,
        "peak_allocated_vram_bytes": max(peak_allocated) if peak_allocated else None,
        "peak_reserved_vram_bytes": max(peak_reserved) if peak_reserved else None,
        "max_torch_current_allocated_bytes": max(current_allocated) if current_allocated else None,
        "max_torch_current_reserved_bytes": max(current_reserved) if current_reserved else None,
        "max_nvml_device_used_bytes": max(nvml_device_used) if nvml_device_used else None,
        "max_nvml_process_used_bytes": max(nvml_process_used) if nvml_process_used else None,
        "peak_process_ram_bytes": max(process_ram) if process_ram else None,
        "total_gaussian_count": rows[0]["total_gaussian_count"] if rows else None,
        "resident_gaussian_count": resident_counts[0] if resident_counts else None,
        "mean_resident_gaussian_count": float(np.mean(resident_counts)) if resident_counts else None,
        "active_gaussian_count": active_counts[0] if active_counts else None,
        "mean_active_gaussian_count": float(np.mean(active_counts)) if active_counts else None,
        "all_loaded_gaussians_resident": all(
            bool(row.get("all_loaded_gaussians_resident")) for row in rows
        ) if rows else None,
        "all_loaded_gaussians_active": all(
            bool(row.get("all_loaded_gaussians_active")) for row in rows
        ) if rows else None,
        "selected_gaussian_count": rows[0].get("selected_gaussian_count") if rows else None,
        "min_selected_gaussian_count": min(selected_counts) if selected_counts else None,
        "mean_selected_gaussian_count": float(np.mean(selected_counts)) if selected_counts else None,
        "median_selected_gaussian_count": percentile(selected_counts, 50),
        "p95_selected_gaussian_count": percentile(selected_counts, 95),
        "max_selected_gaussian_count": max(selected_counts) if selected_counts else None,
        "active_cell_count": rows[0].get("active_cell_count") if rows else None,
        "mean_active_cell_count": float(np.mean(active_cell_counts)) if active_cell_counts else None,
        "selected_cell_count": rows[0].get("selected_cell_count") if rows else None,
        "min_selected_cell_count": min(selected_cell_counts) if selected_cell_counts else None,
        "mean_selected_cell_count": float(np.mean(selected_cell_counts)) if selected_cell_counts else None,
        "median_selected_cell_count": percentile(selected_cell_counts, 50),
        "p95_selected_cell_count": percentile(selected_cell_counts, 95),
        "max_selected_cell_count": max(selected_cell_counts) if selected_cell_counts else None,
        "total_cell_count": rows[0].get("total_cell_count") if rows else None,
        "mean_total_cell_count": float(np.mean(total_cell_counts)) if total_cell_counts else None,
        "mean_visibility_selection_ms": float(np.mean(visibility_times)) if visibility_times else None,
        "mean_lod_selection_ms": float(np.mean(lod_selection_times)) if lod_selection_times else None,
        "mean_selected_cell_assembly_ms": float(np.mean(assembly_times)) if assembly_times else None,
        "mean_visible_gaussians": float(np.mean(finite_values(rows, "visible_gaussians")))
        if finite_values(rows, "visible_gaussians")
        else None,
        "resolution": [config["width"], config["height"]],
        "render_settings": render_settings_metadata(config),
        "environment": environment,
        "camera_path": camera_path.metadata,
        "integrity": benchmark_integrity_metadata(
            config,
            all_loaded_gaussians_active_verified=all(
                int(row.get("active_gaussian_count", -1))
                == int(row.get("total_gaussian_count", -2))
                for row in rows
            )
            if rows
            else True,
        ),
    }


def render_settings_metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "sh_degree": config.get("sh_degree"),
        "render_mode": config.get("render_mode"),
        "background": config.get("background"),
        "near_plane": config.get("near_plane"),
        "far_plane": config.get("far_plane"),
        "radius_clip": config.get("radius_clip"),
        "eps2d": config.get("eps2d"),
        "rasterize_mode": config.get("rasterize_mode"),
        "camera_model": config.get("camera_model"),
        "packed": config.get("packed"),
    }


def scene_stats_metadata(scene, scene_stats) -> dict[str, Any]:
    metadata = {
        "scene_type": scene_stats.scene_type,
        "total_gaussians": int(scene_stats.total_gaussians),
        "sh_degree": scene_stats.sh_degree,
    }
    if scene_stats.scene_type in {
        "citygaussian_fixed_all", "citygaussian_visibility", "citygaussian_lod_release"
    }:
        metadata.update(
            {
                "lod_source_level": scene_stats.citygaussian_lod_source_level,
                "block_dim": list(scene_stats.citygaussian_block_dim or []),
                "aabb": list(scene_stats.citygaussian_aabb or []),
                "total_cell_count": scene_stats.citygaussian_cell_count,
                "per_cell_gaussian_counts": scene_stats.citygaussian_per_cell_counts,
                "cell_construction_ms": scene_stats.citygaussian_cell_construction_ms,
                "lod_policy": scene_stats.citygaussian_lod_policy,
                "lod_level_labels": scene_stats.citygaussian_lod_level_labels,
                "lod_level_semantics": scene_stats.citygaussian_lod_level_semantics,
                "lod_level_gaussian_counts": scene_stats.citygaussian_lod_level_gaussian_counts,
                "lod_level_paths": [
                    str(path) for path in (scene_stats.citygaussian_lod_level_paths or [])
                ],
                "visibility_bounds_level": scene_stats.citygaussian_visibility_bounds_level,
            }
        )
        if hasattr(scene, "get_cell_metadata"):
            metadata["cell_metadata"] = scene.get_cell_metadata()
    return metadata


def flat_integrity_metadata(
    *,
    scene_type_flat_verified: bool = True,
    all_loaded_gaussians_active_verified: bool | None = None,
) -> dict[str, bool | None]:
    return {
        "flat_scene_requested": True,
        "scene_type_flat": scene_type_flat_verified,
        "all_loaded_gaussians_active": all_loaded_gaussians_active_verified,
        "all_loaded_gaussians_active_expected": True,
        "all_loaded_gaussians_resident_expected": True,
        "block_manager_used": False,
        "host_residency_used": False,
        "lod_selection_used": False,
        "hierarchy_traversal_used": False,
        "citygaussian_runtime_selection_used": False,
        "gsplat_renderer_used": True,
    }


def benchmark_integrity_metadata(
    config: dict[str, Any],
    *,
    scene_type_verified: bool = True,
    all_loaded_gaussians_active_verified: bool | None = None,
) -> dict[str, bool | None]:
    scene_type = str(config.get("scene_type", "flat"))
    if scene_type == "flat":
        return flat_integrity_metadata(
            scene_type_flat_verified=scene_type_verified,
            all_loaded_gaussians_active_verified=all_loaded_gaussians_active_verified,
        )
    if scene_type in {
        "citygaussian_fixed_all", "citygaussian_visibility", "citygaussian_lod_release"
    }:
        lod_used = scene_type == "citygaussian_lod_release"
        visibility_used = scene_type in {"citygaussian_visibility", "citygaussian_lod_release"}
        return {
            "flat_scene_requested": False,
            "scene_type_flat": False,
            "scene_type_citygaussian_fixed_all": scene_type_verified and not visibility_used,
            "scene_type_citygaussian_visibility": scene_type_verified
            and scene_type == "citygaussian_visibility",
            "scene_type_citygaussian_lod_release": scene_type_verified and lod_used,
            "all_loaded_gaussians_active": all_loaded_gaussians_active_verified,
            "all_loaded_gaussians_active_expected": requires_all_loaded_gaussians_active(config),
            "all_loaded_gaussians_resident_expected": True,
            "block_manager_used": False,
            "host_residency_used": False,
            "lod_selection_used": lod_used,
            "paper_lod_policy_used": False,
            "hierarchy_traversal_used": False,
            "citygaussian_cell_construction_used": True,
            "citygaussian_runtime_selection_used": visibility_used,
            "citygaussian_visibility_selection_used": visibility_used,
            "adaptive_lod_used": lod_used,
            "vq_reconstruction_used": False,
            "streaming_used": False,
            "gsplat_renderer_used": True,
        }
    return {
        "flat_scene_requested": None,
        "scene_type_flat": None,
        "all_loaded_gaussians_active": all_loaded_gaussians_active_verified,
        "all_loaded_gaussians_active_expected": None,
        "all_loaded_gaussians_resident_expected": None,
        "block_manager_used": None,
        "host_residency_used": None,
        "lod_selection_used": None,
        "hierarchy_traversal_used": None,
        "citygaussian_runtime_selection_used": None,
        "gsplat_renderer_used": True,
    }


def resolved_config_metadata(
    config: dict[str, Any],
    asset: dict[str, Any],
    camera_path: CameraPath,
    record_kind: str,
    max_frames: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "configuration": config,
        "asset": asset,
        "camera_path": camera_path.metadata,
        "record_kind": record_kind,
        "max_frames": max_frames,
        "warmup_frames": config.get("warmup_frames"),
        "num_repeat": config.get("num_repeat"),
        "repetition_index": config.get("repetition_index"),
        "integrity": benchmark_integrity_metadata(config),
        "provenance": {
            "configuration_sha256": canonical_json_sha256(config),
            "trajectory_file_sha256": sha256_file(camera_path.path)
            if camera_path.path.is_file()
            else None,
        },
    }


def build_feasibility_result(
    *,
    config: dict[str, Any],
    asset: dict[str, Any],
    camera_path: CameraPath,
    status: str,
    failure_stage: str | None,
    environment: dict[str, Any],
    cpu_before: dict[str, float | None],
    cpu_after: dict[str, float | None],
    gpu_memory: dict[str, float | None],
    exception: BaseException | None = None,
    frame_result: dict[str, Any] | None = None,
    total_gaussians: int | None = None,
    scene_type_flat_verified: bool = False,
    final_successful_stage: str | None = None,
    stage_memory: dict[str, float | None] | None = None,
    memory_stages: list[dict[str, Any]] | None = None,
    scene_construction_peak: dict[str, float | None] | None = None,
    render_peak: dict[str, float | None] | None = None,
    whole_run_peak: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_kind": "feasibility_result",
        "configuration_id": config["configuration_id"],
        "status": status,
        "failure_stage": failure_stage,
        "final_successful_stage": final_successful_stage,
        "checkpoint_identifier": asset["identifier"],
        "checkpoint_type": asset["checkpoint_type"],
        "checkpoint_path_supplied": asset["path_supplied"],
        "checkpoint_file_size_bytes": asset["file_size_bytes"],
        "total_gaussians": total_gaussians or asset.get("ply_vertex_count"),
        "cpu_ram_before_bytes": cpu_before.get("process_ram_bytes"),
        "cpu_ram_after_bytes": cpu_after.get("process_ram_bytes"),
        "system_ram_before_bytes": cpu_before.get("system_ram_used_bytes"),
        "system_ram_after_bytes": cpu_after.get("system_ram_used_bytes"),
        "gpu_allocated_bytes": gpu_memory.get("allocated_bytes"),
        "gpu_reserved_bytes": gpu_memory.get("reserved_bytes"),
        "gpu_peak_allocated_bytes": gpu_memory.get("peak_allocated_bytes"),
        "gpu_peak_reserved_bytes": gpu_memory.get("peak_reserved_bytes"),
        "torch_current_allocated_bytes": gpu_memory.get("torch_current_allocated_bytes"),
        "torch_current_reserved_bytes": gpu_memory.get("torch_current_reserved_bytes"),
        "torch_peak_allocated_bytes": gpu_memory.get("torch_peak_allocated_bytes"),
        "torch_peak_reserved_bytes": gpu_memory.get("torch_peak_reserved_bytes"),
        "allocator_backend": gpu_memory.get("allocator_backend"),
        "active_device": gpu_memory.get("active_device"),
        "nvml_device_total_bytes": gpu_memory.get("nvml_device_total_bytes"),
        "nvml_device_used_bytes": gpu_memory.get("nvml_device_used_bytes"),
        "nvml_device_free_bytes": gpu_memory.get("nvml_device_free_bytes"),
        "nvml_process_used_bytes": gpu_memory.get("nvml_process_used_bytes"),
        "nvml_other_processes_used_bytes": gpu_memory.get("nvml_other_processes_used_bytes"),
        "nvml_process_accounting_supported": gpu_memory.get(
            "nvml_process_accounting_supported"
        ),
        "nvml_process_accounting_reliable": gpu_memory.get(
            "nvml_process_accounting_reliable"
        ),
        "nvml_process_accounting_note": gpu_memory.get("nvml_process_accounting_note"),
        "process_ram_after_ply_parsing_bytes": (stage_memory or {}).get(
            "process_ram_after_ply_parsing_bytes"
        ),
        "process_ram_after_cpu_tensor_construction_bytes": (stage_memory or {}).get(
            "process_ram_after_cpu_tensor_construction_bytes"
        ),
        "gpu_before_transfer_allocated_bytes": (stage_memory or {}).get(
            "gpu_before_transfer_allocated_bytes"
        ),
        "gpu_before_transfer_reserved_bytes": (stage_memory or {}).get(
            "gpu_before_transfer_reserved_bytes"
        ),
        "gpu_after_transfer_allocated_bytes": (stage_memory or {}).get(
            "gpu_after_transfer_allocated_bytes"
        ),
        "gpu_after_transfer_reserved_bytes": (stage_memory or {}).get(
            "gpu_after_transfer_reserved_bytes"
        ),
        "gpu_before_rasterization_allocated_bytes": (stage_memory or {}).get(
            "gpu_before_rasterization_allocated_bytes"
        ),
        "gpu_before_rasterization_reserved_bytes": (stage_memory or {}).get(
            "gpu_before_rasterization_reserved_bytes"
        ),
        "stage_memory": stage_memory or {},
        "memory_stages": memory_stages or [],
        "scene_construction_peak": scene_construction_peak or {},
        "render_peak": render_peak or {},
        "whole_run_peak": whole_run_peak or {},
        "exception_type": type(exception).__name__ if exception is not None else None,
        "exception_message": str(exception) if exception is not None else None,
        "environment": environment,
        "camera_path": camera_path.metadata,
        "frame_result": frame_result,
        "integrity": flat_integrity_metadata(
            scene_type_flat_verified=scene_type_flat_verified,
            all_loaded_gaussians_active_verified=frame_result is not None,
        ),
    }


def is_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message or "cuda oom" in message


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True, allow_nan=False))


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def peak_from_snapshot(snapshot: dict[str, Any] | None) -> dict[str, float | None]:
    snapshot = snapshot or {}
    return {
        "torch_peak_allocated_bytes": snapshot.get("torch_peak_allocated_bytes"),
        "torch_peak_reserved_bytes": snapshot.get("torch_peak_reserved_bytes"),
    }


def max_peak_dicts(*peaks: dict[str, float | None] | None) -> dict[str, float | None]:
    allocated = [
        float(peak["torch_peak_allocated_bytes"])
        for peak in peaks
        if peak and isinstance(peak.get("torch_peak_allocated_bytes"), (int, float))
    ]
    reserved = [
        float(peak["torch_peak_reserved_bytes"])
        for peak in peaks
        if peak and isinstance(peak.get("torch_peak_reserved_bytes"), (int, float))
    ]
    return {
        "torch_peak_allocated_bytes": max(allocated) if allocated else None,
        "torch_peak_reserved_bytes": max(reserved) if reserved else None,
    }


def sample_camera_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if sample_count >= frame_count:
        return list(range(frame_count))
    raw = np.linspace(0, frame_count - 1, num=sample_count)
    indices = sorted({int(round(value)) for value in raw})
    candidate = 0
    while len(indices) < sample_count and candidate < frame_count:
        if candidate not in indices:
            indices.append(candidate)
        candidate += 1
    return sorted(indices)


def select_sampled_cameras(camera_path: CameraPath, sample_count: int) -> list[tuple[int, CameraSpec]]:
    return [(idx, camera_path.cameras[idx]) for idx in sample_camera_indices(len(camera_path.cameras), sample_count)]


def run_benchmark(
    *,
    config: dict[str, Any],
    asset_path: Path | list[Path],
    asset_supplied: str | list[str],
    camera_path: CameraPath,
    output_dir: Path,
    record_kind: str,
    max_frames: int | None,
    device: torch.device,
    log_gpu_events: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = FeasibilityStageTracker(device)
    tracker.record_memory_stage("process_start")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tracker.record_memory_stage("after_cuda_initialization")
    tracker.record_memory_stage("before_asset_load")
    reset_torch_peak_memory(device)
    asset = inspect_benchmark_asset(asset_path, asset_supplied, config)
    environment = environment_metadata()
    write_json(
        output_dir / RESOLVED_CONFIG_JSON,
        resolved_config_metadata(config, asset, camera_path, record_kind, max_frames),
    )

    scene, scene_stats = load_scene_for_benchmark(
        config,
        asset_path,
        device,
        progress_callback=tracker.enter,
    )
    post_transfer_memory = gpu_memory_snapshot(device, "after_gpu_transfer")
    scene_construction_peak = peak_from_snapshot(post_transfer_memory)
    asset["total_gaussians"] = int(scene_stats.total_gaussians)
    renderer = GsplatRenderer(device)
    warmup_frames = int(config.get("warmup_frames", 0))
    repeat = int(config.get("num_repeat", 1))

    for idx in range(warmup_frames):
        camera = camera_path.cameras[idx % len(camera_path.cameras)]
        render_one_frame(
            scene,
            renderer,
            camera,
            config,
            int(scene_stats.total_gaussians),
            scene_stats.sh_degree,
            device,
            False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        reset_torch_peak_memory(device)

    rows = []
    start = time.perf_counter()
    for frame_index, (repeat_index, camera_index, camera) in enumerate(
        select_measured_cameras(camera_path, max_frames, repeat)
    ):
        metrics = render_one_frame(
            scene,
            renderer,
            camera,
            config,
            int(scene_stats.total_gaussians),
            scene_stats.sh_degree,
            device,
            log_gpu_events,
        )
        rows.append(
            {
                "schema_version": 1,
                "record_kind": "frame_sample",
                "configuration_id": config["configuration_id"],
                "frame_index": frame_index,
                "repeat_index": repeat_index,
                "camera_index": camera_index,
                "camera_id": camera.camera_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_sec": time.perf_counter() - start,
                "width": camera.width,
                "height": camera.height,
                "post_transfer_torch_allocated_bytes": post_transfer_memory[
                    "torch_current_allocated_bytes"
                ],
                "post_transfer_torch_reserved_bytes": post_transfer_memory[
                    "torch_current_reserved_bytes"
                ],
                **metrics,
                "integrity": benchmark_integrity_metadata(
                    config,
                    all_loaded_gaussians_active_verified=int(metrics["active_gaussian_count"])
                    == int(metrics["total_gaussian_count"]),
                ),
            }
        )

    append_jsonl(output_dir / FRAME_JSONL, rows)
    summary = summarize_rows(
        rows,
        record_kind=record_kind,
        config=config,
        asset=asset,
        camera_path=camera_path,
        environment=environment,
    )
    summary["scene"] = scene_stats_metadata(scene, scene_stats)
    render_peak = {
        "torch_peak_allocated_bytes": summary["peak_allocated_vram_bytes"],
        "torch_peak_reserved_bytes": summary["peak_reserved_vram_bytes"],
    }
    summary["memory_stages"] = tracker.memory_stages
    summary["scene_construction_peak"] = scene_construction_peak
    summary["render_peak"] = render_peak
    summary["whole_run_peak"] = max_peak_dicts(scene_construction_peak, render_peak)
    summary["warmup_policy"] = {
        "frame_count": warmup_frames,
        "camera_policy": "trajectory frame 0, then deterministic successive frames if count > 1",
        "included_in_aggregates": False,
        "memory_peaks_reset_after_warmup": True,
    }
    summary["repetition_index"] = config.get("repetition_index")
    summary["provenance"] = {
        "configuration_sha256": canonical_json_sha256(config),
        "trajectory_file_sha256": sha256_file(camera_path.path),
    }
    write_json(output_dir / SUMMARY_JSON, summary)
    return 0


def run_camera_sample(
    *,
    config: dict[str, Any],
    asset_path: Path | list[Path],
    asset_supplied: str | list[str],
    camera_path: CameraPath,
    output_dir: Path,
    sample_frames: int,
    device: torch.device,
    log_gpu_events: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = FeasibilityStageTracker(device)
    tracker.record_memory_stage("process_start")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tracker.record_memory_stage("after_cuda_initialization")
    tracker.record_memory_stage("before_asset_load")
    reset_torch_peak_memory(device)

    asset = inspect_benchmark_asset(asset_path, asset_supplied, config)
    environment = environment_metadata()
    sampled = select_sampled_cameras(camera_path, sample_frames)
    write_json(
        output_dir / RESOLVED_CONFIG_JSON,
        resolved_config_metadata(
            config,
            asset,
            camera_path,
            "camera_sample_validation",
            len(sampled),
        ),
    )

    scene, scene_stats = load_scene_for_benchmark(
        config,
        asset_path,
        device,
        progress_callback=tracker.enter,
    )
    scene_construction_peak = peak_from_snapshot(gpu_memory_snapshot(device, "after_gpu_transfer"))
    asset["total_gaussians"] = int(scene_stats.total_gaussians)
    renderer = GsplatRenderer(device)

    rows = []
    start = time.perf_counter()
    for sample_index, (trajectory_index, camera) in enumerate(sampled):
        metrics = render_one_frame(
            scene,
            renderer,
            camera,
            config,
            int(scene_stats.total_gaussians),
            scene_stats.sh_degree,
            device,
            log_gpu_events,
        )
        rows.append(
            {
                "schema_version": 1,
                "record_kind": "camera_sample_frame",
                "configuration_id": config["configuration_id"],
                "frame_index": sample_index,
                "trajectory_index": trajectory_index,
                "camera_index": trajectory_index,
                "camera_id": camera.camera_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_sec": time.perf_counter() - start,
                "width": camera.width,
                "height": camera.height,
                **metrics,
                "integrity": benchmark_integrity_metadata(
                    config,
                    all_loaded_gaussians_active_verified=int(metrics["active_gaussian_count"])
                    == int(metrics["total_gaussian_count"]),
                ),
            }
        )

    append_jsonl(output_dir / FRAME_JSONL, rows)
    summary = summarize_rows(
        rows,
        record_kind="camera_sample_validation",
        config=config,
        asset=asset,
        camera_path=camera_path,
        environment=environment,
    )
    summary["scene"] = scene_stats_metadata(scene, scene_stats)
    render_peak = {
        "torch_peak_allocated_bytes": summary["peak_allocated_vram_bytes"],
        "torch_peak_reserved_bytes": summary["peak_reserved_vram_bytes"],
    }
    visible_values = finite_values(rows, "visible_gaussians")
    peak_values = finite_values(rows, "torch_peak_allocated_bytes")
    summary.update(
        {
            "sampled_trajectory_indices": [idx for idx, _ in sampled],
            "memory_stages": tracker.memory_stages,
            "scene_construction_peak": scene_construction_peak,
            "render_peak": render_peak,
            "whole_run_peak": max_peak_dicts(scene_construction_peak, render_peak),
            "min_visible_gaussians": min(visible_values) if visible_values else None,
            "max_visible_gaussians": max(visible_values) if visible_values else None,
            "min_render_peak_allocated_bytes": min(peak_values) if peak_values else None,
            "max_render_peak_allocated_bytes": max(peak_values) if peak_values else None,
            "finite_image_all": all(row.get("finite_image") is True for row in rows),
            "finite_alpha_all": all(row.get("finite_alpha") is True for row in rows),
            "integrity": benchmark_integrity_metadata(
                config,
                scene_type_verified=scene_stats.scene_type == str(config.get("scene_type", "flat")),
                all_loaded_gaussians_active_verified=all(
                    int(row.get("active_gaussian_count", -1)) == int(scene_stats.total_gaussians)
                    for row in rows
                ),
            ),
        }
    )
    write_json(output_dir / SUMMARY_JSON, summary)
    write_json(output_dir / CAMERA_SAMPLE_JSON, summary)
    return 0


def run_feasibility(
    *,
    config: dict[str, Any],
    asset_path: Path,
    asset_supplied: str,
    camera_path: CameraPath,
    output_dir: Path,
    max_frames: int | None,
    device: torch.device,
    log_gpu_events: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = FeasibilityStageTracker(device)
    tracker.record_memory_stage("process_start")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tracker.record_memory_stage("after_cuda_initialization")
    tracker.enter("asset_validation")
    asset = inspect_asset(asset_path, asset_supplied, config)
    environment = environment_metadata()
    tracker.complete_current()
    write_json(
        output_dir / RESOLVED_CONFIG_JSON,
        resolved_config_metadata(config, asset, camera_path, "feasibility_result", max_frames),
    )
    cpu_before = cpu_memory_snapshot()
    tracker.record_memory_stage("before_asset_load")
    reset_torch_peak_memory(device)
    scene = scene_stats = renderer = None
    frame_result = None
    exception = None
    failure_gpu_memory = None
    scene_construction_peak = {}
    render_peak = {}
    try:
        scene, scene_stats = load_flat_scene_for_benchmark(
            asset_path,
            device,
            progress_callback=tracker.enter,
        )
        scene_construction_peak = peak_from_snapshot(
            gpu_memory_snapshot(device, "after_gpu_transfer")
        )
        tracker.enter("rasterizer_initialization")
        renderer = GsplatRenderer(device)
        camera = camera_path.cameras[0]
        frame_result = render_one_frame(
            scene,
            renderer,
            camera,
            config,
            int(scene_stats.total_gaussians),
            scene_stats.sh_degree,
            device,
            log_gpu_events,
            stage_callback=tracker.enter,
        )
        render_peak = {
            "torch_peak_allocated_bytes": frame_result.get("torch_peak_allocated_bytes"),
            "torch_peak_reserved_bytes": frame_result.get("torch_peak_reserved_bytes"),
        }
        tracker.record_memory_stage("after_rasterization")
        tracker.record_memory_stage("result_collection")
        tracker.complete_current()
        status = "success"
        failure_stage = None
    except BaseException as exc:
        failure_gpu_memory = gpu_memory_snapshot(device, "result_collection")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        status = "oom" if is_oom(exc) else "failed"
        failure_stage = tracker.current_stage
        exception = exc

    cpu_after = cpu_memory_snapshot()
    gpu = failure_gpu_memory or gpu_memory_snapshot(device, "result_collection")
    whole_run_peak = max_peak_dicts(scene_construction_peak, render_peak)
    result = build_feasibility_result(
        config=config,
        asset=asset,
        camera_path=camera_path,
        status=status,
        failure_stage=failure_stage,
        environment=environment,
        cpu_before=cpu_before,
        cpu_after=cpu_after,
        gpu_memory=gpu,
        exception=exception,
        frame_result=frame_result,
        total_gaussians=int(scene_stats.total_gaussians) if scene_stats is not None else None,
        scene_type_flat_verified=scene_stats is not None and scene_stats.scene_type == "flat",
        final_successful_stage=tracker.final_successful_stage,
        stage_memory=tracker.memory,
        memory_stages=tracker.memory_stages,
        scene_construction_peak=scene_construction_peak,
        render_peak=render_peak,
        whole_run_peak=whole_run_peak,
    )
    write_json(output_dir / FEASIBILITY_JSON, result)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run canonical flat real-scene gsplat benchmark configurations."
    )
    parser.add_argument("--config-id", required=True, choices=sorted(CONFIG_FILES))
    parser.add_argument("--asset", default=None, help="External checkpoint or PLY path.")
    parser.add_argument("--lod-finest-asset", default=None)
    parser.add_argument("--lod-middle-asset", default=None)
    parser.add_argument("--lod-coarsest-asset", default=None)
    parser.add_argument("--camera-path", type=Path, default=None, help="Override tracked camera path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None, help="Measured-frame cap for short validation.")
    parser.add_argument("--num-warmup", type=int, default=None, help="Override warmup frame count.")
    parser.add_argument("--num-repeat", type=int, default=None, help="Override repeat count.")
    parser.add_argument("--repetition-index", type=int, default=None)
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=None,
        help="Render an evenly spaced non-formal camera sample instead of the default run.",
    )
    parser.add_argument("--formal", action="store_true", help="Use record_kind=formal_benchmark.")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--log-gpu-events", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config_id)
    if args.num_warmup is not None:
        config["warmup_frames"] = int(args.num_warmup)
    if args.num_repeat is not None:
        config["num_repeat"] = int(args.num_repeat)
    if args.repetition_index is not None:
        config["repetition_index"] = int(args.repetition_index)
    if args.max_frames is None:
        args.max_frames = int(config.get("measured_frames", 0)) or None

    try:
        if args.config_id == "CG-LOD-RELEASE":
            asset_path, supplied_asset = resolve_lod_asset_paths(
                config,
                args.lod_finest_asset,
                args.lod_middle_asset,
                args.lod_coarsest_asset,
            )
        else:
            asset_path, supplied_asset = resolve_asset_path(config, args.asset)
        camera_json = resolve_camera_path(config, args.camera_path)
        camera_path = load_camera_path(camera_json, trajectory_id=config.get("trajectory_id"))
        device = require_cuda(args.device_index)
    except MissingAssetError as exc:
        print(f"Missing asset: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    record_kind = str(config.get("record_kind_default", "benchmark_validation"))
    if args.formal:
        record_kind = "formal_benchmark"

    if args.sample_frames is not None:
        return run_camera_sample(
            config=config,
            asset_path=asset_path,
            asset_supplied=supplied_asset,
            camera_path=camera_path,
            output_dir=args.output_dir,
            sample_frames=int(args.sample_frames),
            device=device,
            log_gpu_events=args.log_gpu_events,
        )

    if record_kind == "feasibility_result":
        if isinstance(asset_path, list) or isinstance(supplied_asset, list):
            raise ValueError("Feasibility mode supports one flat asset only.")
        return run_feasibility(
            config=config,
            asset_path=asset_path,
            asset_supplied=supplied_asset,
            camera_path=camera_path,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            device=device,
            log_gpu_events=args.log_gpu_events,
        )

    try:
        return run_benchmark(
            config=config,
            asset_path=asset_path,
            asset_supplied=supplied_asset,
            camera_path=camera_path,
            output_dir=args.output_dir,
            record_kind=record_kind,
            max_frames=args.max_frames,
            device=device,
            log_gpu_events=args.log_gpu_events,
        )
    except Exception as exc:
        print(f"Benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
