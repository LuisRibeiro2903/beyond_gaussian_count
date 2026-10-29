"""Benchmark gsplat checkpoint rendering on a reproducible camera path.

This script measures backend rasterization throughput without starting viser or
opening a browser. It writes one CSV row per rendered frame plus a summary JSON
next to the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
GSPLOT_EXAMPLES = ROOT / "gsplat" / "examples"
if str(GSPLOT_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(GSPLOT_EXAMPLES))
GS_FRAMEWORK = ROOT / "gs_framework"
if str(GS_FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(GS_FRAMEWORK))

from scene_runtime import FrameCamera, GsplatRenderer, RenderSettings, SceneLoader  # noqa: E402


try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency
    psutil = None

try:
    import pynvml
except ImportError:  # pragma: no cover - optional runtime dependency
    pynvml = None

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional runtime dependency
    imageio = None


CSV_FIELDS = [
    "timestamp",
    "elapsed_sec",
    "pid",
    "frame_index",
    "repeat_index",
    "camera_index",
    "width",
    "height",
    "frame_time_ms",
    "fps",
    "gpu_render_time_ms",
    "num_gaussians_total",
    "num_gaussians_rendered",
    "visible_gaussians",
    "vram_allocated_gb",
    "vram_reserved_gb",
    "vram_max_allocated_gb",
    "torch_allocated_mb",
    "torch_reserved_mb",
    "torch_frame_peak_allocated_mb",
    "torch_frame_peak_reserved_mb",
    "process_ram_gb",
    "process_cpu_percent",
    "system_ram_used_gb",
    "system_ram_percent",
    "gpu_utilization_percent",
    "gpu_util_percent",
    "vram_used_gb",
    "vram_total_gb",
    "render_frame_index",
    "render_frame_time_ms",
    "render_fps",
    "render_width",
    "render_height",
    "active_blocks",
    "active_block_ids",
    "active_gaussians",
    "active_gaussian_fraction",
    "total_gaussians",
    "total_blocks",
    "resident_blocks",
    "resident_gaussians",
    "resident_gaussian_fraction",
    "block_cull_ms",
    "block_concat_ms",
    "gpu_upload_ms",
    "cache_hits",
    "cache_misses",
    "cache_hit_rate",
    "newly_uploaded_blocks",
    "newly_uploaded_gaussians",
    "evicted_blocks",
    "evicted_gaussians",
    "gpu_resident_bytes",
    "gpu_resident_mb",
    "cpu_resident_bytes",
    "cpu_resident_mb",
    "peak_vram",
    "peak_vram_mb",
    "gpu_budget_exceeded",
    "active_set_exceeds_budget",
    "camera_position_x",
    "camera_position_y",
    "camera_position_z",
]


@dataclass
class CameraSpec:
    name: str
    camtoworld: np.ndarray
    K: np.ndarray
    width: int
    height: int


@dataclass
class CameraPath:
    cameras: List[CameraSpec]
    description: str


@dataclass
class NvmlState:
    handle: object


def _nan() -> float:
    return float("nan")


def _default_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = 0.5 * float(width) / math.tan(math.radians(fov_deg) * 0.5)
    return np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _vertical_fov_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = 0.5 * float(height) / math.tan(math.radians(fov_deg) * 0.5)
    return np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _look_at_camtoworld(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])
) -> np.ndarray:
    forward = target - eye
    forward = forward / max(np.linalg.norm(forward), 1e-8)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = right / max(np.linalg.norm(right), 1e-8)
    down = np.cross(forward, right)
    down = down / max(np.linalg.norm(down), 1e-8)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right.astype(np.float32)
    c2w[:3, 1] = down.astype(np.float32)
    c2w[:3, 2] = forward.astype(np.float32)
    c2w[:3, 3] = eye.astype(np.float32)
    return c2w


def _generate_camera_path(
    scene_stats,
    width: int,
    height: int,
    num_cameras: int,
    mode: str,
    fov_deg: float,
    radius_scale: float,
) -> CameraPath:
    if scene_stats.scene_aabb_min is None or scene_stats.scene_aabb_max is None:
        raise ValueError("Generated camera paths require scene bounds from the loaded scene.")
    mins = np.array(scene_stats.scene_aabb_min, dtype=np.float32)
    maxs = np.array(scene_stats.scene_aabb_max, dtype=np.float32)
    center = ((mins + maxs) * 0.5).astype(np.float32)
    extent = maxs - mins
    radius = float(max(np.linalg.norm(extent[:2]), np.linalg.norm(extent), 1.0) * radius_scale)
    height_offset = float(max(extent[2], radius * 0.35, 1.0))
    K = _default_intrinsics(width, height, fov_deg)

    cameras: List[CameraSpec] = []
    for i in range(num_cameras):
        t = i / max(num_cameras, 1)
        angle = 2.0 * math.pi * t
        if mode == "orbit":
            eye = center + np.array(
                [math.cos(angle) * radius, math.sin(angle) * radius, height_offset],
                dtype=np.float32,
            )
        elif mode == "flyover":
            x = (t - 0.5) * radius * 2.0
            y = math.sin(t * 2.0 * math.pi) * radius * 0.35
            eye = center + np.array([x, y, height_offset * 1.4], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported generated path mode: {mode}")
        c2w = _look_at_camtoworld(eye, center)
        cameras.append(CameraSpec(f"{mode}_{i:04d}", c2w, K.copy(), width, height))

    desc = (
        f"generated {mode}: {num_cameras} cameras, bounds_min={mins.tolist()}, "
        f"bounds_max={maxs.tolist()}, fov_deg={fov_deg}, radius_scale={radius_scale}"
    )
    return CameraPath(cameras=cameras, description=desc)


def _parse_matrix(record: dict, keys: Sequence[str], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    for key in keys:
        if key in record:
            arr = np.array(record[key], dtype=np.float32)
            return arr.reshape(shape)
    return None


def _viser_export_to_render_camtoworld(c2w: np.ndarray, scale_ratio: float) -> np.ndarray:
    c2w = c2w.copy()
    # nerfview writes camera paths with a local 180-degree X rotation and
    # translations divided by its viewer scale. Undo both to match live renders.
    c2w[:3, 1:3] *= -1.0
    c2w[:3, 3] *= float(scale_ratio)
    return c2w


def _load_viser_camera_path(data: dict, path: Path, default_width: int, default_height: int) -> CameraPath:
    records = data["camera_path"]
    if not isinstance(records, list):
        raise ValueError("Viser camera path JSON field 'camera_path' must be a list.")

    width = int(data.get("render_width", data.get("width", default_width)))
    height = int(data.get("render_height", data.get("height", default_height)))
    default_fov = float(data.get("default_fov", data.get("fov_deg", 60.0)))
    scale_ratio = float(data.get("scale_ratio", 10.0))

    cameras: List[CameraSpec] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Viser camera path record {idx} must be an object.")
        c2w = _parse_matrix(record, ["camera_to_world", "camtoworld", "c2w"], (4, 4))
        if c2w is None:
            raise ValueError(f"Viser camera path record {idx} has no camera_to_world matrix.")

        cam_width = int(record.get("width", width))
        cam_height = int(record.get("height", height))
        fov_deg = float(record.get("fov", default_fov))
        cameras.append(
            CameraSpec(
                f"viser_{idx:06d}",
                _viser_export_to_render_camtoworld(c2w, scale_ratio).astype(np.float32),
                _vertical_fov_intrinsics(cam_width, cam_height, fov_deg),
                cam_width,
                cam_height,
            )
        )

    return CameraPath(
        cameras=cameras,
        description=(
            f"viser camera path from {path}: {len(cameras)} sampled cameras, "
            f"scale_ratio={scale_ratio}"
        ),
    )


def _load_json_camera_path(path: Path, default_width: int, default_height: int) -> CameraPath:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        if "camera_path" in data:
            return _load_viser_camera_path(data, path, default_width, default_height)
        records = data.get("cameras", data.get("frames"))
    else:
        records = None
    if records is None:
        raise ValueError("Camera path JSON must be a list or contain 'cameras'/'frames'/'camera_path'.")

    global_K = _parse_matrix(data, ["K", "intrinsics"], (3, 3)) if isinstance(data, dict) else None
    width = int(data.get("width", default_width)) if isinstance(data, dict) else default_width
    height = int(data.get("height", default_height)) if isinstance(data, dict) else default_height
    fov_deg = float(data.get("fov_deg", 60.0)) if isinstance(data, dict) else 60.0

    cameras: List[CameraSpec] = []
    for idx, record in enumerate(records):
        c2w = _parse_matrix(
            record,
            ["camtoworld", "camera_to_world", "c2w", "transform_matrix"],
            (4, 4),
        )
        if c2w is None:
            w2c = _parse_matrix(record, ["world_to_camera", "w2c", "viewmat", "view_matrix"], (4, 4))
            if w2c is None:
                raise ValueError(f"Camera JSON record {idx} has no c2w/camtoworld or w2c matrix.")
            c2w = np.linalg.inv(w2c).astype(np.float32)

        cam_width = int(record.get("width", width))
        cam_height = int(record.get("height", height))
        K = _parse_matrix(record, ["K", "intrinsics"], (3, 3))
        if K is None:
            K = global_K.copy() if global_K is not None else _default_intrinsics(cam_width, cam_height, fov_deg)
        cameras.append(
            CameraSpec(
                str(record.get("name", f"json_{idx:04d}")),
                c2w.astype(np.float32),
                K.astype(np.float32),
                cam_width,
                cam_height,
            )
        )
    return CameraPath(cameras=cameras, description=f"json camera path from {path}")


def _load_image_list(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _load_colmap_camera_path(
    data_dir: Path,
    width: int,
    height: int,
    data_factor: float,
    normalize_world_space: bool,
    train_data_dir: Optional[Path],
    image_list: Optional[Iterable[str]],
) -> CameraPath:
    try:
        from datasets.normalize import align_principal_axes, similarity_from_cameras, transform_cameras, transform_points
        from pycolmap import SceneManager
    except ImportError as exc:
        raise ImportError(
            "COLMAP path mode requires pycolmap and gsplat examples dependencies."
        ) from exc

    def colmap_sparse_dir(root: Path) -> Path:
        candidate = root / "sparse" / "0"
        if candidate.exists():
            return candidate
        candidate = root / "sparse"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not find sparse/0 or sparse under {root}")

    def read_cameras(root: Path) -> Tuple[List[Tuple[str, int, np.ndarray]], Dict[int, object], np.ndarray]:
        manager = SceneManager(str(colmap_sparse_dir(root)))
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()
        bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 4)
        rows = []
        for image in manager.images.values():
            rot = image.R().astype(np.float32)
            trans = image.tvec.reshape(3, 1).astype(np.float32)
            w2c = np.concatenate([np.concatenate([rot, trans], axis=1), bottom], axis=0)
            rows.append((image.name, int(image.camera_id), w2c))
        rows.sort(key=lambda item: item[0])
        return rows, manager.cameras, manager.points3D.astype(np.float32)

    def normalization_transform(camtoworlds: np.ndarray, points: np.ndarray) -> np.ndarray:
        transform1 = similarity_from_cameras(camtoworlds)
        transformed_points = transform_points(transform1, points)
        transform2 = align_principal_axes(transformed_points)
        transform = transform2 @ transform1
        transformed_points = transform_points(transform2, transformed_points)
        if np.median(transformed_points[:, 2]) > np.mean(transformed_points[:, 2]):
            transform3 = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            transform = transform3 @ transform
        return transform.astype(np.float32)

    transform = None
    if normalize_world_space:
        source_dir = train_data_dir or data_dir
        train_rows, _, train_points = read_cameras(source_dir)
        train_c2w = np.linalg.inv(np.stack([row[2] for row in train_rows], axis=0))
        transform = normalization_transform(train_c2w, train_points)

    rows, cameras_by_id, _ = read_cameras(data_dir)
    image_filter = set(image_list or [])
    if image_filter:
        rows = [row for row in rows if row[0] in image_filter]
        missing = sorted(image_filter - {row[0] for row in rows})
        if missing:
            raise ValueError(f"{len(missing)} requested COLMAP images were not found: {missing[:5]}")
    if not rows:
        raise ValueError(f"No COLMAP cameras found in {data_dir}")

    c2ws = np.linalg.inv(np.stack([row[2] for row in rows], axis=0)).astype(np.float32)
    if transform is not None:
        c2ws = transform_cameras(transform, c2ws)

    cameras: List[CameraSpec] = []
    for idx, (name, camera_id, _) in enumerate(rows):
        camera = cameras_by_id[camera_id]
        fx = float(camera.fx) / float(data_factor)
        fy = float(camera.fy) / float(data_factor)
        cx = float(camera.cx) / float(data_factor)
        cy = float(camera.cy) / float(data_factor)
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        base_width = int(round(float(camera.width) / float(data_factor)))
        base_height = int(round(float(camera.height) / float(data_factor)))
        cam_width = width if width > 0 else base_width
        cam_height = height if height > 0 else base_height
        if cam_width != base_width:
            K[0, :] *= float(cam_width) / float(base_width)
        if cam_height != base_height:
            K[1, :] *= float(cam_height) / float(base_height)
        cameras.append(CameraSpec(name, c2ws[idx], K, cam_width, cam_height))

    desc = f"COLMAP cameras from {data_dir}"
    if normalize_world_space:
        desc += f", normalized using {train_data_dir or data_dir}"
    return CameraPath(cameras=cameras, description=desc)


def _init_nvml(device: torch.device) -> Optional[NvmlState]:
    if pynvml is None or device.type != "cuda":
        return None
    try:
        pynvml.nvmlInit()
        index = device.index if device.index is not None else torch.cuda.current_device()
        return NvmlState(handle=pynvml.nvmlDeviceGetHandleByIndex(index))
    except Exception:
        return None


def _gpu_utilization(nvml_state: Optional[NvmlState]) -> float:
    if nvml_state is None:
        return _nan()
    try:
        return float(pynvml.nvmlDeviceGetUtilizationRates(nvml_state.handle).gpu)
    except Exception:
        return _nan()


def _gpu_memory_nvml(nvml_state: Optional[NvmlState]) -> Tuple[float, float]:
    if nvml_state is None:
        return _nan(), _nan()
    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_state.handle)
        return float(mem.used / (1024**3)), float(mem.total / (1024**3))
    except Exception:
        return _nan(), _nan()


def _torch_cuda_memory_snapshot(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "allocated": _nan(),
            "reserved": _nan(),
            "peak_allocated": _nan(),
            "peak_reserved": _nan(),
        }

    peak_reserved = _nan()
    if hasattr(torch.cuda, "max_memory_reserved"):
        peak_reserved = float(torch.cuda.max_memory_reserved(device))
    return {
        "allocated": float(torch.cuda.memory_allocated(device)),
        "reserved": float(torch.cuda.memory_reserved(device)),
        "peak_allocated": float(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": peak_reserved,
    }


def _memory_stats(
    device: torch.device,
    process: Optional[object],
    cumulative_peak_allocated: float = _nan(),
) -> Tuple[float, float, float, float, float, float, float]:
    vram_alloc = vram_reserved = vram_peak = _nan()
    if device.type == "cuda":
        vram_alloc = torch.cuda.memory_allocated(device) / 1e9
        vram_reserved = torch.cuda.memory_reserved(device) / 1e9
        if np.isfinite(cumulative_peak_allocated):
            vram_peak = cumulative_peak_allocated / 1e9
        else:
            vram_peak = torch.cuda.max_memory_allocated(device) / 1e9

    process_ram = process_cpu = system_ram = system_percent = _nan()
    if psutil is not None:
        proc = process or psutil.Process(os.getpid())
        process_ram = proc.memory_info().rss / (1024**3)
        process_cpu = proc.cpu_percent(interval=None)
        system_mem = psutil.virtual_memory()
        system_ram = system_mem.used / (1024**3)
        system_percent = system_mem.percent
    return vram_alloc, vram_reserved, vram_peak, process_ram, process_cpu, system_ram, system_percent


def _render_tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    return (
        image.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .byte()
        .cpu()
        .numpy()
    )


def _validate_visual_output_args(args: argparse.Namespace, cameras: Sequence[CameraSpec]) -> None:
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1.")
    if args.max_saved_frames < 0:
        raise ValueError("--max-saved-frames must be >= 0.")
    if (args.save_frames_dir is not None or args.save_video is not None) and imageio is None:
        raise ImportError("Saving frames/video requires imageio. Install imageio in the benchmark environment.")
    if args.save_video is not None:
        sizes = {(camera.width, camera.height) for camera in cameras}
        if len(sizes) != 1:
            raise ValueError(
                "--save-video requires all rendered frames to have the same width/height. "
                "Pass fixed --width and --height for COLMAP paths."
            )


def _should_save_frame(frame_index: int, save_every: int, max_saved_frames: int, saved_frames: int) -> bool:
    if frame_index % save_every != 0:
        return False
    return max_saved_frames == 0 or saved_frames < max_saved_frames


@torch.no_grad()
def _render_scene_once(
    scene,
    renderer: GsplatRenderer,
    camera: CameraSpec,
    device: torch.device,
    sh_degree: int,
    packed: bool,
    rasterize_mode: str,
    camera_model: str,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    eps2d: float,
    log_gpu_events: bool,
    return_image: bool = False,
) -> Tuple[float, float, float, Optional[torch.Tensor], object]:
    c2w = torch.from_numpy(camera.camtoworld).to(device=device, dtype=torch.float32)
    K = torch.from_numpy(camera.K).to(device=device, dtype=torch.float32)
    frame_camera = FrameCamera(
        c2w=c2w,
        K=K,
        width=camera.width,
        height=camera.height,
    )
    render_settings = RenderSettings(
        width=camera.width,
        height=camera.height,
        packed=packed,
        sh_degree=sh_degree,
        rasterize_mode=rasterize_mode,
        camera_model=camera_model,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        eps2d=eps2d,
        render_mode="RGB",
    )
    active_splats = scene.prepare_frame(frame_camera, render_settings)
    result = renderer.render(
        active_splats,
        frame_camera,
        render_settings,
        log_gpu_events=log_gpu_events,
    )
    image = result.image[..., :3].clamp(0.0, 1.0).detach() if return_image else None
    return (
        result.render_time_ms,
        result.gpu_render_time_ms,
        result.visible_gaussians,
        image,
        active_splats.scene_metrics,
    )


def _summary(
    values: List[float],
    fps_values: List[float],
    vram_values: List[float],
    torch_reserved_values: List[float],
    torch_frame_peak_values: List[float],
    torch_frame_peak_reserved_values: List[float],
    args: argparse.Namespace,
    camera_path: CameraPath,
    scene_stats,
    scene_paths: Sequence[Path],
) -> dict:
    frame_times = np.array(values, dtype=np.float64)
    fps = np.array(fps_values, dtype=np.float64)
    vram = np.array(vram_values, dtype=np.float64)
    finite_vram = vram[np.isfinite(vram)]
    torch_reserved = np.array(torch_reserved_values, dtype=np.float64)
    finite_torch_reserved = torch_reserved[np.isfinite(torch_reserved)]
    torch_frame_peak = np.array(torch_frame_peak_values, dtype=np.float64)
    finite_torch_frame_peak = torch_frame_peak[np.isfinite(torch_frame_peak)]
    torch_frame_peak_reserved = np.array(torch_frame_peak_reserved_values, dtype=np.float64)
    finite_torch_frame_peak_reserved = torch_frame_peak_reserved[
        np.isfinite(torch_frame_peak_reserved)
    ]
    return {
        "average_fps": float(np.mean(fps)) if fps.size else _nan(),
        "median_fps": float(np.median(fps)) if fps.size else _nan(),
        "one_percent_low_fps": float(np.percentile(fps, 1)) if fps.size else _nan(),
        "average_frame_time_ms": float(np.mean(frame_times)) if frame_times.size else _nan(),
        "median_frame_time_ms": float(np.median(frame_times)) if frame_times.size else _nan(),
        "peak_frame_time_ms": float(np.max(frame_times)) if frame_times.size else _nan(),
        "peak_vram_allocated_gb": float(np.max(finite_vram)) if finite_vram.size else _nan(),
        "average_vram_allocated_gb": float(np.mean(finite_vram)) if finite_vram.size else _nan(),
        "peak_torch_reserved_mb": float(np.max(finite_torch_reserved))
        if finite_torch_reserved.size
        else _nan(),
        "average_torch_reserved_mb": float(np.mean(finite_torch_reserved))
        if finite_torch_reserved.size
        else _nan(),
        "peak_torch_frame_peak_allocated_mb": float(np.max(finite_torch_frame_peak))
        if finite_torch_frame_peak.size
        else _nan(),
        "average_torch_frame_peak_allocated_mb": float(np.mean(finite_torch_frame_peak))
        if finite_torch_frame_peak.size
        else _nan(),
        "peak_torch_frame_peak_reserved_mb": float(np.max(finite_torch_frame_peak_reserved))
        if finite_torch_frame_peak_reserved.size
        else _nan(),
        "average_torch_frame_peak_reserved_mb": float(
            np.mean(finite_torch_frame_peak_reserved)
        )
        if finite_torch_frame_peak_reserved.size
        else _nan(),
        "number_of_gaussians": int(scene_stats.total_gaussians),
        "number_of_frames_rendered": int(frame_times.size),
        "scene_type": scene_stats.scene_type,
        "total_blocks": scene_stats.total_blocks,
        "checkpoint_path": [str(path) for path in scene_paths],
        "camera_path_description": camera_path.description,
        "width": args.width,
        "height": args.height,
        "num_warmup": args.num_warmup,
        "num_repeat": args.num_repeat,
        "log_gpu_events": args.log_gpu_events,
        "packed": args.packed,
        "rasterize_mode": args.rasterize_mode,
        "camera_model": args.camera_model,
        "block_residency": args.block_residency,
        "block_keepalive_frames": args.block_keepalive_frames,
        "gpu_block_budget_mb": args.gpu_block_budget_mb,
        "block_cache_policy": args.block_cache_policy,
        "save_frames_dir": str(args.save_frames_dir) if args.save_frames_dir else None,
        "save_video": str(args.save_video) if args.save_video else None,
        "video_fps": args.video_fps,
        "save_every": args.save_every,
        "max_saved_frames": args.max_saved_frames,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a reproducible camera path through gsplat checkpoints and log benchmark CSV metrics."
    )
    parser.add_argument("--scene", type=Path, nargs="+", default=None, help="Flat .pt/.ckpt checkpoint(s), Gaussian .ply file(s), or one blocked scene asset.")
    parser.add_argument("--scene-type", choices=["auto", "flat", "blocked"], default="auto")
    parser.add_argument("--ckpt", type=Path, nargs="+", default=None, help="Legacy alias for --scene with flat checkpoint(s) or Gaussian .ply file(s).")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--width", type=int, default=1600, help="Render width. Use 0 with --colmap-data-dir to keep COLMAP width.")
    parser.add_argument("--height", type=int, default=900, help="Render height. Use 0 with --colmap-data-dir to keep COLMAP height.")
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-repeat", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-gpu-events", action="store_true", help="Log CUDA event timing for the rasterization call.")
    parser.add_argument("--save-frames-dir", type=Path, default=None, help="Optional directory for PNG frame dumps. Saving is excluded from frame_time_ms.")
    parser.add_argument("--save-video", type=Path, default=None, help="Optional output video path, e.g. debug_path.mp4. Saving is excluded from frame_time_ms.")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--save-every", type=int, default=1, help="Save every Nth recorded frame when frame/video output is enabled.")
    parser.add_argument("--max-saved-frames", type=int, default=0, help="Maximum visual frames to save; 0 means no limit.")

    parser.add_argument("--camera-path-json", type=Path, default=None, help="JSON camera path with c2w/camtoworld matrices and optional K.")
    parser.add_argument("--colmap-data-dir", type=Path, default=None, help="COLMAP dataset directory to use as fixed camera path.")
    parser.add_argument("--colmap-train-data-dir", type=Path, default=None, help="Optional train COLMAP dir for simple_trainer-style normalization.")
    parser.add_argument("--image-list", type=Path, default=None, help="Optional newline-delimited COLMAP image names.")
    parser.add_argument("--data-factor", type=float, default=1.0)
    parser.add_argument("--normalize-world-space", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--path-mode", choices=["orbit", "flyover"], default="orbit", help="Generated camera path mode when no JSON/COLMAP path is provided.")
    parser.add_argument("--num-cameras", type=int, default=120, help="Number of cameras for generated paths.")
    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument("--radius-scale", type=float, default=1.25)

    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--packed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rasterize-mode", choices=["classic", "antialiased"], default="classic")
    parser.add_argument("--camera-model", choices=["pinhole", "ortho", "fisheye"], default="pinhole")
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1e10)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--eps2d", type=float, default=0.3)
    parser.add_argument("--block-residency", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--block-keepalive-frames", type=int, default=0)
    parser.add_argument("--gpu-block-budget-mb", type=float, default=None)
    parser.add_argument("--block-cache-policy", choices=["lru"], default="lru")
    parser.add_argument("--force-all-blocks-active", action="store_true")
    return parser.parse_args()


def _resolve_scene_paths(args: argparse.Namespace) -> List[Path]:
    if args.scene is not None:
        if args.ckpt is not None:
            raise ValueError("Use either --scene or legacy --ckpt, not both.")
        return [Path(path) for path in args.scene]
    if args.ckpt is not None:
        return [Path(path) for path in args.ckpt]
    raise ValueError("Pass --scene PATH, or legacy --ckpt PATH.")


def main() -> None:
    args = parse_args()
    if args.width < 0 or args.height < 0:
        raise ValueError("--width and --height must be non-negative.")
    if args.num_warmup < 0 or args.num_repeat < 1:
        raise ValueError("--num-warmup must be >= 0 and --num-repeat must be >= 1.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "This gsplat backend benchmark requires a CUDA device; the local rasterizer "
            "does not support CPU rendering."
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false in this process. "
            "Run in an environment with GPU access."
        )
    scene_paths = _resolve_scene_paths(args)
    culling_mode = (
        "force_all_blocks_active"
        if args.force_all_blocks_active
        else "normal_frustum_culling"
    )
    scene = SceneLoader.load(
        scene_paths,
        device=device,
        scene_type=args.scene_type,
        block_residency=args.block_residency,
        block_keepalive_frames=args.block_keepalive_frames,
        gpu_block_budget_mb=args.gpu_block_budget_mb,
        block_cache_policy=args.block_cache_policy,
        culling_mode=culling_mode,
    )
    scene_stats = scene.get_scene_stats()
    renderer = GsplatRenderer(device)
    if scene_stats.sh_degree is not None:
        args.sh_degree = min(args.sh_degree, scene_stats.sh_degree)

    if args.camera_path_json is not None:
        camera_path = _load_json_camera_path(args.camera_path_json, args.width, args.height)
    elif args.colmap_data_dir is not None:
        camera_path = _load_colmap_camera_path(
            args.colmap_data_dir,
            args.width,
            args.height,
            args.data_factor,
            args.normalize_world_space,
            args.colmap_train_data_dir,
            _load_image_list(args.image_list),
        )
    else:
        if args.width == 0 or args.height == 0:
            raise ValueError("Generated paths require positive --width and --height.")
        camera_path = _generate_camera_path(
            scene_stats,
            args.width,
            args.height,
            args.num_cameras,
            args.path_mode,
            args.fov_deg,
            args.radius_scale,
        )

    if not camera_path.cameras:
        raise ValueError("Camera path contains no cameras.")
    _validate_visual_output_args(args, camera_path.cameras)
    if args.save_frames_dir is not None:
        args.save_frames_dir.mkdir(parents=True, exist_ok=True)
    if args.save_video is not None:
        args.save_video.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Loaded {scene_stats.total_gaussians:,} Gaussians from "
        f"{len(scene_paths)} scene file(s), type={scene_stats.scene_type}."
    )
    print(f"Camera path: {camera_path.description}")

    with torch.no_grad():
        for idx in range(args.num_warmup):
            camera = camera_path.cameras[idx % len(camera_path.cameras)]
            _render_scene_once(
                scene,
                renderer,
                camera,
                device,
                args.sh_degree,
                args.packed,
                args.rasterize_mode,
                args.camera_model,
                args.near_plane,
                args.far_plane,
                args.radius_clip,
                args.eps2d,
                False,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    nvml_state = _init_nvml(device)
    process = psutil.Process(os.getpid()) if psutil is not None else None
    if process is not None:
        process.cpu_percent(interval=None)
    frame_times: List[float] = []
    fps_values: List[float] = []
    vram_values: List[float] = []
    torch_reserved_values: List[float] = []
    torch_frame_peak_values: List[float] = []
    torch_frame_peak_reserved_values: List[float] = []
    cumulative_peak_allocated = _nan()
    if device.type == "cuda":
        cumulative_peak_allocated = float(torch.cuda.max_memory_allocated(device))
    start_time = time.perf_counter()
    frame_index = 0
    saved_frames = 0
    video_writer = imageio.get_writer(args.save_video, fps=args.video_fps) if args.save_video else None

    try:
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            f.flush()

            for repeat_index in range(args.num_repeat):
                for camera_index, camera in enumerate(camera_path.cameras):
                    save_this = _should_save_frame(
                        frame_index, args.save_every, args.max_saved_frames, saved_frames
                    ) and (args.save_frames_dir is not None or video_writer is not None)
                    if device.type == "cuda":
                        cumulative_peak_allocated = max(
                            cumulative_peak_allocated,
                            float(torch.cuda.max_memory_allocated(device)),
                        )
                        torch.cuda.reset_peak_memory_stats(device)
                    frame_time_ms, gpu_time_ms, visible_count, image, scene_metrics = _render_scene_once(
                        scene,
                        renderer,
                        camera,
                        device,
                        args.sh_degree,
                        args.packed,
                        args.rasterize_mode,
                        args.camera_model,
                        args.near_plane,
                        args.far_plane,
                        args.radius_clip,
                        args.eps2d,
                        args.log_gpu_events,
                        save_this,
                    )
                    torch_memory = _torch_cuda_memory_snapshot(device)
                    cumulative_peak_allocated = max(
                        cumulative_peak_allocated,
                        torch_memory["peak_allocated"],
                    )
                    fps = 1000.0 / max(frame_time_ms, 1e-9)
                    (
                        vram_alloc,
                        vram_reserved,
                        vram_peak,
                        process_ram,
                        process_cpu,
                        system_ram,
                        system_percent,
                    ) = _memory_stats(device, process, cumulative_peak_allocated)
                    torch_allocated_mb = torch_memory["allocated"] / (1024.0 * 1024.0)
                    torch_reserved_mb = torch_memory["reserved"] / (1024.0 * 1024.0)
                    torch_frame_peak_allocated_mb = torch_memory["peak_allocated"] / (
                        1024.0 * 1024.0
                    )
                    torch_frame_peak_reserved_mb = torch_memory["peak_reserved"] / (
                        1024.0 * 1024.0
                    )
                    gpu_util = _gpu_utilization(nvml_state)
                    vram_used, vram_total = _gpu_memory_nvml(nvml_state)
                    elapsed_sec = time.perf_counter() - start_time
                    row = {
                        "timestamp": time.time(),
                        "elapsed_sec": elapsed_sec,
                        "pid": os.getpid(),
                        "frame_index": frame_index,
                        "repeat_index": repeat_index,
                        "camera_index": camera_index,
                        "width": camera.width,
                        "height": camera.height,
                        "frame_time_ms": frame_time_ms,
                        "fps": fps,
                        "gpu_render_time_ms": gpu_time_ms,
                        "num_gaussians_total": scene_stats.total_gaussians,
                        "num_gaussians_rendered": visible_count,
                        "visible_gaussians": visible_count,
                        "vram_allocated_gb": vram_alloc,
                        "vram_reserved_gb": vram_reserved,
                        "vram_max_allocated_gb": vram_peak,
                        "torch_allocated_mb": torch_allocated_mb,
                        "torch_reserved_mb": torch_reserved_mb,
                        "torch_frame_peak_allocated_mb": torch_frame_peak_allocated_mb,
                        "torch_frame_peak_reserved_mb": torch_frame_peak_reserved_mb,
                        "process_ram_gb": process_ram,
                        "process_cpu_percent": process_cpu,
                        "system_ram_used_gb": system_ram,
                        "system_ram_percent": system_percent,
                        "gpu_utilization_percent": gpu_util,
                        "gpu_util_percent": gpu_util,
                        "vram_used_gb": vram_used,
                        "vram_total_gb": vram_total,
                        "render_frame_index": frame_index + 1,
                        "render_frame_time_ms": frame_time_ms,
                        "render_fps": fps,
                        "render_width": camera.width,
                        "render_height": camera.height,
                        "active_blocks": scene_metrics.active_blocks,
                        "active_block_ids": scene_metrics.active_block_ids,
                        "active_gaussians": scene_metrics.active_gaussians,
                        "active_gaussian_fraction": scene_metrics.active_gaussian_fraction,
                        "total_gaussians": scene_stats.total_gaussians,
                        "total_blocks": scene_metrics.total_blocks,
                        "resident_blocks": scene_metrics.resident_blocks,
                        "resident_gaussians": scene_metrics.resident_gaussians,
                        "resident_gaussian_fraction": scene_metrics.resident_gaussian_fraction,
                        "block_cull_ms": scene_metrics.block_cull_ms,
                        "block_concat_ms": scene_metrics.block_concat_ms,
                        "gpu_upload_ms": scene_metrics.gpu_upload_ms,
                        "cache_hits": scene_metrics.cache_hits,
                        "cache_misses": scene_metrics.cache_misses,
                        "cache_hit_rate": scene_metrics.cache_hit_rate,
                        "newly_uploaded_blocks": scene_metrics.newly_uploaded_blocks,
                        "newly_uploaded_gaussians": scene_metrics.newly_uploaded_gaussians,
                        "evicted_blocks": scene_metrics.evicted_blocks,
                        "evicted_gaussians": scene_metrics.evicted_gaussians,
                        "gpu_resident_bytes": scene_metrics.gpu_resident_bytes,
                        "gpu_resident_mb": scene_metrics.gpu_resident_mb,
                        "cpu_resident_bytes": scene_metrics.cpu_resident_bytes,
                        "cpu_resident_mb": scene_metrics.cpu_resident_mb,
                        "peak_vram": vram_peak * 1e9 if np.isfinite(vram_peak) else _nan(),
                        "peak_vram_mb": vram_peak * 1e3 if np.isfinite(vram_peak) else _nan(),
                        "gpu_budget_exceeded": scene_metrics.gpu_budget_exceeded,
                        "active_set_exceeds_budget": scene_metrics.active_set_exceeds_budget,
                        "camera_position_x": float(camera.camtoworld[0, 3]),
                        "camera_position_y": float(camera.camtoworld[1, 3]),
                        "camera_position_z": float(camera.camtoworld[2, 3]),
                    }
                    writer.writerow(row)
                    f.flush()

                    if save_this and image is not None:
                        frame_uint8 = _render_tensor_to_uint8(image)
                        if args.save_frames_dir is not None:
                            frame_name = (
                                f"frame_{frame_index:06d}_repeat_{repeat_index:03d}_"
                                f"camera_{camera_index:04d}.png"
                            )
                            imageio.imwrite(args.save_frames_dir / frame_name, frame_uint8)
                        if video_writer is not None:
                            video_writer.append_data(frame_uint8)
                        saved_frames += 1

                    frame_times.append(frame_time_ms)
                    fps_values.append(fps)
                    vram_values.append(vram_alloc)
                    torch_reserved_values.append(torch_reserved_mb)
                    torch_frame_peak_values.append(torch_frame_peak_allocated_mb)
                    torch_frame_peak_reserved_values.append(torch_frame_peak_reserved_mb)
                    frame_index += 1
    finally:
        if video_writer is not None:
            video_writer.close()

    if pynvml is not None and nvml_state is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    summary = _summary(
        frame_times,
        fps_values,
        vram_values,
        torch_reserved_values,
        torch_frame_peak_values,
        torch_frame_peak_reserved_values,
        args,
        camera_path,
        scene_stats,
        scene_paths,
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        f"Wrote {frame_index} frame rows to {args.output} and summary to {summary_path}. "
        f"Average FPS: {summary['average_fps']:.2f}"
    )


if __name__ == "__main__":
    main()
