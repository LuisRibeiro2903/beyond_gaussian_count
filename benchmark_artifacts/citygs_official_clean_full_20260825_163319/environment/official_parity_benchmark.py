#!/usr/bin/env python
"""Instrumentation-only deterministic replay for pinned CityGaussian V1.

This harness calls the released PLY loaders, BlockedGaussian constructor,
visibility/LoD selector, active-set assembly, and GraphDECO rasterizer.  It does
not replace or optimize those method stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from gaussian_renderer import render, render_lod
from scene.cameras import MiniCam
from scene.gaussian_model import BlockedGaussian, GaussianModelLOD
from utils.graphics_utils import getProjectionMatrix


OFFICIAL_COMMIT = "e16c526f78ab01d377b13b3e6862dcc93ddbe953"
GRAPHDECO_COMMIT = "59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d"
SIMPLE_KNN_COMMIT = "44f764299fa305faf6ec5ebd99939e0508331503"
ORACLE_INDICES = [0, 131, 262, 392, 523, 654, 785, 916, 1047, 1177, 1308, 1439]
LEVEL_LABELS = ["light_50_vq", "light_66_vq", "light_75_vq"]
AABB = [-3.5, -4.0, -10.0, 4.5, 2.0, 10.0]
BLOCK_DIM = [6, 6, 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("lod", "flat"), required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--finest", type=Path, required=True)
    parser.add_argument("--middle", type=Path)
    parser.add_argument("--coarsest", type=Path)
    parser.add_argument("--frames", choices=("all", "oracle"), default="all")
    parser.add_argument("--save-oracle-images", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_text(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, check=False, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def gpu_memory() -> dict[str, Any]:
    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    query = command_text(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
    )
    whole_used = whole_total = None
    if query:
        try:
            used_mib, total_mib = [int(v.strip()) for v in query.splitlines()[0].split(",")]
            whole_used = used_mib * 1024 * 1024
            whole_total = total_mib * 1024 * 1024
        except Exception:
            pass
    return {
        "torch_current_allocated_bytes": allocated,
        "torch_current_reserved_bytes": reserved,
        "torch_peak_allocated_bytes": peak_allocated,
        "torch_peak_reserved_bytes": peak_reserved,
        "whole_device_used_bytes": whole_used,
        "whole_device_total_bytes": whole_total,
    }


def load_trajectory(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload["camera_path"]
    width = int(payload.get("render_width", 1920))
    height = int(payload.get("render_height", 1080))
    default_fov = float(payload.get("default_fov", 50.0))
    scale_ratio = float(payload.get("scale_ratio", 10.0))
    cameras = []
    for index, record in enumerate(records):
        source = np.asarray(record["camera_to_world"], dtype=np.float32).reshape(4, 4)
        converted = source.copy()
        converted[:3, 1:3] *= -1.0
        converted[:3, 3] *= scale_ratio
        frame_width = int(record.get("width", width))
        frame_height = int(record.get("height", height))
        fov_y = math.radians(float(record.get("fov", default_fov)))
        focal = 0.5 * frame_height / math.tan(fov_y * 0.5)
        fov_x = 2.0 * math.atan(frame_width / (2.0 * focal))
        world_view = torch.linalg.inv(torch.from_numpy(converted)).T.contiguous().cuda()
        projection = getProjectionMatrix(0.01, 100.0, fov_x, fov_y).T.contiguous().cuda()
        full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
        camera = MiniCam(
            frame_width,
            frame_height,
            fov_y,
            fov_x,
            0.01,
            100.0,
            world_view,
            full_projection,
        )
        cameras.append(
            {
                "index": index,
                "camera_id": f"viser_{index:06d}",
                "camera": camera,
                "source_camera_to_world": source,
                "converted_camera_to_world": converted,
                "world_view_transform": world_view,
                "projection_transform": projection,
                "full_projection_transform": full_projection,
                "fov_x": fov_x,
                "fov_y": fov_y,
                "width": frame_width,
                "height": frame_height,
            }
        )
    metadata = {
        "frame_count": len(cameras),
        "resolution": [width, height],
        "default_fov_degrees": default_fov,
        "scale_ratio": scale_ratio,
        "conversion": "flip source camera Y/Z axes and multiply translation by scale_ratio",
        "matrix_semantics": "world_view=inverse(converted_c2w).T; projection=getProjectionMatrix(...).T; full=world_view@projection",
    }
    return cameras, metadata


def camera_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_index": row["index"],
        "camera_id": row["camera_id"],
        "camera_position": row["camera"].camera_center.detach().cpu().tolist(),
        "fov_x": row["fov_x"],
        "fov_y": row["fov_y"],
        "width": row["width"],
        "height": row["height"],
        "source_camera_to_world": row["source_camera_to_world"].tolist(),
        "converted_camera_to_world": row["converted_camera_to_world"].tolist(),
        "world_view_transform": row["world_view_transform"].detach().cpu().tolist(),
        "projection_transform": row["projection_transform"].detach().cpu().tolist(),
        "full_projection_transform": row["full_projection_transform"].detach().cpu().tolist(),
    }


def load_ply(path: Path) -> GaussianModelLOD:
    model = GaussianModelLOD(sh_degree=2, device="cuda")
    model.load_ply(str(path))
    if int(model.get_xyz.shape[0]) <= 0:
        raise RuntimeError(f"No Gaussians loaded from {path}")
    model.active_sh_degree = model.max_sh_degree
    return model


def load_lod(paths: list[Path]) -> list[BlockedGaussian]:
    params = SimpleNamespace(sh_degree=2, block_dim=BLOCK_DIM, aabb=AABB)
    levels = []
    for path in paths:
        model = load_ply(path)
        blocked = BlockedGaussian(model, params, compute_cov3D_python=False)
        levels.append(blocked)
        del model
        torch.cuda.empty_cache()
    return levels


def pipeline() -> SimpleNamespace:
    return SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)


def selection_record(instrumentation: dict[str, Any], levels: list[BlockedGaussian]) -> dict[str, Any]:
    visible = [int(v) for v in instrumentation["visible_cell_indices_tensor"].detach().cpu().flatten().tolist()]
    lods = [int(v) for v in instrumentation["selected_lod_indices_tensor"].detach().cpu().flatten().tolist()]
    nearest = int(instrumentation["nearest_cell_id_tensor"].detach().cpu().item())
    pre_mask = instrumentation["pre_fallback_mask_tensor"].detach().cpu().flatten()
    final_mask = instrumentation["visible_mask_tensor"].detach().cpu().flatten()
    selected_lod_by_cell = {}
    selected_label_by_cell = {}
    selected_count_by_cell = {}
    count_by_level = [0, 0, 0]
    cells_by_level = [0, 0, 0]
    for cell_id, level_index in zip(visible, lods):
        count = int((levels[level_index].cell_idxs[cell_id + 1] - levels[level_index].cell_idxs[cell_id]).item())
        key = str(cell_id)
        selected_lod_by_cell[key] = level_index
        selected_label_by_cell[key] = LEVEL_LABELS[level_index]
        selected_count_by_cell[key] = count
        count_by_level[level_index] += count
        cells_by_level[level_index] += 1
    return {
        "visible_cell_ids": visible,
        "nearest_cell_id": nearest,
        "fallback_changed_mask": bool(final_mask[nearest].item() and not pre_mask[nearest].item()),
        "selected_lod_by_cell": selected_lod_by_cell,
        "selected_level_label_by_cell": selected_label_by_cell,
        "selected_gaussian_count_by_cell": selected_count_by_cell,
        "selected_cell_count_by_level": cells_by_level,
        "selected_gaussian_count_by_level": count_by_level,
        "total_selected_gaussian_count": sum(count_by_level),
    }


@torch.no_grad()
def render_lod_frame(
    camera: MiniCam,
    levels: list[BlockedGaussian],
    background: torch.Tensor,
    save_alpha: bool,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor | None]:
    instrumentation: dict[str, Any] = {}
    result = render_lod(camera, levels, pipeline(), background, instrumentation=instrumentation)
    row = selection_record(instrumentation, levels)
    for key in (
        "visibility_time_ms",
        "lod_selection_time_ms",
        "selection_time_ms",
        "active_set_assembly_time_ms",
        "render_call_time_ms",
        "cuda_rasterization_time_ms",
        "end_to_end_frame_time_ms",
        "radii_visible_gaussian_count",
    ):
        row[key] = instrumentation[key]
    if row["total_selected_gaussian_count"] != instrumentation["total_selected_gaussian_count"]:
        raise RuntimeError("Instrumented selected-count mismatch")
    image = result["render"].detach()
    alpha = None
    if save_alpha:
        override = torch.ones((row["total_selected_gaussian_count"], 3), dtype=torch.float32, device="cuda")
        alpha_rgb = render_lod(camera, levels, pipeline(), background, override_color=override)["render"]
        torch.cuda.synchronize()
        alpha = alpha_rgb[:1].detach()
    return row, image, alpha


@torch.no_grad()
def render_flat_frame(
    camera: MiniCam,
    model: GaussianModelLOD,
    background: torch.Tensor,
    save_alpha: bool,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor | None]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    cuda_start.record()
    result = render(camera, model, pipeline(), background)
    cuda_end.record()
    torch.cuda.synchronize()
    end = time.perf_counter()
    count = int(model.get_xyz.shape[0])
    row = {
        "visibility_time_ms": 0.0,
        "lod_selection_time_ms": 0.0,
        "selection_time_ms": 0.0,
        "active_set_assembly_time_ms": 0.0,
        "render_call_time_ms": (end - start) * 1000.0,
        "cuda_rasterization_time_ms": cuda_start.elapsed_time(cuda_end),
        "end_to_end_frame_time_ms": (end - start) * 1000.0,
        "total_selected_gaussian_count": count,
        "radii_visible_gaussian_count": int((result["radii"] > 0).sum().item()),
    }
    image = result["render"].detach()
    alpha = None
    if save_alpha:
        override = torch.ones((count, 3), dtype=torch.float32, device="cuda")
        alpha_rgb = render(camera, model, pipeline(), background, override_color=override)["render"]
        torch.cuda.synchronize()
        alpha = alpha_rgb[:1].detach()
    return row, image, alpha


def save_arrays(output_dir: Path, frame_index: int, image: torch.Tensor, alpha: torch.Tensor | None) -> None:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    np.save(image_dir / f"frame_{frame_index:04d}_rgb.npy", image.permute(1, 2, 0).cpu().numpy().astype(np.float32))
    if alpha is not None:
        np.save(image_dir / f"frame_{frame_index:04d}_alpha.npy", alpha.permute(1, 2, 0).cpu().numpy().astype(np.float32))


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.mode == "lod" and (args.middle is None or args.coarsest is None):
        raise ValueError("LoD mode requires --middle and --coarsest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cameras, trajectory_metadata = load_trajectory(args.trajectory)
    selected_indices = ORACLE_INDICES if args.frames == "oracle" else list(range(len(cameras)))
    camera_rows = [camera_record(cameras[index]) for index in ORACLE_INDICES]
    (args.output_dir / "camera_matrices.json").write_text(json.dumps(camera_rows, indent=2, allow_nan=False) + "\n")

    asset_paths = [args.finest] if args.mode == "flat" else [args.finest, args.middle, args.coarsest]
    load_start = time.perf_counter()
    if args.mode == "flat":
        scene = load_ply(args.finest)
        resident_count = int(scene.get_xyz.shape[0])
    else:
        scene = load_lod([Path(v) for v in asset_paths])
        resident_count = sum(int(level.feats.shape[0]) for level in scene)
    torch.cuda.synchronize()
    load_time_ms = (time.perf_counter() - load_start) * 1000.0
    memory_after_load = gpu_memory()
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    # Exactly one excluded deterministic warmup at trajectory frame zero.
    if args.mode == "flat":
        warmup_row, warmup_image, warmup_alpha = render_flat_frame(cameras[0]["camera"], scene, background, False)
    else:
        warmup_row, warmup_image, warmup_alpha = render_lod_frame(cameras[0]["camera"], scene, background, False)
    del warmup_image, warmup_alpha
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    memory_after_warmup = gpu_memory()

    records = []
    frames_path = args.output_dir / "frames.jsonl"
    with frames_path.open("w") as handle:
        for measured_index, trajectory_index in enumerate(selected_indices):
            camera_row = cameras[trajectory_index]
            save_image = args.save_oracle_images and trajectory_index in ORACLE_INDICES
            if args.mode == "flat":
                row, image, alpha = render_flat_frame(camera_row["camera"], scene, background, save_image)
            else:
                row, image, alpha = render_lod_frame(camera_row["camera"], scene, background, save_image)
            finite_rgb = bool(torch.isfinite(image).all().item())
            finite_alpha = bool(torch.isfinite(alpha).all().item()) if alpha is not None else None
            if save_image:
                save_arrays(args.output_dir, trajectory_index, image, alpha)
            row.update(
                {
                    "record_kind": "official_citygaussian_frame",
                    "configuration_id": "OFFICIAL-CITYGS-LOD-RELEASE" if args.mode == "lod" else "OFFICIAL-GRAPHDECO-FINEST-FLAT",
                    "repeat": args.repeat,
                    "measured_frame_index": measured_index,
                    "trajectory_frame_index": trajectory_index,
                    "camera_id": camera_row["camera_id"],
                    "resident_gaussian_count": resident_count,
                    "finite_rgb": finite_rgb,
                    "finite_alpha": finite_alpha,
                    "torch_current_allocated_bytes": int(torch.cuda.memory_allocated()),
                    "torch_current_reserved_bytes": int(torch.cuda.memory_reserved()),
                    "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            )
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            records.append(row)
            del image, alpha

    memory_after_run = gpu_memory()
    timing_keys = [
        "visibility_time_ms",
        "lod_selection_time_ms",
        "selection_time_ms",
        "active_set_assembly_time_ms",
        "render_call_time_ms",
        "cuda_rasterization_time_ms",
        "end_to_end_frame_time_ms",
    ]
    summary = {
        "record_kind": "official_citygaussian_repeat_summary",
        "configuration_id": records[0]["configuration_id"],
        "repeat": args.repeat,
        "measured_frame_count": len(records),
        "warmup_policy": "one deterministic render of trajectory frame 0, excluded",
        "warmup_record": warmup_row,
        "timing_semantics": {
            "selection_time_ms": "synchronized visibility plus released Nyquist LoD selection",
            "active_set_assembly_time_ms": "released per-level get_feats, concatenation, and render-input tensor views/casts",
            "render_call_time_ms": "GraphDECO rasterizer setup and synchronized call",
            "cuda_rasterization_time_ms": "CUDA events around GaussianRasterizer invocation",
            "end_to_end_frame_time_ms": "selection start through synchronized rendered output; excludes load, warmup, checks, serialization, and image writes",
        },
        "timing": {key: stats([float(row[key]) for row in records]) for key in timing_keys},
        "end_to_end_fps": 1000.0 / statistics.fmean([float(row["end_to_end_frame_time_ms"]) for row in records]),
        "selected_gaussian_count": stats([float(row["total_selected_gaussian_count"]) for row in records]),
        "resident_gaussian_count": resident_count,
        "load_time_ms": load_time_ms,
        "memory_after_load": memory_after_load,
        "memory_after_warmup_peak_reset": memory_after_warmup,
        "memory_after_run": memory_after_run,
        "all_rgb_finite": all(bool(row["finite_rgb"]) for row in records),
        "alpha_saved_for_oracle_frames": args.save_oracle_images,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    provenance = {
        "official_commit": OFFICIAL_COMMIT,
        "diff_gaussian_rasterization_commit": GRAPHDECO_COMMIT,
        "simple_knn_commit": SIMPLE_KNN_COMMIT,
        "mode": args.mode,
        "repeat": args.repeat,
        "assets": [{"path": str(path), "sha256": sha256_file(Path(path))} for path in asset_paths],
        "trajectory": {"path": str(args.trajectory), "sha256": sha256_file(args.trajectory), **trajectory_metadata},
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "driver_and_gpu": command_text(["nvidia-smi"]),
        "nvcc": command_text([os.environ.get("CUDA_HOME", "") + "/bin/nvcc", "--version"]) if os.environ.get("CUDA_HOME") else command_text(["nvcc", "--version"]),
        "compiler": command_text(["g++", "--version"]),
        "cwd": os.getcwd(),
        "command": sys.argv,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
