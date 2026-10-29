#!/usr/bin/env python
"""Focused non-optimizing GraphDECO wrapper-stage probe.

The probe preserves the released LoD wrapper's activated tensor semantics,
screen-space tensor construction, per-call settings/rasterizer construction,
and public rasterizer invocation. Visibility, selection, concatenation, input
loading, hashing, and transfer remain outside all stage timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


OFFICIAL_ROOT = Path("/home/luis/tese/CityGaussian_official_parity")
OFFICIAL_EXTENSIONS = OFFICIAL_ROOT / ".official_extensions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-zero-input", type=Path, required=True)
    parser.add_argument("--camera-zero-metadata", type=Path, required=True)
    parser.add_argument("--target-input", type=Path, required=True)
    parser.add_argument("--target-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_input(path: Path, metadata_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text())
    if sha256_file(path) != metadata["canonical_input_file_sha256"]:
        raise RuntimeError(f"Input hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    gaussians = {key: value.cuda().float().contiguous() for key, value in payload["gaussians"].items()}
    camera_tensors = {key: value.cuda().float().contiguous() for key, value in payload["camera_tensors"].items()}
    return gaussians, camera_tensors, metadata


def elapsed_gpu(operation) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


@torch.no_grad()
def one_iteration(
    gaussians: dict[str, torch.Tensor],
    camera_tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor]:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    torch.cuda.synchronize()
    screen_wall_start = time.perf_counter()
    holder: dict[str, Any] = {}

    def create_screen() -> None:
        holder["means2d"] = torch.zeros_like(
            gaussians["means"], dtype=gaussians["means"].dtype, requires_grad=True, device="cuda"
        ) + 0

    screen_gpu = elapsed_gpu(create_screen)
    screen_wall = (time.perf_counter() - screen_wall_start) * 1000.0

    torch.cuda.synchronize()
    attr_start = time.perf_counter()
    means3d = gaussians["means"]
    means2d = holder["means2d"]
    opacity = gaussians["opacities"]
    scales = gaussians["scales"]
    rotations = gaussians["quats"]
    shs = gaussians["sh_coefficients"]
    torch.cuda.synchronize()
    attr_wall = (time.perf_counter() - attr_start) * 1000.0

    torch.cuda.synchronize()
    args_start = time.perf_counter()
    settings = GaussianRasterizationSettings(
        image_height=int(metadata["resolution"][1]),
        image_width=int(metadata["resolution"][0]),
        tanfovx=math.tan(float(metadata["fov_x"]) * 0.5),
        tanfovy=math.tan(float(metadata["fov_y"]) * 0.5),
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=camera_tensors["world_view_transform"],
        projmatrix=camera_tensors["full_projection_transform"],
        sh_degree=2,
        campos=camera_tensors["camera_center"],
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    torch.cuda.synchronize()
    args_wall = (time.perf_counter() - args_start) * 1000.0

    output: dict[str, Any] = {}
    torch.cuda.synchronize()
    call_wall_start = time.perf_counter()

    def rasterize() -> None:
        output["image"], output["radii"] = rasterizer(
            means3D=means3d,
            means2D=means2d,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )

    call_gpu = elapsed_gpu(rasterize)
    call_wall = (time.perf_counter() - call_wall_start) * 1000.0
    record = {
        "screen_space_tensor_creation_gpu_ms": screen_gpu,
        "screen_space_tensor_creation_synchronized_wall_ms": screen_wall,
        "activated_attribute_view_preparation_synchronized_wall_ms": attr_wall,
        "rasterizer_argument_construction_synchronized_wall_ms": args_wall,
        "rasterizer_call_gpu_ms": call_gpu,
        "rasterizer_call_synchronized_wall_ms": call_wall,
        "positive_radii_count": int((output["radii"] > 0).sum().item()),
        "maximum_radius": float(output["radii"].max().item()),
        "torch_current_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_current_reserved_bytes": int(torch.cuda.memory_reserved()),
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    return record, output["image"]


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(OFFICIAL_EXTENSIONS))
    sys.path.insert(0, str(OFFICIAL_ROOT))
    camera_zero, camera_zero_tensors, camera_zero_metadata = load_input(
        args.camera_zero_input, args.camera_zero_metadata
    )
    _warm_record, warm_image = one_iteration(camera_zero, camera_zero_tensors, camera_zero_metadata)
    del warm_image, camera_zero, camera_zero_tensors
    torch.cuda.synchronize()

    target, target_tensors, target_metadata = load_input(args.target_input, args.target_metadata)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    iterations = []
    final_image = None
    for index in range(args.iterations):
        record, final_image = one_iteration(target, target_tensors, target_metadata)
        record["iteration"] = index + 1
        record["target_warmed_before_call"] = index > 0
        iterations.append(record)
        del final_image
    payload = {
        "record_kind": "CITYGS-12CAM-ATTRIBUTION-wrapper-stage-probe",
        "repeat": args.repeat,
        "camera_index": target_metadata["camera_index"],
        "gaussian_count": target_metadata["gaussian_count"],
        "camera_zero_single_excluded_warmup": True,
        "target_iterations": args.iterations,
        "input_file_sha256": target_metadata["canonical_input_file_sha256"],
        "iterations": iterations,
        "interpretation": "Iteration 1 is the target's first encounter after one camera-zero rasterizer warmup; later iterations expose target-shape warm behavior. Stage synchronization is diagnostic and changes scheduling relative to the original wrapper, so values are not an additive accounting identity for the historical campaign.",
        "excluded": ["loading", "hashing", "host-to-device transfer", "visibility", "LoD selection", "active-set gathering", "concatenation"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
