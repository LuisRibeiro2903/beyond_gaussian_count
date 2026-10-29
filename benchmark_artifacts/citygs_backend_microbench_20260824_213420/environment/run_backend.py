#!/usr/bin/env python
"""Run one fresh-process direct rasterizer microbenchmark repetition."""

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
from typing import Any, Callable

import numpy as np
import torch


OFFICIAL_ROOT = Path("/home/luis/tese/CityGaussian_official_parity")
OFFICIAL_EXTENSIONS = OFFICIAL_ROOT / ".official_extensions"
OFFICIAL_COMMIT = "e16c526f78ab01d377b13b3e6862dcc93ddbe953"
GRAPHDECO_COMMIT = "59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d"
GSP_FRAMEWORK_PINNED_COMMIT = "cfc2136459dd63cc47dc3acfde55a3d72a4d8bba"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("graphdeco", "gsplat"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=100)
    return parser.parse_args()


def command_text(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        text = (result.stdout + result.stderr).strip()
        return text or None
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor, chunk_rows: int = 65536) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    if value.ndim == 0:
        digest.update(value.numpy().tobytes(order="C"))
    else:
        for start in range(0, value.shape[0], chunk_rows):
            digest.update(value[start : start + chunk_rows].numpy().tobytes(order="C"))
    return digest.hexdigest()


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
        "std_population": float(array.std(ddof=0)),
        "std_sample": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def gpu_state() -> dict[str, Any]:
    query = command_text(
        [
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,clocks.sm,clocks.mem,pstate,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
    )
    record: dict[str, Any] = {"nvidia_smi_raw": query}
    if query:
        try:
            fields = [part.strip() for part in query.splitlines()[0].split(",")]
            record.update(
                {
                    "name": fields[0],
                    "temperature_c": int(fields[1]),
                    "sm_clock_mhz": int(fields[2]),
                    "memory_clock_mhz": int(fields[3]),
                    "pstate": fields[4],
                    "whole_device_used_bytes": int(fields[5]) * 1024 * 1024,
                    "whole_device_total_bytes": int(fields[6]) * 1024 * 1024,
                }
            )
        except Exception as exc:
            record["parse_error"] = str(exc)
    return record


def allocator_state() -> dict[str, int]:
    return {
        "torch_current_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_current_reserved_bytes": int(torch.cuda.memory_reserved()),
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def graphdeco_call_factory(
    gaussians: dict[str, torch.Tensor], camera_tensors: dict[str, torch.Tensor], camera: dict[str, Any]
) -> tuple[Callable[[], tuple[torch.Tensor, dict[str, torch.Tensor]]], Callable[[], torch.Tensor], dict[str, Any]]:
    sys.path.insert(0, str(OFFICIAL_EXTENSIONS))
    sys.path.insert(0, str(OFFICIAL_ROOT))
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    settings = GaussianRasterizationSettings(
        image_height=int(camera["height"]),
        image_width=int(camera["width"]),
        tanfovx=math.tan(float(camera["fov_x"]) * 0.5),
        tanfovy=math.tan(float(camera["fov_y"]) * 0.5),
        bg=background,
        scale_modifier=1.0,
        viewmatrix=camera_tensors["world_view_transform"],
        projmatrix=camera_tensors["full_projection_transform"],
        sh_degree=2,
        campos=camera_tensors["camera_center"],
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    means2d = torch.zeros_like(gaussians["means"], requires_grad=False)
    opacity_column = gaussians["opacities"].reshape(-1, 1)

    def call() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image, radii = rasterizer(
            means3D=gaussians["means"],
            means2D=means2d,
            shs=gaussians["sh_coefficients"],
            colors_precomp=None,
            opacities=opacity_column,
            scales=gaussians["scales"],
            rotations=gaussians["quats"],
            cov3D_precomp=None,
        )
        return image.permute(1, 2, 0).contiguous(), {"radii": radii}

    ones = torch.ones((gaussians["means"].shape[0], 3), dtype=torch.float32, device="cuda")

    def alpha_call() -> torch.Tensor:
        image, _radii = rasterizer(
            means3D=gaussians["means"],
            means2D=means2d,
            shs=None,
            colors_precomp=ones,
            opacities=opacity_column,
            scales=gaussians["scales"],
            rotations=gaussians["quats"],
            cov3D_precomp=None,
        )
        return image[:1].permute(1, 2, 0).contiguous()

    semantics = {
        "call": "preconstructed GaussianRasterizer.__call__ only",
        "scale_input": "activated positive scales; covariance constructed internally",
        "opacity_input": "activated opacity column; compositing internally",
        "rotation_input": "pre-normalized wxyz quaternion",
        "color_input": "degree-2 SH coefficients; SH evaluated internally",
        "camera_input": "official transposed world-view and full-projection matrices",
        "internal_work": [
            "covariance construction",
            "SH evaluation",
            "camera-space projection",
            "frustum rejection",
            "tile overlap generation",
            "duplicated-instance sorting",
            "alpha compositing",
        ],
    }
    return call, alpha_call, semantics


def gsplat_call_factory(
    gaussians: dict[str, torch.Tensor], camera_tensors: dict[str, torch.Tensor], camera: dict[str, Any]
) -> tuple[Callable[[], tuple[torch.Tensor, dict[str, torch.Tensor]]], Callable[[], torch.Tensor], dict[str, Any]]:
    import gsplat
    from gsplat.rendering import rasterization

    background = torch.zeros((1, 3), dtype=torch.float32, device="cuda")
    kwargs = {
        "means": gaussians["means"],
        "quats": gaussians["quats"],
        "scales": gaussians["scales"],
        "opacities": gaussians["opacities"],
        "colors": gaussians["sh_coefficients"],
        "viewmats": camera_tensors["w2c"].unsqueeze(0),
        "Ks": camera_tensors["K"].unsqueeze(0),
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "near_plane": float(camera["near"]),
        "far_plane": float(camera["gsplat_far"]),
        "radius_clip": 0.0,
        "eps2d": 0.3,
        "sh_degree": 2,
        "backgrounds": background,
        "render_mode": "RGB",
        "rasterize_mode": "classic",
        "camera_model": "pinhole",
        "packed": False,
        "distributed": False,
        "with_ut": False,
        "with_eval3d": False,
    }

    def call() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        colors, alphas, info = rasterization(**kwargs)
        info["returned_alpha"] = alphas[0]
        return colors[0], info

    def alpha_call() -> torch.Tensor:
        _colors, alphas, _info = rasterization(**kwargs)
        return alphas[0]

    semantics = {
        "call": "gsplat.rendering.rasterization only",
        "gsplat_version": gsplat.__version__,
        "gsplat_module": gsplat.__file__,
        "scale_input": "activated positive scales; covariance constructed internally",
        "opacity_input": "activated opacity vector; compositing internally",
        "rotation_input": "pre-normalized wxyz quaternion",
        "color_input": "degree-2 SH coefficients; SH evaluated internally",
        "camera_input": "conventional world-to-camera matrix plus pinhole K",
        "internal_work": [
            "covariance construction",
            "SH evaluation",
            "camera-space projection",
            "frustum rejection",
            "tile overlap generation",
            "duplicated-instance sorting",
            "alpha compositing",
        ],
        "configuration": {
            "packed": False,
            "eps2d": 0.3,
            "radius_clip": 0.0,
            "rasterize_mode": "classic",
            "near_plane": float(camera["near"]),
            "far_plane": float(camera["gsplat_far"]),
        },
    }
    return call, alpha_call, semantics


def diagnostics(backend: str, info: dict[str, torch.Tensor]) -> dict[str, Any]:
    radii = info.get("radii")
    record: dict[str, Any] = {
        "input_gaussian_count": None,
        "positive_radii_projected_gaussian_count": None,
        "maximum_radius": None,
        "tile_intersection_count": None,
        "duplicated_sorted_instance_count": None,
        "tiles_touched": None,
    }
    if radii is not None:
        record["positive_radii_projected_gaussian_count"] = int((radii > 0).sum().item())
        record["maximum_radius"] = float(radii.max().item())
    if backend == "gsplat":
        tiles_per_gauss = info.get("tiles_per_gauss")
        isect_ids = info.get("isect_ids")
        flatten_ids = info.get("flatten_ids")
        if tiles_per_gauss is not None:
            record["tile_intersection_count"] = int(tiles_per_gauss.sum().item())
        if isect_ids is not None:
            record["duplicated_sorted_instance_count"] = int(isect_ids.numel())
        if flatten_ids is not None:
            record["flatten_id_count"] = int(flatten_ids.numel())
        record["tiles_touched"] = "not separately exposed by the public rasterization result"
    else:
        record["tile_intersection_count"] = "not exposed by pinned public GraphDECO API"
        record["duplicated_sorted_instance_count"] = "not exposed by pinned public GraphDECO API"
        record["tiles_touched"] = "not exposed by pinned public GraphDECO API"
    return record


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 5:
        raise ValueError("At least five warmup renders are required")
    if args.measured < 30:
        raise ValueError("At least 30 measured renders are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(args.input_metadata.read_text())
    actual_input_hash = sha256_file(args.input)
    if actual_input_hash != metadata["canonical_input_file_sha256"]:
        raise RuntimeError("Canonical input file hash mismatch")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if payload["format"] != "CITYGS-BACKEND-MICROBENCH-CANONICAL-INPUT-v1":
        raise RuntimeError("Unexpected canonical input format")
    cpu_gaussians = payload["gaussians"]
    cpu_camera_tensors = payload["camera_tensors"]
    observed_hashes = {key: tensor_sha256(value) for key, value in cpu_gaussians.items()}
    expected_hashes = {
        key: value["sha256"] for key, value in metadata["canonical_tensor_records"].items()
    }
    if observed_hashes != expected_hashes:
        raise RuntimeError("Canonical tensor hash mismatch")

    gaussians = {
        key: value.to(device="cuda", dtype=torch.float32).contiguous()
        for key, value in cpu_gaussians.items()
    }
    camera_tensors = {
        key: value.to(device="cuda", dtype=torch.float32).contiguous()
        for key, value in cpu_camera_tensors.items()
    }
    del cpu_gaussians, cpu_camera_tensors, payload
    camera = {
        **json.loads(args.input_metadata.read_text()),
        "height": int(metadata["resolution"][1]),
        "width": int(metadata["resolution"][0]),
        "fov_x": float(metadata["fov_x"]),
        "fov_y": float(metadata["fov_y"]),
        "near": 0.01,
        "gsplat_far": 1.0e10,
    }
    torch.cuda.synchronize()

    if args.backend == "graphdeco":
        render_call, alpha_call, semantics = graphdeco_call_factory(gaussians, camera_tensors, camera)
    else:
        render_call, alpha_call, semantics = gsplat_call_factory(gaussians, camera_tensors, camera)

    # Force allocator setup, CUDA module loading, and any backend laziness before measurement.
    warmup_diagnostics = None
    warmup_image = None
    for _ in range(args.warmup):
        warmup_image, warmup_info = render_call()
        warmup_diagnostics = diagnostics(args.backend, warmup_info)
    torch.cuda.synchronize()
    del warmup_image, warmup_info
    torch.cuda.reset_peak_memory_stats()
    memory_before = allocator_state()
    gpu_before = gpu_state()

    event_pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(args.measured)
    ]
    gpu_samples: list[float] = []
    wall_samples: list[float] = []
    output_image = None
    output_info = None
    for start_event, end_event in event_pairs:
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start_event.record()
        output_image, output_info = render_call()
        end_event.record()
        end_event.synchronize()
        wall_samples.append((time.perf_counter() - wall_start) * 1000.0)
        gpu_samples.append(float(start_event.elapsed_time(end_event)))

    if output_image is None or output_info is None:
        raise RuntimeError("No measured output was produced")
    measured_diagnostics = diagnostics(args.backend, output_info)
    measured_diagnostics["input_gaussian_count"] = int(gaussians["means"].shape[0])
    memory_after = allocator_state()
    gpu_after = gpu_state()

    # Output/alpha sanity is deliberately outside timed samples.
    alpha = alpha_call()
    torch.cuda.synchronize()
    image_cpu = output_image.detach().cpu().numpy().astype(np.float32, copy=False)
    alpha_cpu = alpha.detach().cpu().numpy().astype(np.float32, copy=False)
    np.save(args.output_dir / "rgb.npy", image_cpu)
    np.save(args.output_dir / "alpha.npy", alpha_cpu)

    records = [
        {
            "sample_index": index,
            "backend_gpu_time_ms": gpu_ms,
            "backend_synchronized_wall_time_ms": wall_ms,
        }
        for index, (gpu_ms, wall_ms) in enumerate(zip(gpu_samples, wall_samples))
    ]
    with (args.output_dir / "samples.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    summary = {
        "record_kind": "CITYGS-BACKEND-MICROBENCH-repeat",
        "backend": args.backend,
        "case": metadata["case"],
        "repeat": args.repeat,
        "gaussian_count": int(gaussians["means"].shape[0]),
        "camera_index": metadata["camera_index"],
        "camera_id": metadata["camera_id"],
        "resolution": metadata["resolution"],
        "sh_degree": 2,
        "warmup_render_count": args.warmup,
        "measured_render_count": args.measured,
        "backend_gpu_time_ms": distribution(gpu_samples),
        "backend_synchronized_wall_time_ms": distribution(wall_samples),
        "backend_throughput_equivalent": 1000.0 / statistics.fmean(gpu_samples),
        "input_file_sha256": actual_input_hash,
        "input_tensor_sha256": observed_hashes,
        "all_rgb_finite": bool(np.isfinite(image_cpu).all()),
        "all_alpha_finite": bool(np.isfinite(alpha_cpu).all()),
        "rgb_range": [float(image_cpu.min()), float(image_cpu.max())],
        "alpha_range": [float(alpha_cpu.min()), float(alpha_cpu.max())],
        "warmup_diagnostics": warmup_diagnostics,
        "measured_diagnostics": measured_diagnostics,
        "memory": {
            "allocator_semantics": "PyTorch allocator counters; GraphDECO custom extension behavior may not be representation-neutral",
            "before_measured_series_after_peak_reset": memory_before,
            "after_measured_series": memory_after,
            "whole_device_before": gpu_before,
            "whole_device_after": gpu_after,
        },
        "timing_semantics": {
            "backend_gpu_time_ms": "CUDA events enclosing only one direct backend rasterization invocation; end event synchronized",
            "backend_synchronized_wall_time_ms": "CPU perf_counter enclosing event record, backend call, end-event record, and end-event synchronization",
            "excluded": [
                "scene and PLY loading",
                "activation/conversion",
                "camera construction",
                "visibility and LoD selection",
                "active-set gather and concatenation",
                "canonical input loading and hashing",
                "host-to-device transfer",
                "rasterizer/static argument construction",
                "warmup",
                "diagnostic reduction",
                "alpha sanity render",
                "image transfer and serialization",
            ],
        },
        "backend_input_and_internal_work": semantics,
    }
    (args.output_dir / "repeat.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    provenance = {
        "command": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_compiled_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "official_citygaussian_commit": OFFICIAL_COMMIT,
        "graphdeco_commit": GRAPHDECO_COMMIT,
        "gs_framework_pinned_commit": GSP_FRAMEWORK_PINNED_COMMIT,
        "nvidia_smi": command_text(["nvidia-smi"]),
        "nvcc": command_text(["nvcc", "--version"]),
        "compiler": command_text(["g++", "--version"]),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
