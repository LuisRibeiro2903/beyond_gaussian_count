from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from scene_runtime import FrameCamera, GsplatRenderer, RenderSettings, SceneLoader


BASELINE_CONFIGURATION_NAME = "flat_gsplat_smoke_v1"
FIXTURE_SCENE_ID = "tiny_flat_gaussians_v1"
DEFAULT_WIDTH = 64
DEFAULT_HEIGHT = 48
DETERMINISM_ATOL = 1e-4
DETERMINISM_RTOL = 1e-4
REQUIRED_RUNTIME_MODULES = (
    "gsplat",
    "scene_runtime",
    "scene_runtime.loader",
    "scene_runtime.renderer",
    "scene_runtime.gsplat_compat",
    "gsplat_viewer",
    "simple_viewer",
)


@dataclass(frozen=True)
class ValidationFrame:
    camera_id: str
    camera: FrameCamera
    settings: RenderSettings


def tiny_fixture_splats() -> dict[str, torch.Tensor]:
    means = torch.tensor(
        [
            [0.00, 0.00, 2.00],
            [-0.34, -0.18, 2.24],
            [0.36, 0.16, 2.42],
            [-0.12, 0.34, 2.70],
            [0.22, -0.36, 3.05],
        ],
        dtype=torch.float32,
    )
    n = int(means.shape[0])
    sh0 = torch.tensor(
        [
            [[0.80, 0.18, 0.12]],
            [[0.10, 0.72, 0.20]],
            [[0.16, 0.26, 0.84]],
            [[0.78, 0.70, 0.18]],
            [[0.70, 0.24, 0.74]],
        ],
        dtype=torch.float32,
    )
    return {
        "means": means,
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(n, 1),
        "scales": torch.log(
            torch.tensor(
                [
                    [0.075, 0.075, 0.075],
                    [0.070, 0.055, 0.060],
                    [0.060, 0.070, 0.055],
                    [0.085, 0.060, 0.065],
                    [0.050, 0.060, 0.085],
                ],
                dtype=torch.float32,
            )
        ),
        "opacities": torch.tensor([3.0, 2.6, 2.4, 2.2, 2.0], dtype=torch.float32),
        "sh0": sh0,
        "shN": torch.zeros((n, 15, 3), dtype=torch.float32),
    }


def write_tiny_flat_fixture(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": 0,
        "fixture": {
            "scene_id": FIXTURE_SCENE_ID,
            "seed": None,
            "description": "Five fixed Gaussians for flat gsplat baseline validation.",
        },
        "splats": tiny_fixture_splats(),
    }
    torch.save(payload, path)
    return path


def fixed_validation_frames(width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> list[ValidationFrame]:
    K = torch.tensor(
        [
            [58.0, 0.0, width * 0.5],
            [0.0, 58.0, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    c2w_front = torch.eye(4, dtype=torch.float32)
    c2w_offset = torch.eye(4, dtype=torch.float32)
    c2w_offset[0, 3] = 0.18
    c2w_offset[1, 3] = -0.06
    settings = RenderSettings(
        width=width,
        height=height,
        sh_degree=0,
        near_plane=0.1,
        far_plane=8.0,
        background=(0.01, 0.02, 0.03),
        render_mode="RGB",
        rasterize_mode="classic",
        camera_model="pinhole",
        packed=False,
    )
    return [
        ValidationFrame(
            camera_id="front_identity",
            camera=FrameCamera(c2w=c2w_front, K=K, width=width, height=height),
            settings=settings,
        ),
        ValidationFrame(
            camera_id="offset_x",
            camera=FrameCamera(c2w=c2w_offset, K=K, width=width, height=height),
            settings=settings,
        ),
    ]


def import_runtime_modules() -> dict[str, str]:
    imported = {}
    for module_name in REQUIRED_RUNTIME_MODULES:
        module = import_module(module_name)
        imported[module_name] = str(getattr(module, "__file__", "built-in"))
    return imported


def environment_report() -> dict[str, Any]:
    return {
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        },
        "gpu": _gpu_report(),
        "gsplat": {
            "version": _package_version("gsplat"),
            "module_path": _module_path("gsplat"),
        },
    }


def structural_validation(scene_path: Path, device: torch.device | None = None) -> dict[str, Any]:
    device = device or torch.device("cpu")
    imports = import_runtime_modules()
    scene = SceneLoader.load([scene_path], device=device, scene_type="flat")
    stats = scene.get_scene_stats()
    frame = fixed_validation_frames()[0]
    active = scene.prepare_frame(frame.camera, frame.settings)
    renderer = GsplatRenderer(device)

    expected_n = int(tiny_fixture_splats()["means"].shape[0])
    checks = {
        "scene_type_is_flat": stats.scene_type == "flat",
        "gaussian_count": stats.total_gaussians == expected_n and active.num_gaussians == expected_n,
        "sh_degree": stats.sh_degree == 3,
        "means_shape": tuple(active.means.shape) == (expected_n, 3),
        "quats_shape": tuple(active.quats.shape) == (expected_n, 4),
        "scales_shape": tuple(active.scales.shape) == (expected_n, 3),
        "opacities_shape": tuple(active.opacities.shape) == (expected_n,),
        "colors_shape": tuple(active.colors.shape) == (expected_n, 16, 3),
        "all_float32": all(
            tensor.dtype == torch.float32
            for tensor in (active.means, active.quats, active.scales, active.opacities, active.colors)
        ),
        "device": all(
            tensor.device.type == device.type
            for tensor in (active.means, active.quats, active.scales, active.opacities, active.colors)
        ),
        "active_metrics": (
            active.scene_metrics.active_blocks == 1
            and active.scene_metrics.active_gaussians == expected_n
            and active.scene_metrics.active_gaussian_fraction == 1.0
        ),
        "renderer_constructed": isinstance(renderer, GsplatRenderer),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "imports": imports,
        "checks": checks,
        "scene": {
            "scene_type": stats.scene_type,
            "total_gaussians": stats.total_gaussians,
            "sh_degree": stats.sh_degree,
            "steps": stats.steps,
        },
    }


@torch.no_grad()
def render_validation(
    scene_path: Path,
    device: torch.device,
    frame_index: int = 0,
    repetitions: int = 2,
) -> dict[str, Any]:
    if repetitions < 2:
        raise ValueError("render_validation requires at least two repetitions for determinism.")
    scene = SceneLoader.load([scene_path], device=device, scene_type="flat")
    stats = scene.get_scene_stats()
    frame = fixed_validation_frames()[frame_index]
    renderer = GsplatRenderer(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    images = []
    alphas = []
    results = []
    for _ in range(repetitions):
        active = scene.prepare_frame(frame.camera, frame.settings)
        result = renderer.render(active, frame.camera, frame.settings)
        images.append(result.image.detach().cpu())
        alphas.append(result.alpha.detach().cpu() if result.alpha is not None else None)
        results.append(result)

    image_diffs = [
        float(torch.max(torch.abs(images[0] - image)).item())
        for image in images[1:]
    ]
    alpha_diffs = []
    for alpha in alphas[1:]:
        if alphas[0] is None or alpha is None:
            alpha_diffs.append(float("nan"))
        else:
            alpha_diffs.append(float(torch.max(torch.abs(alphas[0] - alpha)).item()))

    peak_vram = _peak_vram_bytes(device)
    result0 = results[0]
    image0 = images[0]
    alpha0 = alphas[0]
    active_metrics = scene.get_metrics()
    finite_image = bool(torch.isfinite(image0).all().item())
    finite_alpha = bool(alpha0 is not None and torch.isfinite(alpha0).all().item())
    deterministic = all(diff <= DETERMINISM_ATOL for diff in image_diffs) and all(
        math.isnan(diff) or diff <= DETERMINISM_ATOL for diff in alpha_diffs
    )
    valid_shape = tuple(image0.shape) == (
        frame.settings.height,
        frame.settings.width,
        3,
    )
    alpha_shape = alpha0 is not None and tuple(alpha0.shape) == (
        frame.settings.height,
        frame.settings.width,
        1,
    )
    plausible_counts = (
        active_metrics.active_gaussians == stats.total_gaussians
        and 0.0 <= result0.visible_gaussians <= float(stats.total_gaussians)
    )

    return {
        "status": "passed"
        if all([finite_image, finite_alpha, deterministic, valid_shape, alpha_shape, plausible_counts])
        else "failed",
        "determinism": {
            "image_max_abs_diff": max(image_diffs) if image_diffs else 0.0,
            "alpha_max_abs_diff": max(alpha_diffs) if alpha_diffs else 0.0,
            "atol": DETERMINISM_ATOL,
            "rtol": DETERMINISM_RTOL,
        },
        "frame": {
            "camera_id": frame.camera_id,
            "resolution": [frame.settings.width, frame.settings.height],
            "image_shape": list(image0.shape),
            "alpha_shape": list(alpha0.shape) if alpha0 is not None else None,
            "image_min": float(image0.min().item()),
            "image_max": float(image0.max().item()),
            "alpha_min": float(alpha0.min().item()) if alpha0 is not None else float("nan"),
            "alpha_max": float(alpha0.max().item()) if alpha0 is not None else float("nan"),
            "finite_image": finite_image,
            "finite_alpha": finite_alpha,
        },
        "metrics": {
            "render_time_ms": float(result0.render_time_ms),
            "rasterization_time_ms": float(result0.gpu_render_time_ms),
            "active_gaussian_count": int(active_metrics.active_gaussians),
            "rendered_gaussian_count": float(result0.visible_gaussians),
            "visible_gaussian_count": float(result0.visible_gaussians),
            "active_block_count": float(active_metrics.active_blocks),
            "peak_vram_bytes": peak_vram,
        },
        "result_schema": baseline_result_schema(
            scene_identifier=FIXTURE_SCENE_ID,
            camera_identifier=frame.camera_id,
            resolution=[frame.settings.width, frame.settings.height],
            frame_time_ms=float(result0.render_time_ms),
            rasterization_time_ms=float(result0.gpu_render_time_ms),
            active_gaussian_count=int(active_metrics.active_gaussians),
            rendered_gaussian_count=float(result0.visible_gaussians),
            visible_gaussian_count=float(result0.visible_gaussians),
            peak_vram_bytes=peak_vram,
        ),
    }


def baseline_result_schema(
    *,
    scene_identifier: str,
    camera_identifier: str,
    resolution: list[int],
    frame_time_ms: float,
    rasterization_time_ms: float,
    active_gaussian_count: int,
    rendered_gaussian_count: float,
    visible_gaussian_count: float,
    peak_vram_bytes: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_kind": "validation_smoke",
        "configuration_name": BASELINE_CONFIGURATION_NAME,
        "scene_identifier": scene_identifier,
        "camera_identifier": camera_identifier,
        "resolution": resolution,
        "frame_time_ms": frame_time_ms,
        "rasterization_time_ms": rasterization_time_ms,
        "active_gaussian_count": active_gaussian_count,
        "rendered_gaussian_count": rendered_gaussian_count,
        "visible_gaussian_count": visible_gaussian_count,
        "peak_vram_bytes": peak_vram_bytes,
        "gsplat_version": _package_version("gsplat"),
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_baseline_smoke(
    output_dir: Path | None = None,
    device_name: str = "auto",
    write_png: bool = False,
) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    env = environment_report()
    with TemporaryDirectory() as tmpdir:
        fixture_dir = Path(output_dir) if output_dir is not None else Path(tmpdir)
        fixture_path = write_tiny_flat_fixture(fixture_dir / "tiny_flat_gaussians.pt")
        try:
            structural = structural_validation(fixture_path, device=torch.device("cpu"))
        except Exception as exc:
            structural = {"status": "failed", "error": repr(exc)}

        try:
            device = _resolve_render_device(device_name)
            if device is None:
                render = {
                    "status": "skipped",
                    "reason": "CUDA is unavailable; structural validation ran on CPU.",
                }
            else:
                render = render_validation(fixture_path, device=device)
                if write_png and output_dir is not None:
                    render["png_path"] = str(_write_debug_png(output_dir, fixture_path, device))
        except Exception as exc:
            render = {"status": "failed", "error": repr(exc)}

        exit_code = 0 if structural["status"] == "passed" and render["status"] in {"passed", "skipped"} else 1

    report = {
        "status": "passed" if exit_code == 0 else "failed",
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "environment": env,
        "fixture": {
            "scene_identifier": FIXTURE_SCENE_ID,
            "gaussian_count": int(tiny_fixture_splats()["means"].shape[0]),
            "camera_count": len(fixed_validation_frames()),
            "generated": True,
        },
        "structural": structural,
        "render": render,
    }
    return exit_code, report


def dumps_report(report: dict[str, Any]) -> str:
    return json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _resolve_render_device(device_name: str) -> torch.device | None:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested but torch.cuda.is_available() is false.")
        return torch.device("cuda", 0)
    if device_name == "auto":
        return torch.device("cuda", 0) if torch.cuda.is_available() else None
    raise ValueError("device_name must be one of: auto, cuda, cpu.")


def _gpu_report() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": None,
            "total_memory_bytes": None,
        }
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(device),
        "total_memory_bytes": int(props.total_memory),
        "capability": list(torch.cuda.get_device_capability(device)),
    }


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        module = import_module(package_name)
        return str(getattr(module, "__version__", "unknown"))


def _module_path(module_name: str) -> str:
    module = import_module(module_name)
    return str(getattr(module, "__file__", "built-in"))


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip()


def _peak_vram_bytes(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return float(torch.cuda.max_memory_allocated(device))


def _write_debug_png(output_dir: Path, fixture_path: Path, device: torch.device) -> Path:
    try:
        import imageio.v3 as imageio
    except Exception as exc:
        raise RuntimeError(f"imageio is required for --write-png: {exc}") from exc

    scene = SceneLoader.load([fixture_path], device=device, scene_type="flat")
    frame = fixed_validation_frames()[0]
    renderer = GsplatRenderer(device)
    active = scene.prepare_frame(frame.camera, frame.settings)
    result = renderer.render(active, frame.camera, frame.settings)
    image = (result.image.detach().cpu().clamp(0, 1).numpy() * 255.0).astype("uint8")
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "tiny_flat_gaussians.png"
    imageio.imwrite(png_path, image)
    return png_path
