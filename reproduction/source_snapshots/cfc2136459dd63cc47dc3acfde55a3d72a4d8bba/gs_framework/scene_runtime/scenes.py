from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from block_manager import BlockCullingMode, BlockManager

from .types import (
    ActiveSplats,
    BlockDebugInfo,
    FrameCamera,
    RenderSettings,
    SceneFrameMetrics,
    SceneStats,
)
from .ply_splats import is_ply_path, load_gaussian_ply


REQUIRED_SPLAT_KEYS = ("means", "quats", "scales", "opacities", "sh0", "shN")
FLAT_BOUNDS_POLICY = "finite_exact_cpu_minmax_with_8_percent_view_padding"
LoadProgressCallback = Callable[[str], None]


def _notify(callback: LoadProgressCallback | None, stage: str) -> None:
    if callback is not None:
        callback(stage)


def _activate_raw_splats(raw: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {
        "means": raw["means"].to(device=device, dtype=torch.float32).contiguous(),
        "quats": F.normalize(
            raw["quats"].to(device=device, dtype=torch.float32), p=2, dim=-1
        ).contiguous(),
        "scales": torch.exp(raw["scales"].to(device=device, dtype=torch.float32)).contiguous(),
        "opacities": torch.sigmoid(
            raw["opacities"].to(device=device, dtype=torch.float32)
        ).contiguous(),
        "colors": torch.cat(
            [
                raw["sh0"].to(device=device, dtype=torch.float32),
                raw["shN"].to(device=device, dtype=torch.float32),
            ],
            dim=-2,
        ).contiguous(),
    }


def _empty_scene_metrics(total_gaussians: int) -> SceneFrameMetrics:
    return SceneFrameMetrics(
        active_blocks=1,
        active_gaussians=total_gaussians,
        active_gaussian_fraction=1.0,
        resident_gaussians=total_gaussians,
        resident_gaussian_fraction=1.0,
        selected_gaussian_count=total_gaussians,
    )


def compute_scene_bounds_from_means(
    means: Tensor,
    padding_fraction: float = 0.08,
    min_padding: float = 1e-3,
) -> dict[str, object]:
    """Compute flat-scene metadata bounds without changing render tensors.

    Non-finite positions are excluded from metadata bounds, reported in the
    returned counts, and left untouched in the caller's render tensors.
    """
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(f"Expected means with shape (N, 3), got {tuple(means.shape)}.")

    means_cpu = means.detach().to(device="cpu", dtype=torch.float32)
    finite_mask = torch.isfinite(means_cpu).all(dim=-1)
    finite_count = int(finite_mask.sum().item())
    total_count = int(means_cpu.shape[0])
    nonfinite_count = total_count - finite_count
    if finite_count == 0:
        raise ValueError("Cannot compute scene bounds: no finite Gaussian positions.")
    if nonfinite_count:
        warnings.warn(
            "Non-finite Gaussian positions were excluded from metadata bounds "
            f"({nonfinite_count} of {total_count}); render tensors are unchanged.",
            RuntimeWarning,
            stacklevel=2,
        )

    valid = means_cpu[finite_mask]
    scene_min = valid.min(dim=0).values
    scene_max = valid.max(dim=0).values
    extent = scene_max - scene_min
    padding = torch.clamp(extent * float(padding_fraction), min=float(min_padding))
    view_min = scene_min - padding
    view_max = scene_max + padding
    return {
        "scene_min": tuple(float(v) for v in scene_min.tolist()),
        "scene_max": tuple(float(v) for v in scene_max.tolist()),
        "view_min": tuple(float(v) for v in view_min.tolist()),
        "view_max": tuple(float(v) for v in view_max.tolist()),
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "policy": FLAT_BOUNDS_POLICY,
    }


class GaussianScene(ABC):
    @abstractmethod
    def prepare_frame(self, camera: FrameCamera, render_settings: RenderSettings) -> ActiveSplats:
        raise NotImplementedError

    @abstractmethod
    def get_scene_stats(self) -> SceneStats:
        raise NotImplementedError

    def get_debug_geometry(self) -> List[BlockDebugInfo]:
        return []

    def get_metrics(self) -> SceneFrameMetrics:
        return SceneFrameMetrics()

    def set_culling_mode(
        self,
        culling_mode: BlockCullingMode,
        selected_block_id: Optional[int] = None,
    ) -> None:
        return None


class FlatGaussianScene(GaussianScene):
    def __init__(
        self,
        paths: Sequence[Path],
        device: torch.device,
        ply_coordinate_mode: Literal["native", "supersplat"] = "native",
        progress_callback: LoadProgressCallback | None = None,
    ):
        if not paths:
            raise ValueError("FlatGaussianScene requires at least one checkpoint path.")
        self.paths = [Path(path) for path in paths]
        self.device = device
        self.steps: List[int] = []

        loaded: List[Dict[str, Tensor]] = []
        for path in self.paths:
            if is_ply_path(path):
                splats = load_gaussian_ply(
                    path,
                    apply_supersplat_transform=ply_coordinate_mode == "supersplat",
                    progress_callback=progress_callback,
                )
                step = -1
            else:
                _notify(progress_callback, "cpu_tensor_construction")
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if not _is_flat_payload(payload):
                    raise ValueError(f"{path} is not a valid flat gsplat checkpoint.")
                splats = payload["splats"]
                step = int(payload.get("step", -1))
            loaded.append({key: splats[key].detach().cpu() for key in REQUIRED_SPLAT_KEYS})
            self.steps.append(step)

        raw = {
            key: torch.cat([splats[key] for splats in loaded], dim=0).contiguous()
            for key in REQUIRED_SPLAT_KEYS
        }
        _notify(progress_callback, "bounds_or_metadata_computation")
        bounds = compute_scene_bounds_from_means(raw["means"])
        self.scene_min = bounds["scene_min"]
        self.scene_max = bounds["scene_max"]
        self.view_min = bounds["view_min"]
        self.view_max = bounds["view_max"]
        self.bounds_finite_gaussian_count = int(bounds["finite_count"])
        self.bounds_nonfinite_gaussian_count = int(bounds["nonfinite_count"])
        self.bounds_policy = str(bounds["policy"])

        _notify(progress_callback, "gpu_transfer")
        active = _activate_raw_splats(raw, device)
        _notify(progress_callback, "gpu_scene_construction")
        self.active_splats = ActiveSplats(
            means=active["means"],
            quats=active["quats"],
            scales=active["scales"],
            opacities=active["opacities"],
            colors=active["colors"],
            scene_metrics=_empty_scene_metrics(int(active["means"].shape[0])),
        )
        self.sh_degree = int(math.sqrt(self.active_splats.colors.shape[-2]) - 1)

    def prepare_frame(self, camera: FrameCamera, render_settings: RenderSettings) -> ActiveSplats:
        metrics = _empty_scene_metrics(self.active_splats.num_gaussians)
        self.active_splats.scene_metrics = metrics
        return self.active_splats

    def get_scene_stats(self) -> SceneStats:
        return SceneStats(
            scene_type="flat",
            total_gaussians=self.active_splats.num_gaussians,
            sh_degree=self.sh_degree,
            paths=self.paths,
            scene_aabb_min=self.scene_min,
            scene_aabb_max=self.scene_max,
            view_aabb_min=self.view_min,
            view_aabb_max=self.view_max,
            bounds_policy=self.bounds_policy,
            bounds_finite_gaussian_count=self.bounds_finite_gaussian_count,
            bounds_nonfinite_gaussian_count=self.bounds_nonfinite_gaussian_count,
            steps=self.steps,
        )

    def get_metrics(self) -> SceneFrameMetrics:
        return self.active_splats.scene_metrics


class BlockedGaussianScene(GaussianScene):
    def __init__(
        self,
        path: Path,
        device: torch.device,
        residency: str = "gpu",
        keepalive_frames: int = 0,
        gpu_block_budget_mb: Optional[float] = None,
        cache_policy: str = "lru",
        culling_mode: BlockCullingMode = "normal_frustum_culling",
        selected_block_id: Optional[int] = None,
    ):
        self.path = Path(path)
        self.device = device
        if gpu_block_budget_mb is not None or cache_policy != "lru":
            raise ValueError(
                "This source-frozen legacy blocked runtime does not implement a GPU block budget "
                "or alternate cache policy."
            )
        self.manager = BlockManager.from_file(
            self.path,
            device=device,
            residency=residency,
            keepalive_frames=keepalive_frames,
        )
        self.culling_mode: BlockCullingMode = culling_mode
        self.selected_block_id = selected_block_id
        self._last_metrics = SceneFrameMetrics()

        mins = [block.cull_aabb_min.cpu() for block in self.manager.blocks]
        maxs = [block.cull_aabb_max.cpu() for block in self.manager.blocks]
        self.scene_min = tuple(float(v) for v in torch.stack(mins, dim=0).min(dim=0).values.tolist())
        self.scene_max = tuple(float(v) for v in torch.stack(maxs, dim=0).max(dim=0).values.tolist())

    def prepare_frame(self, camera: FrameCamera, render_settings: RenderSettings) -> ActiveSplats:
        cull_c2w, cull_K = camera.cull_camera
        block_splats = self.manager.get_active_splats(
            cull_c2w.detach().cpu(),
            cull_K.detach().cpu(),
            int(render_settings.width),
            int(render_settings.height),
            float(render_settings.near_plane),
            float(render_settings.far_plane),
            culling_mode=self.culling_mode,
            selected_block_id=self.selected_block_id,
        )
        self._last_metrics = SceneFrameMetrics(
            active_blocks=block_splats.stats.active_blocks,
            active_block_ids=block_splats.stats.active_block_ids,
            active_gaussians=block_splats.stats.active_gaussians,
            active_gaussian_fraction=block_splats.stats.active_gaussian_fraction,
            total_blocks=block_splats.stats.total_blocks,
            resident_blocks=block_splats.stats.resident_blocks,
            resident_gaussians=block_splats.stats.resident_gaussians,
            resident_gaussian_fraction=block_splats.stats.resident_gaussian_fraction,
            block_cull_ms=block_splats.stats.block_cull_ms,
            block_concat_ms=block_splats.stats.block_concat_ms,
            gpu_upload_ms=block_splats.stats.gpu_upload_ms,
            cache_hits=block_splats.stats.cache_hits,
            cache_misses=block_splats.stats.cache_misses,
            cache_hit_rate=getattr(block_splats.stats, "cache_hit_rate", float("nan")),
            newly_uploaded_blocks=getattr(block_splats.stats, "newly_uploaded_blocks", float("nan")),
            newly_uploaded_gaussians=getattr(block_splats.stats, "newly_uploaded_gaussians", float("nan")),
            evicted_blocks=getattr(block_splats.stats, "evicted_blocks", float("nan")),
            evicted_gaussians=getattr(block_splats.stats, "evicted_gaussians", float("nan")),
            gpu_resident_bytes=getattr(block_splats.stats, "gpu_resident_bytes", float("nan")),
            gpu_resident_mb=getattr(block_splats.stats, "gpu_resident_mb", float("nan")),
            cpu_resident_bytes=getattr(block_splats.stats, "cpu_resident_bytes", float("nan")),
            cpu_resident_mb=getattr(block_splats.stats, "cpu_resident_mb", float("nan")),
            gpu_budget_exceeded=float(getattr(block_splats.stats, "gpu_budget_exceeded", False)),
            active_set_exceeds_budget=float(
                getattr(block_splats.stats, "active_set_exceeds_budget", False)
            ),
        )
        return ActiveSplats(
            means=block_splats.means,
            quats=block_splats.quats,
            scales=block_splats.scales,
            opacities=block_splats.opacities,
            colors=block_splats.colors,
            scene_metrics=self._last_metrics,
        )

    def get_scene_stats(self) -> SceneStats:
        return SceneStats(
            scene_type="blocked",
            total_gaussians=self.manager.total_gaussians,
            sh_degree=self.manager.sh_degree,
            paths=[self.path],
            scene_aabb_min=self.scene_min,
            scene_aabb_max=self.scene_max,
            view_aabb_min=self.scene_min,
            view_aabb_max=self.scene_max,
            total_blocks=self.manager.total_blocks,
        )

    def get_debug_geometry(self) -> List[BlockDebugInfo]:
        return [
            BlockDebugInfo(
                block_id=block.block_id,
                grid_index=block.grid_index,
                gaussian_count=block.gaussian_count,
                content_aabb_min=block.content_aabb_min,
                content_aabb_max=block.content_aabb_max,
                cull_aabb_min=block.cull_aabb_min,
                cull_aabb_max=block.cull_aabb_max,
                active=block.active,
                resident=block.resident,
                empty=block.empty,
                logical_bounds_min=block.logical_bounds_min,
                logical_bounds_max=block.logical_bounds_max,
                raw_gaussian_bounds_min=block.raw_gaussian_bounds_min,
                raw_gaussian_bounds_max=block.raw_gaussian_bounds_max,
                filtered_render_bounds_min=block.filtered_render_bounds_min,
                filtered_render_bounds_max=block.filtered_render_bounds_max,
                raw_gaussian_count=block.raw_gaussian_count,
                bounds_valid_gaussian_count=block.bounds_valid_gaussian_count,
                raw_to_filtered_aabb_volume_ratio=block.raw_to_filtered_aabb_volume_ratio,
            )
            for block in self.manager.debug_blocks()
        ]

    def get_metrics(self) -> SceneFrameMetrics:
        return self._last_metrics

    def set_culling_mode(
        self,
        culling_mode: BlockCullingMode,
        selected_block_id: Optional[int] = None,
    ) -> None:
        self.culling_mode = culling_mode
        self.selected_block_id = selected_block_id

    @property
    def residency(self) -> str:
        return self.manager.residency

    def get_block_debug_info(self, block_id: int) -> Optional[BlockDebugInfo]:
        for block in self.get_debug_geometry():
            if block.block_id == block_id:
                return block
        return None


def _is_flat_payload(payload: object) -> bool:
    if not isinstance(payload, dict) or "splats" not in payload:
        return False
    splats = payload["splats"]
    if not isinstance(splats, dict):
        return False
    return all(key in splats and torch.is_tensor(splats[key]) for key in REQUIRED_SPLAT_KEYS)
