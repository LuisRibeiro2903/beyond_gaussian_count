"""Block-structured Gaussian scene utilities for rendering.

This module keeps the block abstraction above gsplat. The rasterizer still
receives ordinary flat tensors; BlockManager only decides which block tensors
should be concatenated into that flat working set for the current view.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from gaussian_cleanup import (
    aabb_volume,
    analyze_checkpoint,
    load_bounds_filter_config_from_metadata,
    robust_bounds_for_points,
    tensor_bounds,
)


REQUIRED_SPLAT_KEYS = ("means", "quats", "scales", "opacities", "sh0", "shN")
BLOCK_FILE_VERSION = 1
BlockCullingMode = Literal[
    "normal_frustum_culling",
    "force_all_blocks_active",
    "force_no_blocks_active",
    "selected_block_only",
]


@dataclass
class BlockRenderStats:
    total_blocks: int
    active_blocks: int
    active_block_ids: str
    active_gaussians: int
    active_gaussian_fraction: float
    resident_blocks: int
    resident_gaussians: int
    resident_gaussian_fraction: float
    block_cull_ms: float
    block_concat_ms: float
    gpu_upload_ms: float
    cache_hits: int
    cache_misses: int


@dataclass
class ActiveSplats:
    means: Tensor
    quats: Tensor
    scales: Tensor
    opacities: Tensor
    colors: Tensor
    stats: BlockRenderStats

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])


@dataclass
class GaussianBlock:
    block_id: int
    grid_index: Tuple[int, int]
    gaussian_count: int
    content_aabb_min: Tensor
    content_aabb_max: Tensor
    cull_aabb_min: Tensor
    cull_aabb_max: Tensor
    logical_bounds_min: Tensor
    logical_bounds_max: Tensor
    raw_gaussian_bounds_min: Tensor
    raw_gaussian_bounds_max: Tensor
    filtered_render_bounds_min: Tensor
    filtered_render_bounds_max: Tensor
    raw_gaussian_count: int
    bounds_valid_gaussian_count: int
    raw_to_filtered_aabb_volume_ratio: float
    cpu_tensors: Optional[Dict[str, Tensor]]
    gpu_tensors: Optional[Dict[str, Tensor]]
    last_active_frame: int = -1


@dataclass(frozen=True)
class BlockDebugInfo:
    block_id: int
    grid_index: Tuple[int, int]
    gaussian_count: int
    content_aabb_min: Tuple[float, float, float]
    content_aabb_max: Tuple[float, float, float]
    cull_aabb_min: Tuple[float, float, float]
    cull_aabb_max: Tuple[float, float, float]
    active: bool
    resident: bool
    empty: bool
    logical_bounds_min: Optional[Tuple[float, float, float]] = None
    logical_bounds_max: Optional[Tuple[float, float, float]] = None
    raw_gaussian_bounds_min: Optional[Tuple[float, float, float]] = None
    raw_gaussian_bounds_max: Optional[Tuple[float, float, float]] = None
    filtered_render_bounds_min: Optional[Tuple[float, float, float]] = None
    filtered_render_bounds_max: Optional[Tuple[float, float, float]] = None
    raw_gaussian_count: Optional[int] = None
    bounds_valid_gaussian_count: Optional[int] = None
    raw_to_filtered_aabb_volume_ratio: Optional[float] = None


def _load_raw_splats(
    ckpt_paths: Sequence[Path],
    map_location: str | torch.device,
    ply_coordinate_mode: str = "native",
) -> Tuple[Dict[str, Tensor], List[int]]:
    from scene_runtime.ply_splats import is_ply_path, load_gaussian_ply

    if not ckpt_paths:
        raise ValueError("At least one checkpoint path is required.")
    if ply_coordinate_mode not in {"native", "supersplat"}:
        raise ValueError("ply_coordinate_mode must be 'native' or 'supersplat'.")

    loaded: List[Dict[str, Tensor]] = []
    steps: List[int] = []
    for path in ckpt_paths:
        if is_ply_path(path):
            splats = load_gaussian_ply(
                path,
                apply_supersplat_transform=ply_coordinate_mode == "supersplat",
            )
            step = -1
        else:
            ckpt = torch.load(path, map_location=map_location, weights_only=False)
            if "splats" not in ckpt:
                raise ValueError(f"{path} does not contain ckpt['splats'].")
            splats = ckpt["splats"]
            step = int(ckpt.get("step", -1))
        missing = [key for key in REQUIRED_SPLAT_KEYS if key not in splats]
        if missing:
            raise ValueError(f"{path} is missing required splat keys: {missing}")
        loaded.append({key: splats[key].detach().cpu() for key in REQUIRED_SPLAT_KEYS})
        steps.append(step)

    reference_shapes = {
        key: tuple(loaded[0][key].shape[1:])
        for key in REQUIRED_SPLAT_KEYS
    }
    for path, splats in zip(ckpt_paths[1:], loaded[1:]):
        mismatched = [
            f"{key}: expected [N, {reference_shapes[key]}], got {tuple(splats[key].shape)}"
            for key in REQUIRED_SPLAT_KEYS
            if tuple(splats[key].shape[1:]) != reference_shapes[key]
        ]
        if mismatched:
            raise ValueError(
                f"{path} cannot be concatenated with the first checkpoint because "
                "the splat tensor shapes differ after the Gaussian dimension: "
                + "; ".join(mismatched)
            )

    return {
        key: torch.cat([splats[key] for splats in loaded], dim=0).contiguous()
        for key in REQUIRED_SPLAT_KEYS
    }, steps


def _activate_splats(raw: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {
        "means": raw["means"].to(device=device, dtype=torch.float32).contiguous(),
        "quats": F.normalize(raw["quats"].to(device=device, dtype=torch.float32), p=2, dim=-1).contiguous(),
        "scales": torch.exp(raw["scales"].to(device=device, dtype=torch.float32)).contiguous(),
        "opacities": torch.sigmoid(raw["opacities"].to(device=device, dtype=torch.float32)).contiguous(),
        "colors": torch.cat(
            [
                raw["sh0"].to(device=device, dtype=torch.float32),
                raw["shN"].to(device=device, dtype=torch.float32),
            ],
            dim=-2,
        ).contiguous(),
    }


def _empty_active_splats(device: torch.device, colors_shape: Tuple[int, int]) -> Dict[str, Tensor]:
    return {
        "means": torch.empty((0, 3), device=device),
        "quats": torch.empty((0, 4), device=device),
        "scales": torch.empty((0, 3), device=device),
        "opacities": torch.empty((0,), device=device),
        "colors": torch.empty((0, colors_shape[0], colors_shape[1]), device=device),
    }


def _cat_block_tensors(block_tensors: Sequence[Dict[str, Tensor]], device: torch.device, colors_shape: Tuple[int, int]) -> Dict[str, Tensor]:
    if not block_tensors:
        return _empty_active_splats(device, colors_shape)
    return {
        "means": torch.cat([t["means"] for t in block_tensors], dim=0).contiguous(),
        "quats": torch.cat([t["quats"] for t in block_tensors], dim=0).contiguous(),
        "scales": torch.cat([t["scales"] for t in block_tensors], dim=0).contiguous(),
        "opacities": torch.cat([t["opacities"] for t in block_tensors], dim=0).contiguous(),
        "colors": torch.cat([t["colors"] for t in block_tensors], dim=0).contiguous(),
    }


def _aabb_corners(aabb_min: Tensor, aabb_max: Tensor) -> Tensor:
    mins = aabb_min[:, None, :]
    maxs = aabb_max[:, None, :]
    selector = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        device=aabb_min.device,
        dtype=torch.bool,
    )
    return torch.where(selector[None], maxs, mins)


def aabbs_intersect_frustum(
    aabb_min: Tensor,
    aabb_max: Tensor,
    c2w: Tensor,
    K: Tensor,
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
) -> Tensor:
    """Return a boolean mask for AABBs intersecting the pinhole camera frustum.

    The test uses the same OpenCV-style camera convention as the viewer and
    gsplat projection path: positive camera-space z is in front of the camera.
    """

    if aabb_min.numel() == 0:
        return torch.empty((0,), dtype=torch.bool, device=aabb_min.device)

    device = aabb_min.device
    c2w = c2w.to(device=device, dtype=torch.float32)
    K = K.to(device=device, dtype=torch.float32)
    viewmat = torch.linalg.inv(c2w)
    corners = _aabb_corners(aabb_min.to(torch.float32), aabb_max.to(torch.float32))
    ones = torch.ones((*corners.shape[:2], 1), device=device, dtype=torch.float32)
    corners_h = torch.cat([corners, ones], dim=-1)
    cam = torch.matmul(corners_h, viewmat.T)[..., :3]

    x = cam[..., 0]
    y = cam[..., 1]
    z = cam[..., 2]
    fx = K[0, 0].clamp_min(1e-6)
    fy = K[1, 1].clamp_min(1e-6)
    cx = K[0, 2]
    cy = K[1, 2]

    left = x >= (-cx / fx) * z
    right = x <= ((float(width) - cx) / fx) * z
    top = y >= (-cy / fy) * z
    bottom = y <= ((float(height) - cy) / fy) * z
    near = z >= float(near_plane)
    far = z <= float(far_plane)

    outside = (
        (~left).all(dim=1)
        | (~right).all(dim=1)
        | (~top).all(dim=1)
        | (~bottom).all(dim=1)
        | (~near).all(dim=1)
        | (~far).all(dim=1)
    )
    return ~outside


def _block_padding_extent(
    cull_base_min: Tensor,
    cull_base_max: Tensor,
    cell_min: Tensor,
    cell_max: Tensor,
) -> Tensor:
    base_extent = (cull_base_max - cull_base_min).abs()
    cell_extent = (cell_max - cell_min).abs()
    extent = torch.maximum(base_extent, cell_extent)
    positive = extent[extent > 0.0]
    if positive.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    return positive.min().to(torch.float32)


def _scale_padding_from_splats(
    activated_scales: Tensor,
    mask: Tensor,
    scale_padding_factor: float,
    scale_padding_percentile: float,
    max_padding_fraction: Optional[float],
    cull_base_min: Tensor,
    cull_base_max: Tensor,
    cell_min: Tensor,
    cell_max: Tensor,
) -> float:
    if scale_padding_factor <= 0.0:
        return 0.0
    if scale_padding_percentile < 0.0 or scale_padding_percentile > 100.0:
        raise ValueError("scale_padding_percentile must be in [0, 100].")
    if max_padding_fraction is not None and max_padding_fraction < 0.0:
        raise ValueError("max_padding_fraction must be non-negative.")

    max_scales = activated_scales[mask].max(dim=1).values
    scale = float(torch.quantile(max_scales, float(scale_padding_percentile) / 100.0).item())
    padding = scale * float(scale_padding_factor)
    if max_padding_fraction is not None:
        extent = float(
            _block_padding_extent(
                cull_base_min,
                cull_base_max,
                cell_min,
                cell_max,
            ).item()
        )
        padding = min(padding, extent * float(max_padding_fraction))
    return padding


def build_partitioned_checkpoint(
    ckpt_paths: Sequence[Path],
    output_path: Path,
    grid: Tuple[int, int] = (4, 4),
    scale_padding_factor: float = 3.0,
    scale_padding_percentile: float = 99.5,
    max_padding_fraction: Optional[float] = 0.1,
    metadata_bounds_mode: Literal["exact", "robust"] = "exact",
    overwrite: bool = False,
    cleanup_metadata_path: Optional[Path] = None,
    ply_coordinate_mode: str = "native",
) -> Path:
    """Split existing gsplat checkpoints into XY blocks and save block metadata."""

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass overwrite=True to replace it.")
    if grid[0] < 1 or grid[1] < 1:
        raise ValueError("grid dimensions must be positive.")
    if scale_padding_factor < 0.0:
        raise ValueError("scale_padding_factor must be non-negative.")
    if scale_padding_percentile < 0.0 or scale_padding_percentile > 100.0:
        raise ValueError("scale_padding_percentile must be in [0, 100].")
    if max_padding_fraction is not None and max_padding_fraction < 0.0:
        raise ValueError("max_padding_fraction must be non-negative.")
    if metadata_bounds_mode not in {"exact", "robust"}:
        raise ValueError("metadata_bounds_mode must be 'exact' or 'robust'.")

    raw, steps = _load_raw_splats(
        [Path(p) for p in ckpt_paths],
        map_location="cpu",
        ply_coordinate_mode=ply_coordinate_mode,
    )
    means = raw["means"].to(torch.float32)
    total_gaussians = int(means.shape[0])
    if total_gaussians == 0:
        raise ValueError("Cannot partition an empty checkpoint.")

    cleanup_metadata = None
    bounds_valid_mask = torch.ones((total_gaussians,), dtype=torch.bool)
    bounds_filter_config = None
    if cleanup_metadata_path is not None:
        bounds_filter_config = load_bounds_filter_config_from_metadata(Path(cleanup_metadata_path))
        cleanup_result = analyze_checkpoint(
            {"splats": raw},
            bounds_filter_config,
            input_path=None,
            metadata_output_path=Path(cleanup_metadata_path),
        )
        bounds_valid_mask = cleanup_result.bounds_valid_mask
        cleanup_metadata = {
            "cleanup_metadata_path": str(Path(cleanup_metadata_path)),
            "bounds_filter_config": cleanup_result.diagnostics["bounds_filter_config"],
            "total_gaussian_count": cleanup_result.diagnostics["total_gaussian_count"],
            "bounds_valid_gaussian_count": cleanup_result.diagnostics["bounds_valid_gaussian_count"],
            "bounds_removed_gaussian_count": cleanup_result.diagnostics["bounds_removed_gaussian_count"],
            "bounds_removed_gaussian_percent": cleanup_result.diagnostics["bounds_removed_gaussian_percent"],
        }

    scene_min = means.min(dim=0).values
    scene_max = means.max(dim=0).values
    nx, ny = int(grid[0]), int(grid[1])
    x_edges = torch.linspace(scene_min[0], scene_max[0], nx + 1)
    y_edges = torch.linspace(scene_min[1], scene_max[1], ny + 1)
    ix = torch.bucketize(means[:, 0].contiguous(), x_edges[1:-1], right=False)
    iy = torch.bucketize(means[:, 1].contiguous(), y_edges[1:-1], right=False)
    block_ids = iy * nx + ix

    activated_scales = torch.exp(raw["scales"].to(torch.float32))
    blocks = []
    assigned = 0
    for by in range(ny):
        for bx in range(nx):
            block_id = by * nx + bx
            mask = block_ids == block_id
            count = int(mask.sum().item())
            if count == 0:
                continue
            assigned += count
            block_means = means[mask]
            raw_bounds_min_list, raw_bounds_max_list = tensor_bounds(block_means)
            content_min = torch.tensor(raw_bounds_min_list, dtype=torch.float32)
            content_max = torch.tensor(raw_bounds_max_list, dtype=torch.float32)
            cell_min = torch.tensor([x_edges[bx], y_edges[by], scene_min[2]], dtype=torch.float32)
            cell_max = torch.tensor([x_edges[bx + 1], y_edges[by + 1], scene_max[2]], dtype=torch.float32)
            block_bounds_mask = mask & bounds_valid_mask
            bounds_valid_count = int(block_bounds_mask.sum().item())
            used_filtered_fallback = False
            if cleanup_metadata_path is not None and bounds_valid_count > 0 and bounds_filter_config is not None:
                bounds_points = means[block_bounds_mask]
                if metadata_bounds_mode == "robust":
                    filtered_bounds = robust_bounds_for_points(bounds_points, bounds_filter_config)
                    filtered_min = torch.tensor(filtered_bounds["min"], dtype=torch.float32)
                    filtered_max = torch.tensor(filtered_bounds["max"], dtype=torch.float32)
                else:
                    filtered_min_list, filtered_max_list = tensor_bounds(bounds_points)
                    filtered_min = torch.tensor(filtered_min_list, dtype=torch.float32)
                    filtered_max = torch.tensor(filtered_max_list, dtype=torch.float32)
                scale_pad_source = block_bounds_mask
            else:
                filtered_min = content_min
                filtered_max = content_max
                scale_pad_source = mask
                used_filtered_fallback = cleanup_metadata_path is not None and bounds_valid_count == 0

            block_scale_pad = _scale_padding_from_splats(
                activated_scales,
                scale_pad_source,
                float(scale_padding_factor),
                float(scale_padding_percentile),
                max_padding_fraction,
                filtered_min,
                filtered_max,
                cell_min,
                cell_max,
            )
            if cleanup_metadata_path is not None:
                cull_min = filtered_min - block_scale_pad
                cull_max = filtered_max + block_scale_pad
            else:
                cull_min = cell_min - block_scale_pad
                cull_max = cell_max + block_scale_pad
            raw_volume = aabb_volume((raw_bounds_min_list, raw_bounds_max_list))
            filtered_volume = aabb_volume(
                (
                    [float(v) for v in filtered_min.tolist()],
                    [float(v) for v in filtered_max.tolist()],
                )
            )
            volume_ratio = float("inf") if filtered_volume == 0.0 else raw_volume / filtered_volume

            blocks.append(
                {
                    "block_id": block_id,
                    "grid_index": [bx, by],
                    "gaussian_count": count,
                    "cell_aabb_min": cell_min,
                    "cell_aabb_max": cell_max,
                    "logical_bounds_min": cell_min,
                    "logical_bounds_max": cell_max,
                    "raw_gaussian_bounds_min": content_min,
                    "raw_gaussian_bounds_max": content_max,
                    "filtered_render_bounds_min": filtered_min,
                    "filtered_render_bounds_max": filtered_max,
                    "raw_gaussian_count": count,
                    "bounds_valid_gaussian_count": bounds_valid_count,
                    "raw_to_filtered_aabb_volume_ratio": volume_ratio,
                    "filtered_render_bounds_fallback_to_raw": used_filtered_fallback,
                    "content_aabb_min": content_min,
                    "content_aabb_max": content_max,
                    "cull_aabb_min": cull_min,
                    "cull_aabb_max": cull_max,
                    "tensors": {key: value[mask].contiguous() for key, value in raw.items()},
                }
            )

    if assigned != total_gaussians:
        raise RuntimeError(f"Partition assignment mismatch: assigned {assigned}, expected {total_gaussians}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": BLOCK_FILE_VERSION,
            "source_checkpoint_paths": [str(Path(p)) for p in ckpt_paths],
            "source_steps": steps,
            "ply_coordinate_mode": ply_coordinate_mode,
            "grid": [nx, ny],
            "scale_padding_factor": float(scale_padding_factor),
            "scale_padding_percentile": float(scale_padding_percentile),
            "max_padding_fraction": (
                None if max_padding_fraction is None else float(max_padding_fraction)
            ),
            "metadata_bounds_mode": metadata_bounds_mode,
            "total_gaussians": total_gaussians,
            "scene_aabb_min": scene_min,
            "scene_aabb_max": scene_max,
            "cleanup_metadata": cleanup_metadata,
            "blocks": blocks,
        },
        output_path,
    )
    return output_path


class BlockManager:
    def __init__(
        self,
        blocks: List[GaussianBlock],
        device: torch.device,
        total_gaussians: int,
        sh_degree: int,
        colors_shape: Tuple[int, int],
        residency: str = "gpu",
        keepalive_frames: int = 0,
    ):
        if residency not in {"gpu", "cpu"}:
            raise ValueError("residency must be 'gpu' or 'cpu'.")
        self.blocks = sorted(blocks, key=lambda block: block.block_id)
        self.blocks_by_id = {block.block_id: block for block in self.blocks}
        self.device = device
        self.total_gaussians = int(total_gaussians)
        self.sh_degree = int(sh_degree)
        self.colors_shape = colors_shape
        self.residency = residency
        self.keepalive_frames = int(keepalive_frames)
        self.frame_index = 0
        self._gpu_cache: Dict[int, Dict[str, Tensor]] = {}
        self._active_cache_key: Optional[Tuple[int, ...]] = None
        self._active_cache: Optional[Dict[str, Tensor]] = None
        self._cull_mins = torch.stack([block.cull_aabb_min for block in self.blocks], dim=0).cpu()
        self._cull_maxs = torch.stack([block.cull_aabb_max for block in self.blocks], dim=0).cpu()

    @classmethod
    def from_file(
        cls,
        path: Path,
        device: torch.device,
        residency: str = "gpu",
        keepalive_frames: int = 0,
    ) -> "BlockManager":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        version = int(payload.get("format_version", 0))
        if version != BLOCK_FILE_VERSION:
            raise ValueError(f"Unsupported block file version {version}; expected {BLOCK_FILE_VERSION}.")

        blocks: List[GaussianBlock] = []
        sh_degree = -1
        colors_shape: Optional[Tuple[int, int]] = None
        target_device = device if residency == "gpu" else torch.device("cpu")

        for block_payload in payload["blocks"]:
            activated = _activate_splats(block_payload["tensors"], target_device)
            if colors_shape is None:
                colors_shape = (int(activated["colors"].shape[-2]), int(activated["colors"].shape[-1]))
                sh_degree = int(math.sqrt(colors_shape[0]) - 1)
            block = GaussianBlock(
                block_id=int(block_payload["block_id"]),
                grid_index=tuple(int(v) for v in block_payload["grid_index"]),
                gaussian_count=int(block_payload["gaussian_count"]),
                content_aabb_min=block_payload["content_aabb_min"].to(torch.float32),
                content_aabb_max=block_payload["content_aabb_max"].to(torch.float32),
                cull_aabb_min=block_payload["cull_aabb_min"].to(torch.float32),
                cull_aabb_max=block_payload["cull_aabb_max"].to(torch.float32),
                logical_bounds_min=block_payload.get("logical_bounds_min", block_payload.get("cell_aabb_min", block_payload["content_aabb_min"])).to(torch.float32),
                logical_bounds_max=block_payload.get("logical_bounds_max", block_payload.get("cell_aabb_max", block_payload["content_aabb_max"])).to(torch.float32),
                raw_gaussian_bounds_min=block_payload.get("raw_gaussian_bounds_min", block_payload["content_aabb_min"]).to(torch.float32),
                raw_gaussian_bounds_max=block_payload.get("raw_gaussian_bounds_max", block_payload["content_aabb_max"]).to(torch.float32),
                filtered_render_bounds_min=block_payload.get("filtered_render_bounds_min", block_payload["cull_aabb_min"]).to(torch.float32),
                filtered_render_bounds_max=block_payload.get("filtered_render_bounds_max", block_payload["cull_aabb_max"]).to(torch.float32),
                raw_gaussian_count=int(block_payload.get("raw_gaussian_count", block_payload["gaussian_count"])),
                bounds_valid_gaussian_count=int(block_payload.get("bounds_valid_gaussian_count", block_payload["gaussian_count"])),
                raw_to_filtered_aabb_volume_ratio=float(block_payload.get("raw_to_filtered_aabb_volume_ratio", 1.0)),
                cpu_tensors=activated if residency == "cpu" else None,
                gpu_tensors=activated if residency == "gpu" else None,
            )
            blocks.append(block)

        if colors_shape is None:
            raise ValueError(f"{path} does not contain any non-empty blocks.")

        return cls(
            blocks=blocks,
            device=device,
            total_gaussians=int(payload["total_gaussians"]),
            sh_degree=sh_degree,
            colors_shape=colors_shape,
            residency=residency,
            keepalive_frames=keepalive_frames,
        )

    @property
    def total_blocks(self) -> int:
        return len(self.blocks)

    def resident_block_ids(self) -> List[int]:
        if self.residency == "gpu":
            return [block.block_id for block in self.blocks]
        return sorted(self._gpu_cache.keys())

    def active_block_ids(self) -> List[int]:
        if self._active_cache_key is None:
            return []
        return list(self._active_cache_key)

    def debug_blocks(self) -> List[BlockDebugInfo]:
        active_ids = set(self.active_block_ids())
        resident_ids = set(self.resident_block_ids())
        return [
            BlockDebugInfo(
                block_id=block.block_id,
                grid_index=block.grid_index,
                gaussian_count=block.gaussian_count,
                content_aabb_min=tuple(float(v) for v in block.content_aabb_min.tolist()),
                content_aabb_max=tuple(float(v) for v in block.content_aabb_max.tolist()),
                cull_aabb_min=tuple(float(v) for v in block.cull_aabb_min.tolist()),
                cull_aabb_max=tuple(float(v) for v in block.cull_aabb_max.tolist()),
                logical_bounds_min=tuple(float(v) for v in block.logical_bounds_min.tolist()),
                logical_bounds_max=tuple(float(v) for v in block.logical_bounds_max.tolist()),
                raw_gaussian_bounds_min=tuple(float(v) for v in block.raw_gaussian_bounds_min.tolist()),
                raw_gaussian_bounds_max=tuple(float(v) for v in block.raw_gaussian_bounds_max.tolist()),
                filtered_render_bounds_min=tuple(float(v) for v in block.filtered_render_bounds_min.tolist()),
                filtered_render_bounds_max=tuple(float(v) for v in block.filtered_render_bounds_max.tolist()),
                raw_gaussian_count=block.raw_gaussian_count,
                bounds_valid_gaussian_count=block.bounds_valid_gaussian_count,
                raw_to_filtered_aabb_volume_ratio=block.raw_to_filtered_aabb_volume_ratio,
                active=block.block_id in active_ids,
                resident=block.block_id in resident_ids,
                empty=block.gaussian_count == 0,
            )
            for block in self.blocks
        ]

    def _select_active_ids(
        self,
        c2w: Tensor,
        K: Tensor,
        width: int,
        height: int,
        near_plane: float,
        far_plane: float,
        force_all: bool,
        culling_mode: BlockCullingMode,
        selected_block_id: Optional[int],
    ) -> Tuple[List[int], float]:
        tic = time.perf_counter()
        if force_all:
            culling_mode = "force_all_blocks_active"

        if culling_mode == "force_all_blocks_active":
            visible_now = torch.ones((len(self.blocks),), dtype=torch.bool)
        elif culling_mode == "force_no_blocks_active":
            visible_now = torch.zeros((len(self.blocks),), dtype=torch.bool)
        elif culling_mode == "selected_block_only":
            visible_now = torch.tensor(
                [
                    block.block_id == selected_block_id
                    for block in self.blocks
                ],
                dtype=torch.bool,
            )
        else:
            visible_now = aabbs_intersect_frustum(
                self._cull_mins,
                self._cull_maxs,
                c2w.cpu(),
                K.cpu(),
                width,
                height,
                near_plane,
                far_plane,
            )

        active_ids = []
        for idx, block in enumerate(self.blocks):
            if bool(visible_now[idx]):
                block.last_active_frame = self.frame_index
            if culling_mode in {
                "force_all_blocks_active",
                "force_no_blocks_active",
                "selected_block_only",
            }:
                if bool(visible_now[idx]):
                    active_ids.append(block.block_id)
            elif block.last_active_frame >= 0 and self.frame_index - block.last_active_frame <= self.keepalive_frames:
                active_ids.append(block.block_id)
        active_ids.sort()
        return active_ids, (time.perf_counter() - tic) * 1000.0

    def _get_gpu_block_tensors(self, active_ids: Iterable[int]) -> Tuple[List[Dict[str, Tensor]], float, int, int]:
        upload_ms = 0.0
        cache_hits = 0
        cache_misses = 0
        block_tensors: List[Dict[str, Tensor]] = []

        if self.residency == "gpu":
            for block_id in active_ids:
                tensors = self.blocks_by_id[block_id].gpu_tensors
                if tensors is None:
                    raise RuntimeError(f"GPU-resident block {block_id} has no GPU tensors.")
                block_tensors.append(tensors)
            return block_tensors, upload_ms, cache_hits, cache_misses

        wanted = set(active_ids)
        for cached_id in list(self._gpu_cache.keys()):
            if cached_id not in wanted:
                del self._gpu_cache[cached_id]

        for block_id in active_ids:
            cached = self._gpu_cache.get(block_id)
            if cached is not None:
                cache_hits += 1
                block_tensors.append(cached)
                continue
            cache_misses += 1
            block = self.blocks_by_id[block_id]
            if block.cpu_tensors is None:
                raise RuntimeError(f"CPU-resident block {block_id} has no CPU tensors.")
            tic = time.perf_counter()
            uploaded = {
                key: value.to(device=self.device, non_blocking=False).contiguous()
                for key, value in block.cpu_tensors.items()
            }
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            upload_ms += (time.perf_counter() - tic) * 1000.0
            self._gpu_cache[block_id] = uploaded
            block_tensors.append(uploaded)

        return block_tensors, upload_ms, cache_hits, cache_misses

    def get_active_splats(
        self,
        c2w: Tensor,
        K: Tensor,
        width: int,
        height: int,
        near_plane: float,
        far_plane: float,
        force_all: bool = False,
        culling_mode: BlockCullingMode = "normal_frustum_culling",
        selected_block_id: Optional[int] = None,
    ) -> ActiveSplats:
        self.frame_index += 1
        active_ids, cull_ms = self._select_active_ids(
            c2w,
            K,
            width,
            height,
            near_plane,
            far_plane,
            force_all,
            culling_mode,
            selected_block_id,
        )
        cache_key = tuple(active_ids)
        block_tensors, upload_ms, cache_hits, cache_misses = self._get_gpu_block_tensors(active_ids)

        concat_ms = 0.0
        if cache_key == self._active_cache_key and self._active_cache is not None:
            active = self._active_cache
        else:
            tic = time.perf_counter()
            active = _cat_block_tensors(block_tensors, self.device, self.colors_shape)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            concat_ms = (time.perf_counter() - tic) * 1000.0
            self._active_cache_key = cache_key
            self._active_cache = active

        active_gaussians = int(active["means"].shape[0])
        if self.residency == "gpu":
            resident_blocks = self.total_blocks
            resident_gaussians = self.total_gaussians
        else:
            resident_blocks = len(self._gpu_cache)
            resident_gaussians = sum(self.blocks_by_id[block_id].gaussian_count for block_id in self._gpu_cache)

        stats = BlockRenderStats(
            total_blocks=self.total_blocks,
            active_blocks=len(active_ids),
            active_block_ids=" ".join(str(block_id) for block_id in active_ids),
            active_gaussians=active_gaussians,
            active_gaussian_fraction=active_gaussians / max(self.total_gaussians, 1),
            resident_blocks=resident_blocks,
            resident_gaussians=resident_gaussians,
            resident_gaussian_fraction=resident_gaussians / max(self.total_gaussians, 1),
            block_cull_ms=cull_ms,
            block_concat_ms=concat_ms,
            gpu_upload_ms=upload_ms,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )
        return ActiveSplats(
            means=active["means"],
            quats=active["quats"],
            scales=active["scales"],
            opacities=active["opacities"],
            colors=active["colors"],
            stats=stats,
        )
