from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Literal, Sequence

import torch

from block_manager import BLOCK_FILE_VERSION

from .citygaussian import (
    OFFICIAL_MATRIXCITY_AABB,
    OFFICIAL_MATRIXCITY_BLOCK_DIM,
    CityGaussianFixedAllScene,
    CityGaussianReleaseLodScene,
    CityGaussianVisibilityScene,
)
from .alod.scene import AlodResidentScene
from .alod.host_scene import AlodHostScene
from .alod.types import AlodHostResidencyConfig, AlodRuntimeConfig
from .scenes import REQUIRED_SPLAT_KEYS, BlockedGaussianScene, FlatGaussianScene, GaussianScene
from .ply_splats import is_ply_path


SceneType = Literal[
    "auto",
    "flat",
    "blocked",
    "citygaussian_fixed_all",
    "citygaussian_visibility",
    "citygaussian_lod_release",
    "alod_resident",
    "alod_host_nocache",
    "alod_host_previous_reuse",
    "alod_host_viewer_previous_reuse",
    "alod_host_bounded_cache",
]
PlyCoordinateMode = Literal["native", "supersplat"]
LoadProgressCallback = Callable[[str], None]


class SceneLoader:
    @staticmethod
    def load(
        paths: Sequence[Path],
        device: torch.device,
        scene_type: SceneType = "auto",
        block_residency: str = "gpu",
        block_keepalive_frames: int = 0,
        gpu_block_budget_mb: float | None = None,
        block_cache_policy: str = "lru",
        culling_mode: str = "normal_frustum_culling",
        ply_coordinate_mode: PlyCoordinateMode = "native",
        citygaussian_block_dim: Sequence[int] = OFFICIAL_MATRIXCITY_BLOCK_DIM,
        citygaussian_aabb: Sequence[float] = OFFICIAL_MATRIXCITY_AABB,
        citygaussian_lod_source_level: str = "light_50_vq",
        alod_config: AlodRuntimeConfig | None = None,
        alod_host_config: AlodHostResidencyConfig | None = None,
        progress_callback: LoadProgressCallback | None = None,
    ) -> GaussianScene:
        scene_paths = [Path(path) for path in paths]
        if not scene_paths:
            raise ValueError("SceneLoader requires at least one scene path.")
        if scene_type not in {
            "auto",
            "flat",
            "blocked",
            "citygaussian_fixed_all",
            "citygaussian_visibility",
            "citygaussian_lod_release",
            "alod_resident",
            "alod_host_nocache",
            "alod_host_previous_reuse",
            "alod_host_viewer_previous_reuse",
            "alod_host_bounded_cache",
        }:
            raise ValueError(
                "scene_type must be one of: auto, flat, blocked, citygaussian_fixed_all, "
                "citygaussian_visibility, citygaussian_lod_release, alod_resident, "
                "alod_host_nocache, alod_host_previous_reuse, "
                "alod_host_viewer_previous_reuse, alod_host_bounded_cache."
            )

        host_scene_types = {
            "alod_host_nocache": "host_nocache",
            "alod_host_previous_reuse": "host_previous_reuse",
            "alod_host_viewer_previous_reuse": "host_viewer_previous_reuse",
            "alod_host_bounded_cache": "host_bounded_cache",
        }
        if scene_type in host_scene_types:
            if len(scene_paths) != 1 or scene_paths[0].suffix.lower() != ".dhier":
                raise ValueError("ALoD host scene requires exactly one .dhier asset.")
            if alod_config is None or alod_host_config is None:
                raise ValueError(
                    "ALoD host scene requires AlodRuntimeConfig and AlodHostResidencyConfig."
                )
            expected_mode = host_scene_types[scene_type]
            if alod_host_config.mode != expected_mode:
                raise ValueError(
                    f"scene_type {scene_type!r} requires host mode {expected_mode!r}"
                )
            return AlodHostScene(
                scene_paths[0],
                device=device,
                config=alod_config,
                residency_config=alod_host_config,
                progress_callback=progress_callback,
            )

        if scene_type == "alod_resident":
            if len(scene_paths) != 1 or scene_paths[0].suffix.lower() != ".dhier":
                raise ValueError("ALoD resident scene requires exactly one .dhier asset.")
            if alod_config is None:
                raise ValueError(
                    "ALoD resident scene requires an explicit AlodRuntimeConfig; "
                    "asset-specific evaluation constants are not runtime defaults."
                )
            return AlodResidentScene(
                scene_paths[0],
                device=device,
                config=alod_config,
                progress_callback=progress_callback,
            )

        if scene_type == "citygaussian_fixed_all":
            if len(scene_paths) != 1:
                raise ValueError("CityGaussian fixed-all scene supports exactly one finest-LoD PLY path.")
            if not is_ply_path(scene_paths[0]):
                raise ValueError("CityGaussian fixed-all scene requires a Gaussian PLY asset.")
            return CityGaussianFixedAllScene(
                scene_paths[0],
                device=device,
                block_dim=citygaussian_block_dim,
                aabb=citygaussian_aabb,
                lod_source_level=citygaussian_lod_source_level,
                ply_coordinate_mode=ply_coordinate_mode,
                progress_callback=progress_callback,
            )
        if scene_type == "citygaussian_visibility":
            if len(scene_paths) != 1:
                raise ValueError("CityGaussian visibility scene supports exactly one PLY path.")
            if not is_ply_path(scene_paths[0]):
                raise ValueError("CityGaussian visibility scene requires a Gaussian PLY asset.")
            return CityGaussianVisibilityScene(
                scene_paths[0],
                device=device,
                block_dim=citygaussian_block_dim,
                aabb=citygaussian_aabb,
                lod_source_level=citygaussian_lod_source_level,
                ply_coordinate_mode=ply_coordinate_mode,
                progress_callback=progress_callback,
            )
        if scene_type == "citygaussian_lod_release":
            if len(scene_paths) != 3:
                raise ValueError(
                    "CG-LOD-RELEASE requires exactly three PLY paths in finest, middle, "
                    "coarsest order."
                )
            if not all(is_ply_path(path) for path in scene_paths):
                raise ValueError("CG-LOD-RELEASE requires three Gaussian PLY assets.")
            return CityGaussianReleaseLodScene(
                scene_paths,
                device=device,
                block_dim=citygaussian_block_dim,
                aabb=citygaussian_aabb,
                ply_coordinate_mode=ply_coordinate_mode,
                progress_callback=progress_callback,
            )

        kinds = [_detect_payload_kind(path) for path in scene_paths]
        if scene_type == "auto":
            unique = set(kinds)
            if len(unique) != 1:
                raise ValueError(
                    "Mixed scene inputs are not supported: "
                    + ", ".join(f"{path}={kind}" for path, kind in zip(scene_paths, kinds))
                )
            scene_type = kinds[0]
        else:
            wrong = [
                f"{path}={kind}"
                for path, kind in zip(scene_paths, kinds)
                if kind != scene_type
            ]
            if wrong:
                raise ValueError(
                    f"scene_type={scene_type!r} does not match input payload(s): "
                    + ", ".join(wrong)
                )

        if scene_type == "alod_resident":
            if len(scene_paths) != 1:
                raise ValueError("ALoD resident scene supports exactly one .dhier asset.")
            if alod_config is None:
                raise ValueError(
                    "Auto-detected .dhier requires an explicit AlodRuntimeConfig."
                )
            return AlodResidentScene(
                scene_paths[0],
                device=device,
                config=alod_config,
                progress_callback=progress_callback,
            )
        if scene_type == "flat":
            return FlatGaussianScene(
                scene_paths,
                device=device,
                ply_coordinate_mode=ply_coordinate_mode,
                progress_callback=progress_callback,
            )
        if len(scene_paths) != 1:
            raise ValueError(
                "BlockedGaussianScene currently supports exactly one blocked scene path. "
                "Use partition_checkpoint.py to cook one blocked asset."
            )
        return BlockedGaussianScene(
            scene_paths[0],
            device=device,
            residency=block_residency,
            keepalive_frames=block_keepalive_frames,
            gpu_block_budget_mb=gpu_block_budget_mb,
            cache_policy=block_cache_policy,
            culling_mode=culling_mode,
        )


def _detect_payload_kind(path: Path) -> Literal["flat", "blocked", "alod_resident"]:
    if path.suffix.lower() == ".dhier":
        return "alod_resident"
    if is_ply_path(path):
        return "flat"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    flat_error = _validate_flat_payload(payload)
    blocked_error = _validate_blocked_payload(payload)
    if flat_error is None and blocked_error is None:
        raise ValueError(f"{path} looks like both a flat and blocked scene; refusing ambiguous input.")
    if flat_error is None:
        return "flat"
    if blocked_error is None:
        return "blocked"
    raise ValueError(
        f"{path} is neither a strict flat checkpoint nor a blocked scene. "
        f"Flat error: {flat_error} Blocked error: {blocked_error}"
    )


def _validate_flat_payload(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return "payload is not a dict."
    if "splats" not in payload:
        return "missing 'splats'."
    splats = payload["splats"]
    if not isinstance(splats, dict):
        return "'splats' is not a dict."
    missing = [key for key in REQUIRED_SPLAT_KEYS if key not in splats]
    if missing:
        return f"missing splat keys {missing}."
    non_tensors = [key for key in REQUIRED_SPLAT_KEYS if not torch.is_tensor(splats[key])]
    if non_tensors:
        return f"splat keys are not tensors: {non_tensors}."
    return None


def _validate_blocked_payload(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return "payload is not a dict."
    required_top = ["format_version", "blocks", "total_gaussians"]
    missing_top = [key for key in required_top if key not in payload]
    if missing_top:
        return f"missing top-level keys {missing_top}."
    if int(payload["format_version"]) != BLOCK_FILE_VERSION:
        return f"unsupported format_version {payload['format_version']}."
    blocks = payload["blocks"]
    if not isinstance(blocks, list) or len(blocks) == 0:
        return "'blocks' must be a non-empty list."

    required_block = [
        "block_id",
        "grid_index",
        "gaussian_count",
        "content_aabb_min",
        "content_aabb_max",
        "cull_aabb_min",
        "cull_aabb_max",
        "tensors",
    ]
    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            return f"block {idx} is not a dict."
        missing_block = [key for key in required_block if key not in block]
        if missing_block:
            return f"block {idx} missing keys {missing_block}."
        tensors = block["tensors"]
        if not isinstance(tensors, dict):
            return f"block {idx} tensors is not a dict."
        missing_tensors = [key for key in REQUIRED_SPLAT_KEYS if key not in tensors]
        if missing_tensors:
            return f"block {idx} missing tensor keys {missing_tensors}."
        non_tensors = [key for key in REQUIRED_SPLAT_KEYS if not torch.is_tensor(tensors[key])]
        if non_tensors:
            return f"block {idx} tensor keys are not tensors: {non_tensors}."
    return None
