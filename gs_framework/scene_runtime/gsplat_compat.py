from __future__ import annotations

from typing import Literal


RasterizeMode = Literal["classic", "antialiased"]
CameraModel = Literal["pinhole", "ortho", "fisheye", "ftheta"]

SUPPORTED_RASTERIZE_MODES: tuple[RasterizeMode, ...] = ("classic", "antialiased")
VIEWER_CAMERA_MODELS: tuple[CameraModel, ...] = ("pinhole", "ortho", "fisheye")


def normalize_rasterize_mode(value: str) -> RasterizeMode:
    if value not in SUPPORTED_RASTERIZE_MODES:
        supported = ", ".join(SUPPORTED_RASTERIZE_MODES)
        raise ValueError(f"Unsupported gsplat rasterize_mode {value!r}; expected one of: {supported}")
    return value  # type: ignore[return-value]
