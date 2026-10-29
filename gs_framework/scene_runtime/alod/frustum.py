"""Pinned four-plane frustum operations for the production ALoD path."""

from __future__ import annotations

import numpy as np
import torch


def extract_four_planes(full_projection_transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(full_projection_transform)
    if matrix.dtype != np.dtype("float32") or matrix.shape != (4, 4):
        raise ValueError("full_projection_transform must be float32[4,4]")
    transposed = matrix.T
    planes = np.stack(
        (
            transposed[3] + transposed[0],
            transposed[3] - transposed[0],
            transposed[3] + transposed[1],
            transposed[3] - transposed[1],
        )
    ).astype(np.float32, copy=False)
    squares = np.asarray(planes[:, :3] * planes[:, :3], dtype=np.float32)
    lengths = np.sqrt(
        np.asarray(
            np.asarray(squares[:, 0] + squares[:, 2], dtype=np.float32)
            + squares[:, 1],
            dtype=np.float32,
        )
    ).astype(np.float32)[:, None]
    if np.any(lengths == 0) or not np.all(np.isfinite(lengths)):
        raise ValueError("cannot normalize degenerate frustum plane")
    return np.ascontiguousarray(planes / lengths, dtype=np.float32)


def scale_bounds(log_scaling: np.ndarray) -> np.ndarray:
    values = np.asarray(log_scaling)
    if values.dtype != np.dtype("float32") or values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("log_scaling must be float32[N,3]")
    tensor = torch.from_numpy(np.array(values, copy=True))
    result = torch.exp(torch.max(tensor, dim=-1)[0]) * 3.0
    return np.ascontiguousarray(result.numpy(), dtype=np.float32)


def _sum3(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.asarray(values[..., 0] + values[..., 2], dtype=np.float32)
        + values[..., 1],
        dtype=np.float32,
    )


def visible_spheres(
    points: np.ndarray, radii: np.ndarray, planes: np.ndarray
) -> np.ndarray:
    centers = np.asarray(points)
    bounds = np.asarray(radii)
    active_planes = np.asarray(planes)
    if centers.dtype != np.dtype("float32") or centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("points must be float32[N,3]")
    if bounds.dtype != np.dtype("float32") or bounds.shape != (centers.shape[0],):
        raise ValueError("radii must be float32[N]")
    if active_planes.dtype != np.dtype("float32") or active_planes.shape != (4, 4):
        raise ValueError("planes must be float32[4,4]")
    visible = np.ones(centers.shape[0], dtype=bool)
    for plane in active_planes:
        products = np.asarray(centers * plane[:3], dtype=np.float32)
        signed = np.asarray(_sum3(products) + plane[3], dtype=np.float32)
        visible[signed + bounds < np.float32(0.0)] = False
    return visible


__all__ = ["extract_four_planes", "scale_bounds", "visible_spheres"]
