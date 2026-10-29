"""Pinned-source four-plane frustum math for the CPU ALoD reference.

This is intentionally not a generic graphics frustum helper.  It reproduces
the official evaluation convention: transpose the stored full projection
transform, extract left/right/bottom/top only, normalize each plane, and keep
a sphere unless ``signed_distance + radius < 0`` for any active plane.
"""

from __future__ import annotations

import numpy as np
import torch


def extract_four_frustum_planes(full_projection_transform: np.ndarray) -> np.ndarray:
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
    normal_squares = np.asarray(planes[:, :3] * planes[:, :3], dtype=np.float32)
    normal_lengths = np.sqrt(
        np.asarray(
            np.asarray(normal_squares[:, 0] + normal_squares[:, 2], dtype=np.float32)
            + normal_squares[:, 1],
            dtype=np.float32,
        )
    ).astype(np.float32)[:, None]
    if np.any(normal_lengths == 0) or not np.all(np.isfinite(normal_lengths)):
        raise ValueError("cannot normalize degenerate frustum plane")
    return np.ascontiguousarray(planes / normal_lengths, dtype=np.float32)


def scale_derived_bounds(upper_tree_log_scaling: np.ndarray) -> np.ndarray:
    """Compute ``3 * exp(max(log_scale))`` without leaving CPU.

    The official evaluation computes this expression on CUDA.  CPU PyTorch is
    used here as the closest source-order float32 reference; selection parity,
    rather than an unavailable CUDA-intermediate payload, is authoritative.
    """

    log_scaling = np.asarray(upper_tree_log_scaling)
    if log_scaling.dtype != np.dtype("float32") or log_scaling.ndim != 2 or log_scaling.shape[1] != 3:
        raise ValueError("upper_tree_log_scaling must be float32[N,3]")
    tensor = torch.from_numpy(np.array(log_scaling, copy=True))
    if tensor.device.type != "cpu":
        raise RuntimeError("ALoD reference tensors must remain on CPU")
    result = torch.exp(torch.max(tensor, dim=-1)[0]) * 3.0
    if result.device.type != "cpu":
        raise RuntimeError("ALoD reference tensors must remain on CPU")
    return np.ascontiguousarray(result.numpy(), dtype=np.float32)


def _sum3_cuda_order(values: np.ndarray) -> np.ndarray:
    """Match the observed three-component CUDA reduction order: ``(x+z)+y``."""

    return np.asarray(
        np.asarray(values[..., 0] + values[..., 2], dtype=np.float32)
        + values[..., 1],
        dtype=np.float32,
    )


def frustum_visible_spheres(
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
        signed = np.asarray(
            _sum3_cuda_order(products) + plane[3], dtype=np.float32
        )
        visible[signed + bounds < np.float32(0.0)] = False
    return visible


__all__ = [
    "extract_four_frustum_planes",
    "frustum_visible_spheres",
    "scale_derived_bounds",
]
