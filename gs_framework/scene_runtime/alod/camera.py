"""Camera conversion for the production ALoD selector."""

from __future__ import annotations

import math

import numpy as np
import torch

from scene_runtime.types import FrameCamera, RenderSettings

from .types import AlodSelectionCamera


def _official_projection_matrix(
    znear: float, zfar: float, fov_x: float, fov_y: float, device: torch.device
) -> torch.Tensor:
    tan_half_y = math.tan(float(fov_y) * 0.5)
    tan_half_x = math.tan(float(fov_x) * 0.5)
    top = tan_half_y * float(znear)
    bottom = -top
    right = tan_half_x * float(znear)
    left = -right
    matrix = torch.zeros((4, 4), dtype=torch.float32, device=device)
    matrix[0, 0] = 2.0 * float(znear) / (right - left)
    matrix[1, 1] = 2.0 * float(znear) / (top - bottom)
    matrix[0, 2] = (right + left) / (right - left)
    matrix[1, 2] = (top + bottom) / (top - bottom)
    matrix[3, 2] = 1.0
    matrix[2, 2] = float(zfar) / (float(zfar) - float(znear))
    matrix[2, 3] = -(float(zfar) * float(znear)) / (float(zfar) - float(znear))
    return matrix.T.contiguous()


def selection_camera_from_frame(
    camera: FrameCamera,
    render_settings: RenderSettings,
    *,
    stable_id: str = "runtime-camera",
    projection_znear: float = 0.01,
    projection_zfar: float = 100.0,
) -> AlodSelectionCamera:
    """Create the source-faithful selection camera.

    Frozen validation cameras may provide the exact official camera center and
    full projection matrix through ``FrameCamera``.  Interactive cameras use
    the same official construction on the current tensor device.
    """

    exact_center = camera.selection_camera_center
    exact_full = camera.selection_full_projection_transform
    if exact_center is not None and exact_full is not None:
        return AlodSelectionCamera(
            stable_id=stable_id,
            camera_center=np.ascontiguousarray(
                exact_center.detach().to(device="cpu", dtype=torch.float32).numpy()
            ),
            full_projection_transform=np.ascontiguousarray(
                exact_full.detach().to(device="cpu", dtype=torch.float32).numpy()
            ),
        )

    device = camera.c2w.device
    c2w = camera.c2w.to(device=device, dtype=torch.float32)
    K = camera.K.to(device=device, dtype=torch.float32)
    viewmat = (
        camera.w2c.to(device=device, dtype=torch.float32)
        if camera.w2c is not None
        else torch.linalg.inv(c2w)
    )
    world_view = viewmat.T.contiguous()
    fov_x = 2.0 * math.atan(float(render_settings.width) / (2.0 * float(K[0, 0].item())))
    fov_y = 2.0 * math.atan(float(render_settings.height) / (2.0 * float(K[1, 1].item())))
    projection = _official_projection_matrix(
        projection_znear, projection_zfar, fov_x, fov_y, device
    )
    full = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    center = (
        camera.selection_camera_center.to(device=device, dtype=torch.float32)
        if camera.selection_camera_center is not None
        else c2w[:3, 3]
    )
    return AlodSelectionCamera(
        stable_id=stable_id,
        camera_center=np.ascontiguousarray(center.detach().cpu().numpy(), dtype=np.float32),
        full_projection_transform=np.ascontiguousarray(
            full.detach().cpu().numpy(), dtype=np.float32
        ),
    )

