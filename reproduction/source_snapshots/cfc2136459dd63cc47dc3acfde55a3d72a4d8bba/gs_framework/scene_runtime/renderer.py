from __future__ import annotations

import time

import torch
from gsplat.rendering import rasterization

from .gsplat_compat import normalize_rasterize_mode
from .types import ActiveSplats, FrameCamera, RenderResult, RenderSettings


class GsplatRenderer:
    def __init__(self, device: torch.device):
        self.device = device

    @torch.no_grad()
    def render(
        self,
        active_splats: ActiveSplats,
        camera: FrameCamera,
        render_settings: RenderSettings,
        log_gpu_events: bool = False,
        measure_time: bool = True,
    ) -> RenderResult:
        camera = camera.to(self.device)
        background = torch.tensor(
            [render_settings.background],
            device=self.device,
            dtype=torch.float32,
        )

        start_event = end_event = None
        if log_gpu_events and self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        if self.device.type == "cuda" and measure_time:
            torch.cuda.synchronize(self.device)
        tic = time.perf_counter()
        if start_event is not None:
            start_event.record()

        if active_splats.num_gaussians == 0:
            image = background[0].view(1, 1, 3).expand(
                int(render_settings.height), int(render_settings.width), 3
            ).contiguous()
            alpha = torch.zeros(
                (int(render_settings.height), int(render_settings.width), 1),
                device=self.device,
                dtype=torch.float32,
            )
            info = {}
        else:
            render_colors, render_alphas, info = rasterization(
                means=active_splats.means,
                quats=active_splats.quats,
                scales=active_splats.scales,
                opacities=active_splats.opacities,
                colors=active_splats.colors,
                viewmats=torch.linalg.inv(camera.c2w)[None],
                Ks=camera.K[None],
                width=int(render_settings.width),
                height=int(render_settings.height),
                sh_degree=render_settings.sh_degree,
                near_plane=float(render_settings.near_plane),
                far_plane=float(render_settings.far_plane),
                radius_clip=float(render_settings.radius_clip),
                eps2d=float(render_settings.eps2d),
                backgrounds=background,
                render_mode=render_settings.render_mode,
                rasterize_mode=normalize_rasterize_mode(render_settings.rasterize_mode),
                camera_model=render_settings.camera_model,
                packed=bool(render_settings.packed),
                distributed=bool(render_settings.distributed),
                with_ut=bool(render_settings.with_ut),
                with_eval3d=bool(render_settings.with_eval3d),
            )
            image = render_colors[0]
            alpha = render_alphas[0]

        if start_event is not None and end_event is not None:
            end_event.record()
        if self.device.type == "cuda" and measure_time:
            torch.cuda.synchronize(self.device)
        render_time_ms = (time.perf_counter() - tic) * 1000.0 if measure_time else float("nan")

        gpu_time_ms = float("nan")
        if start_event is not None and end_event is not None:
            gpu_time_ms = float(start_event.elapsed_time(end_event))

        visible_gaussians = _visible_count(info)
        if render_settings.render_mode in {"D", "ED"}:
            depth = image[..., 0:1]
        elif image.shape[-1] > 3:
            depth = image[..., 3:4]
        else:
            depth = None
        return RenderResult(
            image=image,
            alpha=alpha,
            depth=depth,
            render_time_ms=render_time_ms,
            visible_gaussians=visible_gaussians,
            info=info,
            gpu_render_time_ms=gpu_time_ms,
        )


def _visible_count(info: dict) -> float:
    radii = info.get("radii")
    if radii is None:
        return 0.0 if not info else float("nan")
    with torch.no_grad():
        visible = radii[0] if radii.ndim >= 2 else radii
        mask = visible > 0
        if mask.ndim > 1:
            mask = mask.all(dim=-1)
        return float(mask.sum().item())
