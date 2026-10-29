#!/usr/bin/env python3
"""Render an analytic scene without external assets and check basic GPU behavior."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gs_framework'))
import torch
from scene_runtime.renderer import GsplatRenderer
from scene_runtime.types import ActiveSplats, FrameCamera, RenderSettings

def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this smoke test; offline evidence tools need no GPU.')
    device = torch.device('cuda:0')
    scene = ActiveSplats(
        means=torch.tensor([[0., 0., 3.]], device=device),
        quats=torch.tensor([[1., 0., 0., 0.]], device=device),
        scales=torch.full((1, 3), .2, device=device),
        opacities=torch.tensor([.9], device=device),
        colors=torch.tensor([[1., .1, .05]], device=device),
    )
    camera = FrameCamera(torch.eye(4, device=device),
                         torch.tensor([[32., 0., 16.], [0., 32., 16.], [0., 0., 1.]], device=device), 32, 32)
    renderer = GsplatRenderer(device)
    for packed in [False, True]:
        settings = RenderSettings(32, 32, .01, 100., sh_degree=None, packed=packed)
        result = renderer.render(scene, camera, settings)
        assert result.image.shape == (32, 32, 3)
        assert torch.isfinite(result.image).all()
        assert result.image[16, 16, 0] > result.image[16, 16, 1] > 0
        empty = ActiveSplats(*(getattr(scene, field)[:0] for field in ['means', 'quats', 'scales', 'opacities', 'colors']))
        background = renderer.render(empty, camera, settings)
        assert torch.count_nonzero(background.image) == 0
    print(json.dumps({'status':'PASS', 'scene':'one analytic red Gaussian and empty background',
                      'packed_modes':[False,True], 'gpu':torch.cuda.get_device_name(0),
                      'torch':torch.__version__, 'external_assets':False}))

if __name__ == '__main__':
    main()
