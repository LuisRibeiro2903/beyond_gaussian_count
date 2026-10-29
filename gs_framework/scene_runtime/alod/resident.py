"""Fully resident render-property storage for production ALoD scenes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from alod_dhier import DhierAsset
from scene_runtime.types import ActiveSplats


@dataclass(frozen=True)
class AlodResidentProperties:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    global_node_ids: torch.Tensor
    serialized_cpu_bytes: int
    resident_property_bytes: int
    global_id_bytes: int

    @property
    def resident_gpu_bytes(self) -> int:
        return self.resident_property_bytes + self.global_id_bytes

    @classmethod
    def from_dhier(cls, asset: DhierAsset, device: torch.device) -> "AlodResidentProperties":
        means = torch.from_numpy(np.array(asset.positions, dtype=np.float32, copy=True)).to(device)
        raw_quats = torch.from_numpy(
            np.array(asset.rotations, dtype=np.float32, copy=True)
        ).to(device)
        log_scales = torch.from_numpy(
            np.array(asset.log_scales, dtype=np.float32, copy=True)
        ).to(device)
        serialized_opacity = torch.from_numpy(
            np.array(asset.activated_opacities, dtype=np.float32, copy=True)
        )
        colors = torch.from_numpy(
            np.array(asset.sh_coefficients, dtype=np.float32, copy=True)
        ).to(device)

        # Match GaussianModel.create_from_hier followed by compact_gaussians and
        # its render-ready views.  The pinned evaluator computes opacity logits
        # on CPU, stores them there, then transfers selected logits and applies
        # sigmoid on CUDA.  This device boundary is numerically observable.
        serialized_opacity.clamp_(0.0, 0.99999)
        opacity_logits = torch.log(serialized_opacity / (1.0 - serialized_opacity))
        quats = F.normalize(raw_quats, p=2, dim=-1).contiguous()
        scales = torch.exp(log_scales).contiguous()
        opacities = torch.sigmoid(opacity_logits.to(device)).reshape(-1).contiguous()
        means = means.contiguous()
        colors = colors.contiguous()
        global_ids = torch.arange(
            asset.layout.gaussian_count, dtype=torch.int64, device=device
        )
        property_tensors = (means, quats, scales, opacities, colors)
        property_bytes = sum(t.numel() * t.element_size() for t in property_tensors)
        return cls(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            global_node_ids=global_ids,
            serialized_cpu_bytes=int(asset.layout.total_attribute_bytes),
            resident_property_bytes=int(property_bytes),
            global_id_bytes=int(global_ids.numel() * global_ids.element_size()),
        )

    @property
    def row_count(self) -> int:
        return int(self.means.shape[0])

    def gather(self, global_node_ids: torch.Tensor) -> ActiveSplats:
        ids = global_node_ids.to(device=self.means.device, dtype=torch.int64)
        if ids.ndim != 1:
            raise ValueError("global_node_ids must be one-dimensional")
        if ids.numel():
            minimum = int(ids.min().item())
            maximum = int(ids.max().item())
            if minimum < 0 or maximum >= self.row_count:
                raise IndexError(
                    f"global_node_ids outside [0,{self.row_count}): {minimum}..{maximum}"
                )
        return ActiveSplats(
            means=self.means.index_select(0, ids),
            quats=self.quats.index_select(0, ids),
            scales=self.scales.index_select(0, ids),
            opacities=self.opacities.index_select(0, ids),
            colors=self.colors.index_select(0, ids),
        )
