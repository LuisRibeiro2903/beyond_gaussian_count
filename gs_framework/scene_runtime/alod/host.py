"""Canonical host-resident render properties for production ALoD scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from alod_dhier import DhierAsset
from scene_runtime.types import ActiveSplats


PROPERTY_NAMES = ("means", "quats", "scales", "opacities", "colors")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _empty_like_rows(store: "AlodHostProperties", rows: int, device: torch.device) -> ActiveSplats:
    return ActiveSplats(
        means=torch.empty((rows, *store.means.shape[1:]), dtype=store.means.dtype, device=device),
        quats=torch.empty((rows, *store.quats.shape[1:]), dtype=store.quats.dtype, device=device),
        scales=torch.empty((rows, *store.scales.shape[1:]), dtype=store.scales.dtype, device=device),
        opacities=torch.empty((rows,), dtype=store.opacities.dtype, device=device),
        colors=torch.empty((rows, *store.colors.shape[1:]), dtype=store.colors.dtype, device=device),
    )


@dataclass(frozen=True)
class AlodHostProperties:
    """Render-ready CPU properties indexed directly by ``global_node_id``.

    CUDA construction activates bounded chunks on the render device before
    copying them back to CPU.  This preserves the validated resident path's
    device-visible normalization/exp/sigmoid results without requiring the
    complete property table to remain on the GPU.
    """

    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    serialized_cpu_bytes: int
    host_property_bytes: int
    pinned_host_bytes: int
    activation_chunk_rows: int

    @classmethod
    def from_dhier(
        cls,
        asset: DhierAsset,
        activation_device: torch.device,
        *,
        activation_chunk_rows: int = 131_072,
        pin_memory: bool = False,
    ) -> "AlodHostProperties":
        if activation_chunk_rows <= 0:
            raise ValueError("activation_chunk_rows must be positive")
        if pin_memory and activation_device.type != "cuda":
            raise ValueError("pin_memory requires a CUDA activation device")

        count = asset.layout.gaussian_count
        means = torch.from_numpy(np.array(asset.positions, dtype=np.float32, copy=True))
        colors = torch.from_numpy(
            np.array(asset.sh_coefficients, dtype=np.float32, copy=True)
        )
        quats = torch.empty((count, 4), dtype=torch.float32)
        scales = torch.empty((count, 3), dtype=torch.float32)
        opacities = torch.empty((count,), dtype=torch.float32)

        for start in range(0, count, activation_chunk_rows):
            stop = min(start + activation_chunk_rows, count)
            raw_quats = torch.from_numpy(
                np.array(asset.rotations[start:stop], dtype=np.float32, copy=True)
            ).to(activation_device)
            raw_log_scales = torch.from_numpy(
                np.array(asset.log_scales[start:stop], dtype=np.float32, copy=True)
            ).to(activation_device)
            serialized_opacity = torch.from_numpy(
                np.array(
                    asset.activated_opacities[start:stop], dtype=np.float32, copy=True
                )
            )
            serialized_opacity.clamp_(0.0, 0.99999)
            opacity_logits = torch.log(serialized_opacity / (1.0 - serialized_opacity))

            activated_quats = F.normalize(raw_quats, p=2, dim=-1).contiguous()
            activated_scales = torch.exp(raw_log_scales).contiguous()
            activated_opacities = torch.sigmoid(
                opacity_logits.to(activation_device)
            ).reshape(-1).contiguous()
            _synchronize(activation_device)
            quats[start:stop].copy_(activated_quats.cpu())
            scales[start:stop].copy_(activated_scales.cpu())
            opacities[start:stop].copy_(activated_opacities.cpu())

        tensors = [means, quats, scales, opacities, colors]
        if pin_memory:
            tensors = [tensor.pin_memory() for tensor in tensors]
        means, quats, scales, opacities, colors = tensors
        property_bytes = sum(t.numel() * t.element_size() for t in tensors)
        return cls(
            means=means.contiguous(),
            quats=quats.contiguous(),
            scales=scales.contiguous(),
            opacities=opacities.contiguous(),
            colors=colors.contiguous(),
            serialized_cpu_bytes=int(asset.layout.total_attribute_bytes),
            host_property_bytes=int(property_bytes),
            pinned_host_bytes=int(property_bytes if pin_memory else 0),
            activation_chunk_rows=int(activation_chunk_rows),
        )

    @property
    def row_count(self) -> int:
        return int(self.means.shape[0])

    @property
    def property_bytes_per_row(self) -> int:
        if self.row_count == 0:
            return 0
        return self.host_property_bytes // self.row_count

    def validate_global_ids(self, global_node_ids: np.ndarray | torch.Tensor | Iterable[int]) -> np.ndarray:
        if torch.is_tensor(global_node_ids):
            ids = global_node_ids.detach().cpu().numpy()
        else:
            ids = np.asarray(global_node_ids)
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("global_node_ids must be one-dimensional")
        if ids.size:
            minimum = int(ids.min())
            maximum = int(ids.max())
            if minimum < 0 or maximum >= self.row_count:
                raise IndexError(
                    f"global_node_ids outside [0,{self.row_count}): {minimum}..{maximum}"
                )
        return ids

    def gather_cpu(self, global_node_ids: np.ndarray | torch.Tensor | Iterable[int]) -> ActiveSplats:
        ids = self.validate_global_ids(global_node_ids)
        index = torch.from_numpy(ids)
        return ActiveSplats(
            means=self.means.index_select(0, index),
            quats=self.quats.index_select(0, index),
            scales=self.scales.index_select(0, index),
            opacities=self.opacities.index_select(0, index),
            colors=self.colors.index_select(0, index),
        )

    def transfer_rows(
        self,
        global_node_ids: np.ndarray | torch.Tensor | Iterable[int],
        device: torch.device,
    ) -> ActiveSplats:
        cpu = self.gather_cpu(global_node_ids)
        return ActiveSplats(
            means=cpu.means.to(device=device, non_blocking=False),
            quats=cpu.quats.to(device=device, non_blocking=False),
            scales=cpu.scales.to(device=device, non_blocking=False),
            opacities=cpu.opacities.to(device=device, non_blocking=False),
            colors=cpu.colors.to(device=device, non_blocking=False),
        )

    def empty(self, rows: int, device: torch.device) -> ActiveSplats:
        return _empty_like_rows(self, rows, device)


def property_tensor_bytes(splats: ActiveSplats) -> int:
    return int(
        sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                splats.means,
                splats.quats,
                splats.scales,
                splats.opacities,
                splats.colors,
            )
        )
    )


__all__ = [
    "AlodHostProperties",
    "PROPERTY_NAMES",
    "property_tensor_bytes",
]
