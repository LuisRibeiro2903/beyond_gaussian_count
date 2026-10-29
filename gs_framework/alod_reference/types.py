"""Data contracts for the project-owned CPU ALoD reference selector.

The stable identity throughout this package is ``global_node_id``: the row
index in the serialized ``.dhier`` asset.  HSPT-local and SPT-local indices
are derived, transient identities and always carry an explicit mapping back
to global rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from alod_dhier import DhierAsset


@dataclass(frozen=True)
class ReferenceConfig:
    root_global_node_id: int
    skybox_count: int
    spt_root_volume: float
    min_spt_size: int
    target_granularity: float
    distance_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.root_global_node_id < 0:
            raise ValueError("root_global_node_id must be non-negative")
        if self.skybox_count < 0:
            raise ValueError("skybox_count must be non-negative")
        if self.spt_root_volume <= 0:
            raise ValueError("spt_root_volume must be positive")
        if self.min_spt_size < 0:
            raise ValueError("min_spt_size must be non-negative")
        if self.target_granularity <= 0:
            raise ValueError("target_granularity must be positive")
        if self.distance_multiplier <= 0:
            raise ValueError("distance_multiplier must be positive")


@dataclass(frozen=True)
class HierarchyData:
    positions: np.ndarray
    log_scales: np.ndarray
    nodes: np.ndarray

    @classmethod
    def from_dhier(cls, asset: "DhierAsset") -> "HierarchyData":
        return cls(
            positions=np.asarray(asset.positions),
            log_scales=np.asarray(asset.log_scales),
            nodes=np.asarray(asset.nodes),
        )

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions)
        log_scales = np.asarray(self.log_scales)
        nodes = np.asarray(self.nodes)
        if positions.dtype != np.dtype("float32") or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must be float32[N,3]")
        if log_scales.dtype != np.dtype("float32") or log_scales.shape != positions.shape:
            raise ValueError("log_scales must be float32[N,3]")
        if nodes.dtype != np.dtype("int32") or nodes.shape != (positions.shape[0], 6):
            raise ValueError("nodes must be int32[N,6]")

    @property
    def row_count(self) -> int:
        return int(self.nodes.shape[0])


@dataclass(frozen=True)
class ReferenceCamera:
    stable_id: str
    camera_center: np.ndarray
    full_projection_transform: np.ndarray
    registered_image_id: int | None = None

    def __post_init__(self) -> None:
        center = np.asarray(self.camera_center)
        transform = np.asarray(self.full_projection_transform)
        if center.dtype != np.dtype("float32") or center.shape != (3,):
            raise ValueError("camera_center must be float32[3]")
        if transform.dtype != np.dtype("float32") or transform.shape != (4, 4):
            raise ValueError("full_projection_transform must be float32[4,4]")


@dataclass(frozen=True)
class HSPT:
    upper_tree_nodes: np.ndarray
    upper_to_global_node_id: np.ndarray
    upper_tree_xyz: np.ndarray
    upper_tree_log_scaling: np.ndarray
    upper_min_distance_squared: np.ndarray
    spt_starts: np.ndarray
    spt_min: np.ndarray
    spt_max: np.ndarray
    spt_to_global_node_id: np.ndarray
    spt_root_global_node_ids: np.ndarray
    spt_sizes: np.ndarray

    @property
    def upper_row_count(self) -> int:
        return int(self.upper_tree_nodes.shape[0])

    @property
    def spt_count(self) -> int:
        return int(self.spt_sizes.size)

    @property
    def flattened_spt_row_count(self) -> int:
        return int(self.spt_to_global_node_id.size)


@dataclass(frozen=True)
class SPTCutResult:
    spt_id: int
    root_global_node_id: int
    distance: np.float32
    interval_size: int
    emitted_global_node_ids: np.ndarray


@dataclass(frozen=True)
class UpperTraversalResult:
    coarse_cut_upper_local_ids: np.ndarray
    coarse_cut_global_node_ids: np.ndarray
    cut_leaf_upper_local_ids: np.ndarray
    cut_leaf_global_node_ids: np.ndarray
    upper_render_global_node_ids: np.ndarray
    candidate_spt_ids: np.ndarray
    candidate_spt_upper_local_ids: np.ndarray
    candidate_spt_root_global_node_ids: np.ndarray
    candidate_spt_distances: np.ndarray
    frustum_planes: np.ndarray
    frustum_rejected_upper_local_ids: np.ndarray


@dataclass(frozen=True)
class SelectionTimings:
    upper_traversal_seconds: float
    spt_cut_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class SelectionResult:
    camera_stable_id: str
    upper: UpperTraversalResult
    per_spt: tuple[SPTCutResult, ...]
    selected_spt_global_node_ids: np.ndarray
    selected_hierarchy_global_node_ids: np.ndarray
    raster_global_node_ids: np.ndarray
    timings: SelectionTimings
