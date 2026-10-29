"""Production ALoD runtime data contracts.

These types are renderer-oriented and deliberately independent from
``alod_reference``.  The stable identity is always ``global_node_id``, the
serialized row index in the source ``.dhier`` asset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


AlodSelectionMode = Literal["finest", "lod"]
AlodMaterializationFilter = Literal[
    "none", "conservative_pinhole", "conservative_pinhole_gpu"
]
AlodHostResidencyMode = Literal[
    "host_nocache",
    "host_previous_reuse",
    "host_viewer_previous_reuse",
    "host_bounded_cache",
]


@dataclass(frozen=True)
class AlodRuntimeConfig:
    root_global_node_id: int
    skybox_count: int
    spt_root_volume: float
    min_spt_size: int
    target_granularity: float
    distance_multiplier: float = 1.0
    selection_mode: AlodSelectionMode = "lod"
    frustum_enabled: bool = True
    materialization_filter: AlodMaterializationFilter = "none"

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
        if self.selection_mode not in {"finest", "lod"}:
            raise ValueError("selection_mode must be 'finest' or 'lod'")
        if self.materialization_filter not in {
            "none",
            "conservative_pinhole",
            "conservative_pinhole_gpu",
        }:
            raise ValueError(
                "materialization_filter must be 'none', 'conservative_pinhole', "
                "or 'conservative_pinhole_gpu'"
            )


@dataclass(frozen=True)
class AlodHostResidencyConfig:
    mode: AlodHostResidencyMode
    cache_capacity_rows: int | None = None
    activation_chunk_rows: int = 131_072
    pin_memory: bool = False
    validate_mappings: bool = True
    validate_properties: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {
            "host_nocache",
            "host_previous_reuse",
            "host_viewer_previous_reuse",
            "host_bounded_cache",
        }:
            raise ValueError(f"unsupported ALoD host residency mode {self.mode!r}")
        if self.activation_chunk_rows <= 0:
            raise ValueError("activation_chunk_rows must be positive")
        if self.mode == "host_bounded_cache":
            if self.cache_capacity_rows is None or self.cache_capacity_rows <= 0:
                raise ValueError("bounded-cache mode requires a positive cache_capacity_rows")
        elif self.cache_capacity_rows is not None:
            raise ValueError("cache_capacity_rows is only valid for bounded-cache mode")


@dataclass(frozen=True)
class AlodHierarchySource:
    positions: np.ndarray
    log_scales: np.ndarray
    nodes: np.ndarray

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
class AlodHspt:
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

    @property
    def nbytes(self) -> int:
        return int(
            sum(
                value.nbytes
                for value in (
                    self.upper_tree_nodes,
                    self.upper_to_global_node_id,
                    self.upper_tree_xyz,
                    self.upper_tree_log_scaling,
                    self.upper_min_distance_squared,
                    self.spt_starts,
                    self.spt_min,
                    self.spt_max,
                    self.spt_to_global_node_id,
                    self.spt_root_global_node_ids,
                    self.spt_sizes,
                )
            )
        )


@dataclass(frozen=True)
class AlodSelectionCamera:
    stable_id: str
    camera_center: np.ndarray
    full_projection_transform: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.camera_center)
        transform = np.asarray(self.full_projection_transform)
        if center.dtype != np.dtype("float32") or center.shape != (3,):
            raise ValueError("camera_center must be float32[3]")
        if transform.dtype != np.dtype("float32") or transform.shape != (4, 4):
            raise ValueError("full_projection_transform must be float32[4,4]")


@dataclass(frozen=True)
class AlodSptCut:
    spt_id: int
    root_global_node_id: int
    distance: np.float32
    interval_size: int
    emitted_global_node_ids: np.ndarray


@dataclass(frozen=True)
class AlodUpperSelection:
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
class AlodSelectionTimings:
    camera_preparation_seconds: float = 0.0
    upper_traversal_seconds: float = 0.0
    spt_cut_seconds: float = 0.0
    total_selection_seconds: float = 0.0


@dataclass(frozen=True)
class AlodSelection:
    camera_stable_id: str
    mode: AlodSelectionMode
    frustum_enabled: bool
    upper: AlodUpperSelection | None
    per_spt: tuple[AlodSptCut, ...]
    selected_spt_global_node_ids: np.ndarray
    selected_hierarchy_global_node_ids: np.ndarray
    raster_global_node_ids: np.ndarray
    timings: AlodSelectionTimings
