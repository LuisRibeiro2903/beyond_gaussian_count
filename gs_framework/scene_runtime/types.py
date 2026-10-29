from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .gsplat_compat import CameraModel, RasterizeMode


@dataclass
class FrameCamera:
    c2w: Tensor
    K: Tensor
    width: int
    height: int
    cull_c2w: Optional[Tensor] = None
    cull_K: Optional[Tensor] = None
    # Optional exact source-camera values.  Existing scene paths leave these
    # unset and retain the inverse(c2w) renderer/culling behavior.
    w2c: Optional[Tensor] = None
    selection_camera_center: Optional[Tensor] = None
    selection_full_projection_transform: Optional[Tensor] = None
    camera_stable_id: str = ""

    def to(self, device: torch.device) -> "FrameCamera":
        return FrameCamera(
            c2w=self.c2w.to(device=device, dtype=torch.float32),
            K=self.K.to(device=device, dtype=torch.float32),
            width=int(self.width),
            height=int(self.height),
            cull_c2w=(
                self.cull_c2w.to(device=device, dtype=torch.float32)
                if self.cull_c2w is not None
                else None
            ),
            cull_K=(
                self.cull_K.to(device=device, dtype=torch.float32)
                if self.cull_K is not None
                else None
            ),
            w2c=(
                self.w2c.to(device=device, dtype=torch.float32)
                if self.w2c is not None
                else None
            ),
            selection_camera_center=(
                self.selection_camera_center.to(device=device, dtype=torch.float32)
                if self.selection_camera_center is not None
                else None
            ),
            selection_full_projection_transform=(
                self.selection_full_projection_transform.to(
                    device=device, dtype=torch.float32
                )
                if self.selection_full_projection_transform is not None
                else None
            ),
            camera_stable_id=str(self.camera_stable_id),
        )

    @property
    def cull_camera(self) -> Tuple[Tensor, Tensor]:
        return (
            self.cull_c2w if self.cull_c2w is not None else self.c2w,
            self.cull_K if self.cull_K is not None else self.K,
        )


@dataclass
class RenderSettings:
    width: int
    height: int
    near_plane: float
    far_plane: float
    render_mode: str = "RGB"
    sh_degree: Optional[int] = None
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    packed: bool = False
    rasterize_mode: RasterizeMode = "classic"
    camera_model: CameraModel = "pinhole"
    radius_clip: float = 0.0
    eps2d: float = 0.3
    with_ut: bool = False
    with_eval3d: bool = False
    distributed: bool = False


@dataclass
class SceneFrameMetrics:
    active_blocks: float = float("nan")
    active_block_ids: str = ""
    active_gaussians: float = float("nan")
    active_gaussian_fraction: float = float("nan")
    total_blocks: float = float("nan")
    resident_blocks: float = float("nan")
    resident_gaussians: float = float("nan")
    resident_gaussian_fraction: float = float("nan")
    block_cull_ms: float = float("nan")
    block_concat_ms: float = float("nan")
    gpu_upload_ms: float = float("nan")
    cache_hits: float = float("nan")
    cache_misses: float = float("nan")
    cache_hit_rate: float = float("nan")
    newly_uploaded_blocks: float = float("nan")
    newly_uploaded_gaussians: float = float("nan")
    evicted_blocks: float = float("nan")
    evicted_gaussians: float = float("nan")
    gpu_resident_bytes: float = float("nan")
    gpu_resident_mb: float = float("nan")
    cpu_resident_bytes: float = float("nan")
    cpu_resident_mb: float = float("nan")
    peak_vram: float = float("nan")
    peak_vram_mb: float = float("nan")
    gpu_budget_exceeded: float = float("nan")
    active_set_exceeds_budget: float = float("nan")
    total_cell_count: float = float("nan")
    active_cell_count: float = float("nan")
    selected_cell_count: float = float("nan")
    selected_cell_ids: str = ""
    selected_gaussian_count: float = float("nan")
    selected_lod_by_cell: Optional[Dict[int, int]] = None
    selected_level_label_by_cell: Optional[Dict[int, str]] = None
    selected_gaussian_count_by_cell: Optional[Dict[int, int]] = None
    selected_cell_count_by_level: Optional[List[int]] = None
    selected_gaussian_count_by_level: Optional[List[int]] = None
    visibility_selection_ms: float = float("nan")
    lod_selection_ms: float = float("nan")
    selected_cell_assembly_ms: float = float("nan")
    nearest_cell_id: float = float("nan")
    nearest_cell_fallback_changed_mask: Optional[bool] = None
    per_cell_gaussian_counts: Optional[List[int]] = None
    per_cell_bounds: Optional[List[Dict[str, Tuple[float, float, float]]]] = None
    per_cell_distances: Optional[List[float]] = None
    cell_construction_ms: float = float("nan")
    frame_preparation_ms: float = float("nan")
    citygaussian_no_visibility_selection: Optional[bool] = None
    citygaussian_visibility_selection_used: Optional[bool] = None
    citygaussian_no_adaptive_lod: Optional[bool] = None
    citygaussian_adaptive_lod_used: Optional[bool] = None
    citygaussian_lod_policy: Optional[str] = None
    citygaussian_visibility_bounds_level: Optional[str] = None
    citygaussian_no_vq_reconstruction: Optional[bool] = None
    citygaussian_no_streaming: Optional[bool] = None
    alod_selection_mode: Optional[str] = None
    alod_frustum_enabled: Optional[bool] = None
    alod_no_streaming: Optional[bool] = None
    alod_hspt_build_ms: float = float("nan")
    alod_camera_preparation_ms: float = float("nan")
    alod_upper_traversal_ms: float = float("nan")
    alod_spt_cut_ms: float = float("nan")
    alod_selected_index_ms: float = float("nan")
    alod_selected_gather_ms: float = float("nan")
    alod_pre_raster_assembly_ms: float = float("nan")
    alod_total_selection_ms: float = float("nan")
    alod_total_asset_rows: float = float("nan")
    alod_skybox_rows: float = float("nan")
    alod_hierarchy_rows: float = float("nan")
    alod_upper_rows: float = float("nan")
    alod_candidate_spt_count: float = float("nan")
    alod_selected_hierarchy_rows: float = float("nan")
    alod_semantic_selected_rows: float = float("nan")
    alod_semantic_skybox_rows: float = float("nan")
    alod_unique_semantic_rows: float = float("nan")
    alod_duplicate_semantic_occurrences: float = float("nan")
    alod_materialization_rows: float = float("nan")
    alod_unique_materialization_rows: float = float("nan")
    alod_materialization_hierarchy_rows: float = float("nan")
    alod_materialization_skybox_rows: float = float("nan")
    alod_materialization_rejected_rows: float = float("nan")
    alod_materialization_rejected_depth_rows: float = float("nan")
    alod_materialization_rejected_screen_rows: float = float("nan")
    alod_materialization_filter_ms: float = float("nan")
    alod_materialization_id_preparation_ms: float = float("nan")
    alod_materialization_selected_id_h2d_ms: float = float("nan")
    alod_materialization_metadata_gather_ms: float = float("nan")
    alod_materialization_projection_ms: float = float("nan")
    alod_materialization_compaction_ms: float = float("nan")
    alod_materialization_retained_index_d2h_ms: float = float("nan")
    alod_materialization_id_production_ms: float = float("nan")
    alod_materialization_unique_extraction_ms: float = float("nan")
    alod_materialization_occurrence_propagation_ms: float = float("nan")
    alod_materialization_selected_id_bytes: float = float("nan")
    alod_materialization_retained_index_bytes: float = float("nan")
    alod_compact_culling_cpu_bytes: float = float("nan")
    alod_compact_culling_incremental_cpu_bytes: float = float("nan")
    alod_compact_culling_gpu_bytes: float = float("nan")
    alod_total_raster_rows: float = float("nan")
    alod_assembled_renderer_occurrence_rows: float = float("nan")
    alod_occurrence_reindex_ms: float = float("nan")
    alod_hierarchy_metadata_cpu_bytes: float = float("nan")
    alod_hierarchy_metadata_gpu_bytes: float = float("nan")
    alod_gaussian_properties_cpu_bytes: float = float("nan")
    alod_gaussian_properties_gpu_bytes: float = float("nan")
    alod_selected_index_bytes: float = float("nan")
    alod_residency_mode: Optional[str] = None
    alod_host_property_store_bytes: float = float("nan")
    alod_host_pinned_bytes: float = float("nan")
    alod_property_bytes_per_row: float = float("nan")
    alod_persistent_skybox_gpu_bytes: float = float("nan")
    alod_dynamic_cache_capacity_rows: float = float("nan")
    alod_dynamic_cache_gpu_bytes: float = float("nan")
    alod_residency_metadata_cpu_bytes: float = float("nan")
    alod_residency_metadata_gpu_bytes: float = float("nan")
    alod_requested_rows: float = float("nan")
    alod_cache_hit_count: float = float("nan")
    alod_cache_miss_count: float = float("nan")
    alod_cache_eviction_count: float = float("nan")
    alod_cache_occupancy_rows: float = float("nan")
    alod_retained_rows: float = float("nan")
    alod_new_rows: float = float("nan")
    alod_discarded_rows: float = float("nan")
    alod_retention_ratio: float = float("nan")
    alod_selected_set_churn: float = float("nan")
    alod_h2d_rows: float = float("nan")
    alod_h2d_bytes: float = float("nan")
    alod_residency_lookup_ms: float = float("nan")
    alod_host_gather_ms: float = float("nan")
    alod_h2d_transfer_ms: float = float("nan")
    alod_gpu_slot_index_ms: float = float("nan")
    alod_selected_gpu_gather_ms: float = float("nan")
    alod_frame_local_property_bytes: float = float("nan")
    alod_frame_local_index_bytes: float = float("nan")
    alod_cache_management_ms: float = float("nan")
    alod_rows_loaded: float = float("nan")
    alod_cache_occupancy_ratio: float = float("nan")
    alod_previous_working_set_rows: float = float("nan")
    alod_previous_intersection_rows: float = float("nan")


@dataclass
class ActiveSplats:
    means: Tensor
    quats: Tensor
    scales: Tensor
    opacities: Tensor
    colors: Tensor
    global_node_ids: Optional[Tensor] = None
    scene_metrics: SceneFrameMetrics = field(default_factory=SceneFrameMetrics)

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])


@dataclass
class RenderResult:
    image: Tensor
    alpha: Optional[Tensor]
    depth: Optional[Tensor]
    render_time_ms: float
    visible_gaussians: float
    info: Dict[str, Tensor]
    gpu_render_time_ms: float = float("nan")


@dataclass
class SceneStats:
    scene_type: str
    total_gaussians: int
    sh_degree: Optional[int]
    paths: List[Path]
    scene_aabb_min: Optional[Tuple[float, float, float]] = None
    scene_aabb_max: Optional[Tuple[float, float, float]] = None
    view_aabb_min: Optional[Tuple[float, float, float]] = None
    view_aabb_max: Optional[Tuple[float, float, float]] = None
    bounds_policy: Optional[str] = None
    bounds_finite_gaussian_count: Optional[int] = None
    bounds_nonfinite_gaussian_count: Optional[int] = None
    total_blocks: float = float("nan")
    steps: List[int] = field(default_factory=list)
    citygaussian_lod_source_level: Optional[str] = None
    citygaussian_block_dim: Optional[Tuple[int, int, int]] = None
    citygaussian_aabb: Optional[Tuple[float, float, float, float, float, float]] = None
    citygaussian_cell_count: Optional[int] = None
    citygaussian_per_cell_counts: Optional[List[int]] = None
    citygaussian_cell_construction_ms: Optional[float] = None
    citygaussian_lod_policy: Optional[str] = None
    citygaussian_lod_level_labels: Optional[List[str]] = None
    citygaussian_lod_level_semantics: Optional[List[str]] = None
    citygaussian_lod_level_gaussian_counts: Optional[List[int]] = None
    citygaussian_lod_level_paths: Optional[List[Path]] = None
    citygaussian_visibility_bounds_level: Optional[str] = None
    alod_root_global_node_id: Optional[int] = None
    alod_skybox_count: Optional[int] = None
    alod_hierarchy_row_count: Optional[int] = None
    alod_upper_row_count: Optional[int] = None
    alod_spt_count: Optional[int] = None
    alod_flattened_spt_row_count: Optional[int] = None
    alod_selection_mode: Optional[str] = None
    alod_frustum_enabled: Optional[bool] = None
    alod_no_streaming: Optional[bool] = None
    alod_hspt_build_ms: Optional[float] = None
    alod_hierarchy_metadata_cpu_bytes: Optional[int] = None
    alod_hierarchy_metadata_gpu_bytes: Optional[int] = None
    alod_materialization_filter: Optional[str] = None
    alod_compact_culling_cpu_bytes: Optional[int] = None
    alod_compact_culling_incremental_cpu_bytes: Optional[int] = None
    alod_compact_culling_gpu_bytes: Optional[int] = None
    alod_gaussian_properties_cpu_bytes: Optional[int] = None
    alod_gaussian_properties_gpu_bytes: Optional[int] = None
    alod_residency_mode: Optional[str] = None
    alod_host_property_store_bytes: Optional[int] = None
    alod_host_pinned_bytes: Optional[int] = None
    alod_property_bytes_per_row: Optional[int] = None
    alod_persistent_skybox_gpu_bytes: Optional[int] = None
    alod_dynamic_cache_capacity_rows: Optional[int] = None
    alod_dynamic_cache_gpu_bytes: Optional[int] = None
    alod_residency_metadata_cpu_bytes: Optional[int] = None
    alod_residency_metadata_gpu_bytes: Optional[int] = None
    alod_host_store_construction_ms: Optional[float] = None
    alod_skybox_transfer_ms: Optional[float] = None
    alod_cache_allocation_ms: Optional[float] = None


@dataclass(frozen=True)
class BlockDebugInfo:
    block_id: int
    grid_index: Tuple[int, int]
    gaussian_count: int
    content_aabb_min: Tuple[float, float, float]
    content_aabb_max: Tuple[float, float, float]
    cull_aabb_min: Tuple[float, float, float]
    cull_aabb_max: Tuple[float, float, float]
    active: bool
    resident: bool
    empty: bool
    logical_bounds_min: Optional[Tuple[float, float, float]] = None
    logical_bounds_max: Optional[Tuple[float, float, float]] = None
    raw_gaussian_bounds_min: Optional[Tuple[float, float, float]] = None
    raw_gaussian_bounds_max: Optional[Tuple[float, float, float]] = None
    filtered_render_bounds_min: Optional[Tuple[float, float, float]] = None
    filtered_render_bounds_max: Optional[Tuple[float, float, float]] = None
    raw_gaussian_count: Optional[int] = None
    bounds_valid_gaussian_count: Optional[int] = None
    raw_to_filtered_aabb_volume_ratio: Optional[float] = None
