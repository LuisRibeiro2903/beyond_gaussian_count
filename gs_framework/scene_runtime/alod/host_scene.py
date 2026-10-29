"""Host-resident ALoD scenes with explicit GPU property working sets."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from alod_dhier import load_dhier, validate_numeric, validate_topology
from scene_runtime.scenes import GaussianScene, compute_scene_bounds_from_means
from scene_runtime.types import (
    ActiveSplats,
    FrameCamera,
    RenderSettings,
    SceneFrameMetrics,
    SceneStats,
)

from .camera import selection_camera_from_frame
from .host import AlodHostProperties, property_tensor_bytes
from .hspt import build_production_hspt
from .materialization import (
    AlodCompactCullingMetadata,
    AlodMaterializationSelection,
    select_materialization,
    select_multiset_materialization,
)
from .residency import (
    AlodBoundedCache,
    AlodHostNoCache,
    AlodPreviousWorkingSetReuse,
    AlodViewerPreviousWorkingSetReuse,
    AlodResidencyFrame,
)
from .selection import select_production
from .types import (
    AlodHierarchySource,
    AlodHostResidencyConfig,
    AlodRuntimeConfig,
    AlodSelection,
)
from .viewer_reuse import AlodViewerReuseResult, apply_viewer_previous_reuse


LoadProgressCallback = Callable[[str], None]


def _notify(callback: LoadProgressCallback | None, stage: str) -> None:
    if callback is not None:
        callback(stage)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _concat_splats(prefix: ActiveSplats, hierarchy: ActiveSplats) -> ActiveSplats:
    if prefix.num_gaussians == 0:
        return hierarchy
    return ActiveSplats(
        means=torch.cat((prefix.means, hierarchy.means), dim=0),
        quats=torch.cat((prefix.quats, hierarchy.quats), dim=0),
        scales=torch.cat((prefix.scales, hierarchy.scales), dim=0),
        opacities=torch.cat((prefix.opacities, hierarchy.opacities), dim=0),
        colors=torch.cat((prefix.colors, hierarchy.colors), dim=0),
    )


def _gather_splats(splats: ActiveSplats, rows: torch.Tensor) -> ActiveSplats:
    return ActiveSplats(
        means=splats.means.index_select(0, rows),
        quats=splats.quats.index_select(0, rows),
        scales=splats.scales.index_select(0, rows),
        opacities=splats.opacities.index_select(0, rows),
        colors=splats.colors.index_select(0, rows),
    )


class AlodHostScene(GaussianScene):
    """Production selector plus one of three host-property residency policies."""

    def __init__(
        self,
        path: Path,
        device: torch.device,
        config: AlodRuntimeConfig,
        residency_config: AlodHostResidencyConfig,
        progress_callback: LoadProgressCallback | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.paths = [self.path]
        self.device = device
        self.config = config
        self.residency_config = residency_config

        _notify(progress_callback, "cpu_tensor_construction")
        self.asset = load_dhier(self.path)
        if config.skybox_count >= self.asset.layout.gaussian_count:
            raise ValueError("skybox_count must leave at least one hierarchy row")
        if config.root_global_node_id != config.skybox_count:
            raise ValueError("ALoD host modes require hierarchy root == skybox prefix length")
        validate_numeric(self.asset, profile="strict", raise_on_error=True)
        validate_topology(
            self.asset,
            skybox_count=config.skybox_count,
            profile="strict",
            raise_on_error=True,
        )
        self.hierarchy = AlodHierarchySource(
            positions=self.asset.positions,
            log_scales=self.asset.log_scales,
            nodes=self.asset.nodes,
        )

        _notify(progress_callback, "bounds_or_metadata_computation")
        build_start = time.perf_counter()
        self.hspt = build_production_hspt(self.hierarchy, config)
        self.hspt_build_seconds = time.perf_counter() - build_start
        bounds = compute_scene_bounds_from_means(
            torch.from_numpy(np.array(self.asset.positions, dtype=np.float32, copy=True))
        )
        self.scene_min = bounds["scene_min"]
        self.scene_max = bounds["scene_max"]
        self.view_min = bounds["view_min"]
        self.view_max = bounds["view_max"]
        self.bounds_policy = str(bounds["policy"])
        self.bounds_finite_gaussian_count = int(bounds["finite_count"])
        self.bounds_nonfinite_gaussian_count = int(bounds["nonfinite_count"])

        _notify(progress_callback, "host_property_construction")
        host_store_start = time.perf_counter()
        self.host = AlodHostProperties.from_dhier(
            self.asset,
            device,
            activation_chunk_rows=residency_config.activation_chunk_rows,
            pin_memory=residency_config.pin_memory,
        )
        self.host_store_construction_seconds = time.perf_counter() - host_store_start
        self.compact_culling = (
            AlodCompactCullingMetadata.from_log_scales(
                self.asset.positions, self.asset.log_scales
            )
            if config.materialization_filter != "none"
            else None
        )
        self.device_culling = (
            self.compact_culling.to_device(device)
            if config.materialization_filter == "conservative_pinhole_gpu"
            and self.compact_culling is not None
            else None
        )
        _notify(progress_callback, "gpu_transfer")
        skybox_ids = np.arange(config.skybox_count, dtype=np.int64)
        skybox_start = time.perf_counter()
        self.skybox = self.host.transfer_rows(skybox_ids, device)
        _sync(device)
        self.skybox_transfer_seconds = time.perf_counter() - skybox_start
        self.skybox_gpu_bytes = property_tensor_bytes(self.skybox)

        cache_allocation_start = time.perf_counter()
        common = dict(
            host=self.host,
            device=device,
            hierarchy_start=config.skybox_count,
            validate_mappings=residency_config.validate_mappings,
        )
        if residency_config.mode == "host_nocache":
            self.residency = AlodHostNoCache(**common)
        elif residency_config.mode == "host_previous_reuse":
            self.residency = AlodPreviousWorkingSetReuse(**common)
        elif residency_config.mode == "host_viewer_previous_reuse":
            self.residency = AlodViewerPreviousWorkingSetReuse(**common)
        else:
            assert residency_config.cache_capacity_rows is not None
            self.residency = AlodBoundedCache(
                **common, capacity_rows=residency_config.cache_capacity_rows
            )
        _sync(device)
        self.cache_allocation_seconds = time.perf_counter() - cache_allocation_start
        self.last_selection: AlodSelection | None = None
        self.last_viewer_reuse: AlodViewerReuseResult | None = None
        self.last_materialization: AlodMaterializationSelection | None = None
        self.last_residency_frame: AlodResidencyFrame | None = None
        self.last_active_splats: ActiveSplats | None = None
        self.last_metrics = self._base_metrics()
        _notify(progress_callback, "gpu_scene_construction")

    @property
    def scene_type(self) -> str:
        return f"alod_{self.residency_config.mode}"

    @property
    def hierarchy_row_count(self) -> int:
        return self.asset.layout.gaussian_count - self.config.skybox_count

    @property
    def hierarchy_metadata_cpu_bytes(self) -> int:
        return int(self.asset.layout.total_metadata_bytes + self.hspt.nbytes)

    @property
    def compact_culling_cpu_bytes(self) -> int:
        return self.compact_culling.logical_bytes if self.compact_culling is not None else 0

    @property
    def compact_culling_incremental_cpu_bytes(self) -> int:
        return self.compact_culling.incremental_bytes if self.compact_culling is not None else 0

    @property
    def compact_culling_gpu_bytes(self) -> int:
        return self.device_culling.logical_bytes if self.device_culling is not None else 0

    @property
    def persistent_dynamic_gpu_bytes(self) -> int:
        if isinstance(self.residency, AlodBoundedCache):
            return property_tensor_bytes(self.residency.cache)
        if isinstance(self.residency, AlodPreviousWorkingSetReuse):
            if self.residency.previous_splats is not None:
                return property_tensor_bytes(self.residency.previous_splats)
        return 0

    @property
    def residency_metadata_cpu_bytes(self) -> int:
        if isinstance(self.residency, AlodBoundedCache):
            return int(
                self.residency.global_to_slot.nbytes
                + self.residency.slot_to_global.nbytes
                + self.residency.last_used_frame.nbytes
                + self.residency.previous_selected_ids.nbytes
            )
        if isinstance(self.residency, AlodPreviousWorkingSetReuse):
            return int(
                self.residency.global_to_previous_row.nbytes
                + self.residency.previous_selected_ids.nbytes
            )
        return int(self.residency.previous_selected_ids.nbytes)

    def reset_residency(self) -> None:
        self.residency.reset()
        self.last_residency_frame = None

    def _base_metrics(self) -> SceneFrameMetrics:
        total = self.asset.layout.gaussian_count
        dynamic_bytes = self.persistent_dynamic_gpu_bytes
        gpu_property_bytes = self.skybox_gpu_bytes + dynamic_bytes
        total_gpu_bytes = gpu_property_bytes + self.compact_culling_gpu_bytes
        cpu_bytes = (
            self.host.host_property_bytes
            + self.hierarchy_metadata_cpu_bytes
            + self.residency_metadata_cpu_bytes
            + self.compact_culling_incremental_cpu_bytes
        )
        resident_rows = self.config.skybox_count
        if isinstance(self.residency, AlodBoundedCache):
            resident_rows += int(np.count_nonzero(self.residency.slot_to_global >= 0))
        elif isinstance(self.residency, AlodPreviousWorkingSetReuse):
            resident_rows += int(self.residency.previous_selected_ids.size)
        return SceneFrameMetrics(
            active_blocks=1,
            total_blocks=1,
            resident_blocks=1,
            resident_gaussians=resident_rows,
            resident_gaussian_fraction=resident_rows / total,
            gpu_resident_bytes=total_gpu_bytes,
            gpu_resident_mb=total_gpu_bytes / (1024.0**2),
            cpu_resident_bytes=cpu_bytes,
            cpu_resident_mb=cpu_bytes / (1024.0**2),
            alod_selection_mode=self.config.selection_mode,
            alod_frustum_enabled=self.config.frustum_enabled,
            alod_no_streaming=False,
            alod_hspt_build_ms=self.hspt_build_seconds * 1000.0,
            alod_total_asset_rows=total,
            alod_skybox_rows=self.config.skybox_count,
            alod_hierarchy_rows=self.hierarchy_row_count,
            alod_upper_rows=self.hspt.upper_row_count,
            alod_hierarchy_metadata_cpu_bytes=self.hierarchy_metadata_cpu_bytes,
            alod_hierarchy_metadata_gpu_bytes=0,
            alod_gaussian_properties_cpu_bytes=self.host.host_property_bytes,
            alod_gaussian_properties_gpu_bytes=gpu_property_bytes,
            alod_residency_mode=self.residency_config.mode,
            alod_host_property_store_bytes=self.host.host_property_bytes,
            alod_host_pinned_bytes=self.host.pinned_host_bytes,
            alod_property_bytes_per_row=self.host.property_bytes_per_row,
            alod_compact_culling_cpu_bytes=self.compact_culling_cpu_bytes,
            alod_compact_culling_incremental_cpu_bytes=(
                self.compact_culling_incremental_cpu_bytes
            ),
            alod_compact_culling_gpu_bytes=self.compact_culling_gpu_bytes,
            alod_persistent_skybox_gpu_bytes=self.skybox_gpu_bytes,
            alod_dynamic_cache_capacity_rows=(
                self.residency.capacity_rows
                if isinstance(self.residency, AlodBoundedCache)
                else float("nan")
            ),
            alod_dynamic_cache_gpu_bytes=dynamic_bytes,
            alod_residency_metadata_cpu_bytes=self.residency_metadata_cpu_bytes,
            alod_residency_metadata_gpu_bytes=0,
        )

    def _validate_property_rows(
        self, selected_ids: np.ndarray, actual: ActiveSplats
    ) -> None:
        expected = self.host.gather_cpu(selected_ids)
        for name in ("means", "quats", "scales", "opacities", "colors"):
            actual_cpu = getattr(actual, name).detach().cpu()
            if not torch.equal(actual_cpu, getattr(expected, name)):
                raise RuntimeError(f"host/cache property parity failed for {name}")

    def prepare_frame(
        self, camera: FrameCamera, render_settings: RenderSettings
    ) -> ActiveSplats:
        frame_start = time.perf_counter()
        camera_start = frame_start
        selection_camera = selection_camera_from_frame(
            camera,
            render_settings,
            stable_id=camera.camera_stable_id or "runtime-camera",
        )
        camera_seconds = time.perf_counter() - camera_start
        source_selection = select_production(
            self.hierarchy, self.hspt, selection_camera, self.config
        )
        viewer_reuse = None
        if isinstance(self.residency, AlodViewerPreviousWorkingSetReuse):
            viewer_reuse = apply_viewer_previous_reuse(
                source_selection, self.last_selection
            )
            selection = viewer_reuse.selection
        else:
            selection = source_selection
        materialization_selector = (
            select_materialization
            if self.config.materialization_filter == "none"
            else select_multiset_materialization
        )
        materialization = materialization_selector(
            self.config.materialization_filter,
            self.compact_culling,
            selection.raster_global_node_ids,
            camera,
            render_settings,
            device_metadata=self.device_culling,
        )
        materialization_ids = materialization.materialization_global_node_ids
        unique_materialization_ids = (
            materialization.unique_materialization_global_node_ids
            if materialization.unique_materialization_global_node_ids is not None
            else materialization_ids
        )
        skybox_ids = materialization_ids[
            materialization_ids < self.config.skybox_count
        ]
        hierarchy_occurrence_ids = materialization_ids[
            materialization_ids >= self.config.skybox_count
        ]
        unique_skybox_ids = unique_materialization_ids[
            unique_materialization_ids < self.config.skybox_count
        ]
        unique_hierarchy_ids = unique_materialization_ids[
            unique_materialization_ids >= self.config.skybox_count
        ]

        if isinstance(self.residency, AlodViewerPreviousWorkingSetReuse):
            assert viewer_reuse is not None
            residency_frame = self.residency.prepare(
                unique_hierarchy_ids, viewer_reuse.reusable_global_node_ids
            )
        else:
            residency_frame = self.residency.prepare(unique_hierarchy_ids)
        if not np.array_equal(
            residency_frame.ordered_global_node_ids,
            unique_hierarchy_ids,
        ):
            raise RuntimeError("residency changed materialization global-node order")
        if self.residency_config.validate_properties:
            self._validate_property_rows(
                unique_hierarchy_ids,
                residency_frame.hierarchy_splats,
            )

        assembly_start = time.perf_counter()
        skybox_is_full = np.array_equal(
            unique_skybox_ids, np.arange(self.config.skybox_count, dtype=np.int64)
        )
        if skybox_is_full:
            active_skybox = self.skybox
            filtered_skybox_bytes = 0
        else:
            skybox_index = torch.from_numpy(unique_skybox_ids).to(
                device=self.device, dtype=torch.int64
            )
            active_skybox = _gather_splats(self.skybox, skybox_index)
            filtered_skybox_bytes = property_tensor_bytes(active_skybox)
        unique_active = _concat_splats(
            active_skybox, residency_frame.hierarchy_splats
        )
        occurrence_reindex_start = time.perf_counter()
        occurrence_mapping = materialization.materialization_to_unique_indices
        if occurrence_mapping is not None and materialization.materialization_rows != (
            materialization.unique_materialization_rows
        ):
            occurrence_mapping_gpu = torch.from_numpy(occurrence_mapping).to(
                device=self.device, dtype=torch.int64
            )
            active = _gather_splats(unique_active, occurrence_mapping_gpu)
            _sync(self.device)
        else:
            active = unique_active
        occurrence_reindex_seconds = time.perf_counter() - occurrence_reindex_start
        active_global_ids = materialization.materialization_global_node_ids_gpu
        if active_global_ids is None:
            active_global_ids = torch.from_numpy(
                materialization.raster_global_node_ids
            ).to(device=self.device, dtype=torch.int64)
        active.global_node_ids = active_global_ids
        _sync(self.device)
        assembly_seconds = time.perf_counter() - assembly_start
        if active.num_gaussians != materialization.materialization_rows:
            raise RuntimeError("pre-raster row count differs from materialization IDs")

        metrics = self._base_metrics()
        selected_hierarchy_count = int(selection.selected_hierarchy_global_node_ids.size)
        semantic_count = int(selection.raster_global_node_ids.size)
        raster_count = materialization.materialization_rows
        candidate_count = (
            int(selection.upper.candidate_spt_ids.size)
            if selection.upper is not None
            else 0
        )
        metrics.active_gaussians = raster_count
        metrics.active_gaussian_fraction = raster_count / self.asset.layout.gaussian_count
        metrics.selected_gaussian_count = semantic_count
        metrics.visibility_selection_ms = selection.timings.upper_traversal_seconds * 1000.0
        metrics.lod_selection_ms = selection.timings.spt_cut_seconds * 1000.0
        metrics.selected_cell_assembly_ms = assembly_seconds * 1000.0
        metrics.alod_camera_preparation_ms = camera_seconds * 1000.0
        metrics.alod_upper_traversal_ms = selection.timings.upper_traversal_seconds * 1000.0
        metrics.alod_spt_cut_ms = selection.timings.spt_cut_seconds * 1000.0
        metrics.alod_selected_index_ms = residency_frame.gpu_slot_index_ms
        metrics.alod_selected_gather_ms = residency_frame.selected_gpu_gather_ms
        metrics.alod_pre_raster_assembly_ms = assembly_seconds * 1000.0
        metrics.alod_total_selection_ms = selection.timings.total_selection_seconds * 1000.0
        metrics.alod_candidate_spt_count = candidate_count
        metrics.alod_selected_hierarchy_rows = selected_hierarchy_count
        metrics.alod_semantic_selected_rows = semantic_count
        metrics.alod_semantic_skybox_rows = semantic_count - selected_hierarchy_count
        metrics.alod_unique_semantic_rows = materialization.unique_semantic_rows
        metrics.alod_duplicate_semantic_occurrences = (
            semantic_count - materialization.unique_semantic_rows
        )
        metrics.alod_materialization_rows = raster_count
        metrics.alod_unique_materialization_rows = (
            materialization.unique_materialization_rows
        )
        metrics.alod_materialization_skybox_rows = int(unique_skybox_ids.size)
        metrics.alod_materialization_hierarchy_rows = int(unique_hierarchy_ids.size)
        metrics.alod_materialization_rejected_rows = semantic_count - raster_count
        metrics.alod_materialization_rejected_depth_rows = (
            materialization.rejected_depth_rows
        )
        metrics.alod_materialization_rejected_screen_rows = (
            materialization.rejected_screen_bound_rows
        )
        metrics.alod_materialization_filter_ms = materialization.filter_seconds * 1000.0
        filter_timings = materialization.timings
        metrics.alod_materialization_id_preparation_ms = (
            filter_timings.id_preparation_seconds * 1000.0
        )
        metrics.alod_materialization_selected_id_h2d_ms = (
            filter_timings.selected_id_h2d_seconds * 1000.0
        )
        metrics.alod_materialization_metadata_gather_ms = (
            filter_timings.metadata_gather_seconds * 1000.0
        )
        metrics.alod_materialization_projection_ms = (
            filter_timings.projection_seconds * 1000.0
        )
        metrics.alod_materialization_compaction_ms = (
            filter_timings.compaction_seconds * 1000.0
        )
        metrics.alod_materialization_retained_index_d2h_ms = (
            filter_timings.retained_index_d2h_seconds * 1000.0
        )
        metrics.alod_materialization_id_production_ms = (
            filter_timings.materialization_id_production_seconds * 1000.0
        )
        metrics.alod_materialization_unique_extraction_ms = (
            filter_timings.unique_id_extraction_seconds * 1000.0
        )
        metrics.alod_materialization_occurrence_propagation_ms = (
            filter_timings.occurrence_propagation_seconds * 1000.0
        )
        metrics.alod_materialization_selected_id_bytes = (
            filter_timings.selected_id_bytes
        )
        metrics.alod_materialization_retained_index_bytes = (
            filter_timings.retained_index_bytes
        )
        metrics.alod_total_raster_rows = raster_count
        metrics.alod_assembled_renderer_occurrence_rows = raster_count
        metrics.alod_occurrence_reindex_ms = occurrence_reindex_seconds * 1000.0
        metrics.alod_selected_index_bytes = residency_frame.frame_local_index_bytes
        cache_metrics_apply = self.residency_config.mode != "host_nocache"
        if cache_metrics_apply:
            metrics.cache_hits = residency_frame.cache_hits
            metrics.cache_misses = residency_frame.cache_misses
            metrics.cache_hit_rate = residency_frame.cache_hit_rate
            metrics.evicted_gaussians = residency_frame.evictions
        metrics.newly_uploaded_gaussians = residency_frame.h2d_rows
        metrics.gpu_upload_ms = residency_frame.h2d_transfer_ms
        metrics.alod_requested_rows = residency_frame.requested_rows
        if cache_metrics_apply:
            metrics.alod_cache_hit_count = residency_frame.cache_hits
            metrics.alod_cache_miss_count = residency_frame.cache_misses
            metrics.alod_cache_eviction_count = residency_frame.evictions
            metrics.alod_cache_occupancy_rows = residency_frame.cache_occupancy_rows
        metrics.alod_retained_rows = residency_frame.retained_rows
        metrics.alod_new_rows = residency_frame.new_rows
        metrics.alod_discarded_rows = residency_frame.discarded_rows
        metrics.alod_retention_ratio = residency_frame.retention_ratio
        if self.residency_config.mode != "host_nocache":
            metrics.alod_selected_set_churn = residency_frame.selected_set_churn_rows
        metrics.alod_h2d_rows = residency_frame.h2d_rows
        metrics.alod_h2d_bytes = residency_frame.h2d_bytes
        metrics.alod_residency_lookup_ms = residency_frame.residency_lookup_ms
        metrics.alod_host_gather_ms = residency_frame.host_gather_ms
        metrics.alod_h2d_transfer_ms = residency_frame.h2d_transfer_ms
        if self.residency_config.mode == "host_bounded_cache":
            metrics.alod_gpu_slot_index_ms = residency_frame.gpu_slot_index_ms
        if self.residency_config.mode != "host_nocache":
            metrics.alod_selected_gpu_gather_ms = residency_frame.selected_gpu_gather_ms
        metrics.alod_frame_local_property_bytes = (
            residency_frame.frame_local_property_bytes
            + filtered_skybox_bytes
            + property_tensor_bytes(active)
        )
        metrics.alod_frame_local_index_bytes = residency_frame.frame_local_index_bytes
        if cache_metrics_apply:
            metrics.alod_cache_management_ms = residency_frame.cache_management_ms
        metrics.alod_rows_loaded = residency_frame.loads
        if isinstance(self.residency, AlodBoundedCache):
            metrics.alod_cache_occupancy_ratio = (
                residency_frame.cache_occupancy_rows / self.residency.capacity_rows
            )
        if self.residency_config.mode in {
            "host_previous_reuse",
            "host_viewer_previous_reuse",
        }:
            metrics.alod_previous_working_set_rows = (
                residency_frame.previous_working_set_rows
            )
            metrics.alod_previous_intersection_rows = (
                residency_frame.previous_intersection_rows
            )
        metrics.frame_preparation_ms = (time.perf_counter() - frame_start) * 1000.0
        active.scene_metrics = metrics
        self.last_selection = selection
        self.last_viewer_reuse = viewer_reuse
        self.last_materialization = materialization
        self.last_residency_frame = residency_frame
        self.last_active_splats = active
        self.last_metrics = metrics
        return active

    def get_scene_stats(self) -> SceneStats:
        return SceneStats(
            scene_type=self.scene_type,
            total_gaussians=self.asset.layout.gaussian_count,
            sh_degree=self.asset.layout.sh_degree,
            paths=self.paths,
            scene_aabb_min=self.scene_min,
            scene_aabb_max=self.scene_max,
            view_aabb_min=self.view_min,
            view_aabb_max=self.view_max,
            bounds_policy=self.bounds_policy,
            bounds_finite_gaussian_count=self.bounds_finite_gaussian_count,
            bounds_nonfinite_gaussian_count=self.bounds_nonfinite_gaussian_count,
            total_blocks=1,
            alod_root_global_node_id=self.config.root_global_node_id,
            alod_skybox_count=self.config.skybox_count,
            alod_hierarchy_row_count=self.hierarchy_row_count,
            alod_upper_row_count=self.hspt.upper_row_count,
            alod_spt_count=self.hspt.spt_count,
            alod_flattened_spt_row_count=self.hspt.flattened_spt_row_count,
            alod_selection_mode=self.config.selection_mode,
            alod_frustum_enabled=self.config.frustum_enabled,
            alod_no_streaming=False,
            alod_hspt_build_ms=self.hspt_build_seconds * 1000.0,
            alod_hierarchy_metadata_cpu_bytes=self.hierarchy_metadata_cpu_bytes,
            alod_hierarchy_metadata_gpu_bytes=0,
            alod_gaussian_properties_cpu_bytes=self.host.host_property_bytes,
            alod_gaussian_properties_gpu_bytes=(
                self.skybox_gpu_bytes + self.persistent_dynamic_gpu_bytes
            ),
            alod_residency_mode=self.residency_config.mode,
            alod_host_property_store_bytes=self.host.host_property_bytes,
            alod_host_pinned_bytes=self.host.pinned_host_bytes,
            alod_property_bytes_per_row=self.host.property_bytes_per_row,
            alod_materialization_filter=self.config.materialization_filter,
            alod_compact_culling_cpu_bytes=self.compact_culling_cpu_bytes,
            alod_compact_culling_incremental_cpu_bytes=(
                self.compact_culling_incremental_cpu_bytes
            ),
            alod_compact_culling_gpu_bytes=self.compact_culling_gpu_bytes,
            alod_persistent_skybox_gpu_bytes=self.skybox_gpu_bytes,
            alod_dynamic_cache_capacity_rows=(
                self.residency.capacity_rows
                if isinstance(self.residency, AlodBoundedCache)
                else None
            ),
            alod_dynamic_cache_gpu_bytes=self.persistent_dynamic_gpu_bytes,
            alod_residency_metadata_cpu_bytes=self.residency_metadata_cpu_bytes,
            alod_residency_metadata_gpu_bytes=0,
            alod_host_store_construction_ms=self.host_store_construction_seconds * 1000.0,
            alod_skybox_transfer_ms=self.skybox_transfer_seconds * 1000.0,
            alod_cache_allocation_ms=self.cache_allocation_seconds * 1000.0,
        )

    def get_metrics(self) -> SceneFrameMetrics:
        return self.last_metrics


__all__ = ["AlodHostScene"]
