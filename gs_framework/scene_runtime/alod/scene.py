"""Fully resident ALoD scene integrated with the shared gsplat renderer."""

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
from .hspt import build_production_hspt
from .materialization import (
    AlodCompactCullingMetadata,
    AlodMaterializationSelection,
    select_materialization,
)
from .resident import AlodResidentProperties
from .selection import select_production
from .types import AlodHierarchySource, AlodRuntimeConfig, AlodSelection


LoadProgressCallback = Callable[[str], None]


def _notify(callback: LoadProgressCallback | None, stage: str) -> None:
    if callback is not None:
        callback(stage)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class AlodResidentScene(GaussianScene):
    """Production ALoD scene with all render properties resident on ``device``.

    Hierarchy selection remains CPU-side for this correctness-first baseline.
    No selected property is transferred from host on a per-frame basis.
    """

    def __init__(
        self,
        path: Path,
        device: torch.device,
        config: AlodRuntimeConfig,
        progress_callback: LoadProgressCallback | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.paths = [self.path]
        self.device = device
        self.config = config

        _notify(progress_callback, "cpu_tensor_construction")
        self.asset = load_dhier(self.path)
        if config.skybox_count >= self.asset.layout.gaussian_count:
            raise ValueError("skybox_count must leave at least one hierarchy row")
        if config.root_global_node_id != config.skybox_count:
            raise ValueError(
                "the resident baseline requires hierarchy root == skybox prefix length"
            )
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

        _notify(progress_callback, "bounds_or_metadata_computation")
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

        _notify(progress_callback, "gpu_transfer")
        self.resident = AlodResidentProperties.from_dhier(self.asset, device)
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
        self.last_selection: AlodSelection | None = None
        self.last_materialization: AlodMaterializationSelection | None = None
        self.last_active_splats: ActiveSplats | None = None
        self.last_metrics = self._base_metrics()
        _notify(progress_callback, "gpu_scene_construction")

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

    def _base_metrics(self) -> SceneFrameMetrics:
        total = self.asset.layout.gaussian_count
        return SceneFrameMetrics(
            active_blocks=1,
            total_blocks=1,
            resident_blocks=1,
            resident_gaussians=total,
            resident_gaussian_fraction=1.0,
            gpu_resident_bytes=(
                self.resident.resident_gpu_bytes + self.compact_culling_gpu_bytes
            ),
            gpu_resident_mb=(
                self.resident.resident_gpu_bytes + self.compact_culling_gpu_bytes
            )
            / (1024.0**2),
            cpu_resident_bytes=(
                self.resident.serialized_cpu_bytes
                + self.hierarchy_metadata_cpu_bytes
                + self.compact_culling_incremental_cpu_bytes
            ),
            cpu_resident_mb=(
                self.resident.serialized_cpu_bytes
                + self.hierarchy_metadata_cpu_bytes
                + self.compact_culling_incremental_cpu_bytes
            )
            / (1024.0**2),
            alod_selection_mode=self.config.selection_mode,
            alod_frustum_enabled=self.config.frustum_enabled,
            alod_no_streaming=True,
            alod_hspt_build_ms=self.hspt_build_seconds * 1000.0,
            alod_total_asset_rows=total,
            alod_skybox_rows=self.config.skybox_count,
            alod_hierarchy_rows=self.hierarchy_row_count,
            alod_upper_rows=self.hspt.upper_row_count,
            alod_hierarchy_metadata_cpu_bytes=self.hierarchy_metadata_cpu_bytes,
            alod_hierarchy_metadata_gpu_bytes=self.resident.global_id_bytes,
            alod_gaussian_properties_cpu_bytes=self.resident.serialized_cpu_bytes,
            alod_gaussian_properties_gpu_bytes=self.resident.resident_property_bytes,
            alod_compact_culling_cpu_bytes=self.compact_culling_cpu_bytes,
            alod_compact_culling_incremental_cpu_bytes=(
                self.compact_culling_incremental_cpu_bytes
            ),
            alod_compact_culling_gpu_bytes=self.compact_culling_gpu_bytes,
        )

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

        selection = select_production(
            self.hierarchy, self.hspt, selection_camera, self.config
        )
        materialization = select_materialization(
            self.config.materialization_filter,
            self.compact_culling,
            selection.raster_global_node_ids,
            camera,
            render_settings,
            device_metadata=self.device_culling,
        )

        _sync(self.device)
        index_start = time.perf_counter()
        selected_ids = materialization.materialization_global_node_ids_gpu
        if selected_ids is None:
            selected_ids = torch.from_numpy(materialization.raster_global_node_ids).to(
                device=self.device, dtype=torch.int64
            )
        _sync(self.device)
        index_seconds = time.perf_counter() - index_start

        gather_start = time.perf_counter()
        active = self.resident.gather(selected_ids)
        active.global_node_ids = selected_ids
        _sync(self.device)
        gather_seconds = time.perf_counter() - gather_start

        assembly_start = time.perf_counter()
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
        metrics.selected_cell_assembly_ms = gather_seconds * 1000.0
        metrics.alod_camera_preparation_ms = camera_seconds * 1000.0
        metrics.alod_upper_traversal_ms = selection.timings.upper_traversal_seconds * 1000.0
        metrics.alod_spt_cut_ms = selection.timings.spt_cut_seconds * 1000.0
        metrics.alod_selected_index_ms = index_seconds * 1000.0
        metrics.alod_selected_gather_ms = gather_seconds * 1000.0
        metrics.alod_total_selection_ms = selection.timings.total_selection_seconds * 1000.0
        metrics.alod_candidate_spt_count = candidate_count
        metrics.alod_selected_hierarchy_rows = selected_hierarchy_count
        metrics.alod_semantic_selected_rows = semantic_count
        metrics.alod_semantic_skybox_rows = semantic_count - selected_hierarchy_count
        metrics.alod_materialization_rows = raster_count
        metrics.alod_materialization_skybox_rows = int(
            np.count_nonzero(
                materialization.materialization_global_node_ids < self.config.skybox_count
            )
        )
        metrics.alod_materialization_hierarchy_rows = (
            raster_count - metrics.alod_materialization_skybox_rows
        )
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
        metrics.alod_materialization_selected_id_bytes = (
            filter_timings.selected_id_bytes
        )
        metrics.alod_materialization_retained_index_bytes = (
            filter_timings.retained_index_bytes
        )
        metrics.alod_total_raster_rows = raster_count
        metrics.alod_selected_index_bytes = selected_ids.numel() * selected_ids.element_size()
        active.scene_metrics = metrics
        assembly_seconds = time.perf_counter() - assembly_start
        metrics.alod_pre_raster_assembly_ms = assembly_seconds * 1000.0
        metrics.frame_preparation_ms = (time.perf_counter() - frame_start) * 1000.0
        self.last_selection = selection
        self.last_materialization = materialization
        self.last_active_splats = active
        self.last_metrics = metrics
        return active

    def get_scene_stats(self) -> SceneStats:
        return SceneStats(
            scene_type="alod_resident",
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
            alod_no_streaming=True,
            alod_hspt_build_ms=self.hspt_build_seconds * 1000.0,
            alod_hierarchy_metadata_cpu_bytes=self.hierarchy_metadata_cpu_bytes,
            alod_hierarchy_metadata_gpu_bytes=self.resident.global_id_bytes,
            alod_gaussian_properties_cpu_bytes=self.resident.serialized_cpu_bytes,
            alod_gaussian_properties_gpu_bytes=self.resident.resident_property_bytes,
            alod_materialization_filter=self.config.materialization_filter,
            alod_compact_culling_cpu_bytes=self.compact_culling_cpu_bytes,
            alod_compact_culling_incremental_cpu_bytes=(
                self.compact_culling_incremental_cpu_bytes
            ),
            alod_compact_culling_gpu_bytes=self.compact_culling_gpu_bytes,
        )

    def get_metrics(self) -> SceneFrameMetrics:
        return self.last_metrics


__all__ = ["AlodResidentScene"]
