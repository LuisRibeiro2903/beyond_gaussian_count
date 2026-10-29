"""Conservative post-selection materialization filtering for ALoD.

The semantic hierarchy selection is deliberately outside this module.  This
module only removes rows that a pinhole gsplat projection cannot accept, using
an isotropic covariance upper bound derived from each Gaussian's maximum
scale.  False positives are expected; backend-visible false negatives are not.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from scene_runtime.types import FrameCamera, RenderSettings

from .types import AlodMaterializationFilter


_GSPLAT_EXTENT = 3.33
_DEFAULT_SCALE_PADDING = 1.0e-4
_DEFAULT_PIXEL_PADDING = 2.0
_DEFAULT_DEPTH_PADDING = 1.0e-4


@dataclass(frozen=True)
class AlodCompactCullingMetadata:
    """Compact fields sufficient for the conservative pinhole upper bound."""

    means: np.ndarray
    max_scales: np.ndarray

    @classmethod
    def from_log_scales(
        cls, means: np.ndarray, log_scales: np.ndarray
    ) -> "AlodCompactCullingMetadata":
        means_array = np.asarray(means)
        log_scale_array = np.asarray(log_scales)
        if means_array.dtype != np.dtype("float32") or means_array.ndim != 2:
            raise ValueError("means must be float32[N,3]")
        if means_array.shape[1] != 3:
            raise ValueError("means must be float32[N,3]")
        if log_scale_array.dtype != np.dtype("float32") or log_scale_array.shape != means_array.shape:
            raise ValueError("log_scales must be float32[N,3]")
        max_log_scales = np.max(log_scale_array, axis=1).astype(np.float64)
        max_scales = np.exp(max_log_scales)
        max_scales *= 1.0 + _DEFAULT_SCALE_PADDING
        return cls(
            means=means_array,
            max_scales=np.ascontiguousarray(max_scales, dtype=np.float32),
        )

    @property
    def row_count(self) -> int:
        return int(self.means.shape[0])

    @property
    def logical_bytes(self) -> int:
        """Bytes for mean plus maximum scale, whether or not means are shared."""

        return int(self.means.nbytes + self.max_scales.nbytes)

    @property
    def incremental_bytes(self) -> int:
        """New bytes beyond the hierarchy's existing position table."""

        return int(self.max_scales.nbytes)

    @property
    def bytes_per_row(self) -> int:
        return 16

    def to_device(self, device: torch.device) -> "AlodDeviceCullingMetadata":
        """Copy the exact serialized compact table to ``device`` once."""

        return AlodDeviceCullingMetadata(
            means=torch.from_numpy(np.array(self.means, copy=True)).to(
                device=device, dtype=torch.float32
            ),
            max_scales=torch.from_numpy(np.array(self.max_scales, copy=True)).to(
                device=device, dtype=torch.float32
            ),
        )


@dataclass(frozen=True)
class AlodDeviceCullingMetadata:
    """Persistent device copy of the minimal conservative-culling fields."""

    means: torch.Tensor
    max_scales: torch.Tensor

    def __post_init__(self) -> None:
        if self.means.dtype != torch.float32 or self.means.ndim != 2:
            raise ValueError("device means must be float32[N,3]")
        if self.means.shape[1] != 3:
            raise ValueError("device means must be float32[N,3]")
        if self.max_scales.dtype != torch.float32 or self.max_scales.shape != (
            self.means.shape[0],
        ):
            raise ValueError("device max_scales must be float32[N]")
        if self.max_scales.device != self.means.device:
            raise ValueError("device compact metadata tensors must share a device")
        if not self.means.is_contiguous() or not self.max_scales.is_contiguous():
            raise ValueError("device compact metadata must be contiguous")

    @property
    def row_count(self) -> int:
        return int(self.means.shape[0])

    @property
    def logical_bytes(self) -> int:
        return int(
            self.means.numel() * self.means.element_size()
            + self.max_scales.numel() * self.max_scales.element_size()
        )

    @property
    def bytes_per_row(self) -> int:
        return 16


@dataclass(frozen=True)
class AlodMaterializationTimings:
    """Independently measured fast-filter stages."""

    id_preparation_seconds: float = 0.0
    selected_id_h2d_seconds: float = 0.0
    metadata_gather_seconds: float = 0.0
    projection_seconds: float = 0.0
    compaction_seconds: float = 0.0
    retained_index_d2h_seconds: float = 0.0
    materialization_id_production_seconds: float = 0.0
    unique_id_extraction_seconds: float = 0.0
    occurrence_propagation_seconds: float = 0.0
    total_seconds: float = 0.0
    selected_id_bytes: int = 0
    retained_index_bytes: int = 0


@dataclass(frozen=True)
class AlodMaterializationSelection:
    """Stable-ID mapping from semantic selection to actual raster input."""

    semantic_selected_global_node_ids: np.ndarray
    materialization_global_node_ids: np.ndarray
    raster_global_node_ids: np.ndarray
    kept_semantic_indices: np.ndarray
    rejected_depth_rows: int
    rejected_screen_bound_rows: int
    filter_seconds: float
    filter_name: str
    timings: AlodMaterializationTimings = AlodMaterializationTimings()
    materialization_global_node_ids_gpu: torch.Tensor | None = None
    kept_semantic_indices_gpu: torch.Tensor | None = None
    unique_semantic_global_node_ids: np.ndarray | None = None
    semantic_to_unique_indices: np.ndarray | None = None
    unique_materialization_global_node_ids: np.ndarray | None = None
    materialization_to_unique_indices: np.ndarray | None = None
    kept_unique_semantic_indices: np.ndarray | None = None

    @property
    def semantic_selected_rows(self) -> int:
        return int(self.semantic_selected_global_node_ids.size)

    @property
    def materialization_rows(self) -> int:
        return int(self.materialization_global_node_ids.size)

    @property
    def unique_semantic_rows(self) -> int:
        ids = self.unique_semantic_global_node_ids
        return int(ids.size if ids is not None else np.unique(
            self.semantic_selected_global_node_ids
        ).size)

    @property
    def unique_materialization_rows(self) -> int:
        ids = self.unique_materialization_global_node_ids
        return int(ids.size if ids is not None else np.unique(
            self.materialization_global_node_ids
        ).size)


def stable_unique_with_inverse(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return first-occurrence-ordered unique IDs and an exact inverse map."""

    values = np.ascontiguousarray(ids, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("global_node_ids must be one-dimensional")
    if values.size == 0:
        return values.copy(), values.copy()
    sorted_unique, first, sorted_inverse = np.unique(
        values, return_index=True, return_inverse=True
    )
    stable_order = np.argsort(first, kind="stable")
    unique_ids = np.ascontiguousarray(sorted_unique[stable_order], dtype=np.int64)
    sorted_to_stable = np.empty(stable_order.size, dtype=np.int64)
    sorted_to_stable[stable_order] = np.arange(stable_order.size, dtype=np.int64)
    inverse = np.ascontiguousarray(sorted_to_stable[sorted_inverse], dtype=np.int64)
    if not np.array_equal(unique_ids[inverse], values):
        raise RuntimeError("stable unique mapping failed to reconstruct ordered IDs")
    return unique_ids, inverse


def _camera_w2c(camera: FrameCamera) -> np.ndarray:
    matrix = camera.w2c if camera.w2c is not None else torch.linalg.inv(camera.c2w)
    return np.ascontiguousarray(
        matrix.detach().to(device="cpu", dtype=torch.float32).numpy(),
        dtype=np.float32,
    )


def identity_materialization(
    semantic_selected_global_node_ids: np.ndarray,
) -> AlodMaterializationSelection:
    ids = np.ascontiguousarray(semantic_selected_global_node_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("semantic selected global_node_ids must be one-dimensional")
    kept = np.arange(ids.size, dtype=np.int64)
    return AlodMaterializationSelection(
        semantic_selected_global_node_ids=ids.copy(),
        materialization_global_node_ids=ids.copy(),
        raster_global_node_ids=ids.copy(),
        kept_semantic_indices=kept,
        rejected_depth_rows=0,
        rejected_screen_bound_rows=0,
        filter_seconds=0.0,
        filter_name="none",
    )


def conservative_pinhole_materialization(
    metadata: AlodCompactCullingMetadata,
    semantic_selected_global_node_ids: np.ndarray,
    camera: FrameCamera,
    render_settings: RenderSettings,
    *,
    pixel_padding: float = _DEFAULT_PIXEL_PADDING,
    depth_padding: float = _DEFAULT_DEPTH_PADDING,
) -> AlodMaterializationSelection:
    """Conservatively reject depth- or screen-impossible pinhole rows.

    The screen radius uses the maximum 3D scale as an isotropic covariance
    upper bound.  It intentionally ignores opacity tightening.  Numerical
    padding covers float32 camera-transform and CUDA projection differences.
    Unsupported renderer modes fall back to the identity mapping.
    """

    ids = np.ascontiguousarray(semantic_selected_global_node_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("semantic selected global_node_ids must be one-dimensional")
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= metadata.row_count):
        raise IndexError("semantic selected global_node_ids lie outside compact metadata")
    if (
        render_settings.camera_model != "pinhole"
        or render_settings.with_ut
        or render_settings.with_eval3d
    ):
        return identity_materialization(ids)
    if pixel_padding < 0.0 or depth_padding < 0.0:
        raise ValueError("conservative padding must be non-negative")

    start = time.perf_counter()
    if ids.size == 0:
        result = identity_materialization(ids)
        return AlodMaterializationSelection(
            **{**result.__dict__, "filter_seconds": time.perf_counter() - start,
               "filter_name": "conservative_pinhole_mean_max_scale"}
        )

    # The serialized inputs and gsplat kernel are float32.  Scale, depth, and
    # pixel padding deliberately make this analytic upper bound wider than
    # minor CPU/CUDA evaluation-order differences.
    means = metadata.means[ids]
    max_scales = metadata.max_scales[ids]
    w2c = _camera_w2c(camera)
    x = np.asarray(
        means[:, 0] * w2c[0, 0]
        + means[:, 1] * w2c[0, 1]
        + means[:, 2] * w2c[0, 2]
        + w2c[0, 3],
        dtype=np.float32,
    )
    y = np.asarray(
        means[:, 0] * w2c[1, 0]
        + means[:, 1] * w2c[1, 1]
        + means[:, 2] * w2c[1, 2]
        + w2c[1, 3],
        dtype=np.float32,
    )
    z = np.asarray(
        means[:, 0] * w2c[2, 0]
        + means[:, 1] * w2c[2, 1]
        + means[:, 2] * w2c[2, 2]
        + w2c[2, 3],
        dtype=np.float32,
    )

    near = float(render_settings.near_plane)
    far = float(render_settings.far_plane)
    depth_keep = np.isfinite(z) & (z >= near - depth_padding) & (z <= far + depth_padding)

    screen_keep = np.ones(ids.size, dtype=bool)
    projectable = depth_keep & (z > 0.0)
    if np.any(projectable):
        K = camera.K.detach().to(device="cpu", dtype=torch.float32).numpy()
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        width = float(render_settings.width)
        height = float(render_settings.height)
        eps2d = float(render_settings.eps2d)

        indices = np.flatnonzero(projectable)
        px = x[indices]
        py = y[indices]
        pz = z[indices]
        scale = max_scales[indices]
        rz = 1.0 / pz
        tan_fovx = 0.5 * width / fx
        tan_fovy = 0.5 * height / fy
        lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
        lim_x_neg = cx / fx + 0.3 * tan_fovx
        lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
        lim_y_neg = cy / fy + 0.3 * tan_fovy
        tx = pz * np.clip(px * rz, -lim_x_neg, lim_x_pos)
        ty = pz * np.clip(py * rz, -lim_y_neg, lim_y_pos)

        jx_norm2 = (fx * rz) ** 2 + (-fx * tx * rz * rz) ** 2
        jy_norm2 = (fy * rz) ** 2 + (-fy * ty * rz * rz) ** 2
        radius_x = np.ceil(
            _GSPLAT_EXTENT * np.sqrt(np.maximum(scale * scale * jx_norm2 + eps2d, 0.0))
        ) + pixel_padding
        radius_y = np.ceil(
            _GSPLAT_EXTENT * np.sqrt(np.maximum(scale * scale * jy_norm2 + eps2d, 0.0))
        ) + pixel_padding
        mean2d_x = fx * px * rz + cx
        mean2d_y = fy * py * rz + cy
        overlaps = (
            (mean2d_x + radius_x > 0.0)
            & (mean2d_x - radius_x < width)
            & (mean2d_y + radius_y > 0.0)
            & (mean2d_y - radius_y < height)
        )
        screen_keep[indices] = overlaps

    keep = depth_keep & screen_keep
    kept_indices = np.ascontiguousarray(np.flatnonzero(keep), dtype=np.int64)
    materialized = np.ascontiguousarray(ids[keep], dtype=np.int64)
    elapsed = time.perf_counter() - start
    return AlodMaterializationSelection(
        semantic_selected_global_node_ids=ids.copy(),
        materialization_global_node_ids=materialized,
        raster_global_node_ids=materialized.copy(),
        kept_semantic_indices=kept_indices,
        rejected_depth_rows=int(np.count_nonzero(~depth_keep)),
        rejected_screen_bound_rows=int(np.count_nonzero(depth_keep & ~screen_keep)),
        filter_seconds=elapsed,
        filter_name="conservative_pinhole_mean_max_scale",
    )


def _event(device: torch.device) -> torch.cuda.Event | None:
    if device.type != "cuda":
        return None
    return torch.cuda.Event(enable_timing=True)


def _elapsed_seconds(
    start: torch.cuda.Event | None,
    end: torch.cuda.Event | None,
    fallback: float,
) -> float:
    if start is None or end is None:
        return fallback
    return float(start.elapsed_time(end)) / 1000.0


def fast_conservative_pinhole_materialization(
    metadata: AlodDeviceCullingMetadata,
    semantic_selected_global_node_ids: np.ndarray,
    camera: FrameCamera,
    render_settings: RenderSettings,
    *,
    pixel_padding: float = _DEFAULT_PIXEL_PADDING,
    depth_padding: float = _DEFAULT_DEPTH_PADDING,
) -> AlodMaterializationSelection:
    """Run the validated mean/max-scale filter with vectorized torch ops.

    The arithmetic and comparison order intentionally mirror
    :func:`conservative_pinhole_materialization`.  Only compacted semantic
    positions return to CPU; host/cache lookup reconstructs stable global IDs
    from the unchanged CPU selector output.  The compacted global IDs remain
    available on the metadata device for resident rendering and ID logging.
    """

    total_start = time.perf_counter()
    preparation_start = total_start
    ids = np.ascontiguousarray(semantic_selected_global_node_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("semantic selected global_node_ids must be one-dimensional")
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= metadata.row_count):
        raise IndexError("semantic selected global_node_ids lie outside compact metadata")
    if (
        render_settings.camera_model != "pinhole"
        or render_settings.with_ut
        or render_settings.with_eval3d
    ):
        return identity_materialization(ids)
    if pixel_padding < 0.0 or depth_padding < 0.0:
        raise ValueError("conservative padding must be non-negative")

    device = metadata.means.device
    ids_cpu = torch.from_numpy(ids)
    preparation_seconds = time.perf_counter() - preparation_start
    selected_bytes = int(ids_cpu.numel() * ids_cpu.element_size())
    if ids.size == 0:
        empty_gpu = torch.empty(0, dtype=torch.int64, device=device)
        elapsed = time.perf_counter() - total_start
        return AlodMaterializationSelection(
            semantic_selected_global_node_ids=ids.copy(),
            materialization_global_node_ids=ids.copy(),
            raster_global_node_ids=ids.copy(),
            kept_semantic_indices=ids.copy(),
            rejected_depth_rows=0,
            rejected_screen_bound_rows=0,
            filter_seconds=elapsed,
            filter_name="conservative_pinhole_mean_max_scale_gpu",
            timings=AlodMaterializationTimings(
                id_preparation_seconds=preparation_seconds,
                total_seconds=elapsed,
                selected_id_bytes=selected_bytes,
            ),
            materialization_global_node_ids_gpu=empty_gpu,
            kept_semantic_indices_gpu=empty_gpu,
        )

    h2d_start_wall = time.perf_counter()
    h2d_start = _event(device)
    h2d_end = _event(device)
    if h2d_start is not None:
        h2d_start.record()
    ids_gpu = ids_cpu.to(device=device, dtype=torch.int64)
    if h2d_end is not None:
        h2d_end.record()
    h2d_wall = time.perf_counter() - h2d_start_wall

    gather_start_wall = time.perf_counter()
    gather_start = _event(device)
    gather_end = _event(device)
    if gather_start is not None:
        gather_start.record()
    means = metadata.means.index_select(0, ids_gpu)
    max_scales = metadata.max_scales.index_select(0, ids_gpu)
    if gather_end is not None:
        gather_end.record()
    gather_wall = time.perf_counter() - gather_start_wall

    projection_start_wall = time.perf_counter()
    projection_start = _event(device)
    projection_end = _event(device)
    if projection_start is not None:
        projection_start.record()
    w2c = torch.from_numpy(_camera_w2c(camera)).to(device=device)
    x = (
        means[:, 0] * w2c[0, 0]
        + means[:, 1] * w2c[0, 1]
        + means[:, 2] * w2c[0, 2]
        + w2c[0, 3]
    )
    y = (
        means[:, 0] * w2c[1, 0]
        + means[:, 1] * w2c[1, 1]
        + means[:, 2] * w2c[1, 2]
        + w2c[1, 3]
    )
    z = (
        means[:, 0] * w2c[2, 0]
        + means[:, 1] * w2c[2, 1]
        + means[:, 2] * w2c[2, 2]
        + w2c[2, 3]
    )
    near = float(render_settings.near_plane)
    far = float(render_settings.far_plane)
    depth_keep = torch.isfinite(z) & (z >= near - depth_padding) & (
        z <= far + depth_padding
    )
    projectable = depth_keep & (z > 0.0)

    K = camera.K.detach().to(device="cpu", dtype=torch.float32).numpy()
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    width = float(render_settings.width)
    height = float(render_settings.height)
    eps2d = float(render_settings.eps2d)
    rz = 1.0 / z
    tan_fovx = 0.5 * width / fx
    tan_fovy = 0.5 * height / fy
    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx = z * torch.clamp(x * rz, min=-lim_x_neg, max=lim_x_pos)
    ty = z * torch.clamp(y * rz, min=-lim_y_neg, max=lim_y_pos)
    jx_norm2 = (fx * rz) ** 2 + (-fx * tx * rz * rz) ** 2
    jy_norm2 = (fy * rz) ** 2 + (-fy * ty * rz * rz) ** 2
    radius_x = torch.ceil(
        _GSPLAT_EXTENT
        * torch.sqrt(torch.clamp_min(max_scales * max_scales * jx_norm2 + eps2d, 0.0))
    ) + pixel_padding
    radius_y = torch.ceil(
        _GSPLAT_EXTENT
        * torch.sqrt(torch.clamp_min(max_scales * max_scales * jy_norm2 + eps2d, 0.0))
    ) + pixel_padding
    mean2d_x = fx * x * rz + cx
    mean2d_y = fy * y * rz + cy
    screen_keep = (
        (mean2d_x + radius_x > 0.0)
        & (mean2d_x - radius_x < width)
        & (mean2d_y + radius_y > 0.0)
        & (mean2d_y - radius_y < height)
    )
    effective_screen_keep = ~projectable | screen_keep
    keep = depth_keep & effective_screen_keep
    if projection_end is not None:
        projection_end.record()
    projection_wall = time.perf_counter() - projection_start_wall

    compaction_start_wall = time.perf_counter()
    compaction_start = _event(device)
    compaction_end = _event(device)
    if compaction_start is not None:
        compaction_start.record()
    kept_gpu = torch.nonzero(keep, as_tuple=False).reshape(-1)
    materialized_gpu = ids_gpu.index_select(0, kept_gpu)
    rejected_counts_gpu = torch.stack(
        ((~depth_keep).sum(), (depth_keep & ~effective_screen_keep).sum())
    ).to(dtype=torch.int64)
    if compaction_end is not None:
        compaction_end.record()
    compaction_wall = time.perf_counter() - compaction_start_wall

    d2h_start_wall = time.perf_counter()
    d2h_start = _event(device)
    d2h_end = _event(device)
    if d2h_start is not None:
        d2h_start.record()
    kept = np.ascontiguousarray(kept_gpu.detach().cpu().numpy(), dtype=np.int64)
    rejected_counts = rejected_counts_gpu.detach().cpu().numpy()
    if d2h_end is not None:
        d2h_end.record()
        d2h_end.synchronize()
    d2h_wall = time.perf_counter() - d2h_start_wall

    production_start = time.perf_counter()
    materialized = np.ascontiguousarray(ids[kept], dtype=np.int64)
    production_seconds = time.perf_counter() - production_start
    elapsed = time.perf_counter() - total_start
    timings = AlodMaterializationTimings(
        id_preparation_seconds=preparation_seconds,
        selected_id_h2d_seconds=_elapsed_seconds(h2d_start, h2d_end, h2d_wall),
        metadata_gather_seconds=_elapsed_seconds(
            gather_start, gather_end, gather_wall
        ),
        projection_seconds=_elapsed_seconds(
            projection_start, projection_end, projection_wall
        ),
        compaction_seconds=_elapsed_seconds(
            compaction_start, compaction_end, compaction_wall
        ),
        retained_index_d2h_seconds=_elapsed_seconds(d2h_start, d2h_end, d2h_wall),
        materialization_id_production_seconds=production_seconds,
        total_seconds=elapsed,
        selected_id_bytes=selected_bytes,
        retained_index_bytes=int(kept.nbytes),
    )
    return AlodMaterializationSelection(
        semantic_selected_global_node_ids=ids.copy(),
        materialization_global_node_ids=materialized,
        raster_global_node_ids=materialized.copy(),
        kept_semantic_indices=kept,
        rejected_depth_rows=int(rejected_counts[0]),
        rejected_screen_bound_rows=int(rejected_counts[1]),
        filter_seconds=elapsed,
        filter_name="conservative_pinhole_mean_max_scale_gpu",
        timings=timings,
        materialization_global_node_ids_gpu=materialized_gpu,
        kept_semantic_indices_gpu=kept_gpu,
    )


def select_materialization(
    filter_name: AlodMaterializationFilter,
    metadata: AlodCompactCullingMetadata | None,
    semantic_selected_global_node_ids: np.ndarray,
    camera: FrameCamera,
    render_settings: RenderSettings,
    *,
    device_metadata: AlodDeviceCullingMetadata | None = None,
) -> AlodMaterializationSelection:
    if filter_name == "none":
        return identity_materialization(semantic_selected_global_node_ids)
    if metadata is None:
        raise ValueError("conservative materialization requires compact culling metadata")
    if filter_name == "conservative_pinhole_gpu":
        if device_metadata is None:
            raise ValueError("GPU conservative materialization requires device metadata")
        return fast_conservative_pinhole_materialization(
            device_metadata,
            semantic_selected_global_node_ids,
            camera,
            render_settings,
        )
    return conservative_pinhole_materialization(
        metadata,
        semantic_selected_global_node_ids,
        camera,
        render_settings,
    )


def select_multiset_materialization(
    filter_name: AlodMaterializationFilter,
    metadata: AlodCompactCullingMetadata | None,
    semantic_selected_global_node_ids: np.ndarray,
    camera: FrameCamera,
    render_settings: RenderSettings,
    *,
    device_metadata: AlodDeviceCullingMetadata | None = None,
) -> AlodMaterializationSelection:
    """Filter unique property rows once, then restore ordered multiplicity.

    The conservative predicate is row-local: it depends on compact Gaussian
    metadata and the camera/render settings, never on occurrence position or
    neighbouring rows.  Broadcasting a unique row's decision to every
    occurrence is therefore equivalent to evaluating duplicate rows
    repeatedly while allowing residency to remain keyed by unique global ID.
    """

    total_start = time.perf_counter()
    unique_start = total_start
    semantic = np.ascontiguousarray(
        semantic_selected_global_node_ids, dtype=np.int64
    )
    unique_ids, semantic_to_unique = stable_unique_with_inverse(semantic)
    unique_seconds = time.perf_counter() - unique_start

    unique_result = select_materialization(
        filter_name,
        metadata,
        unique_ids,
        camera,
        render_settings,
        device_metadata=device_metadata,
    )

    propagation_start = time.perf_counter()
    kept_unique = np.ascontiguousarray(
        unique_result.kept_semantic_indices, dtype=np.int64
    )
    unique_keep = np.zeros(unique_ids.size, dtype=bool)
    unique_keep[kept_unique] = True
    occurrence_keep = unique_keep[semantic_to_unique]
    kept_occurrences = np.ascontiguousarray(
        np.flatnonzero(occurrence_keep), dtype=np.int64
    )
    materialized_occurrences = np.ascontiguousarray(
        semantic[kept_occurrences], dtype=np.int64
    )
    retained_unique_ids = np.ascontiguousarray(
        unique_result.materialization_global_node_ids, dtype=np.int64
    )
    unique_to_retained = np.full(unique_ids.size, -1, dtype=np.int64)
    unique_to_retained[kept_unique] = np.arange(kept_unique.size, dtype=np.int64)
    materialization_to_unique = np.ascontiguousarray(
        unique_to_retained[semantic_to_unique[kept_occurrences]], dtype=np.int64
    )
    if np.any(materialization_to_unique < 0):
        raise RuntimeError("retained occurrence did not resolve to unique backing")
    if not np.array_equal(
        retained_unique_ids[materialization_to_unique], materialized_occurrences
    ):
        raise RuntimeError("unique backing failed to reconstruct retained occurrences")

    materialized_gpu = None
    kept_occurrences_gpu = None
    if unique_result.materialization_global_node_ids_gpu is not None:
        mapping_gpu = torch.from_numpy(materialization_to_unique).to(
            device=unique_result.materialization_global_node_ids_gpu.device,
            dtype=torch.int64,
        )
        materialized_gpu = unique_result.materialization_global_node_ids_gpu.index_select(
            0, mapping_gpu
        )
        kept_occurrences_gpu = torch.from_numpy(kept_occurrences).to(
            device=materialized_gpu.device, dtype=torch.int64
        )
    propagation_seconds = time.perf_counter() - propagation_start
    elapsed = time.perf_counter() - total_start
    base_timings = unique_result.timings
    timings = AlodMaterializationTimings(
        id_preparation_seconds=base_timings.id_preparation_seconds,
        selected_id_h2d_seconds=base_timings.selected_id_h2d_seconds,
        metadata_gather_seconds=base_timings.metadata_gather_seconds,
        projection_seconds=base_timings.projection_seconds,
        compaction_seconds=base_timings.compaction_seconds,
        retained_index_d2h_seconds=base_timings.retained_index_d2h_seconds,
        materialization_id_production_seconds=(
            base_timings.materialization_id_production_seconds
        ),
        unique_id_extraction_seconds=unique_seconds,
        occurrence_propagation_seconds=propagation_seconds,
        total_seconds=elapsed,
        selected_id_bytes=base_timings.selected_id_bytes,
        retained_index_bytes=base_timings.retained_index_bytes,
    )
    return AlodMaterializationSelection(
        semantic_selected_global_node_ids=semantic.copy(),
        materialization_global_node_ids=materialized_occurrences,
        raster_global_node_ids=materialized_occurrences.copy(),
        kept_semantic_indices=kept_occurrences,
        rejected_depth_rows=unique_result.rejected_depth_rows,
        rejected_screen_bound_rows=unique_result.rejected_screen_bound_rows,
        filter_seconds=elapsed,
        filter_name=f"{unique_result.filter_name}_ordered_multiset",
        timings=timings,
        materialization_global_node_ids_gpu=materialized_gpu,
        kept_semantic_indices_gpu=kept_occurrences_gpu,
        unique_semantic_global_node_ids=unique_ids,
        semantic_to_unique_indices=semantic_to_unique,
        unique_materialization_global_node_ids=retained_unique_ids,
        materialization_to_unique_indices=materialization_to_unique,
        kept_unique_semantic_indices=kept_unique,
    )


__all__ = [
    "AlodCompactCullingMetadata",
    "AlodDeviceCullingMetadata",
    "AlodMaterializationSelection",
    "AlodMaterializationTimings",
    "conservative_pinhole_materialization",
    "fast_conservative_pinhole_materialization",
    "identity_materialization",
    "select_materialization",
    "select_multiset_materialization",
    "stable_unique_with_inverse",
]
