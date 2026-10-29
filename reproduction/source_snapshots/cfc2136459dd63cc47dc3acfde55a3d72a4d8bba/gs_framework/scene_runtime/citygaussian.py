from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .ply_splats import load_gaussian_ply
from .types import ActiveSplats, FrameCamera, RenderSettings, SceneFrameMetrics, SceneStats


OFFICIAL_MATRIXCITY_BLOCK_DIM = (6, 6, 1)
OFFICIAL_MATRIXCITY_AABB = (-3.5, -4.0, -10.0, 4.5, 2.0, 10.0)
OFFICIAL_MAD_MULTIPLIER = 4.0
RELEASE_LOD_POLICY = "CG-LOD-RELEASE"
RELEASE_LOD_LEVEL_LABELS = ("light_50_vq", "light_66_vq", "light_75_vq")
RELEASE_LOD_LEVEL_SEMANTICS = ("finest", "middle", "coarsest")


LoadProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class CityGaussianCell:
    cell_id: int
    grid_index: tuple[int, int, int]
    source_indices: Tensor
    mad_bounds_min: tuple[float, float, float]
    mad_bounds_max: tuple[float, float, float]
    corners: Tensor
    average_scale: float

    @property
    def gaussian_count(self) -> int:
        return int(self.source_indices.shape[0])


@dataclass(frozen=True)
class CityGaussianCellValidation:
    total_gaussians: int
    assigned_total: int
    unassigned_count: int
    duplicated_assignment_count: int
    every_gaussian_assigned_once: bool
    all_cells_nonempty: bool
    attribute_equivalence: bool
    concatenated_source_indices: Tensor


@dataclass(frozen=True)
class CityGaussianOfficialCamera:
    world_view_transform: Tensor
    projection_matrix: Tensor
    full_proj_transform: Tensor
    camera_center: Tensor
    fov_x: float
    fov_y: float
    width: int
    height: int

    def as_official_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(
            world_view_transform=self.world_view_transform,
            projection_matrix=self.projection_matrix,
            full_proj_transform=self.full_proj_transform,
            camera_center=self.camera_center,
            FoVx=float(self.fov_x),
            FoVy=float(self.fov_y),
            image_width=int(self.width),
            image_height=int(self.height),
        )


@dataclass(frozen=True)
class CityGaussianVisibilitySelection:
    mask: Tensor
    selected_cell_ids: list[int]
    selected_cell_count: int
    selected_gaussian_count: int
    nearest_cell_id: int
    nearest_cell_fallback_changed_mask: bool
    per_cell_distances: list[float]
    per_visible_cell_distances: dict[int, float]
    per_visible_cell_projected_boxes: dict[int, list[float]]
    per_cell_valid_front_corner_counts: list[int]


@dataclass(frozen=True)
class CityGaussianReleaseLodSelection:
    visible_cell_ids: list[int]
    selected_lod_by_cell: dict[int, int]
    selected_level_label_by_cell: dict[int, str]
    selected_gaussian_count_by_cell: dict[int, int]
    selected_cell_count_by_level: list[int]
    selected_gaussian_count_by_level: list[int]
    total_selected_gaussian_count: int
    focal_length_pixels: float
    nyquist_scaling_by_cell: dict[int, float]
    average_scales_by_cell: dict[int, list[float]]


def released_nyquist_lod_index(
    average_scales_by_level: Sequence[float],
    nyquist_scaling: float,
) -> int:
    """Return the first strict Nyquist match, or the final/coarsest level."""
    if not average_scales_by_level:
        raise ValueError("At least one LoD average scale is required.")
    for level_index, average_scale in enumerate(average_scales_by_level):
        if float(average_scale) > float(nyquist_scaling):
            return int(level_index)
    return len(average_scales_by_level) - 1


def select_citygaussian_release_lod(
    *,
    visible_cell_ids: Sequence[int],
    per_visible_cell_distances: dict[int, float],
    average_scales_by_level: Sequence[Sequence[float]],
    cell_counts_by_level: Sequence[Sequence[int]],
    level_labels: Sequence[str],
    image_width: int,
    fov_x: float,
) -> CityGaussianReleaseLodSelection:
    """Apply the released V1 Nyquist rule to already-visible logical cells."""
    level_count = len(level_labels)
    if level_count == 0:
        raise ValueError("At least one CityGaussian LoD level is required.")
    if len(average_scales_by_level) != level_count or len(cell_counts_by_level) != level_count:
        raise ValueError("LoD labels, average scales, and cell counts must have equal level counts.")
    if int(image_width) <= 0:
        raise ValueError("image_width must be positive.")
    focal_length = 0.5 * float(image_width) / math.tan(float(fov_x) * 0.5)
    if not math.isfinite(focal_length) or focal_length <= 0.0:
        raise ValueError("Computed focal length must be finite and positive.")

    selected_lod_by_cell: dict[int, int] = {}
    selected_level_label_by_cell: dict[int, str] = {}
    selected_gaussian_count_by_cell: dict[int, int] = {}
    selected_cell_count_by_level = [0 for _ in range(level_count)]
    selected_gaussian_count_by_level = [0 for _ in range(level_count)]
    nyquist_scaling_by_cell: dict[int, float] = {}
    average_scales_by_cell: dict[int, list[float]] = {}
    total_selected = 0

    for raw_cell_id in visible_cell_ids:
        cell_id = int(raw_cell_id)
        if cell_id not in per_visible_cell_distances:
            raise ValueError(f"Missing distance for visible cell {cell_id}.")
        averages = [float(values[cell_id]) for values in average_scales_by_level]
        nyquist_scaling = 2.0 * float(per_visible_cell_distances[cell_id]) / focal_length
        level_index = released_nyquist_lod_index(averages, nyquist_scaling)
        gaussian_count = int(cell_counts_by_level[level_index][cell_id])

        selected_lod_by_cell[cell_id] = level_index
        selected_level_label_by_cell[cell_id] = str(level_labels[level_index])
        selected_gaussian_count_by_cell[cell_id] = gaussian_count
        selected_cell_count_by_level[level_index] += 1
        selected_gaussian_count_by_level[level_index] += gaussian_count
        nyquist_scaling_by_cell[cell_id] = nyquist_scaling
        average_scales_by_cell[cell_id] = averages
        total_selected += gaussian_count

    return CityGaussianReleaseLodSelection(
        visible_cell_ids=[int(cell_id) for cell_id in visible_cell_ids],
        selected_lod_by_cell=selected_lod_by_cell,
        selected_level_label_by_cell=selected_level_label_by_cell,
        selected_gaussian_count_by_cell=selected_gaussian_count_by_cell,
        selected_cell_count_by_level=selected_cell_count_by_level,
        selected_gaussian_count_by_level=selected_gaussian_count_by_level,
        total_selected_gaussian_count=total_selected,
        focal_length_pixels=focal_length,
        nyquist_scaling_by_cell=nyquist_scaling_by_cell,
        average_scales_by_cell=average_scales_by_cell,
    )


def contract_to_unisphere(
    xyz: Tensor,
    aabb: Sequence[float] = OFFICIAL_MATRIXCITY_AABB,
    ord: float = math.inf,
) -> Tensor:
    """Match CityGaussian V1 `utils.large_utils.contract_to_unisphere`."""
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected xyz with shape (N, 3), got {tuple(xyz.shape)}.")
    bounds = torch.tensor(aabb, dtype=xyz.dtype, device=xyz.device)
    aabb_min, aabb_max = torch.split(bounds, 3, dim=-1)
    x = (xyz - aabb_min) / (aabb_max - aabb_min)
    x = x * 2.0 - 1.0
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1.0
    out = x.clone()
    if bool(mask.any()):
        out[mask] = (2.0 - 1.0 / mag[mask]) * (out[mask] / mag[mask])
    return out / 4.0 + 0.5


def official_projection_matrix(znear: float, zfar: float, fov_x: float, fov_y: float) -> Tensor:
    """Match CityGaussian V1 `utils.graphics_utils.getProjectionMatrix(...).T`."""
    tan_half_y = math.tan(float(fov_y) / 2.0)
    tan_half_x = math.tan(float(fov_x) / 2.0)
    top = tan_half_y * float(znear)
    bottom = -top
    right = tan_half_x * float(znear)
    left = -right
    matrix = torch.zeros((4, 4), dtype=torch.float32)
    z_sign = 1.0
    matrix[0, 0] = 2.0 * float(znear) / (right - left)
    matrix[1, 1] = 2.0 * float(znear) / (top - bottom)
    matrix[0, 2] = (right + left) / (right - left)
    matrix[1, 2] = (top + bottom) / (top - bottom)
    matrix[3, 2] = z_sign
    matrix[2, 2] = z_sign * float(zfar) / (float(zfar) - float(znear))
    matrix[2, 3] = -(float(zfar) * float(znear)) / (float(zfar) - float(znear))
    return matrix.T.contiguous()


def citygaussian_official_camera_from_c2w(
    c2w: Tensor,
    K: Tensor,
    *,
    width: int,
    height: int,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> CityGaussianOfficialCamera:
    """Convert the already project-converted runtime camera to official V1 matrices."""
    if c2w.shape != (4, 4):
        raise ValueError(f"Camera c2w must have shape (4, 4), got {tuple(c2w.shape)}.")
    if K.shape != (3, 3):
        raise ValueError(f"Camera K must have shape (3, 3), got {tuple(K.shape)}.")
    if not bool(torch.isfinite(c2w).all().item()):
        raise ValueError("Camera c2w contains non-finite values.")
    if not bool(torch.isfinite(K).all().item()):
        raise ValueError("Camera K contains non-finite values.")
    device = c2w.device
    c2w = c2w.to(device=device, dtype=torch.float32)
    K = K.to(device=device, dtype=torch.float32)
    fov_x = 2.0 * math.atan(float(width) / (2.0 * float(K[0, 0].item())))
    fov_y = 2.0 * math.atan(float(height) / (2.0 * float(K[1, 1].item())))
    world_view = torch.linalg.inv(c2w).T.contiguous()
    projection = official_projection_matrix(znear, zfar, fov_x, fov_y).to(device=device)
    full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    camera_center = torch.linalg.inv(world_view)[3, :3]
    return CityGaussianOfficialCamera(
        world_view_transform=world_view,
        projection_matrix=projection,
        full_proj_transform=full_projection,
        camera_center=camera_center,
        fov_x=fov_x,
        fov_y=fov_y,
        width=int(width),
        height=int(height),
    )


def citygaussian_official_camera_from_frame(
    camera: FrameCamera,
    render_settings: RenderSettings,
    *,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> CityGaussianOfficialCamera:
    cull_c2w, cull_K = camera.cull_camera
    return citygaussian_official_camera_from_c2w(
        cull_c2w,
        cull_K,
        width=int(render_settings.width),
        height=int(render_settings.height),
        znear=znear,
        zfar=zfar,
    )


def citygaussian_in_frustum(
    official_camera: CityGaussianOfficialCamera | SimpleNamespace,
    cell_corners: Tensor,
    cell_counts: Sequence[int],
    *,
    aabb: Sequence[float],
    block_dim: Sequence[int],
) -> CityGaussianVisibilitySelection:
    """Official CityGaussian V1 projected-corner block visibility selection."""
    del aabb, block_dim
    if cell_corners.ndim != 3 or cell_corners.shape[1:] != (8, 3):
        raise ValueError(f"Expected cell_corners with shape (C, 8, 3), got {tuple(cell_corners.shape)}.")
    num_cell = int(cell_corners.shape[0])
    if len(cell_counts) != num_cell:
        raise ValueError(f"Expected {num_cell} cell counts, got {len(cell_counts)}.")
    device = cell_corners.device
    corners_h = torch.cat([cell_corners, torch.ones_like(cell_corners[..., [0]])], dim=-1)
    full_proj_transform = official_camera.full_proj_transform.to(device=device, dtype=torch.float32).repeat(
        num_cell, 1, 1
    )
    viewmatrix = official_camera.world_view_transform.to(device=device, dtype=torch.float32).repeat(
        num_cell, 1, 1
    )

    corners_screen = corners_h.bmm(full_proj_transform)
    corners_screen = corners_screen / corners_screen[..., [-1]]
    corners_screen = corners_screen[..., :-1].reshape(-1, 3)

    corners_cam = corners_h.bmm(viewmatrix)
    dist = torch.norm(corners_cam[:, :, :3], dim=-1)
    dist_min = torch.min(dist, dim=-1).values
    nearest_cell_id = int(torch.argmin(dist_min).item())
    front_mask = corners_cam[..., 2] > 0.2

    flat_front = front_mask.reshape(-1)
    screen_work = corners_screen.clone().reshape(-1, 3)
    screen_work[~flat_front] = torch.inf
    screen_min = screen_work.reshape(num_cell, -1, 3).min(dim=1).values
    screen_min[screen_min == torch.inf] = 0.0

    screen_work = corners_screen.clone().reshape(-1, 3)
    screen_work[~flat_front] = -torch.inf
    screen_max = screen_work.reshape(num_cell, -1, 3).max(dim=1).values
    screen_max[screen_max == -torch.inf] = 0.0

    boxes = torch.cat([screen_min[:, :2], screen_max[:, :2]], dim=1)
    screen = torch.tensor([[-1.0, -1.0, 1.0, 1.0]], dtype=torch.float32, device=device)
    max_xy = torch.min(boxes[:, 2:].unsqueeze(1).expand(num_cell, 1, 2), screen[:, 2:].unsqueeze(0))
    min_xy = torch.max(boxes[:, :2].unsqueeze(1).expand(num_cell, 1, 2), screen[:, :2].unsqueeze(0))
    inter = torch.clamp(max_xy - min_xy, min=0.0)
    pre_fallback_mask = (inter[:, 0, 0] * inter[:, 0, 1]) > 0.0
    mask = pre_fallback_mask.clone()
    mask[nearest_cell_id] = True

    selected_cell_ids = [int(v) for v in torch.nonzero(mask, as_tuple=False).flatten().cpu().tolist()]
    selected_gaussian_count = int(sum(int(cell_counts[cell_id]) for cell_id in selected_cell_ids))
    distances_cpu = dist_min.detach().cpu().to(dtype=torch.float32)
    boxes_cpu = boxes.detach().cpu().to(dtype=torch.float32)
    return CityGaussianVisibilitySelection(
        mask=mask,
        selected_cell_ids=selected_cell_ids,
        selected_cell_count=len(selected_cell_ids),
        selected_gaussian_count=selected_gaussian_count,
        nearest_cell_id=nearest_cell_id,
        nearest_cell_fallback_changed_mask=bool(
            mask[nearest_cell_id].item() and not pre_fallback_mask[nearest_cell_id].item()
        ),
        per_cell_distances=[float(v) for v in distances_cpu.tolist()],
        per_visible_cell_distances={cell_id: float(distances_cpu[cell_id].item()) for cell_id in selected_cell_ids},
        per_visible_cell_projected_boxes={
            cell_id: [float(v) for v in boxes_cpu[cell_id].tolist()] for cell_id in selected_cell_ids
        },
        per_cell_valid_front_corner_counts=[int(v) for v in front_mask.sum(dim=1).detach().cpu().tolist()],
    )


def cell_grid_index(cell_id: int, block_dim: Sequence[int]) -> tuple[int, int, int]:
    x_dim, y_dim, _z_dim = [int(v) for v in block_dim]
    z = int(cell_id) // (x_dim * y_dim)
    y = (int(cell_id) % (x_dim * y_dim)) // x_dim
    x = (int(cell_id) % (x_dim * y_dim)) % x_dim
    return x, y, z


def official_cell_mask(
    cell_id: int,
    xyz: Tensor,
    *,
    block_dim: Sequence[int] = OFFICIAL_MATRIXCITY_BLOCK_DIM,
    aabb: Sequence[float] = OFFICIAL_MATRIXCITY_AABB,
    scale: float = 1.0,
) -> Tensor:
    """Return the same half-open mask as CityGaussian V1 `block_filtering`."""
    dims = [int(v) for v in block_dim]
    x_id, y_id, z_id = cell_grid_index(cell_id, dims)
    contracted = contract_to_unisphere(xyz, aabb, ord=math.inf)

    min_x, max_x = float(x_id) / dims[0], float(x_id + 1) / dims[0]
    min_y, max_y = float(y_id) / dims[1], float(y_id + 1) / dims[1]
    min_z, max_z = float(z_id) / dims[2], float(z_id + 1) / dims[2]

    delta_x = (max_x - min_x) * (float(scale) - 1.0)
    delta_y = (max_y - min_y) * (float(scale) - 1.0)
    delta_z = (max_z - min_z) * (float(scale) - 1.0)
    min_x -= delta_x / 2.0
    max_x += delta_x / 2.0
    min_y -= delta_y / 2.0
    max_y += delta_y / 2.0
    min_z -= delta_z / 2.0
    max_z += delta_z / 2.0

    return (
        (contracted[:, 0] >= min_x)
        & (contracted[:, 0] < max_x)
        & (contracted[:, 1] >= min_y)
        & (contracted[:, 1] < max_y)
        & (contracted[:, 2] >= min_z)
        & (contracted[:, 2] < max_z)
    )


def official_cell_ids(
    xyz: Tensor,
    *,
    block_dim: Sequence[int] = OFFICIAL_MATRIXCITY_BLOCK_DIM,
    aabb: Sequence[float] = OFFICIAL_MATRIXCITY_AABB,
) -> Tensor:
    dims = torch.tensor(block_dim, dtype=torch.long, device=xyz.device)
    contracted = contract_to_unisphere(xyz, aabb, ord=math.inf)
    ids = torch.floor(contracted * dims.to(dtype=contracted.dtype)).to(dtype=torch.long)
    ids = torch.clamp(ids, min=0)
    ids = torch.minimum(ids, dims[None, :] - 1)
    return ids[:, 2] * dims[0] * dims[1] + ids[:, 1] * dims[0] + ids[:, 0]


class CityGaussianFixedAllScene:
    """Official CityGaussian V1 runtime cells with every cell active.

    This intentionally does not implement visibility, adaptive LoD, streaming,
    caching, or the project's custom `BlockManager` partition format.
    """

    def __init__(
        self,
        path: Path,
        device: torch.device,
        *,
        block_dim: Sequence[int] = OFFICIAL_MATRIXCITY_BLOCK_DIM,
        aabb: Sequence[float] = OFFICIAL_MATRIXCITY_AABB,
        lod_source_level: str = "light_50_vq",
        mad_multiplier: float = OFFICIAL_MAD_MULTIPLIER,
        ply_coordinate_mode: Literal["native", "supersplat"] = "native",
        progress_callback: LoadProgressCallback | None = None,
    ):
        self.path = Path(path)
        self.device = device
        self.block_dim = tuple(int(v) for v in block_dim)
        self.aabb = tuple(float(v) for v in aabb)
        self.lod_source_level = str(lod_source_level)
        self.mad_multiplier = float(mad_multiplier)
        self.num_cells = int(math.prod(self.block_dim))
        self._last_metrics = SceneFrameMetrics()

        if progress_callback is not None:
            progress_callback("ply_parsing")
        raw = load_gaussian_ply(
            self.path,
            apply_supersplat_transform=ply_coordinate_mode == "supersplat",
            progress_callback=progress_callback,
        )
        self.total_gaussians = int(raw["means"].shape[0])
        self.steps = [-1]

        construction_start = time.perf_counter()
        activated = _activate_raw_splats(raw, device)
        self.flat_active = ActiveSplats(
            means=activated["means"],
            quats=activated["quats"],
            scales=activated["scales"],
            opacities=activated["opacities"],
            colors=activated["colors"],
        )
        self.cells = construct_citygaussian_cells(
            self.flat_active,
            block_dim=self.block_dim,
            aabb=self.aabb,
            mad_multiplier=self.mad_multiplier,
        )
        self.cell_construction_ms = (time.perf_counter() - construction_start) * 1000.0
        self.cell_counts = [cell.gaussian_count for cell in self.cells]
        self.cell_ids = [cell.cell_id for cell in self.cells]
        self.cell_average_scales = [float(cell.average_scale) for cell in self.cells]
        self.cell_mad_bounds = [
            {"min": cell.mad_bounds_min, "max": cell.mad_bounds_max} for cell in self.cells
        ]
        self.cell_corners = torch.stack([cell.corners for cell in self.cells], dim=0).to(
            device=self.device,
            dtype=torch.float32,
        )
        self.cell_source_indices = [
            cell.source_indices.to(device=self.device, dtype=torch.long) for cell in self.cells
        ]
        self.validation = validate_citygaussian_cells(self.flat_active, self.cells)
        self.concatenated_source_indices = self.validation.concatenated_source_indices.to(
            device=self.device,
            dtype=torch.long,
        )
        self.sh_degree = int(math.sqrt(self.flat_active.colors.shape[-2]) - 1)
        self.scene_min, self.scene_max = _bounds_from_cells(self.cells)

    def prepare_frame(self, camera: FrameCamera, render_settings: RenderSettings) -> ActiveSplats:
        del camera, render_settings
        tic = time.perf_counter()
        source_indices = self.concatenated_source_indices
        active = ActiveSplats(
            means=self.flat_active.means[source_indices],
            quats=self.flat_active.quats[source_indices],
            scales=self.flat_active.scales[source_indices],
            opacities=self.flat_active.opacities[source_indices],
            colors=self.flat_active.colors[source_indices],
        )
        concat_ms = (time.perf_counter() - tic) * 1000.0
        metrics = SceneFrameMetrics(
            active_blocks=float(self.num_cells),
            active_block_ids=",".join(str(cell_id) for cell_id in self.cell_ids),
            resident_blocks=float(self.num_cells),
            resident_gaussians=float(self.total_gaussians),
            resident_gaussian_fraction=1.0,
            active_gaussians=float(active.num_gaussians),
            active_gaussian_fraction=1.0,
            total_blocks=float(self.num_cells),
            total_cell_count=float(self.num_cells),
            active_cell_count=float(self.num_cells),
            selected_cell_count=float(self.num_cells),
            selected_cell_ids=",".join(str(cell_id) for cell_id in self.cell_ids),
            selected_gaussian_count=float(active.num_gaussians),
            visibility_selection_ms=0.0,
            selected_cell_assembly_ms=float(concat_ms),
            per_cell_gaussian_counts=list(self.cell_counts),
            per_cell_bounds=list(self.cell_mad_bounds),
            cell_construction_ms=float(self.cell_construction_ms),
            block_concat_ms=float(concat_ms),
            frame_preparation_ms=float(concat_ms),
            citygaussian_no_visibility_selection=True,
            citygaussian_visibility_selection_used=False,
            citygaussian_no_adaptive_lod=True,
            citygaussian_no_streaming=True,
        )
        active.scene_metrics = metrics
        self._last_metrics = metrics
        return active

    def get_scene_stats(self) -> SceneStats:
        return SceneStats(
            scene_type="citygaussian_fixed_all",
            total_gaussians=self.total_gaussians,
            sh_degree=self.sh_degree,
            paths=[self.path],
            scene_aabb_min=self.scene_min,
            scene_aabb_max=self.scene_max,
            total_blocks=float(self.num_cells),
            steps=self.steps,
            citygaussian_lod_source_level=self.lod_source_level,
            citygaussian_block_dim=self.block_dim,
            citygaussian_aabb=self.aabb,
            citygaussian_cell_count=self.num_cells,
            citygaussian_per_cell_counts=self.cell_counts,
            citygaussian_cell_construction_ms=self.cell_construction_ms,
        )

    def get_metrics(self) -> SceneFrameMetrics:
        return self._last_metrics

    def get_cell_metadata(self) -> dict[str, object]:
        return {
            "block_dim": list(self.block_dim),
            "aabb": list(self.aabb),
            "cell_ids": list(self.cell_ids),
            "per_cell_gaussian_counts": list(self.cell_counts),
            "total_assigned_count": int(self.validation.assigned_total),
            "unassigned_count": int(self.validation.unassigned_count),
            "duplicated_assignment_count": int(self.validation.duplicated_assignment_count),
            "all_cells_nonempty": bool(self.validation.all_cells_nonempty),
            "average_scales": list(self.cell_average_scales),
            "mad_bounds": self.cell_mad_bounds,
        }


class CityGaussianVisibilityScene(CityGaussianFixedAllScene):
    """Official CityGaussian V1 runtime cells with projected-corner visibility.

    All Gaussian tensors stay resident in `flat_active`; visibility only
    controls the selected cells assembled for the current frame. This is not
    streaming, host residency, or adaptive LoD.
    """

    def __init__(
        self,
        path: Path,
        device: torch.device,
        *,
        block_dim: Sequence[int] = OFFICIAL_MATRIXCITY_BLOCK_DIM,
        aabb: Sequence[float] = OFFICIAL_MATRIXCITY_AABB,
        lod_source_level: str = "light_50_vq",
        mad_multiplier: float = OFFICIAL_MAD_MULTIPLIER,
        ply_coordinate_mode: Literal["native", "supersplat"] = "native",
        visibility_znear: float = 0.01,
        visibility_zfar: float = 100.0,
        progress_callback: LoadProgressCallback | None = None,
    ):
        super().__init__(
            path,
            device,
            block_dim=block_dim,
            aabb=aabb,
            lod_source_level=lod_source_level,
            mad_multiplier=mad_multiplier,
            ply_coordinate_mode=ply_coordinate_mode,
            progress_callback=progress_callback,
        )
        self.visibility_znear = float(visibility_znear)
        self.visibility_zfar = float(visibility_zfar)

    def prepare_frame(self, camera: FrameCamera, render_settings: RenderSettings) -> ActiveSplats:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        selection_start = time.perf_counter()
        official_camera = citygaussian_official_camera_from_frame(
            camera.to(self.device),
            render_settings,
            znear=self.visibility_znear,
            zfar=self.visibility_zfar,
        )
        selection = citygaussian_in_frustum(
            official_camera,
            self.cell_corners,
            self.cell_counts,
            aabb=self.aabb,
            block_dim=self.block_dim,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        selection_ms = (time.perf_counter() - selection_start) * 1000.0

        assembly_start = time.perf_counter()
        selected_indices = [self.cell_source_indices[cell_id] for cell_id in selection.selected_cell_ids]
        source_indices = (
            torch.cat(selected_indices, dim=0)
            if selected_indices
            else torch.empty(0, dtype=torch.long, device=self.device)
        )
        active = ActiveSplats(
            means=self.flat_active.means[source_indices],
            quats=self.flat_active.quats[source_indices],
            scales=self.flat_active.scales[source_indices],
            opacities=self.flat_active.opacities[source_indices],
            colors=self.flat_active.colors[source_indices],
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        assembly_ms = (time.perf_counter() - assembly_start) * 1000.0
        frame_preparation_ms = selection_ms + assembly_ms
        selected_ids = ",".join(str(cell_id) for cell_id in selection.selected_cell_ids)
        metrics = SceneFrameMetrics(
            active_blocks=float(selection.selected_cell_count),
            active_block_ids=selected_ids,
            resident_blocks=float(self.num_cells),
            resident_gaussians=float(self.total_gaussians),
            resident_gaussian_fraction=1.0,
            active_gaussians=float(active.num_gaussians),
            active_gaussian_fraction=float(active.num_gaussians) / max(float(self.total_gaussians), 1.0),
            total_blocks=float(self.num_cells),
            total_cell_count=float(self.num_cells),
            active_cell_count=float(selection.selected_cell_count),
            selected_cell_count=float(selection.selected_cell_count),
            selected_cell_ids=selected_ids,
            selected_gaussian_count=float(selection.selected_gaussian_count),
            visibility_selection_ms=float(selection_ms),
            selected_cell_assembly_ms=float(assembly_ms),
            nearest_cell_id=float(selection.nearest_cell_id),
            nearest_cell_fallback_changed_mask=bool(selection.nearest_cell_fallback_changed_mask),
            per_cell_gaussian_counts=list(self.cell_counts),
            per_cell_bounds=list(self.cell_mad_bounds),
            per_cell_distances=list(selection.per_cell_distances),
            cell_construction_ms=float(self.cell_construction_ms),
            block_cull_ms=float(selection_ms),
            block_concat_ms=float(assembly_ms),
            frame_preparation_ms=float(frame_preparation_ms),
            citygaussian_no_visibility_selection=False,
            citygaussian_visibility_selection_used=True,
            citygaussian_no_adaptive_lod=True,
            citygaussian_no_streaming=True,
        )
        active.scene_metrics = metrics
        self._last_metrics = metrics
        return active

    def get_scene_stats(self) -> SceneStats:
        stats = super().get_scene_stats()
        stats.scene_type = "citygaussian_visibility"
        return stats


class CityGaussianReleaseLodScene:
    """Released CityGaussian V1 adaptive LoD with all three levels resident."""

    def __init__(
        self,
        paths: Sequence[Path],
        device: torch.device,
        *,
        block_dim: Sequence[int] = OFFICIAL_MATRIXCITY_BLOCK_DIM,
        aabb: Sequence[float] = OFFICIAL_MATRIXCITY_AABB,
        level_labels: Sequence[str] = RELEASE_LOD_LEVEL_LABELS,
        level_semantics: Sequence[str] = RELEASE_LOD_LEVEL_SEMANTICS,
        mad_multiplier: float = OFFICIAL_MAD_MULTIPLIER,
        ply_coordinate_mode: Literal["native", "supersplat"] = "native",
        visibility_znear: float = 0.01,
        visibility_zfar: float = 100.0,
        progress_callback: LoadProgressCallback | None = None,
    ):
        self.paths = [Path(path) for path in paths]
        self.device = device
        self.block_dim = tuple(int(v) for v in block_dim)
        self.aabb = tuple(float(v) for v in aabb)
        self.level_labels = tuple(str(value) for value in level_labels)
        self.level_semantics = tuple(str(value) for value in level_semantics)
        self.mad_multiplier = float(mad_multiplier)
        self.visibility_znear = float(visibility_znear)
        self.visibility_zfar = float(visibility_zfar)
        self.num_cells = int(math.prod(self.block_dim))
        self._last_metrics = SceneFrameMetrics()
        self.steps = [-1]

        if len(self.paths) != len(RELEASE_LOD_LEVEL_LABELS):
            raise ValueError(
                f"{RELEASE_LOD_POLICY} requires exactly {len(RELEASE_LOD_LEVEL_LABELS)} PLY paths "
                "in finest, middle, coarsest order."
            )
        if self.level_labels != RELEASE_LOD_LEVEL_LABELS:
            raise ValueError(
                f"{RELEASE_LOD_POLICY} level order must be {list(RELEASE_LOD_LEVEL_LABELS)}, "
                f"got {list(self.level_labels)}."
            )
        if len(self.level_semantics) != len(self.level_labels):
            raise ValueError("LoD level labels and semantics must have equal lengths.")

        initialization_start = time.perf_counter()
        self.level_scenes = [
            CityGaussianFixedAllScene(
                path,
                device=device,
                block_dim=self.block_dim,
                aabb=self.aabb,
                lod_source_level=label,
                mad_multiplier=self.mad_multiplier,
                ply_coordinate_mode=ply_coordinate_mode,
                progress_callback=progress_callback,
            )
            for path, label in zip(self.paths, self.level_labels)
        ]
        self.initialization_ms = (time.perf_counter() - initialization_start) * 1000.0
        self.cell_construction_ms = float(
            sum(level.cell_construction_ms for level in self.level_scenes)
        )
        sh_degrees = [int(level.sh_degree) for level in self.level_scenes]
        if len(set(sh_degrees)) != 1:
            raise ValueError(f"All released LoD levels must have the same SH degree, got {sh_degrees}.")
        if any(level.cell_ids != list(range(self.num_cells)) for level in self.level_scenes):
            raise ValueError("Every released LoD level must construct the same complete logical cell ID set.")
        if any(not level.validation.every_gaussian_assigned_once for level in self.level_scenes):
            raise ValueError("Every released LoD Gaussian must be assigned to exactly one official cell.")
        if any(not level.validation.all_cells_nonempty for level in self.level_scenes):
            raise ValueError("Every official cell must be nonempty at every released LoD level.")

        self.sh_degree = sh_degrees[0]
        self.level_gaussian_counts = [int(level.total_gaussians) for level in self.level_scenes]
        self.resident_gaussian_count = int(sum(self.level_gaussian_counts))
        self.total_gaussians = self.resident_gaussian_count
        self.level_cell_counts = [list(level.cell_counts) for level in self.level_scenes]
        self.level_average_scales = [list(level.cell_average_scales) for level in self.level_scenes]
        self.level_cell_construction_ms = [float(level.cell_construction_ms) for level in self.level_scenes]

        coarsest = self.level_scenes[-1]
        self.visibility_level_index = len(self.level_scenes) - 1
        self.visibility_bounds_level = self.level_labels[self.visibility_level_index]
        self.cell_ids = list(coarsest.cell_ids)
        self.cell_counts = list(coarsest.cell_counts)
        self.cell_corners = coarsest.cell_corners
        self.cell_mad_bounds = list(coarsest.cell_mad_bounds)
        self.scene_min = coarsest.scene_min
        self.scene_max = coarsest.scene_max

    def prepare_frame(self, camera: FrameCamera, render_settings: RenderSettings) -> ActiveSplats:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        visibility_start = time.perf_counter()
        official_camera = citygaussian_official_camera_from_frame(
            camera.to(self.device), render_settings,
            znear=self.visibility_znear, zfar=self.visibility_zfar,
        )
        visibility = citygaussian_in_frustum(
            official_camera, self.cell_corners, self.cell_counts,
            aabb=self.aabb, block_dim=self.block_dim,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        visibility_ms = (time.perf_counter() - visibility_start) * 1000.0

        lod_start = time.perf_counter()
        lod_selection = select_citygaussian_release_lod(
            visible_cell_ids=visibility.selected_cell_ids,
            per_visible_cell_distances=visibility.per_visible_cell_distances,
            average_scales_by_level=self.level_average_scales,
            cell_counts_by_level=self.level_cell_counts,
            level_labels=self.level_labels,
            image_width=official_camera.width,
            fov_x=official_camera.fov_x,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        lod_selection_ms = (time.perf_counter() - lod_start) * 1000.0

        assembly_start = time.perf_counter()
        attribute_chunks: dict[str, list[Tensor]] = {
            "means": [], "quats": [], "scales": [], "opacities": [], "colors": []
        }
        for level_index, level in enumerate(self.level_scenes):
            for cell_id in visibility.selected_cell_ids:
                if lod_selection.selected_lod_by_cell[cell_id] != level_index:
                    continue
                source_indices = level.cell_source_indices[cell_id]
                for attribute in attribute_chunks:
                    attribute_chunks[attribute].append(getattr(level.flat_active, attribute)[source_indices])
        active = ActiveSplats(
            means=torch.cat(attribute_chunks["means"], dim=0),
            quats=torch.cat(attribute_chunks["quats"], dim=0),
            scales=torch.cat(attribute_chunks["scales"], dim=0),
            opacities=torch.cat(attribute_chunks["opacities"], dim=0),
            colors=torch.cat(attribute_chunks["colors"], dim=0),
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        assembly_ms = (time.perf_counter() - assembly_start) * 1000.0
        if active.num_gaussians != lod_selection.total_selected_gaussian_count:
            raise RuntimeError(
                f"Released LoD assembly count mismatch: {active.num_gaussians} "
                f"!= {lod_selection.total_selected_gaussian_count}."
            )

        selected_ids = ",".join(str(cell_id) for cell_id in visibility.selected_cell_ids)
        metrics = SceneFrameMetrics(
            active_blocks=float(visibility.selected_cell_count),
            active_block_ids=selected_ids,
            resident_blocks=float(self.num_cells * len(self.level_scenes)),
            resident_gaussians=float(self.resident_gaussian_count),
            resident_gaussian_fraction=1.0,
            active_gaussians=float(active.num_gaussians),
            active_gaussian_fraction=float(active.num_gaussians) / max(float(self.resident_gaussian_count), 1.0),
            total_blocks=float(self.num_cells),
            total_cell_count=float(self.num_cells),
            active_cell_count=float(visibility.selected_cell_count),
            selected_cell_count=float(visibility.selected_cell_count),
            selected_cell_ids=selected_ids,
            selected_gaussian_count=float(lod_selection.total_selected_gaussian_count),
            selected_lod_by_cell=dict(lod_selection.selected_lod_by_cell),
            selected_level_label_by_cell=dict(lod_selection.selected_level_label_by_cell),
            selected_gaussian_count_by_cell=dict(lod_selection.selected_gaussian_count_by_cell),
            selected_cell_count_by_level=list(lod_selection.selected_cell_count_by_level),
            selected_gaussian_count_by_level=list(lod_selection.selected_gaussian_count_by_level),
            visibility_selection_ms=float(visibility_ms),
            lod_selection_ms=float(lod_selection_ms),
            selected_cell_assembly_ms=float(assembly_ms),
            nearest_cell_id=float(visibility.nearest_cell_id),
            nearest_cell_fallback_changed_mask=bool(visibility.nearest_cell_fallback_changed_mask),
            per_cell_gaussian_counts=list(self.cell_counts),
            per_cell_bounds=list(self.cell_mad_bounds),
            per_cell_distances=list(visibility.per_cell_distances),
            cell_construction_ms=float(self.cell_construction_ms),
            block_cull_ms=float(visibility_ms),
            block_concat_ms=float(assembly_ms),
            frame_preparation_ms=float(visibility_ms + lod_selection_ms + assembly_ms),
            citygaussian_no_visibility_selection=False,
            citygaussian_visibility_selection_used=True,
            citygaussian_no_adaptive_lod=False,
            citygaussian_adaptive_lod_used=True,
            citygaussian_lod_policy=RELEASE_LOD_POLICY,
            citygaussian_visibility_bounds_level=self.visibility_bounds_level,
            citygaussian_no_vq_reconstruction=True,
            citygaussian_no_streaming=True,
        )
        active.scene_metrics = metrics
        self._last_metrics = metrics
        return active

    def get_scene_stats(self) -> SceneStats:
        return SceneStats(
            scene_type="citygaussian_lod_release",
            total_gaussians=self.total_gaussians,
            sh_degree=self.sh_degree,
            paths=list(self.paths),
            scene_aabb_min=self.scene_min,
            scene_aabb_max=self.scene_max,
            total_blocks=float(self.num_cells),
            steps=self.steps,
            citygaussian_lod_source_level=None,
            citygaussian_block_dim=self.block_dim,
            citygaussian_aabb=self.aabb,
            citygaussian_cell_count=self.num_cells,
            citygaussian_per_cell_counts=list(self.cell_counts),
            citygaussian_cell_construction_ms=self.cell_construction_ms,
            citygaussian_lod_policy=RELEASE_LOD_POLICY,
            citygaussian_lod_level_labels=list(self.level_labels),
            citygaussian_lod_level_semantics=list(self.level_semantics),
            citygaussian_lod_level_gaussian_counts=list(self.level_gaussian_counts),
            citygaussian_lod_level_paths=list(self.paths),
            citygaussian_visibility_bounds_level=self.visibility_bounds_level,
        )

    def get_metrics(self) -> SceneFrameMetrics:
        return self._last_metrics

    def get_cell_metadata(self) -> dict[str, object]:
        return {
            "policy": RELEASE_LOD_POLICY,
            "block_dim": list(self.block_dim),
            "aabb": list(self.aabb),
            "level_labels": list(self.level_labels),
            "level_semantics": list(self.level_semantics),
            "level_gaussian_counts": list(self.level_gaussian_counts),
            "per_level_cell_gaussian_counts": [list(counts) for counts in self.level_cell_counts],
            "per_level_average_scales": [list(values) for values in self.level_average_scales],
            "per_level_cell_construction_ms": list(self.level_cell_construction_ms),
            "initialization_ms": float(self.initialization_ms),
            "visibility_bounds_level_index": int(self.visibility_level_index),
            "visibility_bounds_level": self.visibility_bounds_level,
            "visibility_mad_bounds": list(self.cell_mad_bounds),
            "resident_gaussian_count": int(self.resident_gaussian_count),
            "vq_reconstruction_used": False,
            "streaming_used": False,
        }


def construct_citygaussian_cells(
    flat_active: ActiveSplats,
    *,
    block_dim: Sequence[int] = OFFICIAL_MATRIXCITY_BLOCK_DIM,
    aabb: Sequence[float] = OFFICIAL_MATRIXCITY_AABB,
    mad_multiplier: float = OFFICIAL_MAD_MULTIPLIER,
) -> list[CityGaussianCell]:
    num_cells = int(math.prod([int(v) for v in block_dim]))
    xyz = flat_active.means
    cells: list[CityGaussianCell] = []
    all_indices = torch.arange(xyz.shape[0], dtype=torch.long, device=xyz.device)
    for cell_id in range(num_cells):
        mask = official_cell_mask(cell_id, xyz, block_dim=block_dim, aabb=aabb)
        indices = all_indices[mask]
        if indices.numel() == 0:
            bounds_min = bounds_max = (float("nan"), float("nan"), float("nan"))
            corners = torch.empty((8, 3), dtype=xyz.dtype, device=xyz.device)
            avg_scale = float("nan")
        else:
            cell_xyz = xyz[mask]
            xyz_min, xyz_max = official_mad_bounds(cell_xyz, mad_multiplier)
            corners = bounds_to_corners(xyz_min, xyz_max)
            bounds_min = tuple(float(v) for v in xyz_min.detach().cpu().tolist())
            bounds_max = tuple(float(v) for v in xyz_max.detach().cpu().tolist())
            avg_scale = float(torch.mean(flat_active.scales[mask], dim=0).max().detach().cpu().item())
        cells.append(
            CityGaussianCell(
                cell_id=cell_id,
                grid_index=cell_grid_index(cell_id, block_dim),
                source_indices=indices.detach().cpu(),
                mad_bounds_min=bounds_min,
                mad_bounds_max=bounds_max,
                corners=corners.detach().cpu(),
                average_scale=avg_scale,
            )
        )
    return cells


def official_mad_bounds(xyz: Tensor, mad_multiplier: float = OFFICIAL_MAD_MULTIPLIER) -> tuple[Tensor, Tensor]:
    if xyz.numel() == 0:
        raise ValueError("Cannot compute MAD bounds for an empty cell.")
    xyz_median = torch.median(xyz, dim=0).values
    delta_median = torch.median(torch.abs(xyz - xyz_median), dim=0).values
    xyz_min = xyz_median - float(mad_multiplier) * delta_median
    xyz_min = torch.max(xyz_min, torch.min(xyz, dim=0).values)
    xyz_max = xyz_median + float(mad_multiplier) * delta_median
    xyz_max = torch.min(xyz_max, torch.max(xyz, dim=0).values)
    return xyz_min, xyz_max


def bounds_to_corners(xyz_min: Tensor, xyz_max: Tensor) -> Tensor:
    return torch.stack(
        [
            torch.stack([xyz_min[0], xyz_min[1], xyz_min[2]]),
            torch.stack([xyz_min[0], xyz_min[1], xyz_max[2]]),
            torch.stack([xyz_min[0], xyz_max[1], xyz_min[2]]),
            torch.stack([xyz_min[0], xyz_max[1], xyz_max[2]]),
            torch.stack([xyz_max[0], xyz_min[1], xyz_min[2]]),
            torch.stack([xyz_max[0], xyz_min[1], xyz_max[2]]),
            torch.stack([xyz_max[0], xyz_max[1], xyz_min[2]]),
            torch.stack([xyz_max[0], xyz_max[1], xyz_max[2]]),
        ],
        dim=0,
    )


def validate_citygaussian_cells(
    flat_active: ActiveSplats,
    cells: Sequence[CityGaussianCell],
) -> CityGaussianCellValidation:
    total = flat_active.num_gaussians
    assignment_counts = torch.zeros(total, dtype=torch.int64)
    source_parts = []
    for cell in cells:
        indices = cell.source_indices.to(dtype=torch.long, device="cpu")
        if indices.numel() > 0:
            assignment_counts.index_add_(0, indices, torch.ones_like(indices, dtype=torch.int64))
            source_parts.append(indices)
    concatenated = torch.cat(source_parts, dim=0) if source_parts else torch.empty(0, dtype=torch.long)
    unassigned = int((assignment_counts == 0).sum().item())
    duplicated = int((assignment_counts > 1).sum().item())
    assigned_total = int(assignment_counts.sum().item())
    every_once = bool(unassigned == 0 and duplicated == 0 and assigned_total == total)
    all_nonempty = bool(all(cell.gaussian_count > 0 for cell in cells))
    equivalence = bool(every_once and torch.equal(torch.sort(concatenated).values, torch.arange(total)))
    return CityGaussianCellValidation(
        total_gaussians=total,
        assigned_total=assigned_total,
        unassigned_count=unassigned,
        duplicated_assignment_count=duplicated,
        every_gaussian_assigned_once=every_once,
        all_cells_nonempty=all_nonempty,
        attribute_equivalence=equivalence,
        concatenated_source_indices=concatenated,
    )


def compare_flat_and_cell_attributes(flat_active: ActiveSplats, cell_active: ActiveSplats, source_indices: Tensor) -> bool:
    source_indices = source_indices.to(device=flat_active.means.device, dtype=torch.long)
    return all(
        torch.equal(getattr(cell_active, name), getattr(flat_active, name)[source_indices])
        for name in ("means", "quats", "scales", "opacities", "colors")
    )


def _activate_raw_splats(raw: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {
        "means": raw["means"].to(device=device, dtype=torch.float32).contiguous(),
        "quats": F.normalize(
            raw["quats"].to(device=device, dtype=torch.float32), p=2, dim=-1
        ).contiguous(),
        "scales": torch.exp(raw["scales"].to(device=device, dtype=torch.float32)).contiguous(),
        "opacities": torch.sigmoid(raw["opacities"].to(device=device, dtype=torch.float32)).view(-1).contiguous(),
        "colors": torch.cat(
            [
                raw["sh0"].to(device=device, dtype=torch.float32),
                raw["shN"].to(device=device, dtype=torch.float32),
            ],
            dim=-2,
        ).contiguous(),
    }


def _bounds_from_cells(cells: Iterable[CityGaussianCell]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mins = []
    maxs = []
    for cell in cells:
        if cell.gaussian_count == 0:
            continue
        mins.append(torch.tensor(cell.mad_bounds_min, dtype=torch.float32))
        maxs.append(torch.tensor(cell.mad_bounds_max, dtype=torch.float32))
    if not mins:
        nan = float("nan")
        return (nan, nan, nan), (nan, nan, nan)
    scene_min = torch.stack(mins, dim=0).min(dim=0).values
    scene_max = torch.stack(maxs, dim=0).max(dim=0).values
    return tuple(float(v) for v in scene_min.tolist()), tuple(float(v) for v in scene_max.tolist())
