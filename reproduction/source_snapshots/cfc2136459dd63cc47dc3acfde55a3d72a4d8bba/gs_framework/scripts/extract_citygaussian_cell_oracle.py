from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from citygaussian_assets import (  # noqa: E402
    REFERENCE_COMMIT,
    compute_citygaussian_cell_counts,
    contract_to_unisphere_np,
    read_ply_header,
)
from scene_runtime.citygaussian import (  # noqa: E402
    citygaussian_official_camera_from_c2w,
    official_projection_matrix,
)


DEFAULT_BLOCK_DIM = [6, 6, 1]
DEFAULT_AABB = [-3.5, -4.0, -10.0, 4.5, 2.0, 10.0]
DEFAULT_CAMERA_PATH = ROOT / "benchmarks" / "camera_paths" / "citygaussian_23m_navigation_v1.json"
DEFAULT_REFERENCE_REPO = Path("/home/luis/tese/CityGaussian")
DEFAULT_PRECISION = 8


@dataclass(frozen=True)
class OracleCamera:
    trajectory_index: int
    camera_id: str
    source_camera_to_world: np.ndarray
    converted_camera_to_world: np.ndarray
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    projection_matrix: torch.Tensor
    camera_center: torch.Tensor
    fov_x: float
    fov_y: float
    width: int
    height: int
    source_fov_degrees: float


@dataclass(frozen=True)
class OfficialVisibilityFunction:
    fn: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    method: str
    source_path: str
    import_error: str | None = None
    source_slice_error: str | None = None


def _round_float(value: Any, precision: int = DEFAULT_PRECISION) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value
        return round(value, precision)
    if isinstance(value, np.generic):
        return _round_float(value.item(), precision)
    if isinstance(value, torch.Tensor):
        return _round_float(value.detach().cpu().tolist(), precision)
    if isinstance(value, dict):
        return {str(k): _round_float(v, precision) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_float(v, precision) for v in value]
    return value


def _sha256_json(value: Any, precision: int = DEFAULT_PRECISION) -> str:
    canonical = json.dumps(_round_float(value, precision), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _vertical_fov_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = 0.5 * float(height) / math.tan(math.radians(float(fov_deg)) * 0.5)
    return np.asarray(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _viser_export_to_project_c2w(c2w: np.ndarray, scale_ratio: float) -> np.ndarray:
    converted = c2w.astype(np.float32, copy=True)
    converted[:3, 1:3] *= -1.0
    converted[:3, 3] *= float(scale_ratio)
    return converted


def _projection_matrix(znear: float, zfar: float, fov_x: float, fov_y: float) -> torch.Tensor:
    return official_projection_matrix(znear, zfar, fov_x, fov_y).T.contiguous()


def _camera_to_official(
    *,
    trajectory_index: int,
    camera_id: str,
    source_c2w: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    scale_ratio: float,
    znear: float,
    zfar: float,
) -> OracleCamera:
    if source_c2w.shape != (4, 4):
        raise ValueError(f"Camera {camera_id} matrix must be 4x4, got {source_c2w.shape}.")
    if not np.isfinite(source_c2w).all():
        raise ValueError(f"Camera {camera_id} has non-finite matrix values.")
    converted_c2w = _viser_export_to_project_c2w(source_c2w, scale_ratio)
    k = _vertical_fov_intrinsics(width, height, fov_deg)
    runtime_camera = citygaussian_official_camera_from_c2w(
        torch.from_numpy(converted_c2w),
        torch.from_numpy(k),
        width=width,
        height=height,
        znear=znear,
        zfar=zfar,
    )
    return OracleCamera(
        trajectory_index=trajectory_index,
        camera_id=camera_id,
        source_camera_to_world=source_c2w.astype(np.float32),
        converted_camera_to_world=converted_c2w,
        world_view_transform=runtime_camera.world_view_transform,
        full_proj_transform=runtime_camera.full_proj_transform,
        projection_matrix=runtime_camera.projection_matrix,
        camera_center=runtime_camera.camera_center,
        fov_x=runtime_camera.fov_x,
        fov_y=runtime_camera.fov_y,
        width=int(width),
        height=int(height),
        source_fov_degrees=float(fov_deg),
    )


def sample_camera_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if sample_count >= frame_count:
        return list(range(frame_count))
    raw = np.linspace(0, frame_count - 1, num=sample_count)
    indices = sorted({int(round(value)) for value in raw})
    candidate = 0
    while len(indices) < sample_count and candidate < frame_count:
        if candidate not in indices:
            indices.append(candidate)
        candidate += 1
    return sorted(indices)


def load_oracle_cameras(
    camera_path: Path,
    *,
    sample_count: int,
    znear: float,
    zfar: float,
) -> tuple[list[OracleCamera], dict[str, Any]]:
    data = json.loads(Path(camera_path).read_text())
    if not isinstance(data, dict) or not isinstance(data.get("camera_path"), list):
        raise ValueError("Expected a Viser camera path JSON with a camera_path list.")
    records = data["camera_path"]
    width = int(data.get("render_width", data.get("width", 1920)))
    height = int(data.get("render_height", data.get("height", 1080)))
    default_fov = float(data.get("default_fov", data.get("fov_deg", 50.0)))
    scale_ratio = float(data.get("scale_ratio", 10.0))
    indices = sample_camera_indices(len(records), sample_count)
    cameras: list[OracleCamera] = []
    for idx in indices:
        record = records[idx]
        source = np.asarray(record["camera_to_world"], dtype=np.float32).reshape(4, 4)
        fov_deg = float(record.get("fov", default_fov))
        cameras.append(
            _camera_to_official(
                trajectory_index=idx,
                camera_id=f"viser_{idx:06d}",
                source_c2w=source,
                width=int(record.get("width", width)),
                height=int(record.get("height", height)),
                fov_deg=fov_deg,
                scale_ratio=scale_ratio,
                znear=znear,
                zfar=zfar,
            )
        )
    return cameras, {
        "camera_path": str(camera_path),
        "trajectory_frame_count": len(records),
        "selected_indices": indices,
        "selection_policy": (
            f"np.linspace over [0, {len(records) - 1}] with sample_count={sample_count}, "
            "rounded to nearest integer, deduplicated, and sorted; includes first and final frames."
        ),
        "source_format": "Viser camera path export",
        "source_matrix": "camera_to_world, row-major JSON, reshaped to 4x4",
        "project_conversion": "flip camera Y/Z axes and multiply translation by scale_ratio",
        "scale_ratio": scale_ratio,
        "intrinsics": "project benchmark convention: JSON fov is vertical FOV; derive K, FoVx and FoVy from width/height/focal",
        "official_matrix_convention": (
            "world_view_transform = inverse(converted camera_to_world).T; "
            "projection_matrix matches CityGaussian utils.graphics_utils.getProjectionMatrix(...).T; "
            "full_proj_transform = world_view_transform @ projection_matrix for row-vector homogeneous corners"
        ),
        "front_convention": "official in_frustum keeps corners with camera-space z > 0.2",
        "resolution": [width, height],
        "default_fov_degrees": default_fov,
        "znear": znear,
        "zfar": zfar,
    }


def load_official_visibility_function(reference_repo: Path) -> OfficialVisibilityFunction:
    source_path = reference_repo / "utils" / "large_utils.py"
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(reference_repo))
        importlib.invalidate_caches()
        try:
            module = importlib.import_module("utils.large_utils")
            return OfficialVisibilityFunction(
                fn=module.in_frustum,
                method="direct_import",
                source_path=str(source_path),
            )
        except Exception as exc:
            import_error = f"{type(exc).__name__}: {exc}"
    finally:
        sys.path[:] = original_path
    try:
        namespace = _load_in_frustum_source_slice(source_path)
        return OfficialVisibilityFunction(
            fn=namespace["in_frustum"],
            method="source_slice_exec",
            source_path=str(source_path),
            import_error=import_error,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not import or source-slice official in_frustum from {source_path}: {type(exc).__name__}: {exc}"
        ) from exc


def _load_in_frustum_source_slice(source_path: Path) -> dict[str, Any]:
    source = Path(source_path).read_text()
    lines = []
    keep = False
    for line in source.splitlines():
        if line.startswith("def in_frustum("):
            keep = True
        if keep and line.startswith("def get_default_aabb("):
            break
        if keep:
            lines.append(line)
    if not lines:
        raise RuntimeError(f"Could not find in_frustum source slice in {source_path}")
    namespace: dict[str, Any] = {"torch": torch, "np": np}
    exec("\n".join(lines), namespace)
    return namespace


def _official_reference_probe(reference_repo: Path, visibility: OfficialVisibilityFunction) -> dict[str, Any]:
    probe: dict[str, Any] = {
        "reference_repo": str(reference_repo),
        "expected_commit": REFERENCE_COMMIT,
        "visibility_function": "utils/large_utils.py::in_frustum",
        "visibility_function_source": visibility.source_path,
        "visibility_load_method": visibility.method,
        "large_utils_import_error": visibility.import_error,
        "official_contract_sample_matches": None,
    }
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(reference_repo))
        try:
            from utils.large_utils import contract_to_unisphere  # type: ignore

            sample = torch.tensor(
                [[-3.5, -4.0, -10.0], [0.5, -1.0, 0.0], [4.5, 2.0, 10.0]],
                dtype=torch.float32,
            )
            aabb = torch.tensor(DEFAULT_AABB, dtype=torch.float32)
            official = contract_to_unisphere(sample.clone(), aabb, ord=torch.inf).cpu().numpy()
            local = contract_to_unisphere_np(sample.cpu().numpy(), DEFAULT_AABB)
            probe["official_contract_sample_matches"] = bool(abs(official - local).max() <= 1e-7)
        except Exception as exc:
            probe["official_contract_probe_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        sys.path[:] = original_path
    return probe


def _bounds_to_corners(bounds_min: Sequence[float], bounds_max: Sequence[float]) -> list[list[float]]:
    mn = [float(v) for v in bounds_min]
    mx = [float(v) for v in bounds_max]
    return [
        [mn[0], mn[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mx[1], mx[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mn[2]],
        [mx[0], mx[1], mx[2]],
    ]


def corners_from_metadata(metadata: dict[str, Any]) -> torch.Tensor:
    corners: list[list[list[float]]] = []
    for bound in metadata["mad_bounds"]:
        if bound["min"] is None or bound["max"] is None:
            raise ValueError(f"Cell {bound['cell_id']} is empty; cannot run visibility oracle.")
        corners.append(_bounds_to_corners(bound["min"], bound["max"]))
    return torch.tensor(corners, dtype=torch.float32)


def _visibility_debug(camera: OracleCamera, cell_corners: torch.Tensor) -> dict[str, Any]:
    num_cell = int(cell_corners.shape[0])
    corners_h = torch.cat([cell_corners, torch.ones_like(cell_corners[..., [0]])], dim=-1)
    full_proj = camera.full_proj_transform.repeat(num_cell, 1, 1)
    view = camera.world_view_transform.repeat(num_cell, 1, 1)
    corners_screen = corners_h.bmm(full_proj)
    corners_screen = corners_screen / corners_screen[..., [-1]]
    corners_screen = corners_screen[..., :-1].reshape(-1, 3)

    corners_cam = corners_h.bmm(view)
    distances = torch.norm(corners_cam[:, :, :3], dim=-1)
    distance_min = torch.min(distances, dim=-1).values
    nearest_cell_id = int(torch.argmin(distance_min).item())
    front_mask = corners_cam[..., 2] > 0.2
    flat_front = front_mask.reshape(-1)

    screen_min_work = corners_screen.clone().reshape(-1, 3)
    screen_min_work[~flat_front] = torch.inf
    screen_min = screen_min_work.reshape(num_cell, -1, 3).min(dim=1).values
    screen_min[screen_min == torch.inf] = 0.0

    screen_max_work = corners_screen.clone().reshape(-1, 3)
    screen_max_work[~flat_front] = -torch.inf
    screen_max = screen_max_work.reshape(num_cell, -1, 3).max(dim=1).values
    screen_max[screen_max == -torch.inf] = 0.0

    boxes = torch.cat([screen_min[:, :2], screen_max[:, :2]], dim=1)
    screen = torch.tensor([[-1.0, -1.0, 1.0, 1.0]], dtype=torch.float32)
    max_xy = torch.min(boxes[:, 2:].unsqueeze(1).expand(num_cell, 1, 2), screen[:, 2:].unsqueeze(0))
    min_xy = torch.max(boxes[:, :2].unsqueeze(1).expand(num_cell, 1, 2), screen[:, :2].unsqueeze(0))
    inter = torch.clamp(max_xy - min_xy, min=0.0)
    pre_fallback_mask = (inter[:, 0, 0] * inter[:, 0, 1]) > 0.0
    return {
        "distance_min": distance_min,
        "nearest_cell_id": nearest_cell_id,
        "front_corner_counts": front_mask.sum(dim=1).to(dtype=torch.int64),
        "projected_boxes": boxes,
        "pre_fallback_mask": pre_fallback_mask,
    }


def run_visibility_oracle(
    *,
    visibility: OfficialVisibilityFunction,
    cameras: Sequence[OracleCamera],
    cell_metadata: dict[str, Any],
    block_dim: list[int],
    aabb: list[float],
    precision: int,
) -> list[dict[str, Any]]:
    cell_corners = corners_from_metadata(cell_metadata)
    cell_counts = [int(v) for v in cell_metadata["cell_gaussian_counts"]]
    expected_cell_count = int(np.prod(np.asarray(block_dim, dtype=np.int64)))
    declared_cell_count = int(cell_metadata["cell_count"])
    if declared_cell_count != expected_cell_count:
        raise ValueError(
            f"Cell metadata declares {declared_cell_count} cells, but block_dim {block_dim} implies "
            f"{expected_cell_count} cells."
        )
    if int(cell_corners.shape[0]) != expected_cell_count:
        raise ValueError(
            f"Cell corner count {int(cell_corners.shape[0])} does not match expected {expected_cell_count}."
        )
    if len(cell_counts) != expected_cell_count:
        raise ValueError(
            f"Cell Gaussian count length {len(cell_counts)} does not match expected {expected_cell_count}."
        )
    records = []
    for camera in cameras:
        official_camera = SimpleNamespace(
            full_proj_transform=camera.full_proj_transform,
            world_view_transform=camera.world_view_transform,
            FoVx=camera.fov_x,
            FoVy=camera.fov_y,
            image_width=camera.width,
            image_height=camera.height,
            camera_center=camera.camera_center,
        )
        with torch.no_grad():
            official_mask, official_distances = visibility.fn(
                official_camera,
                cell_corners.clone(),
                list(aabb),
                list(block_dim),
            )
        mask = official_mask.to(dtype=torch.bool).cpu()
        visible_cell_ids = [int(idx) for idx in torch.nonzero(mask, as_tuple=False).flatten().tolist()]
        debug = _visibility_debug(camera, cell_corners)
        debug_distances = debug["distance_min"].cpu()
        official_distances = official_distances.detach().cpu().to(dtype=torch.float32)
        if official_distances.numel() != len(visible_cell_ids):
            raise RuntimeError(
                f"Official distance output length {official_distances.numel()} does not match "
                f"visible cell count {len(visible_cell_ids)} for {camera.camera_id}."
            )
        pre_fallback = debug["pre_fallback_mask"].cpu()
        nearest_cell_id = int(debug["nearest_cell_id"])
        fallback_changed = bool(mask[nearest_cell_id].item() and not pre_fallback[nearest_cell_id].item())
        projected_boxes = debug["projected_boxes"].cpu().tolist()
        per_cell_bounds = [
            cell_metadata["mad_bounds"][cell_id] for cell_id in range(int(cell_metadata["cell_count"]))
        ]
        records.append(
            {
                "trajectory_index": int(camera.trajectory_index),
                "camera_id": camera.camera_id,
                "visible_cell_ids": visible_cell_ids,
                "visibility_mask": [bool(v) for v in mask.tolist()],
                "visible_cell_count": int(len(visible_cell_ids)),
                "selected_gaussian_count": int(sum(cell_counts[cell_id] for cell_id in visible_cell_ids)),
                "nearest_cell_id": nearest_cell_id,
                "nearest_cell_fallback_changed_mask": fallback_changed,
                "per_visible_cell_camera_distance": {
                    str(cell_id): _round_float(float(official_distances[pos].item()), precision)
                    for pos, cell_id in enumerate(visible_cell_ids)
                },
                "per_visible_cell_projected_box": {
                    str(cell_id): _round_float(projected_boxes[cell_id], precision)
                    for cell_id in visible_cell_ids
                },
                "per_cell_valid_front_corner_count": [
                    int(v) for v in debug["front_corner_counts"].cpu().tolist()
                ],
                "cell_bound_checksum": _sha256_json(per_cell_bounds, precision),
                "projection_matrix_checksum": _sha256_json(camera.projection_matrix.tolist(), precision),
                "full_projection_matrix_checksum": _sha256_json(camera.full_proj_transform.tolist(), precision),
                "camera_center": _round_float(camera.camera_center.tolist(), precision),
                "fov_x_radians": _round_float(camera.fov_x, precision),
                "fov_y_radians": _round_float(camera.fov_y, precision),
                "image_size": [int(camera.width), int(camera.height)],
            }
        )
    return records


def build_fixture(
    *,
    ply: Path,
    asset_identifier: str,
    camera_path: Path,
    reference_repo: Path,
    block_dim: list[int],
    aabb: list[float],
    sample_count: int,
    chunk_size: int,
    znear: float,
    zfar: float,
    precision: int,
) -> dict[str, Any]:
    visibility = load_official_visibility_function(reference_repo)
    cameras, camera_metadata = load_oracle_cameras(camera_path, sample_count=sample_count, znear=znear, zfar=zfar)
    cell_metadata = compute_citygaussian_cell_counts(
        ply,
        block_dim=list(block_dim),
        aabb=list(aabb),
        chunk_size=int(chunk_size),
    )
    header = read_ply_header(ply)
    records = run_visibility_oracle(
        visibility=visibility,
        cameras=cameras,
        cell_metadata=cell_metadata,
        block_dim=list(block_dim),
        aabb=list(aabb),
        precision=precision,
    )
    return {
        "schema_version": "citygaussian_visibility_oracle.v1",
        "provenance": {
            "official_reference_commit": REFERENCE_COMMIT,
            "official_reference_repo": str(reference_repo),
            "official_source_function": "utils/large_utils.py::in_frustum",
            "official_source_path": str(reference_repo / "utils" / "large_utils.py"),
            "official_visibility_load_method": visibility.method,
            "oracle_script": str(Path(__file__).resolve()),
            "oracle_method": (
                "Official in_frustum executed on CPU with independently constructed "
                "official-compatible MAD cell bounds and camera matrices. Runtime "
                "scene visibility code is not imported or used."
            ),
            "reference_probe": _official_reference_probe(reference_repo, visibility),
        },
        "asset": {
            "identifier": asset_identifier,
            "path": str(ply),
            "gaussian_count": int(header.vertex_count),
            "ply_format": header.ply_format,
        },
        "cell_metadata": {
            "block_dim": list(block_dim),
            "aabb": list(aabb),
            "cell_count": int(cell_metadata["cell_count"]),
            "cell_ids": list(cell_metadata["cell_ids"]),
            "cell_gaussian_counts": list(cell_metadata["cell_gaussian_counts"]),
            "total_assigned_count": int(cell_metadata["total_assigned_count"]),
            "unassigned_count": int(cell_metadata["unassigned_count"]),
            "duplicated_assignment_count": int(cell_metadata["duplicated_assignment_count"]),
            "nonempty_cell_count": int(cell_metadata["nonempty_cell_count"]),
            "all_cells_nonempty": bool(cell_metadata["all_cells_nonempty"]),
            "cell_bound_checksum": _sha256_json(cell_metadata["mad_bounds"], precision),
            "mad_bound_precision": precision,
        },
        "camera_metadata": camera_metadata,
        "numeric_serialization": {
            "float_rounding_decimal_places": precision,
            "distance_tolerance_for_future_tests": 1e-5,
            "matrix_checksum_rounding_decimal_places": precision,
        },
        "visibility_records": _round_float(records, precision),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract lightweight CityGaussian V1 cell/visibility oracle metadata for one PLY."
    )
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--asset-identifier", type=str, default=None)
    parser.add_argument("--camera-path", type=Path, default=DEFAULT_CAMERA_PATH)
    parser.add_argument("--reference-repo", type=Path, default=DEFAULT_REFERENCE_REPO)
    parser.add_argument("--block-dim", nargs=3, type=int, default=DEFAULT_BLOCK_DIM)
    parser.add_argument("--aabb", nargs=6, type=float, default=DEFAULT_AABB)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--znear", type=float, default=0.01)
    parser.add_argument("--zfar", type=float, default=100.0)
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    asset_identifier = args.asset_identifier or args.ply.parent.name or args.ply.stem
    result = build_fixture(
        ply=args.ply,
        asset_identifier=asset_identifier,
        camera_path=args.camera_path,
        reference_repo=args.reference_repo,
        block_dim=list(args.block_dim),
        aabb=list(args.aabb),
        sample_count=int(args.sample_count),
        chunk_size=int(args.chunk_size),
        znear=float(args.znear),
        zfar=float(args.zfar),
        precision=int(args.precision),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
