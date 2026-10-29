from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from citygaussian_assets import (  # noqa: E402
    REFERENCE_COMMIT,
    compute_citygaussian_cell_counts,
    infer_sh_degree,
    read_ply_header,
)
from scripts.extract_citygaussian_cell_oracle import (  # noqa: E402
    DEFAULT_AABB,
    DEFAULT_BLOCK_DIM,
    DEFAULT_CAMERA_PATH,
    DEFAULT_PRECISION,
    DEFAULT_REFERENCE_REPO,
    _round_float,
    _sha256_json,
    _visibility_debug,
    corners_from_metadata,
    load_official_visibility_function,
    load_oracle_cameras,
)


DEFAULT_LOD_LEVELS = [
    {
        "label": "light_50_vq",
        "semantic": "finest",
        "path": Path("/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_50_vq/point_cloud.ply"),
    },
    {
        "label": "light_66_vq",
        "semantic": "middle",
        "path": Path("/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_66_vq/point_cloud.ply"),
    },
    {
        "label": "light_75_vq",
        "semantic": "coarsest",
        "path": Path("/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_75_vq/point_cloud.ply"),
    },
]

PAPER_MATRIXCITY_INTERVALS = [
    {"level_index": 0, "label": "light_50_vq", "semantic": "finest", "distance_min": 0.0, "distance_max": 200.0},
    {"level_index": 1, "label": "light_66_vq", "semantic": "middle", "distance_min": 200.0, "distance_max": 400.0},
    {"level_index": 2, "label": "light_75_vq", "semantic": "coarsest", "distance_min": 400.0, "distance_max": None},
]


def focal_length_pixels(image_width: int, fov_x: float) -> float:
    return 0.5 * float(image_width) / math.tan(float(fov_x) * 0.5)


def nyquist_scaling(distance: float, focal_length: float) -> float:
    return 2.0 * float(distance) / float(focal_length)


def release_lod_index(avg_scales_by_level: Sequence[float], nyquist_value: float) -> int:
    """Match `torch.max((avg_scalings > nyquist).uint8, dim=0)` plus fallback."""
    for index, value in enumerate(avg_scales_by_level):
        if float(value) > float(nyquist_value):
            return int(index)
    return int(len(avg_scales_by_level) - 1)


def paper_lod_index(distance: float) -> int:
    if float(distance) < 200.0:
        return 0
    if float(distance) < 400.0:
        return 1
    return 2


def _level_asset(label: str, semantic: str, path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    header = read_ply_header(path)
    sh = infer_sh_degree(header.property_names)
    return {
        "level_index": None,
        "label": str(label),
        "semantic": str(semantic),
        "path": str(path),
        "gaussian_count": int(header.vertex_count),
        "ply_format": header.ply_format,
        "inferred_sh_degree": sh["degree"],
        "cell_ids": list(metadata["cell_ids"]),
        "cell_gaussian_counts": list(metadata["cell_gaussian_counts"]),
        "average_scales": list(metadata["average_scales"]),
        "mad_bounds": list(metadata["mad_bounds"]),
        "total_assigned_count": int(metadata["total_assigned_count"]),
        "unassigned_count": int(metadata["unassigned_count"]),
        "duplicated_assignment_count": int(metadata["duplicated_assignment_count"]),
        "nonempty_cell_count": int(metadata["nonempty_cell_count"]),
        "all_cells_nonempty": bool(metadata["all_cells_nonempty"]),
        "cell_bound_checksum": _sha256_json(metadata["mad_bounds"]),
        "average_scale_checksum": _sha256_json(metadata["average_scales"]),
    }


def _git_reference_summary(reference_repo: Path) -> dict[str, Any]:
    result = {
        "path": str(reference_repo),
        "expected_commit": REFERENCE_COMMIT,
        "branch": None,
        "commit": None,
        "clean_status": None,
        "status_probe_error": None,
    }
    try:
        branch = subprocess.check_output(
            ["git", "-C", str(reference_repo), "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
        ).strip()
        commit = subprocess.check_output(
            ["git", "-C", str(reference_repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(reference_repo), "status", "--short"],
            text=True,
        ).strip()
        result.update(
            {
                "branch": branch,
                "commit": commit,
                "clean_status": status == "",
            }
        )
    except Exception as exc:
        result["status_probe_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _selection_summary(
    *,
    visible_cell_ids: Sequence[int],
    selected_lod_by_cell: dict[str, int],
    selected_gaussian_count_by_cell: dict[str, int],
    level_count: int,
) -> tuple[list[int], list[int], int]:
    cell_counts = [0 for _ in range(level_count)]
    gaussian_counts = [0 for _ in range(level_count)]
    total = 0
    for cell_id in visible_cell_ids:
        key = str(cell_id)
        level_index = int(selected_lod_by_cell[key])
        count = int(selected_gaussian_count_by_cell[key])
        cell_counts[level_index] += 1
        gaussian_counts[level_index] += count
        total += count
    return cell_counts, gaussian_counts, total


def _visible_cells(
    *,
    visibility_fn: Any,
    camera: Any,
    visibility_cell_metadata: dict[str, Any],
    block_dim: list[int],
    aabb: list[float],
    precision: int,
) -> dict[str, Any]:
    cell_corners = corners_from_metadata(visibility_cell_metadata)
    official_camera = type(
        "OfficialCamera",
        (),
        {
            "full_proj_transform": camera.full_proj_transform,
            "world_view_transform": camera.world_view_transform,
            "FoVx": camera.fov_x,
            "FoVy": camera.fov_y,
            "image_width": camera.width,
            "image_height": camera.height,
            "camera_center": camera.camera_center,
        },
    )()
    with torch.no_grad():
        mask, official_distances = visibility_fn(official_camera, cell_corners.clone(), list(aabb), list(block_dim))
    mask = mask.to(dtype=torch.bool).cpu()
    visible_cell_ids = [int(idx) for idx in torch.nonzero(mask, as_tuple=False).flatten().tolist()]
    official_distances = official_distances.detach().cpu().to(dtype=torch.float32)
    if official_distances.numel() != len(visible_cell_ids):
        raise RuntimeError(
            f"Official in_frustum returned {official_distances.numel()} distances for "
            f"{len(visible_cell_ids)} visible cells."
        )
    debug = _visibility_debug(camera, cell_corners)
    pre_fallback = debug["pre_fallback_mask"].cpu()
    nearest_cell_id = int(debug["nearest_cell_id"])
    fallback_changed = bool(mask[nearest_cell_id].item() and not pre_fallback[nearest_cell_id].item())
    return {
        "visible_cell_ids": visible_cell_ids,
        "visibility_mask": [bool(v) for v in mask.tolist()],
        "nearest_cell_id": nearest_cell_id,
        "fallback_changed_mask": fallback_changed,
        "cell_distances": {
            str(cell_id): _round_float(float(official_distances[pos].item()), precision)
            for pos, cell_id in enumerate(visible_cell_ids)
        },
        "per_cell_valid_front_corner_count": [int(v) for v in debug["front_corner_counts"].cpu().tolist()],
        "visibility_source_cell_bound_checksum": _sha256_json(visibility_cell_metadata["mad_bounds"], precision),
    }


def build_lod_frames(
    *,
    policy: Literal["CG-LOD-RELEASE", "CG-LOD-PAPER"],
    visibility_fn: Any,
    cameras: Sequence[Any],
    levels: Sequence[dict[str, Any]],
    visibility_level_index: int,
    block_dim: list[int],
    aabb: list[float],
    precision: int,
) -> list[dict[str, Any]]:
    frames = []
    level_count = len(levels)
    for camera in cameras:
        visibility = _visible_cells(
            visibility_fn=visibility_fn,
            camera=camera,
            visibility_cell_metadata=levels[visibility_level_index],
            block_dim=block_dim,
            aabb=aabb,
            precision=precision,
        )
        focal_length = focal_length_pixels(camera.width, camera.fov_x)
        selected_lod_by_cell: dict[str, int] = {}
        selected_label_by_cell: dict[str, str] = {}
        selected_count_by_cell: dict[str, int] = {}
        nyquist_by_cell: dict[str, float] = {}
        avg_scales_by_cell: dict[str, list[float]] = {}
        for cell_id in visibility["visible_cell_ids"]:
            key = str(cell_id)
            distance = float(visibility["cell_distances"][key])
            if policy == "CG-LOD-RELEASE":
                nyquist_value = nyquist_scaling(distance, focal_length)
                avg_scales = [float(level["average_scales"][cell_id]) for level in levels]
                level_index = release_lod_index(avg_scales, nyquist_value)
                nyquist_by_cell[key] = _round_float(nyquist_value, precision)
                avg_scales_by_cell[key] = _round_float(avg_scales, precision)
            elif policy == "CG-LOD-PAPER":
                level_index = paper_lod_index(distance)
            else:
                raise ValueError(f"Unknown policy {policy}.")
            selected_lod_by_cell[key] = int(level_index)
            selected_label_by_cell[key] = str(levels[level_index]["label"])
            selected_count_by_cell[key] = int(levels[level_index]["cell_gaussian_counts"][cell_id])

        selected_cell_count_by_level, selected_gaussian_count_by_level, total_selected = _selection_summary(
            visible_cell_ids=visibility["visible_cell_ids"],
            selected_lod_by_cell=selected_lod_by_cell,
            selected_gaussian_count_by_cell=selected_count_by_cell,
            level_count=level_count,
        )
        frame = {
            "trajectory_index": int(camera.trajectory_index),
            "camera_id": camera.camera_id,
            "visible_cell_ids": list(visibility["visible_cell_ids"]),
            "visibility_mask": list(visibility["visibility_mask"]),
            "visible_cell_count": int(len(visibility["visible_cell_ids"])),
            "nearest_cell_id": int(visibility["nearest_cell_id"]),
            "fallback_changed_mask": bool(visibility["fallback_changed_mask"]),
            "cell_distances": visibility["cell_distances"],
            "selected_lod_by_cell": selected_lod_by_cell,
            "selected_level_label_by_cell": selected_label_by_cell,
            "selected_gaussian_count_by_cell": selected_count_by_cell,
            "selected_cell_count_by_level": selected_cell_count_by_level,
            "selected_gaussian_count_by_level": selected_gaussian_count_by_level,
            "total_selected_gaussian_count": int(total_selected),
            "focal_length_pixels": _round_float(focal_length, precision),
            "fov_x_radians": _round_float(camera.fov_x, precision),
            "fov_y_radians": _round_float(camera.fov_y, precision),
            "image_size": [int(camera.width), int(camera.height)],
            "visibility_source_level_index": int(visibility_level_index),
            "visibility_source_cell_bound_checksum": visibility["visibility_source_cell_bound_checksum"],
            "per_cell_valid_front_corner_count": visibility["per_cell_valid_front_corner_count"],
        }
        if policy == "CG-LOD-RELEASE":
            frame["nyquist_scaling_by_cell"] = nyquist_by_cell
            frame["average_scales_by_cell"] = avg_scales_by_cell
        frames.append(_round_float(frame, precision))
    return frames


def _fixture_common(
    *,
    policy: Literal["CG-LOD-RELEASE", "CG-LOD-PAPER"],
    reference_repo: Path,
    levels: list[dict[str, Any]],
    camera_metadata: dict[str, Any],
    block_dim: list[int],
    aabb: list[float],
    precision: int,
) -> dict[str, Any]:
    validation_status = "authoritative" if policy == "CG-LOD-RELEASE" else "provisional"
    unresolved = []
    if policy == "CG-LOD-PAPER":
        unresolved = [
            "Paper fixed intervals are reported in world units/meters, but compatibility with the current "
            "project Viser-path scale_ratio camera conversion has not been validated against official renders.",
            "Boundary convention is encoded as half-open [0,200), [200,400), [400,inf) for this provisional oracle.",
        ]
    return {
        "schema_version": "citygaussian_lod_oracle.v1",
        "method": "CityGaussian V1 selected-LoD oracle",
        "policy": policy,
        "reference_repository_commit": REFERENCE_COMMIT,
        "reference_repository": _git_reference_summary(reference_repo),
        "asset_paths_or_roles": [
            {
                "level_index": int(level["level_index"]),
                "label": level["label"],
                "semantic": level["semantic"],
                "path": level["path"],
                "gaussian_count": int(level["gaussian_count"]),
                "inferred_sh_degree": int(level["inferred_sh_degree"]),
            }
            for level in levels
        ],
        "level_order": [level["label"] for level in levels],
        "level_labels": {str(int(level["level_index"])): level["label"] for level in levels},
        "level_semantics": {str(int(level["level_index"])): level["semantic"] for level in levels},
        "level_gaussian_counts": [int(level["gaussian_count"]) for level in levels],
        "level_cell_metadata": [
            {
                "level_index": int(level["level_index"]),
                "label": level["label"],
                "semantic": level["semantic"],
                "cell_ids": list(level["cell_ids"]),
                "cell_gaussian_counts": list(level["cell_gaussian_counts"]),
                "average_scales": _round_float(level["average_scales"], precision),
                "mad_bounds": _round_float(level["mad_bounds"], precision),
                "total_assigned_count": int(level["total_assigned_count"]),
                "unassigned_count": int(level["unassigned_count"]),
                "duplicated_assignment_count": int(level["duplicated_assignment_count"]),
                "nonempty_cell_count": int(level["nonempty_cell_count"]),
                "all_cells_nonempty": bool(level["all_cells_nonempty"]),
                "cell_bound_checksum": _sha256_json(level["mad_bounds"], precision),
                "average_scale_checksum": _sha256_json(level["average_scales"], precision),
            }
            for level in levels
        ],
        "block_dim": list(block_dim),
        "aabb": list(aabb),
        "trajectory_indices": list(camera_metadata["selected_indices"]),
        "camera_conversion": camera_metadata,
        "validation_status": validation_status,
        "unresolved_semantics": unresolved,
        "numeric_serialization": {
            "float_rounding_decimal_places": precision,
            "distance_tolerance_for_future_tests": 1e-5,
            "matrix_checksum_rounding_decimal_places": precision,
        },
        "source_evidence": {
            "release_policy": {
                "selection_function": "gaussian_renderer/__init__.py::render_lod",
                "visibility_function": "utils/large_utils.py::in_frustum",
                "cell_construction": "scene/gaussian_model.py::BlockedGaussian.cell_divider",
                "level_order": "render_large_lod.py loads dataset.lod_configs in YAML order",
                "visibility_source_level": "render_lod uses lod_list[-1].cell_corners, i.e. the last/coarsest level",
                "focal_length_equation": "0.5 * viewpoint_cam.image_width / tan(viewpoint_cam.FoVx * 0.5)",
                "distance_definition": "minimum Euclidean camera-space distance over the eight MAD-bound cell corners",
                "average_scale_statistic": "max(mean(activated_scaling[cell_mask], dim=0)) per cell and level",
                "comparison_operator": ">",
                "selection_order": "first matching YAML-order level via torch.max over uint8 boolean comparisons",
                "fallback_behavior": "when no level matches, lod_idx = len(lod_list) - 1",
                "no_blending_or_hysteresis": True,
            },
            "paper_policy": {
                "intervals": PAPER_MATRIXCITY_INTERVALS,
                "status": "provisional; included as separate fixture only",
            },
        },
    }


def load_level_metadata(
    *,
    level_specs: Sequence[dict[str, Any]],
    block_dim: list[int],
    aabb: list[float],
    chunk_size: int,
) -> list[dict[str, Any]]:
    levels = []
    for index, spec in enumerate(level_specs):
        path = Path(spec["path"])
        metadata = compute_citygaussian_cell_counts(
            path,
            block_dim=list(block_dim),
            aabb=list(aabb),
            chunk_size=int(chunk_size),
        )
        level = _level_asset(str(spec["label"]), str(spec["semantic"]), path, metadata)
        level["level_index"] = int(index)
        levels.append(level)
    if len(levels) < 1:
        raise ValueError("At least one LoD level is required.")
    return levels


def build_fixture_from_metadata(
    *,
    policy: Literal["CG-LOD-RELEASE", "CG-LOD-PAPER"],
    levels: Sequence[dict[str, Any]],
    visibility_fn: Any,
    cameras: Sequence[Any],
    camera_metadata: dict[str, Any],
    reference_repo: Path,
    block_dim: list[int],
    aabb: list[float],
    precision: int,
) -> dict[str, Any]:
    visibility_level_index = len(levels) - 1
    fixture = _fixture_common(
        policy=policy,
        reference_repo=reference_repo,
        levels=levels,
        camera_metadata=camera_metadata,
        block_dim=block_dim,
        aabb=aabb,
        precision=precision,
    )
    fixture["frames"] = build_lod_frames(
        policy=policy,
        visibility_fn=visibility_fn,
        cameras=cameras,
        levels=levels,
        visibility_level_index=visibility_level_index,
        block_dim=block_dim,
        aabb=aabb,
        precision=precision,
    )
    fixture["fixture_checksum"] = hashlib.sha256(
        json.dumps(_round_float(fixture, precision), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return _round_float(fixture, precision)


def build_fixture(
    *,
    policy: Literal["CG-LOD-RELEASE", "CG-LOD-PAPER"],
    level_specs: Sequence[dict[str, Any]],
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
    levels = load_level_metadata(
        level_specs=level_specs,
        block_dim=list(block_dim),
        aabb=list(aabb),
        chunk_size=int(chunk_size),
    )
    return build_fixture_from_metadata(
        policy=policy,
        levels=levels,
        visibility_fn=visibility.fn,
        cameras=cameras,
        camera_metadata=camera_metadata,
        reference_repo=reference_repo,
        block_dim=list(block_dim),
        aabb=list(aabb),
        precision=int(precision),
    )


def _parse_level(value: str) -> dict[str, Any]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("LoD levels must use LABEL=SEMANTIC=/path/to/point_cloud.ply")
    label, semantic, path = parts
    return {"label": label, "semantic": semantic, "path": Path(path)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract CityGaussian V1 selected-LoD oracle fixtures.")
    parser.add_argument("--lod-level", action="append", type=_parse_level, default=None)
    parser.add_argument("--camera-path", type=Path, default=DEFAULT_CAMERA_PATH)
    parser.add_argument("--reference-repo", type=Path, default=DEFAULT_REFERENCE_REPO)
    parser.add_argument("--block-dim", nargs=3, type=int, default=DEFAULT_BLOCK_DIM)
    parser.add_argument("--aabb", nargs=6, type=float, default=DEFAULT_AABB)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--znear", type=float, default=0.01)
    parser.add_argument("--zfar", type=float, default=100.0)
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION)
    parser.add_argument("--policy", choices=["release", "paper", "both"], default="both")
    parser.add_argument("--release-output", type=Path)
    parser.add_argument("--paper-output", type=Path)
    return parser


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    level_specs = args.lod_level if args.lod_level is not None else DEFAULT_LOD_LEVELS
    default_fixture_dir = ROOT / "unit_tests" / "fixtures" / "citygaussian_lod"
    release_output = args.release_output or default_fixture_dir / "release_lod_oracle_v1.json"
    paper_output = args.paper_output or default_fixture_dir / "paper_lod_oracle_v1.json"
    visibility = load_official_visibility_function(args.reference_repo)
    cameras, camera_metadata = load_oracle_cameras(
        args.camera_path,
        sample_count=int(args.sample_count),
        znear=float(args.znear),
        zfar=float(args.zfar),
    )
    levels = load_level_metadata(
        level_specs=level_specs,
        block_dim=list(args.block_dim),
        aabb=list(args.aabb),
        chunk_size=int(args.chunk_size),
    )
    common = dict(
        levels=levels,
        visibility_fn=visibility.fn,
        cameras=cameras,
        camera_metadata=camera_metadata,
        reference_repo=args.reference_repo,
        block_dim=list(args.block_dim),
        aabb=list(args.aabb),
        precision=int(args.precision),
    )
    if args.policy in {"release", "both"}:
        _write_json(release_output, build_fixture_from_metadata(policy="CG-LOD-RELEASE", **common))
    if args.policy in {"paper", "both"}:
        _write_json(paper_output, build_fixture_from_metadata(policy="CG-LOD-PAPER", **common))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
