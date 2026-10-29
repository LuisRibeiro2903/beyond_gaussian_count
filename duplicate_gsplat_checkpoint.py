"""Create a larger gsplat checkpoint by duplicating and translating splats.

This is useful for stress-testing viewers/renderers with an intentionally large
Gaussian count while keeping the original checkpoint structure.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import Tensor


def parse_vec3(text: str) -> Tensor:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected a comma-separated vec3, e.g. 1,0,0")
    return torch.tensor(parts, dtype=torch.float32)


def _as_tensor3(values: object, label: str) -> Tensor:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{label} must be a 3-value list")
    return torch.tensor([float(v) for v in values], dtype=torch.float32)


def load_checkpoint(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu")


def splat_count(splats: Dict[str, Tensor]) -> int:
    if "means" not in splats:
        raise KeyError("checkpoint splats do not contain a 'means' tensor")
    if splats["means"].ndim != 2 or splats["means"].shape[-1] != 3:
        raise ValueError(f"expected splats['means'] to have shape [N, 3], got {tuple(splats['means'].shape)}")
    return int(splats["means"].shape[0])


def auto_spacing(means: Tensor, multiplier: float) -> float:
    mins = means.amin(dim=0)
    maxs = means.amax(dim=0)
    extent = (maxs - mins).max().item()
    if extent <= 0:
        extent = 1.0
    return float(extent * multiplier)


def tensor_bounds(points: Tensor) -> Tuple[Tensor, Tensor]:
    return points.amin(dim=0).to(torch.float32), points.amax(dim=0).to(torch.float32)


def load_json_aabb(path: Path, source: str) -> Tuple[Tensor, Tensor]:
    metadata = json.loads(path.read_text())
    if source == "auto":
        source = "robust" if "xyz_robust_percentiles_after_filtering" in metadata else "exact"

    if source == "robust":
        robust = metadata.get("xyz_robust_percentiles_after_filtering")
        if not isinstance(robust, dict):
            raise ValueError(f"{path} does not contain xyz_robust_percentiles_after_filtering")
        return _as_tensor3(robust.get("min"), "robust min"), _as_tensor3(robust.get("max"), "robust max")
    if source == "exact":
        return (
            _as_tensor3(metadata.get("xyz_min_after_filtering"), "xyz_min_after_filtering"),
            _as_tensor3(metadata.get("xyz_max_after_filtering"), "xyz_max_after_filtering"),
        )
    if source == "raw":
        return (
            _as_tensor3(metadata.get("xyz_min_before_filtering"), "xyz_min_before_filtering"),
            _as_tensor3(metadata.get("xyz_max_before_filtering"), "xyz_max_before_filtering"),
        )
    raise ValueError(f"--aabb-source {source!r} is not available for JSON metadata")


def _union_block_bounds(blocks: object, min_key: str, max_key: str) -> Tuple[Tensor, Tensor]:
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("partitioned checkpoint does not contain a non-empty blocks list")
    mins = []
    maxs = []
    for block in blocks:
        if not isinstance(block, dict) or min_key not in block or max_key not in block:
            raise ValueError(f"partitioned checkpoint block is missing {min_key}/{max_key}")
        mins.append(block[min_key].detach().cpu().to(torch.float32))
        maxs.append(block[max_key].detach().cpu().to(torch.float32))
    return torch.stack(mins, dim=0).amin(dim=0), torch.stack(maxs, dim=0).amax(dim=0)


def load_checkpoint_aabb(path: Path, source: str) -> Tuple[Tensor, Tensor]:
    payload = load_checkpoint(path)
    if not isinstance(payload, dict):
        raise TypeError(f"expected {path} to contain a dict checkpoint")
    if source == "auto":
        source = "cull" if "blocks" in payload else "exact"

    if "blocks" in payload:
        if source == "cull":
            return _union_block_bounds(payload["blocks"], "cull_aabb_min", "cull_aabb_max")
        if source in {"exact", "filtered"}:
            return _union_block_bounds(
                payload["blocks"],
                "filtered_render_bounds_min",
                "filtered_render_bounds_max",
            )
        if source == "raw":
            return _union_block_bounds(
                payload["blocks"],
                "raw_gaussian_bounds_min",
                "raw_gaussian_bounds_max",
            )
        if source == "logical":
            return _union_block_bounds(payload["blocks"], "logical_bounds_min", "logical_bounds_max")
        if source == "scene":
            return (
                payload["scene_aabb_min"].detach().cpu().to(torch.float32),
                payload["scene_aabb_max"].detach().cpu().to(torch.float32),
            )
        raise ValueError(f"--aabb-source {source!r} is not available for partitioned checkpoints")

    splats = payload.get("splats", payload)
    if not isinstance(splats, dict) or "means" not in splats:
        raise ValueError(f"{path} does not contain splats['means']")
    if source not in {"exact", "raw"}:
        raise ValueError(f"--aabb-source {source!r} is not available for flat checkpoints")
    return tensor_bounds(splats["means"].detach().cpu().to(torch.float32))


def load_aabb(path: Path, source: str) -> Tuple[Tensor, Tensor]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json_aabb(path, source)
    return load_checkpoint_aabb(path, source)


def resolve_layout_aabb(args: argparse.Namespace, means: Tensor) -> Tuple[Tensor, Tensor, str]:
    if (args.aabb_min is None) != (args.aabb_max is None):
        raise ValueError("Pass both --aabb-min and --aabb-max, or neither.")
    if args.aabb_metadata is not None and args.aabb_min is not None:
        raise ValueError("Use either --aabb-metadata or explicit --aabb-min/--aabb-max, not both.")

    if args.aabb_metadata is not None:
        mn, mx = load_aabb(args.aabb_metadata, args.aabb_source)
        source = f"{args.aabb_metadata} ({args.aabb_source})"
    elif args.aabb_min is not None and args.aabb_max is not None:
        mn, mx = args.aabb_min, args.aabb_max
        source = "explicit --aabb-min/--aabb-max"
    else:
        mn, mx = tensor_bounds(means)
        source = "input means"

    if torch.any(mx <= mn):
        raise ValueError(f"invalid AABB: min {mn.tolist()} must be smaller than max {mx.tolist()}")
    return mn, mx, source


def grid_offsets(copies: int, spacing: float) -> List[Tensor]:
    cols = math.ceil(math.sqrt(copies))
    offsets = []
    for idx in range(copies):
        row = idx // cols
        col = idx % cols
        offsets.append(torch.tensor([col * spacing, row * spacing, 0.0], dtype=torch.float32))
    return offsets


def aabb_grid_offsets(grid: Tuple[int, int], aabb_min: Tensor, aabb_max: Tensor, overlap: Tensor) -> List[Tensor]:
    nx, ny = grid
    if nx < 1 or ny < 1:
        raise ValueError("--grid dimensions must be positive")
    extent = (aabb_max - aabb_min).to(torch.float32)
    xy_step = extent[:2] - overlap[:2]
    if torch.any(xy_step <= 0):
        raise ValueError(
            "grid overlap must be smaller than the AABB X/Y extent; "
            f"extent={extent[:2].tolist()}, overlap={overlap[:2].tolist()}"
        )
    offsets = []
    for row in range(ny):
        for col in range(nx):
            offsets.append(
                torch.tensor(
                    [float(col * xy_step[0]), float(row * xy_step[1]), 0.0],
                    dtype=torch.float32,
                )
            )
    return offsets


def resolve_grid_overlap(args: argparse.Namespace, aabb_min: Tensor, aabb_max: Tensor) -> Tensor:
    if args.overlap is not None and args.overlap_percent is not None:
        raise ValueError("Use either --overlap or --overlap-percent, not both.")
    extent = (aabb_max - aabb_min).to(torch.float32)
    if args.overlap_percent is not None:
        if args.overlap_percent < 0.0 or args.overlap_percent >= 100.0:
            raise ValueError("--overlap-percent must satisfy 0 <= value < 100")
        overlap_xy = extent[:2] * (float(args.overlap_percent) / 100.0)
    elif args.overlap is not None:
        if args.overlap < 0.0:
            raise ValueError("--overlap must be non-negative")
        overlap_xy = torch.tensor([float(args.overlap), float(args.overlap)], dtype=torch.float32)
    else:
        overlap_xy = torch.zeros((2,), dtype=torch.float32)
    return torch.tensor([float(overlap_xy[0]), float(overlap_xy[1]), 0.0], dtype=torch.float32)


def line_offsets(copies: int, spacing: float, direction: Tensor) -> List[Tensor]:
    norm = torch.linalg.vector_norm(direction)
    if norm <= 0:
        raise ValueError("--direction must not be the zero vector")
    direction = direction / norm
    return [direction * (idx * spacing) for idx in range(copies)]


def make_offsets(args: argparse.Namespace, means: Tensor) -> List[Tensor]:
    if args.offsets:
        offsets = [parse_vec3(text) for text in args.offsets.split(";") if text.strip()]
        if len(offsets) != args.copies:
            raise ValueError(f"--offsets provided {len(offsets)} offsets, but --copies is {args.copies}")
        return offsets

    if args.grid is not None:
        aabb_min, aabb_max, aabb_source = resolve_layout_aabb(args, means)
        overlap = resolve_grid_overlap(args, aabb_min, aabb_max)
        args.copies = int(args.grid[0]) * int(args.grid[1])
        args.layout_aabb_min = aabb_min
        args.layout_aabb_max = aabb_max
        args.layout_aabb_source = aabb_source
        args.layout_overlap = overlap
        return aabb_grid_offsets((int(args.grid[0]), int(args.grid[1])), aabb_min, aabb_max, overlap)

    spacing = args.spacing
    if spacing is None:
        spacing = auto_spacing(means, args.spacing_multiplier)

    if args.layout == "line":
        return line_offsets(args.copies, spacing, args.direction)
    return grid_offsets(args.copies, spacing)


def duplicate_splats(splats: Dict[str, Tensor], offsets: Iterable[Tensor]) -> Dict[str, Tensor]:
    n = splat_count(splats)
    offsets = list(offsets)
    out: Dict[str, Tensor] = {}

    for name, value in splats.items():
        if not torch.is_tensor(value):
            out[name] = value
            continue

        if value.shape[:1] != (n,):
            out[name] = value.clone()
            continue

        chunks = []
        for offset in offsets:
            chunk = value.clone()
            if name == "means":
                chunk = chunk + offset.to(dtype=chunk.dtype).view(1, 3)
            chunks.append(chunk)
        out[name] = torch.cat(chunks, dim=0)

    return out


def describe_offsets(offsets: List[Tensor]) -> str:
    preview = offsets[: min(5, len(offsets))]
    formatted = ["(" + ", ".join(f"{float(v):.3f}" for v in offset.tolist()) + ")" for offset in preview]
    suffix = "" if len(offsets) <= len(preview) else f", ... +{len(offsets) - len(preview)} more"
    return ", ".join(formatted) + suffix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Duplicate a gsplat checkpoint N times by concatenating translated splat tensors."
    )
    parser.add_argument("input", type=Path, help="Input .pt checkpoint")
    parser.add_argument("output", type=Path, help="Output .pt checkpoint")
    parser.add_argument("-n", "--copies", type=int, default=2, help="Number of total copies to write")
    parser.add_argument(
        "--layout",
        choices=("grid", "line"),
        default="grid",
        help="How to place copies when --offsets is not provided",
    )
    parser.add_argument(
        "--grid",
        type=int,
        nargs=2,
        metavar=("NX", "NY"),
        default=None,
        help="Create an AABB-aware NX by NY grid. Overrides --copies and --layout.",
    )
    parser.add_argument(
        "--aabb-metadata",
        type=Path,
        default=None,
        help=(
            "Metadata/checkpoint to define the grid footprint. Supports cleanup JSON, "
            "partitioned checkpoints, and flat gsplat checkpoints."
        ),
    )
    parser.add_argument(
        "--aabb-source",
        choices=("auto", "exact", "robust", "raw", "cull", "filtered", "logical", "scene"),
        default="auto",
        help=(
            "Which AABB to read from --aabb-metadata. JSON supports exact/robust/raw; "
            "partitioned checkpoints support cull/filtered/raw/logical/scene. Default: auto."
        ),
    )
    parser.add_argument(
        "--aabb-min",
        type=parse_vec3,
        default=None,
        help="Explicit grid footprint minimum as x,y,z.",
    )
    parser.add_argument(
        "--aabb-max",
        type=parse_vec3,
        default=None,
        help="Explicit grid footprint maximum as x,y,z.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Absolute X/Y overlap between adjacent AABB grid cells, in scene units.",
    )
    parser.add_argument(
        "--overlap-percent",
        type=float,
        default=None,
        help="X/Y overlap as a percentage of the selected AABB extent.",
    )
    parser.add_argument("--spacing", type=float, default=None, help="Distance between generated copies")
    parser.add_argument(
        "--spacing-multiplier",
        type=float,
        default=1.25,
        help="Auto spacing multiplier applied to the largest bbox extent",
    )
    parser.add_argument(
        "--direction",
        type=parse_vec3,
        default=torch.tensor([1.0, 0.0, 0.0]),
        help="Line layout direction as x,y,z",
    )
    parser.add_argument(
        "--offsets",
        type=str,
        default=None,
        help="Explicit semicolon-separated offsets, e.g. '0,0,0;100,0,0;0,100,0'",
    )
    args = parser.parse_args()

    if args.grid is None and args.copies < 1:
        raise ValueError("--copies must be at least 1")
    if args.grid is not None and args.offsets is not None:
        raise ValueError("--grid and --offsets are mutually exclusive")
    if args.grid is None and (
        args.aabb_metadata is not None
        or args.aabb_min is not None
        or args.aabb_max is not None
        or args.overlap is not None
        or args.overlap_percent is not None
    ):
        raise ValueError("--aabb-* and --overlap* options require --grid")

    ckpt = load_checkpoint(args.input)
    if not isinstance(ckpt, dict):
        raise TypeError(f"expected checkpoint to be a dict, got {type(ckpt).__name__}")

    if "splats" in ckpt:
        splats = ckpt["splats"]
        out_ckpt = copy.copy(ckpt)
    else:
        splats = ckpt
        out_ckpt = None

    if not isinstance(splats, dict):
        raise TypeError("expected checkpoint['splats'] to be a dict-like state_dict")

    original_n = splat_count(splats)
    offsets = make_offsets(args, splats["means"])
    duplicated = duplicate_splats(splats, offsets)
    final_n = splat_count(duplicated)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if out_ckpt is None:
        torch.save(duplicated, args.output)
    else:
        out_ckpt["splats"] = duplicated
        out_ckpt["duplicated_from"] = str(args.input)
        out_ckpt["duplicate_copies"] = args.copies
        out_ckpt["duplicate_offsets"] = [offset.tolist() for offset in offsets]
        if args.grid is not None:
            out_ckpt["duplicate_grid"] = [int(args.grid[0]), int(args.grid[1])]
            out_ckpt["duplicate_layout_aabb_min"] = args.layout_aabb_min.tolist()
            out_ckpt["duplicate_layout_aabb_max"] = args.layout_aabb_max.tolist()
            out_ckpt["duplicate_layout_aabb_source"] = args.layout_aabb_source
            out_ckpt["duplicate_overlap"] = args.layout_overlap.tolist()
        torch.save(out_ckpt, args.output)

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Gaussians: {original_n:,} -> {final_n:,} ({args.copies}x)")
    print(f"Offsets: {describe_offsets(offsets)}")
    if args.grid is not None:
        print(f"Grid: {int(args.grid[0])}x{int(args.grid[1])}")
        print(f"Layout AABB source: {args.layout_aabb_source}")
        print(
            "Layout AABB: "
            f"{tuple(round(float(v), 6) for v in args.layout_aabb_min.tolist())} -> "
            f"{tuple(round(float(v), 6) for v in args.layout_aabb_max.tolist())}"
        )
        print(f"Overlap: {tuple(round(float(v), 6) for v in args.layout_overlap.tolist())}")


if __name__ == "__main__":
    main()
