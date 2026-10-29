"""Partition an already-trained gsplat checkpoint or Gaussian PLY into rendering blocks."""

from __future__ import annotations

import argparse
from pathlib import Path

from block_manager import build_partitioned_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split existing gsplat .pt/.ckpt checkpoint or Gaussian .ply splats into XY rendering blocks."
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        nargs="+",
        required=True,
        help="One or more gsplat checkpoint .pt/.ckpt or Gaussian .ply files to concatenate before partitioning.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output block metadata/tensor file, e.g. /path/to/aerial_blocks_4x4.pt.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        nargs=2,
        default=(4, 4),
        metavar=("NX", "NY"),
        help="XY grid dimensions. Default: 4 4.",
    )
    parser.add_argument(
        "--scale-padding-factor",
        type=float,
        default=3.0,
        help="Expand cull AABBs by this many robust Gaussian scales per block. Default: 3.0.",
    )
    parser.add_argument(
        "--scale-padding-percentile",
        type=float,
        default=99.5,
        help=(
            "Percentile of per-Gaussian max activated scale used for cull padding. "
            "Use 100 to recover previous max-scale behavior. Default: 99.5."
        ),
    )
    parser.add_argument(
        "--max-padding-fraction",
        type=float,
        default=0.1,
        help=(
            "Clamp cull padding to this fraction of the block extent. "
            "Default: 0.1."
        ),
    )
    parser.add_argument(
        "--no-padding-cap",
        action="store_true",
        help="Disable the max-padding-fraction clamp.",
    )
    parser.add_argument(
        "--ply-coordinate-mode",
        choices=["native", "supersplat"],
        default="native",
        help=(
            "Coordinate convention for .ply inputs. Use native for raw Nerfstudio/Inria "
            "exports and supersplat for PLYs that need the SuperSplat viewer transform. "
            "Default: native."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--cleanup-metadata",
        type=Path,
        default=None,
        help=(
            "Optional cleanup metadata JSON produced by cleanup_gaussian_checkpoint.py. "
            "When provided, partitioning keeps raw tensors but uses metadata filters "
            "to compute cull metadata."
        ),
    )
    parser.add_argument(
        "--metadata-bounds-mode",
        choices=["exact", "robust"],
        default="exact",
        help=(
            "How cleanup metadata defines block cull bounds. exact uses min/max of "
            "metadata-valid Gaussians; robust uses the metadata position percentiles. "
            "Default: exact."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_partitioned_checkpoint(
        ckpt_paths=args.ckpt,
        output_path=args.output,
        grid=(int(args.grid[0]), int(args.grid[1])),
        scale_padding_factor=float(args.scale_padding_factor),
        scale_padding_percentile=float(args.scale_padding_percentile),
        max_padding_fraction=(
            None if args.no_padding_cap else float(args.max_padding_fraction)
        ),
        metadata_bounds_mode=args.metadata_bounds_mode,
        overwrite=bool(args.overwrite),
        cleanup_metadata_path=args.cleanup_metadata,
        ply_coordinate_mode=args.ply_coordinate_mode,
    )
    print(f"Saved block-structured checkpoint to {output}")


if __name__ == "__main__":
    main()
