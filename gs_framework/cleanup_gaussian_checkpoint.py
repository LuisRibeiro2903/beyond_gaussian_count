"""Offline cleanup/diagnostics tool for flat gsplat Gaussian checkpoints or PLY files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gaussian_cleanup import (
    BoundsFilterConfig,
    ExportPruneConfig,
    analyze_checkpoint,
    load_flat_checkpoint,
    save_pruned_checkpoint,
    write_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute offline Gaussian cleanup diagnostics for a flat gsplat checkpoint, "
            "or Gaussian PLY file, and optionally export a separately pruned checkpoint."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input flat gsplat checkpoint .pt/.ckpt or Gaussian .ply file.")
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
        "--metadata-output",
        type=Path,
        required=True,
        help="Output JSON sidecar for cleanup diagnostics and robust bounds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output pruned checkpoint path. Required with --export-pruned-checkpoint.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only write diagnostics metadata. This is also the default when export is not requested.",
    )
    parser.add_argument(
        "--export-pruned-checkpoint",
        action="store_true",
        help="Write a separate checkpoint using export-specific prune filters.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing metadata/output files. The input checkpoint is still never overwritten.",
    )
    parser.add_argument("--opacity-min", type=float, default=None, help="Minimum activated opacity for bounds metadata.")
    parser.add_argument(
        "--opacity-percentile",
        type=float,
        default=None,
        help="Keep Gaussians in bounds metadata with activated opacity at or above this percentile threshold.",
    )
    parser.add_argument("--max-scale", type=float, default=None, help="Maximum activated per-Gaussian scale for bounds metadata.")
    parser.add_argument(
        "--max-scale-percentile",
        type=float,
        default=None,
        help="Keep Gaussians in bounds metadata with activated max scale at or below this percentile threshold.",
    )
    parser.add_argument(
        "--roi-min",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional inclusive minimum XYZ scene ROI.",
    )
    parser.add_argument(
        "--roi-max",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional inclusive maximum XYZ scene ROI.",
    )
    parser.add_argument(
        "--position-percentile-min",
        type=float,
        default=0.5,
        help="Lower XYZ percentile for robust render bounds. Default: 0.5.",
    )
    parser.add_argument(
        "--position-percentile-max",
        type=float,
        default=99.5,
        help="Upper XYZ percentile for robust render bounds. Default: 99.5.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Padding added to robust render bounds in scene units. Default: 0.0.",
    )
    parser.add_argument(
        "--export-opacity-min",
        type=float,
        default=None,
        help="Export-only minimum activated opacity. Requires --allow-unprotected-pruning.",
    )
    parser.add_argument(
        "--export-max-scale",
        type=float,
        default=None,
        help="Export-only maximum activated max scale. Requires --allow-unprotected-pruning.",
    )
    parser.add_argument(
        "--export-roi-min",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Export-only inclusive minimum XYZ ROI. Requires --allow-unprotected-pruning.",
    )
    parser.add_argument(
        "--export-roi-max",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Export-only inclusive maximum XYZ ROI. Requires --allow-unprotected-pruning.",
    )
    parser.add_argument(
        "--allow-unprotected-pruning",
        action="store_true",
        help=(
            "Allow destructive export pruning without visibility/contribution protection. "
            "Use only for experiments."
        ),
    )
    return parser.parse_args()


def _bounds_config_from_args(args: argparse.Namespace) -> BoundsFilterConfig:
    return BoundsFilterConfig(
        opacity_min=args.opacity_min,
        opacity_percentile=args.opacity_percentile,
        max_scale=args.max_scale,
        max_scale_percentile=args.max_scale_percentile,
        roi_min=tuple(float(v) for v in args.roi_min) if args.roi_min is not None else None,
        roi_max=tuple(float(v) for v in args.roi_max) if args.roi_max is not None else None,
        position_percentile_min=float(args.position_percentile_min),
        position_percentile_max=float(args.position_percentile_max),
        padding=float(args.padding),
    )


def _export_config_from_args(args: argparse.Namespace) -> ExportPruneConfig:
    return ExportPruneConfig(
        opacity_min=args.export_opacity_min,
        max_scale=args.export_max_scale,
        roi_min=tuple(float(v) for v in args.export_roi_min) if args.export_roi_min is not None else None,
        roi_max=tuple(float(v) for v in args.export_roi_max) if args.export_roi_max is not None else None,
    )


def _has_export_filters(config: ExportPruneConfig) -> bool:
    return any(
        value is not None
        for value in (config.opacity_min, config.max_scale, config.roi_min, config.roi_max)
    )


def run(args: argparse.Namespace) -> None:
    if args.metadata_only and args.export_pruned_checkpoint:
        raise ValueError("--metadata-only and --export-pruned-checkpoint are mutually exclusive.")
    if args.output is not None and not args.export_pruned_checkpoint:
        raise ValueError("--output is only valid with --export-pruned-checkpoint.")
    if args.export_pruned_checkpoint and args.output is None:
        raise ValueError("--output is required with --export-pruned-checkpoint.")
    export_config = _export_config_from_args(args)
    if args.export_pruned_checkpoint:
        if not _has_export_filters(export_config):
            raise ValueError(
                "--export-pruned-checkpoint requires export-specific filters such as "
                "--export-opacity-min, --export-max-scale, --export-roi-min, or --export-roi-max. "
                "Bounds metadata filters are not used for checkpoint pruning."
            )
        if not args.allow_unprotected_pruning:
            raise ValueError(
                "Refusing destructive pruning without --allow-unprotected-pruning. "
                "Visibility/contribution protection is not implemented yet, so pruning can damage quality."
            )
    if args.metadata_output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.metadata_output} already exists. Pass --overwrite to replace it.")
    if args.export_pruned_checkpoint:
        if args.output.resolve() == args.input.resolve():
            raise ValueError("Refusing to overwrite the input checkpoint; choose a separate --output path.")
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"{args.output} already exists. Pass --overwrite to replace it.")

    payload = load_flat_checkpoint(
        args.input,
        ply_coordinate_mode=getattr(args, "ply_coordinate_mode", "native"),
    )
    result = analyze_checkpoint(
        payload,
        _bounds_config_from_args(args),
        export_config=export_config,
        input_path=args.input,
        output_path=args.output if args.export_pruned_checkpoint else None,
        metadata_output_path=args.metadata_output,
    )
    write_metadata(args.metadata_output, result.diagnostics, overwrite=bool(args.overwrite))

    if args.export_pruned_checkpoint:
        print(
            "WARNING: Export pruning is unprotected by visibility/contribution analysis and can reduce quality.",
            file=sys.stderr,
        )
        save_pruned_checkpoint(
            payload,
            result.export_keep_mask,
            input_path=args.input,
            output_path=args.output,
            overwrite=bool(args.overwrite),
        )

    total = int(result.diagnostics["total_gaussian_count"])
    valid = int(result.diagnostics["bounds_valid_gaussian_count"])
    removed = int(result.diagnostics["bounds_removed_gaussian_count"])
    removed_pct = float(result.diagnostics["bounds_removed_gaussian_percent"])
    print(
        f"Cleanup metadata written to {args.metadata_output}\n"
        f"Bounds-valid Gaussians: {valid:,}/{total:,}; excluded from bounds: {removed:,} ({removed_pct:.2f}%)."
    )
    if args.export_pruned_checkpoint:
        export_kept = int(result.diagnostics["export_keep_gaussian_count"])
        export_removed = int(result.diagnostics["export_removed_gaussian_count"])
        export_removed_pct = float(result.diagnostics["export_removed_gaussian_percent"])
        print(
            f"Export-kept Gaussians: {export_kept:,}/{total:,}; pruned: "
            f"{export_removed:,} ({export_removed_pct:.2f}%)."
        )
        print(f"Pruned checkpoint written to {args.output}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
