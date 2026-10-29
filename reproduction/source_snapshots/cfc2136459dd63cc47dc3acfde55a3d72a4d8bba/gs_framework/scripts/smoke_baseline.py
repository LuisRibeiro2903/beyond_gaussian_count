#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline_validation import dumps_report, run_baseline_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the deterministic flat gsplat baseline without launching the viewer."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Render device. auto uses CUDA when available and otherwise runs CPU-only structural checks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for the generated tiny checkpoint and optional PNG.",
    )
    parser.add_argument(
        "--write-png",
        action="store_true",
        help="Write a tiny rendered PNG to --output-dir for manual inspection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_png and args.output_dir is None:
        print("--write-png requires --output-dir", file=sys.stderr)
        return 2
    exit_code, report = run_baseline_smoke(
        output_dir=args.output_dir,
        device_name=args.device,
        write_png=args.write_png,
    )
    print(dumps_report(report))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
