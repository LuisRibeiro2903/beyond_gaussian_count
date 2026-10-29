#!/usr/bin/env python3
"""Inspect a pinned-platform ALoD .dhier asset without CUDA or rendering."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_dhier import (  # noqa: E402
    DhierError,
    DhierFormatError,
    DhierIOError,
    DhierNumericError,
    DhierPlatformError,
    DhierTopologyError,
    ValidationProfile,
    build_inspection_report,
    load_dhier,
)
from alod_dhier.report import SCHEMA_VERSION, canonical_json  # noqa: E402


EXIT_VALID = 0
EXIT_FORMAT = 2
EXIT_NUMERIC = 3
EXIT_TOPOLOGY = 4
EXIT_IO = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--validate-topology", action="store_true")
    parser.add_argument("--skybox-count", type=int, default=0)
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in ValidationProfile],
        default=ValidationProfile.STRICT.value,
        help="strict final-hierarchy semantics or the pinned creator-initial leaf quirk",
    )
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument(
        "--performance",
        action="store_true",
        help="add nondeterministic wall-time/RSS observations to the otherwise deterministic report",
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _failure_report(path: Path, error: BaseException) -> dict[str, Any]:
    category = getattr(error, "category", "io" if isinstance(error, OSError) else "format")
    details = getattr(error, "details", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "asset_path": str(path.expanduser().resolve()),
        "error": {
            "category": category,
            "type": type(error).__name__,
            "message": str(error),
            "details": details,
        },
        "validation": {
            "format_valid": False if category in ("format", "platform") else None,
            "numeric_valid": None,
            "topology_requested": None,
            "topology_valid": None,
            "overall_valid": False,
        },
    }


def _exit_for(error: BaseException) -> int:
    if isinstance(error, DhierNumericError):
        return EXIT_NUMERIC
    if isinstance(error, DhierTopologyError):
        return EXIT_TOPOLOGY
    if isinstance(error, (DhierFormatError, DhierPlatformError)):
        return EXIT_FORMAT
    if isinstance(error, (DhierIOError, OSError)):
        return EXIT_IO
    return EXIT_FORMAT


def _print_summary(report: dict[str, Any]) -> None:
    validation = report["validation"]
    if "error" in report:
        error = report["error"]
        print(f"INVALID [{error['category']}]: {error['message']}")
        return
    fmt = report["format"]
    hierarchy = report["hierarchy"]
    print(
        f"{'VALID' if validation['overall_valid'] else 'INVALID'}: "
        f"{report['asset_path']}\n"
        f"  rows={fmt['gaussian_count']} degree={fmt['sh_degree']} "
        f"bytes={fmt['actual_file_bytes']} sha256={report['asset_sha256']}"
    )
    if hierarchy["topology_validation_performed"]:
        print(
            f"  topology={hierarchy['valid']} profile={hierarchy['profile']} "
            f"root={hierarchy['root_global_id']} reachable={hierarchy['reachable_count']} "
            f"max_depth={hierarchy['max_depth']}"
        )
        if not hierarchy["valid"]:
            print(f"  issue_counts={json.dumps(hierarchy['issues']['counts'], sort_keys=True)}")
    if "performance" in report:
        metrics = report["performance"]
        print(
            f"  parse={metrics['parse_wall_seconds']:.6f}s "
            f"validation_report={metrics['validation_report_wall_seconds']:.6f}s "
            f"peak_rss={metrics['peak_process_rss_bytes']} bytes"
        )


def main() -> int:
    args = parse_args()
    input_path = args.path.expanduser().resolve()
    output_path = args.json_out.expanduser().resolve() if args.json_out else None
    if output_path is not None and output_path == input_path:
        report = _failure_report(input_path, DhierIOError("JSON output must not overwrite input asset"))
        _print_summary(report)
        return EXIT_IO

    try:
        parse_started = time.perf_counter()
        asset = load_dhier(input_path)
        parse_seconds = time.perf_counter() - parse_started
        validation_started = time.perf_counter()
        report = build_inspection_report(
            asset,
            validate_tree=args.validate_topology,
            skybox_count=args.skybox_count,
            profile=args.profile,
            max_examples=args.max_examples,
            repo_root=REPO_ROOT,
        )
        validation_seconds = time.perf_counter() - validation_started
        if args.performance:
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KiB.  This inspector explicitly targets the same
            # validated x86-64 Linux platform as the pinned assets.
            report["performance"] = {
                "parse_wall_seconds": parse_seconds,
                "validation_report_wall_seconds": validation_seconds,
                "peak_process_rss_bytes": int(peak_rss) * 1024,
                "classification": "engineering observation; nondeterministic and not part of canonical parity",
            }
        if not report["validation"]["numeric_valid"]:
            error: DhierError = DhierNumericError("numeric/attribute validation failed", report)
        elif args.validate_topology and not report["validation"]["topology_valid"]:
            error = DhierTopologyError("hierarchy topology validation failed", report)
        else:
            error = None  # type: ignore[assignment]
    except (DhierError, OSError, ValueError) as exc:
        report = getattr(exc, "report", None) or _failure_report(input_path, exc)
        error = exc

    if output_path is not None:
        try:
            _atomic_json(output_path, report)
        except OSError as exc:
            print(f"I/O error writing {output_path}: {exc}", file=sys.stderr)
            return EXIT_IO
    _print_summary(report)
    return EXIT_VALID if error is None else _exit_for(error)


if __name__ == "__main__":
    raise SystemExit(main())
