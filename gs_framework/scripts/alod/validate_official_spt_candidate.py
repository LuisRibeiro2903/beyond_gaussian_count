#!/usr/bin/env python3
"""Run one exact pinned-source ALoD HSPT construction without training/rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


OFFICIAL_COMMIT = "73f547341547ae021ff036ad74fdf73a1240b450"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p, method="linear")) if values.size else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spt-root-volume", type=float, required=True)
    parser.add_argument("--expected-hierarchy-sha256", required=True)
    parser.add_argument("--sh-degree", type=int, default=1)
    parser.add_argument("--cap-max", type=int, default=12_000_000)
    parser.add_argument("--target-pixels", type=float, default=2.0)
    parser.add_argument("--min-spt-size", type=int, default=256)
    parser.add_argument("--llff-hold", type=int, default=10)
    parser.add_argument("--resolution-divisor", type=float, default=1.0)
    args = parser.parse_args()
    if args.resolution_divisor <= 0:
        raise ValueError("resolution divisor must be positive")

    source = args.source.resolve()
    hierarchy = args.hierarchy.resolve()
    output = args.output.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"official source commit mismatch: {commit}")
    if sha256_file(hierarchy) != args.expected_hierarchy_sha256:
        raise RuntimeError("immutable hierarchy SHA-256 mismatch")

    os.chdir(source)
    sys.path.insert(0, str(source))
    from scene.dataset_readers import readColmapSceneInfo
    from scene.gaussian_model import GaussianModel
    from utils.general_utils import safe_state

    safe_state(False)
    info = readColmapSceneInfo(
        str(args.dataset.resolve()),
        str((args.dataset / "images").resolve()),
        "",
        "",
        True,
        False,
        args.llff_hold,
    )
    model = GaussianModel(args.sh_degree)
    model.create_from_hier(
        str(hierarchy),
        info.nerf_normalization["radius"],
        str(args.scaffold.resolve()),
    )
    loaded_rows = len(model.nodes)
    model.compact_gaussians(
        "cpu", args.cap_max, "classic", prune_unused_gaussians=False
    )
    focal = float(info.train_cameras[0].focal_length) / args.resolution_divisor
    target_granularity = args.target_pixels / focal
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    roots = model.build_hierarchical_SPT(
        args.spt_root_volume,
        target_granularity,
        args.min_spt_size,
        use_bounding_spheres=False,
    )
    torch.cuda.synchronize()
    build_seconds = time.perf_counter() - started
    sizes = (
        model.SPT_starts[1:] - model.SPT_starts[:-1]
    ).detach().cpu().numpy().astype(np.int64, copy=False)
    upper_rows = int(len(model.upper_tree_nodes))
    flattened = int(sizes.sum())
    hierarchy_rows = loaded_rows - int(model.skybox_points)
    source_status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    )
    substantive_status = [
        line for line in source_status.splitlines()
        if "__pycache__/" not in line and not line.endswith(".pyc")
    ]
    report = {
        "gate": "PASS-OFFICIAL-HSPT-STRUCTURE",
        "training_performed": False,
        "rendering_performed": False,
        "camera_selection_performed": False,
        "official_source_commit": commit,
        "official_source_clean": not substantive_status,
        "hierarchy": str(hierarchy),
        "hierarchy_sha256": args.expected_hierarchy_sha256,
        "SPT_root_volume": args.spt_root_volume,
        "target_granularity_pixels": args.target_pixels,
        "base_focal_length": focal,
        "resolution_divisor": args.resolution_divisor,
        "target_granularity": target_granularity,
        "min_SPT_size": args.min_spt_size,
        "skybox_rows": int(model.skybox_points),
        "hierarchy_rows": hierarchy_rows,
        "upper_hierarchy_rows": upper_rows,
        "SPT_count": int(sizes.size),
        "SPT_root_count": int(roots.numel()),
        "flattened_SPT_rows": flattened,
        "SPT_coverage": flattened / hierarchy_rows,
        "official_print_coverage": flattened / (flattened + upper_rows),
        "SPT_size": {
            "minimum": int(sizes.min()) if sizes.size else 0,
            "p10": percentile(sizes, 10),
            "median": percentile(sizes, 50),
            "mean": float(sizes.mean()) if sizes.size else 0.0,
            "p90": percentile(sizes, 90),
            "p95": percentile(sizes, 95),
            "p99": percentile(sizes, 99),
            "maximum": int(sizes.max()) if sizes.size else 0,
        },
        "HSPT_build_wall_seconds": build_seconds,
        "process_peak_RSS_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        * 1024,
        "pytorch_peak_allocated_bytes_during_HSPT": int(
            torch.cuda.max_memory_allocated()
        ),
        "pytorch_peak_reserved_bytes_during_HSPT": int(
            torch.cuda.max_memory_reserved()
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
