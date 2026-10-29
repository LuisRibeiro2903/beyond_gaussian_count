#!/usr/bin/env python3
"""Calibrate ALoD SPT structure without training, cameras, or rendering.

This tool reproduces the structural part of the pinned official
``GaussianModel.build_hierarchical_SPT`` implementation: the breadth-first
volume cut, the strict ``>`` threshold, and the strict ``len(SPT) >
min_SPT_size`` admission rule.  Distance thresholds and SPT row ordering do
not affect the requested partition counts or sizes and are deliberately not
materialized by this structural sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from alod_dhier import load_dhier


OFFICIAL_COMMIT = "73f547341547ae021ff036ad74fdf73a1240b450"
DEPTH = 0
PARENT = 1
CHILD_COUNT = 2
FIRST_CHILD = 3
NEXT_SIBLING = 4


@dataclass(frozen=True)
class SweepPolicy:
    skybox_rows: int
    reserve_rows: int
    reduced_cache_rows: int
    min_spt_size: int
    coverage_minimum: float = 0.95
    nondegenerate_min_spts: int = 16
    nondegenerate_max_largest_fraction: float = 0.25
    overpartition_min_median_multiple: int = 4
    comfortable_headroom_fraction: float = 0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile, method="linear"))


def _subtree_sizes(nodes: np.ndarray) -> np.ndarray:
    """Return exact subtree cardinalities for the validated binary tree."""

    sizes = np.ones(nodes.shape[0], dtype=np.int32)
    depths = np.asarray(nodes[:, DEPTH], dtype=np.int32)
    max_depth = int(depths.max())
    for depth in range(max_depth, 0, -1):
        ids = np.flatnonzero(depths == depth)
        parents = np.asarray(nodes[ids, PARENT], dtype=np.int64)
        np.add.at(sizes, parents, sizes[ids])
    return sizes


def _activated_volumes(log_scales: np.ndarray) -> np.ndarray:
    """Match the official CPU PyTorch float32 exp/product expression."""

    rows = torch.from_numpy(np.array(log_scales, dtype=np.float32, copy=True))
    volumes = torch.prod(torch.exp(rows), dim=-1)
    return volumes.numpy().copy()


def _candidate_structure(
    nodes: np.ndarray,
    volumes: np.ndarray,
    subtree_sizes: np.ndarray,
    threshold: float,
    min_spt_size: int,
) -> tuple[np.ndarray, int, int]:
    """Return admitted SPT sizes, upper rows, and below-min candidates."""

    stack = np.asarray([0], dtype=np.int32)
    initial_upper_rows = 0
    cut_interiors: list[np.ndarray] = []
    while stack.size:
        initial_upper_rows += int(stack.size)
        interior = stack[nodes[stack, CHILD_COUNT] > 0]
        if interior.size == 0:
            break
        descend = volumes[interior] > np.float32(threshold)
        if np.any(~descend):
            cut_interiors.append(interior[~descend].copy())
        continuing = interior[descend]
        if continuing.size == 0:
            break
        first = np.asarray(nodes[continuing, FIRST_CHILD], dtype=np.int32)
        second = np.asarray(nodes[first, NEXT_SIBLING], dtype=np.int32)
        stack = np.concatenate((first, second)).astype(np.int32, copy=False)

    cut_ids = (
        np.concatenate(cut_interiors).astype(np.int32, copy=False)
        if cut_interiors
        else np.empty(0, dtype=np.int32)
    )
    raw_sizes = subtree_sizes[cut_ids]
    admitted = raw_sizes > min_spt_size
    spt_sizes = np.asarray(raw_sizes[admitted], dtype=np.int32)
    rejected_sizes = np.asarray(raw_sizes[~admitted], dtype=np.int32)
    # The official builder already included each rejected root in the initial
    # upper cut and then appends the remaining subtree rows.
    upper_rows = initial_upper_rows + int(np.sum(rejected_sizes - 1, dtype=np.int64))
    return spt_sizes, upper_rows, int(rejected_sizes.size)


def _scene_statistics(positions: np.ndarray, volumes: np.ndarray) -> dict[str, Any]:
    minimum = np.min(positions, axis=0).astype(np.float64)
    maximum = np.max(positions, axis=0).astype(np.float64)
    sides = maximum - minimum
    return {
        "bounding_box_min": minimum.tolist(),
        "bounding_box_max": maximum.tolist(),
        "bounding_box_side_lengths": sides.tolist(),
        "bounding_box_volume": float(np.prod(sides)),
        "bounding_box_diagonal": float(np.linalg.norm(sides)),
        "root_position": np.asarray(positions[0], dtype=np.float64).tolist(),
        "root_activated_scale_volume": float(volumes[0]),
        "hierarchy_position_component_min": float(minimum.min()),
        "hierarchy_position_component_max": float(maximum.max()),
    }


def _evaluate_isolated(payload: dict[str, Any]) -> dict[str, Any]:
    started_total = time.perf_counter()
    hierarchy_path = Path(payload["hierarchy"])
    asset = load_dhier(hierarchy_path)
    nodes = np.asarray(asset.nodes)
    positions = np.asarray(asset.positions)
    precompute_started = time.perf_counter()
    subtree_sizes = _subtree_sizes(nodes)
    volumes = _activated_volumes(np.asarray(asset.log_scales))
    precompute_seconds = time.perf_counter() - precompute_started

    threshold = float(payload["threshold"])
    policy = SweepPolicy(**payload["policy"])
    structure_started = time.perf_counter()
    sizes, upper_rows, rejected_count = _candidate_structure(
        nodes, volumes, subtree_sizes, threshold, policy.min_spt_size
    )
    structure_seconds = time.perf_counter() - structure_started
    hierarchy_rows = int(nodes.shape[0])
    flattened = int(np.sum(sizes, dtype=np.int64))
    count = int(sizes.size)
    largest = int(sizes.max()) if count else 0
    median = _percentile(sizes, 50)
    required = policy.skybox_rows + upper_rows + largest + policy.reserve_rows
    headroom = policy.reduced_cache_rows - required
    coverage = flattened / hierarchy_rows
    official_print_coverage = (
        flattened / (flattened + upper_rows) if flattened + upper_rows else 0.0
    )
    max_overpartition_spts = math.floor(
        hierarchy_rows / (policy.overpartition_min_median_multiple * policy.min_spt_size)
    )
    criteria = {
        "coverage": coverage >= policy.coverage_minimum,
        "cache_feasible_strict": required < policy.reduced_cache_rows,
        "nondegenerate_count": count >= policy.nondegenerate_min_spts,
        "nondegenerate_largest_fraction": (
            largest / hierarchy_rows <= policy.nondegenerate_max_largest_fraction
        ),
        "not_overpartitioned_count": count <= max_overpartition_spts,
        "not_overpartitioned_median": (
            median >= policy.overpartition_min_median_multiple * policy.min_spt_size
        ),
        "comfortable_cache_headroom": (
            headroom >= policy.reduced_cache_rows * policy.comfortable_headroom_fraction
        ),
    }
    result = {
        "SPT_root_volume": threshold,
        "hierarchy_rows": hierarchy_rows,
        "upper_hierarchy_rows": upper_rows,
        "SPT_count": count,
        "SPT_coverage": coverage,
        "official_print_coverage": official_print_coverage,
        "flattened_SPT_rows": flattened,
        "SPT_size": {
            "minimum": int(sizes.min()) if count else 0,
            "p10": _percentile(sizes, 10),
            "median": median,
            "mean": float(sizes.mean()) if count else 0.0,
            "p90": _percentile(sizes, 90),
            "p95": _percentile(sizes, 95),
            "p99": _percentile(sizes, 99),
            "maximum": largest,
        },
        "largest_SPT_fraction_of_hierarchy": largest / hierarchy_rows,
        "upper_hierarchy_fraction": upper_rows / hierarchy_rows,
        "below_or_equal_min_SPT_size_candidates": rejected_count,
        "cache_feasible": required < policy.reduced_cache_rows,
        "required_reduced_cache_rows": required,
        "headroom_rows": headroom,
        "criteria": criteria,
        "passes_all_nonstability_criteria": all(criteria.values()),
        "timing": {
            "source_structural_precompute_seconds": precompute_seconds,
            "HSPT_structural_construction_seconds": structure_seconds,
            "isolated_process_wall_seconds": time.perf_counter() - started_total,
        },
        "peak_RSS_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    if payload.get("include_scene_statistics"):
        result["scene_statistics"] = _scene_statistics(positions, volumes)
    return result


def _official_evidence(source: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    )
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"official source commit mismatch: {commit}")
    substantive_status = [
        line for line in status.splitlines()
        if "__pycache__/" not in line and not line.endswith(".pyc")
    ]
    if substantive_status:
        raise RuntimeError("official source checkout has substantive changes")
    evidence_paths = [
        source / "Docs" / "Hyperparameters.md",
        source / "scene" / "gaussian_model.py",
        source / "train_hierarchy.py",
        source / "train.py",
        source / "configs" / "MatrixCity.json",
    ]
    return {
        "commit": commit,
        "checkout_clean": True,
        "files": {str(path): sha256_file(path) for path in evidence_paths},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=float, action="append", required=True)
    parser.add_argument("--expected-hierarchy-sha256", required=True)
    parser.add_argument("--skybox-rows", type=int, default=100_000)
    parser.add_argument("--reserve-rows", type=int, default=100_000)
    parser.add_argument("--reduced-cache-rows", type=int, default=1_600_000)
    parser.add_argument("--min-spt-size", type=int, default=256)
    args = parser.parse_args()

    hierarchy = args.hierarchy.resolve()
    source = args.official_source.resolve()
    actual_hierarchy_hash = sha256_file(hierarchy)
    if actual_hierarchy_hash != args.expected_hierarchy_sha256:
        raise RuntimeError(
            f"hierarchy hash mismatch: {actual_hierarchy_hash} != "
            f"{args.expected_hierarchy_sha256}"
        )
    policy = SweepPolicy(
        skybox_rows=args.skybox_rows,
        reserve_rows=args.reserve_rows,
        reduced_cache_rows=args.reduced_cache_rows,
        min_spt_size=args.min_spt_size,
    )
    candidates = list(dict.fromkeys(float(value) for value in args.candidate))
    results: list[dict[str, Any]] = []
    context = get_context("spawn")
    for index, threshold in enumerate(candidates):
        payload = {
            "hierarchy": str(hierarchy),
            "threshold": threshold,
            "policy": policy.__dict__,
            "include_scene_statistics": index == 0,
        }
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            results.append(executor.submit(_evaluate_isolated, payload).result())

    scene_statistics = results[0].pop("scene_statistics")
    output = {
        "schema_version": 1,
        "method": "source-faithful structural HSPT sweep; no training/cameras/rendering",
        "official_source": _official_evidence(source),
        "hierarchy": {
            "path": str(hierarchy),
            "sha256": actual_hierarchy_hash,
            "rows": results[0]["hierarchy_rows"],
        },
        "scene_statistics": scene_statistics,
        "policy": {
            **policy.__dict__,
            "strict_cache_inequality": "required_reduced_cache_rows < reduced_cache_rows",
            "required_reduced_cache_rows_formula": (
                "skybox_rows + upper_hierarchy_rows + maximum_complete_SPT_rows + reserve_rows"
            ),
            "max_overpartition_spts": math.floor(
                results[0]["hierarchy_rows"]
                / (policy.overpartition_min_median_multiple * policy.min_spt_size)
            ),
            "stability_rule": (
                "selected candidate and immediate tested neighbors must each pass all "
                "non-stability criteria; adjacent SPT-count and maximum-size ratios must "
                "remain within [0.5, 2.0]"
            ),
            "selection_rule": (
                "largest SPT_root_volume in a stable regime passing every criterion"
            ),
        },
        "candidates": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
