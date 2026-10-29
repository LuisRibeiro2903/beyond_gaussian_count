#!/usr/bin/env python3
"""Compute the frozen criteria_v1 and continuous ALoD prevalence measures."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import spearmanr, theilslopes


BOOTSTRAP_SEED = 20260921
BOOTSTRAP_RESAMPLES = 10_000


def weighted_median(histogram: dict[int, int]) -> float:
    total = sum(histogram.values())
    left = (total - 1) // 2
    right = total // 2

    def value_at(target: int) -> int:
        cumulative = 0
        for value, count in sorted(histogram.items()):
            cumulative += count
            if target < cumulative:
                return value
        raise RuntimeError("depth histogram is empty")

    return (value_at(left) + value_at(right)) / 2.0


def response(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(x)) < 5:
        return {"spearman_rho": None, "theil_sen_slope": None, "top_minus_bottom_quintile_median": None}
    order = np.argsort(x, kind="stable")
    quintile_count = max(1, math.ceil(len(x) / 5))
    rho = float(spearmanr(x, y).statistic)
    return {
        "spearman_rho": rho if math.isfinite(rho) else None,
        "theil_sen_slope": float(theilslopes(y, x).slope),
        "top_minus_bottom_quintile_median": float(
            np.median(y[order[-quintile_count:]]) - np.median(y[order[:quintile_count]])
        ),
    }


def bootstrap(x: np.ndarray, y: np.ndarray) -> dict[str, dict[str, float] | None]:
    if len(np.unique(x)) < 5:
        return {key: None for key in response(x, y)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = {key: [] for key in response(x, y)}
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, len(x), size=len(x))
        sample = response(x[indices], y[indices])
        for key, value in sample.items():
            if value is not None and np.isfinite(value):
                values[key].append(value)
    result = {}
    for key, samples in values.items():
        result[key] = (
            None
            if not samples
            else {
                "lower_2_5_percentile": float(np.quantile(samples, 0.025)),
                "upper_97_5_percentile": float(np.quantile(samples, 0.975)),
                "finite_resamples": len(samples),
            }
        )
    return result


def equal_count_bins(x: np.ndarray, y: np.ndarray, counts: np.ndarray, hierarchy_rows: int) -> list[dict]:
    order = np.argsort(x, kind="stable")
    bins = []
    for index, members in enumerate(np.array_split(order, 10)):
        if not len(members):
            continue
        bins.append(
            {
                "bin": index,
                "camera_count": len(members),
                "x_min": float(np.min(x[members])),
                "x_max": float(np.max(x[members])),
                "normalized_mean_depth": {
                    "p25": float(np.quantile(y[members], 0.25)),
                    "median": float(np.median(y[members])),
                    "p75": float(np.quantile(y[members], 0.75)),
                },
                "selected_fraction_of_hierarchy": {
                    "p25": float(np.quantile(counts[members] / hierarchy_rows, 0.25)),
                    "median": float(np.median(counts[members] / hierarchy_rows)),
                    "p75": float(np.quantile(counts[members] / hierarchy_rows, 0.75)),
                },
            }
        )
    return bins


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--frozen-cameras", type=Path, required=True)
    parser.add_argument("--hierarchy-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.oracle_manifest.read_text())
    oracle_root = args.oracle_manifest.parent
    frozen = json.loads(args.frozen_cameras.read_text())
    hierarchy = json.loads(args.hierarchy_validation.read_text())
    frozen_by_index = {
        int(camera["evaluation_sequence_index"]): camera for camera in frozen["cameras"]
    }
    maximum_depth = int(hierarchy["hierarchy"]["max_depth"])
    hierarchy_rows = int(manifest["asset"]["hierarchy_rows"])

    rows = []
    selected_sets = []
    for entry in manifest["cameras"]:
        index = int(entry["evaluation_sequence_index"])
        camera = frozen_by_index[index]
        cut_path = oracle_root / entry["paths"]["spt-cut.json"]
        cut = json.loads(cut_path.read_text())
        hist = {int(depth): int(count) for depth, count in cut["selected_depth"]["distribution"].items()}
        total = sum(hist.values())
        mean_depth = sum(depth * count for depth, count in hist.items()) / total
        payload = cut["arrays"]["selected_hierarchy_global_node_ids"]["payload_path"]
        ids = np.load(cut_path.parent / payload, allow_pickle=False).astype(np.int64, copy=False)
        selected_sets.append(set(int(value) for value in ids))
        rows.append(
            {
                "evaluation_sequence_index": index,
                "image_name": camera["image_name"],
                "projected_nearest_geometry_scale_proxy": camera["projected_nearest_geometry_scale_proxy"],
                "projected_centre_scale_proxy": camera["projected_centre_scale_proxy"],
                "selected_count": total,
                "mean_depth": mean_depth,
                "median_depth": weighted_median(hist),
                "maximum_depth": max(hist),
                "normalized_mean_depth": mean_depth / maximum_depth,
                "depth_distribution": {str(key): hist[key] for key in sorted(hist)},
            }
        )
    rows.sort(key=lambda row: row["evaluation_sequence_index"])

    counts = np.asarray([row["selected_count"] for row in rows], dtype=np.int64)
    means = np.asarray([row["mean_depth"] for row in rows], dtype=np.float64)
    medians = np.asarray([row["median_depth"] for row in rows], dtype=np.float64)
    maximums = np.asarray([row["maximum_depth"] for row in rows], dtype=np.int64)
    y = means / maximum_depth
    q_near = np.asarray([row["projected_nearest_geometry_scale_proxy"] for row in rows])
    q_centre = np.asarray([row["projected_centre_scale_proxy"] for row in rows])
    x_near = np.log2(q_near / np.median(q_near))
    x_centre = np.log2(q_centre / np.median(q_centre))

    jaccards = []
    tvds = []
    hists = [{int(k): int(v) for k, v in row["depth_distribution"].items()} for row in rows]
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            union = selected_sets[left] | selected_sets[right]
            jaccards.append(1.0 if not union else len(selected_sets[left] & selected_sets[right]) / len(union))
            keys = set(hists[left]) | set(hists[right])
            tvds.append(
                0.5
                * sum(
                    abs(
                        hists[left].get(key, 0) / counts[left]
                        - hists[right].get(key, 0) / counts[right]
                    )
                    for key in keys
                )
            )
    relative_range = float((counts.max() - counts.min()) / np.median(counts))
    minimum_jaccard = float(min(jaccards))
    maximum_tvd = float(max(tvds))
    predicates = {
        "material_count_variation": relative_range >= 0.05,
        "meaningful_set_variation": minimum_jaccard <= 0.95,
        "material_depth_variation": bool(maximums.max() > maximums.min() or maximum_tvd >= 0.05),
    }
    degenerate = (
        len(set(counts.tolist())) == 1
        and all(value == selected_sets[0] for value in selected_sets[1:])
        and all(value == hists[0] for value in hists[1:])
    )
    if degenerate:
        classification = "VALID-DEGENERATE-LOD"
    elif all(predicates.values()):
        classification = "VALID-NONTRIVIAL-LOD"
    else:
        classification = "VALID-WEAK-LOD"

    result = {
        "schema_version": "alod_prevalence_lod_response.v1",
        "dataset_id": manifest["dataset"]["id"],
        "oracle_id": manifest["oracle_id"],
        "classification": classification,
        "criteria_v1": {
            "selected_count": {
                "minimum": int(counts.min()),
                "median": float(np.median(counts)),
                "maximum": int(counts.max()),
                "relative_range": relative_range,
            },
            "minimum_pairwise_selected_set_jaccard": minimum_jaccard,
            "maximum_pairwise_depth_distribution_total_variation": maximum_tvd,
            "mean_depth_range": [float(means.min()), float(means.max())],
            "median_depth_range": [float(medians.min()), float(medians.max())],
            "maximum_depth_range": [int(maximums.min()), int(maximums.max())],
            "predicates": predicates,
            "exact_degenerate_predicate": degenerate,
        },
        "continuous_response": {
            "definitions": {
                "x_nearest": "log2((f/r_nearest)/median(f/r_nearest))",
                "x_centre_sensitivity": "log2((f/r_median_sparse_centre)/median(f/r_median_sparse_centre))",
                "y": "mean selected hierarchy depth / final hierarchy maximum depth",
                "quintile": "ceil(n/5) lowest and highest x-ranked cameras",
                "bootstrap": f"two-sided percentile interval, {BOOTSTRAP_RESAMPLES} camera resamples, seed {BOOTSTRAP_SEED}",
            },
            "nearest_geometry_scale": {
                "statistics": response(x_near, y),
                "bootstrap_95_percent": bootstrap(x_near, y),
            },
            "scene_centre_sensitivity": {
                "statistics": response(x_centre, y),
                "bootstrap_95_percent": bootstrap(x_centre, y),
            },
            "hierarchy_scale_response_bins": equal_count_bins(x_near, y, counts, hierarchy_rows),
        },
        "cameras": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"classification": classification, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
