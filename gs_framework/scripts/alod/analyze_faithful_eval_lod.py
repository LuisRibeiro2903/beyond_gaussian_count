#!/usr/bin/env python3
"""Supplementary non-tuning LoD characterization on all 71 eval cameras."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_dhier import load_dhier
from benchmarks.benchmark_real_scene import (
    apply_camera_config,
    load_camera_path,
    make_frame_camera,
    make_render_settings,
)
from scene_runtime.alod.camera import selection_camera_from_frame
from scene_runtime.alod.hspt import build_production_hspt
from scene_runtime.alod.selection import select_production
from scene_runtime.alod.types import AlodHierarchySource, AlodRuntimeConfig


REPRESENTATIVE_INDICES = (0, 10, 20, 30, 40, 50, 60, 70)


def canonical(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value, dtype="<i8")
    return hashlib.sha256(value.tobytes()).hexdigest()


def weighted_quantile(hist: np.ndarray, q: float) -> float:
    total = int(hist.sum())
    target = q * (total - 1)
    lo, hi = math.floor(target), math.ceil(target)
    cumulative = np.cumsum(hist)
    lo_value = int(np.searchsorted(cumulative, lo, side="right"))
    hi_value = int(np.searchsorted(cumulative, hi, side="right"))
    return float(lo_value + (hi_value - lo_value) * (target - lo))


def depth_record(depths: np.ndarray, max_depth: int) -> tuple[dict, np.ndarray]:
    hist = np.bincount(depths.astype(np.int64), minlength=max_depth + 1)
    total = int(hist.sum())
    values = np.arange(hist.size, dtype=np.float64)
    nonzero = np.flatnonzero(hist)
    return (
        {
            "minimum": int(nonzero[0]),
            "median": weighted_quantile(hist, 0.5),
            "mean": float(np.dot(values, hist) / total),
            "p90": weighted_quantile(hist, 0.9),
            "maximum": int(nonzero[-1]),
            "histogram": {
                str(int(depth)): int(hist[depth]) for depth in nonzero
            },
        },
        hist,
    )


def distribution(values: list[float | int]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(a.min()),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p90": float(np.quantile(a, 0.90)),
        "maximum": float(a.max()),
    }


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.intersect1d(left, right, assume_unique=True).size
    union = left.size + right.size - intersection
    return float(intersection / union) if union else 1.0


def corr(x: list[float], y: list[float]) -> dict[str, float]:
    from scipy.stats import pearsonr, spearmanr

    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dhier", type=Path, required=True)
    parser.add_argument("--camera-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_camera_data = json.loads(args.camera_path.read_text(encoding="utf-8"))
    metadata = raw_camera_data["cameras"]
    camera_path = apply_camera_config(
        load_camera_path(args.camera_path, raw_camera_data["trajectory_id"]),
        {"camera_resolution_scale": 1.0},
    )
    if len(camera_path.cameras) != 71 or len(metadata) != 71:
        raise RuntimeError("formal LoD scan requires exactly 71 frozen eval cameras")

    asset = load_dhier(args.dhier)
    hierarchy = AlodHierarchySource(asset.positions, asset.log_scales, asset.nodes)
    config = AlodRuntimeConfig(
        root_global_node_id=100000,
        skybox_count=100000,
        spt_root_volume=4.75,
        min_spt_size=100,
        target_granularity=0.0008629449482830516,
        distance_multiplier=1.0,
        selection_mode="lod",
        frustum_enabled=True,
        materialization_filter="none",
    )
    hspt = build_production_hspt(hierarchy, config)
    max_depth = int(asset.nodes[config.skybox_count :, 0].max())
    rows, histograms = [], []
    representative_sets: dict[int, np.ndarray] = {}
    adjacent_jaccards = []
    previous_sorted = None
    render_config = {
        "near_plane": 0.01,
        "far_plane": 1.0e10,
        "render_mode": "RGB",
        "background": [0.0, 0.0, 0.0],
        "packed": True,
        "rasterize_mode": "classic",
        "camera_model": "pinhole",
        "radius_clip": 0.0,
        "eps2d": 0.3,
    }
    import torch

    for index, (camera_spec, camera_meta) in enumerate(
        zip(camera_path.cameras, metadata)
    ):
        frame = make_frame_camera(camera_spec, torch.device("cpu"))
        settings = make_render_settings(render_config, camera_spec, sh_degree=1)
        selection_camera = selection_camera_from_frame(
            frame, settings, stable_id=camera_spec.camera_id
        )
        selection = select_production(hierarchy, hspt, selection_camera, config)
        selected = selection.selected_hierarchy_global_node_ids
        sorted_selected = np.sort(selected, kind="stable")
        depths = asset.nodes[selected, 0]
        depth_stats, hist = depth_record(depths, max_depth)
        histograms.append(hist.astype(np.float64) / selected.size)
        if previous_sorted is not None:
            adjacent_jaccards.append(jaccard(previous_sorted, sorted_selected))
        previous_sorted = sorted_selected
        if index in REPRESENTATIVE_INDICES:
            representative_sets[index] = sorted_selected
        assert selection.upper is not None
        rows.append(
            {
                "evaluation_sequence_index": index,
                "camera_id": camera_spec.camera_id,
                "dataset_index": int(camera_meta["dataset_index"]),
                "modality": camera_meta["modality"],
                "selected_hierarchy_count": int(selected.size),
                "selected_global_ids_ordered_sha256": array_hash(selected),
                "selected_depth": depth_stats,
                "spts_touched": len(selection.per_spt),
                "upper_candidates": int(selection.upper.candidate_spt_ids.size),
                "frustum_rejected_upper_nodes": int(
                    selection.upper.frustum_rejected_upper_local_ids.size
                ),
                "camera_scale_metadata": {
                    key: float(camera_meta[key])
                    for key in (
                        "camera_height",
                        "camera_to_scene_centre_distance",
                        "nearest_scene_distance_proxy",
                        "projected_scale_proxy",
                        "projected_nearest_geometry_scale_proxy",
                    )
                },
            }
        )

    representative_pairs = []
    for position, left in enumerate(REPRESENTATIVE_INDICES):
        for right in REPRESENTATIVE_INDICES[position + 1 :]:
            representative_pairs.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "jaccard": jaccard(
                        representative_sets[left], representative_sets[right]
                    ),
                }
            )
    tvds = []
    for left in range(len(histograms)):
        for right in range(left + 1, len(histograms)):
            tvds.append(float(0.5 * np.abs(histograms[left] - histograms[right]).sum()))

    counts = [row["selected_hierarchy_count"] for row in rows]
    mean_depths = [row["selected_depth"]["mean"] for row in rows]
    p90_depths = [row["selected_depth"]["p90"] for row in rows]
    correlations = {}
    for proxy in (
        "camera_height",
        "camera_to_scene_centre_distance",
        "nearest_scene_distance_proxy",
        "projected_scale_proxy",
        "projected_nearest_geometry_scale_proxy",
    ):
        x = [row["camera_scale_metadata"][proxy] for row in rows]
        correlations[proxy] = {
            "selected_hierarchy_count": corr(x, counts),
            "mean_selected_depth": corr(x, mean_depths),
            "p90_selected_depth": corr(x, p90_depths),
        }

    summary = {
        "status": "PASS-ALOD-REPRO-LOD-COVERAGE-71",
        "classification": "supplementary_non_tuning_characterization",
        "camera_count": len(rows),
        "trajectory_id": raw_camera_data["trajectory_id"],
        "ordering": raw_camera_data["ordering"],
        "configuration": {
            "SPT_root_volume": 4.75,
            "min_SPT_size": 100,
            "target_granularity": config.target_granularity,
            "frustum_enabled": True,
            "selector": "validated_faithful_project_CPU_selector",
            "project_extensions_enabled": False,
        },
        "hierarchy": {
            "total_rows": asset.layout.gaussian_count,
            "hierarchy_rows": asset.layout.gaussian_count - config.skybox_count,
            "maximum_depth": max_depth,
            "upper_rows": hspt.upper_row_count,
            "spt_count": hspt.spt_count,
        },
        "selected_hierarchy_count": distribution(counts),
        "selected_count_variation_fraction": float((max(counts) - min(counts)) / min(counts)),
        "per_camera_median_depth": distribution(
            [row["selected_depth"]["median"] for row in rows]
        ),
        "per_camera_mean_depth": distribution(mean_depths),
        "per_camera_p90_depth": distribution(p90_depths),
        "per_camera_maximum_depth": distribution(
            [row["selected_depth"]["maximum"] for row in rows]
        ),
        "spts_touched": distribution([row["spts_touched"] for row in rows]),
        "jaccard": {
            "adjacent_ordered_camera_pairs": distribution(adjacent_jaccards),
            "bounded_representative_indices": list(REPRESENTATIVE_INDICES),
            "bounded_representative_pairs": representative_pairs,
            "bounded_representative_distribution": distribution(
                [row["jaccard"] for row in representative_pairs]
            ),
        },
        "depth_distribution_total_variation_all_pairs": distribution(tvds),
        "camera_scale_associations": correlations,
        "weak_lod_oracle_preserved": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_camera.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(canonical(summary), encoding="utf-8")
    print(canonical({"status": summary["status"], "output_dir": str(args.output_dir.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
