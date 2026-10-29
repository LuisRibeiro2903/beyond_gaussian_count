#!/usr/bin/env python3
"""Produce deterministic mixed-scale statistics from a frozen ALoD oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_array(root: Path, descriptor: dict[str, Any]) -> np.ndarray:
    return np.load(root / descriptor["payload_path"], allow_pickle=False)


def id_hash(values: np.ndarray, *, sort: bool = False) -> str:
    values = np.asarray(values, dtype="<i8")
    if sort:
        values = np.sort(values, kind="stable")
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def histogram(record: dict[str, Any]) -> dict[int, int]:
    return {int(key): int(value) for key, value in record.items()}


def weighted_quantile(hist: dict[int, int], q: float) -> float:
    total = sum(hist.values())
    if not total:
        raise ValueError("empty histogram")
    target = q * (total - 1)
    lower_index = int(math.floor(target))
    upper_index = int(math.ceil(target))

    def at(index: int) -> int:
        cumulative = 0
        for value, count in sorted(hist.items()):
            cumulative += count
            if index < cumulative:
                return value
        raise AssertionError("quantile index escaped histogram")

    lower = at(lower_index)
    upper = at(upper_index)
    return lower + (upper - lower) * (target - lower_index)


def depth_stats(hist: dict[int, int], maximum_hierarchy_depth: int) -> dict[str, Any]:
    total = sum(hist.values())
    mean = sum(depth * count for depth, count in hist.items()) / total
    coarse_max = maximum_hierarchy_depth // 3
    deep_min = math.ceil(2 * maximum_hierarchy_depth / 3)
    return {
        "minimum": min(hist),
        "p10": weighted_quantile(hist, 0.10),
        "p25": weighted_quantile(hist, 0.25),
        "median": weighted_quantile(hist, 0.50),
        "mean": mean,
        "p75": weighted_quantile(hist, 0.75),
        "p90": weighted_quantile(hist, 0.90),
        "maximum": max(hist),
        "histogram": {str(key): hist[key] for key in sorted(hist)},
        "coarse_level_definition": f"depth <= {coarse_max} (lowest hierarchy-depth tertile)",
        "coarse_level_fraction": sum(
            count for depth, count in hist.items() if depth <= coarse_max
        ) / total,
        "deep_level_definition": f"depth >= {deep_min} (highest hierarchy-depth tertile)",
        "deep_level_fraction": sum(
            count for depth, count in hist.items() if depth >= deep_min
        ) / total,
        "fraction_depth_ge_30": sum(count for depth, count in hist.items() if depth >= 30) / total,
        "fraction_depth_ge_40": sum(count for depth, count in hist.items() if depth >= 40) / total,
    }


def tvd(first: dict[int, int], second: dict[int, int]) -> float:
    first_total = sum(first.values())
    second_total = sum(second.values())
    return 0.5 * sum(
        abs(first.get(key, 0) / first_total - second.get(key, 0) / second_total)
        for key in set(first) | set(second)
    )


def distribution(values: list[float | int]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "minimum": ordered[0],
        "median": float(statistics.median(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "p90": float(np.quantile(ordered, 0.90)),
        "p95": float(np.quantile(ordered, 0.95)),
        "p99": float(np.quantile(ordered, 0.99)),
        "maximum": ordered[-1],
    }


def pair_summary(matrix: list[list[float]], left: list[int], right: list[int], *, same: bool) -> dict[str, float]:
    values = []
    for i in left:
        for j in right:
            if same and j <= i:
                continue
            values.append(matrix[i][j])
    return distribution(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--hierarchy-preflight", type=Path, required=True)
    parser.add_argument("--project-parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.oracle_manifest.resolve()
    root = manifest_path.parent
    manifest = load_json(manifest_path)
    preflight = load_json(args.hierarchy_preflight)
    parity = load_json(args.project_parity)
    maximum_hierarchy_depth = int(preflight["hierarchy"]["max_depth"])
    hspt_path = root / manifest["hspt"]["path"]
    hspt_record = load_json(hspt_path)
    hspt_root = hspt_path.parent
    spt_sizes = load_array(hspt_root, hspt_record["arrays"]["spt_sizes"])
    spt_roots = load_array(hspt_root, hspt_record["arrays"]["spt_root_global_node_ids"])

    camera_results = []
    depth_hists = []
    candidate_sets = []
    jaccard: list[list[float]] = manifest["lod_oracle_classification"]["evidence"][
        "pairwise_selected_set_jaccard"
    ]
    depth_tvd: list[list[float]] = []
    candidate_jaccard: list[list[float]] = []
    for entry in manifest["cameras"]:
        camera_path = root / entry["paths"]["camera.json"]
        camera_root = camera_path.parent
        camera = load_json(camera_path)
        upper = load_json(camera_root / "upper-traversal.json")
        cut = load_json(camera_root / "spt-cut.json")
        working = load_json(camera_root / "working-set.json")
        render = load_json(camera_root / "optional-image.json")
        selected = load_array(
            camera_root, cut["arrays"]["selected_hierarchy_global_node_ids"]
        ).astype(np.int64, copy=False)
        candidate_ids = load_array(camera_root, cut["arrays"]["candidate_spt_ids"]).astype(np.int64, copy=False)
        candidate_distances = load_array(
            camera_root, cut["arrays"]["candidate_spt_distances"]
        )
        raw = load_array(
            camera_root, cut["arrays"]["raw_cuda_spt_global_node_ids"]
        ).astype(np.int64, copy=False)
        starts = load_array(
            camera_root, cut["arrays"]["raw_cuda_spt_prefix_starts"]
        ).astype(np.int64, copy=False)
        per_spt = []
        for index, spt_id_value in enumerate(candidate_ids):
            start = int(starts[index])
            end = int(starts[index + 1]) if index + 1 < starts.size else int(raw.size)
            emitted = raw[start:end]
            spt_id = int(spt_id_value)
            per_spt.append(
                {
                    "candidate_order_index": index,
                    "spt_id": spt_id,
                    "spt_root_global_node_id": int(spt_roots[spt_id]),
                    "candidate_distance": float(candidate_distances[index]),
                    "input_interval_size": int(spt_sizes[spt_id]),
                    "selected_count": int(emitted.size),
                    "ordered_selected_global_node_ids_sha256": id_hash(emitted),
                    "sorted_selected_global_node_ids_sha256": id_hash(emitted, sort=True),
                }
            )
        hist = histogram(cut["selected_depth"]["distribution"])
        modality = camera["selection_categories"]["modality"]
        candidate_set = set(int(value) for value in candidate_ids)
        candidate_sets.append(candidate_set)
        depth_hists.append(hist)
        camera_results.append(
            {
                "camera_stable_id": entry["camera_stable_id"],
                "filename": entry["filename"],
                "modality": modality,
                "distance_stratum": camera["selection_categories"]["distance_stratum"],
                "region_class": camera["selection_categories"]["region_class"],
                "registered_image_id": entry["registered_image_id"],
                "camera_id": entry["camera_id"],
                "dataset_index": camera["frozen_dataset_camera"]["dataset_index"],
                "evaluation_sequence_index": entry["evaluation_sequence_index"],
                "camera_center": camera["effective_official_runtime_camera"]["camera_center_float32"],
                "source_intrinsics": camera["frozen_dataset_camera"]["source_intrinsics"],
                "effective_intrinsics": camera["effective_official_runtime_camera"]["intrinsics"],
                "source_resolution": [camera["frozen_dataset_camera"]["source_width"], camera["frozen_dataset_camera"]["source_height"]],
                "effective_resolution": [camera["effective_official_runtime_camera"]["image_width"], camera["effective_official_runtime_camera"]["image_height"]],
                "pose_c2w_nerf": camera["frozen_dataset_camera"]["formal_dataset_camera"]["c2w_nerf"],
                "world_to_camera_rotation": camera["frozen_dataset_camera"]["formal_dataset_camera"]["colmap_world_to_camera_rotation"],
                "world_to_camera_translation": camera["frozen_dataset_camera"]["formal_dataset_camera"]["colmap_world_to_camera_translation"],
                "candidate_spt_count": int(candidate_ids.size),
                "spts_touched": int(candidate_ids.size),
                "candidate_spt_ordered_sha256": id_hash(candidate_ids),
                "upper_candidates": int(upper["counts"]["coarse_cut"]),
                "upper_emitted_nodes": int(upper["counts"]["emitted_upper_nodes"]),
                "selected_hierarchy_count": int(selected.size),
                "raster_row_count": int(working["global_node_ids"]["count"]),
                "ordered_selected_global_node_ids_sha256": id_hash(selected),
                "sorted_selected_global_node_ids_sha256": id_hash(selected, sort=True),
                "ordered_raster_global_node_ids_sha256": working["global_node_ids"]["ordered_sha256"],
                "selected_depth": depth_stats(hist, maximum_hierarchy_depth),
                "per_spt": per_spt,
                "pre_raster_bundle_sha256": load_json(camera_root / "pre-raster.json")["pre_raster_bundle_sha256"],
                "render_sha256_diagnostic_only": render["rendered_rgb_chw"]["raw_cpu_c_order_sha256"],
                "cold_determinism": load_json(camera_root / "determinism.json"),
            }
        )

    for left in range(len(camera_results)):
        tvd_row, spt_row = [], []
        for right in range(len(camera_results)):
            tvd_row.append(tvd(depth_hists[left], depth_hists[right]))
            spt_union = candidate_sets[left] | candidate_sets[right]
            spt_row.append(1.0 if not spt_union else len(candidate_sets[left] & candidate_sets[right]) / len(spt_union))
        depth_tvd.append(tvd_row)
        candidate_jaccard.append(spt_row)

    aerial = [index for index, row in enumerate(camera_results) if row["modality"] == "aerial"]
    street = [index for index, row in enumerate(camera_results) if row["modality"] == "street"]

    def aggregate(indices: list[int]) -> dict[str, Any]:
        pooled: dict[int, int] = {}
        for index in indices:
            for depth, count in depth_hists[index].items():
                pooled[depth] = pooled.get(depth, 0) + count
        return {
            "camera_count": len(indices),
            "selected_hierarchy_count": distribution([camera_results[i]["selected_hierarchy_count"] for i in indices]),
            "per_camera_median_depth": distribution([camera_results[i]["selected_depth"]["median"] for i in indices]),
            "per_camera_mean_depth": distribution([camera_results[i]["selected_depth"]["mean"] for i in indices]),
            "per_camera_p90_depth": distribution([camera_results[i]["selected_depth"]["p90"] for i in indices]),
            "per_camera_maximum_depth": distribution([camera_results[i]["selected_depth"]["maximum"] for i in indices]),
            "candidate_spt_count": distribution([camera_results[i]["candidate_spt_count"] for i in indices]),
            "spts_touched": distribution([camera_results[i]["spts_touched"] for i in indices]),
            "pooled_selected_depth": depth_stats(pooled, maximum_hierarchy_depth),
        }

    all_pairs = [jaccard[i][j] for i in range(16) for j in range(i + 1, 16)]
    all_tvd = [depth_tvd[i][j] for i in range(16) for j in range(i + 1, 16)]
    classification = manifest["lod_oracle_classification"]
    boundary_examples = {
        "far_aerial": min(
            (camera_results[i] for i in aerial if camera_results[i]["distance_stratum"] == "far"),
            key=lambda row: row["selected_hierarchy_count"],
        )["camera_stable_id"],
        "near_aerial": max(
            (camera_results[i] for i in aerial if camera_results[i]["distance_stratum"] == "near"),
            key=lambda row: row["selected_hierarchy_count"],
        )["camera_stable_id"],
        "representative_street": min(
            (camera_results[i] for i in street),
            key=lambda row: abs(row["selected_hierarchy_count"] - statistics.median(camera_results[j]["selected_hierarchy_count"] for j in street)),
        )["camera_stable_id"],
    }
    result = {
        "schema_version": "alod_formal_mixed_selection_analysis.v1",
        "oracle_id": manifest["oracle_id"],
        "asset": manifest["asset"],
        "evaluation_configuration": hspt_record["configuration"],
        "training_configuration": {
            "SPT_root_volume": 4.75,
            "configured_min_SPT_size": 256,
            "initial_build_min_SPT_size": 256,
            "densification_rebuild_min_SPT_size_source_default": 100,
            "target_granularity_pixels": 2,
            "use_bounding_spheres": False,
            "use_frustum_culling": True,
            "note": "train_hierarchy.py passes min_SPT_size on the initial build but omits it on densification rebuilds, invoking GaussianModel's default 100; HSPT is transient and is not serialized into the final .dhier",
        },
        "hspt": {
            "counts": hspt_record["counts"],
            "coverage": {
                "hierarchy_rows": manifest["asset"]["hierarchy_rows"],
                "unique_covered_hierarchy_rows": manifest["asset"]["hierarchy_rows"],
                "shared_upper_spt_roots": int(spt_roots.size),
            },
            "spt_size_distribution": distribution(spt_sizes.tolist()),
            "spt_ordered_root_global_node_ids_sha256": id_hash(spt_roots),
            "structural_array_hashes": manifest["hspt"]["structural_array_hashes"],
        },
        "cameras": camera_results,
        "modality_aggregates": {
            "aerial": aggregate(aerial),
            "street": aggregate(street),
        },
        "cross_camera": {
            "selected_set_jaccard": {
                "all_pairs": distribution(all_pairs),
                "aerial_aerial": pair_summary(jaccard, aerial, aerial, same=True),
                "street_street": pair_summary(jaccard, street, street, same=True),
                "aerial_street": pair_summary(jaccard, aerial, street, same=False),
                "matrix": jaccard,
            },
            "depth_distribution_total_variation": {
                "all_pairs": distribution(all_tvd),
                "aerial_aerial": pair_summary(depth_tvd, aerial, aerial, same=True),
                "street_street": pair_summary(depth_tvd, street, street, same=True),
                "aerial_street": pair_summary(depth_tvd, aerial, street, same=False),
                "matrix": depth_tvd,
            },
            "candidate_spt_jaccard": {
                "aerial_aerial": pair_summary(candidate_jaccard, aerial, aerial, same=True),
                "street_street": pair_summary(candidate_jaccard, street, street, same=True),
                "aerial_street": pair_summary(candidate_jaccard, aerial, street, same=False),
            },
        },
        "classification": classification,
        "boundary_examples": boundary_examples,
        "project_reproduction": {
            "status": parity["status"],
            "hspt_parity": parity["hspt"]["parity"],
            "camera_count": len(parity["cameras"]),
            "all_camera_layers_exact": all(all(row["parity"].values()) for row in parity["cameras"]),
        },
        "render_quality_evaluation_performed": False,
        "performance_benchmark_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(result), encoding="utf-8")
    print(canonical_json({"output": str(args.output.resolve()), "classification": classification["classification"], "camera_count": len(camera_results)}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
