#!/usr/bin/env python3
"""Freeze every public Sci-Art validation camera before hierarchy training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from prepare_campus_prevalence_dataset import read_point_sample, read_pose_headers, sha256
from freeze_campus_prevalence_cameras import read_single_pinhole_camera


DATASET_ID = "ALOD-PREV-US3D-SCIART-01"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--oracle-dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["dataset_id"] != DATASET_ID or manifest["status"] != "PASS-FROZEN-DATASET-PREFLIGHT":
        raise RuntimeError("Sci-Art dataset manifest is not sealed")
    train_sparse = args.train_dataset / "sparse/0"
    oracle_sparse = args.oracle_dataset / "sparse/0"
    calibration = read_single_pinhole_camera(oracle_sparse / "cameras.bin")
    train = read_pose_headers(train_sparse / "images.bin")
    oracle = read_pose_headers(oracle_sparse / "images.bin")
    val_names = manifest["split"]["public_val_names"]
    ordered_names = sorted(row["name"] for row in oracle)
    if len(oracle) != 42 or ordered_names[::2] != val_names:
        raise RuntimeError("official llff_hold=2 would not select the public val split")
    train_by_name = {row["name"]: row for row in train}
    oracle_by_name = {row["name"]: row for row in oracle}
    if set(val_names) & set(train_by_name):
        raise RuntimeError("public validation camera leaked into training")
    _, points, _, _ = read_point_sample(train_sparse / "points3D.bin", 250_000)
    tree = cKDTree(points)
    scene_centre = np.median(points, axis=0)
    train_centres = np.asarray([row["centre"] for row in train])
    val_centres = np.asarray([oracle_by_name[name]["centre"] for name in val_names])
    hull = ConvexHull(train_centres)
    signed = val_centres @ hull.equations[:, :3].T + hull.equations[:, 3]
    tolerance = max(1.0, float(np.ptp(train_centres, axis=0).max())) * 1e-9
    inside = np.max(signed, axis=1) <= tolerance
    nearest_camera_distances, nearest_camera_indices = cKDTree(train_centres).query(val_centres, k=1)
    image_rows = {row["name"]: row for row in
                  json.loads((args.manifest.parent / "image-manifest.json").read_text())}
    cameras = []
    for index, name in enumerate(val_names):
        pose = oracle_by_name[name]
        centre = np.asarray(pose["centre"], dtype=np.float64)
        nearest = float(tree.query(centre, k=1)[0])
        centre_distance = float(np.linalg.norm(centre - scene_centre))
        image = image_rows[name]
        fx = float(calibration["fx"]) / 4.0
        cameras.append({
            "camera_center": centre.tolist(),
            "camera_id": int(pose["camera_id"]),
            "camera_model": "PINHOLE",
            "colmap_file_order": next(i for i, row in enumerate(oracle) if row["name"] == name),
            "evaluation_sequence_index": index,
            "stable_sorted_index": ordered_names.index(name),
            "image_name": name,
            "relative_image_path": f"images/{name}",
            "registered_image_id": int(pose["image_id"]),
            "qvec_world_to_camera": np.asarray(pose["qvec_world_to_camera"]).tolist(),
            "tvec_world_to_camera": np.asarray(pose["tvec_world_to_camera"]).tolist(),
            "rotation_world_to_camera": np.asarray(pose["rotation_world_to_camera"]).tolist(),
            "width": int(calibration["width"]),
            "height": int(calibration["height"]),
            "effective_width": 1216,
            "effective_height": 912,
            "intrinsics": {
                "fx": fx, "fy": float(calibration["fy"]) / 4.0,
                "cx": float(calibration["cx"]) / 4.0,
                "cy": float(calibration["cy"]) / 4.0,
            },
            "image_file": {
                "resolved_path": str(args.oracle_dataset / "images" / name),
                "relative_path": f"images/{name}",
                "sha256": image["sha256"], "size_bytes": image["bytes"],
                "width": 4864, "height": 3648,
            },
            "nearest_sampled_geometry_distance": nearest,
            "camera_to_sparse_median_centre_distance": centre_distance,
            "projected_nearest_geometry_scale_proxy": fx / max(nearest, 1e-12),
            "projected_centre_scale_proxy": fx / max(centre_distance, 1e-12),
            "selection_categories": {"capture_view": "UrbanScene3D-Sci-Art",
                                     "split": "public-held-out"},
            "selection_rationale": "Public COLMAP val model; exact oracle llff_hold=2 split",
        })
    result = {
        "schema_version": "alod_sciart_prevalence_evaluation_cameras.v1",
        "dataset_id": DATASET_ID,
        "trajectory_id": "ALOD-PREV-US3D-SCIART-01-EVAL-21-PUBLIC-VAL-V1",
        "ordering": "public val camera name order; oracle sorted-name llff_hold=2 order",
        "resolution_policy": "official factor 4; 1216x912",
        "source_split": "public Sci-Art COLMAP train/val models",
        "source_split_sha256": sha256(args.manifest),
        "camera_count": 21,
        "sparse_sample": {"rule": "every ceil(N/250000)-th training COLMAP sparse point in file order",
                          "count": len(points), "scene_centre": scene_centre.tolist()},
        "cameras": cameras,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    coverage = {
        "schema_version": "alod_sciart_prevalence_coverage.v1",
        "dataset_id": DATASET_ID,
        "train_camera_count": len(train), "evaluation_camera_count": len(val_names),
        "inside_train_camera_convex_hull_count": int(inside.sum()),
        "cameras": [{"name": name,
                     "inside_train_camera_convex_hull": bool(inside[index]),
                     "nearest_train_camera_name": train[int(nearest_camera_indices[index])]["name"],
                     "nearest_train_camera_distance": float(nearest_camera_distances[index])}
                    for index, name in enumerate(val_names)],
    }
    (args.output.parent / "coverage-diagnostic.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    print(f"PASS: 21 frozen Sci-Art public held-out cameras; sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
