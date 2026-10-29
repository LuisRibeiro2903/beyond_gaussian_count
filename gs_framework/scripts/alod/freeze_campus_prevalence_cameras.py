#!/usr/bin/env python3
"""Freeze every public Campus held-out camera before hierarchy evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_campus_prevalence_dataset import (  # noqa: E402
    DATASET_ID,
    read_point_sample,
    read_pose_headers,
    sha256,
)


def read_single_pinhole_camera(path: Path) -> dict[str, float | int | str]:
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        if count != 1:
            raise RuntimeError(f"expected one camera calibration, found {count}")
        camera_id, model_id, width, height = struct.unpack("<iiQQ", handle.read(24))
        if model_id != 1:
            raise RuntimeError(f"expected COLMAP PINHOLE model id 1, found {model_id}")
        fx, fy, cx, cy = struct.unpack("<dddd", handle.read(32))
    return {
        "camera_id": camera_id,
        "camera_model": "PINHOLE",
        "width": int(width),
        "height": int(height),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    sparse = dataset / "sparse" / "0"
    manifest = json.loads(args.manifest.read_text())
    if manifest["dataset_id"] != DATASET_ID or manifest["status"] != "PASS-FROZEN-DATASET-PREFLIGHT":
        raise RuntimeError("dataset manifest is not the frozen Campus preflight")

    calibration = read_single_pinhole_camera(sparse / "cameras.bin")
    poses = read_pose_headers(sparse / "images.bin")
    by_name = {row["name"]: row for row in poses}
    sorted_names = sorted(by_name)
    test_names = manifest["split"]["public_test_names"]
    if test_names != sorted_names[::50]:
        raise RuntimeError("held-out split changed")

    _, points, _, _ = read_point_sample(sparse / "points3D.bin", 250_000)
    point_tree = cKDTree(points)
    scene_centre = np.median(points, axis=0)
    image_manifest = {
        row["name"]: row
        for row in json.loads((args.manifest.parent / "image-manifest.json").read_text())
    }
    cameras = []
    for evaluation_index, name in enumerate(test_names):
        pose = by_name[name]
        centre = np.asarray(pose["centre"], dtype=np.float64)
        nearest_distance = float(point_tree.query(centre, k=1)[0])
        centre_distance = float(np.linalg.norm(centre - scene_centre))
        fx = float(calibration["fx"])
        image = image_manifest[name]
        cameras.append(
            {
                "camera_center": centre.tolist(),
                "camera_id": int(pose["camera_id"]),
                "camera_model": calibration["camera_model"],
                "colmap_file_order": next(i for i, row in enumerate(poses) if row["name"] == name),
                "evaluation_sequence_index": evaluation_index,
                "stable_sorted_index": sorted_names.index(name),
                "image_name": name,
                "relative_image_path": f"images/{name}",
                "registered_image_id": int(pose["image_id"]),
                "qvec_world_to_camera": np.asarray(pose["qvec_world_to_camera"]).tolist(),
                "tvec_world_to_camera": np.asarray(pose["tvec_world_to_camera"]).tolist(),
                "rotation_world_to_camera": np.asarray(pose["rotation_world_to_camera"]).tolist(),
                "width": int(calibration["width"]),
                "height": int(calibration["height"]),
                "effective_width": int(calibration["width"]),
                "effective_height": int(calibration["height"]),
                "intrinsics": {
                    "fx": fx,
                    "fy": float(calibration["fy"]),
                    "cx": float(calibration["cx"]),
                    "cy": float(calibration["cy"]),
                },
                "image_file": {
                    "resolved_path": str(dataset / "images" / name),
                    "relative_path": f"images/{name}",
                    "sha256": image["sha256"],
                    "size_bytes": image["bytes"],
                    "width": image["resolution"][0],
                    "height": image["resolution"][1],
                },
                "nearest_sampled_geometry_distance": nearest_distance,
                "camera_to_sparse_median_centre_distance": centre_distance,
                "projected_nearest_geometry_scale_proxy": fx / max(nearest_distance, 1e-12),
                "projected_centre_scale_proxy": fx / max(centre_distance, 1e-12),
                "selection_categories": {
                    "capture_view": name.split("/", 1)[0],
                    "split": "public-held-out",
                },
                "selection_rationale": (
                    "Named by the public pre-existing test.txt and exactly equivalent to "
                    "the official loader's sorted-name llff_hold=50 split."
                ),
            }
        )

    result = {
        "schema_version": "alod_campus_prevalence_evaluation_cameras.v1",
        "dataset_id": DATASET_ID,
        "trajectory_id": "ALOD-PREV-H3DGS-CAMPUS-CHUNK-01-EVAL-24-PUBLIC-TEST-V1",
        "ordering": "public test.txt order (which equals sorted-name llff_hold=50 order)",
        "resolution_policy": "archive resolution, divisor 1",
        "source_split": str(sparse / "test.txt"),
        "source_split_sha256": sha256(sparse / "test.txt"),
        "camera_count": len(cameras),
        "sparse_sample": {
            "rule": "every ceil(N/250000)-th COLMAP sparse point in file order",
            "count": len(points),
            "scene_centre": scene_centre.tolist(),
        },
        "cameras": cameras,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
