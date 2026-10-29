#!/usr/bin/env python3
"""Freeze the deterministic 71-camera formal ALoD evaluation trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "datasets/alod/ALOD-REPRO-MC-MIXED-01"


def canonical(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_METADATA / "formal_evaluation_cameras.json",
    )
    args = parser.parse_args()

    split_path = args.metadata_root / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    records = sorted(
        (row for row in split["records"] if row["split"] == "eval"),
        key=lambda row: (int(row["dataset_index"]), row["dataset_filename"]),
    )
    if len(records) != 71:
        raise RuntimeError(f"expected 71 formal evaluation cameras, found {len(records)}")

    cameras = []
    for evaluation_index, row in enumerate(records):
        source_width = int(row["image"]["width"])
        source_height = int(row["image"]["height"])
        width = round(source_width / 2)
        height = round(source_height / 2)
        intrinsics = row["intrinsics"]
        rotation = row["colmap_world_to_camera_rotation"]
        translation = row["colmap_world_to_camera_translation"]
        world_to_camera = [
            [float(rotation[r][0]), float(rotation[r][1]), float(rotation[r][2]), float(translation[r])]
            for r in range(3)
        ] + [[0.0, 0.0, 0.0, 1.0]]
        cameras.append(
            {
                "name": row["dataset_filename"],
                "world_to_camera": world_to_camera,
                "camera_center": row["camera_centre"],
                "intrinsics": {
                    "fx": float(intrinsics["fl_x"]) / 2.0,
                    "fy": float(intrinsics["fl_y"]) / 2.0,
                    "cx": float(intrinsics["cx"]) / 2.0,
                    "cy": float(intrinsics["cy"]) / 2.0,
                },
                "width": width,
                "height": height,
                "evaluation_sequence_index": evaluation_index,
                "dataset_index": int(row["dataset_index"]),
                "modality": row["modality"],
                "ground_truth_path": row["image"]["source_path"],
                "ground_truth_sha256": row["image"]["sha256"],
                "camera_height": float(row["camera_height"]),
                "camera_to_scene_centre_distance": float(
                    row["camera_to_scene_centre_distance"]
                ),
                "nearest_scene_distance_proxy": float(
                    row["nearest_scene_distance_proxy"]
                ),
                "projected_scale_proxy": float(row["projected_scale_proxy"]),
                "projected_nearest_geometry_scale_proxy": float(
                    row["projected_nearest_geometry_scale_proxy"]
                ),
            }
        )

    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    result = {
        "schema_version": "alod_formal_evaluation_cameras.v1",
        "dataset_id": split["dataset_id"],
        "trajectory_id": "ALOD-REPRO-MC-MIXED-01-EVAL-71-DATASET-INDEX-V1",
        "ordering": "split == eval, ascending (dataset_index, dataset_filename)",
        "resolution_policy": "official evaluation resolution divisor 2",
        "source_split": str(split_path.resolve()),
        "source_split_sha256": split_sha,
        "camera_count": len(cameras),
        "cameras": cameras,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical(result), encoding="utf-8")
    output_sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(canonical({"output": str(args.output.resolve()), "sha256": output_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
