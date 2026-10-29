#!/usr/bin/env python3
"""Validate and seal the frozen H3DGS Campus standalone prevalence dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull, cKDTree


THESIS_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(THESIS_ROOT))

from tools.audit_alod_dataset_candidates import (  # noqa: E402
    read_cameras,
    read_image_headers,
    read_point_sample,
)


DATASET_ID = "ALOD-PREV-H3DGS-CAMPUS-CHUNK-01"
ARCHIVE_URL = (
    "https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/"
    "datasets/standalone_chunks/campus.zip"
)
EXPECTED = {
    "archive": "a305c06d03e52c68f3f5e8d9c7047b072083b87db6d114e8ebd2c62d7c345032",
    "cameras.bin": "e6b149d42e29eae4251ae5e10d276168f0192fdf2e0e899cdeb14dad1c00ecb2",
    "images.bin": "0734b4645f6b2d07b8ac96ab9c781aee7b9ce0b83a7b93950eebcef00706523c",
    "points3D.bin": "fff9091f2ccdd4f8218c97dd8ab184cbe71d3b71f62521d50d47609408077bf0",
    "test.txt": "5734be7fcaf37148804e40d4cae9d4feae0b17a5079c8fee771a7d8240b77c9c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
    }


def rotation_from_qvec(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def read_pose_headers(path: Path) -> list[dict]:
    """Read the fields needed for coverage while retaining the audited parser output."""
    import struct

    rows = []
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", handle.read(8))
        for _ in range(count):
            image_id = struct.unpack("<i", handle.read(4))[0]
            qvec = np.asarray(struct.unpack("<dddd", handle.read(32)), dtype=np.float64)
            tvec = np.asarray(struct.unpack("<ddd", handle.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<i", handle.read(4))[0]
            name_bytes = bytearray()
            while True:
                value = handle.read(1)
                if value == b"\x00":
                    break
                if not value:
                    raise EOFError("unterminated COLMAP image name")
                name_bytes.extend(value)
            point_count = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(point_count * 24, 1)
            rotation = rotation_from_qvec(qvec)
            rows.append(
                {
                    "image_id": image_id,
                    "camera_id": camera_id,
                    "name": name_bytes.decode("utf-8"),
                    "qvec_world_to_camera": qvec,
                    "tvec_world_to_camera": tvec,
                    "rotation_world_to_camera": rotation,
                    "centre": -rotation.T @ tvec,
                    "view_direction": rotation.T @ np.asarray([0.0, 0.0, 1.0]),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    sparse = dataset / "sparse" / "0"
    images = dataset / "images"
    depths = dataset / "depths"
    test_path = sparse / "test.txt"
    paths = {
        "archive": args.archive.resolve(),
        "cameras.bin": sparse / "cameras.bin",
        "images.bin": sparse / "images.bin",
        "points3D.bin": sparse / "points3D.bin",
        "test.txt": test_path,
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    if actual != EXPECTED:
        raise RuntimeError(f"frozen hash mismatch: {actual}")

    cameras = read_cameras(paths["cameras.bin"])
    audited_rows = read_image_headers(paths["images.bin"])
    pose_rows = read_pose_headers(paths["images.bin"])
    if [row["name"] for row in audited_rows] != [row["name"] for row in pose_rows]:
        raise RuntimeError("independent image parsers disagree")
    test_names = [line.strip() for line in test_path.read_text().splitlines() if line.strip()]
    sorted_names = sorted(row["name"] for row in pose_rows)
    if test_names != sorted_names[::50]:
        raise RuntimeError("public test.txt is not exactly the loader's llff_hold=50 split")
    if len(pose_rows) != 1190 or len(test_names) != 24:
        raise RuntimeError("frozen camera counts changed")
    if set(camera["model"] for camera in cameras.values()) != {"PINHOLE"}:
        raise RuntimeError("unsupported or unexpected camera model")

    image_records = []
    resolutions: dict[str, int] = {}
    for name in sorted_names:
        image_path = images / name
        depth_path = depths / name
        if not image_path.is_file() or not depth_path.is_file():
            raise RuntimeError(f"unresolved image/depth path: {name}")
        with Image.open(image_path) as image:
            width, height = image.size
        key = f"{width}x{height}"
        resolutions[key] = resolutions.get(key, 0) + 1
        image_records.append(
            {
                "name": name,
                "bytes": image_path.stat().st_size,
                "sha256": sha256(image_path),
                "depth_bytes": depth_path.stat().st_size,
                "depth_sha256": sha256(depth_path),
                "resolution": [width, height],
                "split": "evaluation" if name in set(test_names) else "train",
            }
        )
    if resolutions != {"1436x1075": 1190}:
        raise RuntimeError(f"unexpected resolution distribution: {resolutions}")

    point_count, _, _, _ = read_point_sample(paths["points3D.bin"], 250_000)
    if point_count != 359475:
        raise RuntimeError(f"unexpected sparse point count: {point_count}")

    train = [row for row in pose_rows if row["name"] not in set(test_names)]
    evaluation = [row for name in test_names for row in pose_rows if row["name"] == name]
    train_centres = np.asarray([row["centre"] for row in train])
    eval_centres = np.asarray([row["centre"] for row in evaluation])
    train_directions = np.asarray([row["view_direction"] for row in train])
    eval_directions = np.asarray([row["view_direction"] for row in evaluation])

    hull = ConvexHull(train_centres)
    signed = eval_centres @ hull.equations[:, :3].T + hull.equations[:, 3]
    tolerance = max(1.0, float(np.ptp(train_centres, axis=0).max())) * 1e-9
    inside = np.max(signed, axis=1) <= tolerance
    nearest_distance, nearest_index = cKDTree(train_centres).query(eval_centres, k=1)
    nearest_pose_dot = np.sum(eval_directions * train_directions[nearest_index], axis=1)
    nearest_pose_angle = np.degrees(np.arccos(np.clip(nearest_pose_dot, -1.0, 1.0)))
    all_direction_dot = eval_directions @ train_directions.T
    nearest_direction_angle = np.degrees(
        np.arccos(np.clip(np.max(all_direction_dot, axis=1), -1.0, 1.0))
    )

    coverage_rows = []
    for index, row in enumerate(evaluation):
        coverage_rows.append(
            {
                "name": row["name"],
                "inside_training_camera_convex_hull": bool(inside[index]),
                "nearest_training_camera": train[int(nearest_index[index])]["name"],
                "nearest_training_camera_distance": float(nearest_distance[index]),
                "view_direction_angle_to_nearest_position_camera_degrees": float(nearest_pose_angle[index]),
                "nearest_global_training_view_direction_angle_degrees": float(nearest_direction_angle[index]),
            }
        )

    aggregate = hashlib.sha256()
    for row in image_records:
        aggregate.update(
            f"{row['name']}\0{row['sha256']}\0{row['depth_sha256']}\n".encode()
        )
    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "status": "PASS-FROZEN-DATASET-PREFLIGHT",
        "source": {
            "asset": "H3DGS Campus standalone chunk (used for evaluation)",
            "archive_url": ARCHIVE_URL,
            "archive_sha256": actual["archive"],
            "official_dataset_page": "https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/datasets/",
            "paper_project_page": "https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/",
            "gdpr_notice": "https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/Mentions_information_RGPD_GRAPHDECO.pdf",
            "licence_requirements": (
                "No separate data-licence file is embedded in the archive or stated on the download page. "
                "Retain official provenance and hashes, use locally for research/evaluation, do not redistribute "
                "the imagery, and verify redistribution rights with Inria before any public release."
            ),
        },
        "paths": {"dataset": str(dataset), "images": str(images), "sparse": str(sparse)},
        "hashes": actual,
        "counts": {
            "registered_images": len(pose_rows),
            "image_files": len(image_records),
            "depth_files": len(image_records),
            "train_images": len(train),
            "evaluation_images": len(evaluation),
            "sparse_points": point_count,
        },
        "split": {
            "source": str(test_path),
            "public_test_names": test_names,
            "official_loader_equivalence": "sorted registered names with llff_hold=50",
            "llff_hold": 50,
        },
        "camera_models": sorted(set(camera["model"] for camera in cameras.values())),
        "resolution_distribution": resolutions,
        "image_and_depth_manifest_sha256": aggregate.hexdigest(),
        "validation": {
            "all_registered_images_resolve": True,
            "all_depth_paths_resolve": True,
            "finite_camera_centres_and_directions": bool(
                np.isfinite(train_centres).all()
                and np.isfinite(eval_centres).all()
                and np.isfinite(train_directions).all()
                and np.isfinite(eval_directions).all()
            ),
            "public_split_matches_loader": True,
            "supported_intrinsics": True,
        },
    }
    coverage = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "purpose": "pre-training eval-support diagnostic; not an admission or LoD-classification input",
        "definitions": {
            "convex_hull": "3D convex hull of training camera centres",
            "nearest_training_camera_distance": "Euclidean reconstruction-gauge distance",
            "nearest_position_view_direction": "angular separation from the nearest-position training camera",
            "nearest_global_view_direction": "minimum angular separation from any training-camera optical axis",
        },
        "summary": {
            "evaluation_camera_count": len(evaluation),
            "inside_training_camera_convex_hull_count": int(np.sum(inside)),
            "inside_training_camera_convex_hull_fraction": float(np.mean(inside)),
            "nearest_training_camera_distance": distribution(nearest_distance),
            "view_direction_angle_to_nearest_position_camera_degrees": distribution(nearest_pose_angle),
            "nearest_global_training_view_direction_angle_degrees": distribution(nearest_direction_angle),
        },
        "cameras": coverage_rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "image-manifest.json").write_text(
        json.dumps(image_records, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "coverage-diagnostic.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": manifest["status"], "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
