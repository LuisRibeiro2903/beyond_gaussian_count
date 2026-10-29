#!/usr/bin/env python3
"""Audit COLMAP camera/geometry scale before any ALoD training outcome exists.

The script reads COLMAP binary models without materializing image observations
or point tracks.  It reports camera coverage, sparse geometry scale, projected
scale proxies, and the exact source-derived initial hierarchy-size estimate
used by the local ALoD admission gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np
from scipy.spatial import cKDTree


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"expected {size} bytes, received {len(data)}")
    return data


def unpack(handle: BinaryIO, fmt: str) -> tuple:
    return struct.unpack("<" + fmt, read_exact(handle, struct.calcsize("<" + fmt)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def read_cameras(path: Path) -> dict[int, dict]:
    output: dict[int, dict] = {}
    with path.open("rb") as handle:
        (count,) = unpack(handle, "Q")
        for _ in range(count):
            camera_id, model_id, width, height = unpack(handle, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"unsupported COLMAP camera model id {model_id}")
            model_name, parameter_count = CAMERA_MODELS[model_id]
            params = np.asarray(unpack(handle, "d" * parameter_count), dtype=np.float64)
            if model_name == "PINHOLE":
                fx, fy = float(params[0]), float(params[1])
            else:
                fx = fy = float(params[0])
            output[camera_id] = {
                "camera_id": camera_id,
                "model": model_name,
                "width": int(width),
                "height": int(height),
                "fx": fx,
                "fy": fy,
            }
    return output


def read_image_headers(path: Path) -> list[dict]:
    output = []
    with path.open("rb") as handle:
        (count,) = unpack(handle, "Q")
        for _ in range(count):
            (image_id,) = unpack(handle, "i")
            qvec = np.asarray(unpack(handle, "dddd"), dtype=np.float64)
            tvec = np.asarray(unpack(handle, "ddd"), dtype=np.float64)
            (camera_id,) = unpack(handle, "i")
            name_bytes = bytearray()
            while True:
                value = read_exact(handle, 1)
                if value == b"\x00":
                    break
                name_bytes.extend(value)
            (point2d_count,) = unpack(handle, "Q")
            handle.seek(int(point2d_count) * 24, 1)
            rotation = qvec_to_rotmat(qvec)
            output.append(
                {
                    "image_id": image_id,
                    "camera_id": camera_id,
                    "name": name_bytes.decode("utf-8"),
                    "centre": -rotation.T @ tvec,
                }
            )
    return output


def read_point_sample(path: Path, maximum_sample: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        (count,) = unpack(handle, "Q")
        stride = max(1, math.ceil(count / maximum_sample))
        sample = []
        xyz_min = np.full(3, np.inf)
        xyz_max = np.full(3, -np.inf)
        for index in range(count):
            unpack(handle, "Q")
            xyz = np.asarray(unpack(handle, "ddd"), dtype=np.float64)
            read_exact(handle, 3)
            unpack(handle, "d")
            (track_length,) = unpack(handle, "Q")
            handle.seek(int(track_length) * 8, 1)
            xyz_min = np.minimum(xyz_min, xyz)
            xyz_max = np.maximum(xyz_max, xyz)
            if index % stride == 0:
                sample.append(xyz)
    return int(count), np.asarray(sample), xyz_min, xyz_max


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
    }


def resolution_distribution(rows: list[dict], cameras: dict[int, dict], divisor: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        camera = cameras[row["camera_id"]]
        width = round(camera["width"] / divisor)
        height = round(camera["height"] / divisor)
        key = f"{width}x{height}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def audit_model(
    path: Path,
    resolution_divisor: int,
    maximum_sample: int,
    vertical_axis: int,
    vertical_direction: str,
) -> dict:
    cameras_path = path / "cameras.bin"
    images_path = path / "images.bin"
    points_path = path / "points3D.bin"
    cameras = read_cameras(cameras_path)
    images = read_image_headers(images_path)
    point_count, point_sample, point_min, point_max = read_point_sample(points_path, maximum_sample)
    centres = np.asarray([row["centre"] for row in images])
    geometry_centre = np.median(point_sample, axis=0)
    if vertical_direction == "up":
        ground_reference = float(np.quantile(point_sample[:, vertical_axis], 0.05))
        camera_height = centres[:, vertical_axis] - ground_reference
        ground_quantile = "p05"
    else:
        ground_reference = float(np.quantile(point_sample[:, vertical_axis], 0.95))
        camera_height = ground_reference - centres[:, vertical_axis]
        ground_quantile = "p95"
    nearest_distance, _ = cKDTree(point_sample).query(centres, k=1)
    centre_distance = np.linalg.norm(centres - geometry_centre, axis=1)
    focal = np.asarray([cameras[row["camera_id"]]["fx"] / resolution_divisor for row in images])
    projected_centre_scale = focal / np.maximum(centre_distance, 1e-12)
    projected_nearest_scale = focal / np.maximum(nearest_distance, 1e-12)
    extent = np.max(centres, axis=0) - np.min(centres, axis=0)
    return {
        "model_path": str(path.resolve()),
        "content_sha256": {
            "cameras.bin": sha256(cameras_path),
            "images.bin": sha256(images_path),
            "points3D.bin": sha256(points_path),
        },
        "camera_count": len(images),
        "camera_calibration_count": len(cameras),
        "camera_model_count": len({camera["model"] for camera in cameras.values()}),
        "camera_models": sorted({camera["model"] for camera in cameras.values()}),
        "source_resolution_distribution": resolution_distribution(images, cameras, 1),
        "study_resolution_divisor": resolution_divisor,
        "study_resolution_distribution": resolution_distribution(images, cameras, resolution_divisor),
        "sparse_point_count": point_count,
        "sparse_sample_count": len(point_sample),
        "sparse_xyz_min": point_min.tolist(),
        "sparse_xyz_max": point_max.tolist(),
        "sparse_extent_xyz": (point_max - point_min).tolist(),
        "sparse_extent_diagonal": float(np.linalg.norm(point_max - point_min)),
        "camera_xyz_min": np.min(centres, axis=0).tolist(),
        "camera_xyz_max": np.max(centres, axis=0).tolist(),
        "camera_extent_xyz": extent.tolist(),
        "camera_extent_diagonal": float(np.linalg.norm(extent)),
        "vertical_axis": vertical_axis,
        "vertical_direction": vertical_direction,
        "sparse_ground_reference_quantile": ground_quantile,
        "sparse_ground_reference": ground_reference,
        "camera_height_above_sparse_ground_proxy": distribution(camera_height),
        "camera_to_sparse_median_centre_distance": distribution(centre_distance),
        "nearest_sampled_geometry_distance": distribution(nearest_distance),
        "focal_length_at_study_resolution": distribution(focal),
        "projected_centre_scale_proxy": distribution(projected_centre_scale),
        "projected_nearest_geometry_scale_proxy": distribution(projected_nearest_scale),
        "projected_nearest_p90_over_p10": float(
            np.quantile(projected_nearest_scale, 0.90)
            / max(np.quantile(projected_nearest_scale, 0.10), 1e-12)
        ),
        "projected_nearest_max_over_min": float(
            np.max(projected_nearest_scale) / max(np.min(projected_nearest_scale), 1e-12)
        ),
        "estimated_initial_hierarchy_rows_before_skybox": 2 * point_count - 1,
        "estimated_initial_fine_rows_with_100k_skybox": 2 * point_count - 1 + 100_000,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", nargs=3, metavar=("ID", "TRAIN_MODEL", "EVAL_MODEL"))
    parser.add_argument(
        "--single-dataset",
        action="append",
        nargs=3,
        metavar=("ID", "MODEL", "TEST_LIST"),
        help="audit one COLMAP model whose held-out image names are listed in TEST_LIST",
    )
    parser.add_argument(
        "--single-model",
        action="append",
        nargs=2,
        metavar=("ID", "MODEL"),
        help="audit one COLMAP model when no public train/evaluation split is supplied",
    )
    parser.add_argument("--resolution-divisor", type=int, default=4)
    parser.add_argument("--maximum-point-sample", type=int, default=250_000)
    parser.add_argument("--vertical-axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--vertical-direction", choices=("up", "down"), default="up")
    parser.add_argument("--source-archive-url", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.dataset and not args.single_dataset and not args.single_model:
        parser.error("at least one --dataset, --single-dataset, or --single-model is required")

    datasets = {}
    for dataset_id, train_model, eval_model in args.dataset or []:
        train = audit_model(
            Path(train_model),
            args.resolution_divisor,
            args.maximum_point_sample,
            args.vertical_axis,
            args.vertical_direction,
        )
        evaluation = audit_model(
            Path(eval_model),
            args.resolution_divisor,
            args.maximum_point_sample,
            args.vertical_axis,
            args.vertical_direction,
        )
        datasets[dataset_id] = {
            "total_camera_count": train["camera_count"] + evaluation["camera_count"],
            "train": train,
            "evaluation": evaluation,
        }
    for dataset_id, model, test_list in args.single_dataset or []:
        combined = audit_model(
            Path(model),
            args.resolution_divisor,
            args.maximum_point_sample,
            args.vertical_axis,
            args.vertical_direction,
        )
        test_names = [
            line.strip()
            for line in Path(test_list).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        datasets[dataset_id] = {
            "total_camera_count": combined["camera_count"],
            "train_camera_count": combined["camera_count"] - len(test_names),
            "evaluation_camera_count": len(test_names),
            "split_source": str(Path(test_list).resolve()),
            "all_registered": combined,
        }
    for dataset_id, model in args.single_model or []:
        combined = audit_model(
            Path(model),
            args.resolution_divisor,
            args.maximum_point_sample,
            args.vertical_axis,
            args.vertical_direction,
        )
        datasets[dataset_id] = {
            "total_camera_count": combined["camera_count"],
            "split_source": None,
            "all_registered": combined,
        }
    result = {
        "schema_version": 1,
        "purpose": "outcome-independent ALoD prevalence-study candidate audit",
        "source_archive": {
            "url": args.source_archive_url,
            "sha256": args.source_archive_sha256,
        },
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve()), "datasets": sorted(datasets)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
