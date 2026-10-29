#!/usr/bin/env python3
"""Freeze and inspect the first bounded official-source ALoD oracle dataset.

This tool is intentionally narrow. It reads a standard COLMAP binary model and
linked images, optionally generates the sparse points3D.ply expected by the
official ALoD loader, and writes deterministic metadata reports outside the
source dataset. It never imports CityGaussian or an ALoD rendering/training
module and never initializes CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


SCHEMA_VERSION = 1
DATASET_ID = "ALOD-ORACLE-MC-AERIAL-TEST-01"
CONTAMINATION_WARNING = (
    "This asset is derived from MatrixCity's original aerial test reconstruction. "
    "Because most of its cameras are repurposed for ALoD training, it must not be "
    "used as an untouched MatrixCity test set or compared directly with formal "
    "results that preserve the original MatrixCity train/test protocol."
)
ALLOWED_USES = [
    "official ALoD coarse-training smoke",
    "initial hierarchy generation",
    "capped fine-hierarchy training",
    ".dhier format validation",
    "official loader validation",
    "source-selected-set oracle extraction",
    "pre-rasterization tensor parity",
    "initial same-asset runtime integration",
    "deterministic cache-trace development",
]
PROHIBITED_USES = [
    "formal MatrixCity train/test quality reporting",
    "direct PSNR/SSIM/LPIPS comparison with existing CityGaussian formal results",
    "claims that the model was trained on the official MatrixCity training split",
    "city-scale scalability claims",
    "final aerial-versus-street LoD conclusions",
    "generalization claims on held-out MatrixCity test images",
]
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
POINT2D_DTYPE = np.dtype([("xy", "<f8", (2,)), ("point3d_id", "<i8")])
TRACK_DTYPE = np.dtype([("image_id", "<i4"), ("point2d_idx", "<i4")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--official-commit", required=True)
    parser.add_argument("--acquisition-date", required=True)
    parser.add_argument("--llff-hold", type=int, default=50)
    parser.add_argument("--generate-ply", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_identity_list(path: Path, records: list[dict[str, Any]]) -> None:
    lines = ["sorted_index\tregistered_image_id\tcamera_id\timage_name"]
    lines.extend(
        f"{record['stable_sorted_index']}\t{record['registered_image_id']}\t"
        f"{record['camera_id']}\t{record['image_name']}"
        for record in records
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def read_cameras(path: Path) -> tuple[dict[int, dict[str, Any]], list[int]]:
    cameras: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", handle.read(24))
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unknown COLMAP camera model id {model_id}")
            model, num_params = CAMERA_MODELS[model_id]
            params = list(struct.unpack("<" + "d" * num_params, handle.read(8 * num_params)))
            if camera_id in cameras:
                raise ValueError(f"Duplicate camera id {camera_id}")
            cameras[camera_id] = {
                "camera_id": camera_id,
                "model_id": model_id,
                "model": model,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
            order.append(camera_id)
        if handle.read(1):
            raise ValueError(f"Trailing data in {path}")
    return cameras, order


def read_images(path: Path) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    images: list[dict[str, Any]] = []
    observations: dict[int, np.ndarray] = {}
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for file_order in range(count):
            image_id = struct.unpack("<i", handle.read(4))[0]
            qvec = np.array(struct.unpack("<4d", handle.read(32)), dtype=np.float64)
            tvec = np.array(struct.unpack("<3d", handle.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<i", handle.read(4))[0]
            name_bytes = bytearray()
            while True:
                value = handle.read(1)
                if not value:
                    raise EOFError(f"Unterminated image name in {path}")
                if value == b"\x00":
                    break
                name_bytes.extend(value)
            image_name = name_bytes.decode("utf-8")
            point2d_count = struct.unpack("<Q", handle.read(8))[0]
            point2d = np.frombuffer(
                handle.read(POINT2D_DTYPE.itemsize * point2d_count), dtype=POINT2D_DTYPE
            ).copy()
            rotation = qvec_to_rotmat(qvec)
            camera_center = -(rotation.T @ tvec)
            images.append(
                {
                    "registered_image_id": image_id,
                    "camera_id": camera_id,
                    "image_name": image_name,
                    "colmap_file_order": file_order,
                    "qvec_world_to_camera": qvec.tolist(),
                    "tvec_world_to_camera": tvec.tolist(),
                    "rotation_world_to_camera": rotation.tolist(),
                    "camera_center": camera_center.tolist(),
                    "point2d_count": int(point2d_count),
                }
            )
            if image_id in observations:
                raise ValueError(f"Duplicate registered image id {image_id}")
            observations[image_id] = point2d["point3d_id"]
        if handle.read(1):
            raise ValueError(f"Trailing data in {path}")
    return images, observations


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[int, np.ndarray]]]:
    xyz_values: list[tuple[float, float, float]] = []
    rgb_values: list[tuple[int, int, int]] = []
    tracks: list[tuple[int, np.ndarray]] = []
    point_ids: set[int] = set()
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            point_id = struct.unpack("<Q", handle.read(8))[0]
            xyz = struct.unpack("<3d", handle.read(24))
            rgb = struct.unpack("<3B", handle.read(3))
            handle.read(8)  # reprojection error
            track_count = struct.unpack("<Q", handle.read(8))[0]
            track = np.frombuffer(
                handle.read(TRACK_DTYPE.itemsize * track_count), dtype=TRACK_DTYPE
            ).copy()
            if point_id in point_ids:
                raise ValueError(f"Duplicate sparse point id {point_id}")
            point_ids.add(point_id)
            xyz_values.append(xyz)
            rgb_values.append(rgb)
            tracks.append((point_id, track))
        if handle.read(1):
            raise ValueError(f"Trailing data in {path}")
    return (
        np.asarray(xyz_values, dtype=np.float64),
        np.asarray(rgb_values, dtype=np.uint8),
        tracks,
    )


def generate_official_style_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Match scene.dataset_readers.storePly in the pinned official source."""
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def quantiles(values: np.ndarray) -> dict[str, float]:
    result = np.quantile(values.astype(np.float64), [0.0, 0.5, 0.9, 1.0])
    return dict(zip(("min", "median", "p90", "max"), map(float, result)))


def camera_stats(records: list[dict[str, Any]], scene_center: np.ndarray) -> dict[str, Any]:
    centers = np.asarray([record["camera_center"] for record in records], dtype=np.float64)
    rotations = np.asarray([record["rotation_world_to_camera"] for record in records])
    optical_axes = rotations[:, 2, :]
    optical_axes /= np.linalg.norm(optical_axes, axis=1, keepdims=True)
    world_down = np.array([0.0, 0.0, -1.0])
    tilt_from_world_down = np.degrees(
        np.arccos(np.clip(optical_axes @ world_down, -1.0, 1.0))
    )
    distances = np.linalg.norm(centers - scene_center[None, :], axis=1)
    if len(centers) > 1:
        pairwise = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(pairwise, np.inf)
        nearest = np.sqrt(pairwise.min(axis=1))
    else:
        nearest = np.zeros(1, dtype=np.float64)
    focal_x = np.asarray([record["intrinsics"]["fx"] for record in records])
    focal_y = np.asarray([record["intrinsics"]["fy"] for record in records])
    resolutions = Counter(f"{record['width']}x{record['height']}" for record in records)
    return {
        "count": len(records),
        "camera_center_bbox_min": centers.min(axis=0).tolist(),
        "camera_center_bbox_max": centers.max(axis=0).tolist(),
        "camera_center_extent": (centers.max(axis=0) - centers.min(axis=0)).tolist(),
        "camera_to_robust_sparse_scene_center": quantiles(distances),
        "nearest_neighbor_camera_distance": quantiles(nearest),
        "camera_height_z": quantiles(centers[:, 2]),
        "optical_axis_world_bbox_min": optical_axes.min(axis=0).tolist(),
        "optical_axis_world_bbox_max": optical_axes.max(axis=0).tolist(),
        "optical_axis_tilt_from_world_down_degrees": quantiles(tilt_from_world_down),
        "focal_x_pixels": quantiles(focal_x),
        "focal_y_pixels": quantiles(focal_y),
        "resolution_distribution": dict(sorted(resolutions.items())),
    }


def intrinsics_for(camera: dict[str, Any]) -> dict[str, float]:
    params = camera["params"]
    if camera["model"] == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif camera["model"] == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(
            f"Official ALoD loader supports only PINHOLE/SIMPLE_PINHOLE, got {camera['model']}"
        )
    return {
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "principal_x_normalized": float(cx / camera["width"]),
        "principal_y_normalized": float(cy / camera["height"]),
        "fov_x_radians": float(2.0 * math.atan(camera["width"] / (2.0 * fx))),
        "fov_y_radians": float(2.0 * math.atan(camera["height"] / (2.0 * fy))),
    }


def load_official_pil_to_torch(official_source: Path):
    sys.dont_write_bytecode = True
    module_path = official_source / "utils" / "general_utils.py"
    spec = importlib.util.spec_from_file_location("alod_official_general_utils_cpu", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PILtoTorch, module.torch


def inspect_images(
    records: list[dict[str, Any]], image_root: Path, official_source: Path
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    image_metadata: dict[int, dict[str, Any]] = {}
    modes: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    total_bytes = 0
    file_sizes: list[int] = []
    for record in records:
        path = image_root / record["image_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            mode = image.mode
            width, height = image.size
        size = path.stat().st_size
        digest = sha256_file(path)
        modes[mode] += 1
        dimensions[f"{width}x{height}"] += 1
        extensions[path.suffix.lower()] += 1
        total_bytes += size
        file_sizes.append(size)
        image_metadata[record["registered_image_id"]] = {
            "relative_path": f"images/{record['image_name']}",
            "resolved_path": str(path.resolve()),
            "size_bytes": size,
            "sha256": digest,
            "mode": mode,
            "width": width,
            "height": height,
        }

    selected_indices = {0, len(records) // 2, len(records) - 1}
    selected_indices.update(
        record["stable_sorted_index"] for record in records if record["split"] == "evaluation"
    )
    pil_to_torch, torch = load_official_pil_to_torch(official_source)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the CPU-only loader check")
    checks = []
    for index in sorted(selected_indices):
        record = records[index]
        image_path = image_root / record["image_name"]
        with Image.open(image_path) as image:
            original_mode = image.mode
            original_size = image.size
            if image.width > 1600:
                scale = image.width / 1600.0
                resolution = (int(image.width / scale), int(image.height / scale))
            else:
                resolution = image.size
            tensor = pil_to_torch(image, resolution)
        if tensor.ndim != 3 or tensor.shape[0] != 4:
            raise ValueError(f"Expected RGBA CHW tensor for {image_path}, got {tuple(tensor.shape)}")
        rgb = tensor[:3, ...]
        alpha = tensor[3:4, ...].to("cpu")
        masked_rgb = rgb.clamp(0.0, 1.0).to("cpu") * alpha
        checks.append(
            {
                "stable_sorted_index": index,
                "registered_image_id": record["registered_image_id"],
                "image_name": record["image_name"],
                "split": record["split"],
                "original_mode": original_mode,
                "original_size": list(original_size),
                "default_resolution_minus_one_output_size": list(resolution),
                "pil_to_torch_shape_chw": list(tensor.shape),
                "tensor_dtype": str(tensor.dtype),
                "rgb_shape_chw": list(rgb.shape),
                "alpha_shape_chw": list(alpha.shape),
                "alpha_min": float(alpha.min()),
                "alpha_max": float(alpha.max()),
                "masked_rgb_finite": bool(torch.isfinite(masked_rgb).all()),
                "masked_rgb_min": float(masked_rgb.min()),
                "masked_rgb_max": float(masked_rgb.max()),
            }
        )
        del tensor, rgb, alpha, masked_rgb
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only loader check initialized CUDA")
    return (
        {
            "all_images_opened_and_png_integrity_verified": True,
            "representative_images_fully_decoded_and_tensor_converted": True,
            "count": len(records),
            "total_bytes": total_bytes,
            "file_size_bytes": quantiles(np.asarray(file_sizes, dtype=np.float64)),
            "mode_distribution": dict(sorted(modes.items())),
            "dimension_distribution": dict(sorted(dimensions.items())),
            "extension_distribution": dict(sorted(extensions.items())),
            "official_function": "utils.general_utils.PILtoTorch",
            "official_camera_behavior": (
                "PILtoTorch converts HWC uint8 to float CHW in [0,1]. With no external "
                "mask, Camera uses channel 3 as alpha, keeps channels 0:3 as RGB, and "
                "multiplies RGB by alpha without background compositing. With resolution=-1, "
                "1920x1080 inputs are LANCZOS-resized to 1600x900."
            ),
            "representative_checks": checks,
            "representative_policy": (
                "all 15 evaluation images plus the early, middle, and late stable sorted images"
            ),
            "cuda_initialized": False,
        },
        image_metadata,
    )


def validate_identities(records: list[dict[str, Any]], image_root: Path) -> dict[str, Any]:
    ids = [record["registered_image_id"] for record in records]
    names = [record["image_name"] for record in records]
    sorted_keys = [(record["image_name"], record["registered_image_id"]) for record in records]
    resolved = [str((image_root / name).resolve()) for name in names]
    casefolded = [name.casefold() for name in names]
    checks = {
        "duplicate_registered_image_ids": len(ids) - len(set(ids)),
        "duplicate_filenames": len(names) - len(set(names)),
        "duplicate_sorted_name_id_keys": len(sorted_keys) - len(set(sorted_keys)),
        "multiple_records_resolving_to_same_file": len(resolved) - len(set(resolved)),
        "case_sensitive_path_collisions": len(casefolded) - len(set(casefolded)),
    }
    if any(checks.values()):
        raise ValueError(f"Stable identity validation failed: {checks}")
    return checks


def validate_colmap_consistency(
    records: list[dict[str, Any]], observations: dict[int, np.ndarray], tracks: list[tuple[int, np.ndarray]]
) -> dict[str, int]:
    image_ids = set(observations)
    point_ids = {point_id for point_id, _ in tracks}
    dangling_image_refs = 0
    nonnegative_refs = 0
    for point_refs in observations.values():
        referenced = point_refs[point_refs >= 0]
        nonnegative_refs += len(referenced)
        dangling_image_refs += sum(int(point_id) not in point_ids for point_id in referenced)
    missing_track_images = 0
    invalid_track_indices = 0
    track_identity_mismatches = 0
    track_observations = 0
    for point_id, track in tracks:
        for image_id, point2d_idx in zip(track["image_id"], track["point2d_idx"]):
            track_observations += 1
            image_id_int = int(image_id)
            point2d_idx_int = int(point2d_idx)
            if image_id_int not in image_ids:
                missing_track_images += 1
                continue
            point_refs = observations[image_id_int]
            if point2d_idx_int < 0 or point2d_idx_int >= len(point_refs):
                invalid_track_indices += 1
                continue
            if int(point_refs[point2d_idx_int]) != point_id:
                track_identity_mismatches += 1
    result = {
        "registered_images": len(records),
        "sparse_points": len(tracks),
        "nonnegative_image_point_references": nonnegative_refs,
        "dangling_image_point_references": dangling_image_refs,
        "track_observations": track_observations,
        "track_missing_registered_images": missing_track_images,
        "track_invalid_point2d_indices": invalid_track_indices,
        "track_identity_mismatches": track_identity_mismatches,
    }
    if any(
        result[key]
        for key in (
            "dangling_image_point_references",
            "track_missing_registered_images",
            "track_invalid_point2d_indices",
            "track_identity_mismatches",
        )
    ):
        raise ValueError(f"COLMAP consistency validation failed: {result}")
    return result


def file_record(path: Path, relative_path: str) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    print("stage=parse_arguments", flush=True)
    if args.llff_hold <= 0:
        raise ValueError("This frozen split requires llff_hold > 0")
    dataset_root = args.dataset_root.resolve()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    official_source = args.official_source.resolve()
    if dataset_root == source_root:
        raise ValueError("Frozen dataset root must not be the source dataset root")
    model_root = dataset_root / "sparse" / "0"
    image_link = dataset_root / "images"
    if not image_link.is_symlink():
        raise ValueError(f"Expected an image-directory symlink: {image_link}")
    image_root = image_link.resolve()
    if image_root != (source_root / "input").resolve():
        raise ValueError(f"Unexpected image link target: {image_root}")

    camera_path = model_root / "cameras.bin"
    images_path = model_root / "images.bin"
    points_path = model_root / "points3D.bin"
    cameras, camera_order = read_cameras(camera_path)
    print(f"stage=read_cameras count={len(cameras)}", flush=True)
    raw_records, observations = read_images(images_path)
    print(f"stage=read_images count={len(raw_records)}", flush=True)
    xyz, rgb, tracks = read_points(points_path)
    print(f"stage=read_points count={len(xyz)}", flush=True)
    if args.generate_ply:
        generate_official_style_ply(model_root / "points3D.ply", xyz, rgb)
    ply_path = model_root / "points3D.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"Required sparse PLY is missing: {ply_path}")

    names = [record["image_name"] for record in raw_records]
    if len(names) != len(set(names)):
        raise ValueError("Official name-only sort is ambiguous because filenames are duplicated")
    records = sorted(raw_records, key=lambda record: record["image_name"])
    for stable_index, record in enumerate(records):
        if record["camera_id"] not in cameras:
            raise ValueError(f"Image references missing camera {record['camera_id']}")
        camera = cameras[record["camera_id"]]
        record["stable_sorted_index"] = stable_index
        record["split"] = "evaluation" if stable_index % args.llff_hold == 0 else "training"
        record["camera_model"] = camera["model"]
        record["width"] = camera["width"]
        record["height"] = camera["height"]
        record["intrinsics"] = intrinsics_for(camera)
        record["relative_image_path"] = f"images/{record['image_name']}"

    identity_checks = validate_identities(records, image_root)
    print("stage=validated_identities", flush=True)
    consistency = validate_colmap_consistency(records, observations, tracks)
    print("stage=validated_colmap_consistency", flush=True)
    loader_check, image_metadata = inspect_images(records, image_root, official_source)
    print("stage=validated_images_and_loader", flush=True)
    for record in records:
        record["image_file"] = image_metadata[record["registered_image_id"]]

    train_records = [record for record in records if record["split"] == "training"]
    evaluation_records = [record for record in records if record["split"] == "evaluation"]
    if len(train_records) != 726 or len(evaluation_records) != 15:
        raise ValueError(
            f"Unexpected split counts: training={len(train_records)}, evaluation={len(evaluation_records)}"
        )
    if {r["registered_image_id"] for r in train_records} & {
        r["registered_image_id"] for r in evaluation_records
    }:
        raise ValueError("Training and evaluation splits overlap")

    scene_center = np.median(xyz, axis=0)
    all_stats = camera_stats(records, scene_center)
    train_stats = camera_stats(train_records, scene_center)
    eval_stats = camera_stats(evaluation_records, scene_center)
    point_stats = {
        "count": len(xyz),
        "robust_scene_center_sparse_point_coordinate_median": scene_center.tolist(),
        "bbox_min": xyz.min(axis=0).tolist(),
        "bbox_max": xyz.max(axis=0).tolist(),
        "bbox_extent": (xyz.max(axis=0) - xyz.min(axis=0)).tolist(),
    }
    distances = np.asarray(
        [np.linalg.norm(np.asarray(r["camera_center"]) - scene_center) for r in records]
    )
    distance_q25, distance_q75 = np.quantile(distances, [0.25, 0.75])
    heights = np.asarray([r["camera_center"][2] for r in records])
    height_q25, height_q75 = np.quantile(heights, [0.25, 0.75])
    bbox_min = np.asarray(all_stats["camera_center_bbox_min"])
    bbox_max = np.asarray(all_stats["camera_center_bbox_max"])
    bbox_extent = bbox_max - bbox_min
    evaluation_cameras = []
    for sequence_index, record in enumerate(evaluation_records):
        center = np.asarray(record["camera_center"])
        distance = float(np.linalg.norm(center - scene_center))
        distance_category = "near" if distance <= distance_q25 else "far" if distance >= distance_q75 else "central"
        height = center[2]
        height_category = "low" if height <= height_q25 else "high" if height >= height_q75 else "middle"
        boundary = bool(
            np.any(center[:2] <= bbox_min[:2] + 0.1 * bbox_extent[:2])
            or np.any(center[:2] >= bbox_max[:2] - 0.1 * bbox_extent[:2])
        )
        evaluation_cameras.append(
            {
                **record,
                "evaluation_sequence_index": sequence_index,
                "camera_to_robust_scene_center_distance": distance,
                "selection_categories": {
                    "distance": distance_category,
                    "height": height_category,
                    "boundary": boundary,
                    "focal": "fixed-single-camera",
                },
                "selection_rationale": (
                    f"Official source zero-based stable name-sort index {record['stable_sorted_index']} "
                    f"satisfies index % {args.llff_hold} == 0."
                ),
            }
        )

    model_files = {
        "cameras_bin": file_record(camera_path, "sparse/0/cameras.bin"),
        "images_bin": file_record(images_path, "sparse/0/images.bin"),
        "points3d_bin": file_record(points_path, "sparse/0/points3D.bin"),
        "points3d_ply": file_record(ply_path, "sparse/0/points3D.ply"),
    }
    source_model_files = {
        key: file_record(source_root / value["relative_path"], value["relative_path"])
        for key, value in model_files.items()
        if key != "points3d_ply"
    }
    for key in source_model_files:
        if source_model_files[key]["sha256"] != model_files[key]["sha256"]:
            raise ValueError(f"Frozen COLMAP copy differs from source for {key}")

    registered_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "identity_definition": (
            "COLMAP registered image ID plus camera ID and relative image name; stable_sorted_index "
            "is derived using the pinned official source's name-only stable sort"
        ),
        "records": records,
    }
    camera_statistics = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "scene_center_definition": "coordinate-wise median of sparse COLMAP points",
        "sparse_points": point_stats,
        "all_registered_cameras": all_stats,
        "training_cameras": train_stats,
        "evaluation_cameras": eval_stats,
        "image_loader_compatibility": loader_check,
        "limitations": {
            "camera_models": ["PINHOLE"],
            "fixed_focal_length": True,
            "aerial_only": True,
            "camera_height_approximate_range": [1.5, 5.0],
            "camera_to_center_approximate_range": [2.25, 10.65],
            "street_level_approach_retreat": False,
            "large_aerial_to_ground_transition": False,
            "non_trivial_hierarchy_depth_variation": "unverified until a hierarchy exists",
        },
    }
    evaluation_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "split_id": "ALoD local oracle split",
        "llff_hold": args.llff_hold,
        "zero_based_indexing": True,
        "cameras": evaluation_cameras,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "registered-images.json", registered_payload)
    write_json(output_dir / "camera-statistics.json", camera_statistics)
    write_json(output_dir / "evaluation-cameras.json", evaluation_payload)
    write_identity_list(output_dir / "train-images.txt", train_records)
    write_identity_list(output_dir / "evaluation-images.txt", evaluation_records)
    print("stage=wrote_primary_metadata", flush=True)

    generated_files = {
        name: file_record(output_dir / name, name)
        for name in (
            "registered-images.json",
            "camera-statistics.json",
            "evaluation-cameras.json",
            "train-images.txt",
            "evaluation-images.txt",
        )
    }
    command = (
        "/home/luis/miniconda3/envs/alod_official/bin/python "
        "/home/luis/tese/gs_framework/scripts/alod/inspect_oracle_dataset.py "
        f"--dataset-root {dataset_root} --source-root {source_root} --output-dir {output_dir} "
        f"--official-source {official_source} --official-commit {args.official_commit} "
        f"--acquisition-date {args.acquisition_date} --llff-hold {args.llff_hold} --generate-ply"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_name": "MatrixCity Small City aerial block_all_test reconstruction",
        "dataset_role": (
            "first bounded official-source ALoD training smoke, .dhier generation, and "
            "format/selection/rendering oracle asset; not a performance baseline"
        ),
        "formal_evaluation_contamination_warning": CONTAMINATION_WARNING,
        "source_dataset": "MatrixCity Small City",
        "source_modality": "aerial",
        "source_partition": "original MatrixCity/CityGaussian test reconstruction",
        "source_registered_images": 741,
        "project_use": "repurposed for ALoD smoke training and oracle generation",
        "original_test_semantics_are_not_preserved": True,
        "repurposed_original_test_cameras_for_training": 726,
        "allowed_uses": ALLOWED_USES,
        "prohibited_uses": PROHIBITED_USES,
        "sources": {
            "matrixcity_project_url": "https://city-super.github.io/matrixcity/",
            "matrixcity_data_url": "https://huggingface.co/datasets/BoDai/MatrixCity",
            "citygaussian_repository_url": "https://github.com/Linketic/CityGaussian",
            "citygaussian_colmap_results_google_drive": "https://drive.google.com/file/d/1Uz1pSTIpkagTml2jzkkzJ_rglS_z34p7/view",
            "citygaussian_colmap_results_huggingface": "https://huggingface.co/datasets/dylanebert/CityGaussian",
            "local_source_path": str(source_root),
        },
        "license": {
            "matrixcity_images": "CC BY-NC 4.0",
            "matrixcity_license_reference": "https://city-super.github.io/matrixcity/",
            "citygaussian_colmap_reconstruction": "CC BY-NC-SA 4.0",
            "citygaussian_license_reference": "https://github.com/Linketic/CityGaussian/blob/main/LICENSE.md",
            "permitted_use": "non-commercial research with attribution; ShareAlike applies to CityGaussian-derived reconstruction material",
            "redistribution": (
                "Do not redistribute image or COLMAP payloads from this repository; preserve hashes "
                "and regeneration instructions. Any external redistribution must satisfy both source licenses."
            ),
        },
        "local_external_path": str(dataset_root),
        "acquisition_date": args.acquisition_date,
        "archive": {"present": False, "checksum": None, "reason": "frozen from an existing legal local copy"},
        "relative_paths": {
            "images": "images",
            "colmap_model": "sparse/0",
            "cameras": "sparse/0/cameras.bin",
            "registered_images": "sparse/0/images.bin",
            "sparse_points_binary": "sparse/0/points3D.bin",
            "sparse_points_ply": "sparse/0/points3D.ply",
        },
        "image_link": {
            "link_type": "absolute symbolic link",
            "link_text": os.readlink(image_link),
            "resolved_target": str(image_root),
            "images_duplicated": False,
            "reason": "avoid duplicating 3,030,193,206 bytes of third-party image payload",
            "external_dependency": str(source_root / "input"),
            "broken_link_validation": (
                "link resolves, every registered file exists, every PNG passes integrity verification, "
                "and all required representative images fully decode"
            ),
        },
        "file_count_summary": {
            "registered_images": len(records),
            "logical_image_files": len(image_metadata),
            "copied_colmap_binary_files": 3,
            "generated_sparse_ply_files": 1,
            "repository_metadata_files_excluding_readme": 6,
        },
        "colmap": {
            "camera_count": len(cameras),
            "camera_ids_in_file_order": camera_order,
            "camera_models": sorted({camera["model"] for camera in cameras.values()}),
            "registered_image_count": len(records),
            "sparse_point_count": len(xyz),
            "undistortion_status": "already undistorted; only PINHOLE model is present",
            "consistency_validation": consistency,
            "stable_identity_validation": identity_checks,
        },
        "split": {
            "split_id": "ALoD local oracle split",
            "official_source_file": "scene/dataset_readers.py",
            "official_source_function": "readColmapSceneInfo",
            "official_source_commit": args.official_commit,
            "semantics": "stable registered-image name sort; zero-based index; evaluation when index % llff_hold == 0; all remaining images train",
            "eval_resolved": True,
            "llff_hold": args.llff_hold,
            "train_test_exp": False,
            "zero_based_indexing": True,
            "training_count": len(train_records),
            "evaluation_count": len(evaluation_records),
            "training_registered_image_ids": [r["registered_image_id"] for r in train_records],
            "evaluation_registered_image_ids": [r["registered_image_id"] for r in evaluation_records],
            "complete_sorted_registered_image_id_order": [r["registered_image_id"] for r in records],
            "complete_sorted_registered_image_identity_order": [
                {
                    "stable_sorted_index": r["stable_sorted_index"],
                    "registered_image_id": r["registered_image_id"],
                    "camera_id": r["camera_id"],
                    "image_name": r["image_name"],
                }
                for r in records
            ],
        },
        "image_loader_compatibility": loader_check,
        "sparse_ply_generation": {
            "source_function": "scene.dataset_readers.storePly",
            "algorithm": "read points3D.bin in file order; emit x/y/z float32, zero nx/ny/nz float32, and red/green/blue uint8 using plyfile binary little-endian output",
            "input": model_files["points3d_bin"],
            "output": model_files["points3d_ply"],
            "point_count": len(xyz),
            "property_layout": ["x:f4", "y:f4", "z:f4", "nx:f4", "ny:f4", "nz:f4", "red:u1", "green:u1", "blue:u1"],
            "command": command,
            "repeated_generation_byte_identical": True,
        },
        "content_hashes": {
            "frozen_colmap_and_ply": model_files,
            "source_colmap": source_model_files,
            "registered_image_hashes_location": "registered-images.json records[*].image_file.sha256",
            "generated_metadata": generated_files,
        },
        "excluded_files": [
            "source transforms JSON",
            "estimated depths and depth-scale metadata",
            "input_1.2 duplicate image tree",
            "masks",
            "training outputs, checkpoints, Gaussian PLYs, and .dhier assets",
        ],
        "manifest_generation_command": command,
        "suitability": {
            "training_pipeline_validation": "suitable",
            ".dhier_validation": "suitable",
            "hierarchy_selection_oracle_creation": "conditionally suitable; non-trivial depth/LoD variation remains unverified until a hierarchy exists",
            "cache_behavior_development": "suitable",
            "formal_LoD_evaluation": "insufficient",
            "city_scale_claims": "insufficient",
            "mixed_scale_navigation_claims": "insufficient",
        },
        "future_dataset_required": True,
    }
    write_json(output_dir / "dataset-manifest.json", manifest)
    print("stage=wrote_manifest", flush=True)
    print(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "registered_images": len(records),
                "training": len(train_records),
                "evaluation": len(evaluation_records),
                "sparse_points": len(xyz),
                "image_bytes": loader_check["total_bytes"],
                "output_dir": str(output_dir),
                "cuda_initialized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
