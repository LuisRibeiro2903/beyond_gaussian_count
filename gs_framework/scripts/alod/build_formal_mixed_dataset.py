#!/usr/bin/env python3
"""Build the frozen public-source mixed-scale MatrixCity ALoD proxy.

The builder deliberately uses pose/intrinsic metadata only for camera selection.
It never reads image pixels until the selected identities have been frozen.  The
official ALoD loader's llff_hold split is encoded in lexicographic filenames so
the released train/test identities remain deterministic without patching the
pinned source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image as PILImage


DATASET_ID = "ALOD-REPRO-MC-MIXED-01"
SCHEMA_VERSION = 1
STREET_TRAIN_COUNT = 330
STREET_EVAL_COUNT = 45
AERIAL_TRAIN_COUNT = 309
AERIAL_EVAL_COUNT = 26
LLFF_HOLD = 10
DIAGNOSTIC_PER_MODALITY = 8


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_id(modality: str, source_split: str, file_path: str) -> str:
    return f"matrixcity-small-city:{modality}:{source_split}:{file_path}"


def normalized_c2w(frame: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(frame["transform_matrix"], dtype=np.float64).copy()
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid transform for {frame.get('file_path')}")
    matrix[3] = (0.0, 0.0, 0.0, 1.0)
    rotation = matrix[:3, :3]
    u, _, vt = np.linalg.svd(rotation)
    matrix[:3, :3] = u @ vt
    if np.linalg.det(matrix[:3, :3]) < 0:
        raise ValueError(f"left-handed transform for {frame.get('file_path')}")
    return matrix


def load_source_frames(
    path: Path,
    modality: str,
    source_split: str,
    image_root: Path,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    records: list[dict[str, Any]] = []
    for source_index, frame in enumerate(payload["frames"]):
        raw_path = frame["file_path"]
        normalized_path = raw_path.replace("\\", "/")
        if modality == "street" and "/street/" not in normalized_path:
            # The released fusion pose files contain both modalities; the
            # local aerial pool is audited independently below.
            continue
        c2w = normalized_c2w(frame)
        if modality == "aerial":
            image_path = (path.parent / raw_path).resolve()
            if not image_path.is_file():
                image_path = image_path.parent / "input" / image_path.name
            intr = {
                key: float(frame.get(key, payload[key]))
                for key in ("fl_x", "fl_y", "cx", "cy", "w", "h")
            }
        else:
            relative = normalized_path
            marker = "../street/"
            if not relative.startswith(marker):
                raise ValueError(f"unexpected released street path: {raw_path}")
            image_path = (image_root / relative[len(marker) :]).resolve()
            intr = {key: float(frame[key]) for key in ("fl_x", "fl_y", "cx", "cy", "w", "h")}
        forward = -c2w[:3, 2]
        forward /= np.linalg.norm(forward)
        records.append(
            {
                "source_id": source_id(modality, source_split, raw_path),
                "source_index": source_index,
                "source_file_path": raw_path,
                "source_pose_file": str(path),
                "source_image_path": str(image_path),
                "modality": modality,
                "source_split": source_split,
                "c2w": c2w,
                "centre": c2w[:3, 3],
                "forward": forward,
                "intrinsics": intr,
            }
        )
    return records


def deterministic_farthest(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count > len(records):
        raise ValueError(f"requested {count} records from pool of {len(records)}")
    if count == len(records):
        return sorted(records, key=lambda row: row["source_id"])
    ordered = sorted(records, key=lambda row: row["source_id"])
    features = np.asarray(
        [np.r_[row["centre"], row["forward"]] for row in ordered], dtype=np.float64
    )
    scale = np.std(features, axis=0)
    scale[scale < 1e-9] = 1.0
    features = (features - np.mean(features, axis=0)) / scale
    centroid = np.mean(features, axis=0)
    first = int(np.argmax(np.sum((features - centroid) ** 2, axis=1)))
    chosen = [first]
    min_distance = np.sum((features - features[first]) ** 2, axis=1)
    min_distance[first] = -1.0
    while len(chosen) < count:
        index = int(np.argmax(min_distance))
        chosen.append(index)
        distance = np.sum((features - features[index]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[chosen] = -1.0
    return [ordered[index] for index in chosen]


def regime_quotas(total: int) -> dict[str, int]:
    base, remainder = divmod(total, 3)
    names = ("near", "mid", "far")
    return {name: base + (1 if i < remainder else 0) for i, name in enumerate(names)}


def label_aerial_regimes(records: list[dict[str, Any]], scene_reference: np.ndarray) -> None:
    distances = np.asarray([np.linalg.norm(row["centre"] - scene_reference) for row in records])
    q1, q2 = np.quantile(distances, (1 / 3, 2 / 3))
    for row, distance in zip(records, distances):
        row["camera_to_reference_distance"] = float(distance)
        row["distance_stratum"] = "near" if distance <= q1 else "mid" if distance <= q2 else "far"


def select_stratified(
    records: list[dict[str, Any]], count: int, scene_reference: np.ndarray
) -> list[dict[str, Any]]:
    label_aerial_regimes(records, scene_reference)
    selected: list[dict[str, Any]] = []
    for regime, quota in regime_quotas(count).items():
        pool = [row for row in records if row["distance_stratum"] == regime]
        selected.extend(deterministic_farthest(pool, quota))
    if len({row["source_id"] for row in selected}) != count:
        raise AssertionError("stratified selection produced duplicate identities")
    return selected


def assign_common_metrics(records: list[dict[str, Any]], scene_reference: np.ndarray) -> None:
    for modality in ("aerial", "street"):
        modality_records = [row for row in records if row["modality"] == modality]
        radial = np.asarray(
            [np.linalg.norm(row["centre"][:2] - scene_reference[:2]) for row in modality_records]
        )
        boundary_threshold = float(np.quantile(radial, 0.67))
        for row, radial_distance in zip(modality_records, radial):
            forward = row["forward"]
            distance = float(np.linalg.norm(row["centre"] - scene_reference))
            row["camera_to_scene_centre_distance"] = distance
            row["radial_xy_distance"] = float(radial_distance)
            row["region_class"] = "boundary" if radial_distance >= boundary_threshold else "central"
            row["yaw_degrees"] = float(math.degrees(math.atan2(forward[1], forward[0])))
            row["pitch_degrees"] = float(
                math.degrees(math.asin(np.clip(forward[2], -1.0, 1.0)))
            )
            row["projected_scale_proxy"] = float(
                row["intrinsics"]["fl_x"] / max(distance, 1e-9)
            )


def split_order(eval_records: list[dict[str, Any]], train_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(eval_records) + len(train_records)
    eval_slots = list(range(0, total, LLFF_HOLD))
    if len(eval_slots) != len(eval_records):
        raise ValueError(f"llff_hold={LLFF_HOLD} gives {len(eval_slots)} slots, need {len(eval_records)}")
    ordered_eval = sorted(eval_records, key=lambda row: row["source_id"])
    ordered_train = sorted(train_records, key=lambda row: row["source_id"])
    output: list[dict[str, Any] | None] = [None] * total
    for slot, row in zip(eval_slots, ordered_eval):
        output[slot] = row
    iterator = iter(ordered_train)
    for index in range(total):
        if output[index] is None:
            output[index] = next(iterator)
    try:
        next(iterator)
        raise AssertionError("unconsumed training identity")
    except StopIteration:
        pass
    return [row for row in output if row is not None]


def c2w_to_colmap(c2w_nerf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w_colmap = c2w_nerf @ axis
    rotation = c2w_colmap[:3, :3].T
    translation = -rotation @ c2w_colmap[:3, 3]
    return rotation, translation


def write_colmap_cameras(path: Path, camera_defs: dict[int, dict[str, float]]) -> None:
    # COLMAP binary PINHOLE model id is 1 and has fx, fy, cx, cy.
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(camera_defs)))
        for camera_id, intr in sorted(camera_defs.items()):
            handle.write(struct.pack("<iiQQ", camera_id, 1, int(intr["w"]), int(intr["h"])))
            handle.write(struct.pack("<dddd", intr["fl_x"], intr["fl_y"], intr["cx"], intr["cy"]))


def rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    matrix = np.asarray(
        [
            [rxx - ryy - rzz, ryx + rxy, rzx + rxz, ryz - rzy],
            [ryx + rxy, ryy - rxx - rzz, rzy + ryz, rzx - rxz],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, rxy - ryx],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ]
    ) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    qvec = eigenvectors[[3, 0, 1, 2], np.argmax(eigenvalues)]
    return -qvec if qvec[0] < 0 else qvec


def safe_name(index: int, row: dict[str, Any]) -> str:
    original = Path(row["source_file_path"]).stem
    return f"{index:06d}__{row['modality']}__{row['source_split']}__{original}.png"


def write_colmap_images(path: Path, ordered: list[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(ordered)))
        for index, row in enumerate(ordered):
            rotation, translation = c2w_to_colmap(row["c2w"])
            qvec = rotmat_to_qvec(rotation)
            camera_id = 1 if row["modality"] == "aerial" else 2
            name = row["dataset_filename"].encode()
            handle.write(struct.pack("<i", index + 1))
            handle.write(struct.pack("<dddd", *qvec))
            handle.write(struct.pack("<ddd", *translation))
            handle.write(struct.pack("<i", camera_id))
            handle.write(name + b"\x00")
            handle.write(struct.pack("<Q", 0))


def verify_aerial_geometry_frame(
    selected: list[dict[str, Any]], source_images_bins: list[Path]
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from read_write_model import qvec2rotmat, read_images_binary  # type: ignore

    source_images = []
    for source_images_bin in source_images_bins:
        source_images.extend(read_images_binary(str(source_images_bin)).values())
    source_centres = np.asarray(
        [-qvec2rotmat(image.qvec).T @ image.tvec for image in source_images], dtype=np.float64
    )
    errors = []
    for row in selected:
        errors.append(float(np.min(np.linalg.norm(source_centres - row["centre"], axis=1))))
    result = {
        "source_registered_images": len(source_images),
        "source_images_bins": [str(path) for path in source_images_bins],
        "selected_aerial_views_checked": len(selected),
        "nearest_source_colmap_centre_error": summary(errors),
        "threshold": 1e-5,
        "result": "PASS" if max(errors) <= 1e-5 else "FAIL",
    }
    if result["result"] != "PASS":
        raise ValueError(f"aerial pose/COLMAP frame mismatch: max error {max(errors)}")
    return result


def stream_crop_points(
    source: Path, output_bin: Path, output_ply: Path, xy_bounds: tuple[np.ndarray, np.ndarray]
) -> tuple[int, np.ndarray, dict[str, Any]]:
    lower, upper = xy_bounds
    body_bin = Path(f"{output_bin}.body.tmp")
    body_ply = Path(f"{output_ply}.body.tmp")
    sample: list[tuple[float, float, float]] = []
    with source.open("rb") as src, body_bin.open("wb") as dst_bin, body_ply.open("wb") as dst_ply:
        source_count = struct.unpack("<Q", src.read(8))[0]
        sample_stride = max(1, source_count // 250_000)
        selected_count = 0
        min_xyz = np.full(3, np.inf)
        max_xyz = np.full(3, -np.inf)
        for source_index in range(source_count):
            point_id = struct.unpack("<Q", src.read(8))[0]
            xyz = struct.unpack("<ddd", src.read(24))
            rgb = struct.unpack("<BBB", src.read(3))
            error = struct.unpack("<d", src.read(8))[0]
            track_length = struct.unpack("<Q", src.read(8))[0]
            src.seek(track_length * 8, os.SEEK_CUR)
            if lower[0] <= xyz[0] <= upper[0] and lower[1] <= xyz[1] <= upper[1]:
                dst_bin.write(struct.pack("<QdddBBBdQ", point_id, *xyz, *rgb, error, 0))
                dst_ply.write(struct.pack("<ffffffBBB", *xyz, 0.0, 0.0, 0.0, *rgb))
                selected_count += 1
                xyz_np = np.asarray(xyz)
                min_xyz = np.minimum(min_xyz, xyz_np)
                max_xyz = np.maximum(max_xyz, xyz_np)
                if source_index % sample_stride == 0:
                    sample.append(xyz)
    if selected_count == 0:
        raise ValueError("point crop is empty")
    with output_bin.open("wb") as dst, body_bin.open("rb") as body:
        dst.write(struct.pack("<Q", selected_count))
        shutil.copyfileobj(body, dst)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {selected_count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode()
    with output_ply.open("wb") as dst, body_ply.open("rb") as body:
        dst.write(header)
        shutil.copyfileobj(body, dst)
    body_bin.unlink()
    body_ply.unlink()
    return selected_count, np.asarray(sample, dtype=np.float64), {
        "source_count": source_count,
        "selected_count": selected_count,
        "xyz_min": min_xyz.tolist(),
        "xyz_max": max_xyz.tolist(),
        "xy_crop_min": lower.tolist(),
        "xy_crop_max": upper.tolist(),
    }


def nearest_sample_distances(records: list[dict[str, Any]], points: np.ndarray) -> list[float]:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        distances, _ = tree.query(np.asarray([row["centre"] for row in records]), k=1)
        return [float(value) for value in distances]
    except ImportError:
        result = []
        for row in records:
            centre = row["centre"]
            best = math.inf
            for start in range(0, len(points), 50_000):
                best = min(best, float(np.min(np.linalg.norm(points[start : start + 50_000] - centre, axis=1))))
            result.append(best)
        return result


def summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def diagnostic_selection(eval_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    aerial_quotas = {"near": 3, "mid": 2, "far": 3}
    for stratum, count in aerial_quotas.items():
        pool = [
            row
            for row in eval_records
            if row["modality"] == "aerial" and row["distance_stratum"] == stratum
        ]
        selected.extend(deterministic_farthest(pool, count))
    for region_class in ("central", "boundary"):
        pool = [
            row
            for row in eval_records
            if row["modality"] == "street" and row["region_class"] == region_class
        ]
        selected.extend(deterministic_farthest(pool, 4))
    return sorted(selected, key=lambda row: row["dataset_index"])


def public_record(row: dict[str, Any]) -> dict[str, Any]:
    rotation, translation = c2w_to_colmap(row["c2w"])
    return {
        "dataset_index": row["dataset_index"],
        "registered_image_id": row["registered_image_id"],
        "dataset_filename": row["dataset_filename"],
        "source_id": row["source_id"],
        "source_index": row["source_index"],
        "source_file_path": row["source_file_path"],
        "source_pose_file": row["source_pose_file"],
        "modality": row["modality"],
        "source_split": row["source_split"],
        "split": row["split"],
        "distance_stratum": row["distance_stratum"],
        "region_class": row["region_class"],
        "intrinsic_identity": "matrixcity-aerial-pinhole" if row["modality"] == "aerial" else "matrixcity-street-pinhole",
        "intrinsics": row["intrinsics"],
        "c2w_nerf": row["c2w"].tolist(),
        "camera_centre": row["centre"].tolist(),
        "colmap_world_to_camera_rotation": rotation.tolist(),
        "colmap_world_to_camera_translation": translation.tolist(),
        "camera_to_scene_centre_distance": row["camera_to_scene_centre_distance"],
        "camera_height": row["camera_height"],
        "nearest_scene_distance_proxy": row["nearest_scene_distance_proxy"],
        "projected_scale_proxy": row["projected_scale_proxy"],
        "projected_nearest_geometry_scale_proxy": row[
            "projected_nearest_geometry_scale_proxy"
        ],
        "yaw_degrees": row["yaw_degrees"],
        "pitch_degrees": row["pitch_degrees"],
        "image": row["image"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--aerial-root", type=Path, required=True)
    parser.add_argument("--street-pose-root", type=Path, required=True)
    parser.add_argument("--street-image-root", type=Path, required=True)
    parser.add_argument("--source-points", type=Path, required=True)
    parser.add_argument("--source-images-bin", type=Path, required=True)
    parser.add_argument("--source-test-images-bin", type=Path, required=True)
    args = parser.parse_args()

    output_images = args.dataset_root / "images"
    output_sparse = args.dataset_root / "sparse" / "0"
    for path in (output_images, output_sparse, args.metadata_root):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"refusing to overwrite populated frozen output: {path}")
        path.mkdir(parents=True, exist_ok=True)

    aerial_pose_root = args.aerial_root / "pose" / "block_all"
    aerial_train = load_source_frames(
        aerial_pose_root / "transforms_train.json", "aerial", "train", args.aerial_root
    )
    aerial_eval = load_source_frames(
        aerial_pose_root / "transforms_test.json", "aerial", "test", args.aerial_root
    )
    street_train = load_source_frames(
        args.street_pose_root / "transforms_train.json", "street", "train", args.street_image_root
    )
    street_eval = load_source_frames(
        args.street_pose_root / "transforms_test.json", "street", "test", args.street_image_root
    )
    if (len(street_train), len(street_eval)) != (STREET_TRAIN_COUNT, STREET_EVAL_COUNT):
        raise ValueError("released street source counts changed")

    street_all = street_train + street_eval
    street_centres = np.asarray([row["centre"] for row in street_all])
    scene_reference = np.median(street_centres, axis=0)
    street_xy_min = np.min(street_centres[:, :2], axis=0)
    street_xy_max = np.max(street_centres[:, :2], axis=0)
    street_diagonal = float(np.linalg.norm(street_xy_max - street_xy_min))

    # Pick the smallest street-footprint-relative radius that leaves at least
    # 3x selection headroom in both source partitions and exposes the released
    # low (<=2) and high (>=5) aerial height regimes.  The discrete search is
    # deterministic and fails instead of silently dropping a regime.
    overlap_audit = []
    aerial_train_pool: list[dict[str, Any]] | None = None
    aerial_eval_pool: list[dict[str, Any]] | None = None
    overlap_radius = math.nan
    for multiplier in np.arange(1.0, 4.01, 0.25):
        radius = float(multiplier * street_diagonal)
        pools = [
            [row for row in rows if np.linalg.norm(row["centre"][:2] - scene_reference[:2]) <= radius]
            for rows in (aerial_train, aerial_eval)
        ]
        heights = [sorted({round(float(row["centre"][2]), 6) for row in pool}) for pool in pools]
        accepted = (
            len(pools[0]) >= 3 * AERIAL_TRAIN_COUNT
            and len(pools[1]) >= 3 * AERIAL_EVAL_COUNT
            and all(values and min(values) <= 2.0 and max(values) >= 5.0 for values in heights)
        )
        overlap_audit.append(
            {
                "multiplier": float(multiplier),
                "radius": radius,
                "train_candidates": len(pools[0]),
                "eval_candidates": len(pools[1]),
                "train_heights": heights[0],
                "eval_heights": heights[1],
                "accepted": accepted,
            }
        )
        if accepted:
            overlap_radius = radius
            aerial_train_pool, aerial_eval_pool = pools
            break
    if aerial_train_pool is None or aerial_eval_pool is None:
        raise ValueError("no deterministic overlap radius preserves all required aerial regimes")
    selected_aerial_train = select_stratified(aerial_train_pool, AERIAL_TRAIN_COUNT, scene_reference)
    selected_aerial_eval = select_stratified(aerial_eval_pool, AERIAL_EVAL_COUNT, scene_reference)
    selected_aerial = selected_aerial_train + selected_aerial_eval
    alignment = verify_aerial_geometry_frame(
        selected_aerial, [args.source_images_bin, args.source_test_images_bin]
    )

    for row in street_all:
        distance = float(np.linalg.norm(row["centre"] - scene_reference))
        row["camera_to_reference_distance"] = distance
        row["distance_stratum"] = "street"
    train_records = selected_aerial_train + street_train
    eval_records = selected_aerial_eval + street_eval
    all_selected = train_records + eval_records
    assign_common_metrics(all_selected, scene_reference)
    ordered = split_order(eval_records, train_records)
    for index, row in enumerate(ordered):
        row["dataset_index"] = index
        row["registered_image_id"] = index + 1
        row["dataset_filename"] = safe_name(index, row)
        row["split"] = "eval" if index % LLFF_HOLD == 0 else "train"
        expected = "eval" if row["source_id"] in {item["source_id"] for item in eval_records} else "train"
        if row["split"] != expected:
            raise AssertionError("encoded llff split does not match frozen source split")

    camera_defs = {
        1: selected_aerial[0]["intrinsics"],
        2: street_all[0]["intrinsics"],
    }
    write_colmap_cameras(output_sparse / "cameras.bin", camera_defs)
    write_colmap_images(output_sparse / "images.bin", ordered)

    selected_xy = np.asarray([row["centre"][:2] for row in ordered])
    selected_min, selected_max = np.min(selected_xy, axis=0), np.max(selected_xy, axis=0)
    geometry_padding = max(0.1 * float(np.linalg.norm(selected_max - selected_min)), 0.25)
    point_count, point_sample, geometry = stream_crop_points(
        args.source_points,
        output_sparse / "points3D.bin",
        output_sparse / "points3D.ply",
        (selected_min - geometry_padding, selected_max + geometry_padding),
    )
    geometry_centre = np.median(point_sample, axis=0)
    ground = float(np.quantile(point_sample[:, 2], 0.05))
    nearest = nearest_sample_distances(ordered, point_sample)
    for row, nearest_distance in zip(ordered, nearest):
        row["camera_height"] = float(row["centre"][2] - ground)
        row["nearest_scene_distance_proxy"] = nearest_distance
        row["projected_nearest_geometry_scale_proxy"] = float(
            row["intrinsics"]["fl_x"] / max(nearest_distance, 1e-9)
        )

    extension_counts: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    for row in ordered:
        source_image = Path(row["source_image_path"])
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        link = output_images / row["dataset_filename"]
        link.symlink_to(source_image)
        image_hash = sha256_file(source_image)
        with PILImage.open(source_image) as image:
            image.verify()
        with PILImage.open(source_image) as image:
            width, height = image.size
            mode = image.mode
        expected = (int(row["intrinsics"]["w"]), int(row["intrinsics"]["h"]))
        if (width, height) != expected:
            raise ValueError(f"image/intrinsic resolution mismatch for {source_image}")
        row["image"] = {
            "relative_path": f"images/{row['dataset_filename']}",
            "source_path": str(source_image),
            "sha256": image_hash,
            "size_bytes": source_image.stat().st_size,
            "width": width,
            "height": height,
            "mode": mode,
        }
        extension_counts[source_image.suffix.lower()] += 1
        dimensions[f"{width}x{height}"] += 1

    eval_final = [row for row in ordered if row["split"] == "eval"]
    diagnostics = diagnostic_selection(eval_final)
    public_records = [public_record(row) for row in ordered]
    split_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "split_method": "released source train/test identities encoded at every tenth lexicographic filename",
        "official_llff_hold": LLFF_HOLD,
        "counts": dict(Counter(row["split"] for row in ordered)),
        "records": public_records,
    }
    diagnostic_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "selection_method": "frozen-eval-only deterministic farthest-point coverage in standardized position and viewing direction: aerial 3/2/3 near/mid/far; street 4/4 central/boundary",
        "count": len(diagnostics),
        "cameras": [public_record(row) for row in diagnostics],
    }

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        groups["all"].append(row)
        groups[row["modality"]].append(row)
        groups[row["split"]].append(row)
        groups[f"{row['modality']}:{row['split']}"] .append(row)
        groups[f"stratum:{row['distance_stratum']}"] .append(row)
    stats_groups = {}
    for name, rows in sorted(groups.items()):
        stats_groups[name] = {
            "count": len(rows),
            "camera_to_scene_centre_distance": summary(row["camera_to_scene_centre_distance"] for row in rows),
            "camera_height": summary(row["camera_height"] for row in rows),
            "nearest_scene_distance_proxy": summary(row["nearest_scene_distance_proxy"] for row in rows),
            "focal_length_pixels": summary(row["intrinsics"]["fl_x"] for row in rows),
            "projected_scale_proxy": summary(row["projected_scale_proxy"] for row in rows),
            "projected_nearest_geometry_scale_proxy": summary(
                row["projected_nearest_geometry_scale_proxy"] for row in rows
            ),
            "yaw_degrees": summary(row["yaw_degrees"] for row in rows),
            "pitch_degrees": summary(row["pitch_degrees"] for row in rows),
        }
    statistics_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "status": "PASS-ALOD-MIXED-SCALE-CAMERA-COVERAGE",
        "scene_reference_from_street_camera_median": scene_reference.tolist(),
        "sampled_sparse_geometry_centre": geometry_centre.tolist(),
        "sampled_sparse_ground_z_p05": ground,
        "aerial_overlap": {
            "street_xy_min": street_xy_min.tolist(),
            "street_xy_max": street_xy_max.tolist(),
            "street_xy_diagonal": street_diagonal,
            "selected_radius_about_street_median": overlap_radius,
            "aerial_train_candidates": len(aerial_train_pool),
            "aerial_eval_candidates": len(aerial_eval_pool),
            "deterministic_radius_audit": overlap_audit,
        },
        "groups": stats_groups,
        "modality_counts": dict(Counter(row["modality"] for row in ordered)),
        "stratum_counts": dict(Counter(row["distance_stratum"] for row in ordered)),
        "region_counts": dict(Counter(row["region_class"] for row in ordered)),
        "dimension_distribution": dict(dimensions),
        "extension_distribution": dict(extension_counts),
    }
    provenance_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "classification": "mixed-scale MatrixCity proxy for faithful public-source reproduction; not the unreleased exact ALoD paper dataset",
        "license": "CC BY-NC 4.0",
        "source_repository": "https://github.com/city-super/MatrixCity",
        "source_dataset": "https://huggingface.co/datasets/BoDai/MatrixCity",
        "street_release": {
            "identity": "aerial_street_fusion/street.tar and pose/transforms_{train,test}.json",
            "archive_path": str(args.dataset_root / "source" / "archive" / "street.tar"),
            "archive_sha256": sha256_file(args.dataset_root / "source" / "archive" / "street.tar"),
            "pose_hashes": {
                path.name: sha256_file(path)
                for path in sorted(args.street_pose_root.glob("transforms_*.json"))
            },
            "coordinate_evidence": "The released MatrixCity merge_aerial_street.py concatenates normalized aerial and street c2w poses without a transform.",
        },
        "aerial_release": {
            "identity": "small_city/aerial block_all official pose split and RGB payload",
            "root": str(args.aerial_root),
            "pose_hashes": {
                path.name: sha256_file(path)
                for path in sorted(aerial_pose_root.glob("transforms_*.json"))
            },
        },
        "sparse_geometry": {
            "identity": "public-source MatrixCity aerial COLMAP sparse geometry",
            "source_path": str(args.source_points),
            "source_sha256": sha256_file(args.source_points),
            "track_policy": "crop by selected camera XY envelope and remove source-image track references; XYZ/RGB/error remain unchanged",
            "crop": geometry,
            "alignment_validation": alignment,
        },
        "coordinate_system": {
            "world_units": "100 m normalized MatrixCity units",
            "pose_source": "released MatrixCity c2w",
            "camera_convention": "NeRF x-right/y-up/z-back converted exactly to COLMAP x-right/y-down/z-forward",
            "alignment_transform": "identity; no manual or visual alignment",
        },
    }

    write_json(args.metadata_root / "split.json", split_payload)
    write_json(args.metadata_root / "diagnostic_cameras.json", diagnostic_payload)
    write_json(args.metadata_root / "camera_statistics.json", statistics_payload)
    write_json(args.metadata_root / "source_provenance.json", provenance_payload)
    generated_hashes = {
        path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(args.metadata_root.glob("*.json"))
    }
    colmap_hashes = {
        path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(output_sparse.iterdir())
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_name": "MatrixCity Small City mixed aerial and street public-source proxy",
        "classification": provenance_payload["classification"],
        "construction_version": "gs_framework-v1",
        "construction_command": "python scripts/alod/build_formal_mixed_dataset.py --dataset-root /home/luis/tese/data/alod/ALOD-REPRO-MC-MIXED-01 --metadata-root datasets/alod/ALOD-REPRO-MC-MIXED-01 --aerial-root /home/luis/tese/data/matrix_city/aerial --street-pose-root /home/luis/tese/data/alod/ALOD-REPRO-MC-MIXED-01/source/pose --street-image-root /home/luis/tese/data/alod/ALOD-REPRO-MC-MIXED-01/source/street --source-points /home/luis/tese/data/matrix_city/aerial/train/block_all/sparse/0/points3D.bin --source-images-bin /home/luis/tese/data/matrix_city/aerial/train/block_all/sparse/0/images.bin --source-test-images-bin /home/luis/tese/data/matrix_city/aerial/test/block_all_test/sparse/0/images.bin",
        "counts": {
            "cameras": len(ordered),
            "aerial": sum(row["modality"] == "aerial" for row in ordered),
            "street": sum(row["modality"] == "street" for row in ordered),
            "train": sum(row["split"] == "train" for row in ordered),
            "eval": sum(row["split"] == "eval" for row in ordered),
            "diagnostic": len(diagnostics),
            "sparse_points": point_count,
        },
        "split": {"method": split_payload["split_method"], "official_llff_hold": LLFF_HOLD},
        "camera_models": ["PINHOLE"],
        "camera_count": 2,
        "source_licenses": ["MatrixCity dataset: CC BY-NC 4.0"],
        "source_identities": [
            "BoDai/MatrixCity aerial_street_fusion street release",
            "BoDai/MatrixCity Small City aerial block_all release",
            "public-source MatrixCity aerial COLMAP sparse geometry",
        ],
        "content_hashes": {"metadata": generated_hashes, "colmap": colmap_hashes},
        "image_hashes_location": "split.json records[*].image.sha256",
        "payload_policy": "image payload remains external to Git; dataset images are immutable symlinks to preserved source payload",
        "status": "PASS-ALOD-MIXED-SCALE-CAMERA-COVERAGE",
    }
    write_json(args.metadata_root / "manifest.json", manifest)
    manifest_hash = sha256_file(args.metadata_root / "manifest.json")
    (args.metadata_root / "manifest.sha256").write_text(f"{manifest_hash}  manifest.json\n")
    for name in (
        "manifest.json",
        "manifest.sha256",
        "split.json",
        "diagnostic_cameras.json",
        "camera_statistics.json",
        "source_provenance.json",
    ):
        shutil.copy2(args.metadata_root / name, args.dataset_root / name)
    print(json.dumps({"dataset_id": DATASET_ID, "manifest_sha256": manifest_hash, "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
