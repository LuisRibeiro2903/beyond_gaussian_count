#!/usr/bin/env python3
"""Build a memory-safe official-layout Uni10K dataset for ALoD preflight.

The public archive stores a large COLMAP reconstruction with 2D observations and
point tracks.  The pinned ALoD loader does not use those correspondences, but it
does eagerly materialize them.  This builder therefore copies all images and
camera metadata losslessly while replacing observations/tracks with empty lists.
It also emits the exact sparse XYZ/RGB values as a binary PLY so the official
loader never needs to parse the track-bearing source point file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
from typing import BinaryIO


SOURCE_COMMIT = "811af1acf8d274ac5d59b498a89ba9dd8c523f86"
DATASET_ID = "ALOD-REPRO-UNI10K-01"


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_exact(handle: BinaryIO, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise EOFError(f"expected {size} bytes, read {len(value)}")
    return value


def read_c_string(handle: BinaryIO) -> bytes:
    value = bytearray()
    while True:
        char = read_exact(handle, 1)
        if char == b"\0":
            return bytes(value)
        value.extend(char)


def image_semantic_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        count_raw = read_exact(handle, 8)
        count = struct.unpack("<Q", count_raw)[0]
        digest.update(count_raw)
        for _ in range(count):
            header = read_exact(handle, 64)
            raw_name = read_c_string(handle)
            observation_count = struct.unpack("<Q", read_exact(handle, 8))[0]
            handle.seek(24 * observation_count, os.SEEK_CUR)
            digest.update(header)
            digest.update(raw_name + b"\0")
        if handle.read(1):
            raise ValueError(f"unexpected trailing bytes in {path}")
    return count, digest.hexdigest()


def point_semantic_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        count_raw = read_exact(handle, 8)
        count = struct.unpack("<Q", count_raw)[0]
        digest.update(count_raw)
        for _ in range(count):
            fixed = read_exact(handle, 43)
            track_count = struct.unpack("<Q", read_exact(handle, 8))[0]
            handle.seek(8 * track_count, os.SEEK_CUR)
            digest.update(fixed)
        if handle.read(1):
            raise ValueError(f"unexpected trailing bytes in {path}")
    return count, digest.hexdigest()


def strip_image_observations(source: Path, target: Path) -> dict[str, object]:
    names: list[str] = []
    camera_counts: dict[int, int] = {}
    source_observations = 0
    semantic_digest = hashlib.sha256()
    with source.open("rb") as src, target.open("wb") as dst:
        count_raw = read_exact(src, 8)
        count = struct.unpack("<Q", count_raw)[0]
        dst.write(count_raw)
        semantic_digest.update(count_raw)
        for _ in range(count):
            header = read_exact(src, 64)
            camera_id = struct.unpack("<i", header[-4:])[0]
            raw_name = read_c_string(src)
            observation_count_raw = read_exact(src, 8)
            observation_count = struct.unpack("<Q", observation_count_raw)[0]
            src.seek(24 * observation_count, os.SEEK_CUR)
            dst.write(header)
            dst.write(raw_name + b"\0")
            dst.write(struct.pack("<Q", 0))
            semantic_digest.update(header)
            semantic_digest.update(raw_name + b"\0")
            name = raw_name.decode("utf-8")
            names.append(name)
            camera_counts[camera_id] = camera_counts.get(camera_id, 0) + 1
            source_observations += observation_count
        if src.read(1):
            raise ValueError("unexpected trailing bytes in source images.bin")
    sorted_names = sorted(names)
    return {
        "registered_images": count,
        "source_observations_removed": source_observations,
        "camera_id_counts": {str(key): value for key, value in sorted(camera_counts.items())},
        "aerial_images": sum(name.startswith("aerial/") for name in names),
        "ground_images": sum(name.startswith("ground/") for name in names),
        "llff_hold": 8,
        "train_images": count - len(sorted_names[::8]),
        "evaluation_images": len(sorted_names[::8]),
        "first_sorted_name": sorted_names[0],
        "last_sorted_name": sorted_names[-1],
        "ordered_name_sha256": hashlib.sha256("\n".join(sorted_names).encode()).hexdigest(),
        "registered_image_semantic_sha256": semantic_digest.hexdigest(),
    }


def strip_point_tracks_and_write_ply(
    source: Path, target_bin: Path, target_ply: Path
) -> dict[str, object]:
    body_path = target_ply.with_suffix(".body.tmp")
    minimum = [float("inf")] * 3
    maximum = [float("-inf")] * 3
    source_tracks = 0
    semantic_digest = hashlib.sha256()
    with source.open("rb") as src, target_bin.open("wb") as dst, body_path.open("wb") as body:
        count_raw = read_exact(src, 8)
        count = struct.unpack("<Q", count_raw)[0]
        dst.write(count_raw)
        semantic_digest.update(count_raw)
        for _ in range(count):
            fixed = read_exact(src, 43)
            point_id, x, y, z, red, green, blue, error = struct.unpack("<QdddBBBd", fixed)
            track_count_raw = read_exact(src, 8)
            track_count = struct.unpack("<Q", track_count_raw)[0]
            src.seek(8 * track_count, os.SEEK_CUR)
            dst.write(fixed)
            dst.write(struct.pack("<Q", 0))
            semantic_digest.update(fixed)
            body.write(struct.pack("<ffffffBBB", x, y, z, 0.0, 0.0, 0.0, red, green, blue))
            for index, value in enumerate((x, y, z)):
                minimum[index] = min(minimum[index], value)
                maximum[index] = max(maximum[index], value)
            source_tracks += track_count
        if src.read(1):
            raise ValueError("unexpected trailing bytes in source points3D.bin")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode()
    with target_ply.open("wb") as ply, body_path.open("rb") as body:
        ply.write(header)
        shutil.copyfileobj(body, ply, 8 << 20)
    body_path.unlink()
    return {
        "sparse_points": count,
        "source_track_entries_removed": source_tracks,
        "xyz_min": minimum,
        "xyz_max": maximum,
        "sparse_point_semantic_sha256": semantic_digest.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source_sparse = args.source / "undistorted" / "sparse"
    source_images = args.source / "undistorted" / "images"
    output_sparse = args.output / "sparse" / "0"
    output_images = args.output / "images"
    required = [source_sparse / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    if not source_images.is_dir() or not all(path.is_file() for path in required):
        raise FileNotFoundError("source extraction is incomplete")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    output_sparse.mkdir(parents=True)
    output_images.symlink_to(source_images.resolve(), target_is_directory=True)
    shutil.copy2(source_sparse / "cameras.bin", output_sparse / "cameras.bin")
    image_result = strip_image_observations(
        source_sparse / "images.bin", output_sparse / "images.bin"
    )
    point_result = strip_point_tracks_and_write_ply(
        source_sparse / "points3D.bin",
        output_sparse / "points3D.bin",
        output_sparse / "points3D.ply",
    )
    output_image_count, output_image_semantics = image_semantic_digest(
        output_sparse / "images.bin"
    )
    output_point_count, output_point_semantics = point_semantic_digest(
        output_sparse / "points3D.bin"
    )
    if output_image_count != image_result["registered_images"]:
        raise ValueError("derived image record count changed")
    if output_image_semantics != image_result["registered_image_semantic_sha256"]:
        raise ValueError("derived camera pose/intrinsic/name semantics changed")
    if output_point_count != point_result["sparse_points"]:
        raise ValueError("derived sparse point count changed")
    if output_point_semantics != point_result["sparse_point_semantic_sha256"]:
        raise ValueError("derived sparse point id/XYZ/RGB/error semantics changed")

    image_files = sorted(path for path in source_images.rglob("*") if path.is_file())
    if len(image_files) != image_result["registered_images"]:
        raise ValueError(
            f"image count {len(image_files)} != registered count {image_result['registered_images']}"
        )
    payload = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "source": {
            "repository": "mattia-durso/Uni10K",
            "commit": SOURCE_COMMIT,
            "root": str(args.source.resolve()),
            "cameras_bin_sha256": sha256_file(source_sparse / "cameras.bin"),
            "images_bin_sha256": sha256_file(source_sparse / "images.bin"),
            "points3D_bin_sha256": sha256_file(source_sparse / "points3D.bin"),
        },
        "transformation": {
            "images": "byte-identical files exposed through a directory symlink",
            "cameras": "byte-identical cameras.bin",
            "registered_images": "identical headers/names with unused 2D observations removed",
            "sparse_points": "identical id/XYZ/RGB/error with unused tracks removed; binary PLY emitted from XYZ/RGB",
        },
        "counts": {**image_result, **point_result, "image_files": len(image_files)},
        "output": {
            "root": str(args.output.resolve()),
            "cameras_bin_sha256": sha256_file(output_sparse / "cameras.bin"),
            "images_bin_sha256": sha256_file(output_sparse / "images.bin"),
            "points3D_bin_sha256": sha256_file(output_sparse / "points3D.bin"),
            "points3D_ply_sha256": sha256_file(output_sparse / "points3D.ply"),
        },
        "validation": {
            "camera_binary_identical": sha256_file(source_sparse / "cameras.bin")
            == sha256_file(output_sparse / "cameras.bin"),
            "image_count_matches_registered": len(image_files)
            == image_result["registered_images"],
            "registered_image_semantics_identical": True,
            "sparse_point_semantics_identical": True,
            "status": "PASS",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
