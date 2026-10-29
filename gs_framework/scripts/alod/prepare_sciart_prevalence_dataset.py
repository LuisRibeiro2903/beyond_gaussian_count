#!/usr/bin/env python3
"""Stage the public Sci-Art split without changing the pinned ALoD loader.

The train model is copied unchanged. The evaluation model interleaves each
public val camera with a train camera so llff_hold=2 selects all 21 val views.
Raw photographs are undistorted with the official Mega-NeRF metadata, matching
CityGaussian/tools/copy_images.py, before the official loader downsamples /4.
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
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement


ROOT = Path(__file__).resolve().parents[3]
DATASET_ID = "ALOD-PREV-US3D-SCIART-01"
SOURCE = ROOT / "data/alod_prevalence/URBANSCENE3D-SCIART/source"
MODEL = SOURCE / "colmap_results/sciart"
DATASET = ROOT / "data/alod_prevalence/URBANSCENE3D-SCIART/dataset"
OUTPUT = ROOT / "gs_framework/datasets/alod/ALOD-PREV-US3D-SCIART-01"
AUDIT = ROOT / "thesis_notes/data/alod_prevalence_candidate_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_records(path: Path, wanted: set[str] | None = None) -> dict[str, dict]:
    """Read COLMAP pose headers and selected full records without expanding tracks."""
    records = {}
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", handle.read(8))
        for _ in range(count):
            start = handle.tell()
            image_id = struct.unpack("<i", handle.read(4))[0]
            pose = handle.read(56)
            if len(pose) != 56 or not all(math.isfinite(value) for value in struct.unpack("<7d", pose)):
                raise RuntimeError(f"non-finite/truncated COLMAP pose in {path}")
            camera_id = struct.unpack("<i", handle.read(4))[0]
            name = bytearray()
            while (char := handle.read(1)) != b"\0":
                if not char:
                    raise EOFError(f"truncated image name in {path}")
                name.extend(char)
            name = name.decode("utf-8")
            (points,) = struct.unpack("<Q", handle.read(8))
            handle.seek(points * 24, os.SEEK_CUR)
            end = handle.tell()
            raw = None
            if wanted is not None and name in wanted:
                handle.seek(start)
                raw = handle.read(end - start)
                handle.seek(end)
            if name in records:
                raise RuntimeError(f"duplicate COLMAP name {name}")
            records[name] = {"image_id": image_id, "camera_id": camera_id,
                             "pose_sha256": hashlib.sha256(pose).hexdigest(), "raw": raw}
    return records


def points_ply(source: Path, target: Path, expected_count: int) -> None:
    """Make the official loader's PLY format without retaining COLMAP tracks."""
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    points = np.zeros(expected_count, dtype=dtype)
    with source.open("rb") as handle:
        (count,) = struct.unpack("<Q", handle.read(8))
        if count != expected_count:
            raise RuntimeError(f"COLMAP point count {count} != {expected_count}")
        for index in range(count):
            _, x, y, z, r, g, b, _ = struct.unpack("<QdddBBBd", handle.read(43))
            points[index] = (x, y, z, 0, 0, 0, r, g, b)
            (track_length,) = struct.unpack("<Q", handle.read(8))
            handle.seek(track_length * 8, os.SEEK_CUR)
        if handle.read(1):
            raise RuntimeError("COLMAP points file has trailing bytes")
    PlyData([PlyElement.describe(points, "vertex")]).write(target)


def stage_models() -> dict:
    audit = json.loads(AUDIT.read_text())["datasets"]["URBANSCENE3D-SCIART"]
    expected_train = audit["train"]["content_sha256"]
    expected_val = audit["evaluation"]["content_sha256"]
    for split, expected in (("train", expected_train), ("val", expected_val)):
        for name, digest in expected.items():
            path = MODEL / split / "sparse/0" / name
            if sha256(path) != digest:
                raise RuntimeError(f"candidate-audit COLMAP hash mismatch: {path}")
    train_path = MODEL / "train/sparse/0"
    val_path = MODEL / "val/sparse/0"
    if (train_path / "cameras.bin").read_bytes() != (val_path / "cameras.bin").read_bytes():
        raise RuntimeError("Sci-Art train/val calibration differs")
    calibration = (train_path / "cameras.bin").read_bytes()
    if len(calibration) != 64:
        raise RuntimeError("expected one Sci-Art PINHOLE camera")
    count, camera_id, model_id, width, height, fx, fy, cx, cy = struct.unpack("<QiiQQdddd", calibration)
    if (count, model_id, width, height) != (1, 1, 4864, 3648) or camera_id <= 0:
        raise RuntimeError("unsupported Sci-Art calibration")
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
        raise RuntimeError("invalid Sci-Art focal length/principal point")
    train = image_records(train_path / "images.bin")
    val = image_records(val_path / "images.bin")
    if len(train) != 2998 or len(val) != 21 or set(train) & set(val):
        raise RuntimeError("public Sci-Art train/val split changed")
    if any(row["camera_id"] != camera_id for row in list(train.values()) + list(val.values())):
        raise RuntimeError("Sci-Art image references an unsupported calibration")
    if len({row["image_id"] for row in train.values()}) != 2998:
        raise RuntimeError("duplicate train image ID")
    if len({row["image_id"] for row in val.values()}) != 21:
        raise RuntimeError("duplicate val image ID")
    sorted_names = sorted(set(train) | set(val))
    pairs = []
    used = set()
    for val_name in sorted(val):
        successor = next((name for name in sorted_names
                          if name > val_name and name in train and name not in used), None)
        if successor is None:
            raise RuntimeError(f"no train successor for {val_name}")
        used.add(successor)
        pairs.append((val_name, successor))
    selected_names = {name for pair in pairs for name in pair}
    selected_train = image_records(train_path / "images.bin", selected_names)
    selected_val = image_records(val_path / "images.bin", selected_names)
    adapter_ids = {name: 1_000_000 + selected_train[name]["image_id"]
                   for _, name in pairs}
    if set(adapter_ids.values()) & {row["image_id"] for row in val.values()}:
        raise RuntimeError("normalization-only ID remap collides with public val ID")
    ordered = sorted(selected_names)
    if ordered[::2] != [pair[0] for pair in pairs] or ordered[1::2] != [pair[1] for pair in pairs]:
        raise RuntimeError("42-camera llff_hold=2 does not select the public val split")
    for split in ("train", "oracle"):
        (DATASET / split / "sparse/0").mkdir(parents=True, exist_ok=True)
        (DATASET / split / "images").mkdir(parents=True, exist_ok=True)
    for name in expected_train:
        dest = DATASET / "train/sparse/0" / name
        if not dest.exists():
            shutil.copy2(train_path / name, dest)
        if sha256(dest) != expected_train[name]:
            raise RuntimeError(f"staged train COLMAP hash mismatch: {dest}")
    train_ply = DATASET / "train/sparse/0/points3D.ply"
    if not train_ply.exists():
        points_ply(train_path / "points3D.bin", train_ply, 2133151)
    oracle_sparse = DATASET / "oracle/sparse/0"
    for name in ("cameras.bin", "points3D.bin"):
        dest = oracle_sparse / name
        if not dest.exists():
            shutil.copy2(val_path / name, dest)
        if sha256(dest) != expected_val[name]:
            raise RuntimeError(f"staged val COLMAP hash mismatch: {dest}")
    oracle_ply = oracle_sparse / "points3D.ply"
    if not oracle_ply.exists():
        points_ply(val_path / "points3D.bin", oracle_ply, 331)
    output = oracle_sparse / "images.bin"
    if not output.exists():
        with output.open("wb") as handle:
            handle.write(struct.pack("<Q", len(ordered)))
            for name in ordered:
                record = selected_val.get(name) if name in val else selected_train.get(name)
                raw = record["raw"]
                if name in adapter_ids:
                    raw = struct.pack("<i", adapter_ids[name]) + raw[4:]
                handle.write(raw)
    oracle = image_records(output)
    if list(sorted(oracle))[::2] != sorted(val):
        raise RuntimeError("staged oracle split is incorrect")
    if len(oracle) != 42 or len({row["image_id"] for row in oracle.values()}) != 42:
        raise RuntimeError("staged oracle image IDs are not unique")
    for name in ordered:
        source = val[name] if name in val else train[name]
        expected_id = source["image_id"] if name in val else adapter_ids[name]
        if (oracle[name]["image_id"] != expected_id or
                oracle[name]["camera_id"] != source["camera_id"] or
                oracle[name]["pose_sha256"] != source["pose_sha256"]):
            raise RuntimeError(f"oracle pose changed: {name}")
    return {"train": train, "val": val, "pairs": pairs,
            "normalization_only_image_id_map": adapter_ids,
            "colmap_hashes": {"train": expected_train, "val": expected_val,
                              "oracle_images_bin": sha256(output)}}


def raw_tar_inventory(mappings: list[tuple[str, str]]) -> tuple[dict[str, tarfile.TarFile], dict[str, tuple[str, str]]]:
    tar_root = SOURCE / "fastmap_raw"
    archives = {part: tarfile.open(tar_root / f"{part}.tar", "r") for part in ("1", "2", "3", "4")}
    names_by_part = {part: set(archive.getnames()) for part, archive in archives.items()}
    index = {}
    for raw, _ in mappings:
        part = raw.split("/", 1)[0]
        if part not in archives:
            raise RuntimeError(f"unexpected Sci-Art raw-image folder: {raw}")
        relative = raw.split("/", 1)[1]
        candidates = (raw, relative, f"urbn_Sci-Art/{raw}")
        matches = [name for name in candidates if name in names_by_part[part]]
        if len(matches) != 1:
            raise RuntimeError(f"expected one tar member for {raw}, got {matches}")
        index[raw] = (part, matches[0])
    return archives, index


def stage_images(models: dict, selected_part: str | None = None) -> list[dict]:
    metadata_root = SOURCE / "sci-art-pixsfm"
    mappings = [tuple(line.strip().split(",")) for line in
                (metadata_root / "mappings.txt").read_text().splitlines() if line.strip()]
    if len(mappings) != 3019:
        raise RuntimeError("official mapping count changed")
    expected = set(models["train"]) | set(models["val"])
    if {Path(meta).stem + ".JPG" for _, meta in mappings} != expected:
        raise RuntimeError("Mega-NeRF image mapping does not match public COLMAP models")
    rows = []
    archives, member_by_raw = raw_tar_inventory(mappings)
    try:
        for index, (raw_name, metadata_name) in enumerate(mappings, 1):
            if selected_part is not None and raw_name.split("/", 1)[0] != selected_part:
                continue
            name = Path(metadata_name).stem + ".JPG"
            split = "train" if name in models["train"] else "val"
            metadata_path = metadata_root / split / "metadata" / metadata_name
            if not metadata_path.is_file():
                raise RuntimeError(f"missing official metadata {metadata_path}")
            target = DATASET / "train/images" / name if split == "train" else DATASET / "oracle/images" / name
            if not target.exists():
                metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
                intrinsics = metadata["intrinsics"]
                camera_matrix = np.array([[intrinsics[0], 0, intrinsics[2]],
                                          [0, intrinsics[1], intrinsics[3]], [0, 0, 1]], dtype=np.float64)
                part, member = member_by_raw[raw_name]
                source_handle = archives[part].extractfile(member)
                if source_handle is None:
                    raise RuntimeError(f"could not read tar member {member}")
                raw = np.frombuffer(source_handle.read(), dtype=np.uint8)
                distorted = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if distorted is None or distorted.shape[:2] != (3648, 4864):
                    raise RuntimeError(f"unexpected raw image: {raw_name}")
                undistorted = cv2.undistort(distorted, camera_matrix,
                                            np.asarray(metadata["distortion"]))
                if undistorted.shape[:2] != (int(metadata["H"]), int(metadata["W"])):
                    raise RuntimeError(f"undistortion shape mismatch: {name}")
                partial = target.with_name(target.stem + ".partial" + target.suffix)
                if not cv2.imwrite(str(partial), undistorted):
                    raise RuntimeError(f"image write failed: {target}")
                os.replace(partial, target)
            rows.append({"name": name, "split": split, "raw": raw_name,
                         "source_tar_member": member_by_raw[raw_name][1],
                         "metadata_sha256": sha256(metadata_path),
                         "bytes": target.stat().st_size, "sha256": sha256(target)})
            if len(rows) % 100 == 0:
                print(f"prepared {len(rows)} Sci-Art images from part {selected_part or 'all'}", flush=True)
    finally:
        for archive in archives.values():
            archive.close()
    if selected_part is not None:
        return rows
    for _, train_name in models["pairs"]:
        source_image = DATASET / "train/images" / train_name
        oracle_image = DATASET / "oracle/images" / train_name
        if not oracle_image.exists():
            os.link(source_image, oracle_image)
        if sha256(oracle_image) != sha256(source_image):
            raise RuntimeError(f"oracle normalization image differs: {train_name}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--image-part", choices=("1", "2", "3", "4"))
    args = parser.parse_args()
    models = stage_models()
    print("PASS: 2998 train / 21 public val cameras; oracle llff_hold=2 selects all 21", flush=True)
    if args.models_only:
        return 0
    image_rows = stage_images(models, args.image_part)
    if args.image_part is not None:
        print(f"PASS: prepared Sci-Art raw-image part {args.image_part}")
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "image-manifest.json").write_text(json.dumps(image_rows, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1, "dataset_id": DATASET_ID,
        "status": "PASS-FROZEN-DATASET-PREFLIGHT",
        "source": {"image_collection_page": "https://github.com/Linxius/UrbanScene3D#urbanscene3d-v1",
                   "image_mirror": "https://huggingface.co/datasets/whc/fastmap_sfm/tree/main/images/urbn_Sci-Art",
                   "image_tar_sha256": {f"{part}.tar": sha256(SOURCE / "fastmap_raw" / f"{part}.tar")
                                        for part in ("1", "2", "3", "4")},
                   "metadata_archive_url": "https://storage.cmusatyalab.org/mega-nerf-data/sci-art-pixsfm.tgz",
                   "metadata_archive_sha256": sha256(SOURCE / "sci-art-pixsfm.tgz"),
                   "colmap_archive_sha256": "0571c6e575a3265ab025cce18c0f0f7ea23f264e9a8f22d22f6f5e41f2375fba",
                   "license": "UrbanScene3D non-commercial research use; preserve attribution, do not redistribute imagery"},
        "counts": {"train_images": 2998, "evaluation_images": 21,
                   "registered_images": 3019, "image_files": len(image_rows),
                   "sparse_points": 2133151},
        "split": {"source": "public COLMAP train/val models",
                  "train_loader_llff_hold": 0, "oracle_loader_llff_hold": 2,
                  "oracle_pairs": models["pairs"],
                  "normalization_only_image_id_map": models["normalization_only_image_id_map"],
                  "public_val_names": sorted(models["val"])},
        "paths": {"train_dataset": str(DATASET / "train"),
                  "oracle_dataset": str(DATASET / "oracle")},
        "hashes": models["colmap_hashes"],
        "resolution_policy": {"native": [4864, 3648], "loader_divisor": 4,
                              "study": [1216, 912]},
        "image_manifest_sha256": sha256(OUTPUT / "image-manifest.json"),
        "validation": {"all_registered_images_resolve": True,
                       "public_split_matches_loader": True,
                       "all_camera_poses_unchanged": True,
                       "official_undistortion_applied": True},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"PASS: sealed Sci-Art dataset {OUTPUT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
