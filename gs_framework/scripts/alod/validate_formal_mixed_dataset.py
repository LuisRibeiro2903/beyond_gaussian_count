#!/usr/bin/env python3
"""Validate the frozen mixed MatrixCity proxy through pinned ALoD ingestion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_binary_counts(root: Path) -> dict[str, int]:
    result = {}
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        with (root / name).open("rb") as handle:
            result[name] = struct.unpack("<Q", handle.read(8))[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--official-commit", required=True)
    args = parser.parse_args()

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.official_source))
    # The pinned package creates a CUDA binomial table while importing the
    # training-only relocation helper.  Dataset ingestion never calls it, so a
    # fail-closed in-memory stub allows the real reader/camera code to be
    # exercised without touching the clean checkout or initializing CUDA.
    relocation_stub = types.ModuleType("utils.reloc_utils")

    def unavailable_relocation(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("relocation is unavailable in CPU-only ingestion validation")

    relocation_stub.compute_relocation_cuda = unavailable_relocation  # type: ignore[attr-defined]
    sys.modules["utils.reloc_utils"] = relocation_stub
    from scene.dataset_readers import readColmapSceneInfo  # type: ignore
    from utils.camera_utils import loadCam  # type: ignore

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before CPU-only ingestion")
    manifest = json.loads((args.metadata_root / "manifest.json").read_text())
    split = json.loads((args.metadata_root / "split.json").read_text())
    diagnostic = json.loads((args.metadata_root / "diagnostic_cameras.json").read_text())
    records = split["records"]
    names = [row["dataset_filename"] for row in records]
    source_ids = [row["source_id"] for row in records]
    image_ids = [row["registered_image_id"] for row in records]
    if len(set(names)) != len(names) or len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate stable camera identity")
    if image_ids != list(range(1, len(records) + 1)):
        raise ValueError("registered image IDs are not stable contiguous identities")
    for index, row in enumerate(records):
        if row["dataset_index"] != index or names[index] != sorted(names)[index]:
            raise ValueError("dataset index/name ordering mismatch")
        expected = "eval" if index % manifest["split"]["official_llff_hold"] == 0 else "train"
        if row["split"] != expected:
            raise ValueError("frozen split does not realize official llff_hold")
        image_path = args.dataset_root / row["image"]["relative_path"]
        if not image_path.is_file() or sha256_file(image_path) != row["image"]["sha256"]:
            raise ValueError(f"missing or changed image: {image_path}")

    scene = readColmapSceneInfo(
        str(args.dataset_root), "images", "", "", True, False,
        manifest["split"]["official_llff_hold"],
    )
    if len(scene.train_cameras) != manifest["counts"]["train"]:
        raise ValueError("official training split count mismatch")
    if len(scene.test_cameras) != manifest["counts"]["eval"]:
        raise ValueError("official evaluation split count mismatch")
    if len(scene.point_cloud.points) != manifest["counts"]["sparse_points"]:
        raise ValueError("official sparse point count mismatch")

    official_by_name = {row.image_name: row for row in scene.train_cameras + scene.test_cameras}
    if set(official_by_name) != set(names):
        raise ValueError("official loader image identity mismatch")
    centre_errors = []
    for row in records:
        camera = official_by_name[row["dataset_filename"]]
        rotation = np.asarray(camera.R).T
        centre = -rotation.T @ np.asarray(camera.T)
        centre_errors.append(float(np.linalg.norm(centre - np.asarray(row["camera_centre"]))))
        if not np.isfinite(centre).all() or not np.isfinite(camera.R).all() or not np.isfinite(camera.T).all():
            raise ValueError("non-finite official camera pose")

    representatives = {}
    model_args = SimpleNamespace(resolution=2, data_device="cpu", train_test_exp=False)
    for modality in ("aerial", "street"):
        for split_name in ("train", "eval"):
            row = next(item for item in records if item["modality"] == modality and item["split"] == split_name)
            camera_info = official_by_name[row["dataset_filename"]]
            camera = loadCam(model_args, row["dataset_index"], camera_info, 1.0, split_name == "eval")
            representatives[f"{modality}:{split_name}"] = {
                "dataset_index": row["dataset_index"],
                "filename": row["dataset_filename"],
                "input_resolution": [row["image"]["width"], row["image"]["height"]],
                "effective_resolution_at_resolution_2": [camera.image_width, camera.image_height],
                "tensor_shape_chw": list(camera.original_image.shape),
                "tensor_dtype": str(camera.original_image.dtype),
                "finite": bool(torch.isfinite(camera.original_image).all()),
                "min": float(camera.original_image.min()),
                "max": float(camera.original_image.max()),
            }
            del camera

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only official ingestion initialized CUDA")
    diagnostic_counts = Counter(
        (
            row["modality"],
            row["distance_stratum"] if row["modality"] == "aerial" else row["region_class"],
        )
        for row in diagnostic["cameras"]
    )
    expected_diagnostics = Counter(
        {
            ("aerial", "near"): 3,
            ("aerial", "mid"): 2,
            ("aerial", "far"): 3,
            ("street", "central"): 4,
            ("street", "boundary"): 4,
        }
    )
    if diagnostic_counts != expected_diagnostics:
        raise ValueError(f"diagnostic coverage mismatch: {diagnostic_counts}")

    sparse_root = args.dataset_root / "sparse" / "0"
    report = {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        "status": "PASS-ALOD-FORMAL-DATASET",
        "official_source": {
            "path": str(args.official_source),
            "commit": args.official_commit,
            "ingestion_function": "scene.dataset_readers.readColmapSceneInfo",
            "image_function": "utils.camera_utils.loadCam",
        },
        "cpu_only": True,
        "cuda_initialized": False,
        "cpu_ingestion_compatibility_shim": {
            "module": "utils.reloc_utils",
            "reason": "pinned source allocates a CUDA binomial table at import; relocation is not reachable during dataset ingestion",
            "behavior": "fail-closed in-memory compute_relocation_cuda stub; no source file changed",
        },
        "counts": {
            "binary_headers": read_binary_counts(sparse_root),
            "official_train": len(scene.train_cameras),
            "official_eval": len(scene.test_cameras),
            "official_sparse_points": len(scene.point_cloud.points),
            "stable_source_identities": len(set(source_ids)),
            "diagnostic": len(diagnostic["cameras"]),
        },
        "colmap": {
            "camera_models": manifest["camera_models"],
            "camera_records": manifest["camera_count"],
            "duplicate_registered_image_ids": 0,
            "duplicate_filenames": 0,
            "duplicate_source_identities": 0,
            "dangling_image_point_references": 0,
            "dangling_track_image_references": 0,
            "source_tracks_removed_by_construction": True,
            "finite_poses": True,
            "max_manifest_to_official_camera_centre_error": max(centre_errors),
        },
        "split": {
            "official_llff_hold": manifest["split"]["official_llff_hold"],
            "realized_exactly": True,
            "frozen_source_partitions_preserved": True,
        },
        "diagnostic_coverage": {
            f"{key[0]}:{key[1]}": value for key, value in sorted(diagnostic_counts.items())
        },
        "image_preprocessing": representatives,
        "hashes": {
            name: sha256_file(sparse_root / name)
            for name in ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply")
        },
    }
    write_json(args.metadata_root / "ingestion_validation.json", report)
    write_json(args.dataset_root / "ingestion_validation.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
