#!/usr/bin/env python3
"""Normalize the frozen formal diagnostic cameras for the Run005 observer.

This is a schema adapter only: it preserves the exact pre-training diagnostic
set and derives official evaluation indices from the already-frozen LLFF split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from read_write_model import read_cameras_binary, read_images_binary


EXPECTED_FILENAMES = (
    "000060__aerial__test__0049.png",
    "000090__aerial__test__0072.png",
    "000100__aerial__test__0084.png",
    "000130__aerial__test__0129.png",
    "000150__aerial__test__0000.png",
    "000220__aerial__test__0098.png",
    "000240__aerial__test__0124.png",
    "000250__aerial__test__0147.png",
    "000320__street__test__0006.png",
    "000350__street__test__0009.png",
    "000370__street__test__0011.png",
    "000480__street__test__0022.png",
    "000500__street__test__0024.png",
    "000520__street__test__0026.png",
    "000580__street__test__0032.png",
    "000650__street__test__0039.png",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=REPO_ROOT / "datasets/alod/ALOD-REPRO-MC-MIXED-01",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/luis/tese/data/alod/ALOD-REPRO-MC-MIXED-01"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    diagnostic_path = args.metadata_root / "diagnostic_cameras.json"
    split_path = args.metadata_root / "split.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    frozen = diagnostic["cameras"]
    filenames = tuple(row["dataset_filename"] for row in frozen)
    if filenames != EXPECTED_FILENAMES:
        raise RuntimeError("formal diagnostic camera identities/order changed")

    evaluation = [row for row in split["records"] if row["split"] == "eval"]
    evaluation.sort(key=lambda row: row["dataset_filename"])
    eval_index = {
        row["dataset_filename"]: index for index, row in enumerate(evaluation)
    }
    images = read_images_binary(str(args.dataset_root / "sparse/0/images.bin"))
    cameras = read_cameras_binary(str(args.dataset_root / "sparse/0/cameras.bin"))

    normalized = []
    for row in frozen:
        registered = images[int(row["registered_image_id"])]
        camera = cameras[int(registered.camera_id)]
        if registered.name != row["dataset_filename"]:
            raise RuntimeError("registered image/name mismatch")
        if row["dataset_filename"] not in eval_index:
            raise RuntimeError("diagnostic camera is absent from official evaluation split")
        width = int(row["image"]["width"])
        height = int(row["image"]["height"])
        if width != int(camera.width) or height != int(camera.height):
            raise RuntimeError("manifest/COLMAP source resolution mismatch")
        normalized.append(
            {
                "registered_image_id": int(row["registered_image_id"]),
                "camera_id": int(registered.camera_id),
                "image_name": row["dataset_filename"],
                "stable_sorted_index": int(row["dataset_index"]),
                "dataset_index": int(row["dataset_index"]),
                "evaluation_sequence_index": int(eval_index[row["dataset_filename"]]),
                "camera_center": row["camera_centre"],
                "source_width": width,
                "source_height": height,
                "effective_width": round(width / 2),
                "effective_height": round(height / 2),
                "source_intrinsics": row["intrinsics"],
                "selection_categories": {
                    "modality": row["modality"],
                    "distance_stratum": row["distance_stratum"],
                    "region_class": row["region_class"],
                },
                "selection_rationale": diagnostic["selection_method"],
                "formal_dataset_camera": row,
            }
        )

    result = {
        "schema_version": "alod_formal_mixed_oracle_cameras.v1",
        "dataset_id": diagnostic["dataset_id"],
        "source_diagnostic_manifest": str(diagnostic_path.resolve()),
        "source_split_manifest": str(split_path.resolve()),
        "derivation": "exact frozen diagnostic rows plus official sorted LLFF-hold evaluation positions",
        "resolution_divisor": 2,
        "cameras": normalized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(result), encoding="utf-8")
    print(canonical_json({"output": str(args.output.resolve()), "camera_count": len(normalized)}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
