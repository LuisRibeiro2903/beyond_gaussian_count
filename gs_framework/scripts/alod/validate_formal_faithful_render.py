#!/usr/bin/env python3
"""Validate the formal no-cache pre-raster path against the Run009 oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_selection_oracle import load_manifest, sha256_file, validate_oracle_tree
from scene_runtime.alod.host_scene import AlodHostScene
from scene_runtime.alod.types import AlodHostResidencyConfig
from scripts.alod.validate_production_resident import (
    _config,
    _frame_camera,
    _json,
    _payload,
    _pre_raster_hashes,
    _settings,
)


PASS_STATUS = "PASS-ALOD-FORMAL-FAITHFUL-PRE-RASTER"


def validate(dhier: Path, oracle_manifest: Path, device: torch.device) -> dict:
    manifest = load_manifest(oracle_manifest)
    validate_oracle_tree(oracle_manifest)
    if sha256_file(dhier) != manifest["asset"]["sha256"]:
        raise RuntimeError("formal hierarchy hash does not match the Run009 oracle")
    runtime_config, _ = _config(manifest, oracle_manifest.resolve().parent)
    scene = AlodHostScene(
        dhier,
        device,
        runtime_config,
        AlodHostResidencyConfig(
            mode="host_nocache",
            validate_mappings=True,
            validate_properties=True,
        ),
    )
    root = oracle_manifest.resolve().parent
    rows = []
    for camera_index, entry in enumerate(manifest["cameras"]):
        camera_path = root / entry["paths"]["camera.json"]
        record = _json(camera_path)
        camera = _frame_camera(record, device)
        settings = _settings(record)
        active = scene.prepare_frame(camera, settings)
        selection = scene.last_selection
        assert selection is not None
        working = _json(camera_path.parent / "working-set.json")
        expected_raster = _payload(camera_path.parent, working["global_node_id_payload"])
        if not np.array_equal(selection.raster_global_node_ids, expected_raster):
            raise RuntimeError(f"raster-ID order differs for camera {camera_index}")
        expected_selected = expected_raster[expected_raster >= runtime_config.skybox_count]
        if not np.array_equal(
            selection.selected_hierarchy_global_node_ids, expected_selected
        ):
            raise RuntimeError(f"selected global IDs differ for camera {camera_index}")
        hashes = _pre_raster_hashes(active, camera, settings)
        expected_hashes = entry["pre_raster_tensor_hashes"]
        mismatches = {
            key: {"actual": value, "expected": expected_hashes[key]}
            for key, value in hashes.items()
            if value != expected_hashes[key]
        }
        if mismatches:
            raise RuntimeError(
                f"pre-raster properties differ for camera {camera_index}: {mismatches}"
            )
        rows.append(
            {
                "camera_index": camera_index,
                "camera_stable_id": entry["camera_stable_id"],
                "selected_hierarchy_rows": int(expected_selected.size),
                "raster_rows": int(expected_raster.size),
                "raster_id_order_exact": True,
                "pre_raster_properties_exact": True,
                "h2d_rows": int(scene.get_metrics().alod_h2d_rows),
            }
        )
    return {
        "status": PASS_STATUS,
        "asset": {"path": str(dhier.resolve()), "sha256": sha256_file(dhier)},
        "oracle_manifest": str(oracle_manifest.resolve()),
        "camera_count": len(rows),
        "checks": {
            "selected_global_ids_exact": True,
            "raster_global_id_order_exact": True,
            "pre_raster_properties_exact": True,
            "host_nocache_h2d_every_frame": all(
                row["h2d_rows"] == row["selected_hierarchy_rows"] for row in rows
            ),
        },
        "cameras": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dhier", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = validate(args.dhier.resolve(), args.oracle_manifest.resolve(), torch.device(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
