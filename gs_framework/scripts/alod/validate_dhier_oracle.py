#!/usr/bin/env python3
"""Compare the independent parser with a compact official-loader oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_dhier import load_dhier  # noqa: E402
from alod_dhier.report import sha256_file  # noqa: E402


SCHEMA_VERSION = "alod_dhier_oracle_validation.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def array_hash(array: Any) -> str:
    if hasattr(array, "detach"):
        array = array.detach().cpu().contiguous().numpy()
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def descriptor(array: Any, sample_ids: list[int]) -> dict[str, Any]:
    if hasattr(array, "detach"):
        array = array.detach().cpu().contiguous().numpy()
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": [int(value) for value in contiguous.shape],
        "dtype": str(contiguous.dtype),
        "raw_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        "samples": {
            str(index): np.ascontiguousarray(contiguous[index]).tobytes().hex()
            for index in sample_ids
        },
    }


def compare_descriptor(
    name: str, actual: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "shape": actual["shape"] == expected["shape"],
        "dtype": actual["dtype"] == expected["dtype"],
        "raw_sha256": actual["raw_sha256"] == expected["raw_sha256"],
        "sample_raw_hex": all(
            actual["samples"][key] == value["raw_hex"]
            for key, value in expected["samples"].items()
        ),
    }
    return {"name": name, "valid": all(checks.values()), "checks": checks, "actual": actual}


def main() -> int:
    args = parse_args()
    asset_path = args.asset.resolve()
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    if oracle["schema_version"] != "alod_dhier_official_loader_oracle.v1":
        raise ValueError("unsupported oracle schema")
    asset = load_dhier(asset_path)
    if sha256_file(asset_path) != oracle["asset_sha256"]:
        raise ValueError("asset SHA-256 does not match oracle")
    sample_ids = [int(value) for value in oracle["sample_global_node_ids"]]

    raw_arrays = {
        "positions": asset.positions,
        "rotations": asset.rotations,
        "log_scales": asset.log_scales,
        "activated_opacities": asset.activated_opacities,
        "sh_coefficients": asset.sh_coefficients,
        "nodes": asset.nodes,
    }
    expected_a = oracle["layer_a_raw_native_loader"]["arrays"]
    layer_a = [
        compare_descriptor(name, descriptor(array, sample_ids), expected_a[name])
        for name, array in raw_arrays.items()
    ]

    # The format parser remains NumPy-only.  This separate parity checker uses
    # CPU Torch solely to independently reproduce the pinned Python model's
    # documented float32 operations.
    import torch
    import torch.nn.functional as functional

    sh = torch.from_numpy(np.array(asset.sh_coefficients, copy=True))
    opacity = torch.from_numpy(np.array(asset.activated_opacities, copy=True))
    opacity.clamp_(0, 0.99999)
    opacity_logits = torch.log(opacity / (1 - opacity))
    scales = torch.from_numpy(np.array(asset.log_scales, copy=True))
    rotations = torch.from_numpy(np.array(asset.rotations, copy=True))
    nodes = torch.from_numpy(np.array(asset.nodes, copy=True))
    nodes[:, -1] = 0
    model_arrays = {
        "stored_positions": torch.from_numpy(np.array(asset.positions, copy=True)),
        "stored_features_dc": sh[:, :1, :],
        "stored_features_rest": sh[:, 1:, :],
        "stored_opacity_logits": opacity_logits,
        "stored_log_scales": scales,
        "stored_rotations": rotations,
        "stored_nodes": nodes,
        "render_scales_exp": torch.exp(scales),
        "render_rotations_normalized": functional.normalize(rotations),
        "render_opacities_sigmoid": torch.sigmoid(opacity_logits),
    }
    expected_b = oracle["layer_b_GaussianModel_create_from_hier"]["arrays"]
    layer_b = [
        compare_descriptor(name, descriptor(array, sample_ids), expected_b[name])
        for name, array in model_arrays.items()
    ]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "asset": str(asset_path),
        "asset_sha256": oracle["asset_sha256"],
        "official_source_commit": oracle["official_source_commit"],
        "layer_a_raw_format_parity": {
            "valid": all(record["valid"] for record in layer_a),
            "arrays": layer_a,
        },
        "layer_b_model_load_parity": {
            "valid": all(record["valid"] for record in layer_b),
            "arrays": layer_b,
            "comparison": "exact float32 hashes and exact sampled row bytes",
        },
    }
    payload["overall_valid"] = bool(
        payload["layer_a_raw_format_parity"]["valid"]
        and payload["layer_b_model_load_parity"]["valid"]
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "gate": "PASS" if payload["overall_valid"] else "FAIL",
        "asset_sha256": payload["asset_sha256"],
        "layer_a": payload["layer_a_raw_format_parity"]["valid"],
        "layer_b": payload["layer_b_model_load_parity"]["valid"],
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
