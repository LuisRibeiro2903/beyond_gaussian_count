#!/usr/bin/env python3
"""Export compact Layer-A/Layer-B parity evidence from the pinned official loader.

Run this script only in the validated ``alod_official`` environment.  It reads
the asset and writes hashes/samples; it never rewrites a hierarchy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


# The pinned checkout tracks CPython bytecode.  Prevent a read-only oracle run
# from refreshing those protected files when official modules are imported.
sys.dont_write_bytecode = True


SCHEMA_VERSION = "alod_dhier_official_loader_oracle.v1"
PINNED_COMMIT = "73f547341547ae021ff036ad74fdf73a1240b450"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--skybox-count", type=int, default=0)
    parser.add_argument("--sample-global-id", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_descriptor(tensor: torch.Tensor, sample_ids: list[int]) -> dict[str, Any]:
    array = tensor.detach().cpu().contiguous().numpy()
    numeric = np.issubdtype(array.dtype, np.floating)
    finite_count = int(np.count_nonzero(np.isfinite(array))) if numeric else None
    samples = {}
    for global_id in sample_ids:
        row = np.ascontiguousarray(array[global_id])
        samples[str(global_id)] = {
            "value": row.tolist(),
            "raw_hex": row.tobytes().hex(),
        }
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "raw_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "finite_count": finite_count,
        "element_count": int(array.size),
        "min": float(np.min(array)) if array.size else None,
        "max": float(np.max(array)) if array.size else None,
        "samples": samples,
    }


def main() -> int:
    args = parse_args()
    asset = args.asset.resolve()
    source = args.official_source.resolve()
    output = args.output.resolve()
    if output == asset:
        raise ValueError("oracle output must not overwrite the hierarchy")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=source, check=True, capture_output=True, text=True
    ).stdout
    if commit != PINNED_COMMIT or status:
        raise RuntimeError(f"official source is not clean at pinned commit: {commit!r}, dirty={bool(status)}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    with asset.open("rb") as handle:
        gaussian_count, degree = struct.unpack("<ii", handle.read(8))
    if degree not in (0, 1, 2, 3):
        raise ValueError(f"unsupported degree {degree}")
    sample_ids = args.sample_global_id or [0, gaussian_count // 2, gaussian_count - 1]
    if any(index < 0 or index >= gaussian_count for index in sample_ids):
        raise ValueError("sample global ID outside asset")
    sample_ids = list(dict.fromkeys(sample_ids))

    cuda_before = torch.cuda.is_initialized()
    from gaussian_hierarchy._C import load_dynamic_hierarchy

    positions, sh, opacities, scales, rotations, nodes = load_dynamic_hierarchy(str(asset))
    raw_tensors = {
        "positions": positions,
        "rotations": rotations,
        "log_scales": scales,
        "activated_opacities": opacities,
        "sh_coefficients": sh,
        "nodes": nodes,
    }
    layer_a = {name: tensor_descriptor(tensor, sample_ids) for name, tensor in raw_tensors.items()}

    # Layer B intentionally uses scaffold_file="" so the oracle isolates the
    # transformations performed on the serialized rows.  Skybox prepending and
    # link offsetting are separate external-input behavior, not format decoding.
    from scene.gaussian_model import GaussianModel

    model = GaussianModel(degree)
    model.create_from_hier(str(asset), 1.0, "", device="cpu")
    model_tensors = {
        "stored_positions": model._xyz,
        "stored_features_dc": model._features_dc,
        "stored_features_rest": model._features_rest,
        "stored_opacity_logits": model._opacity,
        "stored_log_scales": model._scaling,
        "stored_rotations": model._rotation,
        "stored_nodes": model.nodes,
        "render_scales_exp": model.get_scaling,
        "render_rotations_normalized": model.get_rotation,
        "render_opacities_sigmoid": model.get_opacity,
    }
    layer_b = {name: tensor_descriptor(tensor, sample_ids) for name, tensor in model_tensors.items()}

    payload = {
        "schema_version": SCHEMA_VERSION,
        "label": args.label,
        "official_source_commit": commit,
        "official_source_clean": True,
        "asset_path_recorded_for_provenance": str(asset),
        "asset_sha256": file_sha256(asset),
        "asset_bytes": asset.stat().st_size,
        "gaussian_count": gaussian_count,
        "node_count": int(nodes.shape[0]),
        "sh_degree": degree,
        "skybox_count": args.skybox_count,
        "sample_global_node_ids": sample_ids,
        "global_node_id_contract": "serialized row/index",
        "layer_a_raw_native_loader": {
            "comparison": "exact raw bytes",
            "transformations": [],
            "arrays": layer_a,
        },
        "layer_b_GaussianModel_create_from_hier": {
            "comparison": "independently reproduced transformations; exact hashes preferred",
            "scaffold_file": "",
            "transformations": [
                "split SH into degree-dependent DC/rest stored tensors",
                "clamp serialized opacity to [0, 0.99999]",
                "apply inverse_sigmoid(x) = log(x / (1 - x)) to stored opacity",
                "set hierarchy-node column 5 to zero",
                "render scale view applies exp",
                "render rotation view applies torch.nn.functional.normalize",
                "render opacity view applies sigmoid",
            ],
            "explicitly_not_exercised": [
                "scaffold/skybox attribute prepend",
                "hierarchy link offset by external skybox count",
            ],
            "arrays": layer_b,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_initialized_before": cuda_before,
            "cuda_initialized_after": torch.cuda.is_initialized(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "gate": "PASS",
        "output": str(output),
        "asset_sha256": payload["asset_sha256"],
        "degree": degree,
        "rows": gaussian_count,
        "cuda_initialized": torch.cuda.is_initialized(),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
