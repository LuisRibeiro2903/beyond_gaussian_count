#!/usr/bin/env python3
"""GPU integration check for ALoD load, HSPT rebuild, and serialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from topology_invariants import validate_topology


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_report(model: object) -> dict[str, object]:
    active_size = int(getattr(model, "size", len(model.nodes)))
    report = validate_topology(
        model.nodes,
        active_size=active_size,
        skybox_rows=model.skybox_points,
        root=model.skybox_points,
        max_mismatches=100,
    )
    if not report.valid:
        raise RuntimeError(json.dumps(report.as_dict(), sort_keys=True))
    return report.as_dict()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path)
    parser.add_argument("--serialized-skybox-rows", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--llff-hold", type=int, default=50)
    parser.add_argument("--sh-degree", type=int, default=1)
    parser.add_argument("--cap-max", type=int, default=1_100_000)
    parser.add_argument("--spt-root-volume", type=float, default=150.0)
    parser.add_argument("--target-pixels", type=float, default=2.0)
    parser.add_argument("--min-spt-size", type=int, default=256)
    args = parser.parse_args()

    source = args.source.resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))
    from gaussian_hierarchy._C import load_dynamic_hierarchy
    from scene.dataset_readers import readColmapSceneInfo
    from scene.gaussian_model import GaussianModel
    from utils.general_utils import safe_state

    safe_state(False)
    info = readColmapSceneInfo(
        str(args.dataset),
        str(args.dataset / "images"),
        "",
        "",
        True,
        False,
        args.llff_hold,
    )
    model = GaussianModel(args.sh_degree)
    scaffold = str(args.scaffold.resolve()) if args.scaffold is not None else ""
    model.create_from_hier(
        str(args.hierarchy),
        info.nerf_normalization["radius"],
        scaffold,
    )
    if args.serialized_skybox_rows:
        if args.scaffold is not None:
            raise ValueError("serialized skybox rows and external scaffold are mutually exclusive")
        model.skybox_points = args.serialized_skybox_rows
    loaded = checked_report(model)
    loaded_size = len(model.nodes)
    loaded_hash = tensor_sha256(model.nodes[:loaded_size])

    model.compact_gaussians(
        "cpu",
        args.cap_max,
        "classic",
        prune_unused_gaussians=False,
    )
    compacted = checked_report(model)
    compacted_hash = tensor_sha256(model.nodes[: model.size])
    if compacted_hash != loaded_hash:
        raise RuntimeError("compact_gaussians changed active topology")

    before_hspt = model.nodes[: model.size].clone()
    focal = info.train_cameras[0].focal_length
    started = time.perf_counter()
    model.build_hierarchical_SPT(
        args.spt_root_volume,
        args.target_pixels / focal,
        args.min_spt_size,
        use_bounding_spheres=False,
    )
    hspt_seconds = time.perf_counter() - started
    hspt = checked_report(model)
    hspt_unchanged = torch.equal(before_hspt, model.nodes[: model.size])
    if not hspt_unchanged:
        raise RuntimeError("HSPT rebuild changed hierarchy node rows")

    temporary = Path(tempfile.gettempdir()) / f"alod-topology-roundtrip-{os.getpid()}.dhier"
    try:
        model.save_hierarchy(str(temporary.parent), temporary.name)
        roundtrip_sha256 = file_sha256(temporary)
        roundtrip_bytes = temporary.stat().st_size
        _, _, _, _, _, roundtrip_nodes = load_dynamic_hierarchy(str(temporary))
        roundtrip_equal = torch.equal(
            roundtrip_nodes.cpu(), model.nodes[: model.size].cpu()
        )
        if not roundtrip_equal:
            raise RuntimeError("dynamic hierarchy writer/loader changed node rows")
    finally:
        if temporary.exists():
            temporary.unlink()

    output = {
        "gate": "PASS",
        "source": str(source),
        "dataset": str(args.dataset.resolve()),
        "hierarchy": str(args.hierarchy.resolve()),
        "scaffold": str(args.scaffold.resolve()) if args.scaffold is not None else None,
        "serialized_skybox_rows": args.serialized_skybox_rows,
        "cuda_initialized": torch.cuda.is_initialized(),
        "cuda_device": torch.cuda.get_device_name(0),
        "loaded_topology": loaded,
        "loaded_nodes_sha256": loaded_hash,
        "compacted_topology": compacted,
        "compacted_active_nodes_unchanged": compacted_hash == loaded_hash,
        "hspt_topology": hspt,
        "hspt_nodes_unchanged": hspt_unchanged,
        "hspt_rebuild_seconds": hspt_seconds,
        "roundtrip_nodes_equal": roundtrip_equal,
        "roundtrip_file_sha256": roundtrip_sha256,
        "roundtrip_file_bytes": roundtrip_bytes,
        "temporary_roundtrip_file_removed": not temporary.exists(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
