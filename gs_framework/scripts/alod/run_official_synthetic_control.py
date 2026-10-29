#!/usr/bin/env python3
"""Run the pinned released ALoD loader, selector, and renderer on the synthetic control.

Run this script with the ``alod_official`` Python environment and the pinned
official checkout first on ``PYTHONPATH``.  It intentionally imports the
released modules directly and does not import the project selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from gaussian_hierarchy._C import get_spt_cut_cuda, load_dynamic_hierarchy
from gaussian_renderer import render_gsplat
from scene import GaussianModel
from utils.graphics_utils import getProjectionMatrix


EXPERIMENT_ID = "ALOD-SYNTH-STRONG-LOD-CONTROL-001"
SKYBOX_ROWS = 100_000
SPT_ROOT_VOLUME = 1.5
TARGET_GRANULARITY = 0.00390625
MIN_SPT_SIZE = 100


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def raw_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def camera_tensors(record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c2w = torch.tensor(record["c2w"], dtype=torch.float32, device="cuda")
    world_view = torch.linalg.inv(c2w).T.contiguous()
    width = int(record["width"])
    height = int(record["height"])
    intrinsics = np.asarray(record["K"], dtype=np.float32)
    fov_x = 2.0 * math.atan(width / (2.0 * float(intrinsics[0, 0])))
    fov_y = 2.0 * math.atan(height / (2.0 * float(intrinsics[1, 1])))
    projection = getProjectionMatrix(
        znear=0.01,
        zfar=100.0,
        fovX=fov_x,
        fovY=fov_y,
        primx=float(intrinsics[0, 2]) / width,
        primy=float(intrinsics[1, 2]) / height,
    ).T.cuda()
    full = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    center = c2w[:3, 3].contiguous()
    return world_view, full, center


def prepare_model(asset: Path) -> GaussianModel:
    model = GaussianModel(None)
    model.create_from_hier(str(asset), 1.0, "", device="cpu")
    model.skybox_points = SKYBOX_ROWS
    model.compact_gaussians(
        "cpu", None, False, training=False, prune_unused_gaussians=False
    )
    model.build_hierarchical_SPT(
        SPT_ROOT_VOLUME,
        TARGET_GRANULARITY,
        min_SPT_Size=MIN_SPT_SIZE,
        use_bounding_spheres=False,
    )
    return model


@torch.no_grad()
def select(model: GaussianModel, center: torch.Tensor, full: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
    bounds = torch.exp(torch.max(model.upper_tree_scaling, dim=-1)[0]) * 3.0
    planes = model.extract_frustum_planes(full)
    frustum = lambda indices: model.frustum_cull_spheres(
        model.upper_tree_xyz[indices], bounds[indices], planes
    )
    detail = lambda indices: model.min_distance_squared[indices] > (
        center - model.upper_tree_xyz[indices]
    ).square().sum(dim=-1)
    coarse = model.cut_hierarchy_on_condition(
        model.upper_tree_nodes,
        detail,
        return_upper_tree=False,
        root_node=0,
        leave_out_of_cut_condition=frustum,
    )
    leaves = coarse[model.upper_tree_nodes[coarse, 2] == 0]
    candidate_mask = model.upper_tree_nodes[leaves, 3] >= 0
    spt_indices = model.upper_tree_nodes[leaves[candidate_mask], 3]
    spt_upper = leaves[candidate_mask]
    upper_render = model.upper_tree_nodes[coarse][
        model.upper_tree_nodes[coarse, 3] <= 0, 5
    ]
    distances = (
        (model.upper_tree_xyz[spt_upper] - center).pow(2).sum(1).sqrt()
        if len(spt_upper)
        else torch.empty(0, dtype=torch.float32, device="cuda")
    )
    if len(spt_indices):
        spt_rows, starts = get_spt_cut_cuda(
            len(spt_indices),
            model.SPT_gaussian_indices,
            model.SPT_starts,
            model.SPT_max,
            model.SPT_min,
            spt_indices,
            distances,
        )
    else:
        spt_rows = torch.empty(0, dtype=torch.int32, device="cuda")
        starts = torch.empty(0, dtype=torch.int32, device="cuda")
    hierarchy = torch.cat((upper_render, spt_rows)).to(torch.int32)
    raster = torch.cat(
        (
            torch.arange(SKYBOX_ROWS, dtype=torch.int32, device="cuda"),
            hierarchy,
        )
    )
    return raster, {
        "coarse_cut_rows": int(coarse.numel()),
        "upper_render_rows": int(upper_render.numel()),
        "candidate_spts": int(spt_indices.numel()),
        "spt_rows": int(spt_rows.numel()),
        "spt_start_entries": int(starts.numel()),
    }


@torch.no_grad()
def render(model: GaussianModel, raster_ids: torch.Tensor, camera: Any, output: Path) -> dict[str, Any]:
    rows = raster_ids.to(device="cpu", dtype=torch.long)
    properties = model.properties.index_select(0, rows).cuda()
    means = properties[:, 0:3].contiguous()
    scales = torch.exp(properties[:, 3:6]).contiguous()
    rotations = F.normalize(properties[:, 6:10], dim=-1).contiguous()
    dc = properties[:, 10:13].unsqueeze(1).contiguous()
    opacity = torch.sigmoid(properties[:, 13:14]).contiguous()
    rest = properties[:, 14:23].reshape(len(properties), 3, 3).contiguous()
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    result = render_gsplat(
        camera,
        means,
        opacity,
        scales,
        rotations,
        dc,
        rest,
        SimpleNamespace(),
        background,
        sh_degree=1,
    )
    torch.cuda.synchronize()
    image = result["render"].clamp(0.0, 1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(image, output)
    packed = result["packed_indices"].detach().cpu().to(torch.int64)
    return {
        "image_path": str(output.resolve()),
        "image_sha256": sha256_file(output),
        "packed_projection_rows": int(packed.numel()),
        "packed_indices_sha256": hashlib.sha256(
            packed.contiguous().numpy().tobytes(order="C")
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--render-index", type=int, action="append", default=[])
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    loaded = load_dynamic_hierarchy(str(args.asset))
    loader = {
        "shapes": [[int(v) for v in tensor.shape] for tensor in loaded],
        "dtypes": [str(tensor.dtype) for tensor in loaded],
        "raw_hashes": [raw_hash(tensor) for tensor in loaded],
    }
    model = prepare_model(args.asset)
    camera_payload = json.loads(args.cameras.read_text(encoding="utf-8"))
    records = []
    ids_dir = args.output_root / "selected_ids"
    ids_dir.mkdir(parents=True, exist_ok=True)
    render_set = set(args.render_index)
    for index, record in enumerate(camera_payload["cameras"]):
        world_view, full, center = camera_tensors(record)
        raster_ids, counts = select(model, center, full)
        hierarchy_ids = raster_ids[SKYBOX_ROWS:].detach().cpu().numpy().astype(np.int32)
        ids_path = ids_dir / f"camera_{index:02d}.npy"
        np.save(ids_path, hierarchy_ids, allow_pickle=False)
        row: dict[str, Any] = {
            "camera_index": index,
            "camera_stable_id": record["name"],
            "distance": float(record["distance"]),
            "selected_hierarchy_rows": int(hierarchy_ids.size),
            "selected_raster_rows": int(raster_ids.numel()),
            "selected_ids_path": str(ids_path.resolve()),
            "selected_ids_sha256": sha256_file(ids_path),
            **counts,
        }
        if index in render_set:
            camera = SimpleNamespace(
                image_width=int(record["width"]),
                image_height=int(record["height"]),
                FoVx=2.0 * math.atan(int(record["width"]) / (2.0 * float(record["K"][0][0]))),
                FoVy=2.0 * math.atan(int(record["height"]) / (2.0 * float(record["K"][1][1]))),
                world_view_transform=world_view,
                full_proj_transform=full,
                camera_center=center,
            )
            row["render"] = render(
                model,
                raster_ids,
                camera,
                args.output_root / "renders" / f"camera_{index:02d}.png",
            )
        records.append(row)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "implementation": "pinned official released loader/HSPT/selector/gsplat renderer",
        "asset": str(args.asset.resolve()),
        "asset_sha256": sha256_file(args.asset),
        "camera_path": str(args.cameras.resolve()),
        "camera_path_sha256": sha256_file(args.cameras),
        "configuration": {
            "skybox_rows": SKYBOX_ROWS,
            "root_global_node_id": SKYBOX_ROWS,
            "spt_root_volume": SPT_ROOT_VOLUME,
            "min_spt_size": MIN_SPT_SIZE,
            "target_granularity": TARGET_GRANULARITY,
            "distance_multiplier": 1.0,
            "frustum": "released four-plane scale-derived bounds",
        },
        "loader": loader,
        "hspt": {
            "upper_rows": int(model.upper_tree_nodes.shape[0]),
            "spt_count": int(model.SPT_starts.numel() - 1),
            "flattened_spt_rows": int(model.SPT_gaussian_indices.numel()),
        },
        "cameras": records,
    }
    (args.output_root / "official_summary.json").write_text(
        canonical_json(result), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
