#!/usr/bin/env python
"""Prepare immutable same-input cases for the CityGaussian backend microbenchmark.

This is deliberately outside every timed backend region.  The representative
case executes the pinned released visibility and Nyquist LoD selection once,
then serializes the already assembled activated Gaussian tensors for reuse by
both rasterizers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


OFFICIAL_ROOT = Path("/home/luis/tese/CityGaussian_official_parity")
OFFICIAL_EXTENSIONS = OFFICIAL_ROOT / ".official_extensions"
OFFICIAL_COMMIT = "e16c526f78ab01d377b13b3e6862dcc93ddbe953"
GRAPHDECO_COMMIT = "59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d"
AABB = [-3.5, -4.0, -10.0, 4.5, 2.0, 10.0]
BLOCK_DIM = [6, 6, 1]
LEVEL_LABELS = ["light_50_vq", "light_66_vq", "light_75_vq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("finest_flat", "representative_lod"), required=True)
    parser.add_argument("--camera-index", type=int, default=392)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--finest", type=Path, required=True)
    parser.add_argument("--middle", type=Path)
    parser.add_argument("--coarsest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser.parse_args()


def install_official_import_paths() -> None:
    sys.path.insert(0, str(OFFICIAL_EXTENSIONS))
    sys.path.insert(0, str(OFFICIAL_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor, chunk_rows: int = 65536) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    if value.ndim == 0:
        digest.update(value.numpy().tobytes(order="C"))
    else:
        for start in range(0, value.shape[0], chunk_rows):
            digest.update(value[start : start + chunk_rows].numpy().tobytes(order="C"))
    return digest.hexdigest()


def tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu()
    flat = value.to(dtype=torch.float64).reshape(-1)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": tensor_sha256(value),
        "sum_float64": float(flat.sum().item()),
        "mean_float64": float(flat.mean().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "all_finite": bool(torch.isfinite(value).all().item()),
    }


def load_camera(path: Path, index: int) -> dict[str, torch.Tensor | float | int | str]:
    from utils.graphics_utils import getProjectionMatrix

    payload = json.loads(path.read_text())
    records = payload["camera_path"]
    if index < 0 or index >= len(records):
        raise IndexError(f"Camera index {index} outside trajectory length {len(records)}")
    record = records[index]
    width = int(record.get("width", payload.get("render_width", 1920)))
    height = int(record.get("height", payload.get("render_height", 1080)))
    fov_y = math.radians(float(record.get("fov", payload.get("default_fov", 50.0))))
    focal = 0.5 * height / math.tan(fov_y * 0.5)
    fov_x = 2.0 * math.atan(width / (2.0 * focal))
    scale_ratio = float(payload.get("scale_ratio", 10.0))
    source = np.asarray(record["camera_to_world"], dtype=np.float32).reshape(4, 4)
    converted = source.copy()
    converted[:3, 1:3] *= -1.0
    converted[:3, 3] *= scale_ratio
    c2w = torch.from_numpy(converted).to(dtype=torch.float32)
    w2c = torch.linalg.inv(c2w).contiguous()
    world_view = w2c.T.contiguous()
    projection = getProjectionMatrix(0.01, 100.0, fov_x, fov_y).T.contiguous()
    full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    K = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_center = c2w[:3, 3].contiguous()
    return {
        "camera_id": f"viser_{index:06d}",
        "camera_index": index,
        "width": width,
        "height": height,
        "fov_x": fov_x,
        "fov_y": fov_y,
        "near": 0.01,
        "graphdeco_projection_far": 100.0,
        "gsplat_far": 1.0e10,
        "source_c2w": torch.from_numpy(source.copy()),
        "converted_c2w": c2w,
        "w2c": w2c,
        "K": K,
        "world_view_transform": world_view,
        "projection_transform": projection,
        "full_projection_transform": full_projection,
        "camera_center": camera_center,
    }


def load_model(path: Path):
    from scene.gaussian_model import GaussianModelLOD

    model = GaussianModelLOD(sh_degree=2, device="cuda")
    model.load_ply(str(path))
    model.active_sh_degree = model.max_sh_degree
    return model


def canonical_from_model(model) -> dict[str, torch.Tensor]:
    return {
        "means": model.get_xyz.detach().contiguous(),
        "quats": F.normalize(model.get_rotation.detach(), p=2, dim=-1).contiguous(),
        "scales": model.get_scaling.detach().contiguous(),
        "opacities": model.get_opacity.detach().reshape(-1).contiguous(),
        "sh_coefficients": model.get_features.detach().contiguous(),
    }


def load_blocked_levels(paths: list[Path]):
    from scene.gaussian_model import BlockedGaussian

    params = SimpleNamespace(sh_degree=2, block_dim=BLOCK_DIM, aabb=AABB)
    levels = []
    resident_counts = []
    for path in paths:
        model = load_model(path)
        resident_counts.append(int(model.get_xyz.shape[0]))
        levels.append(BlockedGaussian(model, params, compute_cov3D_python=False))
        del model
        torch.cuda.empty_cache()
    return levels, resident_counts


@torch.no_grad()
def assemble_released_lod(levels, camera_payload: dict[str, Any]):
    from scene.cameras import MiniCam
    from utils.large_utils import in_frustum

    camera = MiniCam(
        int(camera_payload["width"]),
        int(camera_payload["height"]),
        float(camera_payload["fov_y"]),
        float(camera_payload["fov_x"]),
        0.01,
        100.0,
        camera_payload["world_view_transform"].cuda(),
        camera_payload["full_projection_transform"].cuda(),
    )
    visible_mask, distances = in_frustum(
        camera,
        levels[-1].cell_corners,
        levels[-1].aabb,
        levels[-1].block_dim,
    )
    visible_ids = visible_mask.nonzero().squeeze(0)
    focal_length = 0.5 * camera.image_width / math.tan(camera.FoVx * 0.5)
    nyquist_scalings = 2.0 * distances / focal_length
    avg_scalings = torch.stack([level.avg_scalings for level in levels], dim=0)[:, visible_mask]
    values, lod_indices = torch.max(
        (avg_scalings > nyquist_scalings.unsqueeze(0)).to(torch.uint8), dim=0
    )
    lod_indices[values == 0] = len(levels) - 1
    chunks = []
    count_by_level = [0 for _ in levels]
    cells_by_level = [0 for _ in levels]
    count_by_cell: dict[str, int] = {}
    lod_by_cell: dict[str, int] = {}
    for level_index, level in enumerate(levels):
        level_cells = visible_ids[lod_indices == level_index]
        level_chunks = level.get_feats(level_cells)
        chunks.extend(level_chunks)
        cells_by_level[level_index] = int(level_cells.numel())
        for cell_id_tensor, chunk in zip(level_cells, level_chunks):
            cell_id = int(cell_id_tensor.item())
            count = int(chunk.shape[0])
            count_by_level[level_index] += count
            count_by_cell[str(cell_id)] = count
            lod_by_cell[str(cell_id)] = level_index
    feats = torch.cat(chunks, dim=0).contiguous()
    feature_end = 3 * (2 + 1) ** 2 + 4
    canonical = {
        "means": feats[:, :3].float().contiguous(),
        "quats": F.normalize(feats[:, feature_end + 3 :].float(), p=2, dim=-1).contiguous(),
        "scales": feats[:, feature_end : feature_end + 3].float().contiguous(),
        "opacities": feats[:, 3].float().contiguous(),
        "sh_coefficients": feats[:, 4:feature_end].reshape(-1, 9, 3).float().contiguous(),
    }
    selection = {
        "visible_cell_ids": [int(v) for v in visible_ids.detach().cpu().flatten().tolist()],
        "selected_lod_by_cell": lod_by_cell,
        "selected_level_label_by_cell": {
            cell_id: LEVEL_LABELS[level_index] for cell_id, level_index in lod_by_cell.items()
        },
        "selected_gaussian_count_by_cell": count_by_cell,
        "selected_cell_count_by_level": cells_by_level,
        "selected_gaussian_count_by_level": count_by_level,
        "total_selected_gaussian_count": int(feats.shape[0]),
        "focal_length_pixels": float(focal_length),
    }
    return canonical, selection


def cpu_payload(value: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().cpu().contiguous() for key, tensor in value.items()}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to execute the pinned official loaders and selector")
    if args.case == "representative_lod" and (args.middle is None or args.coarsest is None):
        raise ValueError("representative_lod requires --middle and --coarsest")
    install_official_import_paths()
    camera = load_camera(args.trajectory, args.camera_index)

    if args.case == "finest_flat":
        model = load_model(args.finest)
        canonical_gpu = canonical_from_model(model)
        resident_counts = [int(model.get_xyz.shape[0])]
        selection = None
        asset_paths = [args.finest]
    else:
        asset_paths = [args.finest, args.middle, args.coarsest]
        levels, resident_counts = load_blocked_levels([Path(path) for path in asset_paths])
        canonical_gpu, selection = assemble_released_lod(levels, camera)

    canonical = cpu_payload(canonical_gpu)
    count = int(canonical["means"].shape[0])
    if any(int(tensor.shape[0]) != count for tensor in canonical.values()):
        raise RuntimeError("Canonical tensor row counts differ")
    camera_tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in camera.items()
        if torch.is_tensor(value)
    }
    camera_scalars = {key: value for key, value in camera.items() if not torch.is_tensor(value)}
    input_payload = {
        "format": "CITYGS-BACKEND-MICROBENCH-CANONICAL-INPUT-v1",
        "case": args.case,
        "sh_degree": 2,
        "gaussians": canonical,
        "camera_tensors": camera_tensors,
        "camera_scalars": camera_scalars,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(input_payload, args.output)
    tensor_records = {key: tensor_record(value) for key, value in canonical.items()}
    camera_records = {key: tensor_record(value) for key, value in camera_tensors.items()}
    quat_norms = torch.linalg.vector_norm(canonical["quats"], dim=-1)
    metadata = {
        "format": input_payload["format"],
        "case": args.case,
        "gaussian_count": count,
        "sh_degree": 2,
        "camera_index": args.camera_index,
        "camera_id": camera_scalars["camera_id"],
        "resolution": [camera_scalars["width"], camera_scalars["height"]],
        "fov_x": camera_scalars["fov_x"],
        "fov_y": camera_scalars["fov_y"],
        "background": [0.0, 0.0, 0.0],
        "scale_modifier": 1.0,
        "canonical_tensor_records": tensor_records,
        "quaternion_norm": {
            "min": float(quat_norms.min().item()),
            "max": float(quat_norms.max().item()),
            "mean": float(quat_norms.mean().item()),
        },
        "camera_tensor_records": camera_records,
        "camera_equivalence": {
            "world_view_transpose_equals_w2c": bool(
                torch.equal(camera_tensors["world_view_transform"].T, camera_tensors["w2c"])
            ),
            "camera_center_equals_converted_c2w_translation": bool(
                torch.equal(camera_tensors["camera_center"], camera_tensors["converted_c2w"][:3, 3])
            ),
            "focal_x_equals_focal_y": bool(
                camera_tensors["K"][0, 0].item() == camera_tensors["K"][1, 1].item()
            ),
        },
        "selection": selection,
        "resident_gaussian_counts_by_level": resident_counts,
        "resident_gaussian_count_total": int(sum(resident_counts)),
        "assets": [
            {"path": str(path), "sha256": sha256_file(Path(path))} for path in asset_paths
        ],
        "trajectory": {"path": str(args.trajectory), "sha256": sha256_file(args.trajectory)},
        "canonical_input_file": str(args.output),
        "canonical_input_file_sha256": sha256_file(args.output),
        "official_commit": OFFICIAL_COMMIT,
        "graphdeco_commit": GRAPHDECO_COMMIT,
        "preparation_boundary": "all loading, activation, selection, gathering, concatenation, camera conversion, and CPU serialization occur before backend timing",
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
