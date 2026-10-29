#!/usr/bin/env python3
"""Build and analyze the frozen ALoD synthetic strong-LoD control.

The hierarchy and camera formulas are frozen in
``thesis_notes/ALOD_SYNTH_STRONG_LOD_CONTROL_PROTOCOL.md``.  This script does
not contain a selector variant: selection is delegated to the existing
project CPU reference or production implementation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_dhier import load_dhier, validate_numeric, validate_topology  # noqa: E402
from alod_reference import (  # noqa: E402
    HierarchyData,
    ReferenceCamera,
    ReferenceConfig,
    build_hspt,
    select_reference,
)
from benchmarks.benchmark_real_scene import (  # noqa: E402
    apply_camera_config,
    load_camera_path,
)
from scene_runtime.alod.camera import _official_projection_matrix  # noqa: E402


EXPERIMENT_ID = "ALOD-SYNTH-STRONG-LOD-CONTROL-001"
SKYBOX_ROWS = 100_000
SH_DEGREE = 1
SPT_ROOT_VOLUME = 1.5
MIN_SPT_SIZE = 100
TARGET_GRANULARITY = 0.00390625
DISTANCE_MULTIPLIER = 1.0
WIDTH = 1024
HEIGHT = 1024
FX = 512.0
FY = 512.0
CX = 511.5
CY = 511.5
CAMERA_COUNT = 25
LEAF_SCALE = np.asarray([0.028, 0.028, 0.014], dtype=np.float64)
LEAF_OPACITY = 0.82
SH_C0 = 0.28209479177387814
EPS = 1.0e-12


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _split_axes(nx: int, ny: int) -> list[int]:
    if nx <= 0 or ny <= 0 or nx & (nx - 1) or ny & (ny - 1):
        raise ValueError("grid dimensions must be positive powers of two")
    axes: list[int] = []
    while nx > 1 or ny > 1:
        if nx >= ny and nx > 1:
            axes.append(0)
            nx //= 2
        elif ny > 1:
            axes.append(1)
            ny //= 2
        else:
            raise AssertionError("invalid split state")
    return axes


def _leaf_grid_in_tree_order(nx: int, ny: int, axes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    leaf_count = nx * ny
    paths = np.arange(leaf_count, dtype=np.uint32)
    ix = np.zeros(leaf_count, dtype=np.uint32)
    iy = np.zeros(leaf_count, dtype=np.uint32)
    remaining = {0: int(math.log2(nx)), 1: int(math.log2(ny))}
    for depth, axis in enumerate(axes):
        bit = (paths >> np.uint32(len(axes) - depth - 1)) & np.uint32(1)
        remaining[axis] -= 1
        if axis == 0:
            ix |= bit << np.uint32(remaining[axis])
        else:
            iy |= bit << np.uint32(remaining[axis])
    return ix.astype(np.int64), iy.astype(np.int64)


def _heap_to_preorder(node_count: int) -> np.ndarray:
    order = np.empty(node_count, dtype=np.int32)
    stack = [0]
    cursor = 0
    while stack:
        heap_id = stack.pop()
        order[cursor] = heap_id
        cursor += 1
        right = 2 * heap_id + 2
        if right < node_count:
            stack.append(right)
            stack.append(right - 1)
    if cursor != node_count:
        raise AssertionError("preorder traversal did not cover the tree")
    return order


def _rgb(ix: np.ndarray, iy: np.ndarray, nx: int, ny: int) -> np.ndarray:
    u = (ix.astype(np.float64) + 0.5) / nx
    v = (iy.astype(np.float64) + 0.5) / ny
    checker = ((ix // 32 + iy // 32) & 1).astype(np.float64)
    radial = np.clip(1.0 - np.sqrt((2.0 * u - 1.0) ** 2 + (2.0 * v - 1.0) ** 2), 0.0, 1.0)
    return np.stack(
        (
            np.clip(0.12 + 0.68 * u + 0.12 * checker, 0.0, 1.0),
            np.clip(0.12 + 0.68 * v + 0.10 * (1.0 - checker), 0.0, 1.0),
            np.clip(0.18 + 0.52 * radial + 0.12 * checker, 0.0, 1.0),
        ),
        axis=1,
    )


def _area_from_diag(covariance_diag: np.ndarray) -> np.ndarray:
    scales = np.sqrt(np.maximum(covariance_diag, EPS))
    return scales[:, 0] * scales[:, 1] + scales[:, 0] * scales[:, 2] + scales[:, 1] * scales[:, 2]


def _build_heap(nx: int, ny: int) -> dict[str, np.ndarray]:
    axes = _split_axes(nx, ny)
    depth = len(axes)
    leaf_count = nx * ny
    node_count = 2 * leaf_count - 1
    leaf_start = leaf_count - 1

    positions = np.empty((node_count, 3), dtype=np.float64)
    covariance_diag = np.empty((node_count, 3), dtype=np.float64)
    opacity = np.empty(node_count, dtype=np.float64)
    sh = np.zeros((node_count, 4, 3), dtype=np.float64)

    ix, iy = _leaf_grid_in_tree_order(nx, ny, axes)
    positions[leaf_start:, 0] = -32.0 + (ix + 0.5) * (64.0 / nx)
    positions[leaf_start:, 1] = -16.0 + (iy + 0.5) * (32.0 / ny)
    positions[leaf_start:, 2] = 0.0
    covariance_diag[leaf_start:] = LEAF_SCALE**2
    opacity[leaf_start:] = LEAF_OPACITY
    sh[leaf_start:, 0, :] = (_rgb(ix, iy, nx, ny) - 0.5) / SH_C0

    for level in range(depth - 1, -1, -1):
        parent_ids = np.arange((1 << level) - 1, (1 << (level + 1)) - 1, dtype=np.int64)
        first = 2 * parent_ids + 1
        second = first + 1
        child_area_0 = _area_from_diag(covariance_diag[first])
        child_area_1 = _area_from_diag(covariance_diag[second])
        weight_0 = opacity[first] * child_area_0
        weight_1 = opacity[second] * child_area_1
        total = weight_0 + weight_1
        a0 = weight_0 / total
        a1 = weight_1 / total
        positions[parent_ids] = a0[:, None] * positions[first] + a1[:, None] * positions[second]
        diff0 = positions[first] - positions[parent_ids]
        diff1 = positions[second] - positions[parent_ids]
        covariance_diag[parent_ids] = (
            a0[:, None] * (covariance_diag[first] + diff0 * diff0)
            + a1[:, None] * (covariance_diag[second] + diff1 * diff1)
        )
        sh[parent_ids] = a0[:, None, None] * sh[first] + a1[:, None, None] * sh[second]
        parent_area = _area_from_diag(covariance_diag[parent_ids])
        opacity[parent_ids] = np.clip(total / parent_area, 1.0e-4, 0.999)

    return {
        "positions": positions,
        "covariance_diag": covariance_diag,
        "opacity": opacity,
        "sh": sh,
        "axes": np.asarray(axes, dtype=np.int8),
    }


def _tree_nodes(pre_to_heap: np.ndarray, depth: int, global_offset: int) -> np.ndarray:
    node_count = int(pre_to_heap.size)
    heap_to_pre = np.empty(node_count, dtype=np.int32)
    heap_to_pre[pre_to_heap] = np.arange(node_count, dtype=np.int32)
    heap_ids = pre_to_heap.astype(np.int64)
    levels = np.floor(np.log2(heap_ids + 1)).astype(np.int32)
    parent_heap = (heap_ids - 1) // 2
    parents = np.where(heap_ids == 0, -1, heap_to_pre[np.maximum(parent_heap, 0)] + global_offset)
    interior = levels < depth
    first_heap = 2 * heap_ids + 1
    first = np.where(interior, heap_to_pre[np.minimum(first_heap, node_count - 1)] + global_offset, 0)
    is_first_child = np.logical_and(heap_ids > 0, (heap_ids & 1) == 1)
    sibling_heap = heap_ids + 1
    sibling = np.where(is_first_child, heap_to_pre[np.minimum(sibling_heap, node_count - 1)] + global_offset, 0)
    nodes = np.empty((node_count, 6), dtype=np.int32)
    nodes[:, 0] = levels
    nodes[:, 1] = parents.astype(np.int32)
    nodes[:, 2] = np.where(interior, 2, 0).astype(np.int32)
    nodes[:, 3] = first.astype(np.int32)
    nodes[:, 4] = sibling.astype(np.int32)
    nodes[:, 5] = 0
    return nodes


def _skybox() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i = np.arange(SKYBOX_ROWS, dtype=np.float64)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / SKYBOX_ROWS
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = golden * i
    positions = 256.0 * np.stack((np.cos(theta) * radius_xy, y, np.sin(theta) * radius_xy), axis=1)
    rotations = np.zeros((SKYBOX_ROWS, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    log_scales = np.full((SKYBOX_ROWS, 3), math.log(0.01), dtype=np.float32)
    opacity = np.full(SKYBOX_ROWS, 1.0e-6, dtype=np.float32)
    sh = np.zeros((SKYBOX_ROWS, 4, 3), dtype=np.float32)
    nodes = np.full((SKYBOX_ROWS, 6), -99, dtype=np.int32)
    nodes[:, 5] = 0
    return positions.astype(np.float32), rotations, log_scales, opacity, sh, nodes


def _write_dhier(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(arrays["positions"].shape[0])
    with path.open("wb") as handle:
        handle.write(struct.pack("<ii", count, SH_DEGREE))
        for name in ("positions", "rotations", "log_scales", "opacity", "sh"):
            np.asarray(arrays[name], dtype="<f4", order="C").tofile(handle)
        handle.write(struct.pack("<i", count))
        np.asarray(arrays["nodes"], dtype="<i4", order="C").tofile(handle)


def _depth_structure(nodes: np.ndarray, positions: np.ndarray, covariance_diag: np.ndarray) -> dict[str, Any]:
    depths = nodes[:, 0]
    rows = []
    for depth in range(int(depths.max()) + 1):
        mask = depths == depth
        scales = np.sqrt(np.maximum(covariance_diag[mask], EPS))
        extent = 6.0 * np.max(scales, axis=1)
        rows.append(
            {
                "depth": depth,
                "node_count": int(mask.sum()),
                "median_extent_6sigma": float(np.median(extent)),
                "min_extent_6sigma": float(extent.min()),
                "max_extent_6sigma": float(extent.max()),
                "median_scale": [float(v) for v in np.median(scales, axis=0)],
                "position_min": [float(v) for v in positions[mask].min(axis=0)],
                "position_max": [float(v) for v in positions[mask].max(axis=0)],
            }
        )
    for index in range(len(rows) - 1):
        rows[index]["median_parent_child_extent_ratio"] = (
            rows[index]["median_extent_6sigma"] / rows[index + 1]["median_extent_6sigma"]
        )
    rows[-1]["median_parent_child_extent_ratio"] = None
    return {"by_depth": rows}


def build_asset(path: Path, nx: int, ny: int, role: str) -> dict[str, Any]:
    heap = _build_heap(nx, ny)
    axes = heap.pop("axes")
    hierarchy_count = 2 * nx * ny - 1
    depth = len(axes)
    pre_to_heap = _heap_to_preorder(hierarchy_count)
    positions = heap["positions"][pre_to_heap]
    covariance_diag = heap["covariance_diag"][pre_to_heap]
    opacity = heap["opacity"][pre_to_heap]
    sh = heap["sh"][pre_to_heap]

    # The frozen analytic scene remains axis-aligned.  The generic symmetric
    # eigendecomposition is still executed; the covariance reconstruction check
    # below verifies the serialized eigensystem before float32 conversion.
    covariance = np.zeros((hierarchy_count, 3, 3), dtype=np.float64)
    covariance[:, 0, 0] = covariance_diag[:, 0]
    covariance[:, 1, 1] = covariance_diag[:, 1]
    covariance[:, 2, 2] = covariance_diag[:, 2]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, EPS)
    determinant = np.linalg.det(eigenvectors)
    eigenvectors[determinant < 0.0, :, 2] *= -1.0
    scales = np.sqrt(eigenvalues)
    quats_xyzw = Rotation.from_matrix(eigenvectors).as_quat(canonical=False)
    rotations = np.column_stack((quats_xyzw[:, 3], quats_xyzw[:, :3]))
    reconstructed = eigenvectors @ (eigenvalues[:, :, None] * np.swapaxes(eigenvectors, 1, 2))
    covariance_error = float(np.max(np.abs(reconstructed - covariance)))

    hierarchy_nodes = _tree_nodes(pre_to_heap, depth, SKYBOX_ROWS)
    sky_pos, sky_rot, sky_log, sky_opacity, sky_sh, sky_nodes = _skybox()
    arrays = {
        "positions": np.concatenate((sky_pos, positions.astype(np.float32))),
        "rotations": np.concatenate((sky_rot, rotations.astype(np.float32))),
        "log_scales": np.concatenate((sky_log, np.log(scales).astype(np.float32))),
        "opacity": np.concatenate((sky_opacity, opacity.astype(np.float32))),
        "sh": np.concatenate((sky_sh, sh.astype(np.float32))),
        "nodes": np.concatenate((sky_nodes, hierarchy_nodes)),
    }
    _write_dhier(path, arrays)
    structure = _depth_structure(hierarchy_nodes, positions, covariance_diag)
    report = {
        "experiment_id": EXPERIMENT_ID,
        "artifact_role": role,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "file_bytes": path.stat().st_size,
        "grid": {"nx": nx, "ny": ny, "leaf_count": nx * ny},
        "split_axes": ["x" if int(v) == 0 else "y" for v in axes],
        "skybox_rows": SKYBOX_ROWS,
        "hierarchy_rows": hierarchy_count,
        "total_rows": int(arrays["positions"].shape[0]),
        "maximum_depth": depth,
        "root_global_node_id": SKYBOX_ROWS,
        "covariance_eigendecomposition_max_absolute_reconstruction_error": covariance_error,
        "aggregation": {
            "weights": "activated_opacity * (sx*sy + sx*sz + sy*sz)",
            "position_and_sh": "normalized weighted mean",
            "covariance": "weighted mixture moment matching",
            "opacity": "clip(sum child projected mass / parent area, 1e-4, 0.999)",
        },
        "structure": structure,
    }
    write_json(path.with_suffix(".construction.json"), report)
    return report


def _camera_c2w(distance: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    matrix[:3, 3] = np.asarray([0.0, 0.0, distance], dtype=np.float32)
    return matrix


def build_cameras(path: Path) -> dict[str, Any]:
    K = np.asarray([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float32)
    cameras = []
    for index in range(CAMERA_COUNT):
        distance = 40.0 * 2.0 ** (index / 3.0)
        c2w = _camera_c2w(distance)
        cameras.append(
            {
                "name": f"synth-near-to-far-{index:02d}",
                "c2w": c2w.tolist(),
                "K": K.tolist(),
                "camera_center": [0.0, 0.0, distance],
                "width": WIDTH,
                "height": HEIGHT,
                "trajectory_index": index,
                "distance": distance,
                "projected_scale_proxy": FX / distance,
            }
        )
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "trajectory_id": f"{EXPERIMENT_ID}-NEAR-TO-FAR-25-V1",
        "width": WIDTH,
        "height": HEIGHT,
        "K": K.tolist(),
        "camera_count": CAMERA_COUNT,
        "distance_formula": "40 * 2^(j/3), j=0,...,24",
        "orientation": "OpenCV c2w diag(1,-1,-1), centre (0,0,+distance), looks at origin, world up +y",
        "cameras": cameras,
    }
    write_json(path, payload)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), **payload}


def validate_asset(path: Path, output: Path) -> dict[str, Any]:
    asset = load_dhier(path)
    numeric = validate_numeric(asset, profile="strict", raise_on_error=False)
    topology = validate_topology(
        asset, skybox_count=SKYBOX_ROWS, profile="strict", raise_on_error=False
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "asset": str(path.resolve()),
        "asset_sha256": sha256_file(path),
        "layout": asdict(asset.layout),
        "numeric": numeric,
        "topology": topology,
        "status": "PASS-SYNTH-DHIER-INDEPENDENT-VALIDATION"
        if numeric["valid"] and topology["valid"]
        else "FAIL-SYNTH-DHIER-INDEPENDENT-VALIDATION",
    }
    write_json(output, result)
    if not numeric["valid"] or not topology["valid"]:
        raise RuntimeError(result["status"])
    return result


def reference_config() -> ReferenceConfig:
    return ReferenceConfig(
        root_global_node_id=SKYBOX_ROWS,
        skybox_count=SKYBOX_ROWS,
        spt_root_volume=SPT_ROOT_VOLUME,
        min_spt_size=MIN_SPT_SIZE,
        target_granularity=TARGET_GRANULARITY,
        distance_multiplier=DISTANCE_MULTIPLIER,
    )


def _reference_cameras(path: Path) -> list[ReferenceCamera]:
    loaded = apply_camera_config(
        load_camera_path(path, f"{EXPERIMENT_ID}-NEAR-TO-FAR-25-V1"),
        {"camera_resolution_scale": 1.0},
    )
    result = []
    import torch

    device = torch.device("cpu")
    for camera in loaded.cameras:
        c2w = torch.from_numpy(camera.c2w)
        w2c = torch.linalg.inv(c2w)
        world_view = w2c.T.contiguous()
        fov_x = 2.0 * math.atan(WIDTH / (2.0 * FX))
        fov_y = 2.0 * math.atan(HEIGHT / (2.0 * FY))
        projection = _official_projection_matrix(0.01, 100.0, fov_x, fov_y, device)
        full = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
        result.append(
            ReferenceCamera(
                stable_id=str(camera.camera_id),
                camera_center=np.ascontiguousarray(camera.c2w[:3, 3], dtype=np.float32),
                full_projection_transform=np.ascontiguousarray(full.numpy(), dtype=np.float32),
            )
        )
    return result


def select_reference_trajectory(asset_path: Path, camera_path: Path, output: Path) -> dict[str, Any]:
    asset = load_dhier(asset_path)
    hierarchy = HierarchyData.from_dhier(asset)
    config = reference_config()
    hspt = build_hspt(hierarchy, config)
    cameras = _reference_cameras(camera_path)
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    ids_dir = output.parent / f"{output.stem}.selected_ids"
    ids_dir.mkdir(parents=True, exist_ok=True)
    for index, camera in enumerate(cameras):
        selected = select_reference(hspt, camera, config)
        ids = selected.selected_hierarchy_global_node_ids.astype(np.int32)
        ids_path = ids_dir / f"camera_{index:02d}.npy"
        np.save(ids_path, ids, allow_pickle=False)
        depths = np.asarray(asset.nodes[ids, 0], dtype=np.int32)
        unique, counts = np.unique(depths, return_counts=True)
        rows.append(
            {
                "camera_index": index,
                "camera_stable_id": camera.stable_id,
                "distance": 40.0 * 2.0 ** (index / 3.0),
                "projected_scale_proxy": FX / (40.0 * 2.0 ** (index / 3.0)),
                "selected_hierarchy_rows": int(ids.size),
                "selected_raster_rows": int(selected.raster_global_node_ids.size),
                "mean_depth": float(depths.mean(dtype=np.float64)),
                "median_depth": float(np.median(depths)),
                "maximum_depth": int(depths.max()),
                "depth_histogram": {str(int(d)): int(c) for d, c in zip(unique, counts)},
                "selected_ids_path": str(ids_path.resolve()),
                "selected_ids_sha256": sha256_file(ids_path),
            }
        )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "selector": "independent CPU reference implementing pinned official semantics",
        "asset_sha256": sha256_file(asset_path),
        "camera_path_sha256": sha256_file(camera_path),
        "configuration": {
            "root_global_node_id": SKYBOX_ROWS,
            "skybox_count": SKYBOX_ROWS,
            "spt_root_volume": SPT_ROOT_VOLUME,
            "min_spt_size": MIN_SPT_SIZE,
            "target_granularity": TARGET_GRANULARITY,
            "distance_multiplier": DISTANCE_MULTIPLIER,
        },
        "hierarchy_maximum_depth": int(np.max(asset.nodes[config.skybox_count:, 0])),
        "hspt": {
            "upper_rows": int(hspt.upper_tree_nodes.shape[0]),
            "spt_count": int(hspt.spt_count),
            "flattened_spt_rows": int(hspt.flattened_spt_row_count),
        },
        "cameras": rows,
    }
    write_json(output, result)
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    slopes = []
    for i in range(x.size):
        delta = x[i + 1 :] - x[i]
        mask = delta != 0
        slopes.extend(((y[i + 1 :][mask] - y[i]) / delta[mask]).tolist())
    return float(np.median(np.asarray(slopes, dtype=np.float64)))


def classify(selection_path: Path, output: Path) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    records = selection["cameras"]
    ids = [np.load(record["selected_ids_path"], allow_pickle=False) for record in records]
    counts = np.asarray([row["selected_hierarchy_rows"] for row in records], dtype=np.float64)
    means = np.asarray([row["mean_depth"] for row in records], dtype=np.float64)
    maximum_depth = int(selection["hierarchy_maximum_depth"])
    x = np.log2(
        np.asarray([row["projected_scale_proxy"] for row in records], dtype=np.float64)
        / np.median([row["projected_scale_proxy"] for row in records])
    )
    y = means / maximum_depth

    min_jaccard = 1.0
    max_tvd = 0.0
    for i in range(len(records)):
        set_i = set(ids[i].tolist())
        hist_i = records[i]["depth_histogram"]
        for j in range(i + 1, len(records)):
            set_j = set(ids[j].tolist())
            union = len(set_i | set_j)
            min_jaccard = min(min_jaccard, len(set_i & set_j) / union if union else 1.0)
            depths = set(hist_i) | set(records[j]["depth_histogram"])
            tvd = 0.5 * sum(
                abs(hist_i.get(d, 0) / counts[i] - records[j]["depth_histogram"].get(d, 0) / counts[j])
                for d in depths
            )
            max_tvd = max(max_tvd, tvd)

    count_range = float((counts.max() - counts.min()) / np.median(counts))
    criteria = {
        "material_count_variation": bool(count_range >= 0.05),
        "meaningful_set_variation": bool(min_jaccard <= 0.95),
        "material_depth_variation": bool(max_tvd >= 0.05
        or max(row["maximum_depth"] for row in records) > min(row["maximum_depth"] for row in records)),
    }
    rho = float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])
    slope = _theil_sen(x, y)
    order = np.argsort(x)
    quintile = max(1, len(records) // 5)
    far = order[:quintile]
    near = order[-quintile:]
    near_far = float(np.median(y[near]) - np.median(y[far]))
    span = float(y.max() - y.min())
    regimes = [chunk for chunk in np.array_split(np.argsort(-x), 5)]
    regime_medians = [float(np.median(y[chunk])) for chunk in regimes]
    ordered_regimes = all(a >= b for a, b in zip(regime_medians, regime_medians[1:]))
    representatives = [0, len(records) // 2, len(records) - 1]
    rounded = [int(round(means[index])) for index in representatives]
    representative_pass = len(set(rounded)) >= 3 and rounded[0] - rounded[-1] >= math.ceil(0.35 * maximum_depth)
    gate = {
        "criteria_v1_all": bool(all(criteria.values())),
        "spearman_at_least_0_95": bool(rho >= 0.95),
        "theil_sen_at_least_0_06": bool(slope >= 0.06),
        "nearest_minus_farthest_quintile_at_least_0_40": bool(near_far >= 0.40),
        "normalized_mean_depth_span_at_least_0_40": bool(span >= 0.40),
        "five_regime_ordered_response": bool(ordered_regimes),
        "representative_depth_bands": bool(representative_pass),
    }
    passed = all(gate.values())
    result = {
        "experiment_id": EXPERIMENT_ID,
        "classification": "PASS-SYNTH-STRONG-LOD-CONTROL" if passed else "FAIL-SYNTH-STRONG-LOD-CONTROL",
        "criteria_v1": {
            **criteria,
            "selected_count_relative_range": count_range,
            "minimum_pairwise_jaccard": min_jaccard,
            "maximum_pairwise_depth_tvd": max_tvd,
        },
        "positive_control_response": {
            "maximum_hierarchy_depth": maximum_depth,
            "projected_scale_mean_depth_spearman": rho,
            "theil_sen_normalized_depth_per_scale_octave": slope,
            "nearest_minus_farthest_quintile_normalized_depth": near_far,
            "normalized_mean_depth_span": span,
            "five_near_to_far_regime_median_normalized_depths": regime_medians,
            "representative_camera_indices": representatives,
            "representative_rounded_mean_depths": rounded,
            "gate": gate,
        },
        "per_camera": records,
    }
    write_json(output, result)
    return result


def command_build(args: argparse.Namespace) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    camera_report = build_cameras(args.output_root / "cameras.json")
    reports = [
        build_asset(args.output_root / "smoke.dhier", 64, 32, "correctness-smoke"),
        build_asset(args.output_root / "performance.dhier", 1024, 512, "scientific-performance-scale"),
    ]
    write_json(
        args.output_root / "build_manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol": str((WORKSPACE_ROOT / "thesis_notes/ALOD_SYNTH_STRONG_LOD_CONTROL_PROTOCOL.md").resolve()),
            "cameras": camera_report,
            "artifacts": reports,
        },
    )
    return 0


def command_validate(args: argparse.Namespace) -> int:
    validate_asset(args.asset, args.output)
    return 0


def command_select(args: argparse.Namespace) -> int:
    select_reference_trajectory(args.asset, args.cameras, args.output)
    return 0


def command_classify(args: argparse.Namespace) -> int:
    result = classify(args.selection, args.output)
    return 0 if result["classification"].startswith("PASS") else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "runs/alod/ALOD-SYNTH-STRONG-LOD-CONTROL-001",
    )
    build.set_defaults(func=command_build)
    validate = sub.add_parser("validate")
    validate.add_argument("--asset", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.set_defaults(func=command_validate)
    select = sub.add_parser("select")
    select.add_argument("--asset", type=Path, required=True)
    select.add_argument("--cameras", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.set_defaults(func=command_select)
    classify_command = sub.add_parser("classify")
    classify_command.add_argument("--selection", type=Path, required=True)
    classify_command.add_argument("--output", type=Path, required=True)
    classify_command.set_defaults(func=command_classify)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
