#!/usr/bin/env python3
"""Cut-only HSPT/BFS ablation using the pinned official ALoD primitives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.dont_write_bytecode = True


OFFICIAL_COMMIT = "73f547341547ae021ff036ad74fdf73a1240b450"
DIAGNOSTIC_INDICES = (6, 9, 10, 13, 15, 22, 24, 25, 32, 35, 37, 48, 50, 52, 58, 65)


def canonical(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_hash(value: np.ndarray, *, sort: bool = False) -> str:
    ids = np.asarray(value, dtype="<i4")
    if sort:
        ids = np.sort(ids, kind="stable")
    return hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest()


def distribution(values: list[float | int]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(a.min()),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p95": float(np.quantile(a, 0.95)),
        "maximum": float(a.max()),
    }


def logical_bytes(*tensors: torch.Tensor) -> int:
    return int(sum(value.numel() * value.element_size() for value in tensors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--camera-fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    source = args.source.resolve()
    dataset = args.dataset.resolve()
    hierarchy = args.hierarchy.resolve()
    camera_fixture = args.camera_fixture.resolve()
    output_dir = args.output_dir.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"official source commit mismatch: {commit}")
    if sha256_file(hierarchy) != args.expected_sha256:
        raise RuntimeError("formal hierarchy SHA-256 mismatch")
    os.chdir(source)
    sys.path.insert(0, str(source))
    from gaussian_hierarchy._C import get_spt_cut_cuda
    from scene.dataset_readers import readColmapSceneInfo
    from scene.gaussian_model import GaussianModel
    from utils.camera_utils import loadCam
    from utils.general_utils import safe_state

    fixture = json.loads(camera_fixture.read_text(encoding="utf-8"))
    expected_names = [row["name"] for row in fixture["cameras"]]
    safe_state(False)
    info = readColmapSceneInfo(
        str(dataset),
        str(dataset / "images"),
        "",
        "",
        True,
        False,
        10,
    )
    actual_names = [row.image_name for row in info.test_cameras]
    if actual_names != expected_names:
        raise RuntimeError("official eval camera order differs from frozen 71-camera fixture")

    model = GaussianModel(1)
    model.create_from_hier(
        str(hierarchy), info.nerf_normalization["radius"], ""
    )
    model.compact_gaussians("cpu", None, False, False)
    model.skybox_points = 100000
    focal = float(info.train_cameras[0].focal_length)
    target = 2.0 / focal
    model.build_hierarchical_SPT(
        4.75, target, 100, use_bounding_spheres=False
    )
    torch.cuda.synchronize()

    hspt_tensors = (
        model.upper_tree_nodes,
        model.upper_tree_xyz,
        model.upper_tree_scaling,
        model.min_distance_squared,
        model.SPT_gaussian_indices,
        model.SPT_starts,
        model.SPT_max,
        model.SPT_min,
    )
    hspt_logical_gpu_bytes = logical_bytes(*hspt_tensors)
    allocated_after_hspt = int(torch.cuda.memory_allocated())
    reserved_after_hspt = int(torch.cuda.memory_reserved())

    nodes = model.nodes.cuda()
    xyz = model.properties[:, 0:3].cuda()
    log_scales = model.properties[:, 3:6].cuda()
    full_bounds = torch.exp(log_scales).max(dim=-1).values * 3.0
    parents = nodes[:, 1].to(torch.int64).clamp(min=0)
    parent_scales = torch.exp(log_scales.index_select(0, parents))
    full_min_distance_squared = (
        parent_scales[:, 0] * parent_scales[:, 1]
        + parent_scales[:, 0] * parent_scales[:, 2]
        + parent_scales[:, 1] * parent_scales[:, 2]
    ) / (target * target)
    full_min_distance_squared[100000] = 1.0e12
    del parents, parent_scales
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    bfs_logical_gpu_bytes = logical_bytes(
        nodes, xyz, log_scales, full_bounds, full_min_distance_squared
    )
    allocated_with_bfs = int(torch.cuda.memory_allocated())
    reserved_with_bfs = int(torch.cuda.memory_reserved())

    def hspt_cut(camera) -> torch.Tensor:
        center = camera.camera_center.cuda()
        full_proj = camera.full_proj_transform.cuda()
        bounds = torch.exp(model.upper_tree_scaling).max(dim=-1).values * 3.0
        planes = model.extract_frustum_planes(full_proj)
        cull = lambda indices: model.frustum_cull_spheres(
            model.upper_tree_xyz[indices], bounds[indices], planes
        )
        condition = lambda indices: model.min_distance_squared[indices] > (
            center - model.upper_tree_xyz[indices]
        ).square().sum(dim=-1)
        coarse = model.cut_hierarchy_on_condition(
            model.upper_tree_nodes,
            condition,
            return_upper_tree=False,
            root_node=0,
            leave_out_of_cut_condition=cull,
        )
        leaf = coarse[model.upper_tree_nodes[coarse, 2] == 0]
        spt_indices = model.upper_tree_nodes[leaf][
            model.upper_tree_nodes[leaf, 3] >= 0, 3
        ]
        upper = model.upper_tree_nodes[coarse][
            model.upper_tree_nodes[coarse, 3] <= 0, 5
        ]
        spt_upper = leaf[model.upper_tree_nodes[leaf, 3] >= 0]
        distances = (
            model.upper_tree_xyz[spt_upper] - center
        ).pow(2).sum(1).sqrt()
        if spt_indices.numel():
            selected, starts = get_spt_cut_cuda(
                len(spt_indices),
                model.SPT_gaussian_indices,
                model.SPT_starts,
                model.SPT_max,
                model.SPT_min,
                spt_indices,
                distances,
            )
            if starts.numel() and selected.numel() == int(starts[-1].item()):
                starts = starts[:-1]
                distances = distances[:-1]
                spt_indices = spt_indices[:-1]
        else:
            selected = torch.empty(0, dtype=torch.int32, device="cuda")
        return torch.cat((upper, selected))

    def bfs_cut(camera) -> torch.Tensor:
        center = camera.camera_center.cuda()
        planes = model.extract_frustum_planes(camera.full_proj_transform.cuda())
        cull = lambda indices: model.frustum_cull_spheres(
            xyz[indices], full_bounds[indices], planes
        )
        condition = lambda indices: full_min_distance_squared[indices] > (
            center - xyz[indices]
        ).square().sum(dim=-1)
        return model.cut_hierarchy_on_condition(
            nodes,
            condition,
            return_upper_tree=False,
            root_node=100000,
            leave_out_of_cut_condition=cull,
        )

    camera_args = SimpleNamespace(
        resolution=2, data_device="cpu", train_test_exp=False
    )
    payload_dir = output_dir / "selected_global_ids"
    payload_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for diagnostic_order, evaluation_index in enumerate(DIAGNOSTIC_INDICES):
        camera = loadCam(
            camera_args, evaluation_index, info.test_cameras[evaluation_index], 1.0, True
        )
        hspt_cut(camera)
        bfs_cut(camera)
        torch.cuda.synchronize()
        hspt_times, bfs_times = [], []
        hspt_selected = bfs_selected = None
        for _ in range(args.repetitions):
            torch.cuda.synchronize()
            started = time.perf_counter()
            hspt_selected = hspt_cut(camera)
            torch.cuda.synchronize()
            hspt_times.append((time.perf_counter() - started) * 1000.0)
            torch.cuda.synchronize()
            started = time.perf_counter()
            bfs_selected = bfs_cut(camera)
            torch.cuda.synchronize()
            bfs_times.append((time.perf_counter() - started) * 1000.0)
        assert hspt_selected is not None and bfs_selected is not None
        hspt_ids = hspt_selected.detach().cpu().numpy().astype(np.int32, copy=False)
        bfs_ids = bfs_selected.detach().cpu().numpy().astype(np.int32, copy=False)
        payload = payload_dir / f"camera-{evaluation_index:02d}.npz"
        np.savez_compressed(payload, hspt=hspt_ids, bfs=bfs_ids)
        hspt_sorted = np.sort(hspt_ids, kind="stable")
        bfs_sorted = np.sort(bfs_ids, kind="stable")
        intersection = np.intersect1d(
            hspt_sorted, bfs_sorted, assume_unique=True
        ).size
        union = hspt_ids.size + bfs_ids.size - intersection
        hspt_depth = model.nodes[hspt_ids, 0].numpy()
        bfs_depth = model.nodes[bfs_ids, 0].numpy()
        rows.append(
            {
                "diagnostic_order": diagnostic_order,
                "evaluation_sequence_index": evaluation_index,
                "camera_name": expected_names[evaluation_index],
                "payload": str(payload.relative_to(output_dir)),
                "hspt": {
                    "selected_count": int(hspt_ids.size),
                    "ordered_ids_sha256": ids_hash(hspt_ids),
                    "sorted_ids_sha256": ids_hash(hspt_ids, sort=True),
                    "cut_time_ms": distribution(hspt_times),
                    "depth_histogram": {
                        str(int(value)): int(count)
                        for value, count in zip(*np.unique(hspt_depth, return_counts=True))
                    },
                },
                "bfs": {
                    "selected_count": int(bfs_ids.size),
                    "ordered_ids_sha256": ids_hash(bfs_ids),
                    "sorted_ids_sha256": ids_hash(bfs_ids, sort=True),
                    "cut_time_ms": distribution(bfs_times),
                    "depth_histogram": {
                        str(int(value)): int(count)
                        for value, count in zip(*np.unique(bfs_depth, return_counts=True))
                    },
                },
                "identity": {
                    "exact_order": bool(np.array_equal(hspt_ids, bfs_ids)),
                    "exact_set": bool(np.array_equal(hspt_sorted, bfs_sorted)),
                    "intersection_count": int(intersection),
                    "symmetric_difference_count": int(
                        hspt_ids.size + bfs_ids.size - 2 * intersection
                    ),
                    "jaccard": float(intersection / union) if union else 1.0,
                },
            }
        )
        del camera

    hspt_ms = [row["hspt"]["cut_time_ms"]["median"] for row in rows]
    bfs_ms = [row["bfs"]["cut_time_ms"]["median"] for row in rows]
    summary = {
        "status": "PASS-ALOD-REPRO-HSPT-BFS",
        "classification": {
            "hspt": "exact released evaluator cut path",
            "bfs": "minimal source-faithful full-hierarchy reconstruction using released BFS traversal primitive and identical LoD/frustum predicates",
        },
        "official_source_commit": commit,
        "official_source_clean": subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ) == "",
        "hierarchy": {
            "path": str(hierarchy),
            "sha256": args.expected_sha256,
            "total_rows": int(model.nodes.shape[0]),
            "skybox_rows_excluded_from_cut": 100000,
        },
        "camera_fixture": {
            "path": str(camera_fixture),
            "diagnostic_evaluation_indices": list(DIAGNOSTIC_INDICES),
            "camera_count": len(rows),
        },
        "timing": {
            "repetitions_per_camera_after_one_warmup": args.repetitions,
            "synchronization": "torch.cuda.synchronize before and after each cut",
            "hspt_per_camera_median_ms": distribution(hspt_ms),
            "bfs_per_camera_median_ms": distribution(bfs_ms),
            "bfs_over_hspt_ratio_of_means": float(np.mean(bfs_ms) / np.mean(hspt_ms)),
        },
        "identity": {
            "all_exact_sets": all(row["identity"]["exact_set"] for row in rows),
            "all_exact_order": all(row["identity"]["exact_order"] for row in rows),
            "jaccard": distribution([row["identity"]["jaccard"] for row in rows]),
            "selected_count_delta_bfs_minus_hspt": distribution(
                [row["bfs"]["selected_count"] - row["hspt"]["selected_count"] for row in rows]
            ),
        },
        "memory": {
            "hspt_logical_gpu_metadata_bytes": hspt_logical_gpu_bytes,
            "bfs_full_hierarchy_logical_gpu_metadata_bytes": bfs_logical_gpu_bytes,
            "bfs_over_hspt_logical_gpu_ratio": bfs_logical_gpu_bytes / hspt_logical_gpu_bytes,
            "torch_allocated_after_hspt_bytes": allocated_after_hspt,
            "torch_reserved_after_hspt_bytes": reserved_after_hspt,
            "torch_allocated_with_bfs_bytes": allocated_with_bfs,
            "torch_reserved_with_bfs_bytes": reserved_with_bfs,
            "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        },
        "paper_comparability": "trend-only: local Run009 proxy, RTX 5080, and reconstructed BFS entry point differ from the paper assets/harness/GPU",
    }
    (output_dir / "per_camera.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(canonical(summary), encoding="utf-8")
    print(canonical({"status": summary["status"], "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
