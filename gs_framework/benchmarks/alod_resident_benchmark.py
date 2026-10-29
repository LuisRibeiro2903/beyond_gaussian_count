#!/usr/bin/env python3
"""Resident-only ALoD ablation validation, quality, and aggregation tools.

This module intentionally does not implement streaming, caching, prefetching,
working-set reuse, or alternative hierarchy/selection behavior.  It evaluates
the three tracked fully resident configurations around the shared benchmark
runner and production selector.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_dhier import load_dhier, validate_numeric, validate_topology  # noqa: E402
from benchmarks.benchmark_real_scene import (  # noqa: E402
    CONFIG_FILES,
    GsplatRenderer,
    apply_camera_config,
    canonical_json_sha256,
    environment_metadata,
    json_safe,
    load_camera_path,
    load_config,
    load_scene_for_benchmark,
    make_frame_camera,
    make_render_settings,
    require_cuda,
    resolve_camera_path,
    sha256_file,
    write_json,
)
from benchmarks.compare_citygaussian_same_asset import render_for_comparison  # noqa: E402
from scene_runtime.alod import (  # noqa: E402
    AlodHierarchySource,
    AlodRuntimeConfig,
    build_production_hspt,
    select_production,
    selection_camera_from_frame,
)


CONFIG_IDS = (
    "ALOD-HIER-FINEST-RESIDENT",
    "ALOD-HIER-LOD-RESIDENT",
    "ALOD-HIER-LOD-FRUSTUM",
)
PASS_SEMANTICS = "PASS-ALOD-RESIDENT-ABLATION-SEMANTICS"


def _runtime_config(config: dict[str, Any]) -> AlodRuntimeConfig:
    raw = config["alod"]
    return AlodRuntimeConfig(
        root_global_node_id=int(raw["root_global_node_id"]),
        skybox_count=int(raw["skybox_count"]),
        spt_root_volume=float(raw["spt_root_volume"]),
        min_spt_size=int(raw["min_spt_size"]),
        target_granularity=float(raw["target_granularity"]),
        distance_multiplier=float(raw.get("distance_multiplier", 1.0)),
        selection_mode=str(raw["selection_mode"]),
        frustum_enabled=bool(raw["frustum_enabled"]),
        materialization_filter=str(raw.get("materialization_filter", "none")),
    )


def _configured_camera_path(config: dict[str, Any]):
    return apply_camera_config(
        load_camera_path(resolve_camera_path(config), config.get("trajectory_id")),
        config,
    )


def _selection_camera(camera, config: dict[str, Any]):
    frame = make_frame_camera(camera, torch.device("cpu"))
    settings = make_render_settings(config, camera, sh_degree=int(config["sh_degree"]))
    return selection_camera_from_frame(
        frame, settings, stable_id=camera.camera_id
    )


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _depth_histogram(depths: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(depths.astype(np.int64), return_counts=True)
    return {str(int(depth)): int(count) for depth, count in zip(unique, counts)}


def _hierarchy_cut_validation(
    nodes: np.ndarray,
    selected_ids: np.ndarray,
    *,
    root_id: int,
    skybox_count: int,
    require_complete: bool,
    require_no_ancestor_duplication: bool,
) -> dict[str, Any]:
    selected_ids = np.asarray(selected_ids, dtype=np.int64)
    row_count = int(nodes.shape[0])
    if selected_ids.ndim != 1:
        raise RuntimeError("selected hierarchy IDs must be one-dimensional")
    if selected_ids.size != np.unique(selected_ids).size:
        raise RuntimeError("selected hierarchy IDs contain duplicates")
    if selected_ids.size and (
        int(selected_ids.min()) < skybox_count or int(selected_ids.max()) >= row_count
    ):
        raise RuntimeError("selected hierarchy IDs lie outside hierarchy rows")

    selected = np.zeros(row_count, dtype=bool)
    selected[selected_ids] = True
    depths = nodes[:, 0].astype(np.int64, copy=False)
    order = np.argsort(depths[skybox_count:], kind="stable") + skybox_count
    has_selected_ancestor = np.zeros(row_count, dtype=bool)
    selected_path_count = np.zeros(row_count, dtype=np.int16)
    for depth in np.unique(depths[order]):
        level = order[depths[order] == depth]
        parents = nodes[level, 1].astype(np.int64, copy=False)
        valid_parent = parents >= skybox_count
        if np.any(valid_parent):
            child_ids = level[valid_parent]
            parent_ids = parents[valid_parent]
            has_selected_ancestor[child_ids] = (
                selected[parent_ids] | has_selected_ancestor[parent_ids]
            )
            selected_path_count[child_ids] = (
                selected_path_count[parent_ids] + selected[child_ids].astype(np.int16)
            )
        roots = level[~valid_parent]
        if roots.size:
            selected_path_count[roots] = selected[roots].astype(np.int16)

    conflicts = selected & has_selected_ancestor
    conflict_count = int(np.count_nonzero(conflicts))
    if require_no_ancestor_duplication and conflict_count:
        raise RuntimeError(
            f"ancestor/descendant duplication at {int(np.flatnonzero(conflicts)[0])}"
        )
    leaf_ids = np.flatnonzero(
        (np.arange(row_count) >= skybox_count) & (nodes[:, 2] == 0)
    )
    complete = bool(np.all(selected_path_count[leaf_ids] == 1))
    covered = bool(np.all(selected_path_count[leaf_ids] >= 1))
    if require_complete and not complete:
        invalid = int(np.count_nonzero(selected_path_count[leaf_ids] != 1))
        raise RuntimeError(f"hierarchy cut does not cover {invalid} leaf paths exactly once")
    return {
        "unique_selected_ids": True,
        "selected_ids_in_hierarchy_range": True,
        "no_ancestor_descendant_duplication": conflict_count == 0,
        "selected_nodes_with_selected_ancestor": conflict_count,
        "all_leaf_paths_covered": covered,
        "complete_one_node_per_leaf_path": complete,
        "leaf_row_count": int(leaf_ids.size),
        "root_selected": bool(selected[root_id]),
    }


def run_semantics(asset_path: Path, output: Path) -> int:
    asset_path = asset_path.resolve()
    asset = load_dhier(asset_path)
    configs = {config_id: load_config(config_id) for config_id in CONFIG_IDS}
    base = configs[CONFIG_IDS[0]]
    base_runtime = _runtime_config(base)
    validate_numeric(asset, profile="strict", raise_on_error=True)
    validate_topology(
        asset,
        skybox_count=base_runtime.skybox_count,
        profile="strict",
        raise_on_error=True,
    )
    hierarchy = AlodHierarchySource(asset.positions, asset.log_scales, asset.nodes)
    hspt = build_production_hspt(hierarchy, base_runtime)
    camera_path = _configured_camera_path(base)
    expected_skybox = np.arange(base_runtime.skybox_count, dtype=np.int64)

    all_results: dict[str, Any] = {}
    for config_id in CONFIG_IDS:
        config = configs[config_id]
        runtime = _runtime_config(config)
        configured = _configured_camera_path(config)
        if [camera.camera_id for camera in configured.cameras] != [
            camera.camera_id for camera in camera_path.cameras
        ]:
            raise RuntimeError(f"{config_id} camera order differs from the frozen path")
        rows = []
        previous_mask: np.ndarray | None = None
        for camera_index, camera in enumerate(configured.cameras):
            selection = select_production(
                hierarchy, hspt, _selection_camera(camera, config), runtime
            )
            if not np.array_equal(
                selection.raster_global_node_ids[: runtime.skybox_count], expected_skybox
            ):
                raise RuntimeError(f"{config_id} camera {camera_index} skybox prefix differs")
            if not np.array_equal(
                selection.raster_global_node_ids[runtime.skybox_count :],
                selection.selected_hierarchy_global_node_ids,
            ):
                raise RuntimeError(f"{config_id} camera {camera_index} raster order differs")
            cut = _hierarchy_cut_validation(
                asset.nodes,
                selection.selected_hierarchy_global_node_ids,
                root_id=runtime.root_global_node_id,
                skybox_count=runtime.skybox_count,
                require_complete=runtime.selection_mode == "finest",
                require_no_ancestor_duplication=runtime.selection_mode == "finest",
            )
            if runtime.selection_mode == "finest":
                expected_leaves = np.flatnonzero(
                    (np.arange(asset.layout.gaussian_count) >= runtime.skybox_count)
                    & (asset.nodes[:, 2] == 0)
                ).astype(np.int64)
                if not np.array_equal(
                    selection.selected_hierarchy_global_node_ids, expected_leaves
                ):
                    raise RuntimeError("finest selection is not the exact serialized leaf cut")

            selected = selection.selected_hierarchy_global_node_ids
            depths = asset.nodes[selected, 0].astype(np.int64, copy=False)
            current_mask = np.zeros(asset.layout.gaussian_count, dtype=bool)
            current_mask[selected] = True
            churn = None
            if previous_mask is not None:
                intersection = int(np.count_nonzero(previous_mask & current_mask))
                union = int(np.count_nonzero(previous_mask | current_mask))
                churn = {
                    "jaccard": intersection / union if union else 1.0,
                    "added": int(np.count_nonzero(~previous_mask & current_mask)),
                    "removed": int(np.count_nonzero(previous_mask & ~current_mask)),
                }
            previous_mask = current_mask
            upper = selection.upper
            rows.append(
                {
                    "camera_index": camera_index,
                    "camera_id": camera.camera_id,
                    "selected_hierarchy_rows": int(selected.size),
                    "selected_skybox_rows": runtime.skybox_count,
                    "total_raster_rows": int(selection.raster_global_node_ids.size),
                    "candidate_spt_count": int(upper.candidate_spt_ids.size)
                    if upper is not None
                    else 0,
                    "upper_emitted_count": int(upper.upper_render_global_node_ids.size)
                    if upper is not None
                    else 0,
                    "frustum_rejected_upper_count": int(
                        upper.frustum_rejected_upper_local_ids.size
                    )
                    if upper is not None
                    else 0,
                    "selected_min_depth": int(depths.min()),
                    "selected_median_depth": float(np.median(depths)),
                    "selected_max_depth": int(depths.max()),
                    "selected_depth_histogram": _depth_histogram(depths),
                    "ordered_selected_id_sha256": _array_sha256(selected.astype("<i8")),
                    "ordered_raster_id_sha256": _array_sha256(
                        selection.raster_global_node_ids.astype("<i8")
                    ),
                    "cut_validation": cut,
                    "churn_from_previous_camera": churn,
                }
            )
        counts = [row["selected_hierarchy_rows"] for row in rows]
        churns = [
            row["churn_from_previous_camera"]
            for row in rows
            if row["churn_from_previous_camera"] is not None
        ]
        all_results[config_id] = {
            "selection_mode": runtime.selection_mode,
            "frustum_enabled": runtime.frustum_enabled,
            "finest_selection_cost_included_per_frame": runtime.selection_mode == "finest",
            "frames": rows,
            "summary": {
                "minimum_selected_hierarchy_rows": min(counts),
                "median_selected_hierarchy_rows": float(np.median(counts)),
                "maximum_selected_hierarchy_rows": max(counts),
                "minimum_selected_depth": min(row["selected_min_depth"] for row in rows),
                "median_selected_depth": float(
                    np.median([row["selected_median_depth"] for row in rows])
                ),
                "maximum_selected_depth": max(row["selected_max_depth"] for row in rows),
                "minimum_sequential_jaccard": min(
                    (row["jaccard"] for row in churns), default=1.0
                ),
                "mean_sequential_jaccard": float(
                    np.mean([row["jaccard"] for row in churns])
                )
                if churns
                else 1.0,
            },
        }

    report = {
        "schema_version": 1,
        "status": PASS_SEMANTICS,
        "asset": {
            "path": str(asset_path),
            "sha256": sha256_file(asset_path),
            "total_serialized_rows": int(asset.layout.gaussian_count),
            "skybox_rows": base_runtime.skybox_count,
            "hierarchy_rows": int(asset.layout.gaussian_count - base_runtime.skybox_count),
        },
        "trajectory": {
            "id": camera_path.trajectory_id,
            "path": str(camera_path.path),
            "sha256": sha256_file(camera_path.path),
            "camera_count": len(camera_path.cameras),
            "camera_ids": [camera.camera_id for camera in camera_path.cameras],
            "resolution": [camera_path.cameras[0].width, camera_path.cameras[0].height],
            "weak_lod_oracle": True,
        },
        "hspt": {
            "upper_rows": hspt.upper_row_count,
            "spt_count": hspt.spt_count,
            "flattened_spt_rows": hspt.flattened_spt_row_count,
        },
        "configurations": all_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return 0


def _image_nchw(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = image.to(device=device, dtype=torch.float32)
    if value.ndim == 3 and value.shape[-1] in {1, 3}:
        value = value.permute(2, 0, 1).unsqueeze(0)
    if value.ndim != 4:
        raise RuntimeError(f"unexpected image shape {tuple(image.shape)}")
    # Match display/image-evaluation semantics and LPIPS' documented input
    # domain. Raster outputs can exceed [0,1] slightly before presentation.
    return value.clamp(0.0, 1.0).contiguous()


def _render_configuration(
    config: dict[str, Any], asset: Path, cameras, device: torch.device
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene, stats = load_scene_for_benchmark(config, asset, device)
    renderer = GsplatRenderer(device)
    outputs = []
    for camera in cameras:
        rendered = render_for_comparison(scene, renderer, camera, config, stats, device)
        outputs.append(
            {
                "image": rendered["image"].detach().cpu(),
                "alpha": rendered["alpha"].detach().cpu()
                if rendered["alpha"] is not None
                else None,
                "selected_hierarchy_rows": int(
                    rendered["scene_metrics"].alod_selected_hierarchy_rows
                ),
                "total_raster_rows": int(rendered["active_gaussians"]),
                "visible_gaussians": int(rendered["visible_gaussians"]),
            }
        )
    metadata = {
        "total_serialized_rows": int(stats.total_gaussians),
        "selection_mode": stats.alod_selection_mode,
        "frustum_enabled": stats.alod_frustum_enabled,
        "gaussian_properties_gpu_bytes": stats.alod_gaussian_properties_gpu_bytes,
    }
    del scene, renderer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return outputs, metadata


@torch.no_grad()
def run_quality(asset: Path, output_dir: Path, device_index: int) -> int:
    from torchmetrics.functional.image.ssim import structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    device = require_cuda(device_index)
    configs = {config_id: load_config(config_id) for config_id in CONFIG_IDS}
    camera_path = _configured_camera_path(configs[CONFIG_IDS[0]])
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered: dict[str, list[dict[str, Any]]] = {}
    scene_metadata: dict[str, Any] = {}
    for config_id in CONFIG_IDS:
        rendered[config_id], scene_metadata[config_id] = _render_configuration(
            configs[config_id], asset, camera_path.cameras, device
        )

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device)
    pairs = (
        (CONFIG_IDS[0], CONFIG_IDS[1]),
        (CONFIG_IDS[0], CONFIG_IDS[2]),
        (CONFIG_IDS[1], CONFIG_IDS[2]),
    )
    rows = []
    for left_id, right_id in pairs:
        for camera_index, camera in enumerate(camera_path.cameras):
            left = rendered[left_id][camera_index]
            right = rendered[right_id][camera_index]
            left_image = _image_nchw(left["image"], device)
            right_image = _image_nchw(right["image"], device)
            difference = left_image - right_image
            abs_difference = difference.abs()
            mse = float(torch.mean(difference.square()).item())
            psnr = float("inf") if mse == 0.0 else float(-10.0 * math.log10(mse))
            ssim = float(
                structural_similarity_index_measure(
                    right_image, left_image, data_range=1.0
                ).item()
            )
            lpips = float(lpips_metric(right_image, left_image).item())
            alpha_metrics = None
            if left["alpha"] is not None and right["alpha"] is not None:
                alpha_left = _image_nchw(left["alpha"], device)
                alpha_right = _image_nchw(right["alpha"], device)
                alpha_abs = (alpha_left - alpha_right).abs()
                alpha_metrics = {
                    "mean_absolute_error": float(alpha_abs.mean().item()),
                    "maximum_absolute_error": float(alpha_abs.max().item()),
                }
            rows.append(
                {
                    "left_configuration_id": left_id,
                    "right_configuration_id": right_id,
                    "camera_index": camera_index,
                    "camera_id": camera.camera_id,
                    "psnr_db": psnr,
                    "ssim": ssim,
                    "lpips_alex": lpips,
                    "mean_absolute_error": float(abs_difference.mean().item()),
                    "maximum_absolute_error": float(abs_difference.max().item()),
                    "mse": mse,
                    "alpha_difference": alpha_metrics,
                    "left_selected_hierarchy_rows": left["selected_hierarchy_rows"],
                    "right_selected_hierarchy_rows": right["selected_hierarchy_rows"],
                    "left_total_raster_rows": left["total_raster_rows"],
                    "right_total_raster_rows": right["total_raster_rows"],
                    "left_visible_gaussians": left["visible_gaussians"],
                    "right_visible_gaussians": right["visible_gaussians"],
                }
            )

    frames_path = output_dir / "frames.jsonl"
    with frames_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n")
    comparisons = {}
    for left_id, right_id in pairs:
        pair_rows = [
            row
            for row in rows
            if row["left_configuration_id"] == left_id
            and row["right_configuration_id"] == right_id
        ]
        key = f"{left_id}__{right_id}"
        psnrs = [float(row["psnr_db"]) for row in pair_rows]
        finite_psnrs = [value for value in psnrs if math.isfinite(value)]
        comparisons[key] = {
            "frame_count": len(pair_rows),
            "mean_finite_psnr_db": float(np.mean(finite_psnrs))
            if finite_psnrs
            else None,
            "minimum_psnr_db": float(min(psnrs)),
            "infinite_psnr_frame_count": int(
                sum(not math.isfinite(value) for value in psnrs)
            ),
            "mean_ssim": float(np.mean([row["ssim"] for row in pair_rows])),
            "minimum_ssim": float(min(row["ssim"] for row in pair_rows)),
            "mean_lpips_alex": float(
                np.mean([row["lpips_alex"] for row in pair_rows])
            ),
            "maximum_lpips_alex": float(max(row["lpips_alex"] for row in pair_rows)),
            "mean_absolute_error": float(
                np.mean([row["mean_absolute_error"] for row in pair_rows])
            ),
            "maximum_absolute_error": float(
                max(row["maximum_absolute_error"] for row in pair_rows)
            ),
        }
    summary = {
        "schema_version": 1,
        "record_kind": "alod_resident_finest_reference_quality",
        "reference_semantics": "same-asset finest valid hierarchy cut, not photography",
        "asset": {"path": str(asset.resolve()), "sha256": sha256_file(asset)},
        "trajectory": {
            "id": camera_path.trajectory_id,
            "sha256": sha256_file(camera_path.path),
            "camera_count": len(camera_path.cameras),
        },
        "metrics": {
            "input_domain": "both renders clamped to RGB/alpha [0,1]",
            "psnr": "RGB [0,1], same-asset reference",
            "ssim": "torchmetrics, data_range=1",
            "lpips": "torchmetrics AlexNet, normalize=True",
        },
        "scene_metadata": scene_metadata,
        "comparisons": comparisons,
        "environment": environment_metadata(),
    }
    write_json(output_dir / "quality_summary.json", summary)
    return 0


def _finite(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("mean", "median", "p10", "p90", "p95", "std", "min", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _percent_change(child: float, parent: float) -> float | None:
    return None if parent == 0.0 else 100.0 * (child - parent) / parent


def run_aggregate(
    campaign_root: Path,
    asset: Path,
    oracle_manifest: Path,
    output_dir: Path,
) -> int:
    semantics = _read_json(campaign_root / "checks" / "ablation_semantics.json")
    quality = _read_json(campaign_root / "quality" / "quality_summary.json")
    oracle = _read_json(oracle_manifest)
    config_summaries = []
    pooled_rows: dict[str, list[dict[str, Any]]] = {}
    for config_id in CONFIG_IDS:
        rows: list[dict[str, Any]] = []
        replicate_means = []
        startup = []
        for replicate in range(1, 4):
            run_dir = campaign_root / "runs" / config_id / f"replicate-{replicate:02d}"
            run_rows = _read_jsonl(run_dir / "frames.jsonl")
            if len(run_rows) != 100:
                raise RuntimeError(f"{run_dir} contains {len(run_rows)} frames, expected 100")
            summary = _read_json(run_dir / "summary.json")
            if summary.get("record_kind") != "formal_benchmark":
                raise RuntimeError(f"{run_dir} is not marked formal_benchmark")
            rows.extend(run_rows)
            replicate_means.append(
                float(np.mean(_finite(run_rows, "end_to_end_frame_time_ms")))
            )
            startup.append(summary.get("startup", {}))
        pooled_rows[config_id] = rows
        total_stats = _stats(_finite(rows, "end_to_end_frame_time_ms"))
        raster_stats = _stats(_finite(rows, "rasterization_time_ms"))
        config_summaries.append(
            {
                "configuration_id": config_id,
                "frame_count": len(rows),
                "process_replicates": 3,
                "complete_trajectory_repetitions_per_process": 10,
                "total_frame_time_ms": total_stats,
                "fps_from_mean_total_frame_time": 1000.0 / float(total_stats["mean"]),
                "replicate_mean_total_frame_time_ms": replicate_means,
                "replicate_mean_total_frame_time_std_ms": float(
                    statistics.pstdev(replicate_means)
                ),
                "camera_preparation_ms": _stats(
                    _finite(rows, "alod_camera_preparation_ms")
                ),
                "upper_traversal_ms": _stats(_finite(rows, "alod_upper_traversal_ms")),
                "spt_cut_ms": _stats(_finite(rows, "alod_spt_cut_ms")),
                "total_selection_ms": _stats(
                    _finite(rows, "alod_total_selection_ms")
                ),
                "selected_index_ms": _stats(_finite(rows, "alod_selected_index_ms")),
                "selected_gather_ms": _stats(_finite(rows, "alod_selected_gather_ms")),
                "pre_raster_assembly_ms": _stats(
                    _finite(rows, "alod_pre_raster_assembly_ms")
                ),
                "frame_preparation_ms": _stats(_finite(rows, "preparation_time_ms")),
                "rasterization_ms": raster_stats,
                "selected_hierarchy_rows": _stats(
                    _finite(rows, "alod_selected_hierarchy_rows")
                ),
                "total_raster_rows": _stats(_finite(rows, "alod_total_raster_rows")),
                "visible_gaussians": _stats(_finite(rows, "visible_gaussians")),
                "candidate_spt_count": _stats(
                    _finite(rows, "alod_candidate_spt_count")
                ),
                "steady_torch_allocated_bytes": _stats(
                    _finite(rows, "torch_current_allocated_bytes")
                ),
                "peak_torch_allocated_bytes": max(
                    _finite(rows, "torch_peak_allocated_bytes")
                ),
                "peak_torch_reserved_bytes": max(
                    _finite(rows, "torch_peak_reserved_bytes")
                ),
                "peak_whole_device_vram_bytes": max(
                    _finite(rows, "nvml_device_used_bytes")
                ),
                "peak_process_rss_bytes": max(_finite(rows, "process_ram_bytes")),
                "gaussian_properties_gpu_bytes": int(
                    rows[0]["alod_gaussian_properties_gpu_bytes"]
                ),
                "hierarchy_metadata_cpu_bytes": int(
                    rows[0]["alod_hierarchy_metadata_cpu_bytes"]
                ),
                "hierarchy_metadata_gpu_bytes": int(
                    rows[0]["alod_hierarchy_metadata_gpu_bytes"]
                ),
                "selected_index_bytes": _stats(
                    _finite(rows, "alod_selected_index_bytes")
                ),
                "startup": startup,
                "selection_characterization": semantics["configurations"][config_id][
                    "summary"
                ],
            }
        )

    summary_by_id = {row["configuration_id"]: row for row in config_summaries}
    transitions = []
    for parent_id, child_id, label in (
        (CONFIG_IDS[0], CONFIG_IDS[1], "finest_to_lod"),
        (CONFIG_IDS[1], CONFIG_IDS[2], "lod_to_lod_frustum"),
    ):
        parent = summary_by_id[parent_id]
        child = summary_by_id[child_id]
        parent_total = float(parent["total_frame_time_ms"]["mean"])
        child_total = float(child["total_frame_time_ms"]["mean"])
        parent_raster = float(parent["rasterization_ms"]["mean"])
        child_raster = float(child["rasterization_ms"]["mean"])
        parent_selected = float(parent["selected_hierarchy_rows"]["mean"])
        child_selected = float(child["selected_hierarchy_rows"]["mean"])
        transitions.append(
            {
                "transition": label,
                "parent_configuration_id": parent_id,
                "child_configuration_id": child_id,
                "selected_hierarchy_rows_removed_mean": parent_selected - child_selected,
                "selected_hierarchy_rows_change_percent": _percent_change(
                    child_selected, parent_selected
                ),
                "rasterization_time_saved_ms": parent_raster - child_raster,
                "rasterization_time_change_percent": _percent_change(
                    child_raster, parent_raster
                ),
                "total_frame_time_saved_ms": parent_total - child_total,
                "total_frame_time_change_percent": _percent_change(child_total, parent_total),
                "parent_preparation_ms": parent["frame_preparation_ms"]["mean"],
                "child_preparation_ms": child["frame_preparation_ms"]["mean"],
                "parent_selection_ms": parent["total_selection_ms"]["mean"],
                "child_selection_ms": child["total_selection_ms"]["mean"],
                "selection_time_change_ms": child["total_selection_ms"]["mean"]
                - parent["total_selection_ms"]["mean"],
                "parent_selected_index_ms": parent["selected_index_ms"]["mean"],
                "child_selected_index_ms": child["selected_index_ms"]["mean"],
                "parent_gather_ms": parent["selected_gather_ms"]["mean"],
                "child_gather_ms": child["selected_gather_ms"]["mean"],
                "parent_pre_raster_assembly_ms": parent["pre_raster_assembly_ms"]["mean"],
                "child_pre_raster_assembly_ms": child["pre_raster_assembly_ms"]["mean"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "campaign_id": "ALOD-RESIDENT-BENCHMARK-001",
        "status": "PASS-ALOD-RESIDENT-BENCHMARK",
        "interpretation": "resident representation baseline, not final memory-scalable runtime",
        "weak_lod_oracle": True,
        "asset": {"path": str(asset.resolve()), "sha256": sha256_file(asset)},
        "oracle": {
            "path": str(oracle_manifest.resolve()),
            "manifest_file_sha256": sha256_file(oracle_manifest),
            "semantic_manifest_sha256": oracle["deterministic_semantic_manifest"]["sha256"],
        },
        "trajectory": semantics["trajectory"],
        "configurations": config_summaries,
        "controlled_transitions": transitions,
        "quality": quality,
        "environment": environment_metadata(),
        "configuration_hashes": {
            config_id: sha256_file(CONFIG_FILES[config_id]) for config_id in CONFIG_IDS
        },
    }
    write_json(output_dir / "config_summary.json", result)
    write_json(output_dir / "controlled_transitions.json", {"rows": transitions})

    csv_fields = (
        "configuration_id",
        "mean_total_frame_time_ms",
        "median_total_frame_time_ms",
        "p90_total_frame_time_ms",
        "p95_total_frame_time_ms",
        "fps",
        "mean_selection_ms",
        "mean_gather_ms",
        "mean_raster_ms",
        "mean_selected_hierarchy_rows",
        "mean_total_raster_rows",
        "mean_visible_gaussians",
        "peak_torch_allocated_bytes",
        "peak_process_rss_bytes",
    )
    with (output_dir / "config_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in config_summaries:
            writer.writerow(
                {
                    "configuration_id": row["configuration_id"],
                    "mean_total_frame_time_ms": row["total_frame_time_ms"]["mean"],
                    "median_total_frame_time_ms": row["total_frame_time_ms"]["median"],
                    "p90_total_frame_time_ms": row["total_frame_time_ms"]["p90"],
                    "p95_total_frame_time_ms": row["total_frame_time_ms"]["p95"],
                    "fps": row["fps_from_mean_total_frame_time"],
                    "mean_selection_ms": row["total_selection_ms"]["mean"],
                    "mean_gather_ms": row["selected_gather_ms"]["mean"],
                    "mean_raster_ms": row["rasterization_ms"]["mean"],
                    "mean_selected_hierarchy_rows": row["selected_hierarchy_rows"]["mean"],
                    "mean_total_raster_rows": row["total_raster_rows"]["mean"],
                    "mean_visible_gaussians": row["visible_gaussians"]["mean"],
                    "peak_torch_allocated_bytes": row["peak_torch_allocated_bytes"],
                    "peak_process_rss_bytes": row["peak_process_rss_bytes"],
                }
            )

    labels = [row["configuration_id"].replace("ALOD-HIER-", "") for row in config_summaries]
    table_lines = [
        "# ALoD Resident Benchmark Tables",
        "",
        "| Configuration | Mean total ms | Median ms | P95 ms | FPS | Selection ms | Gather ms | Raster ms | Selected hierarchy | Raster rows | Visible | Peak allocated MiB | Peak RSS MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in config_summaries:
        table_lines.append(
            "| {configuration_id} | {mean:.3f} | {median:.3f} | {p95:.3f} | {fps:.2f} | "
            "{selection:.3f} | {gather:.3f} | {raster:.3f} | {selected:,.0f} | "
            "{raster_rows:,.0f} | {visible:,.0f} | {peak:.1f} | {rss:.1f} |".format(
                configuration_id=row["configuration_id"],
                mean=row["total_frame_time_ms"]["mean"],
                median=row["total_frame_time_ms"]["median"],
                p95=row["total_frame_time_ms"]["p95"],
                fps=row["fps_from_mean_total_frame_time"],
                selection=row["total_selection_ms"]["mean"],
                gather=row["selected_gather_ms"]["mean"],
                raster=row["rasterization_ms"]["mean"],
                selected=row["selected_hierarchy_rows"]["mean"],
                raster_rows=row["total_raster_rows"]["mean"],
                visible=row["visible_gaussians"]["mean"],
                peak=row["peak_torch_allocated_bytes"] / (1024**2),
                rss=row["peak_process_rss_bytes"] / (1024**2),
            )
        )
    table_lines.extend(
        [
            "",
            "Quality uses the same-asset finest valid cut as a reference, not photography.",
            "",
            "| Comparison | Min PSNR dB | Mean finite PSNR dB | Mean SSIM | Mean LPIPS | Mean absolute error | Max absolute error |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, values in quality["comparisons"].items():
        table_lines.append(
            f"| {key} | {values['minimum_psnr_db']:.3f} | "
            f"{values['mean_finite_psnr_db']:.3f} | {values['mean_ssim']:.8f} | "
            f"{values['mean_lpips_alex']:.8g} | {values['mean_absolute_error']:.8g} | "
            f"{values['maximum_absolute_error']:.8g} |"
        )
    (output_dir / "thesis_tables.md").write_text(
        "\n".join(table_lines) + "\n", encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt

        figures = campaign_root / "figures"
        figures.mkdir(parents=True, exist_ok=True)

        means = [row["total_frame_time_ms"]["mean"] for row in config_summaries]
        p95 = [row["total_frame_time_ms"]["p95"] for row in config_summaries]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, means, color=["#4C78A8", "#F58518", "#54A24B"])
        ax.scatter(range(len(labels)), p95, color="black", label="P95", zorder=3)
        ax.set_ylabel("End-to-end frame time (ms)")
        ax.set_title("Resident ALoD frame time")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "frame_time_by_configuration.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(
            labels,
            [row["selected_hierarchy_rows"]["mean"] / 1e3 for row in config_summaries],
            color=["#4C78A8", "#F58518", "#54A24B"],
        )
        ax.set_ylabel("Mean selected hierarchy rows (thousands)")
        ax.set_title("Resident ALoD selected workload")
        fig.tight_layout()
        fig.savefig(figures / "selected_rows_by_configuration.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        stage_names = (
            ("total_selection_ms", "Selection"),
            ("selected_index_ms", "Index"),
            ("selected_gather_ms", "Gather"),
            ("pre_raster_assembly_ms", "Assembly"),
            ("rasterization_ms", "Raster"),
        )
        bottom = np.zeros(len(config_summaries))
        for key, label in stage_names:
            values = np.asarray([row[key]["mean"] for row in config_summaries])
            ax.bar(labels, values, bottom=bottom, label=label)
            bottom += values
        ax.set_ylabel("Mean time (ms)")
        ax.set_title("Measured ALoD stage breakdown")
        ax.legend(ncol=3)
        fig.tight_layout()
        fig.savefig(figures / "stage_breakdown.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        (output_dir / "figure_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )

    manifest = {
        "schema_version": 1,
        "campaign_id": result["campaign_id"],
        "status": result["status"],
        "asset": result["asset"],
        "oracle": result["oracle"],
        "trajectory": result["trajectory"],
        "configuration_hashes": result["configuration_hashes"],
        "environment": result["environment"],
        "run_policy": {
            "process_replicates": 3,
            "warmup_frames_per_process": 10,
            "measured_complete_trajectory_repetitions_per_process": 10,
            "measured_frames_per_process": 100,
            "measured_frames_per_configuration": 300,
            "rejected_runs": [],
        },
    }
    write_json(campaign_root / "manifest.json", manifest)
    (campaign_root / "README.md").write_text(
        "# ALoD Resident Benchmark 001\n\n"
        "Formal resident-only finest/LoD/frustum ablation. All Gaussian properties "
        "remain GPU resident; this is not an out-of-core result.\n\n"
        "See `tables/config_summary.json`, `quality/quality_summary.json`, and "
        "`checks/ablation_semantics.json`.\n",
        encoding="utf-8",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    semantics = subparsers.add_parser("semantics")
    semantics.add_argument("--asset", type=Path, required=True)
    semantics.add_argument("--output", type=Path, required=True)

    quality = subparsers.add_parser("quality")
    quality.add_argument("--asset", type=Path, required=True)
    quality.add_argument("--output-dir", type=Path, required=True)
    quality.add_argument("--device-index", type=int, default=0)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--campaign-root", type=Path, required=True)
    aggregate.add_argument("--asset", type=Path, required=True)
    aggregate.add_argument("--oracle-manifest", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "semantics":
        return run_semantics(args.asset, args.output)
    if args.command == "quality":
        return run_quality(args.asset, args.output_dir, args.device_index)
    if args.command == "aggregate":
        return run_aggregate(
            args.campaign_root, args.asset, args.oracle_manifest, args.output_dir
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
