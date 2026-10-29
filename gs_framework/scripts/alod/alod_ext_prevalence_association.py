#!/usr/bin/env python3
"""Execute the frozen natural-hierarchy ALoD-EXT association study.

The script only orchestrates the established selector, conservative CUDA
filter, residency policies, renderer, and benchmark instrumentation.  It does
not alter their semantics.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_selection_oracle import load_manifest, sha256_file, validate_oracle_tree  # noqa: E402
from benchmarks.alod_resident_benchmark import _runtime_config  # noqa: E402
from benchmarks import benchmark_real_scene as benchmark_module  # noqa: E402
from benchmarks.benchmark_real_scene import (  # noqa: E402
    CameraPath,
    CameraSpec,
    apply_camera_config,
    environment_metadata,
    load_camera_path,
    load_config,
    make_render_settings,
    run_benchmark,
)
from scene_runtime.alod import (  # noqa: E402
    AlodHostResidencyConfig,
    conservative_pinhole_materialization,
)
from scene_runtime.alod.host_scene import AlodHostScene  # noqa: E402
from scene_runtime.renderer import GsplatRenderer  # noqa: E402
from scene_runtime.types import FrameCamera  # noqa: E402
from scripts.alod.validate_fast_materialization_filter import _projection_ids  # noqa: E402
from scripts.alod.validate_production_resident import _json, _payload  # noqa: E402


CAMPAIGN_ID = "ALOD-EXT-PREVALENCE-ASSOCIATION-001"
PASS_CORRECTNESS = "PASS-ALOD-EXT-PREVALENCE-CORRECTNESS"
PASS_RESULTS = "PASS-ALOD-EXT-PREVALENCE-ASSOCIATION"
DEFAULT_CONFIG = REPO_ROOT / "configs/alod/ALOD-EXT-PREVALENCE-ASSOCIATION-001.json"
MODE_ORDER = ("nocache", "previous", "bounded")


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_study(path: Path) -> dict[str, Any]:
    study = read_json(path.resolve())
    if study.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("association configuration has the wrong campaign ID")
    return study


def dataset_spec(study: dict[str, Any], key: str) -> dict[str, Any]:
    try:
        return study["datasets"][key]
    except KeyError as error:
        raise ValueError(f"unknown dataset {key!r}") from error


def verify_inputs(spec: dict[str, Any]) -> None:
    checks = (
        (Path(spec["asset_path"]), spec["asset_sha256"]),
        (Path(spec["camera_path"]), spec["camera_path_sha256"]),
        (Path(spec["oracle_manifest"]), spec["oracle_manifest_sha256"]),
    )
    for path, expected in checks:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen input hash mismatch for {path}: {actual} != {expected}")


def build_config(spec: dict[str, Any], mode: str, capacity: int | None = None) -> dict[str, Any]:
    base_id = {
        "nocache": "ALOD-REPRO-HOST-NOCACHE",
        "previous": "ALOD-EXT-PREMATERIALIZATION-CUDA-PREVIOUS",
        "bounded": "ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL",
    }[mode]
    config = deepcopy(load_config(base_id))
    config["configuration_id"] = f"{spec['execution_id']}-{mode.upper()}"
    config["description"] = {
        "nocache": "Released Evaluator (No Cache) under the frozen natural-dataset workload.",
        "previous": "Established CUDA conservative filter plus exact previous-working-set property reuse.",
        "bounded": "Established CUDA conservative filter plus union-covering deterministic row-LRU cache.",
    }[mode]
    config["asset_role"] = spec["dataset_id"]
    config["camera_path"] = spec["camera_path"]
    config["trajectory_id"] = spec["trajectory_id"]
    config["trajectory_notes"] = "Complete frozen held-out split in recorded canonical order."
    config["camera_resolution_scale"] = float(spec["camera_resolution_scale"])
    config["width"] = int(spec["width"])
    config["height"] = int(spec["height"])
    config["warmup_frames"] = 10
    config["measured_frames"] = int(spec["camera_count"])
    config["num_repeat"] = 1
    config["record_warmup_frames"] = True
    config["alod"].update(
        {
            "root_global_node_id": 100000,
            "skybox_count": 100000,
            "spt_root_volume": float(spec["spt_root_volume"]),
            "min_spt_size": int(spec["min_spt_size"]),
            "target_granularity": float(spec["target_granularity"]),
            "distance_multiplier": 1.0,
            "selection_mode": "lod",
            "frustum_enabled": True,
        }
    )
    if mode == "bounded":
        if capacity is None:
            raise ValueError("bounded mode requires the precomputed capacity")
        config["alod_host"]["cache_capacity_rows"] = int(capacity)
    return config


def configured_cameras(spec: dict[str, Any], config: dict[str, Any]):
    path = Path(spec["camera_path"])
    configured = apply_camera_config(load_camera_path(path, spec["trajectory_id"]), config)
    manifest_path = Path(spec["oracle_manifest"])
    manifest = load_manifest(manifest_path)
    if len(configured.cameras) != int(spec["camera_count"]):
        raise RuntimeError("configured camera count differs from the frozen workload")
    if len(manifest["cameras"]) != len(configured.cameras):
        raise RuntimeError("official oracle and frozen camera workload differ")

    exact = []
    root = manifest_path.parent
    for index, (source, entry) in enumerate(zip(configured.cameras, manifest["cameras"])):
        if int(entry["evaluation_sequence_index"]) != index:
            raise RuntimeError("official oracle order differs from the frozen camera order")
        record = _json(root / entry["paths"]["camera.json"])
        runtime = record["effective_official_runtime_camera"]
        world_view = np.asarray(runtime["world_view_transform"]["values"], dtype=np.float32)
        w2c = np.ascontiguousarray(world_view.T, dtype=np.float32)
        c2w = np.linalg.inv(w2c.astype(np.float64)).astype(np.float32)
        intrinsics = runtime["intrinsics"]
        K = np.asarray(
            [
                [intrinsics["fx"], 0.0, intrinsics["cx"]],
                [0.0, intrinsics["fy"], intrinsics["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        width = int(runtime["image_width"])
        height = int(runtime["image_height"])
        if width != source.width or height != source.height:
            raise RuntimeError("official runtime resolution differs from the frozen workload")
        camera = CameraSpec(
            camera_id=entry["camera_stable_id"],
            c2w=c2w,
            K=K,
            width=width,
            height=height,
            w2c=w2c,
            camera_center=np.asarray(runtime["camera_center_float32"], dtype=np.float32),
        )
        object.__setattr__(
            camera,
            "_selection_full_projection_transform",
            np.asarray(runtime["full_projection_transform"]["values"], dtype=np.float32),
        )
        exact.append(camera)
    metadata = dict(configured.metadata)
    metadata["exact_official_selector_camera_values"] = True
    metadata["official_oracle_manifest"] = str(manifest_path)
    return CameraPath(
        path=configured.path,
        trajectory_id=configured.trajectory_id,
        cameras=exact,
        metadata=metadata,
    )


def make_exact_frame_camera(camera: CameraSpec, device: torch.device) -> FrameCamera:
    """Preserve the exact official float32 selector camera without changing shared code."""
    c2w = torch.from_numpy(np.asarray(camera.c2w, dtype=np.float32)).to(device)
    K = torch.from_numpy(np.asarray(camera.K, dtype=np.float32)).to(device)
    w2c = (
        torch.from_numpy(np.asarray(camera.w2c, dtype=np.float32)).to(device)
        if camera.w2c is not None
        else torch.linalg.inv(c2w)
    )
    center = (
        torch.from_numpy(np.asarray(camera.camera_center, dtype=np.float32)).to(device)
        if camera.camera_center is not None
        else None
    )
    full_projection = getattr(camera, "_selection_full_projection_transform", None)
    return FrameCamera(
        c2w=c2w,
        K=K,
        width=int(camera.width),
        height=int(camera.height),
        w2c=w2c,
        selection_camera_center=center,
        selection_full_projection_transform=(
            torch.from_numpy(np.asarray(full_projection, dtype=np.float32)).to(device)
            if full_projection is not None
            else None
        ),
        camera_stable_id=camera.camera_id,
    )


def host_config(mode: str, capacity: int | None = None, *, properties: bool = False):
    return AlodHostResidencyConfig(
        mode={"nocache": "host_nocache", "previous": "host_previous_reuse", "bounded": "host_bounded_cache"}[mode],
        cache_capacity_rows=capacity,
        activation_chunk_rows=131072,
        pin_memory=False,
        validate_mappings=True,
        validate_properties=properties,
    )


def tensor_diff(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    if actual.numel() == 0:
        maximum = mean = 0.0
    else:
        delta = torch.abs(actual.float() - expected.float())
        maximum = float(delta.max().item())
        mean = float(delta.mean().item())
    return {
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "maximum_absolute_error": maximum,
        "mean_absolute_error": mean,
    }


def capacity_from_union(study: dict[str, Any], spec: dict[str, Any], union_rows: int) -> int:
    rule = study["capacity_rule"]
    numerator = int(rule["run009_practical_capacity_rows"])
    denominator = int(rule["run009_union_rows"])
    proposed = math.ceil(union_rows * numerator / denominator)
    return min(int(spec["hierarchy_rows"]), proposed)


@torch.no_grad()
def command_correctness(args: argparse.Namespace) -> int:
    study = load_study(args.config)
    spec = dataset_spec(study, args.dataset)
    verify_inputs(spec)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("association correctness requires CUDA")

    faithful_config = build_config(spec, "nocache")
    fast_config = build_config(spec, "previous")
    faithful_runtime = _runtime_config(faithful_config)
    fast_runtime = _runtime_config(fast_config)
    if faithful_runtime.__dict__ | {"materialization_filter": "none"} != (
        fast_runtime.__dict__ | {"materialization_filter": "none"}
    ):
        raise RuntimeError("extension configuration changed semantic selection")

    manifest_path = Path(spec["oracle_manifest"])
    manifest = load_manifest(manifest_path)
    validate_oracle_tree(manifest_path)
    cameras = configured_cameras(spec, faithful_config)
    if len(manifest["cameras"]) != len(cameras.cameras):
        raise RuntimeError("official oracle and frozen camera workload differ")

    asset = Path(spec["asset_path"])
    faithful = AlodHostScene(asset, device, faithful_runtime, host_config("nocache"))
    fast = AlodHostScene(asset, device, fast_runtime, host_config("previous", properties=True))
    renderer = GsplatRenderer(device)
    root = manifest_path.parent
    rows: list[dict[str, Any]] = []
    trace: dict[str, np.ndarray] = {}
    union = np.empty(0, dtype=np.int64)
    for index, (camera_spec, oracle_entry) in enumerate(zip(cameras.cameras, manifest["cameras"])):
        if int(oracle_entry["evaluation_sequence_index"]) != index:
            raise RuntimeError("official oracle order differs from the frozen camera order")
        camera = make_exact_frame_camera(camera_spec, device)
        settings = make_render_settings(faithful_config, camera_spec, sh_degree=1)
        faithful_active = faithful.prepare_frame(camera, settings)
        faithful_render = renderer.render(faithful_active, camera, settings)
        fast_active = fast.prepare_frame(camera, settings)
        fast_render = renderer.render(fast_active, camera, settings)
        faithful_selection = faithful.last_selection
        fast_selection = fast.last_selection
        materialization = fast.last_materialization
        assert faithful_selection is not None and fast_selection is not None and materialization is not None

        camera_dir = root / Path(oracle_entry["paths"]["camera.json"]).parent
        working = _json(camera_dir / "working-set.json")
        oracle_raster = _payload(camera_dir, working["global_node_id_payload"])
        np.testing.assert_array_equal(faithful_selection.raster_global_node_ids, oracle_raster)
        np.testing.assert_array_equal(fast_selection.raster_global_node_ids, oracle_raster)
        np.testing.assert_array_equal(
            faithful_selection.selected_hierarchy_global_node_ids,
            oracle_raster[faithful_runtime.skybox_count :],
        )

        cpu = conservative_pinhole_materialization(
            fast.compact_culling, oracle_raster, camera, settings
        )
        np.testing.assert_array_equal(
            materialization.materialization_global_node_ids,
            cpu.materialization_global_node_ids,
        )
        np.testing.assert_array_equal(materialization.kept_semantic_indices, cpu.kept_semantic_indices)
        kept = torch.from_numpy(cpu.kept_semantic_indices).to(device=device)
        property_fields: dict[str, bool] = {}
        for field in ("means", "quats", "scales", "opacities", "colors", "global_node_ids"):
            expected = getattr(faithful_active, field).index_select(0, kept)
            actual = getattr(fast_active, field)
            property_fields[field] = bool(torch.equal(actual, expected))
            if not property_fields[field]:
                raise RuntimeError(f"{field} property mismatch at camera {index}")

        faithful_backend = _projection_ids(faithful_render)
        fast_backend = _projection_ids(fast_render)
        np.testing.assert_array_equal(fast_backend, faithful_backend)
        false_negatives = np.setdiff1d(
            faithful_backend, materialization.materialization_global_node_ids, assume_unique=False
        )
        if false_negatives.size:
            raise RuntimeError(f"{false_negatives.size} filter false negatives at camera {index}")
        rgb = tensor_diff(fast_render.image, faithful_render.image)
        alpha = tensor_diff(fast_render.alpha, faithful_render.alpha)
        if not rgb["bitwise_equal"] or not alpha["bitwise_equal"]:
            raise RuntimeError(f"render mismatch at camera {index}")

        semantic_hierarchy = oracle_raster[oracle_raster >= faithful_runtime.skybox_count]
        materialized_hierarchy = cpu.materialization_global_node_ids[
            cpu.materialization_global_node_ids >= faithful_runtime.skybox_count
        ]
        union = np.union1d(union, materialized_hierarchy)
        trace[f"frame_{index:03d}_semantic_hierarchy_ids"] = semantic_hierarchy
        trace[f"frame_{index:03d}_materialized_hierarchy_ids"] = materialized_hierarchy
        trace[f"frame_{index:03d}_packed_survivor_ids"] = faithful_backend
        rows.append(
            {
                "camera_index": index,
                "camera_id": camera_spec.camera_id,
                "camera_stable_id": oracle_entry["camera_stable_id"],
                "semantic_rows": int(oracle_raster.size),
                "semantic_hierarchy_rows": int(semantic_hierarchy.size),
                "materialized_rows": int(cpu.materialization_global_node_ids.size),
                "materialized_hierarchy_rows": int(materialized_hierarchy.size),
                "packed_projection_survivors": int(faithful_backend.size),
                "official_semantic_ids_exact": True,
                "cpu_cuda_ordered_filter_ids_exact": True,
                "false_negative_count": 0,
                "properties_bitwise_exact": property_fields,
                "packed_survivor_ids_exact": True,
                "rgb": rgb,
                "alpha": alpha,
            }
        )
        print(f"[{index + 1}/{len(cameras.cameras)}] {camera_spec.camera_id}: PASS", flush=True)

    capacity = capacity_from_union(study, spec, int(union.size))
    maximum_request = max(row["materialized_hierarchy_rows"] for row in rows)
    if capacity < maximum_request:
        raise RuntimeError("frozen capacity rule cannot hold the maximum frame request")
    result = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "execution_id": spec["execution_id"],
        "status": PASS_CORRECTNESS,
        "dataset": args.dataset,
        "stage_a_label": spec["stage_a_label"],
        "asset": {"path": str(asset), "sha256": sha256_file(asset)},
        "camera_path": {"path": spec["camera_path"], "sha256": sha256_file(Path(spec["camera_path"]))},
        "oracle_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "camera_count": len(rows),
        "criteria": {
            "official_semantic_ids_and_order_exact": True,
            "cpu_cuda_filter_ids_and_order_exact": True,
            "conservative_filter_false_negatives": 0,
            "retained_properties_and_order_bitwise_exact": True,
            "packed_projection_survivor_ids_and_order_exact": True,
            "rgb_bitwise_exact": True,
            "alpha_bitwise_exact": True,
        },
        "capacity": {
            "filtered_canonical_union_rows": int(union.size),
            "maximum_single_frame_hierarchy_rows": maximum_request,
            "hierarchy_rows": int(spec["hierarchy_rows"]),
            "bounded_capacity_rows": capacity,
            "formula": study["capacity_rule"]["formula"],
            "chosen_before_timing": True,
        },
        "cameras": rows,
    }
    write_json(args.output, result)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.trace, **trace)
    del faithful, fast, renderer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    study = load_study(args.config)
    spec = dataset_spec(study, args.dataset)
    verify_inputs(spec)
    config = build_config(spec, args.mode, args.capacity)
    cameras = configured_cameras(spec, config)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("association benchmark requires CUDA")
    asset = Path(spec["asset_path"])
    benchmark_module.make_frame_camera = make_exact_frame_camera
    return run_benchmark(
        config=config,
        asset_path=asset,
        asset_supplied=str(asset),
        camera_path=cameras,
        output_dir=args.output_dir.resolve(),
        record_kind="formal_benchmark",
        max_frames=len(cameras.cameras),
        device=device,
        log_gpu_events=True,
    )


def command_campaign(args: argparse.Namespace) -> int:
    study = load_study(args.config)
    spec = dataset_spec(study, args.dataset)
    correctness = read_json(args.correctness)
    if correctness.get("status") != PASS_CORRECTNESS:
        raise RuntimeError("required correctness gate has not passed")
    if correctness.get("execution_id") != spec["execution_id"]:
        raise RuntimeError("correctness gate belongs to another execution")
    capacity = int(correctness["capacity"]["bounded_capacity_rows"])
    root = args.campaign_root.resolve()
    script = Path(__file__).resolve()
    rows = []
    for replicate in range(1, int(study["replicates"]) + 1):
        rotated = MODE_ORDER[replicate - 1 :] + MODE_ORDER[: replicate - 1]
        for mode in rotated:
            output = root / "runs" / spec["execution_id"] / mode / f"replicate-{replicate:02d}"
            command = [
                sys.executable,
                str(script),
                "benchmark",
                "--config",
                str(args.config.resolve()),
                "--dataset",
                args.dataset,
                "--mode",
                mode,
                "--output-dir",
                str(output),
            ]
            if mode == "bounded":
                command += ["--capacity", str(capacity)]
            rows.append(
                {
                    "sequence_index": len(rows) + 1,
                    "execution_id": spec["execution_id"],
                    "dataset": args.dataset,
                    "replicate": replicate,
                    "mode": mode,
                    "capacity_rows": capacity if mode == "bounded" else None,
                    "output_dir": str(output),
                    "command": command,
                    "status": "planned",
                    "started_utc": None,
                    "finished_utc": None,
                }
            )
    schedule_path = root / "checks" / f"{args.dataset}_schedule.json"
    write_json(schedule_path, {"campaign_id": CAMPAIGN_ID, "schedule": rows})
    for row in rows:
        summary = Path(row["output_dir"]) / "summary.json"
        if summary.exists():
            if not args.resume:
                raise FileExistsError(summary)
            row["status"] = "resumed_complete"
            continue
        row["status"] = "running"
        row["started_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(schedule_path, {"campaign_id": CAMPAIGN_ID, "schedule": rows})
        print(
            f"[{row['sequence_index']}/{len(rows)}] {row['execution_id']} "
            f"{row['mode']} replicate {row['replicate']}",
            flush=True,
        )
        completed = subprocess.run(row["command"], cwd=REPO_ROOT)
        row["finished_utc"] = datetime.now(timezone.utc).isoformat()
        row["status"] = "complete" if completed.returncode == 0 else "failed"
        write_json(schedule_path, {"campaign_id": CAMPAIGN_ID, "schedule": rows})
        if completed.returncode != 0:
            raise RuntimeError(f"benchmark failed: {row['mode']} replicate {row['replicate']}")
    return 0


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    raw = values(rows, key)
    if not raw:
        return {"minimum": None, "mean": None, "median": None, "p95": None, "maximum": None}
    array = np.asarray(raw, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def summarize_mode(run_dirs: list[Path]) -> dict[str, Any]:
    measured_by_rep = [read_jsonl(path / "frames.jsonl") for path in run_dirs]
    warmup_by_rep = [read_jsonl(path / "warmup_frames.jsonl") for path in run_dirs]
    measured = [row for rows in measured_by_rep for row in rows]
    warmup = [row for rows in warmup_by_rep for row in rows]
    time_keys = (
        "end_to_end_frame_time_ms",
        "preparation_time_ms",
        "alod_total_selection_ms",
        "alod_materialization_filter_ms",
        "alod_residency_lookup_ms",
        "alod_host_gather_ms",
        "alod_h2d_transfer_ms",
        "alod_selected_gpu_gather_ms",
        "alod_cache_management_ms",
        "alod_pre_raster_assembly_ms",
        "rasterization_time_ms",
    )
    count_keys = (
        "alod_semantic_selected_rows",
        "alod_materialization_rows",
        "backend_projection_survivor_count",
        "alod_h2d_rows",
        "alod_h2d_bytes",
        "alod_cache_hit_count",
        "alod_cache_miss_count",
        "alod_cache_eviction_count",
    )
    mean_total = float(np.mean(values(measured, "end_to_end_frame_time_ms")))
    mean_selection = float(np.mean(values(measured, "alod_total_selection_ms")))
    return {
        "replicate_count": len(run_dirs),
        "measured_frames": len(measured),
        "warmup_frames": len(warmup),
        "timings_ms": {key: distribution(measured, key) for key in time_keys},
        "counts": {key: distribution(measured, key) for key in count_keys},
        "fps_from_mean_end_to_end": 1000.0 / mean_total,
        "semantic_selection_share_of_total": mean_selection / mean_total,
        "steady_dynamic_property_h2d_rows": int(sum(values(measured, "alod_h2d_rows"))),
        "steady_dynamic_property_h2d_bytes": int(sum(values(measured, "alod_h2d_bytes"))),
        "warmup_dynamic_property_h2d_rows": int(sum(values(warmup, "alod_h2d_rows"))),
        "warmup_dynamic_property_h2d_bytes": int(sum(values(warmup, "alod_h2d_bytes"))),
        "cold_inclusive_dynamic_property_h2d_rows": int(
            sum(values(measured + warmup, "alod_h2d_rows"))
        ),
        "cold_inclusive_dynamic_property_h2d_bytes": int(
            sum(values(measured + warmup, "alod_h2d_bytes"))
        ),
        "replicates": [
            {
                "run_dir": str(path),
                "mean_end_to_end_frame_time_ms": float(
                    np.mean(values(rows, "end_to_end_frame_time_ms"))
                ),
            }
            for path, rows in zip(run_dirs, measured_by_rep)
        ],
    }


def command_aggregate(args: argparse.Namespace) -> int:
    study = load_study(args.config)
    root = args.campaign_root.resolve()
    dataset_results: dict[str, Any] = {}
    for key, spec in study["datasets"].items():
        correctness_path = root / "checks" / f"{key}_correctness.json"
        correctness = read_json(correctness_path)
        if correctness.get("status") != PASS_CORRECTNESS:
            raise RuntimeError(f"{key} correctness gate did not pass")
        schedule = read_json(root / "checks" / f"{key}_schedule.json")["schedule"]
        if any(row["status"] not in {"complete", "resumed_complete"} for row in schedule):
            raise RuntimeError(f"{key} benchmark schedule is incomplete")
        modes = {}
        for mode in MODE_ORDER:
            paths = [Path(row["output_dir"]) for row in schedule if row["mode"] == mode]
            modes[mode] = summarize_mode(paths)
        baseline_ms = modes["nocache"]["timings_ms"]["end_to_end_frame_time_ms"]["mean"]
        baseline_steady = modes["nocache"]["steady_dynamic_property_h2d_bytes"]
        baseline_cold = modes["nocache"]["cold_inclusive_dynamic_property_h2d_bytes"]
        for mode in ("previous", "bounded"):
            mode_ms = modes[mode]["timings_ms"]["end_to_end_frame_time_ms"]["mean"]
            modes[mode]["relative_to_released_evaluator"] = {
                "frame_time_ratio": mode_ms / baseline_ms,
                "frame_time_reduction_fraction": 1.0 - mode_ms / baseline_ms,
                "speedup": baseline_ms / mode_ms,
                "steady_h2d_byte_reduction_fraction": 1.0
                - modes[mode]["steady_dynamic_property_h2d_bytes"] / baseline_steady,
                "cold_inclusive_h2d_byte_reduction_fraction": 1.0
                - modes[mode]["cold_inclusive_dynamic_property_h2d_bytes"] / baseline_cold,
            }
        camera_rows = correctness["cameras"]
        semantic = np.asarray([row["semantic_rows"] for row in camera_rows], dtype=np.float64)
        materialized = np.asarray([row["materialized_rows"] for row in camera_rows], dtype=np.float64)
        survivors = np.asarray([row["packed_projection_survivors"] for row in camera_rows], dtype=np.float64)
        dataset_results[key] = {
            "execution_id": spec["execution_id"],
            "dataset_id": spec["dataset_id"],
            "stage_a_label": spec["stage_a_label"],
            "camera_count": int(spec["camera_count"]),
            "correctness_status": correctness["status"],
            "bounded_capacity_rows": correctness["capacity"]["bounded_capacity_rows"],
            "filtered_trajectory_union_rows": correctness["capacity"]["filtered_canonical_union_rows"],
            "workload_funnel": {
                "semantic_rows": distribution(camera_rows, "semantic_rows"),
                "materialized_rows": distribution(camera_rows, "materialized_rows"),
                "packed_projection_survivors": distribution(camera_rows, "packed_projection_survivors"),
                "pooled_materialized_over_semantic": float(materialized.sum() / semantic.sum()),
                "pooled_survivors_over_semantic": float(survivors.sum() / semantic.sum()),
                "mean_per_camera_materialized_over_semantic": float(np.mean(materialized / semantic)),
                "mean_per_camera_survivors_over_semantic": float(np.mean(survivors / semantic)),
            },
            "modes": modes,
        }

    run009_path = WORKSPACE_ROOT / "benchmark_artifacts/alod_ext_formal_run009_001/tables/formal_summary.json"
    run009 = read_json(run009_path)
    historical = {
        "classification": "historical VALID-WEAK-LOD reference",
        "authority": str(run009_path),
        "methodology_compatibility": "same packed backend, ten warmups, three fresh processes, synchronized end-to-end timing; different asset and 71-camera workload",
        "released_evaluator_mean_ms": run009["faithful_context"]["ALOD-REPRO-HOST-NOCACHE"]["timings_ms"]["end_to_end_frame_time_ms"]["mean"],
        "previous_mean_ms": run009["extension_modes"]["ALOD-EXT-PREMATERIALIZATION-CUDA-PREVIOUS"]["timings_ms"]["end_to_end_frame_time_ms"]["mean"],
        "bounded_mean_ms": run009["extension_modes"]["ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL"]["timings_ms"]["end_to_end_frame_time_ms"]["mean"],
        "mean_semantic_rows": run009["working_sets"]["semantic"]["rows"]["mean"] + 100000.0,
        "mean_materialized_rows": run009["extension_modes"]["ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL"]["counts"]["alod_materialization_rows"]["mean"],
        "mean_packed_survivors": run009["faithful_context"]["ALOD-REPRO-HOST-NOCACHE"]["counts"]["backend_projection_survivor_count"]["mean"],
    }
    historical["materialized_over_semantic"] = historical["mean_materialized_rows"] / historical["mean_semantic_rows"]
    historical["survivors_over_semantic"] = historical["mean_packed_survivors"] / historical["mean_semantic_rows"]

    output = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": PASS_RESULTS,
        "classification": study["classification"],
        "environment": environment_metadata(),
        "protocol": {
            "replicates": int(study["replicates"]),
            "warmup_frames": int(study["warmup_frames"]),
            "measured_sequence_restarts_after_warmup": True,
            "cache_state_survives_warmup": True,
            "canonical_complete_held_out_trajectory": True,
            "property_bytes_per_row": int(study["property_bytes_per_row"]),
            "capacity_rule": study["capacity_rule"],
        },
        "datasets": dataset_results,
        "historical_run009": historical,
        "interpretation": "descriptive comparison of two natural trained hierarchies; no causal or population claim",
    }
    write_json(args.output, output)
    return 0


def command_protocol(args: argparse.Namespace) -> int:
    study = load_study(args.config)
    for spec in study["datasets"].values():
        verify_inputs(spec)
    source_paths = [
        Path(__file__).resolve(),
        args.config.resolve(),
        REPO_ROOT / "benchmarks/benchmark_real_scene.py",
        REPO_ROOT / "scene_runtime/alod/selection.py",
        REPO_ROOT / "scene_runtime/alod/materialization.py",
        REPO_ROOT / "scene_runtime/alod/host_scene.py",
        REPO_ROOT / "scene_runtime/alod/residency.py",
    ]
    protocol_note = Path(study["protocol_note"])
    output = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "FROZEN-BEFORE-NEW-EXT-RESULTS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": study["classification"],
        "protocol_note": {"path": str(protocol_note), "sha256": sha256_file(protocol_note)},
        "configuration": {"path": str(args.config.resolve()), "sha256": sha256_file(args.config)},
        "implementation": {str(path): sha256_file(path) for path in source_paths},
        "datasets": study["datasets"],
        "modes": study["modes"],
        "replicates": study["replicates"],
        "warmup_frames": study["warmup_frames"],
        "capacity_rule": study["capacity_rule"],
        "correctness_required_before_timing": True,
        "stage_b_pair_frozen_before_results": True,
        "stage_b_training_authorized": False,
        "synthetic_control_authorized": False,
    }
    write_json(args.output, output)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    protocol = sub.add_parser("protocol", parents=[common])
    protocol.add_argument("--output", type=Path, required=True)
    protocol.set_defaults(func=command_protocol)

    correctness = sub.add_parser("correctness", parents=[common])
    correctness.add_argument("--dataset", choices=("campus", "sciart"), required=True)
    correctness.add_argument("--output", type=Path, required=True)
    correctness.add_argument("--trace", type=Path, required=True)
    correctness.add_argument("--device", default="cuda:0")
    correctness.set_defaults(func=command_correctness)

    benchmark = sub.add_parser("benchmark", parents=[common])
    benchmark.add_argument("--dataset", choices=("campus", "sciart"), required=True)
    benchmark.add_argument("--mode", choices=MODE_ORDER, required=True)
    benchmark.add_argument("--capacity", type=int)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.set_defaults(func=command_benchmark)

    campaign = sub.add_parser("campaign", parents=[common])
    campaign.add_argument("--dataset", choices=("campus", "sciart"), required=True)
    campaign.add_argument("--correctness", type=Path, required=True)
    campaign.add_argument("--campaign-root", type=Path, required=True)
    campaign.add_argument("--resume", action="store_true")
    campaign.set_defaults(func=command_campaign)

    aggregate = sub.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--campaign-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.set_defaults(func=command_aggregate)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
