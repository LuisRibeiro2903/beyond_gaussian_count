#!/usr/bin/env python3
"""Run the frozen ALoD-EXT campaign on the synthetic strong-LoD control."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
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

from benchmarks.alod_resident_benchmark import _runtime_config  # noqa: E402
from benchmarks import benchmark_real_scene as benchmark_module  # noqa: E402
from benchmarks.benchmark_real_scene import (  # noqa: E402
    apply_camera_config,
    environment_metadata,
    load_camera_path,
    load_config,
    make_frame_camera,
    make_render_settings,
    run_benchmark,
)
from scene_runtime.alod import (  # noqa: E402
    AlodHostResidencyConfig,
    conservative_pinhole_materialization,
)
from scene_runtime.alod.host_scene import AlodHostScene  # noqa: E402
from scene_runtime.renderer import GsplatRenderer  # noqa: E402
from scripts.alod.alod_ext_prevalence_association import (  # noqa: E402
    distribution,
    read_json,
    read_jsonl,
    sha256_file,
    summarize_mode,
    tensor_diff,
    values,
    write_json,
)
from scripts.alod.validate_fast_materialization_filter import (  # noqa: E402
    _preload_gsplat_extension,
    _projection_ids,
)


CAMPAIGN_ID = "ALOD-SYNTH-STRONG-LOD-CONTROL-001"
PASS_CORRECTNESS = "PASS-SYNTH-STRONG-LOD-ALOD-EXT-CORRECTNESS"
PASS_CAMPAIGN = "PASS-SYNTH-STRONG-LOD-ALOD-EXT"
DEFAULT_CONFIG = REPO_ROOT / "configs/alod/ALOD-SYNTH-STRONG-LOD-CONTROL-001.json"
MODE_ORDER = ("nocache", "previous", "bounded")


def load_study(path: Path) -> dict[str, Any]:
    value = read_json(path.resolve())
    if value.get("campaign_id") not in {
        CAMPAIGN_ID,
        "ALOD-SYNTH-STRONG-LOD-EXT-MULTISET-CLOSURE-001",
    }:
        raise RuntimeError("unsupported synthetic campaign ID")
    return value


def pass_correctness(study: dict[str, Any]) -> str:
    return str(study.get("pass_correctness", PASS_CORRECTNESS))


def pass_campaign(study: dict[str, Any]) -> str:
    return str(study.get("pass_campaign", PASS_CAMPAIGN))


def verify_inputs(study: dict[str, Any]) -> None:
    for path_key, hash_key in (
        ("asset_path", "asset_sha256"),
        ("camera_path", "camera_path_sha256"),
        ("official_summary", "official_summary_sha256"),
        ("strong_lod_classification", "strong_lod_classification_sha256"),
        ("gsplat_extension_path", "gsplat_extension_sha256"),
    ):
        path = Path(study[path_key])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != study[hash_key]:
            raise RuntimeError(f"input hash mismatch for {path}: {actual}")
    classification = read_json(Path(study["strong_lod_classification"]))
    if classification.get("classification") != "PASS-SYNTH-STRONG-LOD-CONTROL":
        raise RuntimeError("strong-LoD gate has not passed")


def build_config(study: dict[str, Any], mode: str, capacity: int | None = None) -> dict[str, Any]:
    base_id = {
        "nocache": "ALOD-REPRO-HOST-NOCACHE",
        "previous": "ALOD-EXT-PREMATERIALIZATION-CUDA-PREVIOUS",
        "bounded": "ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL",
    }[mode]
    config = deepcopy(load_config(base_id))
    config["configuration_id"] = f"{study['execution_id']}-{mode.upper()}"
    config["description"] = {
        "nocache": "Released Evaluator (No Cache) on frozen synthetic strong-LoD control.",
        "previous": "Unchanged CUDA conservative filter plus previous-working-set reuse.",
        "bounded": "Unchanged CUDA conservative filter plus frozen union-headroom row-LRU.",
    }[mode]
    config["asset_role"] = study["campaign_id"]
    config["camera_path"] = study["camera_path"]
    config["trajectory_id"] = study["trajectory_id"]
    config["trajectory_notes"] = "Frozen 25-camera logarithmic near-to-far synthetic trajectory."
    config["camera_resolution_scale"] = 1.0
    config["width"] = int(study["width"])
    config["height"] = int(study["height"])
    config["warmup_frames"] = int(study["warmup_frames"])
    config["measured_frames"] = int(study["camera_count"])
    config["num_repeat"] = 1
    config["record_warmup_frames"] = True
    config["output_policy"]["write_images"] = False
    config["alod"].update(study["alod"])
    if mode == "bounded":
        if capacity is None:
            raise ValueError("bounded mode requires frozen capacity")
        config["alod_host"]["cache_capacity_rows"] = int(capacity)
    return config


def configured_cameras(study: dict[str, Any], config: dict[str, Any]):
    cameras = apply_camera_config(
        load_camera_path(Path(study["camera_path"]), study["trajectory_id"]), config
    )
    if len(cameras.cameras) != int(study["camera_count"]):
        raise RuntimeError("frozen camera count mismatch")
    return cameras


def host_config(mode: str, capacity: int | None = None, *, properties: bool = False):
    return AlodHostResidencyConfig(
        mode={
            "nocache": "host_nocache",
            "previous": "host_previous_reuse",
            "bounded": "host_bounded_cache",
        }[mode],
        cache_capacity_rows=capacity,
        activation_chunk_rows=131072,
        pin_memory=False,
        validate_mappings=True,
        validate_properties=properties,
    )


def capacity_from_union(study: dict[str, Any], union_rows: int) -> int:
    rule = study["capacity_rule"]
    proposed = math.ceil(
        union_rows
        * int(rule["run009_practical_capacity_rows"])
        / int(rule["run009_union_rows"])
    )
    return min(int(study["hierarchy_rows"]), proposed)


def output_psnr(actual: torch.Tensor, expected: torch.Tensor) -> float | None:
    mse = float(torch.mean((actual.float() - expected.float()) ** 2).item())
    if mse == 0.0:
        return None
    return float(10.0 * math.log10(1.0 / mse))


def multiset_false_negative_count(
    required: np.ndarray, retained: np.ndarray
) -> int:
    required_ids, required_counts = np.unique(required, return_counts=True)
    retained_ids, retained_counts = np.unique(retained, return_counts=True)
    retained_by_id = dict(zip(retained_ids.tolist(), retained_counts.tolist()))
    return int(
        sum(
            max(0, int(count) - int(retained_by_id.get(int(global_id), 0)))
            for global_id, count in zip(required_ids, required_counts)
        )
    )


@torch.no_grad()
def command_correctness(args: argparse.Namespace) -> int:
    study = load_study(args.config)
    verify_inputs(study)
    _preload_gsplat_extension(Path(study["gsplat_extension_path"]))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("synthetic correctness requires CUDA")
    faithful_config = build_config(study, "nocache")
    fast_config = build_config(study, "previous")
    faithful_runtime = _runtime_config(faithful_config)
    fast_runtime = _runtime_config(fast_config)
    if faithful_runtime.__dict__ | {"materialization_filter": "none"} != (
        fast_runtime.__dict__ | {"materialization_filter": "none"}
    ):
        raise RuntimeError("extension configuration changed semantic selection")

    official = read_json(Path(study["official_summary"]))
    cameras = configured_cameras(study, faithful_config)
    if len(official["cameras"]) != len(cameras.cameras):
        raise RuntimeError("official selector and frozen camera count differ")
    asset = Path(study["asset_path"])
    faithful = AlodHostScene(asset, device, faithful_runtime, host_config("nocache"))
    fast = AlodHostScene(asset, device, fast_runtime, host_config("previous", properties=True))
    renderer = GsplatRenderer(device)
    rows = []
    trace: dict[str, np.ndarray] = {}
    union = np.empty(0, dtype=np.int64)
    agreement = study["image_agreement"]
    requested_indices = (
        [int(value) for value in args.camera_indices.split(",")]
        if args.camera_indices
        else list(range(len(cameras.cameras)))
    )
    if len(set(requested_indices)) != len(requested_indices) or any(
        index < 0 or index >= len(cameras.cameras) for index in requested_indices
    ):
        raise ValueError("camera indices must be unique and in range")
    for index, (camera_spec, official_record) in enumerate(
        zip(cameras.cameras, official["cameras"])
    ):
        if index not in requested_indices:
            continue
        if int(official_record["camera_index"]) != index:
            raise RuntimeError("official selector order mismatch")
        camera = make_frame_camera(camera_spec, device)
        settings = make_render_settings(faithful_config, camera_spec, sh_degree=1)
        faithful_active = faithful.prepare_frame(camera, settings)
        faithful_render = renderer.render(faithful_active, camera, settings)
        fast_active = fast.prepare_frame(camera, settings)
        fast_render = renderer.render(fast_active, camera, settings)
        faithful_selection = faithful.last_selection
        fast_selection = fast.last_selection
        materialization = fast.last_materialization
        assert faithful_selection is not None and fast_selection is not None
        assert materialization is not None

        official_hierarchy = np.load(
            official_record["selected_ids_path"], allow_pickle=False
        ).astype(np.int64)
        official_raster = np.concatenate(
            (np.arange(int(study["skybox_rows"]), dtype=np.int64), official_hierarchy)
        )
        np.testing.assert_array_equal(
            faithful_selection.raster_global_node_ids, official_raster
        )
        np.testing.assert_array_equal(
            fast_selection.raster_global_node_ids, official_raster
        )

        cpu = conservative_pinhole_materialization(
            fast.compact_culling, official_raster, camera, settings
        )
        np.testing.assert_array_equal(
            materialization.materialization_global_node_ids,
            cpu.materialization_global_node_ids,
        )
        np.testing.assert_array_equal(
            materialization.kept_semantic_indices, cpu.kept_semantic_indices
        )
        unique_semantic = materialization.unique_semantic_global_node_ids
        semantic_to_unique = materialization.semantic_to_unique_indices
        unique_materialized = materialization.unique_materialization_global_node_ids
        materialization_to_unique = materialization.materialization_to_unique_indices
        assert unique_semantic is not None and semantic_to_unique is not None
        assert unique_materialized is not None and materialization_to_unique is not None
        np.testing.assert_array_equal(
            unique_semantic[semantic_to_unique], official_raster
        )
        np.testing.assert_array_equal(
            unique_materialized[materialization_to_unique],
            materialization.materialization_global_node_ids,
        )
        kept = torch.from_numpy(cpu.kept_semantic_indices).to(device=device)
        property_fields = {}
        for field in (
            "means",
            "quats",
            "scales",
            "opacities",
            "colors",
            "global_node_ids",
        ):
            expected = getattr(faithful_active, field).index_select(0, kept)
            actual = getattr(fast_active, field)
            property_fields[field] = bool(torch.equal(actual, expected))
            if not property_fields[field]:
                raise RuntimeError(f"retained {field} mismatch at camera {index}")

        assembled_ids = fast_active.global_node_ids.detach().cpu().numpy()
        np.testing.assert_array_equal(
            assembled_ids, materialization.materialization_global_node_ids
        )

        faithful_backend = _projection_ids(faithful_render)
        fast_backend = _projection_ids(fast_render)
        np.testing.assert_array_equal(fast_backend, faithful_backend)
        false_negative_count = multiset_false_negative_count(
            faithful_backend,
            materialization.materialization_global_node_ids,
        )
        if false_negative_count:
            raise RuntimeError(f"{false_negative_count} filter false negatives")
        rgb = tensor_diff(fast_render.image, faithful_render.image)
        alpha = tensor_diff(fast_render.alpha, faithful_render.alpha)
        for name, diff in (("rgb", rgb), ("alpha", alpha)):
            if diff["mean_absolute_error"] > float(agreement["mean_absolute_error_max"]):
                raise RuntimeError(f"{name} mean image error exceeded frozen bound")
            if diff["maximum_absolute_error"] > float(agreement["maximum_absolute_error_max"]):
                raise RuntimeError(f"{name} maximum image error exceeded frozen bound")

        hierarchy_materialized = cpu.materialization_global_node_ids[
            cpu.materialization_global_node_ids >= int(study["skybox_rows"])
        ]
        unique_hierarchy_semantic = unique_semantic[
            unique_semantic >= int(study["skybox_rows"])
        ]
        hierarchy_semantic_to_unique = semantic_to_unique[
            official_raster >= int(study["skybox_rows"])
        ] - int(np.count_nonzero(unique_semantic < int(study["skybox_rows"])))
        unique_hierarchy_materialized = unique_materialized[
            unique_materialized >= int(study["skybox_rows"])
        ]
        union = np.union1d(union, unique_hierarchy_materialized)
        trace[f"frame_{index:03d}_semantic_hierarchy_ids"] = official_hierarchy
        trace[f"frame_{index:03d}_unique_semantic_hierarchy_ids"] = (
            unique_hierarchy_semantic
        )
        trace[f"frame_{index:03d}_semantic_to_unique_hierarchy_index"] = (
            hierarchy_semantic_to_unique
        )
        trace[f"frame_{index:03d}_materialized_hierarchy_ids"] = hierarchy_materialized
        trace[f"frame_{index:03d}_unique_materialized_hierarchy_ids"] = (
            unique_hierarchy_materialized
        )
        trace[f"frame_{index:03d}_assembled_renderer_ids"] = assembled_ids
        trace[f"frame_{index:03d}_packed_survivor_ids"] = faithful_backend
        duplicate_occurrences = int(
            official_hierarchy.size - unique_hierarchy_semantic.size
        )
        duplicate_detail = (
            {
                "official_semantic_ordered_ids": official_hierarchy.tolist(),
                "unique_semantic_ids": unique_hierarchy_semantic.tolist(),
                "semantic_to_unique_index": hierarchy_semantic_to_unique.tolist(),
                "retained_unique_ids": unique_hierarchy_materialized.tolist(),
                "assembled_renderer_ordered_ids": assembled_ids[
                    assembled_ids >= int(study["skybox_rows"])
                ].tolist(),
                "packed_survivor_ids": faithful_backend.tolist(),
            }
            if duplicate_occurrences
            else None
        )
        rows.append(
            {
                "camera_index": index,
                "camera_stable_id": official_record["camera_stable_id"],
                "semantic_rows": int(official_raster.size),
                "semantic_hierarchy_rows": int(official_hierarchy.size),
                "unique_semantic_rows": int(unique_semantic.size),
                "unique_semantic_hierarchy_rows": int(unique_hierarchy_semantic.size),
                "duplicate_occurrence_count": duplicate_occurrences,
                "duplication_ratio": (
                    float(duplicate_occurrences / official_hierarchy.size)
                    if official_hierarchy.size
                    else 0.0
                ),
                "materialized_rows": int(cpu.materialization_global_node_ids.size),
                "materialized_hierarchy_rows": int(hierarchy_materialized.size),
                "unique_materialized_rows": int(unique_materialized.size),
                "unique_materialized_hierarchy_rows": int(
                    unique_hierarchy_materialized.size
                ),
                "assembled_renderer_occurrence_rows": int(assembled_ids.size),
                "packed_projection_survivors": int(faithful_backend.size),
                "official_semantic_ids_exact": True,
                "unique_mapping_reconstructs_ordered_semantics": True,
                "cpu_cuda_ordered_filter_ids_exact": True,
                "false_negative_count": 0,
                "properties_bitwise_exact": property_fields,
                "renderer_input_ids_and_order_exact": True,
                "packed_survivor_ids_exact": True,
                "rgb": {**rgb, "psnr_between_outputs": output_psnr(fast_render.image, faithful_render.image)},
                "alpha": {
                    **alpha,
                    "psnr_between_outputs": output_psnr(
                        fast_render.alpha, faithful_render.alpha
                    ),
                },
                "duplicate_trace": duplicate_detail,
            }
        )
        print(f"[{len(rows)}/{len(requested_indices)}] camera {index} PASS", flush=True)

    capacity = capacity_from_union(study, int(union.size))
    maximum_request = max(
        row["unique_materialized_hierarchy_rows"] for row in rows
    )
    if capacity < maximum_request:
        raise RuntimeError("frozen capacity cannot hold maximum frame request")
    result = {
        "schema_version": 1,
        "campaign_id": study["campaign_id"],
        "controlled_parent_id": study.get("controlled_parent_id"),
        "execution_id": study["execution_id"],
        "status": (
            study.get("pass_duplicate_correctness")
            if requested_indices == study.get("duplicate_camera_indices")
            else pass_correctness(study)
        ),
        "scope": (
            "known duplicate cameras"
            if requested_indices == study.get("duplicate_camera_indices")
            else "complete frozen trajectory"
        ),
        "asset": {"path": str(asset), "sha256": sha256_file(asset)},
        "camera_path": {
            "path": study["camera_path"],
            "sha256": sha256_file(Path(study["camera_path"])),
        },
        "official_summary": {
            "path": study["official_summary"],
            "sha256": sha256_file(Path(study["official_summary"])),
        },
        "camera_count": len(rows),
        "camera_indices": requested_indices,
        "criteria": {
            "official_semantic_ids_and_order_exact": True,
            "unique_mapping_exact": True,
            "cpu_cuda_filter_ids_and_order_exact": True,
            "conservative_filter_false_negatives": 0,
            "retained_properties_and_order_bitwise_exact": True,
            "renderer_input_ids_order_and_multiplicity_exact": True,
            "packed_projection_survivor_ids_and_order_exact": True,
            "image_agreement_bounds": agreement,
        },
        "capacity": {
            "filtered_canonical_union_rows": int(union.size),
            "filtered_union_unique_rows": int(union.size),
            "maximum_single_frame_hierarchy_rows": maximum_request,
            "maximum_single_frame_unique_hierarchy_rows": max(
                row["unique_materialized_hierarchy_rows"] for row in rows
            ),
            "hierarchy_rows": int(study["hierarchy_rows"]),
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
    verify_inputs(study)
    _preload_gsplat_extension(Path(study["gsplat_extension_path"]))
    config = build_config(study, args.mode, args.capacity)
    cameras = configured_cameras(study, config)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("synthetic benchmark requires CUDA")
    return run_benchmark(
        config=config,
        asset_path=Path(study["asset_path"]),
        asset_supplied=study["asset_path"],
        camera_path=cameras,
        output_dir=args.output_dir.resolve(),
        record_kind="formal_benchmark",
        max_frames=len(cameras.cameras),
        device=device,
        log_gpu_events=True,
    )


def command_campaign(args: argparse.Namespace) -> int:
    study = load_study(args.config)
    correctness = read_json(args.correctness)
    if correctness.get("status") != pass_correctness(study):
        raise RuntimeError("required correctness gate has not passed")
    capacity = int(correctness["capacity"]["bounded_capacity_rows"])
    root = args.campaign_root.resolve()
    script = Path(__file__).resolve()
    rows = []
    for replicate in range(1, int(study["replicates"]) + 1):
        rotated = MODE_ORDER[replicate - 1 :] + MODE_ORDER[: replicate - 1]
        for mode in rotated:
            output = root / "runs" / study["execution_id"] / mode / f"replicate-{replicate:02d}"
            command = [
                sys.executable,
                str(script),
                "benchmark",
                "--config",
                str(args.config.resolve()),
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
    schedule_path = root / "checks/schedule.json"
    write_json(schedule_path, {"campaign_id": study["campaign_id"], "schedule": rows})
    for row in rows:
        summary = Path(row["output_dir"]) / "summary.json"
        if summary.exists():
            if not args.resume:
                raise FileExistsError(summary)
            row["status"] = "resumed_complete"
            continue
        row["status"] = "running"
        row["started_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(schedule_path, {"campaign_id": study["campaign_id"], "schedule": rows})
        print(
            f"[{row['sequence_index']}/{len(rows)}] {row['mode']} replicate {row['replicate']}",
            flush=True,
        )
        completed = subprocess.run(row["command"], cwd=REPO_ROOT)
        row["finished_utc"] = datetime.now(timezone.utc).isoformat()
        row["status"] = "complete" if completed.returncode == 0 else "failed"
        write_json(schedule_path, {"campaign_id": study["campaign_id"], "schedule": rows})
        if completed.returncode != 0:
            raise RuntimeError(f"benchmark failed: {row['mode']} replicate {row['replicate']}")
    return 0


def command_aggregate(args: argparse.Namespace) -> int:
    study = load_study(args.config)
    correctness = read_json(args.correctness)
    if correctness.get("status") != pass_correctness(study):
        raise RuntimeError("correctness gate did not pass")
    schedule = read_json(args.campaign_root.resolve() / "checks/schedule.json")["schedule"]
    if any(row["status"] not in {"complete", "resumed_complete"} for row in schedule):
        raise RuntimeError("campaign schedule is incomplete")
    modes = {}
    for mode in MODE_ORDER:
        paths = [Path(row["output_dir"]) for row in schedule if row["mode"] == mode]
        modes[mode] = summarize_mode(paths)
        measured = [item for path in paths for item in read_jsonl(path / "frames.jsonl")]
        modes[mode]["renderer_wall_time_ms"] = distribution(measured, "frame_time_ms")
        modes[mode]["inclusive_preparation_time_ms"] = distribution(
            measured, "frame_preparation_ms"
        )
        for key in (
            "alod_materialization_unique_extraction_ms",
            "alod_materialization_occurrence_propagation_ms",
            "alod_occurrence_reindex_ms",
        ):
            modes[mode].setdefault("multiset_timings_ms", {})[key] = distribution(
                measured, key
            )
        for key in (
            "alod_unique_semantic_rows",
            "alod_duplicate_semantic_occurrences",
            "alod_unique_materialization_rows",
            "alod_assembled_renderer_occurrence_rows",
        ):
            modes[mode].setdefault("multiset_counts", {})[key] = distribution(
                measured, key
            )
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
    unique_semantic = np.asarray(
        [row["unique_semantic_rows"] for row in camera_rows], dtype=np.float64
    )
    unique_materialized = np.asarray(
        [row["unique_materialized_rows"] for row in camera_rows], dtype=np.float64
    )
    survivors = np.asarray(
        [row["packed_projection_survivors"] for row in camera_rows], dtype=np.float64
    )
    output = {
        "schema_version": 1,
        "campaign_id": study["campaign_id"],
        "controlled_parent_id": study.get("controlled_parent_id"),
        "status": pass_campaign(study),
        "classification": study["classification"],
        "environment": environment_metadata(),
        "protocol": {
            "replicates": int(study["replicates"]),
            "warmup_frames": int(study["warmup_frames"]),
            "measured_frames_per_replicate": int(study["camera_count"]),
            "measured_sequence_restarts_after_warmup": True,
            "cache_state_survives_warmup": True,
            "property_bytes_per_row": int(study["property_bytes_per_row"]),
            "capacity_rule": study["capacity_rule"],
            "timing_scope": {
                "preparation_time_ms": "inclusive; selection/filter/cache/gather/H2D/assembly are nested diagnostics",
                "end_to_end": "preparation wall time + renderer wall time",
                "rasterization": "nested within renderer wall time",
                "nested_columns_must_not_be_summed": True,
            },
        },
        "correctness_status": correctness["status"],
        "bounded_capacity_rows": correctness["capacity"]["bounded_capacity_rows"],
        "filtered_trajectory_union_rows": correctness["capacity"]["filtered_canonical_union_rows"],
        "workload_funnel": {
            "semantic_occurrence_rows": distribution(camera_rows, "semantic_rows"),
            "unique_semantic_rows": distribution(camera_rows, "unique_semantic_rows"),
            "duplicate_occurrence_count": distribution(
                camera_rows, "duplicate_occurrence_count"
            ),
            "unique_materialized_rows": distribution(
                camera_rows, "unique_materialized_rows"
            ),
            "assembled_renderer_occurrence_rows": distribution(
                camera_rows, "assembled_renderer_occurrence_rows"
            ),
            "packed_projection_survivors": distribution(
                camera_rows, "packed_projection_survivors"
            ),
            "pooled_unique_materialized_over_unique_semantic": float(
                unique_materialized.sum() / unique_semantic.sum()
            ),
            "pooled_unique_materialized_over_semantic_occurrences": float(
                unique_materialized.sum() / semantic.sum()
            ),
            "pooled_survivors_over_semantic": float(survivors.sum() / semantic.sum()),
            "mean_per_camera_unique_materialized_over_unique_semantic": float(
                np.mean(unique_materialized / unique_semantic)
            ),
            "mean_per_camera_survivors_over_semantic": float(
                np.mean(survivors / semantic)
            ),
        },
        "modes": modes,
        "scientific_question": (
            "When the released selector exhibits deliberately strong monotonic semantic "
            "LoD, does ordered-multiset-faithful ALoD-EXT still reduce physical "
            "materialization, transfer, and end-to-end frame time?"
        ),
    }
    write_json(args.output, output)
    return 0


def command_protocol(args: argparse.Namespace) -> int:
    study = load_study(args.config)
    verify_inputs(study)
    note = Path(study["protocol_note"])
    sources = [
        Path(__file__).resolve(),
        REPO_ROOT / "scripts/alod/run_synthetic_strong_lod_ext_multiset_closure.py",
        args.config.resolve(),
        REPO_ROOT / "benchmarks/benchmark_real_scene.py",
        REPO_ROOT / "scene_runtime/types.py",
        REPO_ROOT / "scene_runtime/alod/selection.py",
        REPO_ROOT / "scene_runtime/alod/materialization.py",
        REPO_ROOT / "scene_runtime/alod/host_scene.py",
        REPO_ROOT / "scene_runtime/alod/residency.py",
        REPO_ROOT / "unit_tests/test_alod_materialization.py",
        REPO_ROOT / "unit_tests/test_alod_host_residency.py",
    ]
    write_json(
        args.output,
        {
            "schema_version": 1,
            "campaign_id": study["campaign_id"],
            "controlled_parent_id": study.get("controlled_parent_id"),
            "status": "FROZEN-BEFORE-MULTISET-CORRECTNESS-AND-TIMING",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_note": {"path": str(note), "sha256": sha256_file(note)},
            "configuration": {
                "path": str(args.config.resolve()),
                "sha256": sha256_file(args.config.resolve()),
            },
            "implementation": {str(path): sha256_file(path) for path in sources},
            "inputs": {
                key: study[key]
                for key in (
                    "asset_path",
                    "asset_sha256",
                    "camera_path",
                    "camera_path_sha256",
                    "official_summary",
                    "official_summary_sha256",
                    "strong_lod_classification",
                    "strong_lod_classification_sha256",
                    "gsplat_extension_path",
                    "gsplat_extension_sha256",
                )
            },
            "modes": list(MODE_ORDER),
            "replicates": study["replicates"],
            "warmup_frames": study["warmup_frames"],
            "capacity_rule": study["capacity_rule"],
            "image_agreement": study["image_agreement"],
            "correctness_required_before_timing": True,
        },
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    protocol = sub.add_parser("protocol", parents=[common])
    protocol.add_argument("--output", type=Path, required=True)
    protocol.set_defaults(func=command_protocol)
    correctness = sub.add_parser("correctness", parents=[common])
    correctness.add_argument("--output", type=Path, required=True)
    correctness.add_argument("--trace", type=Path, required=True)
    correctness.add_argument("--device", default="cuda:0")
    correctness.add_argument(
        "--camera-indices",
        help="optional comma-separated correctness subset, preserving listed order",
    )
    correctness.set_defaults(func=command_correctness)
    benchmark = sub.add_parser("benchmark", parents=[common])
    benchmark.add_argument("--mode", choices=MODE_ORDER, required=True)
    benchmark.add_argument("--capacity", type=int)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.set_defaults(func=command_benchmark)
    campaign = sub.add_parser("campaign", parents=[common])
    campaign.add_argument("--correctness", type=Path, required=True)
    campaign.add_argument("--campaign-root", type=Path, required=True)
    campaign.add_argument("--resume", action="store_true")
    campaign.set_defaults(func=command_campaign)
    aggregate = sub.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--correctness", type=Path, required=True)
    aggregate.add_argument("--campaign-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.set_defaults(func=command_aggregate)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
