#!/usr/bin/env python3
"""Exact Run009 correctness gate for the existing ALOD-EXT runtime ladder."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_selection_oracle import load_manifest, sha256_file, validate_oracle_tree  # noqa: E402
from benchmarks.alod_resident_benchmark import _runtime_config  # noqa: E402
from benchmarks.benchmark_real_scene import load_config  # noqa: E402
from scene_runtime.alod import (  # noqa: E402
    AlodHostResidencyConfig,
    conservative_pinhole_materialization,
)
from scene_runtime.alod.host_scene import AlodHostScene  # noqa: E402
from scene_runtime.renderer import GsplatRenderer  # noqa: E402
from scripts.alod.validate_fast_materialization_filter import _projection_ids  # noqa: E402
from scripts.alod.validate_production_resident import (  # noqa: E402
    _frame_camera,
    _json,
    _payload,
    _pre_raster_hashes,
    _settings,
)


STATUS = "PASS-ALOD-EXT-FORMAL-RUN009-CORRECTNESS"
EXPECTED_ASSET_SHA256 = "38f4abdc0d47f2d20316de33f4fff87086e038a57daac0f2feaf98fa9178eb7b"
MODE_IDS = (
    "ALOD-EXT-ROW-LRU-PRACTICAL",
    "ALOD-EXT-PREMATERIALIZATION-CPU-PRACTICAL",
    "ALOD-EXT-PREMATERIALIZATION-CUDA-PREVIOUS",
    "ALOD-EXT-PREMATERIALIZATION-CUDA-MINIMUM",
    "ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL",
    "ALOD-EXT-PREMATERIALIZATION-CUDA-LARGE",
)


def _host_config(config: dict[str, Any]) -> AlodHostResidencyConfig:
    host = config["alod_host"]
    return AlodHostResidencyConfig(
        mode=host["mode"],
        cache_capacity_rows=host.get("cache_capacity_rows"),
        activation_chunk_rows=int(host.get("activation_chunk_rows", 131072)),
        pin_memory=bool(host.get("pin_memory", False)),
        validate_mappings=True,
        validate_properties=True,
    )


def _hash_active(active) -> str:
    digest = hashlib.sha256()
    for field in ("means", "quats", "scales", "opacities", "colors", "global_node_ids"):
        value = getattr(active, field).detach().cpu().contiguous().numpy()
        digest.update(field.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _diff(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    difference = torch.abs(actual.float() - expected.float())
    return {
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "maximum_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
    }


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--trace-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal extension validation requires CUDA")
    asset = args.asset.resolve()
    if sha256_file(asset) != EXPECTED_ASSET_SHA256:
        raise RuntimeError("Run009 hierarchy hash mismatch")
    manifest_path = args.oracle_manifest.resolve()
    manifest = load_manifest(manifest_path)
    validate_oracle_tree(manifest_path)
    if manifest["asset"]["sha256"] != EXPECTED_ASSET_SHA256:
        raise RuntimeError("oracle does not name the formal Run009 hierarchy")

    faithful_config = load_config("ALOD-REPRO-HOST-NOCACHE")
    fast_config = load_config("ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL")
    faithful_runtime = _runtime_config(faithful_config)
    fast_runtime = _runtime_config(fast_config)
    if faithful_runtime.__dict__ | {"materialization_filter": "none"} != (
        fast_runtime.__dict__ | {"materialization_filter": "none"}
    ):
        raise RuntimeError("extension configuration changed semantic selection")

    faithful = AlodHostScene(
        asset,
        device,
        faithful_runtime,
        AlodHostResidencyConfig(mode="host_nocache", validate_mappings=True, validate_properties=True),
    )
    fast = AlodHostScene(asset, device, fast_runtime, _host_config(fast_config))
    renderer = GsplatRenderer(device)
    root = manifest_path.parent
    reference_rows: list[dict[str, Any]] = []
    camera_rows = []
    for camera_index, entry in enumerate(manifest["cameras"]):
        camera_path = root / entry["paths"]["camera.json"]
        record = _json(camera_path)
        camera = _frame_camera(record, device)
        settings = _settings(record)
        faithful_active = faithful.prepare_frame(camera, settings)
        faithful_render = renderer.render(faithful_active, camera, settings)
        fast_active = fast.prepare_frame(camera, settings)
        fast_render = renderer.render(fast_active, camera, settings)
        faithful_selection = faithful.last_selection
        fast_selection = fast.last_selection
        fast_materialization = fast.last_materialization
        assert faithful_selection is not None and fast_selection is not None
        assert fast_materialization is not None and fast.compact_culling is not None

        working = _json(camera_path.parent / "working-set.json")
        oracle_raster = _payload(camera_path.parent, working["global_node_id_payload"])
        np.testing.assert_array_equal(faithful_selection.raster_global_node_ids, oracle_raster)
        np.testing.assert_array_equal(fast_selection.raster_global_node_ids, oracle_raster)
        np.testing.assert_array_equal(
            faithful_selection.selected_spt_global_node_ids,
            fast_selection.selected_spt_global_node_ids,
        )
        assert faithful_selection.upper is not None and fast_selection.upper is not None
        np.testing.assert_array_equal(
            faithful_selection.upper.candidate_spt_ids,
            fast_selection.upper.candidate_spt_ids,
        )
        cpu = conservative_pinhole_materialization(
            fast.compact_culling, oracle_raster, camera, settings
        )
        np.testing.assert_array_equal(
            fast_materialization.materialization_global_node_ids,
            cpu.materialization_global_node_ids,
        )
        np.testing.assert_array_equal(
            fast_materialization.kept_semantic_indices, cpu.kept_semantic_indices
        )
        kept = torch.from_numpy(cpu.kept_semantic_indices).to(device=device)
        for field in ("means", "quats", "scales", "opacities", "colors", "global_node_ids"):
            expected = getattr(faithful_active, field).index_select(0, kept)
            if not torch.equal(getattr(fast_active, field), expected):
                raise RuntimeError(f"surviving {field} differs at diagnostic camera {camera_index}")
        faithful_backend = _projection_ids(faithful_render)
        fast_backend = _projection_ids(fast_render)
        np.testing.assert_array_equal(fast_backend, faithful_backend)
        false_negatives = np.setdiff1d(
            faithful_backend, fast_materialization.materialization_global_node_ids
        )
        if false_negatives.size:
            raise RuntimeError(f"filter false negatives at diagnostic camera {camera_index}")
        rgb = _diff(fast_render.image, faithful_render.image)
        alpha = _diff(fast_render.alpha, faithful_render.alpha)
        if not rgb["bitwise_equal"] or not alpha["bitwise_equal"]:
            raise RuntimeError(f"render identity differs at diagnostic camera {camera_index}")
        reference_rows.append(
            {
                "camera": camera,
                "settings": settings,
                "semantic_ids": oracle_raster.copy(),
                "materialization_ids": cpu.materialization_global_node_ids.copy(),
                "faithful_hash": _hash_active(faithful_active),
                "filtered_hash": _hash_active(fast_active),
            }
        )
        camera_rows.append(
            {
                "camera_index": camera_index,
                "camera_stable_id": entry["camera_stable_id"],
                "semantic_rows": int(oracle_raster.size),
                "materialized_rows": int(cpu.materialization_rows),
                "projection_survivors": int(faithful_backend.size),
                "cpu_gpu_ordered_materialization_ids_exact": True,
                "backend_false_negatives": 0,
                "properties_bitwise_exact": True,
                "rgb": rgb,
                "alpha": alpha,
                "selected_spts_exact": True,
                "touched_spts_exact": True,
            }
        )

    del faithful, fast
    gc.collect()
    torch.cuda.empty_cache()

    mode_rows = {}
    for config_id in MODE_IDS:
        config = load_config(config_id)
        runtime = _runtime_config(config)
        if faithful_runtime.__dict__ | {"materialization_filter": "none"} != (
            runtime.__dict__ | {"materialization_filter": "none"}
        ):
            raise RuntimeError(f"{config_id} changed semantic selection")
        scene = AlodHostScene(asset, device, runtime, _host_config(config))
        checks = []
        for reference in reference_rows:
            active = scene.prepare_frame(reference["camera"], reference["settings"])
            selection = scene.last_selection
            materialization = scene.last_materialization
            assert selection is not None and materialization is not None
            np.testing.assert_array_equal(selection.raster_global_node_ids, reference["semantic_ids"])
            expected_ids = (
                reference["semantic_ids"]
                if runtime.materialization_filter == "none"
                else reference["materialization_ids"]
            )
            np.testing.assert_array_equal(materialization.materialization_global_node_ids, expected_ids)
            expected_hash = (
                reference["faithful_hash"]
                if runtime.materialization_filter == "none"
                else reference["filtered_hash"]
            )
            if _hash_active(active) != expected_hash:
                raise RuntimeError(f"{config_id} pre-raster properties/order differ")
            checks.append(True)
        mode_rows[config_id] = {
            "diagnostic_camera_count": len(checks),
            "semantic_ids_exact": all(checks),
            "materialization_ids_exact": all(checks),
            "properties_and_order_exact": all(checks),
            "capacity_rows": config["alod_host"].get("cache_capacity_rows"),
        }
        del scene
        gc.collect()
        torch.cuda.empty_cache()

    # Exhaustive 71-camera CPU/GPU materialization-ID parity.  The CPU trace was
    # frozen before timing; replay it through the production CUDA filter.
    trace = np.load(args.trace_npz)
    scene = AlodHostScene(asset, device, fast_runtime, _host_config(fast_config))
    from benchmarks.benchmark_real_scene import (  # local import keeps the gate self-contained
        apply_camera_config, load_camera_path, make_frame_camera,
        make_render_settings, resolve_camera_path,
    )
    camera_path = apply_camera_config(
        load_camera_path(resolve_camera_path(faithful_config), faithful_config["trajectory_id"]),
        faithful_config,
    )
    exhaustive = []
    for index, spec in enumerate(camera_path.cameras):
        camera = make_frame_camera(spec, device)
        settings = make_render_settings(faithful_config, spec, sh_degree=1)
        scene.prepare_frame(camera, settings)
        selection = scene.last_selection
        materialization = scene.last_materialization
        assert selection is not None and materialization is not None
        expected_semantic = trace[f"frame_{index:03d}_semantic_hierarchy_ids"]
        expected_materialized = trace[f"frame_{index:03d}_materialized_hierarchy_ids"]
        np.testing.assert_array_equal(selection.selected_hierarchy_global_node_ids, expected_semantic)
        actual_materialized = materialization.materialization_global_node_ids
        actual_materialized = actual_materialized[actual_materialized >= fast_runtime.skybox_count]
        np.testing.assert_array_equal(actual_materialized, expected_materialized)
        exhaustive.append({"camera_index": index, "ordered_ids_exact": True})

    result = {
        "schema_version": 1,
        "status": STATUS,
        "asset": {"path": str(asset), "sha256": sha256_file(asset)},
        "oracle_manifest": str(manifest_path),
        "diagnostic_camera_count": len(camera_rows),
        "formal_workload_camera_count": len(exhaustive),
        "checks": {
            "semantic_oracle_ids_and_order_exact": True,
            "selected_and_touched_spts_exact": True,
            "cpu_gpu_ordered_materialization_ids_exact_all_71": True,
            "backend_false_negatives_zero": True,
            "surviving_properties_and_order_bitwise_exact": True,
            "rgb_bitwise_exact": True,
            "alpha_bitwise_exact": True,
            "all_extension_modes_exact": True,
        },
        "diagnostic_cameras": camera_rows,
        "extension_modes": mode_rows,
        "exhaustive_filter_parity": exhaustive,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": STATUS, "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
