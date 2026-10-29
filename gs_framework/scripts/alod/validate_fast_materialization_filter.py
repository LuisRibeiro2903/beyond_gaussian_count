#!/usr/bin/env python3
"""Validate and time the production torch pre-materialization filter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.alod_resident_benchmark import _runtime_config  # noqa: E402
from benchmarks.benchmark_real_scene import (  # noqa: E402
    apply_camera_config,
    load_camera_path,
    load_config,
    make_frame_camera,
    make_render_settings,
    resolve_camera_path,
    sha256_file,
)
from scene_runtime.alod import (  # noqa: E402
    AlodHostResidencyConfig,
    conservative_pinhole_materialization,
    fast_conservative_pinhole_materialization,
)
from scene_runtime.alod.host import property_tensor_bytes  # noqa: E402
from scene_runtime.alod.host_scene import AlodHostScene  # noqa: E402
from scene_runtime.alod.scene import AlodResidentScene  # noqa: E402
from scene_runtime.renderer import GsplatRenderer  # noqa: E402


PASS_STATUS = "PASS-ALOD-FAST-PREMATERIALIZATION-CULLING"
EXPECTED_ASSET_SHA256 = (
    "6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c"
)


def _preload_gsplat_extension(path: Path | None) -> dict[str, str] | None:
    """Use an already-built extension when launcher CUDA discovery is unsuitable."""

    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    spec = importlib.util.spec_from_file_location("gsplat_cuda", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load gsplat extension {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import gsplat

    sys.modules["gsplat.csrc"] = module
    gsplat.csrc = module
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def _assert_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual, expected):
        difference = torch.max(torch.abs(actual.float() - expected.float())).item()
        raise RuntimeError(f"{name} differs; maximum absolute difference {difference}")


def _projection_ids(rendered) -> np.ndarray:
    values = rendered.info.get("backend_projection_survivor_global_node_ids")
    if values is None:
        raise RuntimeError("renderer did not expose backend projection survivor IDs")
    return np.ascontiguousarray(
        values.detach().to(device="cpu", dtype=torch.int64).numpy(),
        dtype=np.int64,
    )


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


@torch.no_grad()
def validate(
    asset_path: Path,
    device: torch.device,
    *,
    warmup_repetitions: int = 2,
    timing_repetitions: int = 10,
    gsplat_extension: Path | None = None,
) -> dict[str, Any]:
    if device.type != "cuda":
        raise ValueError("fast production validation requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if warmup_repetitions < 0 or timing_repetitions <= 0:
        raise ValueError("invalid timing repetitions")

    extension_record = _preload_gsplat_extension(gsplat_extension)
    asset_path = asset_path.resolve()
    asset_sha256 = sha256_file(asset_path)
    if asset_sha256 != EXPECTED_ASSET_SHA256:
        raise RuntimeError(f"unexpected hierarchy SHA-256 {asset_sha256}")
    base_config = load_config("ALOD-HIER-LOD-FRUSTUM")
    reduced_config = load_config("ALOD-HOST-BOUNDED-CACHE-REDUCED")
    cameras = apply_camera_config(
        load_camera_path(
            resolve_camera_path(base_config), base_config.get("trajectory_id")
        ),
        base_config,
    )
    base_runtime = _runtime_config(base_config)
    fast_runtime = _runtime_config(reduced_config)
    if base_runtime.__dict__ | {"materialization_filter": "none"} != (
        fast_runtime.__dict__ | {"materialization_filter": "none"}
    ):
        raise RuntimeError("fast configuration changed semantic selector settings")

    baseline = AlodResidentScene(asset_path, device, base_runtime)
    fast_host = AlodHostScene(
        asset_path,
        device,
        fast_runtime,
        AlodHostResidencyConfig(
            mode="host_bounded_cache",
            cache_capacity_rows=int(
                reduced_config["alod_host"]["cache_capacity_rows"]
            ),
            validate_mappings=True,
            validate_properties=True,
        ),
    )
    if fast_host.compact_culling is None or fast_host.device_culling is None:
        raise RuntimeError("fast scene did not construct compact metadata")
    if not torch.equal(
        fast_host.device_culling.means.cpu(),
        torch.from_numpy(np.array(fast_host.compact_culling.means, copy=True)),
    ):
        raise RuntimeError("CPU/GPU compact means differ")
    if not torch.equal(
        fast_host.device_culling.max_scales.cpu(),
        torch.from_numpy(fast_host.compact_culling.max_scales),
    ):
        raise RuntimeError("CPU/GPU compact max scales differ")

    renderer = GsplatRenderer(device)
    rows: list[dict[str, Any]] = []
    timing_inputs: list[tuple[np.ndarray, Any, Any]] = []
    total_backend_survivors = 0
    for camera_index, camera_spec in enumerate(cameras.cameras):
        camera = make_frame_camera(camera_spec, device)
        settings = make_render_settings(
            base_config, camera_spec, sh_degree=int(base_config["sh_degree"])
        )
        baseline_active = baseline.prepare_frame(camera, settings)
        baseline_render = renderer.render(baseline_active, camera, settings)
        fast_active = fast_host.prepare_frame(camera, settings)
        fast_render = renderer.render(fast_active, camera, settings)
        baseline_selection = baseline.last_selection
        fast_selection = fast_host.last_selection
        fast_materialization = fast_host.last_materialization
        assert baseline_selection is not None
        assert fast_selection is not None
        assert fast_materialization is not None
        cpu_reference = conservative_pinhole_materialization(
            fast_host.compact_culling,
            baseline_selection.raster_global_node_ids,
            camera,
            settings,
        )

        np.testing.assert_array_equal(
            baseline_selection.selected_hierarchy_global_node_ids,
            fast_selection.selected_hierarchy_global_node_ids,
        )
        np.testing.assert_array_equal(
            baseline_selection.raster_global_node_ids,
            fast_selection.raster_global_node_ids,
        )
        np.testing.assert_array_equal(
            fast_materialization.materialization_global_node_ids,
            cpu_reference.materialization_global_node_ids,
        )
        np.testing.assert_array_equal(
            fast_materialization.kept_semantic_indices,
            cpu_reference.kept_semantic_indices,
        )
        if fast_materialization.materialization_global_node_ids_gpu is None:
            raise RuntimeError("fast filter did not retain GPU materialization IDs")
        _assert_equal(
            "GPU materialization IDs",
            fast_materialization.materialization_global_node_ids_gpu,
            torch.from_numpy(cpu_reference.materialization_global_node_ids).to(device),
        )

        kept = torch.from_numpy(cpu_reference.kept_semantic_indices).to(device)
        for field in ("means", "quats", "scales", "opacities", "colors"):
            _assert_equal(
                f"surviving {field}",
                getattr(fast_active, field),
                getattr(baseline_active, field).index_select(0, kept),
            )
        _assert_equal("fast image", fast_render.image, baseline_render.image)
        _assert_equal("fast alpha", fast_render.alpha, baseline_render.alpha)

        baseline_backend_ids = _projection_ids(baseline_render)
        fast_backend_ids = _projection_ids(fast_render)
        np.testing.assert_array_equal(fast_backend_ids, baseline_backend_ids)
        false_negatives = np.setdiff1d(
            baseline_backend_ids,
            fast_materialization.materialization_global_node_ids,
        )
        if false_negatives.size:
            raise RuntimeError("fast filter removed backend projection survivors")
        total_backend_survivors += int(baseline_backend_ids.size)

        timings = fast_materialization.timings
        rows.append(
            {
                "camera_index": camera_index,
                "camera_id": camera_spec.camera_id,
                "semantic_selected_rows": int(
                    baseline_selection.raster_global_node_ids.size
                ),
                "materialization_rows": fast_materialization.materialization_rows,
                "materialization_ids_sha256": _hash(
                    fast_materialization.materialization_global_node_ids
                ),
                "backend_projection_survivors": int(baseline_backend_ids.size),
                "backend_false_negatives": int(false_negatives.size),
                "surviving_properties_bitwise_equal": True,
                "backend_ids_exact": True,
                "rgb_bitwise_equal": True,
                "alpha_bitwise_equal": True,
                "cpu_reference_filter_ms": cpu_reference.filter_seconds * 1000.0,
                "fast_filter_ms": fast_materialization.filter_seconds * 1000.0,
                "selected_id_bytes": timings.selected_id_bytes,
                "retained_index_bytes": timings.retained_index_bytes,
                "active_property_bytes_before_filter": property_tensor_bytes(
                    baseline_active
                ),
                "active_property_bytes_after_filter": property_tensor_bytes(
                    fast_active
                ),
                "frame_local_property_bytes": int(
                    fast_active.scene_metrics.alod_frame_local_property_bytes
                ),
                "h2d_property_rows": int(fast_host.last_residency_frame.h2d_rows),
                "h2d_property_bytes": int(fast_host.last_residency_frame.h2d_bytes),
            }
        )
        timing_inputs.append(
            (baseline_selection.raster_global_node_ids, camera, settings)
        )

    for _ in range(warmup_repetitions):
        for selected_ids, camera, settings in timing_inputs:
            fast_conservative_pinhole_materialization(
                fast_host.device_culling, selected_ids, camera, settings
            )
    torch.cuda.synchronize(device)

    cpu_times: list[float] = []
    for _ in range(timing_repetitions):
        for selected_ids, camera, settings in timing_inputs:
            result = conservative_pinhole_materialization(
                fast_host.compact_culling, selected_ids, camera, settings
            )
            cpu_times.append(result.filter_seconds * 1000.0)

    stage_values: dict[str, list[float]] = {
        "total_filter_ms": [],
        "id_preparation_ms": [],
        "selected_id_h2d_ms": [],
        "metadata_gather_ms": [],
        "projection_culling_ms": [],
        "mask_compaction_ms": [],
        "retained_index_d2h_ms": [],
        "materialization_id_production_ms": [],
    }
    for _ in range(timing_repetitions):
        for selected_ids, camera, settings in timing_inputs:
            cpu_reference = conservative_pinhole_materialization(
                fast_host.compact_culling, selected_ids, camera, settings
            )
            result = fast_conservative_pinhole_materialization(
                fast_host.device_culling, selected_ids, camera, settings
            )
            np.testing.assert_array_equal(
                result.materialization_global_node_ids,
                cpu_reference.materialization_global_node_ids,
            )
            timing = result.timings
            stage_values["total_filter_ms"].append(result.filter_seconds * 1000.0)
            stage_values["id_preparation_ms"].append(
                timing.id_preparation_seconds * 1000.0
            )
            stage_values["selected_id_h2d_ms"].append(
                timing.selected_id_h2d_seconds * 1000.0
            )
            stage_values["metadata_gather_ms"].append(
                timing.metadata_gather_seconds * 1000.0
            )
            stage_values["projection_culling_ms"].append(
                timing.projection_seconds * 1000.0
            )
            stage_values["mask_compaction_ms"].append(
                timing.compaction_seconds * 1000.0
            )
            stage_values["retained_index_d2h_ms"].append(
                timing.retained_index_d2h_seconds * 1000.0
            )
            stage_values["materialization_id_production_ms"].append(
                timing.materialization_id_production_seconds * 1000.0
            )

    stage_stats = {key: _stats(values) for key, values in stage_values.items()}
    cpu_stats = _stats(cpu_times)
    compact_gpu_bytes = fast_host.compact_culling_gpu_bytes
    old_dynamic_cache_bytes = 525_000 * fast_host.host.property_bytes_per_row
    reduced_dynamic_cache_bytes = (
        fast_host.residency.capacity_rows * fast_host.host.property_bytes_per_row
    )
    return {
        "schema_version": 1,
        "status": PASS_STATUS,
        "asset": {"path": str(asset_path), "sha256": asset_sha256},
        "device": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gsplat_extension": extension_record,
        },
        "camera_count": len(rows),
        "semantic_selector_implementation_unchanged": True,
        "all_cpu_fast_materialization_ids_exact": True,
        "all_backend_false_negatives_zero": True,
        "total_backend_projection_survivors": total_backend_survivors,
        "all_surviving_properties_bitwise_equal": True,
        "all_backend_projection_ids_exact": True,
        "all_rgb_bitwise_equal": True,
        "all_alpha_bitwise_equal": True,
        "compact_metadata": {
            "fields": ["mean_float32[3]", "padded_max_scale_float32"],
            "bytes_per_gaussian": fast_host.compact_culling.bytes_per_row,
            "row_count": fast_host.compact_culling.row_count,
            "logical_cpu_bytes": fast_host.compact_culling.logical_bytes,
            "incremental_cpu_bytes": fast_host.compact_culling.incremental_bytes,
            "gpu_bytes": compact_gpu_bytes,
            "cpu_gpu_bitwise_equal": True,
        },
        "filter_timing": {
            "warmup_repetitions": warmup_repetitions,
            "measured_repetitions": timing_repetitions,
            "measured_calls": timing_repetitions * len(timing_inputs),
            "cpu_reference_filter_ms": cpu_stats,
            "fast_stages": stage_stats,
            "mean_speedup": cpu_stats["mean"]
            / stage_stats["total_filter_ms"]["mean"],
        },
        "selected_id_transfer": {
            "dtype": "int64",
            "mean_bytes_per_frame": float(
                np.mean([row["selected_id_bytes"] for row in rows])
            ),
            "maximum_bytes_per_frame": max(
                row["selected_id_bytes"] for row in rows
            ),
            "mean_retained_index_bytes_per_frame": float(
                np.mean([row["retained_index_bytes"] for row in rows])
            ),
            "host_lookup_consumes_cpu_ids": True,
            "raster_path_reuses_gpu_materialization_ids": True,
        },
        "persistent_memory_tradeoff": {
            "historical_525k_dynamic_property_bytes": old_dynamic_cache_bytes,
            "reduced_160k_dynamic_property_bytes": reduced_dynamic_cache_bytes,
            "dynamic_property_bytes_saved": (
                old_dynamic_cache_bytes - reduced_dynamic_cache_bytes
            ),
            "additional_compact_gpu_bytes": compact_gpu_bytes,
            "net_persistent_gpu_bytes_saved_vs_525k_cpu_filter": (
                old_dynamic_cache_bytes
                - reduced_dynamic_cache_bytes
                - compact_gpu_bytes
            ),
        },
        "reduced_cache_configuration": {
            "configuration_id": "ALOD-HOST-BOUNDED-CACHE-REDUCED",
            "capacity_rows": fast_host.residency.capacity_rows,
            "historical_525k_configuration_unchanged": True,
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--timing-repetitions", type=int, default=10)
    parser.add_argument(
        "--gsplat-extension",
        type=Path,
        default=(
            Path(os.environ["GS_FRAMEWORK_GSPLAT_EXTENSION"])
            if os.environ.get("GS_FRAMEWORK_GSPLAT_EXTENSION")
            else None
        ),
    )
    args = parser.parse_args()
    result = validate(
        args.asset,
        torch.device(args.device),
        warmup_repetitions=args.warmup_repetitions,
        timing_repetitions=args.timing_repetitions,
        gsplat_extension=args.gsplat_extension,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
