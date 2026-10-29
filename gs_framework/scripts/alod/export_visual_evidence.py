#!/usr/bin/env python3
"""Export the frozen Campus/Sci-Art ALoD visual-evidence package.

This script is intentionally an exporter, not a benchmark or selector study.
It reuses the frozen Stage-A camera records, semantic distributions, hierarchy
assets, and ALOD-EXT-PREVALENCE-ASSOCIATION-001 runtime configurations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.alod_resident_benchmark import _runtime_config  # noqa: E402
from scene_runtime.alod.host_scene import AlodHostScene  # noqa: E402
from scene_runtime.renderer import GsplatRenderer  # noqa: E402
from scripts.alod import alod_ext_prevalence_association as association  # noqa: E402
from scripts.alod.validate_fast_materialization_filter import _projection_ids  # noqa: E402


PACKAGE_ID = "ALOD-VISUAL-EVIDENCE-001"
DEFAULT_OUTPUT = REPO_ROOT / "benchmark_artifacts/alod_visual_evidence_001"
ASSOCIATION_ROOT = WORKSPACE_ROOT / "benchmark_artifacts/alod_ext_prevalence_association_001"
DATASETS = {
    "campus": {
        "display_name": "Campus",
        "label": "VALID-NONTRIVIAL-LOD",
        "oracle_root": WORKSPACE_ROOT
        / "runs/alod/oracles/ALOD-PREV-H3DGS-CAMPUS-CHUNK-01-ORACLE-001",
    },
    "sciart": {
        "display_name": "Sci-Art",
        "label": "VALID-WEAK-LOD",
        "oracle_root": WORKSPACE_ROOT
        / "runs/alod/oracles/ALOD-PREV-US3D-SCIART-01-ORACLE-001",
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def raw_array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def load_official_render(
    oracle_root: Path, semantic_manifest: dict[str, Any], index: int
) -> tuple[np.ndarray, dict[str, Any]]:
    entry = next(
        row
        for row in semantic_manifest["cameras"]
        if int(row["evaluation_sequence_index"]) == index
    )
    camera_dir = oracle_root / Path(entry["paths"]["camera.json"]).parent
    record_path = camera_dir / "optional-image.json"
    record = read_json(record_path)
    descriptor = record["rendered_rgb_chw"]
    payload_path = camera_dir / descriptor["payload_path"]
    if sha256_file(payload_path) != descriptor["payload_file_sha256"]:
        raise RuntimeError(f"official render payload file hash mismatch: {payload_path}")
    chw = np.load(payload_path, allow_pickle=False)
    if raw_array_sha256(chw) != descriptor["raw_cpu_c_order_sha256"]:
        raise RuntimeError(f"official render raw tensor hash mismatch: {payload_path}")
    if chw.dtype != np.float32 or list(chw.shape) != descriptor["shape"]:
        raise RuntimeError(f"official render payload metadata mismatch: {payload_path}")
    return np.ascontiguousarray(chw.transpose(1, 2, 0)), {
        "record_path": str(record_path),
        "payload_path": str(payload_path),
        "payload_file_sha256": descriptor["payload_file_sha256"],
        "raw_cpu_c_order_sha256": descriptor["raw_cpu_c_order_sha256"],
    }


def distribution_array(record: dict[str, Any], maximum_depth: int) -> np.ndarray:
    result = np.zeros(maximum_depth + 1, dtype=np.int64)
    for key, value in record["depth_distribution"].items():
        result[int(key)] = int(value)
    if int(result.sum()) != int(record["selected_count"]):
        raise RuntimeError("frozen depth distribution does not sum to selected_count")
    return result


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    left_p = left.astype(np.float64) / float(left.sum())
    right_p = right.astype(np.float64) / float(right.sum())
    return float(0.5 * np.abs(left_p - right_p).sum())


def select_records(
    response: dict[str, Any], manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cameras = response["cameras"]
    by_index = {int(row["evaluation_sequence_index"]): row for row in cameras}
    manifest_by_index = {
        int(row["evaluation_sequence_index"]): row for row in manifest["cameras"]
    }
    counts = np.asarray([int(row["selected_count"]) for row in cameras], dtype=np.int64)
    indices = np.asarray(
        [int(row["evaluation_sequence_index"]) for row in cameras], dtype=np.int64
    )
    frozen_median = float(response["criteria_v1"]["selected_count"]["median"])

    minimum = min(cameras, key=lambda row: (int(row["selected_count"]), int(row["evaluation_sequence_index"])))
    maximum = max(cameras, key=lambda row: (int(row["selected_count"]), -int(row["evaluation_sequence_index"])))
    median_order = np.lexsort((indices, np.abs(counts.astype(np.float64) - frozen_median)))
    median = cameras[int(median_order[0])]
    chosen = []
    for role, row in (("minimum", minimum), ("median", median), ("maximum", maximum)):
        index = int(row["evaluation_sequence_index"])
        enriched = dict(row)
        enriched["selection_role"] = role
        enriched["camera_id"] = manifest_by_index[index]["camera_stable_id"]
        chosen.append(enriched)

    maximum_depth = max(int(row["maximum_depth"]) for row in cameras)
    arrays = {
        int(row["evaluation_sequence_index"]): distribution_array(row, maximum_depth)
        for row in cameras
    }
    best_pair: tuple[int, int] | None = None
    best_tvd = -1.0
    for left_pos, left_index in enumerate(sorted(arrays)):
        for right_index in sorted(arrays)[left_pos + 1 :]:
            tvd = total_variation(arrays[left_index], arrays[right_index])
            pair = (left_index, right_index)
            if tvd > best_tvd + 1e-15 or (
                abs(tvd - best_tvd) <= 1e-15 and (best_pair is None or pair < best_pair)
            ):
                best_pair = pair
                best_tvd = tvd
    assert best_pair is not None
    frozen_tvd = float(
        response["criteria_v1"]["maximum_pairwise_depth_distribution_total_variation"]
    )
    if not np.isclose(best_tvd, frozen_tvd, rtol=0.0, atol=1e-15):
        raise RuntimeError(f"max-TVD reconstruction mismatch: {best_tvd} != {frozen_tvd}")
    pair_record = {
        "camera_indices": list(best_pair),
        "camera_ids": [manifest_by_index[index]["camera_stable_id"] for index in best_pair],
        "total_variation_distance": best_tvd,
        "distributions": [arrays[index].tolist() for index in best_pair],
    }
    return chosen, pair_record


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(to_uint8_rgb(image), mode="RGB").save(path, format="PNG", optimize=False)


def save_alpha(path: Path, alpha: np.ndarray) -> None:
    channel = np.rint(np.clip(alpha[..., 0], 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(channel, mode="L").save(path, format="PNG", optimize=False)


def save_difference(path: Path, difference: np.ndarray, maximum: float) -> None:
    canvas = Image.fromarray(to_uint8_rgb(difference), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=24)
    label = (
        "bitwise identical — max |Δ| = 0"
        if maximum == 0.0
        else f"absolute RGB difference — max |Δ| = {maximum:.8g}"
    )
    box = draw.textbbox((0, 0), label, font=font)
    padding = 12
    draw.rectangle(
        (0, 0, box[2] - box[0] + 2 * padding, box[3] - box[1] + 2 * padding),
        fill=(255, 255, 255),
    )
    draw.text((padding, padding), label, fill=(0, 0, 0), font=font)
    canvas.save(path, format="PNG", optimize=False)


def plot_histogram(path: Path, dataset_name: str, row: dict[str, Any]) -> None:
    depths = np.asarray(sorted(int(key) for key in row["depth_distribution"]), dtype=np.int64)
    counts = np.asarray([row["depth_distribution"][str(depth)] for depth in depths], dtype=np.int64)
    ratio = 100.0 * float(row["materialized_rows"]) / float(row["semantic_rows"])
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.bar(depths, counts, width=0.86, color="#4472C4")
    ax.set_xlabel("Hierarchy depth")
    ax.set_ylabel("Selected hierarchy rows")
    ax.set_title(f"{dataset_name} — {row['selection_role']} selected-count camera\n{row['camera_id']}")
    ax.grid(axis="y", alpha=0.22)
    text = (
        f"semantic hierarchy: {row['selected_count']:,}   "
        f"materialized: {row['materialized_rows']:,}   packed survivors: {row['packed_projection_survivors']:,}\n"
        f"materialized / semantic raster: {ratio:.2f}%   "
        f"depth mean / median / max: {row['mean_depth']:.3f} / {row['median_depth']:.0f} / {row['maximum_depth']}"
    )
    ax.text(0.01, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92})
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_dataset_summary(
    path: Path,
    dataset_name: str,
    label: str,
    rows: list[dict[str, Any]],
    released_images: dict[int, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 9.2), constrained_layout=True)
    for column, row in enumerate(rows):
        index = int(row["evaluation_sequence_index"])
        axes[0, column].imshow(np.clip(released_images[index], 0.0, 1.0))
        axes[0, column].axis("off")
        ratio = 100.0 * float(row["materialized_rows"]) / float(row["semantic_rows"])
        axes[0, column].set_title(
            f"{row['selection_role'].capitalize()} count — camera {index:04d}\n"
            f"{row['camera_id']}\n"
            f"semantic hierarchy {row['selected_count']:,}  |  mean depth {row['mean_depth']:.3f}\n"
            f"materialized / semantic raster {ratio:.2f}%",
            fontsize=10,
        )
        depths = np.asarray(sorted(int(key) for key in row["depth_distribution"]), dtype=np.int64)
        counts = np.asarray([row["depth_distribution"][str(depth)] for depth in depths], dtype=np.int64)
        axes[1, column].bar(depths, counts, width=0.86, color="#4472C4")
        axes[1, column].set_xlabel("Hierarchy depth")
        axes[1, column].grid(axis="y", alpha=0.22)
        if column == 0:
            axes[1, column].set_ylabel("Selected hierarchy rows")
    fig.suptitle(f"{dataset_name} — {label}", fontsize=18)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_tvd_comparison(path: Path, datasets: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 8.0), constrained_layout=True)
    for ax, key in zip(axes, ("campus", "sciart")):
        data = datasets[key]
        pair = data["max_tvd_pair"]
        for camera_id, raw, style in zip(
            pair["camera_ids"], pair["distributions"], ("-", "--")
        ):
            values = np.asarray(raw, dtype=np.float64)
            values /= values.sum()
            ax.plot(np.arange(values.size), values, style, linewidth=2.0, label=camera_id)
        ax.set_title(
            f"{data['display_name']} — maximum pairwise depth TVD = "
            f"{pair['total_variation_distance']:.6f}"
        )
        ax.set_xlabel("Hierarchy depth")
        ax.set_ylabel("Proportion of selected rows")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=9)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


@torch.no_grad()
def render_mode(
    *,
    spec: dict[str, Any],
    cameras: Any,
    selected_indices: set[int],
    mode: str,
    capacity: int | None,
    device: torch.device,
) -> dict[int, dict[str, Any]]:
    config = association.build_config(spec, mode, capacity)
    runtime = _runtime_config(config)
    scene = AlodHostScene(
        Path(spec["asset_path"]), device, runtime, association.host_config(mode, capacity)
    )
    renderer = GsplatRenderer(device)

    def render_index(index: int) -> tuple[Any, Any, np.ndarray]:
        camera_spec = cameras.cameras[index]
        camera = association.make_exact_frame_camera(camera_spec, device)
        settings = association.make_render_settings(config, camera_spec, sh_degree=1)
        active = scene.prepare_frame(camera, settings)
        result = renderer.render(active, camera, settings, measure_time=False)
        survivors = _projection_ids(result)
        return active, result, survivors

    # Match the frozen association workload: first ten canonical cameras warm
    # the residency state, then the measured sequence restarts at camera zero.
    for index in range(10):
        render_index(index)
    outputs: dict[int, dict[str, Any]] = {}
    for index in range(max(selected_indices) + 1):
        active, result, survivors = render_index(index)
        if index in selected_indices:
            outputs[index] = {
                "image": result.image.detach().cpu().numpy().copy(),
                "alpha": result.alpha.detach().cpu().numpy().copy(),
                "packed_survivor_ids": survivors.copy(),
                "semantic_rows": int(active.scene_metrics.alod_semantic_selected_rows),
                "materialized_rows": int(active.scene_metrics.alod_materialization_rows),
            }
    del scene, renderer
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def enrich_with_correctness(
    selected: list[dict[str, Any]], correctness: dict[str, Any]
) -> None:
    by_index = {int(row["camera_index"]): row for row in correctness["cameras"]}
    for row in selected:
        frozen = by_index[int(row["evaluation_sequence_index"])]
        row.update(
            {
                "semantic_rows": int(frozen["semantic_rows"]),
                "semantic_hierarchy_rows": int(frozen["semantic_hierarchy_rows"]),
                "materialized_rows": int(frozen["materialized_rows"]),
                "materialized_hierarchy_rows": int(frozen["materialized_hierarchy_rows"]),
                "packed_projection_survivors": int(frozen["packed_projection_survivors"]),
            }
        )
        if int(row["selected_count"]) != int(row["semantic_hierarchy_rows"]):
            raise RuntimeError("Stage-A and association semantic hierarchy counts differ")


def relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"refusing to modify non-empty immutable package: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in (*DATASETS, "comparison"):
        (output / name).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("canonical visual export requires CUDA")
    study = association.load_study(association.DEFAULT_CONFIG)
    package: dict[str, Any] = {
        "schema_version": 1,
        "package_id": PACKAGE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS-ALOD-VISUAL-EVIDENCE",
        "purpose": (
            "Explanatory visualizations derived from already-frozen experiments; "
            "not a new scientific outcome or camera-selection study."
        ),
        "selection_rule": {
            "minimum": "minimum frozen Stage-A selected hierarchy-row count; ties use lowest canonical camera index",
            "median": (
                "minimum absolute distance to the frozen Stage-A numerical median selected count; "
                "ties use lowest canonical camera index"
            ),
            "maximum": "maximum frozen Stage-A selected hierarchy-row count; ties use lowest canonical camera index",
            "visual_appearance_used": False,
        },
        "rendering": {
            "renderer": "gs_framework.scene_runtime.renderer.GsplatRenderer using gsplat rasterization",
            "released": "Released Evaluator (No Cache)",
            "released_base_configuration_id": "ALOD-REPRO-HOST-NOCACHE",
            "extension": "ALoD-EXT CUDA Bounded Cache",
            "extension_base_configuration_id": "ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL",
            "campaign": "ALOD-EXT-PREVALENCE-ASSOCIATION-001",
            "warmup_frames": 10,
            "measured_sequence_restarts_at_camera_zero": True,
            "new_performance_benchmark": False,
            "png_encoding": "8-bit lossless PNG derived by round(clamp(float RGB, 0, 1) * 255)",
            "parity_check_domain": "native float render tensors before PNG quantization",
        },
        "datasets": {},
        "files": [],
    }

    saved_for_parity: dict[str, dict[str, Path]] = {}
    saved_for_official_comparison: dict[str, dict[str, Path]] = {}
    tvd_data: dict[str, dict[str, Any]] = {}
    for key, metadata in DATASETS.items():
        spec = association.dataset_spec(study, key)
        association.verify_inputs(spec)
        oracle_root = metadata["oracle_root"]
        response_path = oracle_root / "validation/prevalence-lod-response.json"
        manifest_path = oracle_root / "semantic-manifest.json"
        response = read_json(response_path)
        semantic_manifest = read_json(manifest_path)
        selected, max_tvd_pair = select_records(response, semantic_manifest)
        correctness_path = ASSOCIATION_ROOT / f"checks/{key}_correctness.json"
        correctness = read_json(correctness_path)
        enrich_with_correctness(selected, correctness)
        capacity = int(correctness["capacity"]["bounded_capacity_rows"])

        released_config = association.build_config(spec, "nocache")
        cameras = association.configured_cameras(spec, released_config)
        selected_indices = {int(row["evaluation_sequence_index"]) for row in selected}
        print(f"[{key}] rendering Released Evaluator", flush=True)
        released = render_mode(
            spec=spec,
            cameras=cameras,
            selected_indices=selected_indices,
            mode="nocache",
            capacity=None,
            device=device,
        )
        print(f"[{key}] rendering ALoD-EXT bounded", flush=True)
        bounded = render_mode(
            spec=spec,
            cameras=cameras,
            selected_indices=selected_indices,
            mode="bounded",
            capacity=capacity,
            device=device,
        )

        dataset_dir = output / key
        parity_camera_paths: dict[str, Path] | None = None
        official_camera_paths: dict[str, Path] | None = None
        for row in selected:
            index = int(row["evaluation_sequence_index"])
            stem = f"camera_{index:04d}_{row['selection_role']}"
            faithful = released[index]
            extension = bounded[index]
            if faithful["semantic_rows"] != row["semantic_rows"]:
                raise RuntimeError("released semantic count differs from frozen correctness record")
            if extension["materialized_rows"] != row["materialized_rows"]:
                raise RuntimeError("bounded materialized count differs from frozen correctness record")
            if faithful["packed_survivor_ids"].size != row["packed_projection_survivors"]:
                raise RuntimeError("packed survivor count differs from frozen correctness record")
            if not np.array_equal(faithful["packed_survivor_ids"], extension["packed_survivor_ids"]):
                raise RuntimeError("released and bounded packed survivor IDs differ")
            delta = np.abs(extension["image"].astype(np.float32) - faithful["image"].astype(np.float32))
            maximum = float(delta.max())
            nonzero = int(np.count_nonzero(delta))
            alpha_delta = np.abs(extension["alpha"].astype(np.float32) - faithful["alpha"].astype(np.float32))
            alpha_maximum = float(alpha_delta.max())
            alpha_nonzero = int(np.count_nonzero(alpha_delta))
            if maximum != 0.0 or nonzero != 0 or alpha_maximum != 0.0 or alpha_nonzero != 0:
                raise RuntimeError(f"bitwise render parity failed for {key} camera {index}")

            released_path = dataset_dir / f"{stem}_released_rgb.png"
            bounded_path = dataset_dir / f"{stem}_alod_ext_bounded_rgb.png"
            difference_path = dataset_dir / f"{stem}_absolute_rgb_difference.png"
            alpha_path = dataset_dir / f"{stem}_released_alpha.png"
            histogram_path = dataset_dir / f"{stem}_depth_histogram.png"
            official, official_provenance = load_official_render(
                oracle_root, semantic_manifest, index
            )
            official_delta = np.abs(
                extension["image"].astype(np.float32) - official.astype(np.float32)
            )
            official_maximum = float(official_delta.max())
            official_mean = float(official_delta.mean(dtype=np.float64))
            official_nonzero = int(np.count_nonzero(official_delta))
            official_path = dataset_dir / f"{stem}_official_source_rgb.png"
            official_difference_path = (
                dataset_dir / f"{stem}_official_vs_alod_ext_absolute_rgb_difference.png"
            )
            save_rgb(released_path, faithful["image"])
            save_rgb(bounded_path, extension["image"])
            save_difference(difference_path, delta, maximum)
            save_alpha(alpha_path, faithful["alpha"])
            save_rgb(official_path, official)
            save_difference(official_difference_path, official_delta, official_maximum)
            row.update(
                {
                    "materialized_over_semantic_percent": 100.0
                    * float(row["materialized_rows"])
                    / float(row["semantic_rows"]),
                    "rgb_parity": {
                        "bitwise_equal": True,
                        "maximum_absolute_difference": maximum,
                        "nonzero_element_count": nonzero,
                    },
                    "alpha_parity": {
                        "bitwise_equal": True,
                        "maximum_absolute_difference": alpha_maximum,
                        "nonzero_element_count": alpha_nonzero,
                    },
                    "official_source_vs_alod_ext": {
                        "bitwise_equal": bool(np.array_equal(official, extension["image"])),
                        "maximum_absolute_difference": official_maximum,
                        "mean_absolute_difference": official_mean,
                        "nonzero_element_count": official_nonzero,
                        "comparison_domain": "native float32 RGB tensors before PNG quantization",
                        "official_render": official_provenance,
                    },
                    "outputs": {
                        "released_rgb": relative(released_path, output),
                        "alod_ext_bounded_rgb": relative(bounded_path, output),
                        "absolute_rgb_difference": relative(difference_path, output),
                        "released_alpha": relative(alpha_path, output),
                        "depth_histogram": relative(histogram_path, output),
                        "official_source_rgb": relative(official_path, output),
                        "official_vs_alod_ext_absolute_rgb_difference": relative(
                            official_difference_path, output
                        ),
                    },
                }
            )
            plot_histogram(histogram_path, metadata["display_name"], row)
            if row["selection_role"] == "median":
                parity_camera_paths = {
                    "released": released_path,
                    "bounded": bounded_path,
                    "difference": difference_path,
                }
                official_camera_paths = {
                    "official": official_path,
                    "bounded": bounded_path,
                    "difference": official_difference_path,
                }
        assert parity_camera_paths is not None
        assert official_camera_paths is not None
        saved_for_parity[key] = parity_camera_paths
        saved_for_official_comparison[key] = official_camera_paths
        summary_path = dataset_dir / f"{key}_summary.png"
        plot_dataset_summary(
            summary_path,
            metadata["display_name"],
            metadata["label"],
            selected,
            {index: payload["image"] for index, payload in released.items()},
        )
        package["datasets"][key] = {
            "display_name": metadata["display_name"],
            "stage_a_label": metadata["label"],
            "dataset_id": spec["dataset_id"],
            "hierarchy_asset": {
                "path": spec["asset_path"],
                "sha256": spec["asset_sha256"],
            },
            "frozen_camera_path": {
                "path": spec["camera_path"],
                "sha256": spec["camera_path_sha256"],
            },
            "semantic_manifest": {
                "path": spec["oracle_manifest"],
                "sha256": spec["oracle_manifest_sha256"],
            },
            "stage_a_response": {
                "path": str(response_path),
                "sha256": sha256_file(response_path),
            },
            "association_correctness": {
                "path": str(correctness_path),
                "sha256": sha256_file(correctness_path),
            },
            "resolution": [int(spec["width"]), int(spec["height"])],
            "bounded_capacity_rows": capacity,
            "representative_cameras": selected,
            "max_tvd_pair": {
                key: value for key, value in max_tvd_pair.items() if key != "distributions"
            },
            "summary_figure": relative(summary_path, output),
        }
        tvd_data[key] = {
            "display_name": metadata["display_name"],
            "max_tvd_pair": max_tvd_pair,
        }
        del released, bounded
        gc.collect()

    parity_path = output / "comparison/alod_ext_rgb_parity.png"
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8), constrained_layout=True)
    for row_number, key in enumerate(("campus", "sciart")):
        camera = next(
            item
            for item in package["datasets"][key]["representative_cameras"]
            if item["selection_role"] == "median"
        )
        for column, field in enumerate(("released", "bounded", "difference")):
            image = Image.open(saved_for_parity[key][field])
            axes[row_number, column].imshow(image, cmap="gray" if field == "difference" else None)
            axes[row_number, column].axis("off")
        axes[row_number, 0].set_title(
            f"{DATASETS[key]['display_name']} — Released Evaluator RGB\n{camera['camera_id']}"
        )
        axes[row_number, 1].set_title("ALoD-EXT bounded RGB")
        axes[row_number, 2].set_title("Absolute RGB difference\nbitwise identical — max |Δ| = 0")
    fig.suptitle("ALoD-EXT changes execution/materialization, not rendered output", fontsize=17)
    fig.savefig(parity_path, dpi=200, facecolor="white")
    plt.close(fig)

    official_comparison_path = output / "comparison/official_source_vs_alod_ext.png"
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8), constrained_layout=True)
    for row_number, key in enumerate(("campus", "sciart")):
        camera = next(
            item
            for item in package["datasets"][key]["representative_cameras"]
            if item["selection_role"] == "median"
        )
        comparison = camera["official_source_vs_alod_ext"]
        for column, field in enumerate(("official", "bounded", "difference")):
            image = Image.open(saved_for_official_comparison[key][field])
            axes[row_number, column].imshow(
                image, cmap="gray" if field == "difference" else None
            )
            axes[row_number, column].axis("off")
        axes[row_number, 0].set_title(
            f"{DATASETS[key]['display_name']} — pinned official evaluator RGB\n"
            f"{camera['camera_id']}"
        )
        axes[row_number, 1].set_title("ALoD-EXT bounded RGB")
        axes[row_number, 2].set_title(
            "Absolute RGB difference (unamplified)\n"
            f"max |Δ| = {comparison['maximum_absolute_difference']:.8g}, "
            f"mean |Δ| = {comparison['mean_absolute_difference']:.8g}\n"
            f"nonzero elements = {comparison['nonzero_element_count']:,}"
        )
    fig.suptitle("Pinned official ALoD evaluator versus ALoD-EXT bounded", fontsize=17)
    fig.savefig(official_comparison_path, dpi=200, facecolor="white")
    plt.close(fig)

    tvd_path = output / "comparison/depth_tvd_comparison.png"
    plot_tvd_comparison(tvd_path, tvd_data)
    package["comparison_figures"] = {
        "alod_ext_rgb_parity": relative(parity_path, output),
        "official_source_vs_alod_ext": relative(official_comparison_path, output),
        "depth_tvd_comparison": relative(tvd_path, output),
    }

    for path in sorted(output.rglob("*.png")):
        with Image.open(path) as image:
            width, height = image.size
        package["files"].append(
            {
                "path": relative(path, output),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "pixel_size": [width, height],
            }
        )
    (output / "manifest.json").write_text(canonical_json(package), encoding="utf-8")
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
