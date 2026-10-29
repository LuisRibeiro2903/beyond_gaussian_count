#!/usr/bin/env python3
"""Run the pinned official cold evaluator and assemble a selection oracle.

The script applies an observational patch to a temporary source archive.  It
never edits the canonical official checkout, environments, dataset, or Run005.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_COMMIT = "73f547341547ae021ff036ad74fdf73a1240b450"
ASSET_SHA256 = "6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c"
CAMERA_INDICES = (0, 2, 4, 6, 7, 8, 9, 10, 12, 14)
ORACLE_ID = "ALOD-RUN005-SELECTION-ORACLE-001"
HOOK_PATH = REPO_ROOT / "scripts/alod/official_selection_oracle_hooks.py"
ENVIRONMENT_MANIFEST = REPO_ROOT / "environments/alod-official/build-manifest.json"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, default=Path("/home/luis/tese/LoDOfGaussians"))
    parser.add_argument("--official-python", type=Path, default=Path("/home/luis/miniconda3/envs/alod_official/bin/python"))
    parser.add_argument("--asset", type=Path, default=Path("/home/luis/tese/runs/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/ALOD-SMOKE-TRAIN-005/output/final.dhier"))
    parser.add_argument("--dataset", type=Path, default=Path("/home/luis/tese/data/alod/ALOD-ORACLE-MC-AERIAL-TEST-01"))
    parser.add_argument("--frozen-cameras", type=Path, default=REPO_ROOT / "datasets/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/evaluation-cameras.json")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/alod/ALOD-ORACLE-MC-AERIAL-TEST-01-capped.json")
    parser.add_argument("--patch", type=Path, default=REPO_ROOT / "patches/alod/ALOD-RUN005-SELECTION-ORACLE-INSTRUMENTATION.patch")
    parser.add_argument("--output-root", type=Path, default=Path("/home/luis/tese/runs/alod/oracles") / ORACLE_ID)
    parser.add_argument("--oracle-id", default=ORACLE_ID)
    parser.add_argument("--asset-sha256", default=ASSET_SHA256)
    parser.add_argument("--asset-rows", type=int, default=1_099_999)
    parser.add_argument("--skybox-rows", type=int, default=100_000)
    parser.add_argument("--root-global-node-id", type=int, default=100_000)
    parser.add_argument("--sh-degree", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--dataset-id", default="ALOD-ORACLE-MC-AERIAL-TEST-01")
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--dataset-manifest-sha256")
    parser.add_argument("--camera-index", type=int, action="append", default=[])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def git_value(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()


def substantive_git_status(source: Path) -> list[str]:
    """Return checkout changes other than CPython bytecode artifacts."""
    return [
        line
        for line in git_value(source, "status", "--porcelain").splitlines()
        if "__pycache__/" not in line and not line.endswith(".pyc")
    ]


def prepare_source(source: Path, patch: Path, config: Path, destination: Path) -> None:
    archive = destination.parent / "official-source.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", f"--output={archive}", OFFICIAL_COMMIT],
        cwd=source,
        check=True,
    )
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r") as handle:
        handle.extractall(destination, filter="fully_trusted")
    archive.unlink()
    subprocess.run(["git", "apply", str(patch)], cwd=destination, check=True)
    shutil.copy2(config, destination / "configs" / "oracle-selection.json")


def stable_id(camera: dict[str, Any]) -> str:
    return (
        f"eval-{int(camera['evaluation_sequence_index']):04d}-"
        f"img-{int(camera['registered_image_id']):06d}-"
        f"cam-{int(camera['camera_id']):04d}"
    )


def official_environment(base: dict[str, str], *, source: Path, output: Path, camera_index: int, asset: Path, frozen: Path, patch_sha: str, payload_mode: str, hspt_payloads: bool, asset_sha256: str, asset_rows: int, skybox_rows: int, root_global_node_id: int, sh_degree: int) -> dict[str, str]:
    environment = dict(base)
    conda_root = Path("/home/luis/miniconda3/envs/alod_official")
    task_tmp = Path("/tmp") / f"alod-selection-oracle-{os.getuid()}"
    extension_dir = Path("/tmp") / f"alod-selection-oracle-torch-ext-{os.getuid()}"
    task_tmp.mkdir(parents=True, exist_ok=True, mode=0o700)
    extension_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    environment.update(
        {
            "TMPDIR": str(task_tmp),
            "TEMP": str(task_tmp),
            "TMP": str(task_tmp),
            "CUDA_HOME": str(conda_root),
            "PATH": f"{conda_root / 'bin'}:{environment.get('PATH', '')}",
            "TORCH_CUDA_ARCH_LIST": "12.0",
            "MAX_JOBS": "2",
            "CC": str(conda_root / "bin/x86_64-conda-linux-gnu-gcc"),
            "CXX": str(conda_root / "bin/x86_64-conda-linux-gnu-g++"),
            "CUDACXX": str(conda_root / "bin/nvcc"),
            "CPATH": f"{conda_root / 'include'}:{conda_root / 'targets/x86_64-linux/include'}",
            "LD_LIBRARY_PATH": f"{conda_root / 'lib'}:{conda_root / 'targets/x86_64-linux/lib'}",
            "TORCH_EXTENSIONS_DIR": str(extension_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": f"{REPO_ROOT / 'scripts/alod'}:{REPO_ROOT}:{source}",
            "ALOD_ORACLE_OUTPUT_DIR": str(output),
            "ALOD_ORACLE_CAMERA_INDEX": str(camera_index),
            "ALOD_ORACLE_FROZEN_CAMERAS": str(frozen),
            "ALOD_ORACLE_ASSET": str(asset),
            "ALOD_ORACLE_SOURCE_COMMIT": OFFICIAL_COMMIT,
            "ALOD_ORACLE_PATCH_SHA256": patch_sha,
            "ALOD_ORACLE_PAYLOAD_MODE": payload_mode,
            "ALOD_ORACLE_HSPT_PAYLOADS": "1" if hspt_payloads else "0",
            "ALOD_ORACLE_SCAFFOLD_FILE": "",
            "ALOD_ORACLE_EXPECTED_ASSET_SHA256": asset_sha256,
            "ALOD_ORACLE_EXPECTED_ROWS": str(asset_rows),
            "ALOD_ORACLE_EXPECTED_SKYBOX_ROWS": str(skybox_rows),
            "ALOD_ORACLE_EXPECTED_ROOT_GLOBAL_ID": str(root_global_node_id),
            "ALOD_ORACLE_EXPECTED_SH_DEGREE": str(sh_degree),
        }
    )
    return environment


def run_camera(
    *,
    python: Path,
    source: Path,
    dataset: Path,
    asset: Path,
    frozen: Path,
    patch_sha: str,
    output: Path,
    camera_index: int,
    payload_mode: str,
    hspt_payloads: bool,
    asset_sha256: str,
    asset_rows: int,
    skybox_rows: int,
    root_global_node_id: int,
    sh_degree: int,
    resolution: int,
) -> None:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite attempt directory {output}")
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="alod-oracle-model-") as model_tmp:
        command = [
            str(python),
            str(source / "eval_hierarchy.py"),
            "--source_path",
            str(dataset),
            "--model_path",
            model_tmp,
            "--images",
            "images",
            "--alpha_masks",
            "",
            "--depths",
            "",
            "--resolution",
            str(resolution),
            "--eval",
            "--hierarchy",
            str(asset),
            "--scaffold_file",
            "",
            "--config",
            "oracle-selection.json",
        ]
        environment = official_environment(
            os.environ,
            source=source,
            output=output,
            camera_index=camera_index,
            asset=asset,
            frozen=frozen,
            patch_sha=patch_sha,
            payload_mode=payload_mode,
            hspt_payloads=hspt_payloads,
            asset_sha256=asset_sha256,
            asset_rows=asset_rows,
            skybox_rows=skybox_rows,
            root_global_node_id=root_global_node_id,
            sh_degree=sh_degree,
        )
        result = subprocess.run(
            command,
            cwd=source,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    (output / "official.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"official camera {camera_index} failed with {result.returncode}; see {output / 'official.log'}"
        )
    if not (output / "complete.json").is_file():
        raise RuntimeError(f"official camera {camera_index} did not finalize its oracle")


def _raw_hashes(record: dict[str, Any], key: str = "arrays") -> dict[str, str]:
    return {
        name: descriptor["raw_cpu_c_order_sha256"]
        for name, descriptor in record[key].items()
    }


def attempt_signature(path: Path) -> dict[str, Any]:
    load = lambda name: json.loads((path / name).read_text(encoding="utf-8"))
    hspt = load("hspt.json")
    camera = load("camera.json")
    upper = load("upper-traversal.json")
    cut = load("spt-cut.json")
    working = load("working-set.json")
    pre = load("pre-raster.json")
    image = load("optional-image.json")
    return {
        "hspt_configuration": hspt["configuration"],
        "hspt_counts": hspt["counts"],
        "hspt_arrays": _raw_hashes(hspt),
        "camera_identity": camera["identity"],
        "camera_effective_matrix_hashes": {
            name: camera["effective_official_runtime_camera"][name]["raw_cpu_c_order_sha256"]
            for name in ("world_view_transform", "projection_matrix", "full_projection_transform")
        },
        "upper_counts": upper["counts"],
        "upper_identities": upper["identity_summaries"],
        "upper_arrays": _raw_hashes(upper),
        "cut_counts": cut["counts"],
        "cut_depth": cut["selected_depth"],
        "cut_identities": cut["identity_summaries"],
        "cut_arrays": _raw_hashes(cut),
        "source_rows": {
            name: descriptor["raw_cpu_c_order_sha256"]
            for name, descriptor in cut["selected_source_rows"]["arrays"].items()
        },
        "working_ids": working["global_node_ids"],
        "working_correspondence": working["row_correspondence"],
        "pre_counts": pre["counts"],
        "pre_tensors": {
            name: descriptor["raw_cpu_c_order_sha256"]
            for name, descriptor in pre["tensors"].items()
        },
        "pre_bundle": pre["pre_raster_bundle_sha256"],
        "render": image["rendered_rgb_chw"]["raw_cpu_c_order_sha256"],
    }


def compare_attempts(paths: list[Path]) -> str:
    signatures = [attempt_signature(path) for path in paths]
    baseline = signatures[0]
    for index, signature in enumerate(signatures[1:], start=2):
        if signature != baseline:
            raise RuntimeError(f"cold determinism mismatch in repetition {index}")
    return hashlib.sha256(
        json.dumps(baseline, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_selected_ids(camera_dir: Path) -> np.ndarray:
    record = json.loads((camera_dir / "spt-cut.json").read_text(encoding="utf-8"))
    relative = record["arrays"]["selected_hierarchy_global_node_ids"]["payload_path"]
    return np.load(camera_dir / relative, allow_pickle=False).astype(np.int64, copy=False)


def depth_distribution(record: dict[str, Any]) -> dict[int, int]:
    return {int(key): int(value) for key, value in record["selected_depth"]["distribution"].items()}


def depth_tvd(first: dict[int, int], second: dict[int, int]) -> float:
    first_total = sum(first.values())
    second_total = sum(second.values())
    keys = set(first) | set(second)
    return 0.5 * sum(
        abs(first.get(key, 0) / first_total - second.get(key, 0) / second_total)
        for key in keys
    )


def histogram_median(histogram: dict[int, int]) -> float:
    total = sum(histogram.values())
    middle = (total - 1) / 2.0
    cumulative = 0
    lower = upper = None
    for value, count in sorted(histogram.items()):
        previous = cumulative
        cumulative += count
        if lower is None and previous <= int(middle) < cumulative:
            lower = value
        if previous <= int(middle + 0.5) < cumulative:
            upper = value
            break
    if lower is None or upper is None:
        raise RuntimeError("cannot compute median of an empty depth histogram")
    return (lower + upper) / 2.0


def summarize_cameras(camera_dirs: list[Path]) -> tuple[dict[str, Any], str]:
    sets = [set(int(value) for value in load_selected_ids(path)) for path in camera_dirs]
    cuts = [json.loads((path / "spt-cut.json").read_text(encoding="utf-8")) for path in camera_dirs]
    counts = [int(record["counts"]["selected_hierarchy_nodes"]) for record in cuts]
    maximum_depths = [int(record["selected_depth"]["maximum"]) for record in cuts]
    depth_distributions = [depth_distribution(record) for record in cuts]
    jaccard = []
    maximum_depth_tvd = 0.0
    for left, left_set in enumerate(sets):
        row = []
        for right, right_set in enumerate(sets):
            union = left_set | right_set
            row.append(1.0 if not union else len(left_set & right_set) / len(union))
            maximum_depth_tvd = max(
                maximum_depth_tvd,
                depth_tvd(depth_distributions[left], depth_distributions[right]),
            )
        jaccard.append(row)
    median_count = float(statistics.median(counts))
    relative_count_range = (max(counts) - min(counts)) / median_count if median_count else 0.0
    pairwise_jaccard = [
        jaccard[left][right]
        for left in range(len(jaccard))
        for right in range(left + 1, len(jaccard))
    ]
    # A one-camera export is supported for bounded instrumentation smoke tests.
    # With no camera pair, the only defensible overlap value is identity.
    minimum_pairwise_jaccard = min(pairwise_jaccard, default=1.0)
    all_sets_equal = all(value == sets[0] for value in sets[1:])
    all_depth_distributions_equal = all(
        value == depth_distributions[0] for value in depth_distributions[1:]
    )
    meaningful_set_variation = minimum_pairwise_jaccard <= 0.95
    material_count_variation = relative_count_range >= 0.05
    material_depth_variation = (
        max(maximum_depths) > min(maximum_depths) or maximum_depth_tvd >= 0.05
    )
    if all_sets_equal and len(set(counts)) == 1 and all_depth_distributions_equal:
        classification = "DEGENERATE-LOD-ORACLE"
    elif meaningful_set_variation and material_count_variation and material_depth_variation:
        classification = "NONTRIVIAL-LOD-ORACLE"
    else:
        classification = "WEAK-LOD-ORACLE"
    distinct_depths = sorted({depth for distribution in depth_distributions for depth in distribution})
    aggregate_depth_distribution: dict[int, int] = {}
    for distribution in depth_distributions:
        for depth, count in distribution.items():
            aggregate_depth_distribution[depth] = (
                aggregate_depth_distribution.get(depth, 0) + count
            )
    summary = {
        "classification": classification,
        "criteria_v1": {
            "meaningful_set_variation": "minimum pairwise Jaccard <= 0.95",
            "material_count_variation": "(maximum-minimum selected count)/median selected count >= 0.05",
            "material_depth_variation": "maximum selected depth varies or maximum pairwise depth-distribution TVD >= 0.05",
            "nontrivial": "all three variation predicates are true",
            "degenerate": "all selected sets, counts, and depth distributions are identical",
            "weak": "neither nontrivial nor degenerate",
        },
        "evidence": {
            "selected_count": {
                "minimum": min(counts),
                "median": median_count,
                "maximum": max(counts),
                "relative_range": relative_count_range,
            },
            "maximum_selected_depth": {
                "minimum": min(maximum_depths),
                "median": float(statistics.median(maximum_depths)),
                "maximum": max(maximum_depths),
            },
            "distinct_selected_hierarchy_depths": distinct_depths,
            "distinct_selected_hierarchy_depth_count": len(distinct_depths),
            "selected_depth_across_all_selected_nodes": {
                "minimum": min(aggregate_depth_distribution),
                "median": histogram_median(aggregate_depth_distribution),
                "maximum": max(aggregate_depth_distribution),
                "distribution": {
                    str(key): aggregate_depth_distribution[key]
                    for key in sorted(aggregate_depth_distribution)
                },
            },
            "minimum_pairwise_selected_set_jaccard": minimum_pairwise_jaccard,
            "maximum_pairwise_depth_distribution_total_variation": maximum_depth_tvd,
            "all_selected_sets_equal": all_sets_equal,
            "pairwise_selected_set_jaccard": jaccard,
            "sequential_selected_set_jaccard": [jaccard[index][index + 1] for index in range(len(jaccard) - 1)],
        },
    }
    return summary, classification


def build_manifests(output_root: Path, camera_records: list[dict[str, Any]], hspt_reference: Path, patch_sha: str, asset: Path, frozen: Path, config: Path, python: Path, classification: dict[str, Any], *, oracle_id: str, asset_sha256: str, asset_rows: int, skybox_rows: int, root_global_node_id: int, sh_degree: int, dataset_id: str, dataset_manifest: Path | None, dataset_manifest_sha256: str | None) -> None:
    hspt = json.loads(hspt_reference.read_text(encoding="utf-8"))
    semantic = {
        "schema_version": "alod_selection_oracle_manifest.v1",
        "oracle_id": oracle_id,
        "global_node_id_contract": "serialized .dhier row index",
        "asset": {
            "path": str(asset),
            "sha256": asset_sha256,
            "rows": asset_rows,
            "node_rows": asset_rows,
            "sh_degree": sh_degree,
            "skybox_rows": skybox_rows,
            "hierarchy_rows": asset_rows - skybox_rows,
            "root_global_node_id": root_global_node_id,
        },
        "official_runtime": {
            "source_path": "/home/luis/tese/LoDOfGaussians",
            "source_commit": OFFICIAL_COMMIT,
            "authoritative_entrypoint": "eval_hierarchy.py",
            "cold_reset_boundary": "fresh OS process, fresh model, rebuilt HSPT, no prior SPT cut or cache",
            "instrumentation_patch_sha256": patch_sha,
            "instrumentation_hook_sha256": sha256_file(HOOK_PATH),
            "run005_correctness_patch_sha256": "a7412fff7315153774c93b504105804c2a182308807a2c6ec8de34672b9cd7d6",
            "python": str(python),
            "environment_history_sha256": sha256_file(Path("/home/luis/miniconda3/envs/alod_official/conda-meta/history")),
            "environment_build_manifest": str(ENVIRONMENT_MANIFEST),
            "environment_build_manifest_sha256": sha256_file(ENVIRONMENT_MANIFEST),
            "method_modes_not_conflated": {
                "evaluation": "cold/no useful inter-camera cache; selected rows rebuilt per camera",
                "viewer": "previous-working-set reuse; not exported here",
                "fine_training": "bounded retained/write-back cache; not exported here",
            },
        },
        "dataset": {
            "id": dataset_id,
            "manifest": str(dataset_manifest) if dataset_manifest else None,
            "manifest_sha256": dataset_manifest_sha256,
            "frozen_camera_manifest": str(frozen),
            "frozen_camera_manifest_sha256": sha256_file(frozen),
            "configuration": str(config),
            "configuration_sha256": sha256_file(config),
        },
        "hspt": {
            "path": str(hspt_reference.relative_to(output_root)),
            "json_sha256": sha256_file(hspt_reference),
            "configuration": hspt["configuration"],
            "counts": hspt["counts"],
            "structural_array_hashes": _raw_hashes(hspt),
        },
        "frustum": {
            "active_planes": ["left", "right", "bottom", "top"],
            "inactive_planes": ["near", "far"],
            "bounding_sphere_construction": False,
            "scale_derived_bounds": True,
        },
        "cameras": camera_records,
        "lod_oracle_classification": classification,
    }
    semantic_path = output_root / "semantic-manifest.json"
    semantic_path.write_text(canonical_json(semantic), encoding="utf-8")
    manifest = dict(semantic)
    manifest["creation_timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["deterministic_semantic_manifest"] = {
        "path": "semantic-manifest.json",
        "sha256": sha256_file(semantic_path),
    }
    (output_root / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")


def main() -> int:
    args = parse_args()
    source = args.official_source.resolve()
    python = args.official_python.resolve()
    asset = args.asset.resolve()
    dataset = args.dataset.resolve()
    frozen_path = args.frozen_cameras.resolve()
    config = args.config.resolve()
    patch = args.patch.resolve()
    output_root = args.output_root.resolve()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if git_value(source, "rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise RuntimeError("canonical official source is not at the pinned commit")
    if substantive_git_status(source):
        raise RuntimeError("canonical official source has substantive changes")
    if sha256_file(asset) != args.asset_sha256:
        raise RuntimeError("hierarchy SHA-256 mismatch")
    if args.asset_rows <= args.skybox_rows:
        raise ValueError("asset rows must exceed skybox rows")
    if args.dataset_manifest is not None:
        args.dataset_manifest = args.dataset_manifest.resolve()
        actual_dataset_sha = sha256_file(args.dataset_manifest)
        if args.dataset_manifest_sha256 and actual_dataset_sha != args.dataset_manifest_sha256:
            raise RuntimeError("dataset manifest SHA-256 mismatch")
        args.dataset_manifest_sha256 = actual_dataset_sha
    patch_sha = sha256_file(patch)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_order = tuple(
        int(camera["evaluation_sequence_index"]) for camera in frozen["cameras"]
    )
    default_indices = (
        CAMERA_INDICES
        if args.dataset_id == "ALOD-ORACLE-MC-AERIAL-TEST-01"
        else frozen_order
    )
    camera_indices = tuple(args.camera_index or default_indices)
    if camera_indices != tuple(index for index in frozen_order if index in camera_indices):
        raise ValueError(
            f"camera indices must preserve the frozen order subset {frozen_order}"
        )
    by_index = {int(camera["evaluation_sequence_index"]): camera for camera in frozen["cameras"]}
    if any(index not in by_index for index in camera_indices):
        raise RuntimeError("requested camera missing from frozen metadata")
    output_root.mkdir(parents=True, exist_ok=True)
    camera_root = output_root / "cameras"
    attempt_root = output_root / "attempts"
    camera_root.mkdir(exist_ok=True)
    attempt_root.mkdir(exist_ok=True)
    camera_records = []
    with tempfile.TemporaryDirectory(prefix="alod-oracle-source-") as temporary:
        temporary_path = Path(temporary)
        instrumented_source = temporary_path / "source"
        prepare_source(source, patch, config, instrumented_source)
        hspt_payload_claimed = False
        for camera_index in camera_indices:
            camera = by_index[camera_index]
            camera_id = stable_id(camera)
            canonical_dir = camera_root / camera_id
            if canonical_dir.exists():
                if not args.resume:
                    raise RuntimeError(f"refusing to overwrite canonical camera {canonical_dir}")
                signature_sha = json.loads((canonical_dir / "determinism.json").read_text())["semantic_signature_sha256"]
            else:
                attempts = []
                for repetition in range(1, args.repetitions + 1):
                    attempt = attempt_root / camera_id / f"cold-{repetition:02d}"
                    run_camera(
                        python=python,
                        source=instrumented_source,
                        dataset=dataset,
                        asset=asset,
                        frozen=frozen_path,
                        patch_sha=patch_sha,
                        output=attempt,
                        camera_index=camera_index,
                        payload_mode="full" if repetition == 1 else "hash-only",
                        hspt_payloads=not hspt_payload_claimed and repetition == 1,
                        asset_sha256=args.asset_sha256,
                        asset_rows=args.asset_rows,
                        skybox_rows=args.skybox_rows,
                        root_global_node_id=args.root_global_node_id,
                        sh_degree=args.sh_degree,
                        resolution=args.resolution,
                    )
                    attempts.append(attempt)
                signature_sha = compare_attempts(attempts)
                shutil.move(str(attempts[0]), canonical_dir)
                hspt_payload_claimed = True
                determinism = {
                    "schema_version": "alod_selection_oracle_determinism.v1",
                    "camera_stable_id": camera_id,
                    "cold_process_repetitions": args.repetitions,
                    "semantic_signature_sha256": signature_sha,
                    "exact_match": True,
                    "compared": [
                        "HSPT structural hashes",
                        "effective camera matrices",
                        "upper traversal order and sets",
                        "ordered and sorted selected global IDs",
                        "source-row hashes",
                        "working-set mapping",
                        "pre-raster tensor hashes",
                        "render hash",
                    ],
                }
                (canonical_dir / "determinism.json").write_text(canonical_json(determinism), encoding="utf-8")
            cut = json.loads((canonical_dir / "spt-cut.json").read_text(encoding="utf-8"))
            pre = json.loads((canonical_dir / "pre-raster.json").read_text(encoding="utf-8"))
            upper = json.loads((canonical_dir / "upper-traversal.json").read_text(encoding="utf-8"))
            camera_record = {
                "camera_stable_id": camera_id,
                "registered_image_id": int(camera["registered_image_id"]),
                "camera_id": int(camera["camera_id"]),
                "filename": camera["image_name"],
                "stable_sorted_index": int(camera["stable_sorted_index"]),
                "evaluation_sequence_index": camera_index,
                "selection_categories": camera["selection_categories"],
                "selection_rationale": camera["selection_rationale"],
                "upper_candidate_spt_count": int(upper["counts"]["candidate_spts"]),
                "selected_hierarchy_count": int(cut["counts"]["selected_hierarchy_nodes"]),
                "skybox_raster_rows": int(pre["counts"]["skybox_raster_rows"]),
                "total_raster_gaussians": int(pre["counts"]["total_raster_gaussians"]),
                "minimum_selected_depth": int(cut["selected_depth"]["minimum"]),
                "maximum_selected_depth": int(cut["selected_depth"]["maximum"]),
                "selected_depth_distribution": cut["selected_depth"]["distribution"],
                "ordered_selected_global_ids_sha256": cut["identity_summaries"]["selected_hierarchy_global_node_ids"]["ordered_sha256"],
                "sorted_selected_global_ids_sha256": cut["identity_summaries"]["selected_hierarchy_global_node_ids"]["sorted_sha256"],
                "pre_raster_tensor_hashes": {
                    name: descriptor["raw_cpu_c_order_sha256"]
                    for name, descriptor in pre["tensors"].items()
                },
                "determinism_signature_sha256": signature_sha,
                "paths": {
                    name: f"cameras/{camera_id}/{name}"
                    for name in (
                        "camera.json",
                        "upper-traversal.json",
                        "spt-cut.json",
                        "working-set.json",
                        "pre-raster.json",
                        "optional-image.json",
                        "determinism.json",
                    )
                },
            }
            camera_records.append(camera_record)
    camera_dirs = [camera_root / record["camera_stable_id"] for record in camera_records]
    classification, _ = summarize_cameras(camera_dirs)
    hspt_reference = camera_dirs[0] / "hspt.json"
    build_manifests(
        output_root,
        camera_records,
        hspt_reference,
        patch_sha,
        asset,
        frozen_path,
        config,
        python,
        classification,
        oracle_id=args.oracle_id,
        asset_sha256=args.asset_sha256,
        asset_rows=args.asset_rows,
        skybox_rows=args.skybox_rows,
        root_global_node_id=args.root_global_node_id,
        sh_degree=args.sh_degree,
        dataset_id=args.dataset_id,
        dataset_manifest=args.dataset_manifest,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
    )
    execution = {
        "schema_version": "alod_selection_oracle_execution.v1",
        "creation_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "camera_indices": list(camera_indices),
        "repetitions": args.repetitions,
        "instrumented_source_disposable": True,
        "canonical_source_modified": False,
    }
    (output_root / "execution-metadata.json").write_text(canonical_json(execution), encoding="utf-8")
    print(
        canonical_json(
            {
                "gate": "PASS",
                "oracle_id": args.oracle_id,
                "output_root": str(output_root),
                "camera_count": len(camera_records),
                "classification": classification["classification"],
                "semantic_manifest_sha256": sha256_file(output_root / "semantic-manifest.json"),
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
