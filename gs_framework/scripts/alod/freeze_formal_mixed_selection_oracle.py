#!/usr/bin/env python3
"""Freeze compact formal-oracle records after validation and exact parity."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_selection_oracle import sha256_file, validate_oracle_tree


PASS_GATE = "PASS-ALOD-FORMAL-SELECTION-ORACLE"
OFFICIAL_COMMIT = "73f547341547ae021ff036ad74fdf73a1240b450"
RUN005_SEMANTIC_SHA256 = "20bb87caf1694ede18ac2d2a71cdc146c002325ddbf8f6060625f18feda70591"


def canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_value(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--project-parity", type=Path, required=True)
    parser.add_argument("--hierarchy-preflight", type=Path, required=True)
    parser.add_argument("--frozen-cameras", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--hook", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--locator", type=Path, required=True)
    args = parser.parse_args()

    root = args.oracle_root.resolve()
    manifest_path = root / "manifest.json"
    validation = validate_oracle_tree(manifest_path)
    manifest = load_json(manifest_path)
    analysis = load_json(args.analysis)
    parity = load_json(args.project_parity)
    preflight = load_json(args.hierarchy_preflight)
    frozen = load_json(args.frozen_cameras)
    official_head = git_value(args.official_source, "rev-parse", "HEAD")
    official_status = git_value(args.official_source, "status", "--porcelain")

    checks = {
        "oracle_tree_valid": validation["valid"],
        "camera_count_16": validation["camera_count"] == 16,
        "all_cold_determinism_exact": all(
            camera["cold_determinism"]["exact_match"]
            and camera["cold_determinism"]["cold_process_repetitions"] >= 2
            for camera in analysis["cameras"]
        ),
        "project_reference_status": parity["status"] == "PASS-ALOD-CPU-REFERENCE-SELECTION",
        "project_hspt_parity": parity["hspt"]["parity"],
        "project_all_camera_layers_exact": all(
            all(camera["parity"].values()) for camera in parity["cameras"]
        ),
        "strict_hierarchy_valid": preflight["validation"]["overall_valid"],
        "frozen_camera_count_16": len(frozen["cameras"]) == 16,
        "official_commit_pinned": official_head == OFFICIAL_COMMIT,
        "official_source_clean": official_status == "",
        "dataset_manifest_exact": sha256_file(args.dataset_manifest)
        == manifest["dataset"]["manifest_sha256"],
        "hierarchy_exact": sha256_file(Path(manifest["asset"]["path"]))
        == manifest["asset"]["sha256"],
        "run005_oracle_unchanged": sha256_file(
            Path("/home/luis/tese/runs/alod/oracles/ALOD-RUN005-SELECTION-ORACLE-001/semantic-manifest.json")
        ) == RUN005_SEMANTIC_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"formal selection oracle freeze gate failed: {checks}")

    compact_cameras = []
    for camera in analysis["cameras"]:
        compact_cameras.append(
            {
                "camera_stable_id": camera["camera_stable_id"],
                "filename": camera["filename"],
                "modality": camera["modality"],
                "distance_stratum": camera["distance_stratum"],
                "registered_image_id": camera["registered_image_id"],
                "camera_id": camera["camera_id"],
                "dataset_index": camera["dataset_index"],
                "evaluation_sequence_index": camera["evaluation_sequence_index"],
                "selected_hierarchy_count": camera["selected_hierarchy_count"],
                "candidate_spt_count": camera["candidate_spt_count"],
                "selected_depth": {
                    key: camera["selected_depth"][key]
                    for key in ("minimum", "p10", "p25", "median", "mean", "p75", "p90", "maximum")
                },
                "ordered_selected_global_ids_sha256": camera["ordered_selected_global_node_ids_sha256"],
                "sorted_selected_global_ids_sha256": camera["sorted_selected_global_node_ids_sha256"],
                "ordered_raster_global_ids_sha256": camera["ordered_raster_global_node_ids_sha256"],
                "pre_raster_bundle_sha256": camera["pre_raster_bundle_sha256"],
                "determinism_signature_sha256": camera["cold_determinism"]["semantic_signature_sha256"],
            }
        )

    locator = {
        "schema_version": "alod_selection_oracle_locator.v1",
        "oracle_id": manifest["oracle_id"],
        "external_manifest": str(manifest_path),
        "deterministic_semantic_manifest_sha256": manifest["deterministic_semantic_manifest"]["sha256"],
        "formal_analysis": str(args.analysis.resolve()),
        "formal_analysis_sha256": sha256_file(args.analysis),
        "project_reference_parity": str(args.project_parity.resolve()),
        "project_reference_parity_sha256": sha256_file(args.project_parity),
        "hierarchy_preflight_sha256": sha256_file(args.hierarchy_preflight),
        "asset_sha256": manifest["asset"]["sha256"],
        "dataset_manifest_sha256": manifest["dataset"]["manifest_sha256"],
        "frozen_camera_manifest_sha256": manifest["dataset"]["frozen_camera_manifest_sha256"],
        "official_source_commit": official_head,
        "instrumentation_hook_sha256": sha256_file(args.hook),
        "instrumentation_patch_sha256": sha256_file(args.patch),
        "hspt_structural_array_hashes": manifest["hspt"]["structural_array_hashes"],
        "lod_classification": analysis["classification"],
        "project_reproduction_status": parity["status"],
        "cameras": compact_cameras,
    }
    args.locator.parent.mkdir(parents=True, exist_ok=True)
    args.locator.write_text(canonical_json(locator), encoding="utf-8")
    completion = {
        "schema_version": "alod_formal_selection_oracle_completion.v1",
        "gate": PASS_GATE,
        "oracle_id": manifest["oracle_id"],
        "classification": analysis["classification"]["classification"],
        "checks": checks,
        "oracle_validation": validation,
        "locator": str(args.locator.resolve()),
        "locator_sha256": sha256_file(args.locator),
        "compact_artifact_hashes": {
            "semantic_manifest": manifest["deterministic_semantic_manifest"]["sha256"],
            "formal_analysis": sha256_file(args.analysis),
            "project_reference_parity": sha256_file(args.project_parity),
            "hierarchy_strict_preflight": sha256_file(args.hierarchy_preflight),
        },
        "protected_state": {
            "official_source_status_porcelain": official_status,
            "run005_semantic_manifest_sha256": RUN005_SEMANTIC_SHA256,
            "formal_hierarchy_sha256": manifest["asset"]["sha256"],
            "formal_dataset_manifest_sha256": manifest["dataset"]["manifest_sha256"],
        },
    }
    completion_path = root / "completion-report.json"
    completion_path.write_text(canonical_json(completion), encoding="utf-8")
    print(canonical_json({"gate": PASS_GATE, "locator": str(args.locator.resolve()), "locator_sha256": sha256_file(args.locator)}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
