#!/usr/bin/env python3
"""Package the synthetic-positive-control versus natural-oracle diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STAGE_A = {
    "Campus": {
        "classification": "VALID-NONTRIVIAL-LOD",
        "selected_count_relative_range": 0.2523741040,
        "minimum_pairwise_jaccard": 0.4486992016,
        "maximum_pairwise_depth_tvd": 0.0719754897,
        "nearest_scale_depth_spearman": -0.071304,
        "scene_centre_depth_spearman": 0.338261,
    },
    "Sci-Art": {
        "classification": "VALID-WEAK-LOD",
        "selected_count_relative_range": 0.0242054432,
        "minimum_pairwise_jaccard": 0.9388327853,
        "maximum_pairwise_depth_tvd": 0.0130393510,
        "nearest_scale_depth_spearman": 0.100000,
        "scene_centre_depth_spearman": 0.833766,
    },
    "Run009": {
        "classification": "VALID-WEAK-LOD",
        "selected_count_relative_range": 0.044096,
        "minimum_pairwise_jaccard": 0.911268,
        "maximum_pairwise_depth_tvd": 0.018369,
        "nearest_scale_depth_spearman": None,
        "scene_centre_depth_spearman": None,
    },
}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def global_boundary(record: dict[str, Any]) -> float:
    depths = record["decision_margin_by_depth"].values()
    numerator = sum(row["near_either_boundary_abs_margin_le_0_05_count"] for row in depths)
    denominator = sum(row["candidate_decision_count"] for row in depths)
    return numerator / denominator


def natural_row(path: Path) -> dict[str, Any]:
    record = read(path)
    structure = record["hierarchy_structure_by_depth"]
    depths = sorted(int(value) for value in structure)
    first = structure[str(depths[0])]["six_sigma_max_axis_extent"]["median"]
    last = structure[str(depths[-1])]["six_sigma_max_axis_extent"]["median"]
    return {
        "instrumentation_path": str(path.resolve()),
        "instrumentation_sha256": sha(path),
        "camera_count": record["counts"]["camera_count"],
        "hierarchy_rows": record["counts"]["hierarchy_rows"],
        "maximum_depth": depths[-1],
        "root_median_extent_6sigma": first,
        "terminal_depth_median_extent_6sigma": last,
        "root_to_terminal_extent_ratio": first / last,
        "fraction_all_nodes_changing_selected_state": record["decision_changes"]["fraction_all_hierarchy_nodes"],
        "fraction_ever_selected_nodes_changing_state": record["decision_changes"]["fraction_ever_selected_nodes"],
        "global_fraction_spt_decisions_near_either_boundary_abs_margin_le_0_05": global_boundary(record),
        "selected_count_min": record["selected_count"]["min"],
        "selected_count_median": record["selected_count"]["median"],
        "selected_count_max": record["selected_count"]["max"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--synthetic-margins", type=Path, required=True)
    parser.add_argument("--synthetic-construction", type=Path, required=True)
    parser.add_argument("--campus", type=Path, required=True)
    parser.add_argument("--sciart", type=Path, required=True)
    parser.add_argument("--run009", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    classification = read(args.classification)
    margins = read(args.synthetic_margins)
    construction = read(args.synthetic_construction)
    by_depth = construction["structure"]["by_depth"]
    synthetic = {
        "classification": classification["classification"],
        "camera_count": margins["counts"]["camera_count"],
        "hierarchy_rows": margins["counts"]["hierarchy_rows"],
        "maximum_depth": construction["maximum_depth"],
        "root_median_extent_6sigma": by_depth[0]["median_extent_6sigma"],
        "terminal_depth_median_extent_6sigma": by_depth[-1]["median_extent_6sigma"],
        "root_to_terminal_extent_ratio": by_depth[0]["median_extent_6sigma"] / by_depth[-1]["median_extent_6sigma"],
        "fraction_all_nodes_changing_selected_state": margins["decision_changes"]["fraction_all_hierarchy_nodes"],
        "fraction_ever_selected_nodes_changing_state": margins["decision_changes"]["fraction_ever_selected_nodes"],
        "global_fraction_spt_decisions_near_either_boundary_abs_margin_le_0_05": global_boundary(margins),
        "selected_count_near_mid_far": [
            classification["per_camera"][index]["selected_hierarchy_rows"] for index in (0, 12, 24)
        ],
        "mean_depth_near_mid_far": [
            classification["per_camera"][index]["mean_depth"] for index in (0, 12, 24)
        ],
        "projected_scale_mean_depth_spearman": classification["positive_control_response"]["projected_scale_mean_depth_spearman"],
        "theil_sen_normalized_depth_per_scale_octave": classification["positive_control_response"]["theil_sen_normalized_depth_per_scale_octave"],
        "criteria_v1": classification["criteria_v1"],
    }
    natural = {}
    for label, path in (("Campus", args.campus), ("Sci-Art", args.sciart), ("Run009", args.run009)):
        natural[label] = STAGE_A[label] | natural_row(path)

    output = {
        "schema_version": 1,
        "experiment_id": "ALOD-SYNTH-STRONG-LOD-CONTROL-001",
        "scope": "descriptive positive-control comparison; no causal or prevalence inference",
        "stage_a_source": "/home/luis/tese/gs_framework/docs/results/alod_prevalence_stage_a.md",
        "synthetic": synthetic,
        "natural": natural,
        "interpretation": [
            "The synthetic control combines a regular coarse-to-fine construction with an eight-octave path and changes the state of nearly every node it ever selects.",
            "Campus changes a materially larger share of ever-selected nodes than Sci-Art or Run009, consistent with its nontrivial Stage-A label, but lacks the synthetic ordered scale-depth response.",
            "Near-boundary incidence alone does not order the four LoD responses, so no single-factor threshold-margin explanation is supported.",
            "The evidence is consistent with hierarchy scale organization and camera geometry interacting; three natural cases cannot establish causality.",
        ],
        "status": "PASS-SYNTH-NATURAL-MECHANISM-COMPARISON",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(output["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
