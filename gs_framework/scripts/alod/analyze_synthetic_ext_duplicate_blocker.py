#!/usr/bin/env python3
"""Freeze the released far-cut duplicate-ID barrier to synthetic ALoD-EXT."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_dhier import load_dhier  # noqa: E402
from alod_reference import HierarchyData, build_hspt  # noqa: E402
from scripts.alod.synthetic_strong_lod_control import (  # noqa: E402
    reference_config,
    sha256_file,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    official = json.loads(args.official.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    asset = load_dhier(args.asset)
    hspt = build_hspt(HierarchyData.from_dhier(asset), reference_config())

    cameras = []
    all_parity = True
    first_duplicate = None
    for official_row, reference_row in zip(official["cameras"], reference["cameras"]):
        official_ids = np.load(official_row["selected_ids_path"], allow_pickle=False)
        reference_ids = np.load(reference_row["selected_ids_path"], allow_pickle=False)
        exact = bool(np.array_equal(official_ids, reference_ids))
        all_parity &= exact
        counts = Counter(int(value) for value in official_ids)
        duplicates = [
            {"global_node_id": key, "multiplicity": value}
            for key, value in sorted(counts.items())
            if value > 1
        ]
        row = {
            "camera_index": int(official_row["camera_index"]),
            "distance": float(official_row["distance"]),
            "selected_rows": int(official_ids.size),
            "unique_rows": int(np.unique(official_ids).size),
            "official_reference_exact_ordered_parity": exact,
            "duplicates": duplicates,
        }
        if duplicates and first_duplicate is None:
            first_duplicate = row
        cameras.append(row)

    first_spt_root = int(hspt.spt_root_global_node_ids[0])
    first_spt_start = int(hspt.spt_starts[0])
    output = {
        "schema_version": 1,
        "experiment_id": "ALOD-SYNTH-STRONG-LOD-CONTROL-001",
        "status": "BLOCKED-SYNTH-ALOD-EXT-OFFICIAL-DUPLICATE-SEMANTIC-IDS",
        "asset": {"path": str(args.asset.resolve()), "sha256": sha256_file(args.asset)},
        "official_summary_sha256": sha256_file(args.official),
        "reference_selection_sha256": sha256_file(args.reference),
        "official_reference_exact_ordered_parity_all_cameras": all_parity,
        "first_duplicate_camera": first_duplicate,
        "duplicate_camera_indices": [
            row["camera_index"] for row in cameras if row["duplicates"]
        ],
        "mechanism": {
            "first_spt_id": 0,
            "first_spt_root_global_node_id": first_spt_root,
            "first_spt_flattened_root_global_node_id": int(
                hspt.spt_to_global_node_id[first_spt_start]
            ),
            "released_upper_render_condition": "first_child <= 0",
            "released_candidate_spt_condition": "first_child >= 0",
            "released_spt_population": (
                "when camera distance exceeds the first SPT root minimum threshold, "
                "SPT 0 root is emitted by both the upper-render and SPT paths"
            ),
            "validated_ext_barrier": (
                "AlodHostScene residency requires unique selected global_node_ids; "
                "deduplicating or adding duplicate-aware residency would modify the already "
                "validated ALoD-EXT implementation or the exact official semantic order"
            ),
        },
        "correctness_execution": {
            "cameras_passed_before_barrier": list(range(18)),
            "first_rejected_camera": 18,
            "performance_timing_started": False,
            "selector_changed": False,
            "official_cut_replaced_or_deduplicated": False,
            "filter_or_residency_retuned": False,
        },
        "cameras": cameras,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(output), encoding="utf-8")
    print(output["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
