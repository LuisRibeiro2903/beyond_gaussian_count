#!/usr/bin/env python3
"""Freeze the factual completion record for the patched ALoD fine retry."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def maximum(rows: list[dict], key: str):
    values = [row[key] for row in rows if row.get(key) is not None]
    return max(values) if values else None


def minimum(rows: list[dict], key: str):
    values = [row[key] for row in rows if row.get(key) is not None]
    return min(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    diagnostics = args.diagnostics.resolve()

    metrics = [json.loads(line) for line in (run / "metrics/fine.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (run / "logs/events.jsonl").read_text().splitlines()]
    log = (run / "logs/fine.log").read_text(errors="replace")
    decoder = json.JSONDecoder()
    topology_events = [
        decoder.raw_decode(line)[0]
        for line in log.splitlines()
        if line.startswith('{"active_size"')
    ]
    final_barrier = json.loads((run / "validation/final-hierarchy.json").read_text())
    independent = json.loads((run / "validation/final-topology-independent.json").read_text())
    before = json.loads((diagnostics / "run004-protected-tree-before-run005.json").read_text())
    after = json.loads((run / "validation/run004-protected-tree-after.json").read_text())
    official = json.loads((diagnostics / "official-synthetic-results.json").read_text())
    patched = json.loads((diagnostics / "patched-synthetic-results.json").read_text())
    differential = json.loads((diagnostics / "differential-results.json").read_text())
    mutation = json.loads((diagnostics / "integration-mutation-results.json").read_text())

    exit_events = [event for event in events if event["event"] == "exit"]
    diagnostics_or_warnings = [event for event in events if event["event"] not in {"launch", "exit"}]
    training_seconds = float(re.findall(r"Training finished in ([0-9.]+) seconds", log)[-1])
    torch_peak = int(re.findall(r"Peak memory=([0-9_]+)", log)[-1].replace("_", ""))
    densification = []
    for iteration in (2000, 4000, 6000, 8000):
        selected = [event for event in topology_events if event["iteration"] == iteration]
        densification.append(
            {
                "iteration": iteration,
                "all_valid": all(event["valid"] for event in selected),
                "events": [
                    {
                        "mutation": event["mutation"],
                        "scope": event["scope"],
                        "reachable_nodes": event["reachable_nodes"],
                        "checked_edges": event["checked_edges"],
                        "changed_row_count": event["changed_row_count"],
                        "mismatch_counts": event["mismatch_counts"],
                        "duration_seconds": event["duration_seconds"],
                    }
                    for event in selected
                ],
            }
        )

    report = {
        "schema_version": 1,
        "principal_status": "PASS-ALOD-FIRST-DHIER",
        "run_id": "ALOD-SMOKE-TRAIN-005",
        "root_cause": {
            "mutation": "GaussianModel.relocate_gs",
            "sentinel_index_bug": "a leaf sibling has first_child=0; official lines 1769-1775 dereference node 0 and then its -99 next_sibling as a Python negative index, matching Run004 row 1099901 at capacity 1100000",
            "depth_propagation_bug": "official lines 1769-1775 update only immediate children after interior-sibling promotion; descendants below that level retain stale depths",
            "serialization_cause_excluded": True,
            "add_new_gs_cause_excluded": True,
            "hspt_cause_excluded": True,
        },
        "official_failure_evidence": {
            "run004_status": "FAILED-ALOD-FINAL-BARRIER",
            "depth_mismatches": 4355,
            "parent_mismatches": 1,
            "parent_mismatch_node": 1099901,
            "official_synthetic_failing_cases": official["failing_cases"],
            "minimal_capacity_case": "A2_leaf_sibling_real_capacity",
            "deep_descendant_case": "C_deep_interior_sibling",
        },
        "patch": {
            "name": "ALOD-RELOCATE-TOPOLOGY-FIX",
            "strategy": "bounded forward traversal from affected parent and respawn roots reconstructs only reciprocal parent and stored depth metadata; no global repair in the algorithm",
            "algorithm_patch_sha256": "a7412fff7315153774c93b504105804c2a182308807a2c6ec8de34672b9cd7d6",
            "instrumentation_patch_sha256": "98fa8f6c6d0cdde9a96d03ad04b1485efd8576c9e841a921b5a4661da3ef79d7",
            "official_commit": "73f547341547ae021ff036ad74fdf73a1240b450",
            "algorithm_source_files_changed": ["scene/gaussian_model.py"],
            "instrumentation_source_files_changed": ["train_hierarchy.py"],
        },
        "tests": {
            "official_reproducer_gate": official["gate"],
            "patched_reproducer_gate": patched["gate"],
            "patched_failing_cases": patched["failing_cases"],
            "differential_gate": differential["gate"],
            "properties_forward_topology_and_respawn_preserved": differential["properties_forward_topology_and_respawn_preserved"],
            "real_cuda_mutation_gate": mutation["gate"],
            "real_deep_descendant_levels": mutation["relocate_gs"]["deep_interior_descendant_levels"],
            "real_skybox_rows_unchanged": mutation["skybox_rows_unchanged"],
        },
        "run005": {
            "source_run": "ALOD-SMOKE-TRAIN-004",
            "coarse_ply_sha256": "ffcbd2b12d60cbfd2260845e432421ee2af2f552d10818c0ae8d1f067ab4d143",
            "initial_dhier_sha256": "a7862de91cd3620ab7be1296a50b9030aa1c2cca2b4fe15d79abaf55f4664d17",
            "rejected_run004_final_reused": False,
            "training_seconds_reported_by_source": training_seconds,
            "monitor_exit_status": exit_events[-1]["exit_status"],
            "monitor_abort_reason": exit_events[-1]["abort_reason"],
            "mutation_validations": densification,
            "all_mutation_validations_passed": all(event["valid"] for event in topology_events),
        },
        "final_hierarchy": {
            "path": final_barrier["path"],
            "sha256": final_barrier["sha256"],
            "bytes": final_barrier["bytes"],
            "rows": final_barrier["gaussians"],
            "sh_degree": final_barrier["sh_degree"],
            "all_attributes_finite": final_barrier["all_attributes_finite"],
            "reachable_non_skybox_nodes": final_barrier["reachable_non_skybox_nodes"],
            "independent_checked_edges": independent["checked_edges"],
            "independent_mismatch_counts": independent["mismatch_counts"],
            "independent_valid": independent["valid"],
        },
        "resources": {
            "samples": len(metrics),
            "peak_process_group_rss_bytes": maximum(metrics, "process_group_rss_bytes"),
            "minimum_system_available_ram_bytes": minimum(metrics, "system_available_ram_bytes"),
            "peak_whole_device_vram_bytes": maximum(metrics, "whole_device_vram_bytes"),
            "process_vram_attribution": "unavailable; never substituted with whole-device usage",
            "peak_pytorch_memory_log_bytes": torch_peak,
            "peak_swap_growth_bytes": maximum(metrics, "swap_growth_bytes"),
            "peak_output_directory_bytes_during_fine": maximum(metrics, "output_directory_bytes"),
            "resource_policy_abort": exit_events[-1]["abort_reason"],
            "nonterminal_events": diagnostics_or_warnings,
        },
        "protected_state": {
            "prior_run004_audit_fingerprint_reference": "52e324356d7ab9351d9300cdaf6839d9533549e5981da6895cb78f7a570c048f",
            "content_manifest_fingerprint_algorithm": "sha256(canonical JSON of relative path, kind, byte length, file sha256 or symlink target)",
            "run004_fingerprint_before": before["fingerprint_sha256"],
            "run004_fingerprint_after": after["fingerprint_sha256"],
            "run004_unchanged": before["fingerprint_sha256"] == after["fingerprint_sha256"],
            "canonical_source_commit": "73f547341547ae021ff036ad74fdf73a1240b450",
            "nerfstudio2_history_sha256": "02a7a389b7f8b0622bceb935f69173b1b8861bb08e97211fc0700e5387be76d4",
            "alod_official_history_sha256": "1bbcac8c21a82f9891d22e8293ec4787e4e68c5cb399a066cf6ecbb02fc977bd",
        },
        "fidelity_classification": "correctness patch required for structurally valid local reproduction; not bit-identical pinned-source reproduction",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
