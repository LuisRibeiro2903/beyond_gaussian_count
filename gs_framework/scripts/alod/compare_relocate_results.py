#!/usr/bin/env python3
"""Compare official and patched synthetic relocation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--patched", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    official = json.loads(args.official.read_text(encoding="utf-8"))
    patched = json.loads(args.patched.read_text(encoding="utf-8"))
    official_cases = {case["case"]: case for case in official["cases"]}
    patched_cases = {case["case"]: case for case in patched["cases"]}
    if official_cases.keys() != patched_cases.keys():
        raise RuntimeError("official and patched case sets differ")

    comparisons = []
    for name in official_cases:
        before = official_cases[name]
        after = patched_cases[name]
        comparison = {
            "case": name,
            "properties_equal": (
                before["properties_after_sha256"] == after["properties_after_sha256"]
            ),
            "forward_topology_equal": (
                before["forward_topology_after_sha256"]
                == after["forward_topology_after_sha256"]
            ),
            "respawn_behavior_equal": before["respawn_checks"] == after["respawn_checks"],
            "official_topology_valid": before["after"]["valid"],
            "patched_topology_valid": after["after"]["valid"],
            "official_skybox_unchanged": before["skybox_unchanged"],
            "patched_skybox_unchanged": after["skybox_unchanged"],
        }
        comparisons.append(comparison)

    preserved = all(
        result["properties_equal"]
        and result["forward_topology_equal"]
        and result["respawn_behavior_equal"]
        for result in comparisons
    )
    corrected = all(
        result["patched_topology_valid"] and result["patched_skybox_unchanged"]
        for result in comparisons
    )
    report = {
        "gate": "PASS" if preserved and corrected else "FAIL",
        "official_report": str(args.official.resolve()),
        "patched_report": str(args.patched.resolve()),
        "properties_forward_topology_and_respawn_preserved": preserved,
        "all_patched_topology_and_skybox_checks_pass": corrected,
        "cases": comparisons,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
