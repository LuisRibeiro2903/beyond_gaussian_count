#!/usr/bin/env python3
"""Run the frozen ALoD resident benchmark process-replicate schedule."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CONFIG_IDS = (
    "ALOD-HIER-FINEST-RESIDENT",
    "ALOD-HIER-LOD-RESIDENT",
    "ALOD-HIER-LOD-FRUSTUM",
)

# Balanced interleaving: every configuration occupies each run-order position.
SCHEDULE = (
    (1, CONFIG_IDS[0]),
    (1, CONFIG_IDS[1]),
    (1, CONFIG_IDS[2]),
    (2, CONFIG_IDS[2]),
    (2, CONFIG_IDS[0]),
    (2, CONFIG_IDS[1]),
    (3, CONFIG_IDS[1]),
    (3, CONFIG_IDS[2]),
    (3, CONFIG_IDS[0]),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = args.campaign_root.resolve()
    asset = args.asset.resolve()
    benchmark = Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark_real_scene.py"
    commands = []
    for sequence_index, (replicate, config_id) in enumerate(SCHEDULE, start=1):
        output_dir = root / "runs" / config_id / f"replicate-{replicate:02d}"
        summary = output_dir / "summary.json"
        if summary.exists():
            if args.resume:
                print(f"[{sequence_index}/9] skip complete {config_id} replicate {replicate}", flush=True)
                continue
            raise FileExistsError(f"refusing to overwrite {summary}")
        command = [
            sys.executable,
            str(benchmark),
            "--config-id",
            config_id,
            "--asset",
            str(asset),
            "--output-dir",
            str(output_dir),
            "--formal",
            "--max-frames",
            "100",
            "--num-warmup",
            "10",
            "--num-repeat",
            "10",
            "--log-gpu-events",
        ]
        print(f"[{sequence_index}/9] start {config_id} replicate {replicate}", flush=True)
        started = datetime.now(timezone.utc).isoformat()
        subprocess.run(command, check=True, cwd=benchmark.parents[1])
        finished = datetime.now(timezone.utc).isoformat()
        print(f"[{sequence_index}/9] complete {config_id} replicate {replicate}", flush=True)
        commands.append(
            {
                "sequence_index": sequence_index,
                "replicate": replicate,
                "configuration_id": config_id,
                "command": command,
                "started_utc": started,
                "finished_utc": finished,
            }
        )
    checks = root / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    (checks / "formal_run_commands.json").write_text(
        json.dumps({"schedule": commands}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
