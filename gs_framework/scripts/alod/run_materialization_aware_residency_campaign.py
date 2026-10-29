#!/usr/bin/env python3
"""Run the frozen four-mode materialization-aware ALoD campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


CAMPAIGN_ID = "ALOD-MATERIALIZATION-AWARE-RESIDENCY-BENCHMARK-001"
CONFIG_IDS = (
    "ALOD-HIER-LOD-FRUSTUM",
    "ALOD-HOST-PREVIOUS-REUSE-FAST-PREMATERIALIZATION",
    "ALOD-HOST-BOUNDED-CACHE-FAST-PREMATERIALIZATION",
    "ALOD-HOST-BOUNDED-CACHE-REDUCED",
)

# Preserve the accepted host-residency campaign's process/position pattern.
SCHEDULE = (
    (1, CONFIG_IDS[0]),
    (1, CONFIG_IDS[1]),
    (1, CONFIG_IDS[2]),
    (1, CONFIG_IDS[3]),
    (2, CONFIG_IDS[2]),
    (2, CONFIG_IDS[3]),
    (2, CONFIG_IDS[0]),
    (2, CONFIG_IDS[1]),
    (3, CONFIG_IDS[1]),
    (3, CONFIG_IDS[0]),
    (3, CONFIG_IDS[3]),
    (3, CONFIG_IDS[2]),
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "schedule_policy": (
                    "Exact A/B/C/D substitution into the accepted 12-process "
                    "ALOD-HOST-RESIDENCY-BENCHMARK-001 schedule"
                ),
                "schedule": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
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
    schedule_path = root / "checks" / "formal_run_commands.json"
    rows: list[dict] = []
    for sequence_index, (replicate, config_id) in enumerate(SCHEDULE, start=1):
        output_dir = root / "runs" / config_id / f"replicate-{replicate:02d}"
        command = [
            sys.executable,
            str(benchmark),
            "--config-id", config_id,
            "--asset", str(asset),
            "--output-dir", str(output_dir),
            "--formal",
            "--max-frames", "100",
            "--num-warmup", "10",
            "--num-repeat", "10",
            "--log-gpu-events",
            "--record-warmup-frames",
        ]
        rows.append(
            {
                "sequence_index": sequence_index,
                "replicate": replicate,
                "configuration_id": config_id,
                "output_dir": str(output_dir),
                "command": command,
                "status": "planned",
                "started_utc": None,
                "finished_utc": None,
            }
        )
    _write(schedule_path, rows)

    for record in rows:
        summary = Path(record["output_dir"]) / "summary.json"
        if summary.exists():
            if not args.resume:
                raise FileExistsError(f"refusing to overwrite {summary}")
            record["status"] = "resumed_complete"
            _write(schedule_path, rows)
            continue
        print(
            f"[{record['sequence_index']}/{len(SCHEDULE)}] start "
            f"{record['configuration_id']} replicate {record['replicate']}",
            flush=True,
        )
        record["status"] = "running"
        record["started_utc"] = datetime.now(timezone.utc).isoformat()
        _write(schedule_path, rows)
        try:
            subprocess.run(record["command"], check=True, cwd=benchmark.parents[1])
        except BaseException:
            record["status"] = "failed"
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _write(schedule_path, rows)
            raise
        record["status"] = "complete"
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write(schedule_path, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
