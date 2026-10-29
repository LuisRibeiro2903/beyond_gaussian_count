#!/usr/bin/env python3
"""Run the frozen four-mode ALoD residency benchmark schedule."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


CAMPAIGN_ID = "ALOD-HOST-RESIDENCY-BENCHMARK-001"
CONFIG_IDS = (
    "ALOD-HIER-LOD-FRUSTUM",
    "ALOD-HOST-NOCACHE",
    "ALOD-HOST-PREVIOUS-REUSE",
    "ALOD-HOST-BOUNDED-CACHE",
)

# Frozen before execution.  Three replicates preserve the resident campaign's
# 300 measured frames/configuration.  Each mode occupies three distinct order
# positions; perfect four-position balance would require a fourth replicate.
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


def _write_schedule(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "schedule_policy": (
                    "three independent processes/configuration; every mode occupies "
                    "three distinct order positions; no mode always first or last"
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
    benchmark = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "benchmark_real_scene.py"
    )
    schedule_path = root / "checks" / "formal_run_commands.json"
    rows: list[dict] = []
    for sequence_index, (replicate, config_id) in enumerate(SCHEDULE, start=1):
        output_dir = root / "runs" / config_id / f"replicate-{replicate:02d}"
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
        record = {
            "sequence_index": sequence_index,
            "replicate": replicate,
            "configuration_id": config_id,
            "output_dir": str(output_dir),
            "command": command,
            "status": "planned",
            "started_utc": None,
            "finished_utc": None,
        }
        rows.append(record)
    _write_schedule(schedule_path, rows)

    for record in rows:
        sequence_index = int(record["sequence_index"])
        replicate = int(record["replicate"])
        config_id = str(record["configuration_id"])
        output_dir = Path(record["output_dir"])
        summary = output_dir / "summary.json"
        command = list(record["command"])
        if summary.exists():
            if not args.resume:
                raise FileExistsError(f"refusing to overwrite {summary}")
            record["status"] = "resumed_complete"
            print(
                f"[{sequence_index}/{len(SCHEDULE)}] skip complete "
                f"{config_id} replicate {replicate}",
                flush=True,
            )
            _write_schedule(schedule_path, rows)
            continue

        print(
            f"[{sequence_index}/{len(SCHEDULE)}] start {config_id} "
            f"replicate {replicate}",
            flush=True,
        )
        record["status"] = "running"
        record["started_utc"] = datetime.now(timezone.utc).isoformat()
        _write_schedule(schedule_path, rows)
        try:
            subprocess.run(command, check=True, cwd=benchmark.parents[1])
        except BaseException:
            record["status"] = "failed"
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _write_schedule(schedule_path, rows)
            raise
        record["status"] = "complete"
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_schedule(schedule_path, rows)
        print(
            f"[{sequence_index}/{len(SCHEDULE)}] complete {config_id} "
            f"replicate {replicate}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
