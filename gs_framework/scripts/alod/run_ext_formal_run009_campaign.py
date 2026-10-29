#!/usr/bin/env python3
"""Run the frozen six-mode formal Run009 ALOD-EXT campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


CAMPAIGN_ID = "ALOD-EXT-FORMAL-RUN009-001"
CONFIG_IDS = (
    "ALOD-EXT-ROW-LRU-PRACTICAL",
    "ALOD-EXT-PREMATERIALIZATION-CPU-PRACTICAL",
    "ALOD-EXT-PREMATERIALIZATION-CUDA-PREVIOUS",
    "ALOD-EXT-PREMATERIALIZATION-CUDA-MINIMUM",
    "ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL",
    "ALOD-EXT-PREMATERIALIZATION-CUDA-LARGE",
)
SCHEDULE = tuple(
    (replicate, CONFIG_IDS[(position + replicate - 1) % len(CONFIG_IDS)])
    for replicate in range(1, 4)
    for position in range(len(CONFIG_IDS))
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "schedule_policy": "three fresh processes per mode; balanced cyclic ordering",
                "camera_policy": "same 10 ordered warmups then all 71 frozen Run009 evaluation cameras once",
                "schedule": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    benchmark = Path(__file__).resolve().parents[2] / "benchmarks/benchmark_real_scene.py"
    rows = []
    for sequence, (replicate, config_id) in enumerate(SCHEDULE, start=1):
        output = root / "runs" / config_id / f"replicate-{replicate:02d}"
        command = [
            sys.executable,
            str(benchmark),
            "--config-id", config_id,
            "--asset", str(args.asset.resolve()),
            "--output-dir", str(output),
            "--formal",
            "--max-frames", "71",
            "--num-warmup", "10",
            "--num-repeat", "1",
            "--log-gpu-events",
            "--record-warmup-frames",
        ]
        rows.append(
            {
                "sequence_index": sequence,
                "replicate": replicate,
                "configuration_id": config_id,
                "output_dir": str(output),
                "command": command,
                "status": "planned",
                "started_utc": None,
                "finished_utc": None,
            }
        )
    schedule = root / "checks/formal_run_schedule.json"
    _write(schedule, rows)
    for row in rows:
        summary = Path(row["output_dir"]) / "summary.json"
        if summary.exists():
            if not args.resume:
                raise FileExistsError(f"refusing to overwrite {summary}")
            row["status"] = "resumed_complete"
            _write(schedule, rows)
            continue
        row["status"] = "running"
        row["started_utc"] = datetime.now(timezone.utc).isoformat()
        _write(schedule, rows)
        print(
            f"[{row['sequence_index']}/{len(rows)}] {row['configuration_id']} "
            f"replicate {row['replicate']}",
            flush=True,
        )
        try:
            subprocess.run(row["command"], check=True, cwd=benchmark.parents[1])
        except BaseException:
            row["status"] = "failed"
            row["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _write(schedule, rows)
            raise
        row["status"] = "complete"
        row["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write(schedule, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
