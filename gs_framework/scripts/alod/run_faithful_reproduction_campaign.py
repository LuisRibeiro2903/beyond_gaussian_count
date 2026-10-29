#!/usr/bin/env python3
"""Run the frozen three-mode ALOD-REPRO rendering campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


CAMPAIGN_ID = "ALOD-FAITHFUL-REPRODUCTION-001"
CONFIG_IDS = (
    "ALOD-REPRO-HOST-NOCACHE",
    "ALOD-REPRO-NOFRUSTUM",
    "ALOD-REPRO-PREVIOUS-REUSE",
)
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


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    asset = args.asset.resolve()
    benchmark = Path(__file__).resolve().parents[2] / "benchmarks/benchmark_real_scene.py"
    schedule_path = root / "checks/formal_render_schedule.json"
    rows = []
    for sequence, (replicate, config_id) in enumerate(SCHEDULE, start=1):
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
            "71",
            "--num-warmup",
            "10",
            "--num-repeat",
            "1",
            "--log-gpu-events",
        ]
        rows.append(
            {
                "sequence_index": sequence,
                "replicate": replicate,
                "configuration_id": config_id,
                "output_dir": str(output_dir),
                "command": command,
                "status": "planned",
                "started_utc": None,
                "finished_utc": None,
            }
        )
    envelope = {
        "campaign_id": CAMPAIGN_ID,
        "schedule_policy": "three fresh processes per mode; balanced cyclic ordering",
        "camera_policy": "10 ordered warmups then all 71 frozen eval cameras once",
        "schedule": rows,
    }
    write(schedule_path, envelope)
    for row in rows:
        summary = Path(row["output_dir"]) / "summary.json"
        if summary.exists():
            if not args.resume:
                raise FileExistsError(f"refusing to overwrite {summary}")
            row["status"] = "resumed_complete"
            write(schedule_path, envelope)
            continue
        row["status"] = "running"
        row["started_utc"] = datetime.now(timezone.utc).isoformat()
        write(schedule_path, envelope)
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
            write(schedule_path, envelope)
            raise
        row["status"] = "complete"
        row["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(schedule_path, envelope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
