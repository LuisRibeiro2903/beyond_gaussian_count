#!/usr/bin/env python
"""Capture a durable pre/post-run NVIDIA state without changing settings."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


FIELDS = (
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "pstate",
    "clocks.current.graphics",
    "clocks.current.memory",
    "power.draw",
)


def command(args: list[str]) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--phase", choices=("pre", "post"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = command(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(FIELDS),
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
    )
    values = next(csv.reader(io.StringIO(query), skipinitialspace=True))
    record = dict(zip(FIELDS, values))
    for key in (
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "clocks.current.graphics",
        "clocks.current.memory",
    ):
        record[key] = int(float(record[key]))
    record["power.draw"] = float(record["power.draw"])
    payload = {
        "record_kind": "CITYGS-OFFICIAL-CLEAN-FULL-gpu-state",
        "repeat": args.repeat,
        "phase": args.phase,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": record,
        "nvidia_smi_full": command(["nvidia-smi"]),
        "settings_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
