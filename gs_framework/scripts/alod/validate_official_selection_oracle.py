#!/usr/bin/env python3
"""Validate a project-owned official ALoD selection-oracle export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alod_selection_oracle import validate_oracle_tree


DEFAULT_MANIFEST = Path(
    "/home/luis/tese/runs/alod/oracles/"
    "ALOD-RUN005-SELECTION-ORACLE-001/manifest.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    result = validate_oracle_tree(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
