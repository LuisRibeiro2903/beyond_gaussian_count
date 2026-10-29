#!/usr/bin/env python3
"""Run the prospective ordered-multiset closure on the strong-LoD control."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.alod.run_synthetic_strong_lod_ext_campaign import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
