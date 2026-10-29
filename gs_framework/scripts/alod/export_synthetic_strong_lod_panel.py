#!/usr/bin/env python3
"""Export the frozen near/middle/far synthetic strong-LoD visual panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    classification = json.loads(args.classification.read_text(encoding="utf-8"))
    indices = (0, 12, 24)
    labels = ("Near", "Middle", "Far")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    rows = []
    for column, (index, label) in enumerate(zip(indices, labels)):
        record = classification["per_camera"][index]
        image_path = args.render_root / f"camera_{index:02d}.png"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        axes[0, column].imshow(image)
        axes[0, column].axis("off")
        axes[0, column].set_title(
            f"{label}: camera {index}, distance {record['distance']:.1f}\n"
            f"selected {record['selected_hierarchy_rows']:,}; "
            f"mean depth {record['mean_depth']:.2f}; median {record['median_depth']:.1f}"
        )
        histogram = {int(k): int(v) for k, v in record["depth_histogram"].items()}
        depth = np.asarray(sorted(histogram), dtype=np.int32)
        count = np.asarray([histogram[int(value)] for value in depth], dtype=np.int64)
        axes[1, column].bar(depth, count, width=0.8, color="#3569a8")
        axes[1, column].set_xlim(-0.5, 19.5)
        axes[1, column].set_xticks(range(0, 20, 2))
        axes[1, column].set_xlabel("Selected hierarchy depth")
        axes[1, column].set_ylabel("Rows")
        axes[1, column].grid(axis="y", alpha=0.25)
        rows.append(
            {
                "camera_index": index,
                "label": label.lower(),
                "render": str(image_path.resolve()),
                "render_sha256": sha256_file(image_path),
                "selected_hierarchy_rows": record["selected_hierarchy_rows"],
                "mean_depth": record["mean_depth"],
                "median_depth": record["median_depth"],
                "maximum_depth": record["maximum_depth"],
            }
        )
    figure.suptitle(
        "ALOD-SYNTH-STRONG-LOD-CONTROL-001 — unchanged released selector",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    manifest = {
        "experiment_id": "ALOD-SYNTH-STRONG-LOD-CONTROL-001",
        "classification": classification["classification"],
        "classification_source": str(args.classification.resolve()),
        "classification_sha256": sha256_file(args.classification),
        "panel": str(args.output.resolve()),
        "panel_sha256": sha256_file(args.output),
        "representatives": rows,
        "quantitative_acceptance_use": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
