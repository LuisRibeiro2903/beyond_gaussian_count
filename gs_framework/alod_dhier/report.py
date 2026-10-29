"""Deterministic machine-readable reports for project-owned ALoD inspection."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .format import OFFICIAL_SOURCE_COMMIT, DhierAsset, load_dhier
from .validation import (
    DhierNumericError,
    DhierTopologyError,
    ValidationProfile,
    validate_numeric,
    validate_topology,
)


SCHEMA_VERSION = "alod_dhier_inspection.v1"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_span(
    path: Path, offset: int, byte_size: int, chunk_size: int = 8 * 1024 * 1024
) -> str:
    digest = hashlib.sha256()
    remaining = int(byte_size)
    with path.open("rb") as handle:
        handle.seek(int(offset))
        while remaining:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                raise OSError(f"short read while hashing {path} at byte {handle.tell()}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"inspector_project_commit": commit, "inspector_project_dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"inspector_project_commit": None, "inspector_project_dirty": None}


def build_inspection_report(
    asset: DhierAsset,
    *,
    validate_tree: bool = False,
    skybox_count: int = 0,
    profile: ValidationProfile | str = ValidationProfile.STRICT,
    max_examples: int = 32,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    numeric = validate_numeric(asset, profile=profile)
    topology = (
        validate_topology(
            asset,
            skybox_count=skybox_count,
            profile=profile,
            max_examples=max_examples,
        )
        if validate_tree
        else None
    )

    attributes: dict[str, Any] = {}
    for name, summary in numeric["arrays"].items():
        descriptor = dict(summary)
        descriptor["offset"] = int(asset.layout.offsets[name])
        descriptor["raw_sha256"] = sha256_span(
            asset.path,
            asset.layout.offsets[name],
            asset.layout.byte_sizes[name],
        )
        attributes[name] = descriptor

    hierarchy_summary: dict[str, Any] = {
        "dtype": str(asset.nodes.dtype),
        "shape": [int(value) for value in asset.nodes.shape],
        "byte_size": int(asset.nodes.nbytes),
        "offset": int(asset.layout.offsets["nodes"]),
        "raw_sha256": sha256_span(
            asset.path,
            asset.layout.offsets["nodes"],
            asset.layout.byte_sizes["nodes"],
        ),
        "field_order": [
            "depth",
            "parent",
            "child_count",
            "first_child",
            "next_sibling",
            "max_side_or_index",
        ],
        "topology_validation_performed": validate_tree,
    }
    if topology is not None:
        hierarchy_summary.update(topology)

    topology_valid: bool | None = topology["valid"] if topology is not None else None
    overall_valid = bool(numeric["valid"] and (topology_valid is not False))
    root = repo_root or Path(__file__).resolve().parents[1]
    provenance = {
        **_git_provenance(root),
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "fidelity_notes": [
            "project parser is an independent binary-format implementation",
            "official native loader is used only by separate oracle tooling",
            "Run005 is a correctness-patched local reproduction, not a bit-identical pinned-source reproduction",
        ],
    }
    layout = asset.layout
    return {
        "schema_version": SCHEMA_VERSION,
        "asset_path": str(asset.path),
        "asset_size": layout.actual_file_bytes,
        "asset_sha256": sha256_file(asset.path),
        "platform_assumption": {
            "label": "PINNED-ASSET PLATFORM ASSUMPTION",
            "producer_consumer_platform": "validated x86-64 Linux",
            "decoded_byte_order": "explicit little-endian",
            "native_scalar_widths": {"int": 32, "float": 32},
            "format_has_endianness_marker": False,
            "cross_endian_supported": False,
        },
        "identity": {
            "global_node_id": "serialized row/index",
            "not_serialized_identities": [
                "upper_tree_local_id",
                "spt_id",
                "cache_slot",
                "frame_local_gaussian_index",
            ],
        },
        "format": {
            "gaussian_count": layout.gaussian_count,
            "node_count": layout.node_count,
            "sh_degree": layout.sh_degree,
            "sh_coefficients_per_gaussian": layout.sh_coefficients_per_gaussian,
            "bytes_per_gaussian_attributes": layout.bytes_per_gaussian_attributes,
            "bytes_per_node_metadata": layout.bytes_per_node_metadata,
            "total_attribute_bytes": layout.total_attribute_bytes,
            "total_metadata_bytes": layout.total_metadata_bytes,
            "expected_file_bytes": layout.expected_file_bytes,
            "actual_file_bytes": layout.actual_file_bytes,
            "header_and_count_bytes": 12,
        },
        "attributes": attributes,
        "attribute_validation": {
            key: value for key, value in numeric.items() if key != "arrays"
        },
        "hierarchy": hierarchy_summary,
        "official_load_stages": {
            "serialized": {
                "opacity": "activated opacity",
                "scale": "log scale",
                "rotation": "raw, not normalized",
            },
            "native_dynamic_loader": "exact raw numeric load; no activation transform",
            "GaussianModel.create_from_hier_stored": [
                "positions, SH, log scales, and raw rotations remain unchanged",
                "activated opacity is clamped in-place to [0, 0.99999] then inverse_sigmoid is applied",
                "the creator's ellipse-area merge can serialize interior opacity values above 1 before that clamp",
                "node metadata column 5 is set to zero",
                "when a scaffold path is supplied, external skybox attributes/nodes are prepended and hierarchy links are offset",
            ],
            "GaussianModel_render_ready_views": [
                "get_scaling applies exp",
                "get_rotation applies torch.nn.functional.normalize",
                "get_opacity applies sigmoid to stored opacity logits",
            ],
        },
        "validation": {
            "format_valid": True,
            "numeric_valid": bool(numeric["valid"]),
            "topology_requested": validate_tree,
            "topology_valid": topology_valid,
            "overall_valid": overall_valid,
        },
        "provenance": provenance,
    }


def inspect_dhier(
    path: str | Path,
    *,
    validate_tree: bool = False,
    skybox_count: int = 0,
    profile: ValidationProfile | str = ValidationProfile.STRICT,
    max_examples: int = 32,
    strict: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    asset = load_dhier(path)
    report = build_inspection_report(
        asset,
        validate_tree=validate_tree,
        skybox_count=skybox_count,
        profile=profile,
        max_examples=max_examples,
        repo_root=repo_root,
    )
    if strict and not report["validation"]["numeric_valid"]:
        raise DhierNumericError("numeric/attribute validation failed", report)
    if strict and validate_tree and not report["validation"]["topology_valid"]:
        raise DhierTopologyError("hierarchy topology validation failed", report)
    return report


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
