"""Binary layout and read-only parser for the pinned ALoD ``.dhier`` stream.

PINNED-ASSET PLATFORM ASSUMPTION
--------------------------------
The official writer uses native C++ ``int`` and ``float`` writes.  The format
has no magic, schema/version field, checksum, or byte-order marker.  The
validated assets were written and consumed on x86-64 Linux, where those fields
are little-endian 32-bit values.  This parser therefore uses explicit ``<i4``
and ``<f4`` decoding and refuses to run on a big-endian host.  That is an asset
compatibility assumption, not a claim that the upstream format specifies
little-endian portability.

Serialized array row index is the canonical ``global_node_id``.  Upper-tree
local IDs, SPT IDs, cache slots, and frame-local Gaussian indices are not part
of this format layer.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


OFFICIAL_SOURCE_COMMIT = "73f547341547ae021ff036ad74fdf73a1240b450"
SUPPORTED_SH_DEGREES = (0, 1, 2, 3)
FLOAT32_BYTES = 4
INT32_BYTES = 4
NODE_FIELD_COUNT = 6
NODE_BYTES = NODE_FIELD_COUNT * INT32_BYTES
HEADER_BYTES = 2 * INT32_BYTES
NODE_COUNT_BYTES = INT32_BYTES


class DhierError(Exception):
    """Base class for project-owned ``.dhier`` inspection errors."""

    category = "dhier"

    def __init__(self, message: str, *, details: Mapping[str, object] | None = None):
        super().__init__(message)
        self.details = dict(details or {})


class DhierIOError(DhierError):
    category = "io"


class DhierPlatformError(DhierError):
    category = "platform"


class DhierFormatError(DhierError):
    category = "format"


class DhierCountMismatchError(DhierFormatError):
    category = "format"


@dataclass(frozen=True)
class DhierLayout:
    gaussian_count: int
    node_count: int
    sh_degree: int
    sh_coefficients_per_gaussian: int
    bytes_per_gaussian_attributes: int
    bytes_per_node_metadata: int
    total_attribute_bytes: int
    total_metadata_bytes: int
    expected_file_bytes: int
    actual_file_bytes: int
    offsets: Mapping[str, int]
    byte_sizes: Mapping[str, int]


@dataclass(frozen=True)
class DhierAsset:
    """Read-only memory-mapped dynamic hierarchy.

    Every first-axis array row has the same stable identity.  In other words,
    ``global_node_id == serialized row index``.
    """

    path: Path
    layout: DhierLayout
    positions: np.memmap
    rotations: np.memmap
    log_scales: np.memmap
    activated_opacities: np.memmap
    sh_coefficients: np.memmap
    nodes: np.memmap

    @property
    def global_node_id_contract(self) -> str:
        return "serialized_row_index"

    def gaussian_row(self, global_node_id: int) -> dict[str, np.ndarray]:
        index = int(global_node_id)
        if not 0 <= index < self.layout.gaussian_count:
            raise IndexError(f"global_node_id {index} outside serialized rows")
        return {
            "position": self.positions[index],
            "rotation": self.rotations[index],
            "log_scale": self.log_scales[index],
            "activated_opacity": self.activated_opacities[index],
            "sh_coefficients": self.sh_coefficients[index],
            "hierarchy_node": self.nodes[index],
        }


def _read_exact(path: Path, offset: int, size: int) -> bytes:
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(size)
    except OSError as exc:
        raise DhierIOError(f"cannot read {path}: {exc}") from exc
    if len(payload) != size:
        raise DhierFormatError(
            f"truncated .dhier field at byte {offset}: expected {size}, got {len(payload)}",
            details={"offset": offset, "expected_bytes": size, "actual_bytes": len(payload)},
        )
    return payload


def _checked_positive_count(value: int, name: str, file_size: int) -> None:
    if value <= 0:
        raise DhierFormatError(
            f"{name} must be positive, got {value}", details={name: value}
        )
    # Even the smallest row consumes more than four bytes.  This file-derived
    # bound rejects corrupted counts before any multiplication or mmap request.
    if value > file_size // FLOAT32_BYTES:
        raise DhierFormatError(
            f"{name} {value} is not sensible for a {file_size}-byte file",
            details={name: value, "actual_file_bytes": file_size},
        )


def load_dhier(path: str | Path) -> DhierAsset:
    """Parse and memory-map a pinned-platform dynamic hierarchy read-only."""
    if sys.byteorder != "little":
        raise DhierPlatformError(
            "cross-endian .dhier parsing is unsupported because the official "
            "native stream has no byte-order marker",
            details={"host_byteorder": sys.byteorder},
        )

    resolved = Path(path).expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError as exc:
        raise DhierIOError(f"cannot stat {resolved}: {exc}") from exc
    if not resolved.is_file():
        raise DhierIOError(f"not a regular file: {resolved}")
    file_size = int(stat.st_size)
    if file_size < HEADER_BYTES + NODE_COUNT_BYTES:
        raise DhierFormatError(
            f"truncated .dhier header/payload: {file_size} bytes",
            details={"actual_file_bytes": file_size},
        )

    gaussian_count, sh_degree = struct.unpack(
        "<ii", _read_exact(resolved, 0, HEADER_BYTES)
    )
    _checked_positive_count(gaussian_count, "gaussian_count", file_size)
    if sh_degree not in SUPPORTED_SH_DEGREES:
        raise DhierFormatError(
            f"unsupported SH degree {sh_degree}; expected one of {SUPPORTED_SH_DEGREES}",
            details={"sh_degree": sh_degree},
        )

    coefficient_count = (sh_degree + 1) ** 2
    offsets: dict[str, int] = {"positions": HEADER_BYTES}
    byte_sizes: dict[str, int] = {
        "positions": gaussian_count * 3 * FLOAT32_BYTES,
        "rotations": gaussian_count * 4 * FLOAT32_BYTES,
        "log_scales": gaussian_count * 3 * FLOAT32_BYTES,
        "activated_opacities": gaussian_count * FLOAT32_BYTES,
        "sh_coefficients": gaussian_count * coefficient_count * 3 * FLOAT32_BYTES,
    }
    offsets["rotations"] = offsets["positions"] + byte_sizes["positions"]
    offsets["log_scales"] = offsets["rotations"] + byte_sizes["rotations"]
    offsets["activated_opacities"] = (
        offsets["log_scales"] + byte_sizes["log_scales"]
    )
    offsets["sh_coefficients"] = (
        offsets["activated_opacities"] + byte_sizes["activated_opacities"]
    )
    offsets["node_count"] = (
        offsets["sh_coefficients"] + byte_sizes["sh_coefficients"]
    )

    if offsets["node_count"] + NODE_COUNT_BYTES > file_size:
        raise DhierFormatError(
            "truncated .dhier attribute payload before node count",
            details={
                "node_count_offset": offsets["node_count"],
                "actual_file_bytes": file_size,
            },
        )
    (node_count,) = struct.unpack(
        "<i", _read_exact(resolved, offsets["node_count"], NODE_COUNT_BYTES)
    )
    _checked_positive_count(node_count, "node_count", file_size)
    if node_count != gaussian_count:
        raise DhierCountMismatchError(
            f"Gaussian/node count mismatch: {gaussian_count} != {node_count}",
            details={"gaussian_count": gaussian_count, "node_count": node_count},
        )

    offsets["nodes"] = offsets["node_count"] + NODE_COUNT_BYTES
    byte_sizes["nodes"] = node_count * NODE_BYTES
    expected_size = offsets["nodes"] + byte_sizes["nodes"]
    if expected_size < file_size:
        raise DhierFormatError(
            f"trailing data: expected {expected_size} bytes, got {file_size}",
            details={
                "expected_file_bytes": expected_size,
                "actual_file_bytes": file_size,
                "trailing_bytes": file_size - expected_size,
            },
        )
    if expected_size > file_size:
        raise DhierFormatError(
            f"truncated payload: expected {expected_size} bytes, got {file_size}",
            details={
                "expected_file_bytes": expected_size,
                "actual_file_bytes": file_size,
                "missing_bytes": expected_size - file_size,
            },
        )

    attribute_floats = 3 + 4 + 3 + 1 + coefficient_count * 3
    layout = DhierLayout(
        gaussian_count=gaussian_count,
        node_count=node_count,
        sh_degree=sh_degree,
        sh_coefficients_per_gaussian=coefficient_count,
        bytes_per_gaussian_attributes=attribute_floats * FLOAT32_BYTES,
        bytes_per_node_metadata=NODE_BYTES,
        total_attribute_bytes=gaussian_count * attribute_floats * FLOAT32_BYTES,
        total_metadata_bytes=node_count * NODE_BYTES,
        expected_file_bytes=expected_size,
        actual_file_bytes=file_size,
        offsets=dict(offsets),
        byte_sizes=dict(byte_sizes),
    )

    try:
        positions = np.memmap(
            resolved, dtype="<f4", mode="r", offset=offsets["positions"],
            shape=(gaussian_count, 3),
        )
        rotations = np.memmap(
            resolved, dtype="<f4", mode="r", offset=offsets["rotations"],
            shape=(gaussian_count, 4),
        )
        log_scales = np.memmap(
            resolved, dtype="<f4", mode="r", offset=offsets["log_scales"],
            shape=(gaussian_count, 3),
        )
        activated_opacities = np.memmap(
            resolved, dtype="<f4", mode="r", offset=offsets["activated_opacities"],
            shape=(gaussian_count, 1),
        )
        sh_coefficients = np.memmap(
            resolved, dtype="<f4", mode="r", offset=offsets["sh_coefficients"],
            shape=(gaussian_count, coefficient_count, 3),
        )
        nodes = np.memmap(
            resolved, dtype="<i4", mode="r", offset=offsets["nodes"],
            shape=(node_count, NODE_FIELD_COUNT),
        )
    except (OSError, ValueError) as exc:
        raise DhierIOError(f"cannot memory-map {resolved}: {exc}") from exc

    return DhierAsset(
        path=resolved,
        layout=layout,
        positions=positions,
        rotations=rotations,
        log_scales=log_scales,
        activated_opacities=activated_opacities,
        sh_coefficients=sh_coefficients,
        nodes=nodes,
    )
