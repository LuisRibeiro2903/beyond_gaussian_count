from __future__ import annotations

import argparse
import json
import math
import os
import pickletools
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


REFERENCE_COMMIT = "e16c526f78ab01d377b13b3e6862dcc93ddbe953"
SCHEMA_VERSION = "citygaussian_asset_manifest.v1"
REQUIRED_VQ_FILES = (
    "codebook.npz",
    "metadata.npz",
    "non_vq_feats.npz",
    "non_vq_mask.npz",
    "other_attribute.npz",
    "vq_indexs.npz",
    "xyz.npz",
)
PLY_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass(frozen=True)
class PlyHeader:
    path: Path
    file_size_bytes: int
    header_bytes: int
    ply_format: str
    vertex_count: int
    properties: tuple[tuple[str, str], ...]
    comments: tuple[str, ...]

    @property
    def property_names(self) -> list[str]:
        return [name for name, _ in self.properties]

    @property
    def dtype(self) -> np.dtype:
        return np.dtype([(name, PLY_SCALAR_DTYPES[prop_type]) for name, prop_type in self.properties])

    @property
    def expected_size_bytes(self) -> int:
        return self.header_bytes + self.vertex_count * self.dtype.itemsize


def read_ply_header(path: Path) -> PlyHeader:
    path = Path(path)
    file_size = path.stat().st_size
    header: list[str] = []
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if line == b"":
                raise ValueError(f"{path} ended before PLY end_header.")
            try:
                decoded = line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path} has a non-ASCII PLY header.") from exc
            header.append(decoded)
            if decoded == "end_header":
                header_bytes = f.tell()
                break

    if not header or header[0] != "ply":
        raise ValueError(f"{path} is not a PLY file.")
    if len(header) < 2:
        raise ValueError(f"{path} has no PLY format line.")
    format_parts = header[1].split()
    if len(format_parts) != 3 or format_parts[0] != "format":
        raise ValueError(f"{path} has malformed PLY format line: {header[1]!r}.")
    ply_format = format_parts[1]
    if ply_format not in {"binary_little_endian", "binary_big_endian", "ascii"}:
        raise ValueError(f"{path} has unsupported PLY format {ply_format!r}.")

    vertex_count: int | None = None
    properties: list[tuple[str, str]] = []
    comments: list[str] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "comment":
            comments.append(line[len("comment") :].strip())
            continue
        if parts[0] == "element":
            if len(parts) != 3:
                raise ValueError(f"{path} has malformed element line: {line!r}.")
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
            continue
        if in_vertex and parts[0] == "property":
            if len(parts) == 5 and parts[1] == "list":
                raise ValueError(f"{path} contains unsupported vertex list property {line!r}.")
            if len(parts) != 3:
                raise ValueError(f"{path} has malformed vertex property line: {line!r}.")
            prop_type, name = parts[1], parts[2]
            if prop_type not in PLY_SCALAR_DTYPES:
                raise ValueError(f"{path} has unsupported property type {prop_type!r}.")
            properties.append((name, prop_type))

    if vertex_count is None:
        raise ValueError(f"{path} has no vertex element.")
    if vertex_count < 0:
        raise ValueError(f"{path} has a negative vertex count.")
    if not properties:
        raise ValueError(f"{path} has no vertex properties.")
    return PlyHeader(
        path=path,
        file_size_bytes=file_size,
        header_bytes=header_bytes,
        ply_format=ply_format,
        vertex_count=vertex_count,
        properties=tuple(properties),
        comments=tuple(comments),
    )


def infer_sh_degree(property_names: Iterable[str]) -> dict[str, Any]:
    names = list(property_names)
    f_dc = _numbered_names(names, "f_dc_")
    f_rest = _numbered_names(names, "f_rest_")
    errors: list[str] = []
    if len(f_dc) != 3:
        errors.append(f"expected exactly 3 f_dc_* properties, found {len(f_dc)}")
    if len(f_rest) % 3 != 0:
        errors.append(f"expected f_rest_* count to be divisible by 3, found {len(f_rest)}")
    coeffs_per_channel = 1 + len(f_rest) // 3 if len(f_rest) % 3 == 0 else None
    degree: int | None = None
    if coeffs_per_channel is not None:
        root = math.sqrt(coeffs_per_channel)
        if root.is_integer():
            degree = int(root - 1)
        else:
            errors.append(
                f"{coeffs_per_channel} SH coefficients per channel does not map to an integer degree"
            )
    return {
        "degree": degree,
        "f_dc_count": len(f_dc),
        "f_rest_count": len(f_rest),
        "coefficients_per_channel": coeffs_per_channel,
        "valid": not errors,
        "errors": errors,
    }


def summarize_ply(path: Path, role: str) -> dict[str, Any]:
    try:
        header = read_ply_header(path)
    except Exception as exc:
        return {
            "path": str(path),
            "official_asset_role": role,
            "exists": Path(path).exists(),
            "validation_status": "invalid",
            "errors": [str(exc)],
        }

    property_names = header.property_names
    sh = infer_sh_degree(property_names)
    required = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
    missing = [name for name in required if name not in property_names]
    errors = list(sh["errors"])
    if missing:
        errors.append(f"missing required Gaussian properties: {missing}")
    if header.ply_format != "binary_little_endian":
        errors.append(f"expected binary_little_endian, found {header.ply_format}")
    if header.file_size_bytes < header.expected_size_bytes:
        errors.append(
            f"file is truncated: expected at least {header.expected_size_bytes} bytes, got {header.file_size_bytes}"
        )
    warnings = []
    if header.file_size_bytes > header.expected_size_bytes:
        warnings.append(
            f"file has {header.file_size_bytes - header.expected_size_bytes} trailing bytes"
        )
    return {
        "path": str(path),
        "official_asset_role": role,
        "exists": True,
        "file_size_bytes": header.file_size_bytes,
        "ply_format": header.ply_format,
        "header_bytes": header.header_bytes,
        "gaussian_vertex_count": header.vertex_count,
        "available_gaussian_properties": property_names,
        "property_types": {name: prop_type for name, prop_type in header.properties},
        "property_count": len(property_names),
        "inferred_sh_degree": sh["degree"],
        "sh_metadata": sh,
        "finite_header_property_metadata": {
            "vertex_count_is_finite": math.isfinite(header.vertex_count),
            "file_size_is_finite": math.isfinite(header.file_size_bytes),
            "expected_body_size_bytes": header.expected_size_bytes,
            "body_size_matches_header": header.file_size_bytes == header.expected_size_bytes,
        },
        "warnings": warnings,
        "errors": errors,
        "validation_status": "valid" if not errors else "invalid",
    }


def inspect_npz_file(path: Path) -> dict[str, Any]:
    path = Path(path)
    result: dict[str, Any] = {
        "path": str(path),
        "file_name": path.name,
        "exists": path.exists(),
        "arrays": [],
        "metadata_fields": {},
        "pickle_executed": False,
        "errors": [],
    }
    if not path.exists():
        result["errors"].append("missing")
        return result

    try:
        loaded = np.load(path, allow_pickle=False)
        for array_name in loaded.files:
            array = loaded[array_name]
            result["arrays"].append(
                {
                    "array_name": array_name,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "element_count": int(array.size),
                }
            )
        return result
    except ValueError as exc:
        if "Object arrays cannot be loaded" not in str(exc):
            result["errors"].append(str(exc))
            return result
    except Exception as exc:
        result["errors"].append(str(exc))
        return result

    with zipfile.ZipFile(path) as archive:
        for member_name in archive.namelist():
            with archive.open(member_name) as member:
                version = np.lib.format.read_magic(member)
                shape, fortran_order, dtype = np.lib.format._read_array_header(member, version)
                payload = member.read()
                result["arrays"].append(
                    {
                        "array_name": member_name.removesuffix(".npy"),
                        "member_name": member_name,
                        "shape": list(shape),
                        "dtype": str(dtype),
                        "element_count": int(np.prod(shape, dtype=np.int64)) if shape else 1,
                        "fortran_order": bool(fortran_order),
                        "object_array_header_only": dtype.hasobject,
                    }
                )
                if dtype.hasobject and path.name == "metadata.npz":
                    result["metadata_fields"] = _extract_simple_metadata_from_pickle(payload)
                    result["metadata_load_mode"] = "pickletools_disassembly_no_execution"
    return result


def inspect_vq_directory(extreme_saving_dir: Path, adjacent_ply_count: int | None = None) -> dict[str, Any]:
    extreme_saving_dir = Path(extreme_saving_dir)
    files = {name: inspect_npz_file(extreme_saving_dir / name) for name in REQUIRED_VQ_FILES}
    errors: list[str] = []
    for name, info in files.items():
        if not info["exists"]:
            errors.append(f"missing {name}")
        errors.extend(f"{name}: {error}" for error in info["errors"])

    metadata = files["metadata.npz"].get("metadata_fields", {})
    input_pc_num = _as_int(metadata.get("input_pc_num"))
    input_pc_dim = _as_int(metadata.get("input_pc_dim"))
    codebook_size = _as_int(metadata.get("codebook_size"))
    codebook_dim = _as_int(metadata.get("codebook_dim"))
    bit_length = int(math.log2(codebook_size)) if codebook_size and codebook_size > 0 else None

    shapes = _npz_primary_shapes(files)
    xyz_count = _shape_dim0(shapes.get("xyz.npz"))
    other_count = _shape_dim0(shapes.get("other_attribute.npz"))
    non_vq_count = _shape_dim0(shapes.get("non_vq_feats.npz"))
    vq_count = input_pc_num - non_vq_count if input_pc_num is not None and non_vq_count is not None else None
    expected_vq_index_bytes = (
        (vq_count * bit_length + 7) // 8 if vq_count is not None and bit_length is not None else None
    )
    actual_vq_index_bytes = _element_count(files["vq_indexs.npz"])
    one_to_one = bool(
        adjacent_ply_count is not None
        and input_pc_num == adjacent_ply_count
        and xyz_count == adjacent_ply_count
        and other_count == adjacent_ply_count
    )
    if adjacent_ply_count is not None and input_pc_num != adjacent_ply_count:
        errors.append(
            f"metadata input_pc_num {input_pc_num} does not match adjacent PLY count {adjacent_ply_count}"
        )
    if xyz_count is not None and input_pc_num is not None and xyz_count != input_pc_num:
        errors.append(f"xyz count {xyz_count} does not match metadata input_pc_num {input_pc_num}")
    if other_count is not None and input_pc_num is not None and other_count != input_pc_num:
        errors.append(
            f"other_attribute count {other_count} does not match metadata input_pc_num {input_pc_num}"
        )
    if expected_vq_index_bytes is not None and actual_vq_index_bytes is not None:
        if actual_vq_index_bytes != expected_vq_index_bytes:
            errors.append(
                f"vq_indexs byte count {actual_vq_index_bytes} does not match expected {expected_vq_index_bytes}"
            )

    return {
        "path": str(extreme_saving_dir),
        "npz_file_names": list(REQUIRED_VQ_FILES),
        "files": files,
        "metadata_fields": metadata,
        "reconstructed_gaussian_count": input_pc_num,
        "input_pc_dim": input_pc_dim,
        "codebook_size": codebook_size,
        "codebook_dim": codebook_dim,
        "bit_length": bit_length,
        "non_vq_gaussian_count": non_vq_count,
        "vq_gaussian_count": vq_count,
        "vq_index_bytes": actual_vq_index_bytes,
        "expected_vq_index_bytes": expected_vq_index_bytes,
        "vector_quantized_attributes": {
            "source_function": "utils.vq_utils.load_vqgaussian",
            "columns": [6, 6 + codebook_dim] if codebook_dim is not None else None,
            "attributes": "SH color features only: f_dc_0..f_dc_2 and f_rest_0..f_rest_23 for SH degree 2",
            "selection": "Gaussians where unpacked non_vq_mask is false use codebook/vq_indexs.",
        },
        "non_vq_attributes": {
            "all_gaussians": "xyz and other_attribute (opacity, scale_0..2, rot_0..3)",
            "non_vq_mask_true": "SH color features stored directly in non_vq_feats",
            "implicit": "nx, ny, nz are zero-filled by load_vqgaussian",
        },
        "one_to_one_with_adjacent_ply": one_to_one,
        "official_source_function_loads_it": "utils.vq_utils.load_vqgaussian via scene.gaussian_model.GaussianModelLOD.load_vq",
        "errors": errors,
        "validation_status": "valid" if not errors else "invalid",
    }


def parse_citygaussian_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    model = data.get("model_params", {}) if isinstance(data, dict) else {}
    return {
        "path": str(path),
        "model_config": model.get("model_config"),
        "sh_degree": model.get("sh_degree"),
        "source_path": model.get("source_path"),
        "partition_name": model.get("partition_name"),
        "model_path": model.get("model_path"),
        "images": model.get("images"),
        "block_dim": model.get("block_dim"),
        "aabb": model.get("aabb"),
        "lod_configs": model.get("lod_configs") or [],
        "white_background": model.get("white_background"),
        "data_device": model.get("data_device"),
        "eval": model.get("eval"),
        "pipeline_params": data.get("pipeline_params", {}) if isinstance(data, dict) else {},
    }


def compute_citygaussian_cell_counts(
    ply_path: Path,
    block_dim: list[int],
    aabb: list[float],
    chunk_size: int = 1_000_000,
) -> dict[str, Any]:
    header = read_ply_header(ply_path)
    if header.ply_format != "binary_little_endian":
        raise ValueError("cell counting supports binary_little_endian PLY only")
    names = header.property_names
    for name in ("x", "y", "z"):
        if name not in names:
            raise ValueError(f"{ply_path} has no {name} property for cell counting")

    arr = np.memmap(ply_path, dtype=header.dtype, mode="r", offset=header.header_bytes, shape=(header.vertex_count,))
    cell_count = int(np.prod(block_dim, dtype=np.int64))
    counts = np.zeros(cell_count, dtype=np.int64)
    unassigned = 0
    duplicated = 0
    finite_xyz = True
    cell_positions: list[list[np.ndarray]] = [[] for _ in range(cell_count)]
    cell_scales: list[list[np.ndarray]] = [[] for _ in range(cell_count)]
    for start in range(0, header.vertex_count, chunk_size):
        stop = min(start + chunk_size, header.vertex_count)
        xyz = np.stack(
            [
                np.asarray(arr[name][start:stop], dtype=np.float32)
                for name in ("x", "y", "z")
            ],
            axis=1,
        )
        scales = np.stack(
            [
                np.asarray(arr[name][start:stop], dtype=np.float32)
                for name in ("scale_0", "scale_1", "scale_2")
            ],
            axis=1,
        )
        finite_xyz = bool(finite_xyz and np.isfinite(xyz).all())
        contracted = contract_to_unisphere_np(xyz, aabb)
        assignment_count = np.zeros(xyz.shape[0], dtype=np.int16)
        for cell_id in range(cell_count):
            mask = citygaussian_official_cell_mask_np(
                contracted,
                cell_id=cell_id,
                block_dim=block_dim,
                scale=1.0,
            )
            assignment_count[mask] += 1
            counts[cell_id] += int(mask.sum())
            if not np.any(mask):
                continue
            cell_positions[int(cell_id)].append(xyz[mask])
            cell_scales[int(cell_id)].append(np.exp(scales[mask]))
        unassigned += int((assignment_count == 0).sum())
        duplicated += int((assignment_count > 1).sum())

    mad_bounds = []
    average_scales = []
    for cell_id in range(cell_count):
        if counts[cell_id] == 0:
            mad_bounds.append({"cell_id": cell_id, "min": None, "max": None})
            average_scales.append(None)
            continue
        positions = np.concatenate(cell_positions[cell_id], axis=0)
        scales = np.concatenate(cell_scales[cell_id], axis=0)
        xyz_min, xyz_max = citygaussian_mad_bounds_np(positions, n=4.0)
        mad_bounds.append(
            {
                "cell_id": cell_id,
                "min": [float(v) for v in xyz_min.tolist()],
                "max": [float(v) for v in xyz_max.tolist()],
            }
        )
        average_scales.append(float(np.mean(scales, axis=0).max()))

    return {
        "block_dim": list(block_dim),
        "aabb": list(aabb),
        "cell_count": int(counts.shape[0]),
        "cell_ids": list(range(cell_count)),
        "cell_gaussian_counts": [int(v) for v in counts.tolist()],
        "total_assigned_count": int(counts.sum()),
        "unassigned_count": int(unassigned),
        "duplicated_assignment_count": int(duplicated),
        "nonempty_cell_count": int(np.count_nonzero(counts)),
        "all_cells_nonempty": bool(np.all(counts > 0)),
        "min_cell_gaussian_count": int(counts.min()) if counts.size else 0,
        "max_cell_gaussian_count": int(counts.max()) if counts.size else 0,
        "mad_bounds": mad_bounds,
        "average_scales": average_scales,
        "finite_xyz": finite_xyz,
    }


def citygaussian_official_assignments(
    xyz: np.ndarray,
    block_dim: list[int],
    aabb: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    dims = np.asarray(block_dim, dtype=np.int64)
    contracted = contract_to_unisphere_np(xyz, aabb)
    assigned = np.isfinite(contracted).all(axis=1)
    assigned &= np.all(contracted >= 0.0, axis=1)
    assigned &= np.all(contracted < 1.0, axis=1)
    ids = np.floor(contracted * dims[None, :]).astype(np.int64)
    ids = np.clip(ids, 0, dims[None, :] - 1)
    flat_ids = ids[:, 2] * dims[0] * dims[1] + ids[:, 1] * dims[0] + ids[:, 0]
    return assigned, flat_ids


def citygaussian_official_cell_mask_np(
    contracted_xyz: np.ndarray,
    *,
    cell_id: int,
    block_dim: list[int],
    scale: float = 1.0,
) -> np.ndarray:
    dims = [int(v) for v in block_dim]
    block_id_z = int(cell_id) // (dims[0] * dims[1])
    block_id_y = (int(cell_id) % (dims[0] * dims[1])) // dims[0]
    block_id_x = (int(cell_id) % (dims[0] * dims[1])) % dims[0]

    min_x, max_x = float(block_id_x) / dims[0], float(block_id_x + 1) / dims[0]
    min_y, max_y = float(block_id_y) / dims[1], float(block_id_y + 1) / dims[1]
    min_z, max_z = float(block_id_z) / dims[2], float(block_id_z + 1) / dims[2]
    delta_x = (max_x - min_x) * (float(scale) - 1.0)
    delta_y = (max_y - min_y) * (float(scale) - 1.0)
    delta_z = (max_z - min_z) * (float(scale) - 1.0)
    min_x -= delta_x / 2.0
    max_x += delta_x / 2.0
    min_y -= delta_y / 2.0
    max_y += delta_y / 2.0
    min_z -= delta_z / 2.0
    max_z += delta_z / 2.0

    return (
        (contracted_xyz[:, 0] >= min_x)
        & (contracted_xyz[:, 0] < max_x)
        & (contracted_xyz[:, 1] >= min_y)
        & (contracted_xyz[:, 1] < max_y)
        & (contracted_xyz[:, 2] >= min_z)
        & (contracted_xyz[:, 2] < max_z)
    )


def citygaussian_block_ids(xyz: np.ndarray, block_dim: list[int], aabb: list[float]) -> np.ndarray:
    _assigned, flat_ids = citygaussian_official_assignments(xyz, block_dim=block_dim, aabb=aabb)
    return flat_ids


def citygaussian_mad_bounds_np(xyz: np.ndarray, n: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] == 0:
        raise ValueError("Cannot compute MAD bounds for an empty cell.")
    xyz_median = _torch_like_median_np(xyz, axis=0)
    delta_median = _torch_like_median_np(np.abs(xyz - xyz_median[None, :]), axis=0)
    xyz_min = xyz_median - float(n) * delta_median
    xyz_min = np.maximum(xyz_min, np.min(xyz, axis=0))
    xyz_max = xyz_median + float(n) * delta_median
    xyz_max = np.minimum(xyz_max, np.max(xyz, axis=0))
    return xyz_min.astype(np.float32), xyz_max.astype(np.float32)


def _torch_like_median_np(values: np.ndarray, axis: int = 0) -> np.ndarray:
    sorted_values = np.sort(values, axis=axis)
    index = (values.shape[axis] - 1) // 2
    return np.take(sorted_values, index, axis=axis)


def contract_to_unisphere_np(xyz: np.ndarray, aabb: list[float]) -> np.ndarray:
    bounds = np.asarray(aabb, dtype=np.float32)
    mins = bounds[:3]
    maxs = bounds[3:]
    x = (xyz.astype(np.float32, copy=True) - mins[None, :]) / (maxs - mins)[None, :]
    x = x * 2.0 - 1.0
    mag = np.max(np.abs(x), axis=1)
    mask = mag > 1.0
    if np.any(mask):
        m = mag[mask, None]
        x[mask] = (2.0 - 1.0 / m) * x[mask] / m
    return x / 4.0 + 0.5


def validate_assets(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    reference_repo = Path(args.reference_repo) if args.reference_repo else None
    fused_model = Path(args.fused_model) if args.fused_model else None
    lod_root = Path(args.lod_root) if args.lod_root else None
    camera_metadata = Path(args.camera_metadata) if args.camera_metadata else None
    yaml_path = Path(args.yaml) if args.yaml else None

    errors: list[str] = []
    warnings: list[str] = []
    yaml_info: dict[str, Any] = {}
    if yaml_path is not None:
        try:
            yaml_info = parse_citygaussian_yaml(yaml_path)
        except Exception as exc:
            errors.append(f"failed to parse YAML {yaml_path}: {exc}")

    block_dim = list(args.block_dim or yaml_info.get("block_dim") or [])
    aabb = list(args.aabb or yaml_info.get("aabb") or [])

    fused_summary = summarize_ply(fused_model, "fused_no_lod_model") if fused_model else None
    if fused_summary and fused_summary["validation_status"] != "valid":
        errors.extend(f"fused_model: {error}" for error in fused_summary["errors"])

    lod_models: list[dict[str, Any]] = []
    vq_metadata: list[dict[str, Any]] = []
    lod_order: list[dict[str, Any]] = []
    yaml_lod_names = [Path(item).name for item in yaml_info.get("lod_configs", [])]
    if lod_root is not None:
        for index, expected_name in enumerate(yaml_lod_names):
            level_dir = lod_root / expected_name
            ply_path = level_dir / "point_cloud.ply"
            if not level_dir.exists():
                errors.append(f"missing LoD directory {level_dir}")
                lod_order.append({"yaml_index": index, "yaml_path": yaml_info["lod_configs"][index], "exists": False})
                continue
            ply_summary = summarize_ply(ply_path, f"lod_level_{index}")
            if block_dim and aabb and not args.skip_cell_counts and ply_summary["validation_status"] == "valid":
                ply_summary["cell_counts"] = compute_citygaussian_cell_counts(ply_path, block_dim, aabb)
                if not ply_summary["cell_counts"]["all_cells_nonempty"]:
                    errors.append(f"{expected_name}: not every official cell has Gaussians")
            if ply_summary["validation_status"] != "valid":
                errors.extend(f"{expected_name}: {error}" for error in ply_summary["errors"])
            vq_info = inspect_vq_directory(
                level_dir / "extreme_saving",
                adjacent_ply_count=ply_summary.get("gaussian_vertex_count"),
            )
            if vq_info["validation_status"] != "valid":
                errors.extend(f"{expected_name}: {error}" for error in vq_info["errors"])
            lod_models.append(
                {
                    "name": expected_name,
                    "directory": str(level_dir),
                    "yaml_index": index,
                    "yaml_path": yaml_info["lod_configs"][index],
                    "ply": ply_summary,
                }
            )
            vq_metadata.append({"name": expected_name, **vq_info})
            lod_order.append(
                {
                    "yaml_index": index,
                    "yaml_path": yaml_info["lod_configs"][index],
                    "directory_name": expected_name,
                    "gaussian_count": ply_summary.get("gaussian_vertex_count"),
                    "inferred_sh_degree": ply_summary.get("inferred_sh_degree"),
                    "source_runtime_print_label": f"LoD {len(yaml_lod_names) - index}",
                    "semantic_order": ["finest", "middle", "coarsest"][index]
                    if len(yaml_lod_names) == 3
                    else None,
                }
            )
        actual_dirs = sorted(path.name for path in lod_root.iterdir() if path.is_dir()) if lod_root.exists() else []
        missing_from_yaml = [name for name in yaml_lod_names if name not in actual_dirs]
        extra_dirs = [name for name in actual_dirs if name not in yaml_lod_names]
        if missing_from_yaml:
            errors.append(f"LoD root is missing YAML levels: {missing_from_yaml}")
        if extra_dirs:
            warnings.append(f"LoD root contains directories not listed by YAML: {extra_dirs}")
    elif yaml_lod_names:
        warnings.append("YAML declares LoD configs but no --lod-root was supplied.")

    camera_info = _camera_metadata_summary(camera_metadata) if camera_metadata else None
    if camera_info and camera_info["validation_status"] != "valid":
        errors.extend(f"camera_metadata: {error}" for error in camera_info["errors"])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": "CityGaussian V1",
        "scene": args.scene,
        "reference_repository_commit": args.reference_repository_commit,
        "reference_repository": _git_reference_summary(reference_repo, args.reference_repository_commit)
        if reference_repo
        else None,
        "fused_model": fused_summary,
        "lod_models": lod_models,
        "lod_order": lod_order,
        "block_dim": block_dim,
        "aabb": aabb,
        "camera_metadata": camera_info,
        "yaml": yaml_info,
        "gaussian_counts": _gaussian_counts(fused_summary, lod_models),
        "sh_degrees": _sh_degrees(fused_summary, lod_models),
        "vq_metadata": vq_metadata,
        "runtime_construction": {
            "loads_whole_scene_lod_models": True,
            "load_path": "render_large_lod.py::render_sets -> load_gaussians",
            "vq_load_path": "scene/__init__.py::LargeScene.__init__ -> GaussianModelLOD.load_vq -> utils.vq_utils.load_vqgaussian",
            "cell_construction_path": "scene/gaussian_model.py::BlockedGaussian.cell_divider",
            "block_assignment": "utils/large_utils.py::block_filtering via contract_to_unisphere, block_dim and aabb",
            "matrixcity_block_dim": [6, 6, 1],
            "matrixcity_cell_count": 36,
            "bounds": "MAD bounds with n=4 per cell",
            "avg_scale": "max over mean scaling vector per cell",
            "requires_pregenerated_chunk_directories": False,
            "requires_lod_meta_json": False,
        },
        "lod_semantics": _lod_semantics(fused_summary, lod_models, vq_metadata, yaml_info),
        "asset_classification": [_classify_asset(Path(path), fused_model, lod_root, camera_metadata, yaml_path) for path in args.custom_asset],
        "validation_status": "valid" if not errors else "invalid",
        "validation_errors": errors,
        "validation_warnings": warnings,
    }
    return manifest, 0 if not errors else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate CityGaussian V1 assets without CUDA.")
    parser.add_argument("--reference-repo", default="/home/luis/tese/CityGaussian")
    parser.add_argument("--reference-repository-commit", default=REFERENCE_COMMIT)
    parser.add_argument("--scene", default="MatrixCity aerial c36")
    parser.add_argument("--fused-model")
    parser.add_argument("--lod-root")
    parser.add_argument("--camera-metadata")
    parser.add_argument("--yaml")
    parser.add_argument("--custom-asset", action="append", default=[])
    parser.add_argument("--block-dim", nargs=3, type=int)
    parser.add_argument("--aabb", nargs=6, type=float)
    parser.add_argument("--skip-cell-counts", action="store_true")
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest, exit_code = validate_assets(args)
    text = json.dumps(manifest, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)
    return exit_code


def _numbered_names(property_names: list[str], prefix: str) -> list[str]:
    names = [name for name in property_names if name.startswith(prefix)]
    for name in names:
        suffix = name[len(prefix) :]
        if not suffix.isdigit():
            raise ValueError(f"property {name!r} should end with a numeric suffix")
    return sorted(names, key=lambda name: int(name[len(prefix) :]))


def _extract_simple_metadata_from_pickle(payload: bytes) -> dict[str, int]:
    allowed_keys = {"input_pc_num", "input_pc_dim", "codebook_size", "codebook_dim"}
    fields: dict[str, int] = {}
    pending_key: str | None = None
    for opcode, arg, _pos in pickletools.genops(payload):
        if opcode.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"} and isinstance(arg, str):
            pending_key = arg if arg in allowed_keys else None
            continue
        if pending_key and opcode.name in {"BININT", "BININT1", "BININT2", "LONG1", "LONG4"}:
            fields[pending_key] = int(arg)
            pending_key = None
    return fields


def _npz_primary_shapes(files: dict[str, dict[str, Any]]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for name, info in files.items():
        arrays = info.get("arrays") or []
        if arrays:
            shapes[name] = list(arrays[0].get("shape") or [])
    return shapes


def _shape_dim0(shape: list[int] | None) -> int | None:
    return int(shape[0]) if shape else None


def _element_count(info: dict[str, Any]) -> int | None:
    arrays = info.get("arrays") or []
    if not arrays:
        return None
    return int(arrays[0]["element_count"])


def _as_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _git_reference_summary(repo: Path | None, expected_commit: str) -> dict[str, Any]:
    if repo is None:
        return {}
    summary = {"path": str(repo), "expected_commit": expected_commit, "errors": []}
    for key, cmd in {
        "commit": ["git", "-C", str(repo), "rev-parse", "HEAD"],
        "branch": ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
        "status_short": ["git", "-C", str(repo), "status", "--short"],
    }.items():
        try:
            completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
            summary[key] = completed.stdout.strip()
        except Exception as exc:
            summary["errors"].append(str(exc))
    summary["commit_matches_expected"] = summary.get("commit") == expected_commit
    summary["dirty"] = bool(summary.get("status_short"))
    return summary


def _camera_metadata_summary(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text())
    except Exception as exc:
        return {"path": str(path), "exists": Path(path).exists(), "validation_status": "invalid", "errors": [str(exc)]}
    if isinstance(data, list):
        count = len(data)
        sample_keys = sorted(data[0].keys()) if data and isinstance(data[0], dict) else []
    elif isinstance(data, dict):
        count = len(data.get("cameras", [])) if isinstance(data.get("cameras"), list) else None
        sample_keys = sorted(data.keys())
    else:
        count = None
        sample_keys = []
    return {
        "path": str(path),
        "exists": True,
        "file_size_bytes": Path(path).stat().st_size,
        "camera_count": count,
        "top_level_type": type(data).__name__,
        "sample_keys": sample_keys,
        "validation_status": "valid",
        "errors": [],
    }


def _gaussian_counts(fused_summary: dict[str, Any] | None, lod_models: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    if fused_summary and fused_summary.get("gaussian_vertex_count") is not None:
        counts["fused_no_lod"] = int(fused_summary["gaussian_vertex_count"])
    for model in lod_models:
        count = model["ply"].get("gaussian_vertex_count")
        if count is not None:
            counts[model["name"]] = int(count)
    return counts


def _sh_degrees(fused_summary: dict[str, Any] | None, lod_models: list[dict[str, Any]]) -> dict[str, int]:
    degrees: dict[str, int] = {}
    if fused_summary and fused_summary.get("inferred_sh_degree") is not None:
        degrees["fused_no_lod"] = int(fused_summary["inferred_sh_degree"])
    for model in lod_models:
        degree = model["ply"].get("inferred_sh_degree")
        if degree is not None:
            degrees[model["name"]] = int(degree)
    return degrees


def _lod_semantics(
    fused_summary: dict[str, Any] | None,
    lod_models: list[dict[str, Any]],
    vq_metadata: list[dict[str, Any]],
    yaml_info: dict[str, Any],
) -> dict[str, Any]:
    fused_count = fused_summary.get("gaussian_vertex_count") if fused_summary else None
    levels = []
    for model in lod_models:
        count = model["ply"].get("gaussian_vertex_count")
        suffix = model["name"].split("_light_")[-1].split("_")[0] if "_light_" in model["name"] else None
        pruned_percent_from_count = None
        retained_percent_from_count = None
        if fused_count and count is not None:
            retained_percent_from_count = 100.0 * count / fused_count
            pruned_percent_from_count = 100.0 - retained_percent_from_count
        levels.append(
            {
                "name": model["name"],
                "name_number": int(suffix) if suffix and suffix.isdigit() else suffix,
                "gaussian_count": count,
                "retained_percent_vs_fused": retained_percent_from_count,
                "pruned_percent_vs_fused": pruned_percent_from_count,
            }
        )
    return {
        "exact_lod_order": [model["name"] for model in lod_models],
        "yaml_order_is_runtime_order": True,
        "light_50_vq_is_finest_released_lod": bool(levels and levels[0]["name"].endswith("light_50_vq")),
        "level_name_numbers_mean": "prune percentage, verified by counts and CityGaussian custom_dataset.md pruning note",
        "levels_are_complete_alternatives_not_deltas": True,
        "all_levels_cover_complete_scene": all(
            model["ply"].get("cell_counts", {}).get("all_cells_nonempty") for model in lod_models
        )
        if lod_models
        else None,
        "sh_degree_consistent_for_lod_levels": len({model["ply"].get("inferred_sh_degree") for model in lod_models}) <= 1,
        "ply_and_extreme_saving_encode_equivalent_counts": all(
            item.get("one_to_one_with_adjacent_ply") for item in vq_metadata
        )
        if vq_metadata
        else None,
        "official_runtime_loads_by_default": "VQ extreme_saving when render_large_lod.py/viewer.py is invoked with --load_vq; otherwise adjacent PLY point_cloud.ply",
        "nyquist_selector_assumes_yaml_order": True,
        "yaml_sh_degree": yaml_info.get("sh_degree"),
        "levels": levels,
        "released_policy": {
            "name": "CG-LOD-RELEASE",
            "source_function": "gaussian_renderer/__init__.py::render_lod",
            "selection": "Nyquist: first YAML-ordered level whose avg cell scale is greater than 2 * distance / focal_length, fallback to last level",
            "focal_length": "0.5 * image_width / tan(FoVx / 2)",
        },
        "paper_policy": {
            "name": "CG-LOD-PAPER",
            "selection": "fixed distance intervals from paper/Table S3",
            "matrixcity_intervals": "[0,200] finest, [200,400] middle, [400,inf] coarsest",
        },
        "recommendation": "Keep CG-LOD-RELEASE and CG-LOD-PAPER as separately labeled policies.",
    }


def _classify_asset(
    path: Path,
    fused_model: Path | None,
    lod_root: Path | None,
    camera_metadata: Path | None,
    yaml_path: Path | None,
) -> dict[str, str]:
    resolved = path.resolve()
    official_roots = [candidate.resolve() for candidate in (fused_model, camera_metadata, yaml_path) if candidate]
    if lod_root is not None:
        official_roots.append(lod_root.resolve())
    if any(resolved == root or root in resolved.parents for root in official_roots):
        classification = "official_citygaussian_asset"
    elif "partitioned" in resolved.parts:
        classification = "project_custom_partition_not_official_citygaussian"
    else:
        classification = "external_or_unclassified"
    return {"path": str(path), "classification": classification}


if __name__ == "__main__":
    raise SystemExit(main())
