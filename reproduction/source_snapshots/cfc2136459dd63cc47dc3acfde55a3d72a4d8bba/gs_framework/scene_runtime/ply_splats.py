from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor


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

REQUIRED_BASE_FIELDS = (
    "x",
    "y",
    "z",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def is_ply_path(path: Path) -> bool:
    return Path(path).suffix.lower() == ".ply"


LoadProgressCallback = Callable[[str], None]


def load_gaussian_ply(
    path: Path,
    apply_supersplat_transform: bool = True,
    progress_callback: LoadProgressCallback | None = None,
) -> Dict[str, Tensor]:
    """Load a Gaussian Splatting PLY into raw gsplat checkpoint tensors."""
    path = Path(path)
    _notify(progress_callback, "ply_parsing")
    header, data_offset = _read_header(path)
    _validate_header(path, header)

    vertex_count = _vertex_count(header)
    properties = _vertex_properties(header)
    property_names = [name for name, _ in properties]
    _warn_about_unexpected_layout(path, property_names)

    missing = [field for field in REQUIRED_BASE_FIELDS if field not in property_names]
    if missing:
        raise ValueError(f"{path} is missing required Gaussian PLY fields: {missing}.")
    f_dc_names = _numbered_names(property_names, "f_dc_")
    f_rest_names = _numbered_names(property_names, "f_rest_")
    if len(f_dc_names) != 3:
        raise ValueError(
            f"{path} must contain exactly 3 f_dc_* fields, found {len(f_dc_names)}."
        )
    _validate_numbered_range(path, f_dc_names, "f_dc_")
    _validate_numbered_range(path, f_rest_names, "f_rest_")
    if len(f_rest_names) % 3 != 0:
        raise ValueError(
            f"{path} has {len(f_rest_names)} f_rest_* fields; expected a multiple of 3."
        )
    sh_coeff_count = 1 + len(f_rest_names) // 3
    sh_degree = math.sqrt(sh_coeff_count) - 1
    if sh_degree < 0 or not float(sh_degree).is_integer():
        warnings.warn(
            f"{path} has {sh_coeff_count} SH coefficients per color channel, which "
            "does not correspond to an integer SH degree.",
            RuntimeWarning,
            stacklevel=2,
        )

    dtype = np.dtype([(name, PLY_SCALAR_DTYPES[prop_type]) for name, prop_type in properties])
    expected_bytes = data_offset + vertex_count * dtype.itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"{path} is truncated: expected at least {expected_bytes} bytes, got {actual_bytes}."
        )
    if actual_bytes > expected_bytes:
        warnings.warn(
            f"{path} has {actual_bytes - expected_bytes} trailing bytes after vertex data.",
            RuntimeWarning,
            stacklevel=2,
        )

    arr = np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=data_offset,
        shape=(vertex_count,),
    )
    _notify(progress_callback, "cpu_tensor_construction")
    means = _stack_fields(arr, ["x", "y", "z"])
    quats = _stack_fields(arr, ["rot_0", "rot_1", "rot_2", "rot_3"])
    if apply_supersplat_transform:
        _apply_supersplat_ply_transform(means, quats)

    return {
        "means": means,
        "quats": quats,
        "scales": _stack_fields(arr, ["scale_0", "scale_1", "scale_2"]),
        "opacities": torch.from_numpy(np.asarray(arr["opacity"], dtype=np.float32).copy()),
        "sh0": _stack_fields(arr, f_dc_names).reshape(vertex_count, 1, 3),
        "shN": _stack_sh_rest_fields(arr, f_rest_names),
    }


def _notify(callback: LoadProgressCallback | None, stage: str) -> None:
    if callback is not None:
        callback(stage)


def _read_header(path: Path) -> Tuple[List[str], int]:
    header: List[str] = []
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
                return header, f.tell()


def _validate_header(path: Path, header: List[str]) -> None:
    if not header or header[0] != "ply":
        raise ValueError(f"{path} is not a PLY file.")
    if len(header) < 2 or header[1] != "format binary_little_endian 1.0":
        raise ValueError(
            f"{path} must be binary_little_endian PLY 1.0; found {header[1] if len(header) > 1 else 'nothing'}."
        )
    unsupported = [line for line in header if line.startswith("property list ")]
    if unsupported:
        raise ValueError(f"{path} contains list properties, which are not supported.")


def _vertex_count(header: List[str]) -> int:
    for line in header:
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            return int(parts[2])
    raise ValueError("PLY header has no element vertex declaration.")


def _vertex_properties(header: List[str]) -> List[Tuple[str, str]]:
    properties: List[Tuple[str, str]] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            continue
        if not in_vertex or len(parts) != 3 or parts[0] != "property":
            continue
        prop_type, name = parts[1], parts[2]
        if prop_type not in PLY_SCALAR_DTYPES:
            raise ValueError(f"PLY property {name!r} has unsupported type {prop_type!r}.")
        properties.append((name, prop_type))
    if not properties:
        raise ValueError("PLY vertex element has no scalar properties.")
    return properties


def _numbered_names(property_names: List[str], prefix: str) -> List[str]:
    names = [name for name in property_names if name.startswith(prefix)]
    for name in names:
        suffix = name[len(prefix) :]
        if not suffix.isdigit():
            raise ValueError(f"PLY property {name!r} should end with a numeric suffix.")
    return sorted(names, key=lambda name: int(name[len(prefix) :]))


def _validate_numbered_range(path: Path, names: List[str], prefix: str) -> None:
    indices = [int(name[len(prefix) :]) for name in names]
    if indices != list(range(len(indices))):
        raise ValueError(
            f"{path} has non-contiguous {prefix} fields: {names}; expected suffixes "
            f"0..{len(indices) - 1}."
        )


def _warn_about_unexpected_layout(path: Path, property_names: List[str]) -> None:
    allowed = set(REQUIRED_BASE_FIELDS)
    allowed.update({"nx", "ny", "nz"})
    allowed.update(name for name in property_names if name.startswith("f_dc_"))
    allowed.update(name for name in property_names if name.startswith("f_rest_"))
    ignored = [name for name in property_names if name not in allowed]
    if ignored:
        warnings.warn(
            f"{path} contains unsupported PLY properties that will be ignored: {ignored}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not {"nx", "ny", "nz"}.issubset(property_names):
        warnings.warn(
            f"{path} has no complete nx/ny/nz normal fields; normals are optional and will be ignored.",
            RuntimeWarning,
            stacklevel=2,
        )


def _stack_fields(arr: np.ndarray, names: List[str]) -> Tensor:
    stacked = np.stack([np.asarray(arr[name], dtype=np.float32) for name in names], axis=-1)
    return torch.from_numpy(stacked.copy())


def _stack_sh_rest_fields(arr: np.ndarray, names: List[str]) -> Tensor:
    if not names:
        return torch.empty((arr.shape[0], 0, 3), dtype=torch.float32)
    # Nerfstudio/Inria PLY stores rest SH as channel-major:
    # R coeffs, then G coeffs, then B coeffs. gsplat expects coeff-major KxRGB.
    flat = _stack_fields(arr, names)
    return flat.reshape(arr.shape[0], 3, -1).transpose(1, 2).contiguous()


def _apply_supersplat_ply_transform(means: Tensor, quats: Tensor) -> None:
    # SuperSplat/splat-transform tags PLY sources with Transform.PLY, a 180-degree
    # Z rotation, and applies it as the splat entity rotation. Bake the same
    # transform into flat gsplat tensors because this viewer has no scene node.
    means[:, 0].neg_()
    means[:, 1].neg_()

    qw = quats[:, 0].clone()
    qx = quats[:, 1].clone()
    qy = quats[:, 2].clone()
    qz = quats[:, 3].clone()
    quats[:, 0] = -qz
    quats[:, 1] = -qy
    quats[:, 2] = qx
    quats[:, 3] = qw
