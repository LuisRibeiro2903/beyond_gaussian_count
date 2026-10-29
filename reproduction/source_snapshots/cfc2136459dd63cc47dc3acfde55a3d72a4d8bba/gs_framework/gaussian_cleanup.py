"""Offline cleanup helpers for flat gsplat Gaussian checkpoints."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


REQUIRED_SPLAT_KEYS = ("means", "quats", "scales", "opacities", "sh0", "shN")
DEFAULT_PERCENTILES = (0.0, 0.5, 1.0, 5.0, 50.0, 95.0, 99.0, 99.5, 100.0)


@dataclass(frozen=True)
class BoundsFilterConfig:
    opacity_min: Optional[float] = None
    opacity_percentile: Optional[float] = None
    max_scale: Optional[float] = None
    max_scale_percentile: Optional[float] = None
    roi_min: Optional[Tuple[float, float, float]] = None
    roi_max: Optional[Tuple[float, float, float]] = None
    position_percentile_min: float = 0.5
    position_percentile_max: float = 99.5
    padding: float = 0.0


CleanupConfig = BoundsFilterConfig


@dataclass(frozen=True)
class ExportPruneConfig:
    opacity_min: Optional[float] = None
    max_scale: Optional[float] = None
    roi_min: Optional[Tuple[float, float, float]] = None
    roi_max: Optional[Tuple[float, float, float]] = None


@dataclass
class CleanupResult:
    bounds_valid_mask: Tensor
    export_keep_mask: Tensor
    diagnostics: Dict[str, object]

    @property
    def metadata_valid_mask(self) -> Tensor:
        return self.bounds_valid_mask


def load_flat_checkpoint(
    path: Path,
    ply_coordinate_mode: str = "native",
) -> Dict[str, object]:
    from scene_runtime.ply_splats import is_ply_path, load_gaussian_ply

    if ply_coordinate_mode not in {"native", "supersplat"}:
        raise ValueError("ply_coordinate_mode must be 'native' or 'supersplat'.")
    if is_ply_path(path):
        payload: Dict[str, object] = {
            "step": -1,
            "splats": load_gaussian_ply(
                path,
                apply_supersplat_transform=ply_coordinate_mode == "supersplat",
            ),
            "source_path": str(Path(path)),
            "source_format": "ply",
            "ply_coordinate_mode": ply_coordinate_mode,
        }
        validate_flat_checkpoint(payload, path)
        return payload
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_flat_checkpoint(payload, path)
    return payload


def validate_flat_checkpoint(payload: object, path: Optional[Path] = None) -> None:
    label = str(path) if path is not None else "checkpoint"
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a dictionary checkpoint.")
    if "splats" not in payload:
        raise ValueError(f"{label} does not contain checkpoint['splats'].")
    splats = payload["splats"]
    if not isinstance(splats, dict):
        raise ValueError(f"{label} checkpoint['splats'] is not a dictionary.")
    missing = [key for key in REQUIRED_SPLAT_KEYS if key not in splats]
    if missing:
        raise ValueError(f"{label} is missing required splat keys: {missing}")
    non_tensors = [key for key in REQUIRED_SPLAT_KEYS if not torch.is_tensor(splats[key])]
    if non_tensors:
        raise ValueError(f"{label} has non-tensor splat keys: {non_tensors}")

    n = int(splats["means"].shape[0])
    for key in REQUIRED_SPLAT_KEYS:
        if int(splats[key].shape[0]) != n:
            raise ValueError(
                f"{label} splat key {key!r} has first dimension {splats[key].shape[0]}, expected {n}."
            )
    if splats["means"].ndim != 2 or int(splats["means"].shape[1]) != 3:
        raise ValueError(f"{label} splats['means'] must have shape (N, 3).")
    if splats["scales"].ndim != 2 or int(splats["scales"].shape[1]) != 3:
        raise ValueError(f"{label} splats['scales'] must have shape (N, 3).")
    if splats["quats"].ndim != 2 or int(splats["quats"].shape[1]) != 4:
        raise ValueError(f"{label} splats['quats'] must have shape (N, 4).")
    if splats["opacities"].ndim != 1:
        raise ValueError(f"{label} splats['opacities'] must have shape (N,).")
    if splats["sh0"].ndim != 3 or int(splats["sh0"].shape[-1]) != 3:
        raise ValueError(f"{label} splats['sh0'] must have shape (N, M, 3).")
    if splats["shN"].ndim != 3 or int(splats["shN"].shape[-1]) != 3:
        raise ValueError(f"{label} splats['shN'] must have shape (N, M, 3).")
    if n == 0:
        raise ValueError(f"{label} contains no Gaussians.")


def analyze_checkpoint(
    payload: Dict[str, object],
    config: BoundsFilterConfig,
    export_config: Optional[ExportPruneConfig] = None,
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    metadata_output_path: Optional[Path] = None,
) -> CleanupResult:
    validate_flat_checkpoint(payload, input_path)
    splats = payload["splats"]
    assert isinstance(splats, dict)

    means = splats["means"].detach().cpu().to(torch.float32)
    opacities = torch.sigmoid(splats["opacities"].detach().cpu().to(torch.float32))
    activated_scales = torch.exp(splats["scales"].detach().cpu().to(torch.float32))
    max_scales = activated_scales.max(dim=1).values

    bounds_specs = _bounds_filter_specs(config, means, opacities, max_scales)
    bounds_mask, bounds_filter_counts = _apply_filter_specs(means.shape[0], bounds_specs)
    bounds_valid_count = int(bounds_mask.sum().item())
    if bounds_valid_count == 0:
        raise ValueError("Cleanup filters removed every Gaussian; relax thresholds or ROI bounds.")

    if export_config is None:
        export_config = ExportPruneConfig()
    export_specs = _export_filter_specs(export_config, means, opacities, max_scales)
    export_mask, export_filter_counts = _apply_filter_specs(means.shape[0], export_specs)

    raw_bounds = _bounds(means)
    filtered_means = means[bounds_mask]
    filtered_bounds = _bounds(filtered_means)
    raw_robust_bounds = _robust_bounds(
        means,
        float(config.position_percentile_min),
        float(config.position_percentile_max),
        float(config.padding),
    )
    filtered_robust_bounds = _robust_bounds(
        filtered_means,
        float(config.position_percentile_min),
        float(config.position_percentile_max),
        float(config.padding),
    )
    raw_volume = _aabb_volume(raw_bounds)
    filtered_volume = _aabb_volume(filtered_robust_bounds)

    total_count = int(means.shape[0])
    diagnostics: Dict[str, object] = {
        "input_path": str(input_path) if input_path is not None else None,
        "output_path": str(output_path) if output_path is not None else None,
        "metadata_output_path": str(metadata_output_path) if metadata_output_path is not None else None,
        "bounds_filter_config": _config_to_json(config),
        "export_prune_config": _config_to_json(export_config),
        "total_gaussian_count": total_count,
        "bounds_valid_gaussian_count": bounds_valid_count,
        "bounds_removed_gaussian_count": total_count - bounds_valid_count,
        "bounds_removed_gaussian_percent": 100.0 * float(total_count - bounds_valid_count) / float(total_count),
        "bounds_filter_counts": bounds_filter_counts,
        "bounds_filter_ablation_counts": _filter_ablation_counts(total_count, bounds_specs),
        "export_keep_gaussian_count": int(export_mask.sum().item()),
        "export_removed_gaussian_count": total_count - int(export_mask.sum().item()),
        "export_removed_gaussian_percent": 100.0 * float(total_count - int(export_mask.sum().item())) / float(total_count),
        "export_filter_counts": export_filter_counts,
        "export_filter_ablation_counts": _filter_ablation_counts(total_count, export_specs),
        "opacity_percentiles": _percentiles(opacities),
        "scale_percentiles": _percentiles(activated_scales.reshape(-1)),
        "max_scale_percentiles": _percentiles(max_scales),
        "xyz_min_before_filtering": raw_bounds[0],
        "xyz_max_before_filtering": raw_bounds[1],
        "xyz_robust_percentiles_before_filtering": raw_robust_bounds,
        "xyz_min_after_filtering": filtered_bounds[0],
        "xyz_max_after_filtering": filtered_bounds[1],
        "xyz_robust_percentiles_after_filtering": filtered_robust_bounds,
        "raw_aabb_volume": raw_volume,
        "robust_filtered_aabb_volume": filtered_volume,
        "raw_to_filtered_aabb_volume_ratio": (
            float("inf") if filtered_volume == 0.0 else raw_volume / filtered_volume
        ),
    }
    diagnostics["config"] = diagnostics["bounds_filter_config"]
    diagnostics["valid_gaussian_count"] = diagnostics["bounds_valid_gaussian_count"]
    diagnostics["removed_gaussian_count"] = diagnostics["bounds_removed_gaussian_count"]
    diagnostics["removed_gaussian_percent"] = diagnostics["bounds_removed_gaussian_percent"]
    diagnostics["filter_counts"] = diagnostics["bounds_filter_counts"]
    return CleanupResult(
        bounds_valid_mask=bounds_mask,
        export_keep_mask=export_mask,
        diagnostics=diagnostics,
    )


def export_pruned_checkpoint(payload: Dict[str, object], mask: Tensor) -> Dict[str, object]:
    validate_flat_checkpoint(payload)
    pruned = copy.copy(payload)
    splats = payload["splats"]
    assert isinstance(splats, dict)
    bool_mask = mask.cpu().to(torch.bool)
    pruned["splats"] = {
        key: value[bool_mask].detach().cpu().contiguous()
        for key, value in splats.items()
        if torch.is_tensor(value) and int(value.shape[0]) == int(bool_mask.shape[0])
    }

    for key, value in splats.items():
        if key not in pruned["splats"]:
            pruned["splats"][key] = value
    return pruned


def write_metadata(path: Path, diagnostics: Dict[str, object], overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(diagnostics), indent=2, sort_keys=True) + "\n")
    return path


def save_pruned_checkpoint(
    payload: Dict[str, object],
    mask: Tensor,
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path)
    if output_path.resolve() == input_path:
        raise ValueError("Refusing to overwrite the input checkpoint; choose a separate --output path.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_pruned_checkpoint(payload, mask), output_path)
    return output_path


def load_bounds_filter_config_from_metadata(path: Path) -> BoundsFilterConfig:
    metadata = json.loads(Path(path).read_text())
    config = metadata.get("bounds_filter_config", metadata.get("config"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} does not contain a bounds_filter_config object.")
    return bounds_filter_config_from_dict(config)


def bounds_filter_config_from_dict(data: Dict[str, object]) -> BoundsFilterConfig:
    return BoundsFilterConfig(
        opacity_min=_optional_float(data.get("opacity_min")),
        opacity_percentile=_optional_float(data.get("opacity_percentile")),
        max_scale=_optional_float(data.get("max_scale")),
        max_scale_percentile=_optional_float(data.get("max_scale_percentile")),
        roi_min=_optional_vec3(data.get("roi_min")),
        roi_max=_optional_vec3(data.get("roi_max")),
        position_percentile_min=float(data.get("position_percentile_min", 0.5)),
        position_percentile_max=float(data.get("position_percentile_max", 99.5)),
        padding=float(data.get("padding", 0.0)),
    )


def with_position_bounds_config(config: BoundsFilterConfig, percentile_min: float, percentile_max: float, padding: float) -> BoundsFilterConfig:
    return replace(
        config,
        position_percentile_min=float(percentile_min),
        position_percentile_max=float(percentile_max),
        padding=float(padding),
    )


def robust_bounds_for_points(points: Tensor, config: BoundsFilterConfig) -> Dict[str, List[float]]:
    return _robust_bounds(
        points,
        float(config.position_percentile_min),
        float(config.position_percentile_max),
        float(config.padding),
    )


def tensor_bounds(points: Tensor) -> Tuple[List[float], List[float]]:
    return _bounds(points)


def aabb_volume(bounds: Tuple[List[float], List[float]] | Dict[str, List[float]]) -> float:
    return _aabb_volume(bounds)


def _config_to_json(config: BoundsFilterConfig | ExportPruneConfig) -> Dict[str, object]:
    data = asdict(config)
    for key in ("roi_min", "roi_max"):
        if data[key] is not None:
            data[key] = [float(v) for v in data[key]]
    return data


def _bounds_filter_specs(
    config: BoundsFilterConfig,
    means: Tensor,
    opacities: Tensor,
    max_scales: Tensor,
) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    if config.opacity_min is not None:
        specs.append(_filter_spec("opacity_min", opacities >= float(config.opacity_min), float(config.opacity_min)))
    if config.opacity_percentile is not None:
        threshold = _percentile(opacities, float(config.opacity_percentile))
        specs.append(_filter_spec("opacity_percentile", opacities >= threshold, threshold))
    if config.max_scale is not None:
        specs.append(_filter_spec("max_scale", max_scales <= float(config.max_scale), float(config.max_scale)))
    if config.max_scale_percentile is not None:
        threshold = _percentile(max_scales, float(config.max_scale_percentile))
        specs.append(_filter_spec("max_scale_percentile", max_scales <= threshold, threshold))
    if config.roi_min is not None:
        roi_min = torch.tensor(config.roi_min, dtype=torch.float32)
        specs.append(_filter_spec("roi_min", (means >= roi_min).all(dim=1)))
    if config.roi_max is not None:
        roi_max = torch.tensor(config.roi_max, dtype=torch.float32)
        specs.append(_filter_spec("roi_max", (means <= roi_max).all(dim=1)))
    return specs


def _export_filter_specs(
    config: ExportPruneConfig,
    means: Tensor,
    opacities: Tensor,
    max_scales: Tensor,
) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    if config.opacity_min is not None:
        specs.append(_filter_spec("export_opacity_min", opacities >= float(config.opacity_min), float(config.opacity_min)))
    if config.max_scale is not None:
        specs.append(_filter_spec("export_max_scale", max_scales <= float(config.max_scale), float(config.max_scale)))
    if config.roi_min is not None:
        roi_min = torch.tensor(config.roi_min, dtype=torch.float32)
        specs.append(_filter_spec("export_roi_min", (means >= roi_min).all(dim=1)))
    if config.roi_max is not None:
        roi_max = torch.tensor(config.roi_max, dtype=torch.float32)
        specs.append(_filter_spec("export_roi_max", (means <= roi_max).all(dim=1)))
    return specs


def _filter_spec(name: str, mask: Tensor, threshold: Optional[float] = None) -> Dict[str, object]:
    spec: Dict[str, object] = {"name": name, "mask": mask.cpu().to(torch.bool)}
    if threshold is not None:
        spec["threshold"] = float(threshold)
    return spec


def _apply_filter_specs(count: int, specs: Sequence[Dict[str, object]]) -> Tuple[Tensor, List[Dict[str, object]]]:
    mask = torch.ones((int(count),), dtype=torch.bool)
    filter_counts: List[Dict[str, object]] = []
    for spec in specs:
        before = int(mask.sum().item())
        filter_mask = spec["mask"]
        assert torch.is_tensor(filter_mask)
        mask = mask & filter_mask.cpu().to(torch.bool)
        after = int(mask.sum().item())
        entry: Dict[str, object] = {
            "name": str(spec["name"]),
            "valid_before": before,
            "valid_after": after,
            "removed_by_filter": before - after,
        }
        if "threshold" in spec:
            entry["threshold"] = float(spec["threshold"])
        filter_counts.append(entry)
    return mask, filter_counts


def _filter_ablation_counts(count: int, specs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    results = []
    for spec in specs:
        filter_mask = spec["mask"]
        assert torch.is_tensor(filter_mask)
        kept = int(filter_mask.cpu().to(torch.bool).sum().item())
        entry: Dict[str, object] = {
            "name": str(spec["name"]),
            "valid_after_filter_alone": kept,
            "removed_by_filter_alone": int(count) - kept,
            "removed_percent_filter_alone": 100.0 * float(int(count) - kept) / float(max(int(count), 1)),
        }
        if "threshold" in spec:
            entry["threshold"] = float(spec["threshold"])
        results.append(entry)
    return results


def _optional_float(value: object) -> Optional[float]:
    return None if value is None else float(value)


def _optional_vec3(value: object) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected a 3-vector or null, got {value!r}.")
    return (float(value[0]), float(value[1]), float(value[2]))


def _percentile(values: Tensor, percentile: float) -> float:
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("Percentiles must be in [0, 100].")
    values = values.detach().cpu().to(torch.float32).reshape(-1)
    return float(torch.quantile(values, percentile / 100.0).item())


def _percentiles(values: Tensor, percentiles: Tuple[float, ...] = DEFAULT_PERCENTILES) -> Dict[str, float]:
    values = values.detach().cpu().to(torch.float32).reshape(-1)
    return {f"p{_percentile_key(p)}": _percentile(values, p) for p in percentiles}


def _percentile_key(percentile: float) -> str:
    if float(percentile).is_integer():
        return str(int(percentile))
    return str(percentile).replace(".", "_")


def _bounds(points: Tensor) -> Tuple[List[float], List[float]]:
    points = points.detach().cpu().to(torch.float32)
    return (
        [float(v) for v in points.min(dim=0).values.tolist()],
        [float(v) for v in points.max(dim=0).values.tolist()],
    )


def _robust_bounds(points: Tensor, percentile_min: float, percentile_max: float, padding: float) -> Dict[str, List[float]]:
    if percentile_min < 0.0 or percentile_max > 100.0 or percentile_min > percentile_max:
        raise ValueError("Position percentiles must satisfy 0 <= min <= max <= 100.")
    if padding < 0.0:
        raise ValueError("padding must be non-negative.")
    points = points.detach().cpu().to(torch.float32)
    qs = torch.tensor([percentile_min / 100.0, percentile_max / 100.0], dtype=torch.float32)
    quantiles = torch.quantile(points, qs, dim=0)
    lower = quantiles[0] - float(padding)
    upper = quantiles[1] + float(padding)
    return {
        "percentile_min": float(percentile_min),
        "percentile_max": float(percentile_max),
        "padding": float(padding),
        "min": [float(v) for v in lower.tolist()],
        "max": [float(v) for v in upper.tolist()],
    }


def _aabb_volume(bounds: Tuple[List[float], List[float]] | Dict[str, List[float]]) -> float:
    if isinstance(bounds, dict):
        min_values = bounds["min"]
        max_values = bounds["max"]
    else:
        min_values, max_values = bounds
    extent = [max(float(hi) - float(lo), 0.0) for lo, hi in zip(min_values, max_values)]
    return float(extent[0] * extent[1] * extent[2])


def _json_safe(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
    return value
