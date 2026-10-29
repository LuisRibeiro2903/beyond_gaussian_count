#!/usr/bin/env python3
"""Observational capture hooks called by a disposable official-source patch.

This module deliberately contains no hierarchy traversal, SPT construction, or
selection implementation.  It serializes values already produced by the
pinned official evaluation path and proves their mapping to serialized rows.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alod_dhier import load_dhier
from alod_selection_oracle.format import (
    CAMERA_SCHEMA_VERSION,
    HSPT_SCHEMA_VERSION,
    PRE_RASTER_SCHEMA_VERSION,
    SPT_CUT_SCHEMA_VERSION,
    UPPER_TRAVERSAL_SCHEMA_VERSION,
    OracleValidationError,
    array_descriptor,
    canonical_json,
    global_id_hash,
    save_array_payload,
    sha256_file,
    validate_global_ids,
)


EXPECTED_ASSET_SHA256 = os.environ.get(
    "ALOD_ORACLE_EXPECTED_ASSET_SHA256",
    "6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c",
)
EXPECTED_SOURCE_COMMIT = "73f547341547ae021ff036ad74fdf73a1240b450"
EXPECTED_ROWS = int(os.environ.get("ALOD_ORACLE_EXPECTED_ROWS", "1099999"))
EXPECTED_SKYBOX_ROWS = int(os.environ.get("ALOD_ORACLE_EXPECTED_SKYBOX_ROWS", "100000"))
EXPECTED_ROOT_GLOBAL_ID = int(os.environ.get("ALOD_ORACLE_EXPECTED_ROOT_GLOBAL_ID", "100000"))
EXPECTED_SH_DEGREE = int(os.environ.get("ALOD_ORACLE_EXPECTED_SH_DEGREE", "1"))


def selected_camera_index() -> int | None:
    value = os.environ.get("ALOD_ORACLE_CAMERA_INDEX")
    return None if value in (None, "") else int(value)


def _device(value: Any) -> str | None:
    return str(value.device) if hasattr(value, "device") else None


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(value)


def _histogram(values: Any) -> dict[str, int]:
    return {
        str(int(key)): int(count)
        for key, count in sorted(Counter(int(value) for value in _numpy(values).reshape(-1)).items())
    }


def _id_summary(ids: Any) -> dict[str, Any]:
    raw = _numpy(ids)
    canonical = validate_global_ids(raw, row_count=EXPECTED_ROWS, allow_duplicates=True)
    return {
        "count": int(canonical.size),
        "runtime_array": array_descriptor(raw, device=_device(ids)),
        "canonical_encoding": "little-endian signed int64, contiguous C order",
        "ordered_sha256": global_id_hash(canonical),
        "sorted_sha256": global_id_hash(canonical, sorted_ids=True),
        "unique_count": int(np.unique(canonical).size),
        "min": int(canonical.min()) if canonical.size else None,
        "max": int(canonical.max()) if canonical.size else None,
    }


class OracleCapture:
    """Stateful serializer for one cold official camera process."""

    def __init__(
        self,
        *,
        gaussians: Any,
        scene: Any,
        optimization: Any,
        dataset: Any,
        spt_root_global_ids: Any,
        base_focal_length: float,
        target_granularity: float,
    ) -> None:
        output_value = os.environ.get("ALOD_ORACLE_OUTPUT_DIR")
        frozen_value = os.environ.get("ALOD_ORACLE_FROZEN_CAMERAS")
        asset_value = os.environ.get("ALOD_ORACLE_ASSET")
        if not output_value or not frozen_value or not asset_value:
            raise RuntimeError("official oracle environment is incomplete")
        self.output_dir = Path(output_value).resolve()
        self.payload_dir = self.output_dir / "payloads"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.payload_dir.mkdir(parents=True, exist_ok=True)
        self.payload_mode = os.environ.get("ALOD_ORACLE_PAYLOAD_MODE", "hash-only")
        self.capture_hspt_payloads = os.environ.get("ALOD_ORACLE_HSPT_PAYLOADS") == "1"
        self.asset_path = Path(asset_value).resolve()
        self.asset = load_dhier(self.asset_path)
        self.gaussians = gaussians
        self.scene = scene
        self.optimization = optimization
        self.dataset = dataset
        self.base_focal_length = float(base_focal_length)
        self.target_granularity = float(target_granularity)
        self.camera_index = selected_camera_index()
        if self.camera_index is None:
            raise RuntimeError("ALOD_ORACLE_CAMERA_INDEX is required")
        frozen = json.loads(Path(frozen_value).read_text(encoding="utf-8"))
        matches = [
            camera
            for camera in frozen["cameras"]
            if int(camera["evaluation_sequence_index"]) == self.camera_index
        ]
        if len(matches) != 1:
            raise OracleValidationError(
                f"expected one frozen camera for evaluation index {self.camera_index}"
            )
        self.frozen_camera = matches[0]
        self.camera_stable_id = (
            f"eval-{self.camera_index:04d}-"
            f"img-{int(self.frozen_camera['registered_image_id']):06d}-"
            f"cam-{int(self.frozen_camera['camera_id']):04d}"
        )
        self.stage_files: dict[str, str] = {}
        self.hierarchy_selected_ids: np.ndarray | None = None
        self.raster_global_ids: np.ndarray | None = None
        self.row_correspondence: dict[str, Any] | None = None
        self.pre_raster_hashes: dict[str, str] | None = None
        self._validate_asset_identity()
        self._record_hspt(spt_root_global_ids)

    @classmethod
    def from_environment(cls, **kwargs: Any) -> "OracleCapture | None":
        if not os.environ.get("ALOD_ORACLE_OUTPUT_DIR"):
            return None
        return cls(**kwargs)

    def _validate_asset_identity(self) -> None:
        actual_sha = sha256_file(self.asset_path)
        layout = self.asset.layout
        checks = {
            "asset_sha256": actual_sha == EXPECTED_ASSET_SHA256,
            "rows": layout.gaussian_count == EXPECTED_ROWS,
            "node_rows": layout.node_count == EXPECTED_ROWS,
            "sh_degree": layout.sh_degree == EXPECTED_SH_DEGREE,
            "skybox_rows": int(self.gaussians.skybox_points) == EXPECTED_SKYBOX_ROWS,
            "model_size": int(self.gaussians.size) == EXPECTED_ROWS,
            "root_depth": int(self.asset.nodes[EXPECTED_ROOT_GLOBAL_ID, 0]) == 0,
            "root_parent": int(self.asset.nodes[EXPECTED_ROOT_GLOBAL_ID, 1]) == -1,
            "scaffold_file_empty": os.environ.get("ALOD_ORACLE_SCAFFOLD_FILE", "") == "",
            "source_commit": os.environ.get("ALOD_ORACLE_SOURCE_COMMIT")
            == EXPECTED_SOURCE_COMMIT,
        }
        if not all(checks.values()):
            raise OracleValidationError(f"oracle asset identity assertion failed: {checks}")
        payload = {
            "schema_version": "alod_selection_oracle_asset.v1",
            "asset_path": str(self.asset_path),
            "asset_sha256": actual_sha,
            "serialized_rows": EXPECTED_ROWS,
            "skybox_rows": EXPECTED_SKYBOX_ROWS,
            "hierarchy_rows": EXPECTED_ROWS - EXPECTED_SKYBOX_ROWS,
            "root_global_node_id": EXPECTED_ROOT_GLOBAL_ID,
            "sh_degree": EXPECTED_SH_DEGREE,
            "official_source_commit": EXPECTED_SOURCE_COMMIT,
            "instrumentation_patch_sha256": os.environ.get("ALOD_ORACLE_PATCH_SHA256"),
            "run005_correctness_patch_sha256": "a7412fff7315153774c93b504105804c2a182308807a2c6ec8de34672b9cd7d6",
            "checks": checks,
        }
        self._write_json("asset", "asset.json", payload)

    def _describe(
        self,
        stage: str,
        name: str,
        value: Any,
        *,
        save_payload: bool,
    ) -> dict[str, Any]:
        if self.payload_mode == "full" and save_payload:
            return save_array_payload(
                self.payload_dir / stage / f"{name}.npy",
                value,
                device=_device(value),
                relative_to=self.output_dir,
            )
        return array_descriptor(value, device=_device(value))

    def _write_json(self, stage: str, filename: str, payload: dict[str, Any]) -> None:
        path = self.output_dir / filename
        path.write_text(canonical_json(payload), encoding="utf-8")
        self.stage_files[stage] = sha256_file(path)

    def _record_hspt(self, spt_root_global_ids: Any) -> None:
        nodes = self.gaussians.upper_tree_nodes
        upper_to_global = nodes[:, 5]
        upper_global = validate_global_ids(
            upper_to_global,
            row_count=EXPECTED_ROWS,
            label="upper_to_global",
        )
        if upper_global.size == 0 or int(upper_global[0]) != EXPECTED_ROOT_GLOBAL_ID:
            raise OracleValidationError("upper hierarchy does not begin at the serialized root")
        starts = _numpy(self.gaussians.SPT_starts)
        if starts.ndim != 1 or starts.size == 0 or int(starts[0]) != 0:
            raise OracleValidationError("invalid SPT_starts")
        if np.any(starts[1:] < starts[:-1]):
            raise OracleValidationError("SPT_starts is not monotonic")
        spt_ids = validate_global_ids(
            self.gaussians.SPT_gaussian_indices,
            row_count=EXPECTED_ROWS,
            label="SPT_gaussian_indices",
        )
        if int(starts[-1]) != spt_ids.size:
            raise OracleValidationError("SPT_starts does not cover flattened SPT rows")
        roots = validate_global_ids(
            spt_root_global_ids,
            row_count=EXPECTED_ROWS,
            label="SPT_root_global_ids",
        )
        if roots.size != starts.size - 1:
            raise OracleValidationError("SPT root count disagrees with SPT starts")
        if roots.size and not np.array_equal(spt_ids[starts[:-1]], roots):
            raise OracleValidationError("SPT roots do not map to first flattened SPT rows")
        sizes = np.diff(starts).astype("<i4", copy=False)
        upper_depths = np.asarray(self.asset.nodes[upper_global, 0], dtype="<i4")
        payload_arrays = self.capture_hspt_payloads
        arrays = {
            "upper_tree_nodes": self._describe(
                "hspt", "upper_tree_nodes", nodes, save_payload=payload_arrays
            ),
            "upper_to_global_node_id": self._describe(
                "hspt", "upper_to_global_node_id", upper_to_global, save_payload=payload_arrays
            ),
            "upper_tree_xyz": self._describe(
                "hspt", "upper_tree_xyz", self.gaussians.upper_tree_xyz, save_payload=payload_arrays
            ),
            "upper_tree_log_scaling": self._describe(
                "hspt", "upper_tree_log_scaling", self.gaussians.upper_tree_scaling, save_payload=payload_arrays
            ),
            "upper_min_distance_squared": self._describe(
                "hspt", "upper_min_distance_squared", self.gaussians.min_distance_squared, save_payload=payload_arrays
            ),
            "spt_starts": self._describe(
                "hspt", "spt_starts", self.gaussians.SPT_starts, save_payload=payload_arrays
            ),
            "spt_sizes": self._describe(
                "hspt", "spt_sizes", sizes, save_payload=payload_arrays
            ),
            "spt_min": self._describe(
                "hspt", "spt_min", self.gaussians.SPT_min, save_payload=payload_arrays
            ),
            "spt_max": self._describe(
                "hspt", "spt_max", self.gaussians.SPT_max, save_payload=payload_arrays
            ),
            "spt_to_global_node_id": self._describe(
                "hspt", "spt_to_global_node_id", self.gaussians.SPT_gaussian_indices, save_payload=payload_arrays
            ),
            "spt_root_global_node_ids": self._describe(
                "hspt", "spt_root_global_node_ids", roots, save_payload=payload_arrays
            ),
        }
        payload = {
            "schema_version": HSPT_SCHEMA_VERSION,
            "global_node_id_contract": "serialized .dhier row index",
            "configuration": {
                "SPT_root_volume": float(self.optimization.SPT_root_volume),
                "conceptual_target_granularity_pixels": float(
                    self.optimization.target_granularity_pixels
                ),
                "base_focal_length_original_colmap_pixels": self.base_focal_length,
                "resolved_target_granularity": self.target_granularity,
                "min_SPT_size_effective_evaluation_default": 100,
                "min_SPT_size_training_not_used": int(self.optimization.min_SPT_size),
                "use_bounding_spheres": bool(self.optimization.use_bounding_spheres),
            },
            "counts": {
                "upper_hierarchy_rows": int(upper_global.size),
                "spt_count": int(roots.size),
                "flattened_spt_metadata_rows": int(spt_ids.size),
                "total_hspt_metadata_rows": int(upper_global.size + spt_ids.size),
                "minimum_spt_size": int(sizes.min()) if sizes.size else None,
                "maximum_spt_size": int(sizes.max()) if sizes.size else None,
            },
            "spt_size_histogram": _histogram(sizes),
            "upper_hierarchy_depth_distribution": _histogram(upper_depths),
            "mappings": {
                "upper_local_id": "row index into upper_tree_nodes",
                "upper_local_to_global": "upper_tree_nodes[:, 5]",
                "spt_id": "interval index into SPT_starts",
                "spt_local_to_global": "SPT_gaussian_indices[SPT_starts[spt_id] + local_index]",
                "spt_roots": "SPT_gaussian_indices[SPT_starts[:-1]]",
            },
            "arrays": arrays,
        }
        self._write_json("hspt", "hspt.json", payload)

    def record_camera(self, camera: Any) -> None:
        frozen = self.frozen_camera
        center = _numpy(camera.camera_center)
        frozen_center = np.asarray(frozen["camera_center"], dtype=np.float64)
        # The official camera is constructed in a DataLoader worker and inverts a
        # float32 CPU view matrix.  Allow four coordinate-scaled ULPs for the
        # inversion; a fixed 2e-6 bound is tighter than its roundoff.
        center_atol = max(
            2e-6,
            4.0 * np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(frozen_center)))),
        )
        checks = {
            "evaluation_index": int(camera.uid) == int(frozen["evaluation_sequence_index"]),
            "camera_id": int(camera.colmap_id) == int(frozen["camera_id"]),
            "filename": camera.image_name == frozen["image_name"],
            "effective_width": int(camera.image_width)
            == int(frozen.get("effective_width", 960)),
            "effective_height": int(camera.image_height)
            == int(frozen.get("effective_height", 540)),
            "camera_center": bool(np.allclose(center, frozen_center, rtol=0.0, atol=center_atol)),
        }
        if not all(checks.values()):
            raise OracleValidationError(f"official/frozen camera mismatch: {checks}")
        width = int(camera.image_width)
        height = int(camera.image_height)
        fx = width / (2.0 * np.tan(float(camera.FoVx) * 0.5))
        fy = height / (2.0 * np.tan(float(camera.FoVy) * 0.5))
        effective_k = np.asarray(
            [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        payload = {
            "schema_version": CAMERA_SCHEMA_VERSION,
            "camera_stable_id": self.camera_stable_id,
            "identity": {
                "registered_image_id": int(frozen["registered_image_id"]),
                "camera_id": int(frozen["camera_id"]),
                "filename": frozen["image_name"],
                "stable_sorted_index": int(frozen["stable_sorted_index"]),
                "evaluation_sequence_index": int(frozen["evaluation_sequence_index"]),
            },
            "selection_rationale": frozen["selection_rationale"],
            "selection_categories": frozen["selection_categories"],
            "frozen_dataset_camera": frozen,
            "effective_official_runtime_camera": {
                "image_width": width,
                "image_height": height,
                "FoVx_radians": float(camera.FoVx),
                "FoVy_radians": float(camera.FoVy),
                "camera_focal_length_field_original_pixels": float(camera.focal_length),
                "intrinsics": {
                    "fx": float(fx),
                    "fy": float(fy),
                    "cx": width / 2.0,
                    "cy": height / 2.0,
                    "matrix_float64_values": effective_k.tolist(),
                },
                "camera_center_float32": center.tolist(),
                "world_view_transform": {
                    **array_descriptor(camera.world_view_transform, device=_device(camera.world_view_transform)),
                    "values": _numpy(camera.world_view_transform).tolist(),
                },
                "projection_matrix": {
                    **array_descriptor(camera.projection_matrix, device=_device(camera.projection_matrix)),
                    "values": _numpy(camera.projection_matrix).tolist(),
                },
                "full_projection_transform": {
                    **array_descriptor(camera.full_proj_transform, device=_device(camera.full_proj_transform)),
                    "values": _numpy(camera.full_proj_transform).tolist(),
                },
            },
            "checks": checks,
        }
        self._write_json("camera", "camera.json", payload)

    def record_upper_traversal(
        self,
        *,
        camera: Any,
        frustum_planes: Any,
        scale_derived_bounds: Any,
        coarse_cut_upper_local_ids: Any,
        cut_leaf_upper_local_ids: Any,
        candidate_spt_ids: Any,
        candidate_spt_upper_local_ids: Any,
        candidate_spt_distances: Any,
        upper_render_global_ids: Any,
        distance_multiplier: float,
    ) -> None:
        upper_to_global = self.gaussians.upper_tree_nodes[:, 5]
        coarse_global = upper_to_global[coarse_cut_upper_local_ids]
        leaf_global = upper_to_global[cut_leaf_upper_local_ids]
        candidate_roots = upper_to_global[candidate_spt_upper_local_ids]
        for label, ids in (
            ("coarse_cut_global_ids", coarse_global),
            ("cut_leaf_global_ids", leaf_global),
            ("candidate_spt_root_global_ids", candidate_roots),
            ("upper_render_global_ids", upper_render_global_ids),
        ):
            validate_global_ids(ids, row_count=EXPECTED_ROWS, label=label)
        payload = {
            "schema_version": UPPER_TRAVERSAL_SCHEMA_VERSION,
            "camera_stable_id": self.camera_stable_id,
            "official_semantic_output": "coarse_cut plus its SPT candidates and emitted upper global rows",
            "frustum": {
                "active_planes": ["left", "right", "bottom", "top"],
                "inactive_planes": ["near", "far"],
                "bounding_sphere_construction": False,
                "bounds": "3 * exp(max(log_scale_components))",
                "planes": {
                    **array_descriptor(frustum_planes, device=_device(frustum_planes)),
                    "values": _numpy(frustum_planes).tolist(),
                },
                "scale_derived_bounds": self._describe(
                    "upper", "scale_derived_bounds", scale_derived_bounds, save_payload=False
                ),
            },
            "lod": {
                "predicate": "min_distance_squared > squared_camera_distance * distance_multiplier",
                "distance_multiplier": float(distance_multiplier),
                "camera_center": _numpy(camera.camera_center).tolist(),
            },
            "counts": {
                "coarse_cut": int(len(coarse_cut_upper_local_ids)),
                "cut_leaf_nodes": int(len(cut_leaf_upper_local_ids)),
                "candidate_spts": int(len(candidate_spt_ids)),
                "emitted_upper_nodes": int(len(upper_render_global_ids)),
            },
            "identity_summaries": {
                "coarse_cut_global_node_ids": _id_summary(coarse_global),
                "cut_leaf_global_node_ids": _id_summary(leaf_global),
                "candidate_spt_root_global_node_ids": _id_summary(candidate_roots),
                "upper_render_global_node_ids": _id_summary(upper_render_global_ids),
            },
            "arrays": {
                "coarse_cut_upper_local_ids": self._describe(
                    "upper", "coarse_cut_upper_local_ids", coarse_cut_upper_local_ids, save_payload=True
                ),
                "coarse_cut_global_node_ids": self._describe(
                    "upper", "coarse_cut_global_node_ids", coarse_global, save_payload=True
                ),
                "cut_leaf_upper_local_ids": self._describe(
                    "upper", "cut_leaf_upper_local_ids", cut_leaf_upper_local_ids, save_payload=True
                ),
                "cut_leaf_global_node_ids": self._describe(
                    "upper", "cut_leaf_global_node_ids", leaf_global, save_payload=True
                ),
                "candidate_spt_ids": self._describe(
                    "upper", "candidate_spt_ids", candidate_spt_ids, save_payload=True
                ),
                "candidate_spt_upper_local_ids": self._describe(
                    "upper", "candidate_spt_upper_local_ids", candidate_spt_upper_local_ids, save_payload=True
                ),
                "candidate_spt_root_global_node_ids": self._describe(
                    "upper", "candidate_spt_root_global_node_ids", candidate_roots, save_payload=True
                ),
                "candidate_spt_distances": self._describe(
                    "upper", "candidate_spt_distances", candidate_spt_distances, save_payload=True
                ),
                "upper_render_global_node_ids": self._describe(
                    "upper", "upper_render_global_node_ids", upper_render_global_ids, save_payload=True
                ),
            },
        }
        self._write_json("upper_traversal", "upper-traversal.json", payload)

    def record_spt_cut(
        self,
        *,
        candidate_spt_ids: Any,
        candidate_spt_distances: Any,
        raw_cuda_spt_global_ids: Any,
        raw_cuda_spt_prefix_starts: Any,
        post_adjust_spt_ids: Any,
        post_adjust_spt_distances: Any,
        upper_render_global_ids: Any,
    ) -> None:
        raw_spt = validate_global_ids(
            raw_cuda_spt_global_ids,
            row_count=EXPECTED_ROWS,
            label="raw_cuda_spt_global_ids",
        )
        upper = validate_global_ids(
            upper_render_global_ids,
            row_count=EXPECTED_ROWS,
            label="upper_render_global_ids",
        )
        combined = np.concatenate((upper, raw_spt)).astype("<i8", copy=False)
        validate_global_ids(
            combined,
            row_count=EXPECTED_ROWS,
            label="selected_hierarchy_global_ids",
        )
        self.hierarchy_selected_ids = combined
        depths = np.asarray(self.asset.nodes[combined, 0], dtype="<i4")
        source_rows = {}
        for name, array in (
            ("positions", self.asset.positions),
            ("rotations", self.asset.rotations),
            ("serialized_log_scales", self.asset.log_scales),
            ("serialized_activated_opacities", self.asset.activated_opacities),
            ("serialized_sh_coefficients", self.asset.sh_coefficients),
            ("serialized_hierarchy_nodes", self.asset.nodes),
        ):
            source_rows[name] = array_descriptor(np.asarray(array[combined]))
        payload = {
            "schema_version": SPT_CUT_SCHEMA_VERSION,
            "camera_stable_id": self.camera_stable_id,
            "primary_selection_scope": "hierarchy-selected rows excluding always-on skybox",
            "official_order": "emitted upper global IDs followed by raw CUDA SPT-cut global IDs",
            "counts": {
                "candidate_spts": int(len(candidate_spt_ids)),
                "post_adjust_candidate_spts": int(len(post_adjust_spt_ids)),
                "selected_upper_nodes": int(upper.size),
                "selected_spt_nodes": int(raw_spt.size),
                "selected_hierarchy_nodes": int(combined.size),
            },
            "selected_depth": {
                "minimum": int(depths.min()) if depths.size else None,
                "maximum": int(depths.max()) if depths.size else None,
                "distribution": _histogram(depths),
                "distinct_depths": int(np.unique(depths).size),
            },
            "identity_summaries": {
                "raw_cuda_spt_global_node_ids": _id_summary(raw_spt),
                "selected_hierarchy_global_node_ids": _id_summary(combined),
            },
            "arrays": {
                "candidate_spt_ids": self._describe(
                    "spt-cut", "candidate_spt_ids", candidate_spt_ids, save_payload=True
                ),
                "candidate_spt_distances": self._describe(
                    "spt-cut", "candidate_spt_distances", candidate_spt_distances, save_payload=True
                ),
                "raw_cuda_spt_global_node_ids": self._describe(
                    "spt-cut", "raw_cuda_spt_global_node_ids", raw_spt, save_payload=True
                ),
                "raw_cuda_spt_prefix_starts": self._describe(
                    "spt-cut", "raw_cuda_spt_prefix_starts", raw_cuda_spt_prefix_starts, save_payload=True
                ),
                "post_adjust_spt_ids": self._describe(
                    "spt-cut", "post_adjust_spt_ids", post_adjust_spt_ids, save_payload=True
                ),
                "post_adjust_spt_distances": self._describe(
                    "spt-cut", "post_adjust_spt_distances", post_adjust_spt_distances, save_payload=True
                ),
                "selected_hierarchy_global_node_ids": self._describe(
                    "spt-cut", "selected_hierarchy_global_node_ids", combined, save_payload=True
                ),
                "sorted_selected_hierarchy_global_node_ids": self._describe(
                    "spt-cut",
                    "sorted_selected_hierarchy_global_node_ids",
                    np.sort(combined, kind="stable"),
                    save_payload=True,
                ),
            },
            "selected_source_rows": {
                "row_order": "selected_hierarchy_global_node_ids official order",
                "global_node_ids": _id_summary(combined),
                "arrays": source_rows,
            },
        }
        self._write_json("spt_cut", "spt-cut.json", payload)

    def record_working_set(
        self,
        *,
        load_from_disk_global_ids: Any,
        means: Any,
        opacity_logits: Any,
        log_scales: Any,
        raw_quaternions: Any,
        features_dc: Any,
        features_rest: Any,
    ) -> None:
        if self.hierarchy_selected_ids is None:
            raise OracleValidationError("SPT cut must be recorded before working-set assembly")
        loaded = validate_global_ids(
            load_from_disk_global_ids,
            row_count=EXPECTED_ROWS,
            label="load_from_disk_global_ids",
        )
        if not np.array_equal(loaded, self.hierarchy_selected_ids):
            raise OracleValidationError("working-set hierarchy order differs from selected global IDs")
        raster_ids = np.concatenate(
            (np.arange(EXPECTED_SKYBOX_ROWS, dtype="<i8"), loaded)
        )
        validate_global_ids(
            raster_ids,
            row_count=EXPECTED_ROWS,
            label="raster_global_ids",
        )
        self.raster_global_ids = raster_ids
        properties = self.gaussians.properties[
            torch.from_numpy(raster_ids.astype(np.int64, copy=False))
        ]
        sh_rest_count = 3
        expected = {
            "means": properties[:, 0:3],
            "log_scales": properties[:, 3:6],
            "raw_quaternions": properties[:, 6:10],
            "features_dc": properties[:, 10:13].unsqueeze(1),
            "opacity_logits": properties[:, 13:14],
            "features_rest": properties[:, 14:23].reshape(len(properties), sh_rest_count, 3),
        }
        actual = {
            "means": means,
            "log_scales": log_scales,
            "raw_quaternions": raw_quaternions,
            "features_dc": features_dc,
            "opacity_logits": opacity_logits,
            "features_rest": features_rest,
        }
        equality = {
            name: bool(torch.equal(expected[name].cpu(), actual[name].detach().cpu()))
            for name in expected
        }
        row_counts = {
            name: int(value.shape[0]) == int(raster_ids.size)
            for name, value in actual.items()
        }
        if not all(equality.values()) or not all(row_counts.values()):
            raise OracleValidationError(
                f"working-set/tensor row correspondence failed: equality={equality}, rows={row_counts}"
            )
        self.row_correspondence = {
            "mapping": "raster row -> raster_global_node_ids[row] -> serialized .dhier row",
            "skybox_rows": EXPECTED_SKYBOX_ROWS,
            "hierarchy_selected_rows": int(loaded.size),
            "total_raster_rows": int(raster_ids.size),
            "stored_tensor_exact_equality": equality,
            "tensor_row_count_agreement": row_counts,
            "valid": True,
        }
        descriptor = self._describe(
            "pre-raster", "raster_global_node_ids", raster_ids, save_payload=True
        )
        self._write_json(
            "working_set",
            "working-set.json",
            {
                "schema_version": "alod_selection_oracle_working_set.v1",
                "camera_stable_id": self.camera_stable_id,
                "global_node_ids": _id_summary(raster_ids),
                "global_node_id_payload": descriptor,
                "row_correspondence": self.row_correspondence,
            },
        )

    def record_pre_raster(
        self,
        *,
        means: Any,
        quaternions: Any,
        scales: Any,
        opacities: Any,
        colors: Any,
        view_matrices: Any,
        intrinsics: Any,
        width: int,
        height: int,
        background: Any,
        sh_degree: int,
        render_mode: str,
        packed: bool,
    ) -> None:
        if self.raster_global_ids is None or self.row_correspondence is None:
            raise OracleValidationError("working set must be recorded before raster inputs")
        gaussian_tensors = {
            "means": means,
            "quaternions": quaternions,
            "scales": scales,
            "opacities": opacities,
            "sh_features": colors,
        }
        row_counts = {
            name: int(value.shape[0]) == int(self.raster_global_ids.size)
            for name, value in gaussian_tensors.items()
        }
        if not all(row_counts.values()):
            raise OracleValidationError(f"pre-raster row-count mismatch: {row_counts}")
        tensors = {
            name: array_descriptor(value, device=_device(value))
            for name, value in {
                **gaussian_tensors,
                "view_matrices": view_matrices,
                "intrinsics": intrinsics,
                "background": background,
            }.items()
        }
        self.pre_raster_hashes = {
            name: descriptor["raw_cpu_c_order_sha256"] for name, descriptor in tensors.items()
        }
        combined_digest = hashlib.sha256(
            json.dumps(self.pre_raster_hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        payload = {
            "schema_version": PRE_RASTER_SCHEMA_VERSION,
            "camera_stable_id": self.camera_stable_id,
            "global_node_id_contract": "raster_global_node_ids[row] is the serialized source row",
            "counts": {
                "skybox_raster_rows": EXPECTED_SKYBOX_ROWS,
                "hierarchy_selected_raster_rows": int(
                    self.raster_global_ids.size - EXPECTED_SKYBOX_ROWS
                ),
                "total_raster_gaussians": int(self.raster_global_ids.size),
            },
            "row_correspondence": self.row_correspondence,
            "raster_global_node_ids": _id_summary(self.raster_global_ids),
            "tensors": tensors,
            "rasterization_arguments": {
                "width": int(width),
                "height": int(height),
                "sh_degree": int(sh_degree),
                "render_mode": render_mode,
                "packed": bool(packed),
            },
            "pre_raster_bundle_sha256": combined_digest,
        }
        self._write_json("pre_raster", "pre-raster.json", payload)

    def record_render(self, image: Any) -> None:
        descriptor = self._describe(
            "render", "rendered_rgb_chw", image, save_payload=self.payload_mode == "full"
        )
        payload = {
            "schema_version": "alod_selection_oracle_render.v1",
            "camera_stable_id": self.camera_stable_id,
            "rendered_rgb_chw": descriptor,
            "quality_evaluation_performed": False,
        }
        self._write_json("render", "optional-image.json", payload)
        complete = {
            "schema_version": "alod_selection_oracle_attempt_complete.v1",
            "camera_stable_id": self.camera_stable_id,
            "payload_mode": self.payload_mode,
            "stage_json_sha256": dict(sorted(self.stage_files.items())),
            "selected_hierarchy_ordered_sha256": global_id_hash(self.hierarchy_selected_ids),
            "selected_hierarchy_sorted_sha256": global_id_hash(
                self.hierarchy_selected_ids, sorted_ids=True
            ),
            "raster_global_ids_ordered_sha256": global_id_hash(self.raster_global_ids),
            "pre_raster_tensor_hashes": self.pre_raster_hashes,
            "render_sha256": descriptor["raw_cpu_c_order_sha256"],
        }
        self._write_json("complete", "complete.json", complete)
