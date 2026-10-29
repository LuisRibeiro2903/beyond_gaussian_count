import argparse
import inspect
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

import citygaussian_assets
from citygaussian_assets import (
    citygaussian_block_ids,
    citygaussian_mad_bounds_np,
    citygaussian_official_assignments,
    inspect_vq_directory,
    summarize_ply,
    validate_assets,
)


def _write_ply(path: Path, rows, sh_degree: int = 2, malformed: str | None = None) -> None:
    f_rest_count = 3 * ((sh_degree + 1) ** 2 - 1)
    if malformed == "invalid_sh":
        f_rest_count = 6
    properties = [
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
        ("nx", "float"),
        ("ny", "float"),
        ("nz", "float"),
    ]
    if malformed != "missing_dc":
        properties.extend((f"f_dc_{idx}", "float") for idx in range(3))
    properties.extend((f"f_rest_{idx}", "float") for idx in range(f_rest_count))
    properties.extend(
        [
            ("opacity", "float"),
            ("scale_0", "float"),
            ("scale_1", "float"),
            ("scale_2", "float"),
            ("rot_0", "float"),
            ("rot_1", "float"),
            ("rot_2", "float"),
            ("rot_3", "float"),
        ]
    )
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(rows)}"]
    header.extend(f"property {prop_type} {name}" for name, prop_type in properties)
    header.append("end_header")
    path.write_bytes(("\n".join(header) + "\n").encode("ascii"))
    with path.open("ab") as f:
        for row in rows:
            values = [float(row.get(name, 0.0)) for name, _type in properties]
            f.write(struct.pack("<" + "f" * len(values), *values))


def _write_vq(extreme_dir: Path, count: int, metadata_count: int | None = None, missing: str | None = None) -> None:
    extreme_dir.mkdir(parents=True)
    input_pc_num = metadata_count if metadata_count is not None else count
    codebook_size = 8
    codebook_dim = 27
    bit_length = int(math.log2(codebook_size))
    non_vq_count = max(1, count // 2)
    vq_count = input_pc_num - non_vq_count
    fixtures = {
        "codebook.npz": {"arr_0": np.zeros((codebook_size, codebook_dim), dtype=np.float16)},
        "metadata.npz": {
            "metadata": np.array(
                {
                    "input_pc_num": input_pc_num,
                    "input_pc_dim": 41,
                    "codebook_size": codebook_size,
                    "codebook_dim": codebook_dim,
                },
                dtype=object,
            )
        },
        "non_vq_feats.npz": {"arr_0": np.zeros((non_vq_count, codebook_dim), dtype=np.float16)},
        "non_vq_mask.npz": {
            "arr_0": np.packbits(np.array([1] * non_vq_count + [0] * max(0, count - non_vq_count), dtype=np.uint8))
        },
        "other_attribute.npz": {"arr_0": np.zeros((count, 8), dtype=np.float16)},
        "vq_indexs.npz": {
            "arr_0": np.zeros(((max(0, vq_count) * bit_length + 7) // 8,), dtype=np.uint8)
        },
        "xyz.npz": {"arr_0": np.zeros((count, 3), dtype=np.float32)},
    }
    for name, arrays in fixtures.items():
        if name == missing:
            continue
        np.savez(extreme_dir / name, **arrays)


def _write_yaml(path: Path, names) -> None:
    lod_lines = ",\n                  ".join(f'"output/{name}"' for name in names)
    path.write_text(
        f"""model_params: {{
    model_config: {{
        name: "GaussianModelLOD",
        kwargs: {{}}
    }},
    sh_degree: 2,
    source_path: "data/matrix_city/aerial/train/block_all",
    partition_name: "mc_aerial_c36",
    model_path: "",
    images: "images",
    block_dim: [2, 2, 1],
    aabb: [0, 0, 0, 2, 2, 1],
    lod_configs: [{lod_lines}],
    white_background: False,
    data_device: "cuda",
    eval: False,
}}
pipeline_params: {{
    compute_cov3D_python: False
}}
"""
    )


def _make_level(root: Path, name: str, rows) -> None:
    level = root / name
    level.mkdir(parents=True)
    _write_ply(level / "point_cloud.ply", rows, sh_degree=2)
    _write_vq(level / "extreme_saving", len(rows))


def _cell_rows() -> list[dict[str, float]]:
    return [
        {"x": 0.25, "y": 0.25, "z": 0.5, "rot_0": 1.0},
        {"x": 1.25, "y": 0.25, "z": 0.5, "rot_0": 1.0},
        {"x": 0.25, "y": 1.25, "z": 0.5, "rot_0": 1.0},
        {"x": 1.25, "y": 1.25, "z": 0.5, "rot_0": 1.0},
    ]


class CityGaussianAssetValidationTest(unittest.TestCase):
    def test_valid_fused_asset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ply = Path(tmpdir) / "fused.ply"
            _write_ply(ply, _cell_rows(), sh_degree=3)

            summary = summarize_ply(ply, "fused_no_lod_model")

            self.assertEqual(summary["validation_status"], "valid")
            self.assertEqual(summary["gaussian_vertex_count"], 4)
            self.assertEqual(summary["inferred_sh_degree"], 3)

    def test_valid_three_level_asset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fused = tmp / "fused.ply"
            yaml_path = tmp / "lod.yaml"
            lod_root = tmp / "LoD"
            names = [
                "mc_aerial_c36_light_50_vq",
                "mc_aerial_c36_light_66_vq",
                "mc_aerial_c36_light_75_vq",
            ]
            _write_ply(fused, _cell_rows(), sh_degree=3)
            _write_yaml(yaml_path, names)
            for name in names:
                _make_level(lod_root, name, _cell_rows())

            manifest, exit_code = validate_assets(
                argparse.Namespace(
                    reference_repo=None,
                    reference_repository_commit="test",
                    scene="fixture",
                    fused_model=str(fused),
                    lod_root=str(lod_root),
                    camera_metadata=None,
                    yaml=str(yaml_path),
                    custom_asset=[],
                    block_dim=None,
                    aabb=None,
                    skip_cell_counts=False,
                    output=None,
                )
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(manifest["validation_status"], "valid")
            self.assertEqual(manifest["gaussian_counts"]["mc_aerial_c36_light_50_vq"], 4)
            self.assertTrue(manifest["lod_models"][0]["ply"]["cell_counts"]["all_cells_nonempty"])

    def test_missing_lod_directory_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fused = tmp / "fused.ply"
            yaml_path = tmp / "lod.yaml"
            lod_root = tmp / "LoD"
            _write_ply(fused, _cell_rows(), sh_degree=3)
            _write_yaml(yaml_path, ["mc_aerial_c36_light_50_vq"])

            manifest, exit_code = validate_assets(
                argparse.Namespace(
                    reference_repo=None,
                    reference_repository_commit="test",
                    scene="fixture",
                    fused_model=str(fused),
                    lod_root=str(lod_root),
                    camera_metadata=None,
                    yaml=str(yaml_path),
                    custom_asset=[],
                    block_dim=None,
                    aabb=None,
                    skip_cell_counts=False,
                    output=None,
                )
            )

            self.assertNotEqual(exit_code, 0)
            self.assertIn("missing LoD directory", "\n".join(manifest["validation_errors"]))

    def test_missing_npz_member_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            level = Path(tmpdir) / "level"
            level.mkdir()
            _write_ply(level / "point_cloud.ply", _cell_rows(), sh_degree=2)
            _write_vq(level / "extreme_saving", 4, missing="xyz.npz")

            info = inspect_vq_directory(level / "extreme_saving", adjacent_ply_count=4)

            self.assertEqual(info["validation_status"], "invalid")
            self.assertIn("missing xyz.npz", info["errors"])

    def test_yaml_order_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fused = tmp / "fused.ply"
            yaml_path = tmp / "lod.yaml"
            lod_root = tmp / "LoD"
            _write_ply(fused, _cell_rows(), sh_degree=3)
            _write_yaml(yaml_path, ["mc_aerial_c36_light_50_vq", "mc_aerial_c36_light_66_vq"])
            _make_level(lod_root, "mc_aerial_c36_light_50_vq", _cell_rows())

            manifest, exit_code = validate_assets(
                argparse.Namespace(
                    reference_repo=None,
                    reference_repository_commit="test",
                    scene="fixture",
                    fused_model=str(fused),
                    lod_root=str(lod_root),
                    camera_metadata=None,
                    yaml=str(yaml_path),
                    custom_asset=[],
                    block_dim=None,
                    aabb=None,
                    skip_cell_counts=False,
                    output=None,
                )
            )

            self.assertNotEqual(exit_code, 0)
            self.assertIn("mc_aerial_c36_light_66_vq", "\n".join(manifest["validation_errors"]))

    def test_inconsistent_gaussian_counts_between_ply_and_vq_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            level = Path(tmpdir) / "level"
            level.mkdir()
            _write_ply(level / "point_cloud.ply", _cell_rows(), sh_degree=2)
            _write_vq(level / "extreme_saving", 4, metadata_count=5)

            info = inspect_vq_directory(level / "extreme_saving", adjacent_ply_count=4)

            self.assertEqual(info["validation_status"], "invalid")
            self.assertIn("does not match adjacent PLY count", "\n".join(info["errors"]))

    def test_malformed_ply_properties_are_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ply = Path(tmpdir) / "bad.ply"
            _write_ply(ply, _cell_rows(), malformed="missing_dc")

            summary = summarize_ply(ply, "bad")

            self.assertEqual(summary["validation_status"], "invalid")
            self.assertIn("f_dc", "\n".join(summary["errors"]))

    def test_invalid_sh_layout_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ply = Path(tmpdir) / "bad_sh.ply"
            _write_ply(ply, _cell_rows(), malformed="invalid_sh")

            summary = summarize_ply(ply, "bad")

            self.assertEqual(summary["validation_status"], "invalid")
            self.assertIn("does not map to an integer degree", "\n".join(summary["errors"]))

    def test_incomplete_cell_level_expectations_are_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fused = tmp / "fused.ply"
            yaml_path = tmp / "lod.yaml"
            lod_root = tmp / "LoD"
            name = "mc_aerial_c36_light_50_vq"
            rows = [{"x": 0.25, "y": 0.25, "z": 0.5, "rot_0": 1.0}]
            _write_ply(fused, rows, sh_degree=3)
            _write_yaml(yaml_path, [name])
            _make_level(lod_root, name, rows)

            manifest, exit_code = validate_assets(
                argparse.Namespace(
                    reference_repo=None,
                    reference_repository_commit="test",
                    scene="fixture",
                    fused_model=str(fused),
                    lod_root=str(lod_root),
                    camera_metadata=None,
                    yaml=str(yaml_path),
                    custom_asset=[],
                    block_dim=None,
                    aabb=None,
                    skip_cell_counts=False,
                    output=None,
                )
            )

            self.assertNotEqual(exit_code, 0)
            self.assertIn("not every official cell", "\n".join(manifest["validation_errors"]))

    def test_official_versus_custom_asset_classification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fused = tmp / "official" / "point_cloud.ply"
            custom = tmp / "partitioned" / "mc_aerial_6x6.pt"
            fused.parent.mkdir()
            custom.parent.mkdir()
            _write_ply(fused, _cell_rows(), sh_degree=3)
            custom.write_bytes(b"not a real checkpoint")

            manifest, _exit_code = validate_assets(
                argparse.Namespace(
                    reference_repo=None,
                    reference_repository_commit="test",
                    scene="fixture",
                    fused_model=str(fused),
                    lod_root=None,
                    camera_metadata=None,
                    yaml=None,
                    custom_asset=[str(fused), str(custom)],
                    block_dim=None,
                    aabb=None,
                    skip_cell_counts=False,
                    output=None,
                )
            )

            classes = {entry["path"]: entry["classification"] for entry in manifest["asset_classification"]}
            self.assertEqual(classes[str(fused)], "official_citygaussian_asset")
            self.assertEqual(classes[str(custom)], "project_custom_partition_not_official_citygaussian")

    def test_manifest_serialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ply = Path(tmpdir) / "fused.ply"
            _write_ply(ply, _cell_rows(), sh_degree=3)

            manifest, exit_code = validate_assets(
                argparse.Namespace(
                    reference_repo=None,
                    reference_repository_commit="test",
                    scene="fixture",
                    fused_model=str(ply),
                    lod_root=None,
                    camera_metadata=None,
                    yaml=None,
                    custom_asset=[],
                    block_dim=None,
                    aabb=None,
                    skip_cell_counts=False,
                    output=None,
                )
            )

            self.assertEqual(exit_code, 0)
            self.assertIsInstance(json.dumps(manifest), str)
            self.assertIn("schema_version", manifest)

    def test_validation_module_does_not_import_torch_or_cuda(self):
        source = inspect.getsource(citygaussian_assets)
        self.assertNotIn("import torch", source)
        self.assertNotIn("cuda", source.lower().split("runtime_construction")[0])

    def test_block_ids_follow_official_flattening_order(self):
        xyz = np.array([[0.25, 0.25, 0.5], [1.25, 0.25, 0.5], [0.25, 1.25, 0.5]], dtype=np.float32)
        ids = citygaussian_block_ids(xyz, block_dim=[2, 2, 1], aabb=[0, 0, 0, 2, 2, 1])
        self.assertEqual(ids.tolist(), [0, 1, 2])

    def test_official_assignment_reports_exact_once_diagnostics(self):
        xyz = np.array([[0.25, 0.25, 0.5], [1.25, 0.25, 0.5], [0.25, 1.25, 0.5]], dtype=np.float32)
        assigned, ids = citygaussian_official_assignments(
            xyz,
            block_dim=[2, 2, 1],
            aabb=[0, 0, 0, 2, 2, 1],
        )

        self.assertEqual(assigned.tolist(), [True, True, True])
        self.assertEqual(ids.tolist(), [0, 1, 2])

    def test_cell_metadata_includes_mad_bounds_and_average_scales(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ply = Path(tmpdir) / "cells.ply"
            rows = [
                {"x": 0.25, "y": 0.25, "z": 0.5, "scale_0": 0.0, "scale_1": 0.0, "scale_2": 0.0, "rot_0": 1.0},
                {"x": 1.25, "y": 0.25, "z": 0.5, "scale_0": math.log(2.0), "scale_1": 0.0, "scale_2": 0.0, "rot_0": 1.0},
                {"x": 0.25, "y": 1.25, "z": 0.5, "scale_0": 0.0, "scale_1": math.log(3.0), "scale_2": 0.0, "rot_0": 1.0},
                {"x": 1.25, "y": 1.25, "z": 0.5, "scale_0": 0.0, "scale_1": 0.0, "scale_2": math.log(4.0), "rot_0": 1.0},
            ]
            _write_ply(ply, rows, sh_degree=2)

            metadata = citygaussian_assets.compute_citygaussian_cell_counts(
                ply,
                block_dim=[2, 2, 1],
                aabb=[0, 0, 0, 2, 2, 1],
            )

        self.assertEqual(metadata["total_assigned_count"], 4)
        self.assertEqual(metadata["unassigned_count"], 0)
        self.assertEqual(metadata["duplicated_assignment_count"], 0)
        self.assertEqual(metadata["cell_gaussian_counts"], [1, 1, 1, 1])
        self.assertEqual(metadata["mad_bounds"][0]["min"], [0.25, 0.25, 0.5])
        self.assertAlmostEqual(metadata["average_scales"][3], 4.0)

    def test_numpy_mad_bounds_match_reference_formula(self):
        xyz = np.array([[0.0, 0.5, 0.5], [1.0, 0.5, 0.5], [10.0, 0.5, 0.5]], dtype=np.float32)
        xyz_min, xyz_max = citygaussian_mad_bounds_np(xyz, n=4.0)

        self.assertEqual(xyz_min.tolist(), [0.0, 0.5, 0.5])
        self.assertEqual(xyz_max.tolist(), [5.0, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
