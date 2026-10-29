import argparse
import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from cleanup_gaussian_checkpoint import run
from gaussian_cleanup import (
    BoundsFilterConfig,
    ExportPruneConfig,
    analyze_checkpoint,
    export_pruned_checkpoint,
    save_pruned_checkpoint,
)
from scene_runtime.loader import _validate_flat_payload


def _write_gaussian_ply(path: Path, rows):
    properties = [
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
        ("nx", "float"),
        ("ny", "float"),
        ("nz", "float"),
        ("f_dc_0", "float"),
        ("f_dc_1", "float"),
        ("f_dc_2", "float"),
    ]
    properties.extend((f"f_rest_{idx}", "float") for idx in range(45))
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
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(rows)}",
    ]
    header.extend(f"property {prop_type} {name}" for name, prop_type in properties)
    header.append("end_header")
    with path.open("wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        for row in rows:
            values = [float(row.get(name, 0.0)) for name, _ in properties]
            f.write(struct.pack("<" + "f" * len(values), *values))


def _fake_payload():
    means = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [100.0, 100.0, 100.0],
        ],
        dtype=torch.float32,
    )
    n = int(means.shape[0])
    return {
        "step": 17,
        "pose_adjust": {"kept": torch.tensor([1.0])},
        "splats": {
            "means": means,
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(n, 1),
            "scales": torch.log(
                torch.tensor(
                    [
                        [0.5, 0.5, 0.5],
                        [0.6, 0.6, 0.6],
                        [5.0, 0.5, 0.5],
                        [0.7, 0.7, 0.7],
                    ],
                    dtype=torch.float32,
                )
            ),
            "opacities": torch.logit(
                torch.tensor([0.005, 0.5, 0.9, 0.8], dtype=torch.float32)
            ),
            "sh0": torch.arange(n, dtype=torch.float32).reshape(n, 1, 1).repeat(1, 1, 3),
            "shN": torch.zeros((n, 15, 3), dtype=torch.float32),
        },
    }


class GaussianCleanupTest(unittest.TestCase):
    def test_metadata_only_accepts_gaussian_ply(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ply = tmp / "splats.ply"
            metadata = tmp / "metadata.json"
            _write_gaussian_ply(
                ply,
                [
                    {"x": 0.0, "y": 0.0, "z": 0.0, "opacity": 0.0, "rot_0": 1.0},
                    {"x": 1.0, "y": 1.0, "z": 1.0, "opacity": 0.0, "rot_0": 1.0},
                ],
            )

            run(
                argparse.Namespace(
                    input=ply,
                    metadata_output=metadata,
                    output=None,
                    metadata_only=True,
                    export_pruned_checkpoint=False,
                    overwrite=False,
                    ply_coordinate_mode="native",
                    opacity_min=None,
                    opacity_percentile=None,
                    max_scale=None,
                    max_scale_percentile=None,
                    roi_min=None,
                    roi_max=None,
                    position_percentile_min=0.0,
                    position_percentile_max=100.0,
                    padding=0.0,
                    export_opacity_min=None,
                    export_max_scale=None,
                    export_roi_min=None,
                    export_roi_max=None,
                    allow_unprotected_pruning=False,
                )
            )

            data = json.loads(metadata.read_text())
            self.assertEqual(data["total_gaussian_count"], 2)
            self.assertEqual(data["input_path"], str(ply))

    def test_opacity_scale_and_roi_filters_use_activated_values(self):
        payload = _fake_payload()
        result = analyze_checkpoint(
            payload,
            BoundsFilterConfig(
                opacity_min=0.01,
                max_scale=1.0,
                roi_min=(0.0, 0.0, 0.0),
                roi_max=(10.0, 10.0, 10.0),
            ),
        )

        self.assertEqual(result.bounds_valid_mask.tolist(), [False, True, False, False])
        self.assertEqual(result.export_keep_mask.tolist(), [True, True, True, True])
        counts = {entry["name"]: entry["valid_after"] for entry in result.diagnostics["bounds_filter_counts"]}
        self.assertEqual(counts["opacity_min"], 3)
        self.assertEqual(counts["max_scale"], 2)
        self.assertEqual(counts["roi_max"], 1)

    def test_robust_percentile_bounds_and_padding(self):
        payload = _fake_payload()
        result = analyze_checkpoint(
            payload,
            BoundsFilterConfig(
                position_percentile_min=0.0,
                position_percentile_max=50.0,
                padding=1.0,
            ),
        )

        robust = result.diagnostics["xyz_robust_percentiles_after_filtering"]
        self.assertEqual(robust["min"], [-1.0, -1.0, -1.0])
        self.assertEqual(robust["max"], [2.5, 2.5, 2.5])
        self.assertGreater(
            result.diagnostics["raw_to_filtered_aabb_volume_ratio"],
            1.0,
        )

    def test_export_mask_is_separate_from_bounds_mask(self):
        payload = _fake_payload()
        result = analyze_checkpoint(
            payload,
            BoundsFilterConfig(opacity_min=0.01, max_scale=1.0),
            export_config=ExportPruneConfig(opacity_min=0.7),
        )

        self.assertEqual(result.bounds_valid_mask.tolist(), [False, True, False, True])
        self.assertEqual(result.export_keep_mask.tolist(), [False, False, True, True])

    def test_export_prunes_all_splat_tensors_and_preserves_top_level_metadata(self):
        payload = _fake_payload()
        mask = torch.tensor([False, True, False, True])

        pruned = export_pruned_checkpoint(payload, mask)

        self.assertEqual(pruned["step"], 17)
        self.assertIn("pose_adjust", pruned)
        for value in pruned["splats"].values():
            if torch.is_tensor(value):
                self.assertEqual(int(value.shape[0]), 2)
        self.assertTrue(torch.equal(pruned["splats"]["means"], payload["splats"]["means"][mask]))

    def test_metadata_only_does_not_rewrite_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ckpt = tmp / "ckpt.pt"
            metadata = tmp / "metadata.json"
            torch.save(_fake_payload(), ckpt)
            before = torch.load(ckpt, map_location="cpu", weights_only=False)

            run(
                argparse.Namespace(
                    input=ckpt,
                    metadata_output=metadata,
                    output=None,
                    metadata_only=True,
                    export_pruned_checkpoint=False,
                    overwrite=False,
                    opacity_min=0.01,
                    opacity_percentile=None,
                    max_scale=None,
                    max_scale_percentile=None,
                    roi_min=None,
                    roi_max=None,
                    position_percentile_min=0.5,
                    position_percentile_max=99.5,
                    padding=0.0,
                    export_opacity_min=None,
                    export_max_scale=None,
                    export_roi_min=None,
                    export_roi_max=None,
                    allow_unprotected_pruning=False,
                )
            )

            after = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.assertTrue(metadata.exists())
            self.assertTrue(torch.equal(before["splats"]["means"], after["splats"]["means"]))
            self.assertEqual(json.loads(metadata.read_text())["total_gaussian_count"], 4)

    def test_export_refuses_overwrite_and_exported_checkpoint_is_flat_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ckpt = tmp / "ckpt.pt"
            output = tmp / "cleaned.pt"
            metadata = tmp / "metadata.json"
            torch.save(_fake_payload(), ckpt)
            output.write_text("already here")

            with self.assertRaises(FileExistsError):
                run(
                    argparse.Namespace(
                        input=ckpt,
                        metadata_output=metadata,
                        output=output,
                        metadata_only=False,
                        export_pruned_checkpoint=True,
                        overwrite=False,
                        opacity_min=0.01,
                        opacity_percentile=None,
                        max_scale=1.0,
                        max_scale_percentile=None,
                        roi_min=None,
                        roi_max=None,
                        position_percentile_min=0.5,
                        position_percentile_max=99.5,
                        padding=0.0,
                        export_opacity_min=0.7,
                        export_max_scale=None,
                        export_roi_min=None,
                        export_roi_max=None,
                        allow_unprotected_pruning=True,
                    )
                )
            self.assertFalse(metadata.exists())

            output.unlink()
            payload = _fake_payload()
            result = analyze_checkpoint(
                payload,
                BoundsFilterConfig(opacity_min=0.01, max_scale=1.0),
                export_config=ExportPruneConfig(opacity_min=0.7),
            )
            save_pruned_checkpoint(payload, result.export_keep_mask, ckpt, output, overwrite=False)
            exported = torch.load(output, map_location="cpu", weights_only=False)

            self.assertIsNone(_validate_flat_payload(exported))
            self.assertEqual(int(exported["splats"]["means"].shape[0]), 2)

    def test_export_requires_explicit_filters_and_unsafe_acknowledgement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ckpt = tmp / "ckpt.pt"
            output = tmp / "cleaned.pt"
            metadata = tmp / "metadata.json"
            torch.save(_fake_payload(), ckpt)

            base_args = dict(
                input=ckpt,
                metadata_output=metadata,
                output=output,
                metadata_only=False,
                export_pruned_checkpoint=True,
                overwrite=False,
                opacity_min=0.01,
                opacity_percentile=None,
                max_scale=1.0,
                max_scale_percentile=None,
                roi_min=None,
                roi_max=None,
                position_percentile_min=0.5,
                position_percentile_max=99.5,
                padding=0.0,
                export_opacity_min=None,
                export_max_scale=None,
                export_roi_min=None,
                export_roi_max=None,
                allow_unprotected_pruning=False,
            )

            with self.assertRaisesRegex(ValueError, "requires export-specific filters"):
                run(argparse.Namespace(**base_args))

            with self.assertRaisesRegex(ValueError, "Refusing destructive pruning"):
                run(
                    argparse.Namespace(
                        **{
                            **base_args,
                            "export_opacity_min": 0.7,
                        }
                    )
                )


if __name__ == "__main__":
    unittest.main()
