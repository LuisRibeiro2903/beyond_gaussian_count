import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks import citygaussian_canonical_campaign as campaign


class CanonicalCampaignAggregationTest(unittest.TestCase):
    def test_across_run_statistics_use_run_means(self):
        result = campaign.across_run_stats([10.0, 12.0, 14.0])
        self.assertEqual(result["mean_of_run_means"], 12.0)
        self.assertEqual(result["standard_deviation"], 2.0)
        self.assertAlmostEqual(result["coefficient_of_variation_percent"], 100.0 / 6.0)

    def test_validate_run_rejects_render_call_substituted_for_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            row = {
                "preparation_time_ms": 2.0,
                "render_call_time_ms": 3.0,
                "cuda_rasterization_time_ms": 2.5,
                "end_to_end_frame_time_ms": 3.0,
                "resident_gaussian_count": 10,
                "selected_gaussian_count": 5,
                "active_gaussian_count": 5,
            }
            (run_dir / "frames.jsonl").write_text(
                "\n".join(json.dumps(row) for _ in range(1440)) + "\n"
            )
            (run_dir / "summary.json").write_text(json.dumps({
                "frame_count": 1440,
                "configuration_id": "CG-LOD-RELEASE",
                "repetition_index": 1,
            }))
            (run_dir / "resolved_config.json").write_text(json.dumps({
                "warmup_frames": 1,
                "repetition_index": 1,
            }))
            with self.assertRaisesRegex(ValueError, "invalid end-to-end decomposition"):
                campaign.validate_run(run_dir, "CG-LOD-RELEASE", 1)

    def test_validate_run_accepts_canonical_timing_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            row = {
                "preparation_time_ms": 2.0,
                "render_call_time_ms": 3.0,
                "cuda_rasterization_time_ms": 2.5,
                "end_to_end_frame_time_ms": 5.0,
                "resident_gaussian_count": 10,
                "selected_gaussian_count": 5,
                "active_gaussian_count": 5,
            }
            (run_dir / "frames.jsonl").write_text(
                "\n".join(json.dumps(row) for _ in range(1440)) + "\n"
            )
            (run_dir / "summary.json").write_text(json.dumps({
                "frame_count": 1440,
                "configuration_id": "CG-LOD-RELEASE",
                "repetition_index": 2,
            }))
            (run_dir / "resolved_config.json").write_text(json.dumps({
                "warmup_frames": 1,
                "repetition_index": 2,
            }))
            rows, _ = campaign.validate_run(run_dir, "CG-LOD-RELEASE", 2)
            self.assertEqual(len(rows), 1440)

    def test_validate_run_rejects_nonfinite_selected_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            row = {
                "preparation_time_ms": 2.0,
                "render_call_time_ms": 3.0,
                "cuda_rasterization_time_ms": 2.5,
                "end_to_end_frame_time_ms": 5.0,
                "resident_gaussian_count": 10,
                "selected_gaussian_count": float("nan"),
                "active_gaussian_count": 10,
            }
            (run_dir / "frames.jsonl").write_text(
                "\n".join(json.dumps(row) for _ in range(1440)) + "\n"
            )
            (run_dir / "summary.json").write_text(json.dumps({
                "frame_count": 1440,
                "configuration_id": "CG-FINEST-FLAT",
                "repetition_index": 1,
            }))
            (run_dir / "resolved_config.json").write_text(json.dumps({
                "warmup_frames": 1,
                "repetition_index": 1,
            }))
            with self.assertRaisesRegex(ValueError, "non-finite selected_gaussian_count"):
                campaign.validate_run(run_dir, "CG-FINEST-FLAT", 1)

    def test_manifest_records_hashes_and_clean_provenance(self):
        fake_inputs = {
            "trajectory": {"path": "camera.json", "size_bytes": 1, "sha256": "a" * 64}
        }
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.object(campaign, "git_is_dirty", return_value=False), \
             patch.object(campaign, "git_value", return_value="b" * 40), \
             patch.object(campaign, "input_manifest", return_value=fake_inputs):
            manifest = campaign.initialize_manifest(Path(tmpdir))
        self.assertFalse(manifest["source"]["git_dirty"])
        self.assertEqual(manifest["source"]["git_commit"], "b" * 40)
        self.assertEqual(manifest["inputs"]["trajectory"]["sha256"], "a" * 64)
        self.assertEqual(set(manifest["configuration_files"]), set(campaign.CONFIG_IDS))

    def test_dirty_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.object(campaign, "git_is_dirty", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "clean Git worktree"):
                campaign.initialize_manifest(Path(tmpdir))


if __name__ == "__main__":
    unittest.main()
