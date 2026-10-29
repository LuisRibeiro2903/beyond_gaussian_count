#!/usr/bin/env python3
"""Evaluate Run009 quality with the pinned official no-cache evaluator path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.dont_write_bytecode = True


OFFICIAL_COMMIT = "73f547341547ae021ff036ad74fdf73a1240b450"


def canonical(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def distribution(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(data.min()),
        "median": float(np.median(data)),
        "mean": float(data.mean()),
        "p95": float(np.quantile(data, 0.95)),
        "maximum": float(data.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--camera-fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    dataset = args.dataset.resolve()
    hierarchy = args.hierarchy.resolve()
    fixture_path = args.camera_fixture.resolve()
    output_dir = args.output_dir.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"official source commit mismatch: {commit}")
    if sha256_file(hierarchy) != args.expected_sha256:
        raise RuntimeError("formal hierarchy SHA-256 mismatch")

    os.chdir(source)
    sys.path.insert(0, str(source))
    from gaussian_hierarchy._C import get_spt_cut_cuda
    from gaussian_renderer import render_gsplat
    from lpipsPyTorch.modules.lpips import LPIPS
    from scene.dataset_readers import readColmapSceneInfo
    from scene.gaussian_model import GaussianModel
    from utils.camera_utils import loadCam
    from utils.general_utils import safe_state
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected_names = [row["name"] for row in fixture["cameras"]]
    safe_state(False)
    info = readColmapSceneInfo(
        str(dataset), str(dataset / "images"), "", "", True, False, 10
    )
    if [row.image_name for row in info.test_cameras] != expected_names:
        raise RuntimeError("official eval camera order differs from frozen fixture")

    model = GaussianModel(1)
    model.create_from_hier(str(hierarchy), info.nerf_normalization["radius"], "")
    model.compact_gaussians("cpu", None, False, False)
    model.skybox_points = 100000
    target = 2.0 / float(info.train_cameras[0].focal_length)
    model.build_hierarchical_SPT(4.75, target, 100, use_bounding_spheres=False)
    props = model.properties
    means = props[:100000, 0:3].cuda().contiguous()
    scales = props[:100000, 3:6].cuda().contiguous()
    rotations = props[:100000, 6:10].cuda().contiguous()
    features_dc = props[:100000, 10:13].cuda().unsqueeze(1).contiguous()
    opacity = props[:100000, 13].cuda().unsqueeze(1).contiguous()
    features_rest = props[:100000, 14:23].cuda().reshape(100000, 3, 3).contiguous()
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    lpips_metric = LPIPS("vgg", "0.1").cuda().eval()
    pipeline = SimpleNamespace()
    camera_args = SimpleNamespace(
        resolution=2, data_device="cpu", train_test_exp=False
    )

    rows = []
    with torch.no_grad():
        for index, camera_info in enumerate(info.test_cameras):
            camera = loadCam(camera_args, index, camera_info, 1.0, True)
            camera.world_view_transform = camera.world_view_transform.cuda()
            camera.full_proj_transform = camera.full_proj_transform.cuda()
            camera.camera_center = camera.camera_center.cuda()
            bounds = torch.exp(model.upper_tree_scaling).max(dim=-1).values * 3.0
            planes = model.extract_frustum_planes(camera.full_proj_transform)
            cull = lambda indices: model.frustum_cull_spheres(
                model.upper_tree_xyz[indices], bounds[indices], planes
            )
            condition = lambda indices: model.min_distance_squared[indices] > (
                camera.camera_center - model.upper_tree_xyz[indices]
            ).square().sum(dim=-1)
            coarse = model.cut_hierarchy_on_condition(
                model.upper_tree_nodes,
                condition,
                return_upper_tree=False,
                root_node=0,
                leave_out_of_cut_condition=cull,
            )
            leaf = coarse[model.upper_tree_nodes[coarse, 2] == 0]
            spt_indices = model.upper_tree_nodes[leaf][
                model.upper_tree_nodes[leaf, 3] >= 0, 3
            ]
            upper = model.upper_tree_nodes[coarse][
                model.upper_tree_nodes[coarse, 3] <= 0, 5
            ]
            spt_upper = leaf[model.upper_tree_nodes[leaf, 3] >= 0]
            distances = (
                model.upper_tree_xyz[spt_upper] - camera.camera_center
            ).pow(2).sum(1).sqrt()
            selected, starts = get_spt_cut_cuda(
                len(spt_indices),
                model.SPT_gaussian_indices,
                model.SPT_starts,
                model.SPT_max,
                model.SPT_min,
                spt_indices,
                distances,
            )
            if starts.numel() and selected.numel() == int(starts[-1].item()):
                selected = selected
            selected_ids = torch.cat((upper, selected)).to("cpu")
            load = props[selected_ids, :].cuda()
            frame_means = torch.cat((means, load[:, 0:3])).contiguous()
            frame_opacity = torch.cat((opacity, load[:, 13:14])).contiguous()
            frame_scales = torch.cat((scales, load[:, 3:6])).contiguous()
            frame_rotations = torch.cat((rotations, load[:, 6:10])).contiguous()
            frame_dc = torch.cat(
                (features_dc, load[:, 10:13].unsqueeze(1))
            ).contiguous()
            frame_rest = torch.cat(
                (features_rest, load[:, 14:23].reshape(len(load), 3, 3))
            ).contiguous()
            torch.cuda.synchronize()
            started = time.perf_counter()
            render = render_gsplat(
                camera,
                frame_means,
                model.opacity_activation(frame_opacity),
                model.scaling_activation(frame_scales),
                model.rotation_activation(frame_rotations),
                frame_dc,
                frame_rest,
                pipeline,
                background,
                override_color=None,
                sh_degree=model.active_sh_degree,
            )
            torch.cuda.synchronize()
            render_ms = (time.perf_counter() - started) * 1000.0
            image = render["render"]
            gt = camera.original_image.cuda()
            if camera.alpha_mask is not None:
                alpha = camera.alpha_mask.cuda()
                image = image * alpha
                gt = gt * alpha
            psnr_value = float(psnr(image, gt).mean().double().item())
            ssim_value = float(ssim(image, gt).mean().double().item())
            lpips_value = float(lpips_metric(image[None], gt[None]).mean().double().item())

            no_frustum = lambda indices: torch.ones(
                len(indices), dtype=torch.bool, device=indices.device
            )
            coarse_nf = model.cut_hierarchy_on_condition(
                model.upper_tree_nodes,
                condition,
                return_upper_tree=False,
                root_node=0,
                leave_out_of_cut_condition=no_frustum,
            )
            leaf_nf = coarse_nf[model.upper_tree_nodes[coarse_nf, 2] == 0]
            spt_indices_nf = model.upper_tree_nodes[leaf_nf][
                model.upper_tree_nodes[leaf_nf, 3] >= 0, 3
            ]
            upper_nf = model.upper_tree_nodes[coarse_nf][
                model.upper_tree_nodes[coarse_nf, 3] <= 0, 5
            ]
            spt_upper_nf = leaf_nf[model.upper_tree_nodes[leaf_nf, 3] >= 0]
            distances_nf = (
                model.upper_tree_xyz[spt_upper_nf] - camera.camera_center
            ).pow(2).sum(1).sqrt()
            selected_nf, _ = get_spt_cut_cuda(
                len(spt_indices_nf),
                model.SPT_gaussian_indices,
                model.SPT_starts,
                model.SPT_max,
                model.SPT_min,
                spt_indices_nf,
                distances_nf,
            )
            selected_ids_nf = torch.cat((upper_nf, selected_nf)).to("cpu")
            load_nf = props[selected_ids_nf, :].cuda()
            render_nf = render_gsplat(
                camera,
                torch.cat((means, load_nf[:, 0:3])).contiguous(),
                model.opacity_activation(
                    torch.cat((opacity, load_nf[:, 13:14])).contiguous()
                ),
                model.scaling_activation(
                    torch.cat((scales, load_nf[:, 3:6])).contiguous()
                ),
                model.rotation_activation(
                    torch.cat((rotations, load_nf[:, 6:10])).contiguous()
                ),
                torch.cat((features_dc, load_nf[:, 10:13].unsqueeze(1))).contiguous(),
                torch.cat(
                    (features_rest, load_nf[:, 14:23].reshape(len(load_nf), 3, 3))
                ).contiguous(),
                pipeline,
                background,
                override_color=None,
                sh_degree=model.active_sh_degree,
            )["render"]
            if camera.alpha_mask is not None:
                render_nf = render_nf * camera.alpha_mask.cuda()
            difference = torch.abs(image - render_nf)
            rows.append(
                {
                    "evaluation_sequence_index": index,
                    "camera_name": expected_names[index],
                    "modality": fixture["cameras"][index]["modality"],
                    "width": int(camera.image_width),
                    "height": int(camera.image_height),
                    "selected_hierarchy_rows": int(selected_ids.numel()),
                    "total_raster_rows": int(frame_means.shape[0]),
                    "packed_projection_survivors": int(
                        torch.unique(render["packed_indices"]).numel()
                    ),
                    "render_time_ms": render_ms,
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                    "lpips_vgg": lpips_value,
                    "rendered_rgb_float32_sha256": tensor_hash(image),
                    "frustum_equivalence": {
                        "no_frustum_selected_hierarchy_rows": int(
                            selected_ids_nf.numel()
                        ),
                        "avoided_hierarchy_rows": int(
                            selected_ids_nf.numel() - selected_ids.numel()
                        ),
                        "rgb_exact": bool(torch.equal(image, render_nf)),
                        "rgb_max_abs": float(difference.max().item()),
                        "rgb_mean_abs": float(difference.mean().item()),
                    },
                }
            )
            del camera, load, frame_means, frame_opacity, frame_scales
            del frame_rotations, frame_dc, frame_rest, render, image, gt
            del selected_ids_nf, load_nf, render_nf, difference
            torch.cuda.empty_cache()

    def quality(indices: list[int]) -> dict:
        return {
            "camera_count": len(indices),
            "psnr": distribution([rows[index]["psnr"] for index in indices]),
            "ssim": distribution([rows[index]["ssim"] for index in indices]),
            "lpips_vgg": distribution(
                [rows[index]["lpips_vgg"] for index in indices]
            ),
        }

    aerial = [index for index, row in enumerate(rows) if row["modality"] == "aerial"]
    street = [index for index, row in enumerate(rows) if row["modality"] == "street"]
    summary = {
        "status": "PASS-ALOD-REPRO-QUALITY",
        "configuration_id": "ALOD-REPRO-QUALITY",
        "quality_path": "pinned official evaluator selection, no-cache host gather, released render_gsplat packed renderer",
        "metric_semantics": {
            "psnr": "official project utility; per-channel PSNR then channel mean",
            "ssim": "official project 11x11 Gaussian-window SSIM",
            "lpips": "official bundled calibrated LPIPS 0.1 with ImageNet VGG16; explicit batch dimension correction",
            "alpha": "released evaluator alpha application, including its second multiplication of already-masked GT",
        },
        "camera_fixture": str(fixture_path),
        "hierarchy_sha256": args.expected_sha256,
        "official_source_commit": commit,
        "official_source_clean": subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ) == "",
        "all": quality(list(range(len(rows)))),
        "aerial": quality(aerial),
        "street": quality(street),
        "render_time_ms": distribution([row["render_time_ms"] for row in rows]),
        "selected_hierarchy_rows": distribution(
            [row["selected_hierarchy_rows"] for row in rows]
        ),
        "packed_projection_survivors": distribution(
            [row["packed_projection_survivors"] for row in rows]
        ),
        "frustum_rendering_equivalence": {
            "exact_image_count": sum(
                row["frustum_equivalence"]["rgb_exact"] for row in rows
            ),
            "rgb_max_abs": distribution(
                [row["frustum_equivalence"]["rgb_max_abs"] for row in rows]
            ),
            "rgb_mean_abs": distribution(
                [row["frustum_equivalence"]["rgb_mean_abs"] for row in rows]
            ),
            "avoided_hierarchy_rows": distribution(
                [
                    row["frustum_equivalence"]["avoided_hierarchy_rows"]
                    for row in rows
                ]
            ),
            "no_frustum_boolean_device_correction": "CUDA boolean mask; released false-branch constructs a CPU mask incompatible with CUDA indexing",
        },
        "project_extensions_enabled": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_camera.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(canonical(summary), encoding="utf-8")
    print(canonical({"status": summary["status"], "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
