"""Evaluate gsplat checkpoints on a fixed COLMAP camera/image set.

This is intentionally separate from gsplat/examples/simple_trainer.py because
that trainer's validation split depends on --test_every. For stride/subset
experiments, this script keeps the evaluation cameras fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
GSPLOT_EXAMPLES = ROOT / "gsplat" / "examples"
if str(GSPLOT_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(GSPLOT_EXAMPLES))
GS_FRAMEWORK = ROOT / "gs_framework"
if str(GS_FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(GS_FRAMEWORK))

from datasets.normalize import (  # noqa: E402
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)
from gsplat.rendering import rasterization  # noqa: E402
from pycolmap import SceneManager  # noqa: E402
from scene_runtime.ply_splats import is_ply_path, load_gaussian_ply  # noqa: E402


DEFAULT_EVAL_DATA_DIR = (
    ROOT / "CityGaussian" / "data" / "matrix_city" / "aerial" / "test" / "block_all_test"
)
DEFAULT_TRAIN_DATA_DIR = (
    ROOT / "CityGaussian" / "data" / "matrix_city" / "aerial" / "train" / "block_all"
)


@dataclass
class EvalSample:
    name: str
    image_path: Path
    camtoworld: np.ndarray
    K: np.ndarray
    width: int
    height: int
    camera_id: int
    params: np.ndarray
    mapx: Optional[np.ndarray] = None
    mapy: Optional[np.ndarray] = None
    roi: Optional[Tuple[int, int, int, int]] = None
    mask: Optional[np.ndarray] = None


def _get_rel_paths(path_dir: Path) -> List[str]:
    paths: List[str] = []
    for dp, _, filenames in os.walk(path_dir):
        for filename in filenames:
            paths.append(os.path.relpath(Path(dp) / filename, path_dir))
    return paths


def _camera_kind(camera_type: object) -> str:
    text = str(camera_type).upper()
    if "FISHEYE" in text:
        return "fisheye"
    if "PINHOLE" in text or "RADIAL" in text or "OPENCV" in text:
        return "perspective"
    if text in {"0", "1", "2", "3", "4"}:
        return "perspective"
    raise ValueError(f"Unsupported COLMAP camera type: {camera_type}")


def _camera_params(camera: object) -> Tuple[np.ndarray, np.ndarray, str, Tuple[int, int]]:
    fx = float(camera.fx)
    fy = float(camera.fy)
    cx = float(camera.cx)
    cy = float(camera.cy)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    params = np.array(getattr(camera, "params", []), dtype=np.float32)
    kind = _camera_kind(getattr(camera, "camera_type", "PINHOLE"))
    imsize = (int(camera.width), int(camera.height))
    return K, params, kind, imsize


def _resolve_image_dir(data_dir: Path, image_dir_name: Optional[str]) -> Path:
    if image_dir_name:
        image_dir = data_dir / image_dir_name
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        return image_dir

    for candidate in ("images", "input"):
        image_dir = data_dir / candidate
        if image_dir.exists():
            return image_dir

    raise FileNotFoundError(
        f"Could not find an image directory under {data_dir}. "
        "Pass --eval-image-dir-name or --train-image-dir-name explicitly."
    )


def _build_image_path_map(image_dir: Path, image_names: Sequence[str]) -> Dict[str, Path]:
    image_files = sorted(_get_rel_paths(image_dir))
    direct = {name: image_dir / name for name in image_names if (image_dir / name).exists()}
    if len(direct) == len(image_names):
        return direct

    if len(image_files) < len(image_names):
        missing = sorted(set(image_names) - set(direct))[:5]
        raise FileNotFoundError(
            f"Image directory {image_dir} has too few files for COLMAP image list. "
            f"Missing examples: {missing}"
        )

    # Downsampled folders sometimes use a different extension but preserve sort order.
    return {
        image_name: image_dir / image_file
        for image_name, image_file in zip(sorted(image_names), image_files)
    }


class ColmapFixedSet:
    """Minimal COLMAP loader for fixed-camera evaluation."""

    def __init__(
        self,
        data_dir: Path,
        image_dir_name: Optional[str],
        data_factor: float,
        normalize: bool,
        apply_transform: Optional[np.ndarray] = None,
        image_list: Optional[Iterable[str]] = None,
    ) -> None:
        self.data_dir = data_dir
        self.image_dir = _resolve_image_dir(data_dir, image_dir_name)
        colmap_dir = data_dir / "sparse" / "0"
        if not colmap_dir.exists():
            colmap_dir = data_dir / "sparse"
        if not colmap_dir.exists():
            raise FileNotFoundError(f"COLMAP sparse directory does not exist: {colmap_dir}")

        manager = SceneManager(str(colmap_dir))
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        if len(manager.images) == 0:
            raise ValueError(f"No registered COLMAP images found in {colmap_dir}")

        bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 4)
        records = []
        for image_id, image in manager.images.items():
            rot = image.R().astype(np.float32)
            trans = image.tvec.reshape(3, 1).astype(np.float32)
            w2c = np.concatenate([np.concatenate([rot, trans], axis=1), bottom], axis=0)
            records.append((image.name, int(image.camera_id), w2c))

        records.sort(key=lambda item: item[0])
        image_names = [record[0] for record in records]
        image_filter = set(image_list or [])
        if image_filter:
            records = [record for record in records if record[0] in image_filter]
            missing = sorted(image_filter - {record[0] for record in records})
            if missing:
                raise ValueError(f"{len(missing)} requested images were not in COLMAP: {missing[:5]}")

        image_path_map = _build_image_path_map(self.image_dir, image_names)
        camtoworlds = np.linalg.inv(np.stack([record[2] for record in records], axis=0))

        points = manager.points3D.astype(np.float32)
        if normalize:
            normalization_transform = self._normalization_transform(camtoworlds, points)
            camtoworlds = transform_cameras(normalization_transform, camtoworlds)
            points = transform_points(normalization_transform, points)
        elif apply_transform is not None:
            camtoworlds = transform_cameras(apply_transform, camtoworlds)

        self.transform = normalization_transform if normalize else np.eye(4, dtype=np.float32)
        self.samples: List[EvalSample] = []
        cameras = {}
        for camera_id, camera in manager.cameras.items():
            K, params, kind, imsize = _camera_params(camera)
            K[:2, :] /= float(data_factor)
            width = int(round(imsize[0] / float(data_factor)))
            height = int(round(imsize[1] / float(data_factor)))
            cameras[int(camera_id)] = (K, params, kind, width, height)

        for idx, (name, camera_id, _) in enumerate(records):
            K, params, kind, width, height = cameras[camera_id]
            K = K.copy()
            mapx = mapy = mask = None
            roi = None

            if len(params) > 0:
                if kind == "perspective":
                    K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                        K, params, (width, height), 0
                    )
                    mapx, mapy = cv2.initUndistortRectifyMap(
                        K, params, None, K_undist, (width, height), cv2.CV_32FC1
                    )
                    K = K_undist
                    roi = tuple(int(v) for v in roi_undist)
                    width, height = int(roi[2]), int(roi[3])
                elif kind == "fisheye":
                    grid_x, grid_y = np.meshgrid(
                        np.arange(width, dtype=np.float32),
                        np.arange(height, dtype=np.float32),
                        indexing="xy",
                    )
                    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                    x1 = (grid_x - cx) / fx
                    y1 = (grid_y - cy) / fy
                    theta = np.sqrt(x1**2 + y1**2)
                    padded = np.pad(params[:4], (0, max(0, 4 - len(params[:4]))))
                    r = (
                        1.0
                        + padded[0] * theta**2
                        + padded[1] * theta**4
                        + padded[2] * theta**6
                        + padded[3] * theta**8
                    )
                    mapx = (fx * x1 * r + width // 2).astype(np.float32)
                    mapy = (fy * y1 * r + height // 2).astype(np.float32)
                    full_mask = (mapx > 0) & (mapy > 0) & (mapx < width - 1) & (mapy < height - 1)
                    y_indices, x_indices = np.nonzero(full_mask)
                    y_min, y_max = y_indices.min(), y_indices.max() + 1
                    x_min, x_max = x_indices.min(), x_indices.max() + 1
                    mask = full_mask[y_min:y_max, x_min:x_max]
                    K[0, 2] -= x_min
                    K[1, 2] -= y_min
                    roi = (x_min, y_min, x_max - x_min, y_max - y_min)
                    width, height = int(roi[2]), int(roi[3])

            self.samples.append(
                EvalSample(
                    name=name,
                    image_path=image_path_map[name],
                    camtoworld=camtoworlds[idx].astype(np.float32),
                    K=K.astype(np.float32),
                    width=width,
                    height=height,
                    camera_id=camera_id,
                    params=params,
                    mapx=mapx,
                    mapy=mapy,
                    roi=roi,
                    mask=mask,
                )
            )

        print(
            f"[EvalSet] {len(self.samples)} images from {data_dir} "
            f"using image dir {self.image_dir.name}"
        )

    @staticmethod
    def _normalization_transform(camtoworlds: np.ndarray, points: np.ndarray) -> np.ndarray:
        transform1 = similarity_from_cameras(camtoworlds)
        transformed_points = transform_points(transform1, points)
        transform2 = align_principal_axes(transformed_points)
        transform = transform2 @ transform1
        transformed_points = transform_points(transform2, transformed_points)
        if np.median(transformed_points[:, 2]) > np.mean(transformed_points[:, 2]):
            transform3 = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            transform = transform3 @ transform
        return transform.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, sample: EvalSample) -> Tuple[Tensor, Optional[Tensor]]:
        image = imageio.imread(sample.image_path)[..., :3]
        if sample.mapx is not None and sample.mapy is not None and sample.roi is not None:
            image = cv2.remap(image, sample.mapx, sample.mapy, cv2.INTER_LINEAR)
            x, y, w, h = sample.roi
            image = image[y : y + h, x : x + w]
        pixels = torch.from_numpy(image).float() / 255.0
        mask = torch.from_numpy(sample.mask).bool() if sample.mask is not None else None
        return pixels, mask


def _load_image_list(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    names = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def _load_single_splat_source(
    path: Path,
    device: torch.device,
    apply_supersplat_transform: bool,
) -> Tuple[Dict[str, Tensor], int]:
    if is_ply_path(path):
        return load_gaussian_ply(path, apply_supersplat_transform=apply_supersplat_transform), -1
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "splats" not in ckpt:
        raise ValueError(f"{path} is not a valid gsplat checkpoint with a 'splats' dict.")
    return ckpt["splats"], int(ckpt.get("step", -1))


def _load_splats(
    scene_paths: Sequence[Path],
    device: torch.device,
    apply_supersplat_transform: bool,
) -> Tuple[Dict[str, Tensor], int]:
    if not scene_paths:
        raise ValueError("At least one scene path is required.")

    loaded = [
        _load_single_splat_source(path, device, apply_supersplat_transform)
        for path in scene_paths
    ]
    splat_dicts = [splats for splats, _ in loaded]
    required = ["means", "quats", "scales", "opacities", "sh0", "shN"]
    for path, splat_dict in zip(scene_paths, splat_dicts):
        missing = [key for key in required if key not in splat_dict]
        if missing:
            raise ValueError(f"{path} is missing required splat keys: {missing}")
    reference_shapes = {
        key: tuple(splat_dicts[0][key].shape[1:])
        for key in required
    }
    for path, splat_dict in zip(scene_paths[1:], splat_dicts[1:]):
        mismatched = [
            f"{key}: expected [N, {reference_shapes[key]}], got {tuple(splat_dict[key].shape)}"
            for key in required
            if tuple(splat_dict[key].shape[1:]) != reference_shapes[key]
        ]
        if mismatched:
            raise ValueError(
                f"{path} cannot be concatenated with the first scene file because "
                "the splat tensor shapes differ after the Gaussian dimension: "
                + "; ".join(mismatched)
            )

    splats = {
        key: torch.cat([splat_dict[key].to(device) for splat_dict in splat_dicts], dim=0)
        for key in required
    }
    step = next((step for _, step in loaded if step >= 0), -1)
    return splats, step


def _render(
    splats: Dict[str, Tensor],
    camtoworld: Tensor,
    K: Tensor,
    width: int,
    height: int,
    sh_degree: int,
    packed: bool,
    rasterize_mode: str,
    camera_model: str,
    near_plane: float,
    far_plane: float,
) -> Tensor:
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    render_colors, _, _ = rasterization(
        means=splats["means"],
        quats=F.normalize(splats["quats"], p=2, dim=-1),
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=colors,
        viewmats=torch.linalg.inv(camtoworld),
        Ks=K,
        width=width,
        height=height,
        packed=packed,
        sh_degree=sh_degree,
        rasterize_mode=rasterize_mode,
        camera_model=camera_model,
        near_plane=near_plane,
        far_plane=far_plane,
    )
    return torch.clamp(render_colors[..., :3], 0.0, 1.0)


def _safe_name(path: Path) -> str:
    name = path.name
    for suffix in (".ckpt", ".ply", ".pt"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def evaluate_checkpoint(
    ckpt_paths: Sequence[Path],
    eval_set: ColmapFixedSet,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    splats, step = _load_splats(
        ckpt_paths,
        device,
        apply_supersplat_transform=args.ply_coordinate_mode == "supersplat",
    )
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type=args.lpips_net, normalize=False
    ).to(device)

    metrics: Dict[str, List[float]] = {"psnr": [], "ssim": [], "lpips": [], "render_time": []}
    per_image = []
    render_dir = args.output_dir / _safe_name(ckpt_paths[0]) / "renders"
    if args.save_renders:
        render_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(eval_set.samples, desc=f"Evaluating {ckpt_paths[0].name}")):
            pixels, mask = eval_set.load_image(sample)
            pixels = pixels.to(device)
            if mask is not None:
                mask = mask.to(device)
                pixels = pixels * mask[..., None]

            camtoworld = torch.from_numpy(sample.camtoworld).to(device).unsqueeze(0)
            K = torch.from_numpy(sample.K).to(device).unsqueeze(0)

            if device.type == "cuda":
                torch.cuda.synchronize()
            tic = time.perf_counter()
            colors = _render(
                splats=splats,
                camtoworld=camtoworld,
                K=K,
                width=sample.width,
                height=sample.height,
                sh_degree=args.sh_degree,
                packed=args.packed,
                rasterize_mode=args.rasterize_mode,
                camera_model=args.camera_model,
                near_plane=args.near_plane,
                far_plane=args.far_plane,
            ).squeeze(0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            render_time = time.perf_counter() - tic

            if mask is not None:
                colors = colors * mask[..., None]

            pixels_p = pixels.permute(2, 0, 1).unsqueeze(0)
            colors_p = colors.permute(2, 0, 1).unsqueeze(0)
            image_stats = {
                "image": sample.name,
                "psnr": float(psnr_metric(colors_p, pixels_p).item()),
                "ssim": float(ssim_metric(colors_p, pixels_p).item()),
                "lpips": float(lpips_metric(colors_p, pixels_p).item()),
                "render_time": float(render_time),
            }
            for key in metrics:
                metrics[key].append(image_stats[key])
            per_image.append(image_stats)

            if args.save_renders and idx < args.max_save_renders:
                canvas = torch.cat([pixels, colors], dim=1).detach().cpu().numpy()
                imageio.imwrite(render_dir / f"{idx:04d}_{Path(sample.name).stem}.png", (canvas * 255).astype(np.uint8))

    summary = {
        "scene": [str(path) for path in ckpt_paths],
        "ckpt": [str(path) for path in ckpt_paths],
        "step": step,
        "num_images": len(eval_set),
        "num_GS": int(splats["means"].shape[0]),
        "psnr": float(np.mean(metrics["psnr"])),
        "ssim": float(np.mean(metrics["ssim"])),
        "lpips": float(np.mean(metrics["lpips"])),
        "avg_render_time": float(np.mean(metrics["render_time"])),
        "fps": float(1.0 / max(np.mean(metrics["render_time"]), 1e-10)),
    }
    ckpt_out = args.output_dir / _safe_name(ckpt_paths[0])
    ckpt_out.mkdir(parents=True, exist_ok=True)
    (ckpt_out / "metrics.json").write_text(json.dumps(summary, indent=2))
    with (ckpt_out / "metrics.jsonl").open("w") as f:
        for row in per_image:
            f.write(json.dumps(row) + "\n")
    print(
        f"{ckpt_paths[0].name}: PSNR {summary['psnr']:.3f}, "
        f"SSIM {summary['ssim']:.4f}, LPIPS {summary['lpips']:.3f}, "
        f"{summary['avg_render_time']:.4f}s/image, GS {summary['num_GS']}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate gsplat checkpoints or Gaussian PLY files on the same fixed COLMAP test set."
    )
    parser.add_argument("--scene", type=Path, nargs="+", default=None, help="Scene .pt/.ckpt checkpoint(s) or Gaussian .ply file(s). Pass multiple rank files together for one distributed model.")
    parser.add_argument("--ckpt", type=Path, nargs="+", default=None, help="Legacy alias for --scene.")
    parser.add_argument("--eval-data-dir", type=Path, default=DEFAULT_EVAL_DATA_DIR)
    parser.add_argument("--train-data-dir", type=Path, default=DEFAULT_TRAIN_DATA_DIR, help="Training COLMAP data dir used to compute the normalization transform.")
    parser.add_argument("--eval-image-dir-name", type=str, default="images", help="Image folder under eval-data-dir, e.g. input or images.")
    parser.add_argument("--train-image-dir-name", type=str, default="images", help="Image folder under train-data-dir, e.g. input or images.")
    parser.add_argument("--data-factor", type=float, default=1.0)
    parser.add_argument("--normalize-world-space", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dataparser-transforms",
        type=Path,
        default=None,
        help=(
            "Optional Nerfstudio dataparser_transforms.json. Use this when evaluating "
            "a Nerfstudio Splatfacto PLY exported in dataparser coordinates."
        ),
    )
    parser.add_argument(
        "--ply-coordinate-mode",
        choices=["supersplat", "native"],
        default="supersplat",
        help=(
            "Coordinate conversion for PLY inputs. Use 'native' for Nerfstudio-exported "
            "Gaussian PLYs; the default preserves the previous SuperSplat-compatible behavior."
        ),
    )
    parser.add_argument("--image-list", type=Path, default=None, help="Optional newline-delimited COLMAP image names to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval_results" / "fixed_gsplat")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--packed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rasterize-mode", choices=["classic", "antialiased"], default="classic")
    parser.add_argument("--camera-model", choices=["pinhole", "ortho", "fisheye"], default="pinhole")
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1e10)
    parser.add_argument("--lpips-net", choices=["alex", "vgg"], default="alex")
    parser.add_argument("--save-renders", action="store_true")
    parser.add_argument("--max-save-renders", type=int, default=32)
    return parser.parse_args()


def _load_nerfstudio_dataparser_transform(path: Path) -> np.ndarray:
    meta = json.loads(path.read_text())
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :4] = np.array(meta["transform"], dtype=np.float32)
    scale = float(meta.get("scale", 1.0))
    similarity = np.eye(4, dtype=np.float32)
    similarity[:3, :3] = transform[:3, :3] * scale
    similarity[:3, 3] = transform[:3, 3] * scale
    return similarity


def main() -> None:
    args = parse_args()
    if args.scene is not None and args.ckpt is not None:
        raise ValueError("Use either --scene or legacy --ckpt, not both.")
    args.scene = args.scene if args.scene is not None else args.ckpt
    if args.scene is None:
        raise ValueError("Pass --scene PATH, or legacy --ckpt PATH.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    image_list = _load_image_list(args.image_list)

    train_transform = None
    if args.dataparser_transforms is not None:
        train_transform = _load_nerfstudio_dataparser_transform(args.dataparser_transforms)
    elif args.normalize_world_space:
        train_set = ColmapFixedSet(
            data_dir=args.train_data_dir,
            image_dir_name=args.train_image_dir_name,
            data_factor=args.data_factor,
            normalize=True,
            apply_transform=None,
            image_list=None,
        )
        train_transform = train_set.transform

    eval_set = ColmapFixedSet(
        data_dir=args.eval_data_dir,
        image_dir_name=args.eval_image_dir_name,
        data_factor=args.data_factor,
        normalize=False,
        apply_transform=train_transform,
        image_list=image_list,
    )

    all_summaries = []
    # One --scene invocation represents one model. For multi-rank checkpoints, pass
    # all rank files together. Run the script again for another stride/checkpoint.
    all_summaries.append(evaluate_checkpoint(args.scene, eval_set, args, device))
    (args.output_dir / "summary.json").write_text(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
