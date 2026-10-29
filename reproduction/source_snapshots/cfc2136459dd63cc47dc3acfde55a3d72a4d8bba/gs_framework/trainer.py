"""
Gaussian Splatting Trainer
Supports two data formats:
  --data-type nerf   : reads transforms_train.json + transforms_test.json (MatrixCity, Blender)
  --data-type colmap : reads sparse/0/ + images/ (Mill-19, UrbanScene3D, any COLMAP capture)

Usage examples:

  # MatrixCity block_A
  python trainer.py \
    --data-type nerf \
    --train-json  data/small_city/aerial/pose/block_A/transforms_train.json \
    --test-json   data/small_city/aerial/pose/block_A/transforms_test.json \
    --train-imgs  data/small_city/aerial/train/block_1 \
    --test-imgs   data/small_city/aerial/test/block_1_test \
    --pointcloud  data/point_clouds/aerial/Block_A.ply \
    --result-dir  results/matrixcity_block_A

  # Mill-19 / UrbanScene3D (COLMAP format)
  python trainer.py \
    --data-type colmap \
    --data-dir  data/building \
    --result-dir results/building
"""

import os
import sys
import json
import time
import math
import struct
import argparse
import collections
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
from tqdm import tqdm

# gsplat
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

try:
    from fused_ssim import fused_ssim as _fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False

def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k nearest neighbour distances for each point. Returns (N, k) distances.

    Uses sklearn KD-tree (O(N log N)) instead of torch.cdist (O(N²) memory).
    """
    from sklearn.neighbors import NearestNeighbors
    x_np = x.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(x_np)
    distances, _ = nbrs.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)

def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [0,1] to SH DC component."""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

# Metrics
try:
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim_fn
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Data
    data_type: str = "nerf"           # "nerf" or "colmap"

    # NeRF-style paths (used when data_type == "nerf")
    train_json: Optional[str] = None
    test_json:  Optional[str] = None
    train_imgs: Optional[str] = None  # folder with train images
    test_imgs:  Optional[str] = None  # folder with test images
    pointcloud: Optional[str] = None  # .ply file for initialization

    # COLMAP-style paths (used when data_type == "colmap")
    data_dir:   Optional[str] = None  # root containing sparse/ and images/
    test_every: int = 8               # every Nth image is test (COLMAP only)

    # Output
    result_dir: str = "results/default"

    # Training
    max_steps:  int = 30000
    eval_steps: list = field(default_factory=lambda: [7000, 30000])
    save_steps: list = field(default_factory=lambda: [7000, 30000])
    batch_size: int = 1

    # Gaussian init
    init_type:    str   = "sfm"       # "sfm" (from pointcloud) or "random"
    init_num_pts: int   = 100_000     # used when init_type == "random"
    init_extent:  float = 3.0        # scene scale multiplier for random init
    max_init_pts: int   = 0     # subsample PLY to this many points (0 = no limit)

    # Gaussian parameters
    sh_degree:          int   = 3
    sh_degree_interval: int   = 1000
    init_opa:           float = 0.1
    init_scale:         float = 1.0

    # Loss
    ssim_lambda: float = 0.2

    # Optimization — learning rates
    means_lr:     float = 1.6e-4
    scales_lr:    float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr:     float = 1e-3
    sh0_lr:       float = 2.5e-3
    shN_lr:       float = 1.25e-4

    # Densification strategy
    strategy_type:       str   = "default"   # "default" or "mcmc"
    refine_start_iter:   int   = 500
    refine_stop_iter:    int   = 25_000      # extended from gsplat default of 15k
    refine_every:        int   = 100
    reset_every:         int   = 3000
    prune_opa:           float = 0.005
    grow_grad2d:         float = 0.0002
    grow_scale3d:        float = 0.01
    prune_scale3d:       float = 0.1

    # Scene
    near_plane:             float = 0.01
    far_plane:              float = 1e10
    packed:                 bool  = False
    normalize_world_space:  bool  = False     # off by default — bad for aerial

    # Misc
    disable_viewer: bool  = True
    save_ply:       bool  = True
    data_factor:    int   = 1         # image downscale factor


# ---------------------------------------------------------------------------
# Pose utilities
# ---------------------------------------------------------------------------

def nerf_c2w_to_colmap(c2w: np.ndarray):
    """
    Convert NeRF/OpenGL camera-to-world matrix to COLMAP world-to-camera.
    NeRF: Y up,  -Z forward
    COLMAP: Y down, +Z forward
    """
    c2w = c2w.copy()
    c2w[:3, 1:3] *= -1          # flip Y and Z
    R = c2w[:3, :3].T           # rotation: invert = transpose for ortho matrix
    t = -R @ c2w[:3, 3]         # translation
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3,  3] = t
    return w2c


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    return np.linalg.inv(w2c)


# ---------------------------------------------------------------------------
# NeRF dataset (transforms.json)
# ---------------------------------------------------------------------------

class NeRFDataset:
    """
    Loads MatrixCity-style datasets with separate transforms_train.json
    and transforms_test.json, and separate train/test image folders.
    """

    def __init__(self, cfg: Config, split: str):
        assert split in ("train", "test")
        self.split = split
        self.cfg   = cfg

        json_path = cfg.train_json if split == "train" else cfg.test_json
        img_dir   = cfg.train_imgs  if split == "train" else cfg.test_imgs

        if json_path is None:
            raise ValueError(f"--{split}-json is required for data-type=nerf")
        if img_dir is None:
            raise ValueError(f"--{split}-imgs is required for data-type=nerf")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON not found: {json_path}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image folder not found: {img_dir}")

        with open(json_path) as f:
            meta = json.load(f)

        self.fl_x = float(meta["fl_x"])
        self.fl_y = float(meta["fl_y"])
        self.cx   = float(meta["cx"])
        self.cy   = float(meta["cy"])
        self.W    = int(meta["w"])
        self.H    = int(meta["h"])

        # Apply data_factor downscale to intrinsics
        factor = cfg.data_factor
        if factor > 1:
            self.fl_x /= factor
            self.fl_y /= factor
            self.cx   /= factor
            self.cy   /= factor
            self.W    = self.W  // factor
            self.H    = self.H  // factor

        frames = meta["frames"]

        self.image_paths = []
        self.c2w_mats    = []

        missing = 0
        for frame in frames:
            fname    = os.path.basename(frame["file_path"])
            img_path = os.path.join(img_dir, fname)
            if not os.path.exists(img_path):
                missing += 1
                continue
            self.image_paths.append(img_path)
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            self.c2w_mats.append(c2w)

        if missing > 0:
            print(f"[NeRFDataset/{split}] Warning: {missing} images not found on disk.")

        self.c2w_mats = np.stack(self.c2w_mats, axis=0).astype(np.float32)
        print(f"[NeRFDataset/{split}] {len(self.image_paths)} images loaded.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.cfg.data_factor > 1:
            img = img.resize((self.W, self.H), Image.BICUBIC)
        img_t = torch.from_numpy(np.array(img)).float() / 255.0  # (H, W, 3)
        c2w   = torch.from_numpy(self.c2w_mats[idx])             # (4, 4)
        return {"image": img_t, "c2w": c2w, "idx": idx}

    def get_intrinsics(self) -> Tensor:
        """Returns (3, 3) K matrix."""
        K = torch.tensor([
            [self.fl_x, 0,         self.cx],
            [0,         self.fl_y, self.cy],
            [0,         0,         1.0    ],
        ], dtype=torch.float32)
        return K


# ---------------------------------------------------------------------------
# COLMAP dataset
# ---------------------------------------------------------------------------

# Minimal COLMAP binary readers
def _read_next_bytes(f, num_bytes, fmt):
    data = f.read(num_bytes)
    return struct.unpack(fmt, data)

def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        n = _read_next_bytes(f, 8, "<Q")[0]
        for _ in range(n):
            cam_id    = _read_next_bytes(f, 4, "<i")[0]
            model_id  = _read_next_bytes(f, 4, "<i")[0]
            width     = _read_next_bytes(f, 8, "<Q")[0]
            height    = _read_next_bytes(f, 8, "<Q")[0]
            # PINHOLE has 4 params; SIMPLE_PINHOLE has 3
            num_params = 4 if model_id == 1 else 3
            params = _read_next_bytes(f, 8 * num_params, f"<{num_params}d")
            cameras[cam_id] = {
                "width": width, "height": height,
                "model_id": model_id, "params": params
            }
    return cameras

def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        n = _read_next_bytes(f, 8, "<Q")[0]
        for _ in range(n):
            img_id    = _read_next_bytes(f, 4,  "<i")[0]
            qvec      = _read_next_bytes(f, 32, "<4d")   # w x y z
            tvec      = _read_next_bytes(f, 24, "<3d")
            cam_id    = _read_next_bytes(f, 4,  "<i")[0]
            name      = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            num_pts = _read_next_bytes(f, 8, "<Q")[0]
            f.read(num_pts * 24)   # skip 2D point data
            images[img_id] = {
                "name": name.decode("utf-8"),
                "qvec": np.array(qvec),
                "tvec": np.array(tvec),
                "camera_id": cam_id,
            }
    return images

def read_points3d_binary(path):
    xyz, rgb = [], []
    with open(path, "rb") as f:
        n = _read_next_bytes(f, 8, "<Q")[0]
        for _ in range(n):
            _read_next_bytes(f, 8, "<Q")          # point3D_id
            pt  = _read_next_bytes(f, 24, "<3d")
            col = _read_next_bytes(f,  3, "<3B")
            _read_next_bytes(f, 8, "<d")           # error
            track_len = _read_next_bytes(f, 8, "<Q")[0]
            f.read(track_len * 8)                  # skip track
            xyz.append(pt)
            rgb.append(col)
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.uint8)

def qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


class ColmapDataset:
    """
    Loads COLMAP-format datasets (Mill-19, UrbanScene3D, any real capture).
    Uses --test-every to split train/test since there is no pre-defined split.
    """

    def __init__(self, cfg: Config, split: str,
                 all_image_paths, all_c2ws, K, W, H):
        assert split in ("train", "test")
        self.split = split
        self.cfg   = cfg
        self.K     = K
        self.W     = W
        self.H     = H

        # Split by index
        indices = list(range(len(all_image_paths)))
        if split == "test":
            self.indices = [i for i in indices if i % cfg.test_every == 0]
        else:
            self.indices = [i for i in indices if i % cfg.test_every != 0]

        self.image_paths = [all_image_paths[i] for i in self.indices]
        self.c2w_mats    = np.stack([all_c2ws[i] for i in self.indices])
        print(f"[ColmapDataset/{split}] {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.cfg.data_factor > 1:
            img = img.resize((self.W, self.H), Image.BICUBIC)
        img_t = torch.from_numpy(np.array(img)).float() / 255.0
        c2w   = torch.from_numpy(self.c2w_mats[idx].astype(np.float32))
        return {"image": img_t, "c2w": c2w, "idx": idx}

    def get_intrinsics(self) -> Tensor:
        return self.K


def load_colmap(cfg: Config):
    """Load COLMAP dataset, return (train_ds, test_ds, points_xyz, points_rgb)."""
    sparse_dir = os.path.join(cfg.data_dir, "sparse", "0")
    if not os.path.exists(sparse_dir):
        raise FileNotFoundError(f"COLMAP sparse dir not found: {sparse_dir}")

    cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images  = read_images_binary (os.path.join(sparse_dir, "images.bin"))
    pts_xyz, pts_rgb = read_points3d_binary(os.path.join(sparse_dir, "points3D.bin"))

    # Image folder
    factor = cfg.data_factor
    img_dir = os.path.join(cfg.data_dir, f"images_{factor}" if factor > 1 else "images")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image folder not found: {img_dir}")

    # Sort images by name for reproducibility
    sorted_ids   = sorted(images.keys(), key=lambda k: images[k]["name"])
    image_paths  = []
    c2w_mats     = []

    for img_id in sorted_ids:
        im   = images[img_id]
        path = os.path.join(img_dir, im["name"])
        if not os.path.exists(path):
            continue
        image_paths.append(path)
        R   = qvec_to_rotmat(im["qvec"])
        t   = im["tvec"]
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3,  3] = t
        c2w_mats.append(np.linalg.inv(w2c))

    # Intrinsics from first camera (assumes single camera)
    cam    = list(cameras.values())[0]
    W, H   = cam["width"], cam["height"]
    params = cam["params"]
    if cam["model_id"] == 1:   # PINHOLE
        fl_x, fl_y, cx, cy = params
    else:                       # SIMPLE_PINHOLE
        fl_x = fl_y = params[0]
        cx, cy = params[1], params[2]

    if factor > 1:
        fl_x /= factor; fl_y /= factor
        cx   /= factor; cy   /= factor
        W    //= factor; H   //= factor

    K = torch.tensor([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0,    0,  1],
    ], dtype=torch.float32)

    train_ds = ColmapDataset(cfg, "train", image_paths, c2w_mats, K, W, H)
    test_ds  = ColmapDataset(cfg, "test",  image_paths, c2w_mats, K, W, H)
    return train_ds, test_ds, pts_xyz, pts_rgb


# ---------------------------------------------------------------------------
# Point cloud loader
# ---------------------------------------------------------------------------

def load_pointcloud(path: str, max_pts: int = 0):
    """Load .ply and return (xyz, rgb) as numpy arrays.

    If max_pts > 0 and the cloud is larger, uniformly subsample to max_pts.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d required for .ply loading: pip install open3d")
    pcd  = o3d.io.read_point_cloud(path)
    xyz  = np.asarray(pcd.points, dtype=np.float32)
    if pcd.has_colors():
        rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
    print(f"[PointCloud] {len(xyz):,} points loaded from {path}")
    if max_pts > 0 and len(xyz) > max_pts:
        idx = np.random.choice(len(xyz), max_pts, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
        print(f"[PointCloud] subsampled to {max_pts:,} points")
    return xyz, rgb


# ---------------------------------------------------------------------------
# Scene normalisation (optional)
# ---------------------------------------------------------------------------

def compute_scene_scale(c2w_mats: np.ndarray) -> float:
    """Estimate scene scale from camera positions."""
    positions = c2w_mats[:, :3, 3]
    dists     = np.linalg.norm(positions - positions.mean(0), axis=-1)
    return float(np.percentile(dists, 90))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(pred: Tensor, gt: Tensor) -> float:
    mse = F.mse_loss(pred, gt).item()
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)

def ssim(pred: Tensor, gt: Tensor) -> float:
    if not SKIMAGE_AVAILABLE:
        return 0.0
    p = pred.cpu().numpy()
    g = gt.cpu().numpy()
    score = ssim_fn(p, g, data_range=1.0, channel_axis=-1)
    return float(score)

lpips_model = None
def lpips(pred: Tensor, gt: Tensor) -> float:
    global lpips_model
    if not LPIPS_AVAILABLE:
        return 0.0
    if lpips_model is None:
        lpips_model = lpips_lib.LPIPS(net="alex").cuda()
    # expects (1, 3, H, W) in [-1, 1]
    p = pred.permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
    g = gt.permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
    with torch.no_grad():
        score = lpips_model(p, g).item()
    return score


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.device = torch.device("cuda")
        os.makedirs(cfg.result_dir, exist_ok=True)

        # ---- Load data ----
        if cfg.data_type == "nerf":
            self.train_ds = NeRFDataset(cfg, "train")
            self.test_ds  = NeRFDataset(cfg, "test")
            # Load point cloud for initialization
            if cfg.pointcloud and os.path.exists(cfg.pointcloud):
                pts_xyz, pts_rgb = load_pointcloud(cfg.pointcloud, max_pts=cfg.max_init_pts)
            else:
                pts_xyz, pts_rgb = None, None

        elif cfg.data_type == "colmap":
            self.train_ds, self.test_ds, pts_xyz, pts_rgb = load_colmap(cfg)

        else:
            raise ValueError(f"Unknown data-type: {cfg.data_type}")

        # ---- Scene scale ----
        self.scene_scale = compute_scene_scale(self.train_ds.c2w_mats)
        print(f"[Trainer] Scene scale: {self.scene_scale:.4f}")

        # ---- Initialize Gaussians ----
        if cfg.init_type == "sfm" and pts_xyz is not None:
            points = torch.from_numpy(pts_xyz).float()
            colors = torch.from_numpy(pts_rgb).float() / 255.0
            print(f"[Trainer] Initialized from point cloud: {len(points):,} points")
        else:
            print(f"[Trainer] Random initialization: {cfg.init_num_pts:,} points")
            points = cfg.init_extent * self.scene_scale * (
                torch.rand((cfg.init_num_pts, 3)) * 2 - 1
            )
            colors = torch.rand((cfg.init_num_pts, 3))

        N = len(points)

        # Scale init: average dist to 3 nearest neighbours, matching simple_trainer exactly.
        dist_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1).sqrt()  # (N,)
        scales_init = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)  # (N, 3)

        # SH colour init from point cloud colours
        num_sh_bases = (cfg.sh_degree + 1) ** 2
        sh_all = torch.zeros(N, num_sh_bases, 3)
        sh_all[:, :1, :] = rgb_to_sh(colors).unsqueeze(1)  # DC component

        # All Gaussian parameters in a ParameterDict so the strategy can update
        # them in-place during densification and all accesses see the new tensors
        self.splats = torch.nn.ParameterDict({
            "means":     torch.nn.Parameter(points),
            "scales":    torch.nn.Parameter(scales_init),
            "quats":     torch.nn.Parameter(torch.rand(N, 4)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), cfg.init_opa))),
            "sh0":       torch.nn.Parameter(sh_all[:, :1, :]),
            "shN":       torch.nn.Parameter(sh_all[:, 1:, :]),
        }).to(self.device)

        # ---- Optimisers — one per parameter, keyed by name ----
        # fused=True uses a faster CUDA Adam kernel
        # eps=1e-15 is important for GS training stability (same as simple_trainer)
        self.optimisers = {
            name: torch.optim.Adam(
                [{"params": self.splats[name], "lr": lr, "name": name}],
                eps=1e-15,
                fused=True,
            )
            for name, lr in [
                ("means",     cfg.means_lr * self.scene_scale),
                ("scales",    cfg.scales_lr),
                ("quats",     cfg.quats_lr),
                ("opacities", cfg.opacities_lr),
                ("sh0",       cfg.sh0_lr),
                ("shN",       cfg.shN_lr),
            ]
        }

        # ---- Densification strategy ----
        strategy_cfg = dict(
            prune_opa       = cfg.prune_opa,
            grow_grad2d     = cfg.grow_grad2d,
            grow_scale3d    = cfg.grow_scale3d,
            prune_scale3d   = cfg.prune_scale3d,
            refine_start_iter = cfg.refine_start_iter,
            refine_stop_iter  = cfg.refine_stop_iter,
            refine_every      = cfg.refine_every,
            reset_every       = cfg.reset_every,
            verbose           = True,
        )
        if cfg.strategy_type == "mcmc":
            self.strategy = MCMCStrategy(**strategy_cfg)
        else:
            self.strategy = DefaultStrategy(**strategy_cfg)

        self.strategy_state = self.strategy.initialize_state()

        # ---- LPIPS for eval ----
        self._sh_degree_current = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_sh_degree(self, step: int) -> int:
        return min(
            self.cfg.sh_degree,
            step // self.cfg.sh_degree_interval
        )

    def _get_gaussians(self):
        """Return activated Gaussian parameters from the ParameterDict."""
        return dict(
            means     = self.splats["means"],
            scales    = torch.exp(self.splats["scales"]),
            quats     = self.splats["quats"],  # gsplat normalizes internally
            opacities = torch.sigmoid(self.splats["opacities"]),
            sh0       = self.splats["sh0"],
            shN       = self.splats["shN"],
        )

    def _render(self, c2w: Tensor, K: Tensor, W: int, H: int,
                sh_degree: int, training: bool = False):
        """
        Render a single view.
        Returns (image, info):
          image: (H, W, 3) float32 in [0, 1]
          info:  dict from gsplat rasterizer (contains means2d for densification)
        c2w: (4, 4) camera-to-world
        K:   (3, 3) intrinsics
        """
        gs = self._get_gaussians()
        sh = torch.cat([gs["sh0"], gs["shN"]], dim=1)  # (N, total_sh, 3)

        # Build world-to-camera viewmat from NeRF c2w (torch-native, stays on GPU)
        # Flip Y and Z axes (OpenGL -> OpenCV convention)
        c2w_conv = c2w.clone()
        c2w_conv[:3, 1:3] *= -1
        # Invert: for rotation R, inv = R^T; for translation t, inv_t = -R^T @ t
        R = c2w_conv[:3, :3]
        t = c2w_conv[:3,  3]
        w2c = torch.eye(4, device=self.device, dtype=torch.float32)
        w2c[:3, :3] = R.T
        w2c[:3,  3] = -(R.T @ t)
        viewmat = w2c.unsqueeze(0)  # (1, 4, 4)

        renders, alphas, info = rasterization(
            means      = gs["means"],
            quats      = gs["quats"],
            scales     = gs["scales"],
            opacities  = gs["opacities"],
            colors     = sh,
            viewmats   = viewmat,
            Ks         = K.to(self.device).unsqueeze(0),
            width      = W,
            height     = H,
            sh_degree  = sh_degree,
            packed     = self.cfg.packed,
            near_plane = self.cfg.near_plane,
            far_plane  = self.cfg.far_plane,
            render_mode = "RGB",
            absgrad    = getattr(self.strategy, "absgrad", False),
        )

        # means2d must retain grad so the strategy can compute screen-space gradients
        if training and "means2d" in info:
            info["means2d"].retain_grad()

        return renders[0].clamp(0, 1), info

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg    = self.cfg
        device = self.device
        N_train = len(self.train_ds)
        K       = self.train_ds.get_intrinsics().to(device)

        # Resolve image size (after data_factor)
        sample = self.train_ds[0]
        H, W   = sample["image"].shape[:2]

        # DataLoader with background workers — images prefetched while GPU trains
        trainloader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size  = 1,
            shuffle     = True,
            num_workers = 4,
            persistent_workers = True,
            pin_memory  = True,
        )
        trainloader_iter = iter(trainloader)

        pbar = tqdm(range(cfg.max_steps), desc="Training")
        t_start = time.time()

        for step in pbar:
            sh_deg = self._current_sh_degree(step)

            # Get next batch — workers prefetch in background
            try:
                batch = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                batch = next(trainloader_iter)

            gt  = batch["image"][0].to(device, non_blocking=True)  # (H, W, 3)
            c2w = batch["c2w"][0].to(device,   non_blocking=True)  # (4, 4)

            # Render — training=True so means2d retains grad for densification
            render, info = self._render(c2w, K, W, H, sh_deg, training=True)

            # Loss: L1 + SSIM
            l1_loss = F.l1_loss(render, gt)
            if FUSED_SSIM_AVAILABLE:
                ssim_val = _fused_ssim(
                    render.permute(2, 0, 1).unsqueeze(0),
                    gt.permute(2, 0, 1).unsqueeze(0),
                    padding="valid",
                )
            else:
                ssim_val = self._ssim_loss(render, gt)
            ssim_loss = 1.0 - ssim_val
            loss      = (1.0 - cfg.ssim_lambda) * l1_loss + cfg.ssim_lambda * ssim_loss

            for opt in self.optimisers.values(): opt.zero_grad()

            # Pre-backward: strategy registers hooks on means2d before backward
            self.strategy.step_pre_backward(
                params     = self._param_dict(),
                optimizers = self.optimisers,
                state      = self.strategy_state,
                step       = step,
                info       = info,
            )

            loss.backward()
            for opt in self.optimisers.values(): opt.step()

            # Post-backward: strategy uses accumulated screen-space gradients
            self.strategy.step_post_backward(
                params     = self._param_dict(),
                optimizers = self.optimisers,
                state      = self.strategy_state,
                step       = step,
                info       = info,
                packed     = cfg.packed,
            )

            # Logging
            pbar.set_postfix({
                "loss":  f"{loss.item():.3f}",
                "sh":    sh_deg,
                "N":     len(self.splats["means"]),
                "vram":  f"{torch.cuda.max_memory_allocated()/1024**3:.1f}GB",
            })

            # Eval
            if step + 1 in cfg.eval_steps:
                self.evaluate(step + 1)

            # Save
            if step + 1 in cfg.save_steps and cfg.save_ply:
                self.save_ply(step + 1)

        total_time = time.time() - t_start
        print(f"\nTraining done in {total_time/60:.1f} min")
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        self.evaluate(cfg.max_steps)
        if cfg.save_ply:
            self.save_ply(cfg.max_steps)

    def _param_dict(self):
        # Return the ParameterDict directly — strategy updates tensors in-place here
        # so subsequent accesses via self.splats["means"] etc. see the updated tensors
        return self.splats

    @staticmethod
    def _ssim_loss(pred: Tensor, gt: Tensor, window_size: int = 11) -> Tensor:
        """Fast differentiable SSIM approximation."""
        C1, C2 = 0.01**2, 0.03**2
        pred_p = pred.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        gt_p   = gt.permute(2, 0, 1).unsqueeze(0)
        pad    = window_size // 2
        kernel = torch.ones(3, 1, window_size, window_size,
                            device=pred.device) / (window_size ** 2)
        mu1  = F.conv2d(pred_p, kernel, padding=pad, groups=3)
        mu2  = F.conv2d(gt_p,   kernel, padding=pad, groups=3)
        mu1_sq = mu1 ** 2; mu2_sq = mu2 ** 2; mu1_mu2 = mu1 * mu2
        s1  = F.conv2d(pred_p * pred_p, kernel, padding=pad, groups=3) - mu1_sq
        s2  = F.conv2d(gt_p   * gt_p,   kernel, padding=pad, groups=3) - mu2_sq
        s12 = F.conv2d(pred_p * gt_p,   kernel, padding=pad, groups=3) - mu1_mu2
        ssim_map = ((2*mu1_mu2 + C1) * (2*s12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
        return ssim_map.mean()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, step: int):
        cfg    = self.cfg
        device = self.device
        K      = self.test_ds.get_intrinsics().to(device)
        sample0 = self.test_ds[0]
        H, W    = sample0["image"].shape[:2]

        sh_deg  = cfg.sh_degree   # use full SH for eval

        psnr_sum = ssim_sum = lpips_sum = 0.0
        render_times = []

        renders_dir = os.path.join(cfg.result_dir, "renders", f"step_{step:06d}")
        os.makedirs(renders_dir, exist_ok=True)

        for i in range(len(self.test_ds)):
            sample = self.test_ds[i]
            gt     = sample["image"].to(device)
            c2w    = sample["c2w"].to(device)

            t0     = time.time()
            render, _ = self._render(c2w, K, W, H, sh_deg, training=False)
            render_times.append(time.time() - t0)

            psnr_sum  += psnr(render, gt)
            ssim_sum  += ssim(render, gt)
            lpips_sum += lpips(render, gt)

            # Save side-by-side comparison
            if i < 20:
                comp = torch.cat([gt, render], dim=1)  # (H, 2W, 3)
                img  = Image.fromarray(
                    (comp.cpu().numpy() * 255).astype(np.uint8)
                )
                img.save(os.path.join(renders_dir, f"{i:04d}.png"))

        n = len(self.test_ds)
        mean_psnr  = psnr_sum  / n
        mean_ssim  = ssim_sum  / n
        mean_lpips = lpips_sum / n
        mean_time  = sum(render_times) / len(render_times)

        print(f"\n[Eval step {step}]")
        print(f"  PSNR:  {mean_psnr:.3f}")
        print(f"  SSIM:  {mean_ssim:.4f}")
        print(f"  LPIPS: {mean_lpips:.4f}")
        print(f"  Render time: {mean_time*1000:.1f} ms/image")
        print(f"  Num Gaussians: {len(self.splats["means"]):,}")
        print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

        # Save stats
        stats = {
            "step": step, "psnr": mean_psnr, "ssim": mean_ssim,
            "lpips": mean_lpips, "num_gaussians": len(self.splats["means"]),
            "render_ms": mean_time * 1000,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        stats_path = os.path.join(cfg.result_dir, "stats.jsonl")
        with open(stats_path, "a") as f:
            f.write(json.dumps(stats) + "\n")

    # ------------------------------------------------------------------
    # Save PLY
    # ------------------------------------------------------------------

    def save_ply(self, step: int):
        path = os.path.join(self.cfg.result_dir, f"gaussians_{step:06d}.ply")
        gs   = self._get_gaussians()

        means     = gs["means"].detach().cpu().numpy()
        scales    = gs["scales"].detach().cpu().numpy()
        quats     = gs["quats"].detach().cpu().numpy()
        opacities = gs["opacities"].detach().cpu().numpy()
        sh0       = gs["sh0"].detach().cpu().numpy().reshape(len(means), -1)
        shN       = gs["shN"].detach().cpu().numpy().reshape(len(means), -1)

        # Build PLY header
        attrs = (
            [("x","f4"),("y","f4"),("z","f4")] +
            [(f"scale_{i}","f4") for i in range(3)] +
            [(f"rot_{i}","f4") for i in range(4)] +
            [("opacity","f4")] +
            [(f"f_dc_{i}","f4") for i in range(sh0.shape[1])] +
            [(f"f_rest_{i}","f4") for i in range(shN.shape[1])]
        )
        dtype = np.dtype(attrs)
        arr   = np.zeros(len(means), dtype=dtype)
        arr["x"] = means[:,0]; arr["y"] = means[:,1]; arr["z"] = means[:,2]
        for i in range(3): arr[f"scale_{i}"] = np.log(scales[:,i] + 1e-8)
        for i in range(4): arr[f"rot_{i}"]   = quats[:,i]
        arr["opacity"] = np.log(opacities / (1 - opacities + 1e-8))
        for i in range(sh0.shape[1]): arr[f"f_dc_{i}"]   = sh0[:,i]
        for i in range(shN.shape[1]): arr[f"f_rest_{i}"] = shN[:,i]

        with open(path, "wb") as f:
            f.write(f"ply\nformat binary_little_endian 1.0\n".encode())
            f.write(f"element vertex {len(means)}\n".encode())
            for name, dt in attrs:
                f.write(f"property {dt} {name}\n".encode())
            f.write(b"end_header\n")
            f.write(arr.tobytes())

        print(f"[Save] PLY saved to {path} ({len(means):,} Gaussians)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Gaussian Splatting Trainer")

    # Data type
    p.add_argument("--data-type",   default="nerf",   choices=["nerf","colmap"])

    # NeRF paths
    p.add_argument("--train-json",  default=None)
    p.add_argument("--test-json",   default=None)
    p.add_argument("--train-imgs",  default=None)
    p.add_argument("--test-imgs",   default=None)
    p.add_argument("--pointcloud",    default=None)
    p.add_argument("--max-init-pts",  type=int, default=100_000,
                   help="subsample PLY to this many points before KNN init (0 = no limit)")

    # COLMAP path
    p.add_argument("--data-dir",    default=None)
    p.add_argument("--test-every",  type=int, default=8)

    # Output
    p.add_argument("--result-dir",  default="results/default")

    # Training
    p.add_argument("--max-steps",   type=int,   default=30000)
    p.add_argument("--data-factor", type=int,   default=1)
    p.add_argument("--save-ply",    action="store_true", default=True)

    # Strategy
    p.add_argument("--strategy-type",     default="default", choices=["default","mcmc"])
    p.add_argument("--refine-stop-iter",  type=int, default=25000)

    # Scene
    p.add_argument("--no-normalize-world-space", action="store_true", default=True)

    args = p.parse_args()

    cfg = Config(
        data_type            = args.data_type,
        train_json           = args.train_json,
        test_json            = args.test_json,
        train_imgs           = args.train_imgs,
        test_imgs            = args.test_imgs,
        pointcloud           = args.pointcloud,
        max_init_pts         = args.max_init_pts,
        data_dir             = args.data_dir,
        test_every           = args.test_every,
        result_dir           = args.result_dir,
        max_steps            = args.max_steps,
        data_factor          = args.data_factor,
        save_ply             = args.save_ply,
        strategy_type        = args.strategy_type,
        refine_stop_iter     = args.refine_stop_iter,
        normalize_world_space = not args.no_normalize_world_space,
    )
    return cfg


if __name__ == "__main__":
    cfg     = parse_args()
    trainer = Trainer(cfg)
    trainer.train()