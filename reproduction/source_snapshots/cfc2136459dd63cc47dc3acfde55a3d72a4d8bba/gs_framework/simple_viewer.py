import argparse
import csv
import os
import threading
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import viser
from pathlib import Path
from gsplat.distributed import cli

from nerfview import CameraState, RenderTabState, apply_float_colormap
from nerfview._renderer import Renderer, RenderTask
from viser import _messages as viser_messages

from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from scene_runtime import FrameCamera, GsplatRenderer, RenderSettings, SceneLoader
from scene_runtime.types import BlockDebugInfo, SceneStats


BlockCullingMode = Literal[
    "normal_frustum_culling",
    "force_all_blocks_active",
    "force_no_blocks_active",
    "selected_block_only",
]


class RuntimeMetricsLogger:
    CSV_FIELDS = [
        "timestamp",
        "elapsed_sec",
        "pid",
        "process_cpu_percent",
        "process_ram_gb",
        "system_ram_used_gb",
        "system_ram_percent",
        "gpu_util_percent",
        "vram_used_gb",
        "vram_total_gb",
        "render_frame_index",
        "render_frame_time_ms",
        "render_fps",
        "render_width",
        "render_height",
        "active_blocks",
        "active_block_ids",
        "active_gaussians",
        "active_gaussian_fraction",
        "visible_gaussians",
        "total_gaussians",
        "total_blocks",
        "resident_blocks",
        "resident_gaussians",
        "resident_gaussian_fraction",
        "block_cull_ms",
        "block_concat_ms",
        "gpu_upload_ms",
        "cache_hits",
        "cache_misses",
        "cache_hit_rate",
        "newly_uploaded_blocks",
        "newly_uploaded_gaussians",
        "evicted_blocks",
        "evicted_gaussians",
        "gpu_resident_bytes",
        "gpu_resident_mb",
        "cpu_resident_bytes",
        "cpu_resident_mb",
        "peak_vram",
        "peak_vram_mb",
        "torch_allocated_mb",
        "torch_reserved_mb",
        "torch_frame_peak_allocated_mb",
        "torch_frame_peak_reserved_mb",
        "gpu_budget_exceeded",
        "active_set_exceeds_budget",
    ]

    def __init__(self, path: str, interval: float, gpu_index: int, render_metrics):
        self.path = Path(path)
        self.interval = interval
        self.gpu_index = gpu_index
        self.render_metrics = render_metrics
        self.pid = os.getpid()
        self.start_time = time.time()
        self.stop_event = threading.Event()
        self.thread = None
        self.file = None
        self.writer = None
        self.psutil = None
        self.process = None
        self.pynvml = None
        self.gpu_handle = None

    def start(self):
        try:
            import psutil
        except ImportError:
            print(
                "Warning: psutil is not installed; runtime metrics logging is disabled."
            )
            return

        self.psutil = psutil
        self.process = psutil.Process(self.pid)
        self.process.cpu_percent(interval=None)
        self._init_nvml()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.CSV_FIELDS)
        self.writer.writeheader()
        self.file.flush()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"Runtime metrics logging started: {self.path}")

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.pynvml is not None:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass
        if self.file is not None:
            self.file.close()

    def _init_nvml(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self.pynvml = pynvml
        except Exception as exc:
            print(
                "Warning: GPU metrics are disabled because pynvml/NVML could not "
                f"be initialized: {exc}"
            )
            self.pynvml = None
            self.gpu_handle = None

    def _run(self):
        while not self.stop_event.is_set():
            self.writer.writerow(self._sample())
            self.file.flush()
            self.stop_event.wait(self.interval)

    def _sample(self):
        now = time.time()
        process_mem = self.process.memory_info()
        system_mem = self.psutil.virtual_memory()
        gpu_util_percent = float("nan")
        vram_used_gb = float("nan")
        vram_total_gb = float("nan")

        if self.pynvml is not None and self.gpu_handle is not None:
            try:
                util = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_util_percent = util.gpu
                vram_used_gb = mem.used / (1024**3)
                vram_total_gb = mem.total / (1024**3)
            except Exception as exc:
                print(
                    "Warning: GPU metrics sampling failed; disabling GPU metrics: "
                    f"{exc}"
                )
                self.pynvml = None
                self.gpu_handle = None

        sample = {
            "timestamp": now,
            "elapsed_sec": now - self.start_time,
            "pid": self.pid,
            "process_cpu_percent": self.process.cpu_percent(interval=None),
            "process_ram_gb": process_mem.rss / (1024**3),
            "system_ram_used_gb": system_mem.used / (1024**3),
            "system_ram_percent": system_mem.percent,
            "gpu_util_percent": gpu_util_percent,
            "vram_used_gb": vram_used_gb,
            "vram_total_gb": vram_total_gb,
        }
        sample.update(self.render_metrics.snapshot())
        return sample


class RenderMetricsState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_index = 0
        self.frame_time_ms = float("nan")
        self.fps = float("nan")
        self.width = 0
        self.height = 0
        self.active_blocks = float("nan")
        self.active_block_ids = ""
        self.active_gaussians = float("nan")
        self.active_gaussian_fraction = float("nan")
        self.visible_gaussians = float("nan")
        self.total_gaussians = float("nan")
        self.total_blocks = float("nan")
        self.resident_blocks = float("nan")
        self.resident_gaussians = float("nan")
        self.resident_gaussian_fraction = float("nan")
        self.block_cull_ms = float("nan")
        self.block_concat_ms = float("nan")
        self.gpu_upload_ms = float("nan")
        self.cache_hits = float("nan")
        self.cache_misses = float("nan")
        self.cache_hit_rate = float("nan")
        self.newly_uploaded_blocks = float("nan")
        self.newly_uploaded_gaussians = float("nan")
        self.evicted_blocks = float("nan")
        self.evicted_gaussians = float("nan")
        self.gpu_resident_bytes = float("nan")
        self.gpu_resident_mb = float("nan")
        self.cpu_resident_bytes = float("nan")
        self.cpu_resident_mb = float("nan")
        self.peak_vram = float("nan")
        self.peak_vram_mb = float("nan")
        self.torch_allocated_mb = float("nan")
        self.torch_reserved_mb = float("nan")
        self.torch_frame_peak_allocated_mb = float("nan")
        self.torch_frame_peak_reserved_mb = float("nan")
        self.gpu_budget_exceeded = float("nan")
        self.active_set_exceeds_budget = float("nan")

    def record_frame(
        self,
        frame_time_ms: float,
        width: int,
        height: int,
        active_blocks: int,
        active_gaussians: int,
        visible_gaussians: int,
        total_gaussians: int,
        active_block_ids: str = "",
        active_gaussian_fraction: float = float("nan"),
        total_blocks: float = float("nan"),
        resident_blocks: float = float("nan"),
        resident_gaussians: float = float("nan"),
        resident_gaussian_fraction: float = float("nan"),
        block_cull_ms: float = float("nan"),
        block_concat_ms: float = float("nan"),
        gpu_upload_ms: float = float("nan"),
        cache_hits: float = float("nan"),
        cache_misses: float = float("nan"),
        cache_hit_rate: float = float("nan"),
        newly_uploaded_blocks: float = float("nan"),
        newly_uploaded_gaussians: float = float("nan"),
        evicted_blocks: float = float("nan"),
        evicted_gaussians: float = float("nan"),
        gpu_resident_bytes: float = float("nan"),
        gpu_resident_mb: float = float("nan"),
        cpu_resident_bytes: float = float("nan"),
        cpu_resident_mb: float = float("nan"),
        peak_vram: float = float("nan"),
        peak_vram_mb: float = float("nan"),
        torch_allocated_mb: float = float("nan"),
        torch_reserved_mb: float = float("nan"),
        torch_frame_peak_allocated_mb: float = float("nan"),
        torch_frame_peak_reserved_mb: float = float("nan"),
        gpu_budget_exceeded: float = float("nan"),
        active_set_exceeds_budget: float = float("nan"),
    ):
        with self.lock:
            self.frame_index += 1
            self.frame_time_ms = frame_time_ms
            self.fps = 1000.0 / max(frame_time_ms, 1e-10)
            self.width = width
            self.height = height
            self.active_blocks = active_blocks
            self.active_block_ids = active_block_ids
            self.active_gaussians = active_gaussians
            self.active_gaussian_fraction = active_gaussian_fraction
            self.visible_gaussians = visible_gaussians
            self.total_gaussians = total_gaussians
            self.total_blocks = total_blocks
            self.resident_blocks = resident_blocks
            self.resident_gaussians = resident_gaussians
            self.resident_gaussian_fraction = resident_gaussian_fraction
            self.block_cull_ms = block_cull_ms
            self.block_concat_ms = block_concat_ms
            self.gpu_upload_ms = gpu_upload_ms
            self.cache_hits = cache_hits
            self.cache_misses = cache_misses
            self.cache_hit_rate = cache_hit_rate
            self.newly_uploaded_blocks = newly_uploaded_blocks
            self.newly_uploaded_gaussians = newly_uploaded_gaussians
            self.evicted_blocks = evicted_blocks
            self.evicted_gaussians = evicted_gaussians
            self.gpu_resident_bytes = gpu_resident_bytes
            self.gpu_resident_mb = gpu_resident_mb
            self.cpu_resident_bytes = cpu_resident_bytes
            self.cpu_resident_mb = cpu_resident_mb
            self.peak_vram = peak_vram
            self.peak_vram_mb = peak_vram_mb
            self.torch_allocated_mb = torch_allocated_mb
            self.torch_reserved_mb = torch_reserved_mb
            self.torch_frame_peak_allocated_mb = torch_frame_peak_allocated_mb
            self.torch_frame_peak_reserved_mb = torch_frame_peak_reserved_mb
            self.gpu_budget_exceeded = gpu_budget_exceeded
            self.active_set_exceeds_budget = active_set_exceeds_budget

    def snapshot(self):
        with self.lock:
            return {
                "render_frame_index": self.frame_index,
                "render_frame_time_ms": self.frame_time_ms,
                "render_fps": self.fps,
                "render_width": self.width,
                "render_height": self.height,
                "active_blocks": self.active_blocks,
                "active_block_ids": self.active_block_ids,
                "active_gaussians": self.active_gaussians,
                "active_gaussian_fraction": self.active_gaussian_fraction,
                "visible_gaussians": self.visible_gaussians,
                "total_gaussians": self.total_gaussians,
                "total_blocks": self.total_blocks,
                "resident_blocks": self.resident_blocks,
                "resident_gaussians": self.resident_gaussians,
                "resident_gaussian_fraction": self.resident_gaussian_fraction,
                "block_cull_ms": self.block_cull_ms,
                "block_concat_ms": self.block_concat_ms,
                "gpu_upload_ms": self.gpu_upload_ms,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hit_rate,
                "newly_uploaded_blocks": self.newly_uploaded_blocks,
                "newly_uploaded_gaussians": self.newly_uploaded_gaussians,
                "evicted_blocks": self.evicted_blocks,
                "evicted_gaussians": self.evicted_gaussians,
                "gpu_resident_bytes": self.gpu_resident_bytes,
                "gpu_resident_mb": self.gpu_resident_mb,
                "cpu_resident_bytes": self.cpu_resident_bytes,
                "cpu_resident_mb": self.cpu_resident_mb,
                "peak_vram": self.peak_vram,
                "peak_vram_mb": self.peak_vram_mb,
                "torch_allocated_mb": self.torch_allocated_mb,
                "torch_reserved_mb": self.torch_reserved_mb,
                "torch_frame_peak_allocated_mb": self.torch_frame_peak_allocated_mb,
                "torch_frame_peak_reserved_mb": self.torch_frame_peak_reserved_mb,
                "gpu_budget_exceeded": self.gpu_budget_exceeded,
                "active_set_exceeds_budget": self.active_set_exceeds_budget,
            }


@dataclass
class BlockDebugState:
    show_block_bounds: bool = False
    block_debug_aabb: str = "cull"
    show_camera_frustum: bool = False
    show_block_labels: bool = False
    culling_mode: BlockCullingMode = "normal_frustum_culling"
    selected_block_id: int = -1
    frozen_culling_camera_state: Optional[CameraState] = None

    @property
    def culling_camera_frozen(self) -> bool:
        return self.frozen_culling_camera_state is not None


def _box_mesh(aabb_min, aabb_max):
    mn = np.array(aabb_min, dtype=np.float32)
    mx = np.array(aabb_max, dtype=np.float32)
    vertices = np.array(
        [
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]],
            [mn[0], mx[1], mx[2]],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _frustum_mesh(c2w, K, width, height, near_plane, far_plane):
    c2w = np.asarray(c2w, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    fx = max(float(K[0, 0]), 1e-6)
    fy = max(float(K[1, 1]), 1e-6)
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    def plane_corners(z):
        return np.array(
            [
                [(-cx / fx) * z, (-cy / fy) * z, z],
                [((float(width) - cx) / fx) * z, (-cy / fy) * z, z],
                [((float(width) - cx) / fx) * z, ((float(height) - cy) / fy) * z, z],
                [(-cx / fx) * z, ((float(height) - cy) / fy) * z, z],
            ],
            dtype=np.float32,
        )

    camera_vertices = np.concatenate(
        [plane_corners(float(near_plane)), plane_corners(float(far_plane))], axis=0
    )
    vertices_h = np.concatenate(
        [camera_vertices, np.ones((camera_vertices.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    vertices = (vertices_h @ c2w.T)[:, :3].astype(np.float32)
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _torch_cuda_memory_snapshot(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "allocated": float("nan"),
            "reserved": float("nan"),
            "peak_allocated": float("nan"),
            "peak_reserved": float("nan"),
        }

    peak_reserved = float("nan")
    if hasattr(torch.cuda, "max_memory_reserved"):
        peak_reserved = float(torch.cuda.max_memory_reserved(device))
    return {
        "allocated": float(torch.cuda.memory_allocated(device)),
        "reserved": float(torch.cuda.memory_reserved(device)),
        "peak_allocated": float(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": peak_reserved,
    }


def _scene_debug_extent(scene) -> float:
    mins = []
    maxs = []
    for block in scene.get_debug_geometry():
        mins.append(np.asarray(block.cull_aabb_min, dtype=np.float32))
        maxs.append(np.asarray(block.cull_aabb_max, dtype=np.float32))
    if not mins:
        return 1.0
    scene_min = np.stack(mins, axis=0).min(axis=0)
    scene_max = np.stack(maxs, axis=0).max(axis=0)
    return float(max(np.linalg.norm(scene_max - scene_min), 1.0))


def _observer_pose_from_render_camera(camera_state: CameraState, scene_extent: float):
    c2w = np.asarray(camera_state.c2w, dtype=np.float32)
    position = c2w[:3, 3]
    right = c2w[:3, 0]
    up = -c2w[:3, 1]
    forward = c2w[:3, 2]
    scale = max(float(scene_extent) * 0.12, 1.0)
    target = position + forward * scale * 0.65
    observer = position - forward * scale * 1.25 + up * scale * 0.55 + right * scale * 0.35
    return observer.astype(np.float64), target.astype(np.float64), up.astype(np.float64)


class BlockDebugVisualizer:
    def __init__(self, server: viser.ViserServer, scene, state: BlockDebugState):
        self.server = server
        self.scene = scene
        self.state = state
        self.block_handles = {}
        self.block_visual_keys = {}
        self.label_handles = {}
        self.label_visual_keys = {}
        self.frustum_handle = None
        self.frustum_key = None
        self.latest_active_ids = set()
        self.latest_debug_blocks = scene.get_debug_geometry()
        self.latest_resident_ids = {
            block.block_id for block in self.latest_debug_blocks if block.resident
        }
        self.scene_extent = _scene_debug_extent(scene)
        self.selected_info_handle = None
        self.warning_handle = None

    def _block_aabb(self, block: BlockDebugInfo):
        if self.state.block_debug_aabb == "logical" and block.logical_bounds_min is not None:
            return block.logical_bounds_min, block.logical_bounds_max
        if self.state.block_debug_aabb == "raw" and block.raw_gaussian_bounds_min is not None:
            return block.raw_gaussian_bounds_min, block.raw_gaussian_bounds_max
        if self.state.block_debug_aabb == "filtered" and block.filtered_render_bounds_min is not None:
            return block.filtered_render_bounds_min, block.filtered_render_bounds_max
        if self.state.block_debug_aabb == "content":
            return block.content_aabb_min, block.content_aabb_max
        return block.cull_aabb_min, block.cull_aabb_max

    def _block_color(self, block: BlockDebugInfo):
        if block.empty:
            return (255, 170, 40)
        if block.active:
            return (55, 235, 95)
        if block.resident and getattr(self.scene, "residency", "gpu") == "cpu":
            return (75, 155, 255)
        return (130, 130, 130)

    def _label_text(self, block: BlockDebugInfo):
        active = "active" if block.active else "inactive"
        resident = "resident" if block.resident else "non-resident"
        raw_count = block.raw_gaussian_count if block.raw_gaussian_count is not None else block.gaussian_count
        valid_count = block.bounds_valid_gaussian_count
        count_text = f"{raw_count:,} raw gs"
        if valid_count is not None:
            count_text += f"\n{valid_count:,} bounds-valid"
        return f"block {block.block_id}\n{count_text}\n{active}, {resident}"

    def _upsert_block_mesh(self, block: BlockDebugInfo):
        aabb_min, aabb_max = self._block_aabb(block)
        color = self._block_color(block)
        key = (
            self.state.show_block_bounds,
            self.state.block_debug_aabb,
            color,
            tuple(round(v, 5) for v in aabb_min),
            tuple(round(v, 5) for v in aabb_max),
        )
        name = f"/block_debug/bounds/{block.block_id}"
        handle = self.block_handles.get(block.block_id)
        if handle is not None and self.block_visual_keys.get(block.block_id) != key:
            handle.remove()
            self.block_handles.pop(block.block_id, None)
            handle = None
        if handle is None:
            vertices, faces = _box_mesh(aabb_min, aabb_max)
            handle = self.server.scene.add_mesh_simple(
                name,
                vertices=vertices,
                faces=faces,
                color=color,
                wireframe=True,
                opacity=0.75,
                side="double",
                visible=self.state.show_block_bounds,
            )
            self.block_handles[block.block_id] = handle
            self.block_visual_keys[block.block_id] = key
        handle.visible = self.state.show_block_bounds

    def _upsert_label(self, block: BlockDebugInfo):
        aabb_min, aabb_max = self._block_aabb(block)
        center = (np.array(aabb_min, dtype=np.float32) + np.array(aabb_max, dtype=np.float32)) * 0.5
        key = (
            self.state.show_block_labels,
            self.state.block_debug_aabb,
            block.active,
            block.resident,
            block.gaussian_count,
        )
        name = f"/block_debug/labels/{block.block_id}"
        handle = self.label_handles.get(block.block_id)
        if handle is not None and self.label_visual_keys.get(block.block_id) != key:
            handle.remove()
            self.label_handles.pop(block.block_id, None)
            handle = None
        if handle is None:
            handle = self.server.scene.add_label(
                name,
                text=self._label_text(block),
                position=center,
                visible=self.state.show_block_labels,
            )
            self.label_handles[block.block_id] = handle
            self.label_visual_keys[block.block_id] = key
        handle.visible = self.state.show_block_labels

    def _selected_info(self):
        get_block = getattr(self.scene, "get_block_debug_info", None)
        block = get_block(self.state.selected_block_id) if get_block is not None else None
        if block is None:
            return f"Selected block: {self.state.selected_block_id} (not found)"
        raw_count = block.raw_gaussian_count if block.raw_gaussian_count is not None else block.gaussian_count
        valid_count = block.bounds_valid_gaussian_count
        ratio = block.raw_to_filtered_aabb_volume_ratio
        def fmt(values):
            return [round(float(v), 4) for v in values]
        return (
            f"Selected block: {block.block_id}\n"
            f"grid: {block.grid_index}\n"
            f"raw gaussians: {raw_count:,}\n"
            f"bounds-valid: {valid_count if valid_count is not None else 'n/a'}\n"
            f"volume ratio: {round(float(ratio), 4) if ratio is not None else 'n/a'}\n"
            f"logical min: {fmt(block.logical_bounds_min or block.content_aabb_min)}\n"
            f"logical max: {fmt(block.logical_bounds_max or block.content_aabb_max)}\n"
            f"raw min: {fmt(block.raw_gaussian_bounds_min or block.content_aabb_min)}\n"
            f"raw max: {fmt(block.raw_gaussian_bounds_max or block.content_aabb_max)}\n"
            f"filtered min: {fmt(block.filtered_render_bounds_min or block.cull_aabb_min)}\n"
            f"filtered max: {fmt(block.filtered_render_bounds_max or block.cull_aabb_max)}\n"
            f"cull min: {fmt(block.cull_aabb_min)}\n"
            f"cull max: {fmt(block.cull_aabb_max)}"
        )

    def bind_info_handles(self, selected_info_handle, warning_handle):
        self.selected_info_handle = selected_info_handle
        self.warning_handle = warning_handle
        self.update_info_text()

    def update_info_text(self):
        if self.selected_info_handle is not None:
            self.selected_info_handle.value = self._selected_info()
        if self.warning_handle is not None:
            get_block = getattr(self.scene, "get_block_debug_info", None)
            selected_exists = (
                get_block(self.state.selected_block_id) is not None
                if get_block is not None
                else False
            )
            invalid_selected = (
                self.state.culling_mode == "selected_block_only"
                and not selected_exists
            )
            self.warning_handle.value = (
                "Selected block id does not exist; render active set is empty."
                if invalid_selected
                else ""
            )

    def update_blocks(self):
        self.latest_debug_blocks = self.scene.get_debug_geometry()
        for block in self.latest_debug_blocks:
            self._upsert_block_mesh(block)
            self._upsert_label(block)
        self.update_info_text()

    def update_frustum(self, c2w, K, width, height, near_plane, far_plane):
        key = (
            self.state.show_camera_frustum,
            tuple(round(float(v), 4) for v in np.asarray(c2w).reshape(-1)),
            tuple(round(float(v), 4) for v in np.asarray(K).reshape(-1)),
            int(width),
            int(height),
            round(float(near_plane), 5),
            round(float(far_plane), 5),
        )
        if self.frustum_handle is not None and self.frustum_key != key:
            self.frustum_handle.remove()
            self.frustum_handle = None
        if self.frustum_handle is None:
            vertices, faces = _frustum_mesh(c2w, K, width, height, near_plane, far_plane)
            self.frustum_handle = self.server.scene.add_mesh_simple(
                "/block_debug/render_camera_frustum",
                vertices=vertices,
                faces=faces,
                color=(255, 225, 70),
                wireframe=True,
                opacity=0.95,
                side="double",
                visible=self.state.show_camera_frustum,
            )
            self.frustum_key = key
        self.frustum_handle.visible = self.state.show_camera_frustum

    def update(self, c2w, K, width, height, near_plane, far_plane):
        self.update_blocks()
        self.update_frustum(c2w, K, width, height, near_plane, far_plane)


class MetricsGsplatViewer(GsplatViewer):
    def __init__(
        self,
        server: viser.ViserServer,
        render_fn,
        output_dir: Path,
        mode: str = "rendering",
        scene_stats: Optional[SceneStats] = None,
        block_debug_state: Optional[BlockDebugState] = None,
        block_debug_visualizer: Optional[BlockDebugVisualizer] = None,
    ):
        self.scene_stats = scene_stats
        self.block_debug_state = block_debug_state
        self.block_debug_visualizer = block_debug_visualizer
        self.mouse_look_enabled = True
        self._last_camera_position_gui_update = 0.0
        super().__init__(server, render_fn, output_dir, mode)

    def _send_camera_control_mode(self, client: viser.ClientHandle):
        mode = "first_person" if self.mouse_look_enabled else "orbit"
        client._websock_connection.queue_message(
            viser_messages.SetCameraControlModeMessage(mode)
        )

    def _connect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id] = Renderer(
            viewer=self, client=client, lock=self.lock
        )
        self._renderers[client_id].start()
        self._send_camera_control_mode(client)

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            self._last_move_time = time.time()
            with self.server.atomic():
                camera_state = self.get_camera_state(client)
                self._renderers[client_id].submit(RenderTask("move", camera_state))

        initial_pose = _initial_camera_pose_from_scene(self.scene_stats)
        if initial_pose is not None:
            position, look_at, up = initial_pose
            with self.server.atomic():
                client.camera.position = position
                client.camera.look_at = look_at
                client.camera.up_direction = up
            camera_state = self.get_camera_state(client)
            self._renderers[client_id].submit(RenderTask("move", camera_state))

    def _populate_rendering_tab(self):
        with self._rendering_folder:
            with self.server.gui.add_folder("Navigation"):
                mouse_look = self.server.gui.add_checkbox(
                    "Mouse Look",
                    initial_value=self.mouse_look_enabled,
                    hint=(
                        "Keep cursor drags rotational in-place instead of orbiting "
                        "around a fixed look-at point."
                    ),
                )

                @mouse_look.on_update
                def _(_) -> None:
                    self.mouse_look_enabled = bool(mouse_look.value)
                    for client in self.server.get_clients().values():
                        self._send_camera_control_mode(client)

            with self.server.gui.add_folder("Metrics"):
                fps_number = self.server.gui.add_number(
                    "FPS",
                    initial_value=0.0,
                    disabled=True,
                    hint="Most recent completed viewer render FPS.",
                )
                frame_time_number = self.server.gui.add_number(
                    "Frame ms",
                    initial_value=0.0,
                    disabled=True,
                    hint="Most recent completed viewer render time in milliseconds.",
                )
                camera_position_text = self.server.gui.add_text(
                    "Camera Position",
                    initial_value="x: -, y: -, z: -",
                    disabled=True,
                    hint="Current render camera position in world coordinates.",
                )
            if self.block_debug_state is not None and self.block_debug_visualizer is not None:
                with self.server.gui.add_folder("Block Debug"):
                    show_bounds = self.server.gui.add_checkbox(
                        "Show Block Bounds",
                        initial_value=self.block_debug_state.show_block_bounds,
                    )
                    aabb_dropdown = self.server.gui.add_dropdown(
                        "Bounds AABB",
                        ("cull", "filtered", "raw", "logical", "content"),
                        initial_value=self.block_debug_state.block_debug_aabb,
                    )
                    show_frustum = self.server.gui.add_checkbox(
                        "Show Render Frustum",
                        initial_value=self.block_debug_state.show_camera_frustum,
                    )
                    show_labels = self.server.gui.add_checkbox(
                        "Show Block Labels",
                        initial_value=self.block_debug_state.show_block_labels,
                    )
                    culling_dropdown = self.server.gui.add_dropdown(
                        "Culling Mode",
                        (
                            "normal_frustum_culling",
                            "force_all_blocks_active",
                            "force_no_blocks_active",
                            "selected_block_only",
                        ),
                        initial_value=self.block_debug_state.culling_mode,
                    )
                    selected_block = self.server.gui.add_number(
                        "Selected Block",
                        initial_value=self.block_debug_state.selected_block_id,
                        step=1,
                    )
                    freeze_button = self.server.gui.add_button(
                        "Freeze Culling Camera",
                        icon=viser.Icon.CAMERA,
                    )
                    unfreeze_button = self.server.gui.add_button(
                        "Unfreeze Culling Camera",
                        icon=viser.Icon.CAMERA_OFF,
                    )
                    frozen_status = self.server.gui.add_text(
                        "Culling Camera",
                        initial_value="live",
                        disabled=True,
                    )
                    selected_info = self.server.gui.add_text(
                        "Selected Info",
                        initial_value="",
                        disabled=True,
                    )
                    warning_text = self.server.gui.add_text(
                        "Warning",
                        initial_value="",
                        disabled=True,
                    )
                    self.block_debug_visualizer.bind_info_handles(
                        selected_info, warning_text
                    )

                    def _request_debug_rerender(event):
                        self.block_debug_visualizer.update_blocks()
                        self.rerender(event)

                    @show_bounds.on_update
                    def _(event) -> None:
                        self.block_debug_state.show_block_bounds = bool(show_bounds.value)
                        _request_debug_rerender(event)

                    @aabb_dropdown.on_update
                    def _(event) -> None:
                        self.block_debug_state.block_debug_aabb = aabb_dropdown.value
                        _request_debug_rerender(event)

                    @show_frustum.on_update
                    def _(event) -> None:
                        self.block_debug_state.show_camera_frustum = bool(show_frustum.value)
                        self.rerender(event)

                    @show_labels.on_update
                    def _(event) -> None:
                        self.block_debug_state.show_block_labels = bool(show_labels.value)
                        _request_debug_rerender(event)

                    @culling_dropdown.on_update
                    def _(event) -> None:
                        self.block_debug_state.culling_mode = culling_dropdown.value
                        _request_debug_rerender(event)

                    @selected_block.on_update
                    def _(event) -> None:
                        self.block_debug_state.selected_block_id = int(selected_block.value)
                        _request_debug_rerender(event)

                    @freeze_button.on_click
                    def _(event) -> None:
                        client = getattr(event, "client", None)
                        if client is None:
                            clients = self.server.get_clients()
                            client = next(iter(clients.values()), None)
                        if client is None:
                            return
                        frozen_state = self.get_camera_state(client)
                        self.block_debug_state.frozen_culling_camera_state = frozen_state
                        self.block_debug_state.show_camera_frustum = True
                        show_frustum.value = True
                        observer, target, up = _observer_pose_from_render_camera(
                            frozen_state,
                            self.block_debug_visualizer.scene_extent,
                        )
                        with self.server.atomic():
                            client.camera.position = observer
                            client.camera.look_at = target
                            client.camera.up_direction = up
                        frozen_status.value = "frozen: live camera is observer"
                        self.rerender(event)

                    @unfreeze_button.on_click
                    def _(event) -> None:
                        self.block_debug_state.frozen_culling_camera_state = None
                        frozen_status.value = "live"
                        self.rerender(event)
        self._rendering_tab_handles.update(
            {
                "fps_number": fps_number,
                "frame_time_number": frame_time_number,
                "camera_position_text": camera_position_text,
            }
        )
        super()._populate_rendering_tab()

    @staticmethod
    def _format_camera_position(position) -> str:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        return (
            f"x: {position[0]:.3f}, "
            f"y: {position[1]:.3f}, "
            f"z: {position[2]:.3f}"
        )

    def _after_render(self):
        super()._after_render()
        if "fps_number" in self._rendering_tab_handles:
            self._rendering_tab_handles["fps_number"].value = getattr(
                self.render_tab_state, "fps", 0.0
            )
            self._rendering_tab_handles[
                "frame_time_number"
            ].value = getattr(self.render_tab_state, "frame_time_ms", 0.0)
        if "camera_position_text" in self._rendering_tab_handles:
            now = time.monotonic()
            if now - self._last_camera_position_gui_update >= 0.2:
                camera_position = getattr(
                    self.render_tab_state, "camera_position", None
                )
                if camera_position is not None:
                    self._rendering_tab_handles["camera_position_text"].value = (
                        self._format_camera_position(camera_position)
                    )
                    self._last_camera_position_gui_update = now


def _initial_camera_pose_from_scene(
    scene_stats: Optional[SceneStats],
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if (
        scene_stats is None
        or scene_stats.scene_aabb_min is None
        or scene_stats.scene_aabb_max is None
    ):
        return None
    bounds_min = (
        scene_stats.view_aabb_min
        if scene_stats.view_aabb_min is not None
        else scene_stats.scene_aabb_min
    )
    bounds_max = (
        scene_stats.view_aabb_max
        if scene_stats.view_aabb_max is not None
        else scene_stats.scene_aabb_max
    )
    scene_min = np.asarray(bounds_min, dtype=np.float64)
    scene_max = np.asarray(bounds_max, dtype=np.float64)
    if not np.all(np.isfinite(scene_min)) or not np.all(np.isfinite(scene_max)):
        return None
    extent = scene_max - scene_min
    radius = float(np.linalg.norm(extent) * 0.5)
    if radius <= 0.0:
        return None
    center = (scene_min + scene_max) * 0.5
    return _supersplat_style_initial_pose(center, radius)


def _supersplat_style_initial_pose(
    center: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # SuperSplat initializes at azim=-45/elev=-10/zoom=1 and preserves that
    # small absolute camera distance when model bounds arrive. Starting near
    # the view center is more useful than fitting the full raw AABB.
    azim = np.deg2rad(-45.0)
    elev = np.deg2rad(10.0)
    target_to_camera = np.asarray(
        [
            np.sin(azim) * np.cos(elev),
            np.cos(azim) * np.cos(elev),
            np.sin(elev),
        ],
        dtype=np.float64,
    )
    target_to_camera /= np.linalg.norm(target_to_camera)
    supersplat_unit_distance = 1.0 / np.sin(np.deg2rad(75.0) * 0.5)
    distance = min(max(radius * 0.18, supersplat_unit_distance), radius * 2.4)
    position = center + target_to_camera * distance
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return position, center, up


def _resolve_scene_paths(args) -> list[Path]:
    if args.scene is not None:
        if args.ckpt is not None or args.block_metadata is not None:
            raise ValueError("Use either --scene or legacy --ckpt/--block-metadata, not both.")
        return [Path(path) for path in args.scene]
    if args.block_metadata is not None:
        return [Path(args.block_metadata)]
    if args.ckpt is not None:
        return [Path(path) for path in args.ckpt]
    raise ValueError("Pass --scene PATH, or legacy --ckpt/--block-metadata.")


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)
    scene_paths = _resolve_scene_paths(args)
    culling_mode: BlockCullingMode = (
        "force_all_blocks_active"
        if args.force_all_blocks_active
        else "normal_frustum_culling"
    )
    scene = SceneLoader.load(
        scene_paths,
        device=device,
        scene_type=args.scene_type,
        block_residency=args.block_residency,
        block_keepalive_frames=args.block_keepalive_frames,
        gpu_block_budget_mb=args.gpu_block_budget_mb,
        block_cache_policy=args.block_cache_policy,
        culling_mode=culling_mode,
        ply_coordinate_mode=args.ply_coordinate_mode,
    )
    scene_stats = scene.get_scene_stats()
    renderer = GsplatRenderer(device)
    print(
        "Loaded scene:",
        scene_stats.scene_type,
        f"{scene_stats.total_gaussians:,} total Gaussians,",
        f"sh_degree={scene_stats.sh_degree}",
    )
    if scene_stats.scene_type == "blocked":
        print(
            "Blocked scene:",
            f"{int(scene_stats.total_blocks)} blocks,",
            f"residency={args.block_residency},",
            f"gpu_block_budget_mb={args.gpu_block_budget_mb}",
        )

    render_metrics = RenderMetricsState()
    metrics_logger = None
    if args.metrics_log_path is not None and world_rank == 0:
        metrics_logger = RuntimeMetricsLogger(
            args.metrics_log_path,
            args.metrics_interval,
            args.metrics_gpu_index,
            render_metrics,
        )
        metrics_logger.start()

    block_debug_state = None
    block_debug_visualizer = None
    if scene_stats.scene_type == "blocked":
        block_debug_state = BlockDebugState(
            show_block_bounds=args.show_block_bounds,
            block_debug_aabb=args.block_debug_aabb,
            show_camera_frustum=args.show_camera_frustum,
            culling_mode=culling_mode,
        )

    cumulative_peak_vram = float("nan")
    if device.type == "cuda":
        cumulative_peak_vram = float(torch.cuda.max_memory_allocated(device))

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        nonlocal cumulative_peak_vram
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        render_c2w_np = camera_state.c2w
        render_tab_state.camera_position = render_c2w_np[:3, 3].copy()
        render_K_np = camera_state.get_K((width, height))
        cull_camera_state = camera_state
        if block_debug_state is not None and block_debug_state.culling_camera_frozen:
            cull_camera_state = block_debug_state.frozen_culling_camera_state
        cull_c2w_np = cull_camera_state.c2w
        cull_K_np = cull_camera_state.get_K((width, height))

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        if block_debug_state is not None:
            scene.set_culling_mode(
                block_debug_state.culling_mode,
                selected_block_id=block_debug_state.selected_block_id,
            )
        frame_camera = FrameCamera(
            c2w=torch.from_numpy(render_c2w_np).float(),
            K=torch.from_numpy(render_K_np).float(),
            width=width,
            height=height,
            cull_c2w=torch.from_numpy(cull_c2w_np).float(),
            cull_K=torch.from_numpy(cull_K_np).float(),
        )
        render_settings = RenderSettings(
            width=width,
            height=height,
            sh_degree=(
                min(render_tab_state.max_sh_degree, scene_stats.sh_degree)
                if scene_stats.sh_degree is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            background=tuple(
                float(v) / 255.0 for v in render_tab_state.backgrounds
            ),
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
            with_ut=args.with_ut,
            with_eval3d=args.with_eval3d,
            distributed=world_size > 1,
        )
        if device.type == "cuda":
            cumulative_peak_vram = max(
                cumulative_peak_vram,
                float(torch.cuda.max_memory_allocated(device)),
            )
            torch.cuda.reset_peak_memory_stats(device)
        active_splats = scene.prepare_frame(frame_camera, render_settings)
        result = renderer.render(active_splats, frame_camera, render_settings)
        scene_metrics = active_splats.scene_metrics
        torch_memory = _torch_cuda_memory_snapshot(device)
        cumulative_peak_vram = max(cumulative_peak_vram, torch_memory["peak_allocated"])
        peak_vram = cumulative_peak_vram
        peak_vram_mb = peak_vram / (1024.0 * 1024.0)
        torch_allocated_mb = torch_memory["allocated"] / (1024.0 * 1024.0)
        torch_reserved_mb = torch_memory["reserved"] / (1024.0 * 1024.0)
        torch_frame_peak_allocated_mb = torch_memory["peak_allocated"] / (
            1024.0 * 1024.0
        )
        torch_frame_peak_reserved_mb = torch_memory["peak_reserved"] / (
            1024.0 * 1024.0
        )

        render_tab_state.total_gs_count = scene_stats.total_gaussians
        render_tab_state.rendered_gs_count = int(result.visible_gaussians)

        if active_splats.num_gaussians == 0:
            renders = result.image[..., 0:3].cpu().numpy()
        elif render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = result.image[..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = result.depth if result.depth is not None else result.image[..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = result.alpha[..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        frame_time_ms = result.render_time_ms
        render_tab_state.frame_time_ms = frame_time_ms
        render_tab_state.fps = 1000.0 / max(frame_time_ms, 1e-10)
        render_metrics.record_frame(
            frame_time_ms=frame_time_ms,
            width=width,
            height=height,
            active_blocks=scene_metrics.active_blocks,
            active_block_ids=scene_metrics.active_block_ids,
            active_gaussians=scene_metrics.active_gaussians,
            active_gaussian_fraction=scene_metrics.active_gaussian_fraction,
            visible_gaussians=render_tab_state.rendered_gs_count,
            total_gaussians=render_tab_state.total_gs_count,
            total_blocks=scene_metrics.total_blocks,
            resident_blocks=scene_metrics.resident_blocks,
            resident_gaussians=scene_metrics.resident_gaussians,
            resident_gaussian_fraction=scene_metrics.resident_gaussian_fraction,
            block_cull_ms=scene_metrics.block_cull_ms,
            block_concat_ms=scene_metrics.block_concat_ms,
            gpu_upload_ms=scene_metrics.gpu_upload_ms,
            cache_hits=scene_metrics.cache_hits,
            cache_misses=scene_metrics.cache_misses,
            cache_hit_rate=scene_metrics.cache_hit_rate,
            newly_uploaded_blocks=scene_metrics.newly_uploaded_blocks,
            newly_uploaded_gaussians=scene_metrics.newly_uploaded_gaussians,
            evicted_blocks=scene_metrics.evicted_blocks,
            evicted_gaussians=scene_metrics.evicted_gaussians,
            gpu_resident_bytes=scene_metrics.gpu_resident_bytes,
            gpu_resident_mb=scene_metrics.gpu_resident_mb,
            cpu_resident_bytes=scene_metrics.cpu_resident_bytes,
            cpu_resident_mb=scene_metrics.cpu_resident_mb,
            peak_vram=peak_vram,
            peak_vram_mb=peak_vram_mb,
            torch_allocated_mb=torch_allocated_mb,
            torch_reserved_mb=torch_reserved_mb,
            torch_frame_peak_allocated_mb=torch_frame_peak_allocated_mb,
            torch_frame_peak_reserved_mb=torch_frame_peak_reserved_mb,
            gpu_budget_exceeded=scene_metrics.gpu_budget_exceeded,
            active_set_exceeds_budget=scene_metrics.active_set_exceeds_budget,
        )
        if block_debug_visualizer is not None:
            block_debug_visualizer.update(
                cull_c2w_np,
                cull_K_np,
                width,
                height,
                render_tab_state.near_plane,
                render_tab_state.far_plane,
            )
        return renders

    server = viser.ViserServer(port=args.port, verbose=False)
    if scene_stats.scene_type == "blocked":
        block_debug_visualizer = BlockDebugVisualizer(
            server,
            scene,
            block_debug_state,
        )
    _ = MetricsGsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
        scene_stats=scene_stats,
        block_debug_state=block_debug_state,
        block_debug_visualizer=block_debug_visualizer,
    )
    print("Viewer running... Ctrl+C to exit.")
    try:
        time.sleep(100000)
    finally:
        if metrics_logger is not None:
            metrics_logger.stop()


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --ckpt results/garden/ckpts/ckpt_6999_rank0.pt \
        --output_dir results/garden/ \
        --port 8082
    
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --output_dir results/garden/ \
        --port 8082
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--scene",
        type=str,
        nargs="+",
        default=None,
        help="path to a flat .pt checkpoint, Gaussian .ply splat file, or blocked scene asset",
    )
    parser.add_argument(
        "--scene-type",
        choices=["auto", "flat", "blocked"],
        default="auto",
        help="scene loader mode; auto strictly detects flat checkpoints vs blocked assets",
    )
    parser.add_argument(
        "--ply-coordinate-mode",
        choices=["native", "supersplat"],
        default="native",
        help=(
            "coordinate mode for Gaussian PLY files; use native for Nerfstudio/Inria "
            "exports and supersplat for PLYs that need the SuperSplat viewer transform"
        ),
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--block-metadata",
        type=str,
        default=None,
        help="path to a block-structured checkpoint produced by partition_checkpoint.py",
    )
    parser.add_argument(
        "--block-residency",
        choices=["gpu", "cpu"],
        default="gpu",
        help="where block tensors live by default in block mode; cpu mode uploads active blocks synchronously",
    )
    parser.add_argument(
        "--force-all-blocks-active",
        action="store_true",
        help="render every block in block mode for flat-vs-block correctness checks",
    )
    parser.add_argument(
        "--block-keepalive-frames",
        type=int,
        default=0,
        help="keep recently visible blocks active for this many additional frames",
    )
    parser.add_argument(
        "--gpu-block-budget-mb",
        type=float,
        default=None,
        help="optional persistent per-block GPU cache budget in MiB for CPU-resident block mode",
    )
    parser.add_argument(
        "--block-cache-policy",
        choices=["lru"],
        default="lru",
        help="persistent GPU block cache eviction policy",
    )
    parser.add_argument(
        "--show-block-bounds",
        action="store_true",
        help="show block AABB wireframes on startup when block mode is enabled",
    )
    parser.add_argument(
        "--show-camera-frustum",
        action="store_true",
        help="show the render camera frustum on startup when block mode is enabled",
    )
    parser.add_argument(
        "--block-debug-aabb",
        choices=["content", "cull", "logical", "raw", "filtered"],
        default="cull",
        help="which block AABB to draw in the debug overlay",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--with_ut", action="store_true", help="use uncentered transform"
    )
    parser.add_argument("--with_eval3d", action="store_true", help="use eval 3D")
    parser.add_argument(
        "--metrics-log-path",
        type=str,
        default=None,
        help="optional CSV path for runtime CPU/RAM/GPU metrics",
    )
    parser.add_argument(
        "--metrics-interval",
        type=float,
        default=0.5,
        help="seconds between runtime metrics samples",
    )
    parser.add_argument(
        "--metrics-gpu-index",
        type=int,
        default=0,
        help="NVML GPU index used for runtime GPU metrics",
    )
    args = parser.parse_args()

    cli(main, args, verbose=True)
