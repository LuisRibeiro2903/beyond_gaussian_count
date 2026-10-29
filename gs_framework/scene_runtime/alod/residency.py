"""Read-only hierarchy-property residency policies for production ALoD."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np
import torch

from scene_runtime.types import ActiveSplats

from .host import AlodHostProperties, property_tensor_bytes


AlodHostResidencyMode = Literal[
    "host_nocache",
    "host_previous_reuse",
    "host_viewer_previous_reuse",
    "host_bounded_cache",
]


class AlodCacheCapacityError(RuntimeError):
    pass


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_unique(ids: np.ndarray) -> None:
    if ids.size != np.unique(ids).size:
        raise ValueError("selected global_node_ids contain duplicates")


def _copy_rows(
    destination: ActiveSplats,
    destination_rows: torch.Tensor,
    source: ActiveSplats,
    source_rows: torch.Tensor | None = None,
) -> None:
    if source_rows is None:
        source_rows = torch.arange(
            destination_rows.numel(), dtype=torch.int64, device=destination_rows.device
        )
    for name in ("means", "quats", "scales", "opacities", "colors"):
        target_tensor = getattr(destination, name)
        source_tensor = getattr(source, name).index_select(0, source_rows)
        target_tensor.index_copy_(0, destination_rows, source_tensor)


@dataclass(frozen=True)
class AlodResidencyFrame:
    hierarchy_splats: ActiveSplats
    ordered_global_node_ids: np.ndarray
    ordered_gpu_slots: torch.Tensor
    requested_rows: int
    cache_hits: int
    cache_misses: int
    loads: int
    evictions: int
    retained_rows: int
    new_rows: int
    discarded_rows: int
    cache_occupancy_rows: int
    h2d_rows: int
    h2d_bytes: int
    residency_lookup_ms: float
    host_gather_ms: float
    h2d_transfer_ms: float
    gpu_slot_index_ms: float
    selected_gpu_gather_ms: float
    persistent_dynamic_property_bytes: int
    residency_metadata_cpu_bytes: int
    residency_metadata_gpu_bytes: int
    frame_local_property_bytes: int
    frame_local_index_bytes: int
    selected_set_churn_rows: int
    cache_management_ms: float
    previous_working_set_rows: int
    previous_intersection_rows: int

    @property
    def retention_ratio(self) -> float:
        return self.retained_rows / self.requested_rows if self.requested_rows else 1.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.requested_rows if self.requested_rows else 1.0


class _AlodResidencyBase:
    requires_unique_ids = True

    def __init__(
        self,
        host: AlodHostProperties,
        device: torch.device,
        *,
        hierarchy_start: int,
        validate_mappings: bool,
    ) -> None:
        self.host = host
        self.device = device
        self.hierarchy_start = int(hierarchy_start)
        self.validate_mappings = bool(validate_mappings)
        self.previous_selected_ids = np.empty(0, dtype=np.int64)

    @property
    def capacity_rows(self) -> int:
        return 0

    def reset(self) -> None:
        self.previous_selected_ids = np.empty(0, dtype=np.int64)

    def _ids(self, selected_global_node_ids: np.ndarray) -> np.ndarray:
        ids = self.host.validate_global_ids(selected_global_node_ids)
        if ids.size and int(ids.min()) < self.hierarchy_start:
            raise ValueError("dynamic residency request contains a skybox global ID")
        if self.validate_mappings and self.requires_unique_ids:
            _validate_unique(ids)
        return ids

class AlodHostNoCache(_AlodResidencyBase):
    """Evaluation-style host gather and transfer with no cross-frame reuse."""

    # No-cache evaluation has no global-ID-to-resident-slot mapping.  Ordered
    # duplicate semantic occurrences are therefore valid and are gathered and
    # transferred as separate renderer rows, matching released evaluation.
    requires_unique_ids = False

    def prepare(self, selected_global_node_ids: np.ndarray) -> AlodResidencyFrame:
        lookup_start = time.perf_counter()
        ids = self._ids(selected_global_node_ids)
        previous_count = int(self.previous_selected_ids.size)
        lookup_ms = (time.perf_counter() - lookup_start) * 1000.0

        gather_start = time.perf_counter()
        cpu = self.host.gather_cpu(ids)
        gather_ms = (time.perf_counter() - gather_start) * 1000.0
        transfer_start = time.perf_counter()
        active = ActiveSplats(
            means=cpu.means.to(self.device, non_blocking=False),
            quats=cpu.quats.to(self.device, non_blocking=False),
            scales=cpu.scales.to(self.device, non_blocking=False),
            opacities=cpu.opacities.to(self.device, non_blocking=False),
            colors=cpu.colors.to(self.device, non_blocking=False),
        )
        _sync(self.device)
        transfer_ms = (time.perf_counter() - transfer_start) * 1000.0
        self.previous_selected_ids = ids.copy()
        empty_slots = torch.empty(0, dtype=torch.int64, device=self.device)
        return AlodResidencyFrame(
            hierarchy_splats=active,
            ordered_global_node_ids=ids.copy(),
            ordered_gpu_slots=empty_slots,
            requested_rows=int(ids.size),
            cache_hits=0,
            cache_misses=int(ids.size),
            loads=int(ids.size),
            evictions=0,
            retained_rows=0,
            new_rows=int(ids.size),
            discarded_rows=previous_count,
            cache_occupancy_rows=0,
            h2d_rows=int(ids.size),
            h2d_bytes=int(ids.size * self.host.property_bytes_per_row),
            residency_lookup_ms=lookup_ms,
            host_gather_ms=gather_ms,
            h2d_transfer_ms=transfer_ms,
            gpu_slot_index_ms=0.0,
            selected_gpu_gather_ms=0.0,
            persistent_dynamic_property_bytes=0,
            residency_metadata_cpu_bytes=int(self.previous_selected_ids.nbytes),
            residency_metadata_gpu_bytes=0,
            frame_local_property_bytes=property_tensor_bytes(active),
            frame_local_index_bytes=0,
            selected_set_churn_rows=0,
            cache_management_ms=0.0,
            previous_working_set_rows=previous_count,
            previous_intersection_rows=0,
        )


class AlodPreviousWorkingSetReuse(_AlodResidencyBase):
    """Source-inspired exact-row reuse of only the immediately prior set."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.global_to_previous_row = np.full(self.host.row_count, -1, dtype=np.int64)
        self.previous_splats: ActiveSplats | None = None

    def reset(self) -> None:
        super().reset()
        self.global_to_previous_row.fill(-1)
        self.previous_splats = None

    def prepare(self, selected_global_node_ids: np.ndarray) -> AlodResidencyFrame:
        lookup_start = time.perf_counter()
        ids = self._ids(selected_global_node_ids)
        previous_rows = self.global_to_previous_row[ids]
        hit_mask = previous_rows >= 0
        retained = int(hit_mask.sum())
        misses = int(ids.size - retained)
        discarded = int(self.previous_selected_ids.size - retained)
        lookup_ms = (time.perf_counter() - lookup_start) * 1000.0

        active = self.host.empty(int(ids.size), self.device)
        gpu_copy_start = time.perf_counter()
        if retained:
            assert self.previous_splats is not None
            destination_rows = torch.from_numpy(
                np.flatnonzero(hit_mask).astype(np.int64)
            ).to(self.device)
            source_rows = torch.from_numpy(previous_rows[hit_mask].astype(np.int64)).to(
                self.device
            )
            _copy_rows(active, destination_rows, self.previous_splats, source_rows)
        _sync(self.device)
        gpu_copy_ms = (time.perf_counter() - gpu_copy_start) * 1000.0

        miss_ids = ids[~hit_mask]
        gather_start = time.perf_counter()
        cpu = self.host.gather_cpu(miss_ids)
        gather_ms = (time.perf_counter() - gather_start) * 1000.0
        transfer_start = time.perf_counter()
        transferred = ActiveSplats(
            means=cpu.means.to(self.device, non_blocking=False),
            quats=cpu.quats.to(self.device, non_blocking=False),
            scales=cpu.scales.to(self.device, non_blocking=False),
            opacities=cpu.opacities.to(self.device, non_blocking=False),
            colors=cpu.colors.to(self.device, non_blocking=False),
        )
        if misses:
            destination_rows = torch.from_numpy(
                np.flatnonzero(~hit_mask).astype(np.int64)
            ).to(self.device)
            _copy_rows(active, destination_rows, transferred)
        _sync(self.device)
        transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

        management_start = time.perf_counter()
        if self.previous_selected_ids.size:
            self.global_to_previous_row[self.previous_selected_ids] = -1
        self.global_to_previous_row[ids] = np.arange(ids.size, dtype=np.int64)
        self.previous_selected_ids = ids.copy()
        self.previous_splats = active
        if self.validate_mappings:
            np.testing.assert_array_equal(
                self.global_to_previous_row[ids], np.arange(ids.size, dtype=np.int64)
            )
        management_ms = (time.perf_counter() - management_start) * 1000.0
        return AlodResidencyFrame(
            hierarchy_splats=active,
            ordered_global_node_ids=ids.copy(),
            ordered_gpu_slots=torch.empty(0, dtype=torch.int64, device=self.device),
            requested_rows=int(ids.size),
            cache_hits=retained,
            cache_misses=misses,
            loads=misses,
            evictions=0,
            retained_rows=retained,
            new_rows=misses,
            discarded_rows=discarded,
            cache_occupancy_rows=int(ids.size),
            h2d_rows=misses,
            h2d_bytes=int(misses * self.host.property_bytes_per_row),
            residency_lookup_ms=lookup_ms,
            host_gather_ms=gather_ms,
            h2d_transfer_ms=transfer_ms,
            gpu_slot_index_ms=0.0,
            selected_gpu_gather_ms=gpu_copy_ms,
            persistent_dynamic_property_bytes=property_tensor_bytes(active),
            residency_metadata_cpu_bytes=int(
                self.global_to_previous_row.nbytes + self.previous_selected_ids.nbytes
            ),
            residency_metadata_gpu_bytes=0,
            frame_local_property_bytes=property_tensor_bytes(transferred),
            frame_local_index_bytes=int(retained * 2 * 8 + misses * 8),
            selected_set_churn_rows=misses + discarded,
            cache_management_ms=management_ms,
            previous_working_set_rows=int(retained + discarded),
            previous_intersection_rows=retained,
        )


class AlodViewerPreviousWorkingSetReuse(AlodPreviousWorkingSetReuse):
    """Reuse only rows belonging to viewer-compatible prior SPT cuts."""

    def prepare(
        self,
        selected_global_node_ids: np.ndarray,
        reusable_global_node_ids: np.ndarray,
    ) -> AlodResidencyFrame:
        lookup_start = time.perf_counter()
        ids = self._ids(selected_global_node_ids)
        reusable_ids = self._ids(reusable_global_node_ids)
        reusable_mask = np.zeros(ids.size, dtype=bool)
        if reusable_ids.size:
            if reusable_ids.size > ids.size or not np.array_equal(
                ids[-reusable_ids.size :], reusable_ids
            ):
                raise RuntimeError(
                    "viewer-compatible reused SPT rows are not the ordered selection suffix"
                )
            reusable_mask[-reusable_ids.size :] = True
        previous_rows = self.global_to_previous_row[ids]
        hit_mask = reusable_mask & (previous_rows >= 0)
        retained = int(hit_mask.sum())
        misses = int(ids.size - retained)
        discarded = int(self.previous_selected_ids.size - retained)
        lookup_ms = (time.perf_counter() - lookup_start) * 1000.0

        active = self.host.empty(int(ids.size), self.device)
        gpu_copy_start = time.perf_counter()
        if retained:
            assert self.previous_splats is not None
            destination_rows = torch.from_numpy(
                np.flatnonzero(hit_mask).astype(np.int64)
            ).to(self.device)
            source_rows = torch.from_numpy(previous_rows[hit_mask].astype(np.int64)).to(
                self.device
            )
            _copy_rows(active, destination_rows, self.previous_splats, source_rows)
        _sync(self.device)
        gpu_copy_ms = (time.perf_counter() - gpu_copy_start) * 1000.0

        miss_ids = ids[~hit_mask]
        gather_start = time.perf_counter()
        cpu = self.host.gather_cpu(miss_ids)
        gather_ms = (time.perf_counter() - gather_start) * 1000.0
        transfer_start = time.perf_counter()
        transferred = ActiveSplats(
            means=cpu.means.to(self.device, non_blocking=False),
            quats=cpu.quats.to(self.device, non_blocking=False),
            scales=cpu.scales.to(self.device, non_blocking=False),
            opacities=cpu.opacities.to(self.device, non_blocking=False),
            colors=cpu.colors.to(self.device, non_blocking=False),
        )
        if misses:
            destination_rows = torch.from_numpy(
                np.flatnonzero(~hit_mask).astype(np.int64)
            ).to(self.device)
            _copy_rows(active, destination_rows, transferred)
        _sync(self.device)
        transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

        management_start = time.perf_counter()
        if self.previous_selected_ids.size:
            self.global_to_previous_row[self.previous_selected_ids] = -1
        self.global_to_previous_row[ids] = np.arange(ids.size, dtype=np.int64)
        self.previous_selected_ids = ids.copy()
        self.previous_splats = active
        if self.validate_mappings:
            np.testing.assert_array_equal(
                self.global_to_previous_row[ids], np.arange(ids.size, dtype=np.int64)
            )
            if retained != reusable_ids.size:
                raise RuntimeError(
                    "viewer-compatible SPT rows were not all present in the prior working set"
                )
        management_ms = (time.perf_counter() - management_start) * 1000.0
        return AlodResidencyFrame(
            hierarchy_splats=active,
            ordered_global_node_ids=ids.copy(),
            ordered_gpu_slots=torch.empty(0, dtype=torch.int64, device=self.device),
            requested_rows=int(ids.size),
            cache_hits=retained,
            cache_misses=misses,
            loads=misses,
            evictions=0,
            retained_rows=retained,
            new_rows=misses,
            discarded_rows=discarded,
            cache_occupancy_rows=int(ids.size),
            h2d_rows=misses,
            h2d_bytes=int(misses * self.host.property_bytes_per_row),
            residency_lookup_ms=lookup_ms,
            host_gather_ms=gather_ms,
            h2d_transfer_ms=transfer_ms,
            gpu_slot_index_ms=0.0,
            selected_gpu_gather_ms=gpu_copy_ms,
            persistent_dynamic_property_bytes=property_tensor_bytes(active),
            residency_metadata_cpu_bytes=int(
                self.global_to_previous_row.nbytes + self.previous_selected_ids.nbytes
            ),
            residency_metadata_gpu_bytes=0,
            frame_local_property_bytes=property_tensor_bytes(transferred),
            frame_local_index_bytes=int(retained * 2 * 8 + misses * 8),
            selected_set_churn_rows=misses + discarded,
            cache_management_ms=management_ms,
            previous_working_set_rows=int(retained + discarded),
            previous_intersection_rows=retained,
        )


class AlodBoundedCache(_AlodResidencyBase):
    """Fixed-capacity, read-only GPU property cache with deterministic LRU."""

    def __init__(self, *args, capacity_rows: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if capacity_rows <= 0:
            raise ValueError("capacity_rows must be positive")
        if capacity_rows > self.host.row_count - self.hierarchy_start:
            raise ValueError("capacity_rows exceeds hierarchy row count")
        self._capacity_rows = int(capacity_rows)
        self.cache = self.host.empty(self._capacity_rows, self.device)
        self.global_to_slot = np.full(self.host.row_count, -1, dtype=np.int64)
        self.slot_to_global = np.full(self._capacity_rows, -1, dtype=np.int64)
        self.last_used_frame = np.full(self._capacity_rows, -1, dtype=np.int64)
        self.frame_index = 0

    @property
    def capacity_rows(self) -> int:
        return self._capacity_rows

    def reset(self) -> None:
        super().reset()
        self.global_to_slot.fill(-1)
        self.slot_to_global.fill(-1)
        self.last_used_frame.fill(-1)
        self.frame_index = 0

    def _validate_bijection(self) -> None:
        live_slots = np.flatnonzero(self.slot_to_global >= 0)
        live_ids = self.slot_to_global[live_slots]
        if live_ids.size != np.unique(live_ids).size:
            raise RuntimeError("duplicate global IDs in cache reverse mapping")
        if not np.array_equal(self.global_to_slot[live_ids], live_slots):
            raise RuntimeError("global-ID/cache-slot mappings are not bijective")
        mapped_ids = np.flatnonzero(self.global_to_slot >= 0)
        if not np.array_equal(
            self.slot_to_global[self.global_to_slot[mapped_ids]], mapped_ids
        ):
            raise RuntimeError("cache reverse mapping contains a stale slot")

    def prepare(self, selected_global_node_ids: np.ndarray) -> AlodResidencyFrame:
        lookup_start = time.perf_counter()
        ids = self._ids(selected_global_node_ids)
        previous_count = int(self.previous_selected_ids.size)
        if ids.size > self.capacity_rows:
            raise AlodCacheCapacityError(
                f"selected hierarchy rows {ids.size} exceed cache capacity {self.capacity_rows}"
            )
        initial_slots = self.global_to_slot[ids]
        hit_mask = initial_slots >= 0
        hits = int(hit_mask.sum())
        misses = int(ids.size - hits)
        logical_intersection = int(
            np.count_nonzero(
                self.last_used_frame[initial_slots[hit_mask]] == self.frame_index - 1
            )
        )
        logical_new = int(ids.size - logical_intersection)
        logical_discarded = int(previous_count - logical_intersection)
        lookup_ms = (time.perf_counter() - lookup_start) * 1000.0

        management_start = time.perf_counter()
        protected = np.zeros(self.capacity_rows, dtype=bool)
        protected[initial_slots[hit_mask]] = True
        free_slots = np.flatnonzero(self.slot_to_global < 0)
        eviction_count = max(0, misses - int(free_slots.size))
        eviction_slots = np.empty(0, dtype=np.int64)
        if eviction_count:
            candidates = np.flatnonzero((self.slot_to_global >= 0) & ~protected)
            if candidates.size < eviction_count:
                raise AlodCacheCapacityError("no evictable slots remain for complete frame")
            order = np.lexsort((candidates, self.last_used_frame[candidates]))
            eviction_slots = candidates[order[:eviction_count]]
        target_slots = np.concatenate(
            (free_slots[: misses - eviction_count], eviction_slots)
        ).astype(np.int64, copy=False)
        if target_slots.size != misses:
            raise RuntimeError("cache allocator did not produce one slot per miss")
        evicted_ids = self.slot_to_global[eviction_slots].copy()
        if evicted_ids.size:
            self.global_to_slot[evicted_ids] = -1
        miss_ids = ids[~hit_mask]
        self.slot_to_global[target_slots] = miss_ids
        self.global_to_slot[miss_ids] = target_slots
        ordered_slots_cpu = self.global_to_slot[ids].copy()
        if np.any(ordered_slots_cpu < 0):
            raise RuntimeError("selected global ID did not resolve to a cache slot")
        management_ms = (time.perf_counter() - management_start) * 1000.0

        gather_start = time.perf_counter()
        cpu = self.host.gather_cpu(miss_ids)
        gather_ms = (time.perf_counter() - gather_start) * 1000.0
        transfer_start = time.perf_counter()
        transferred = ActiveSplats(
            means=cpu.means.to(self.device, non_blocking=False),
            quats=cpu.quats.to(self.device, non_blocking=False),
            scales=cpu.scales.to(self.device, non_blocking=False),
            opacities=cpu.opacities.to(self.device, non_blocking=False),
            colors=cpu.colors.to(self.device, non_blocking=False),
        )
        if misses:
            target_slots_gpu = torch.from_numpy(target_slots).to(self.device)
            _copy_rows(self.cache, target_slots_gpu, transferred)
        _sync(self.device)
        transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

        slot_start = time.perf_counter()
        ordered_slots = torch.from_numpy(ordered_slots_cpu).to(self.device)
        _sync(self.device)
        slot_ms = (time.perf_counter() - slot_start) * 1000.0
        gpu_gather_start = time.perf_counter()
        active = ActiveSplats(
            means=self.cache.means.index_select(0, ordered_slots),
            quats=self.cache.quats.index_select(0, ordered_slots),
            scales=self.cache.scales.index_select(0, ordered_slots),
            opacities=self.cache.opacities.index_select(0, ordered_slots),
            colors=self.cache.colors.index_select(0, ordered_slots),
        )
        _sync(self.device)
        gpu_gather_ms = (time.perf_counter() - gpu_gather_start) * 1000.0

        self.last_used_frame[ordered_slots_cpu] = self.frame_index
        self.frame_index += 1
        self.previous_selected_ids = ids.copy()
        if self.validate_mappings:
            self._validate_bijection()
            if not np.array_equal(self.slot_to_global[ordered_slots_cpu], ids):
                raise RuntimeError("ordered cache slots contain stale global IDs")
        occupancy = int(np.count_nonzero(self.slot_to_global >= 0))
        return AlodResidencyFrame(
            hierarchy_splats=active,
            ordered_global_node_ids=ids.copy(),
            ordered_gpu_slots=ordered_slots,
            requested_rows=int(ids.size),
            cache_hits=hits,
            cache_misses=misses,
            loads=misses,
            evictions=eviction_count,
            retained_rows=hits,
            new_rows=misses,
            discarded_rows=eviction_count,
            cache_occupancy_rows=occupancy,
            h2d_rows=misses,
            h2d_bytes=int(misses * self.host.property_bytes_per_row),
            residency_lookup_ms=lookup_ms,
            host_gather_ms=gather_ms,
            h2d_transfer_ms=transfer_ms,
            gpu_slot_index_ms=slot_ms,
            selected_gpu_gather_ms=gpu_gather_ms,
            persistent_dynamic_property_bytes=property_tensor_bytes(self.cache),
            residency_metadata_cpu_bytes=int(
                self.global_to_slot.nbytes
                + self.slot_to_global.nbytes
                + self.last_used_frame.nbytes
                + self.previous_selected_ids.nbytes
            ),
            residency_metadata_gpu_bytes=0,
            frame_local_property_bytes=int(
                property_tensor_bytes(active) + property_tensor_bytes(transferred)
            ),
            frame_local_index_bytes=int(ordered_slots.numel() * ordered_slots.element_size()),
            selected_set_churn_rows=logical_new + logical_discarded,
            cache_management_ms=management_ms,
            previous_working_set_rows=previous_count,
            previous_intersection_rows=logical_intersection,
        )


__all__ = [
    "AlodBoundedCache",
    "AlodCacheCapacityError",
    "AlodHostNoCache",
    "AlodHostResidencyMode",
    "AlodPreviousWorkingSetReuse",
    "AlodViewerPreviousWorkingSetReuse",
    "AlodResidencyFrame",
]
