"""Deterministic metadata-only replay of ALoD property residency policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Iterable, Sequence

import numpy as np


def _ids(values: np.ndarray | Iterable[int]) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.int64)
    if result.ndim != 1:
        raise ValueError("request global_node_ids must be one-dimensional")
    if result.size and int(result.min()) < 0:
        raise ValueError("request global_node_ids must be non-negative")
    if np.unique(result).size != result.size:
        raise ValueError("request global_node_ids must be unique within a frame")
    return result


@dataclass(frozen=True)
class AlodRequestTraceFrame:
    frame_index: int
    requested_rows: int
    overlap_with_previous_rows: int
    new_rows: int
    discarded_rows: int
    cumulative_union_rows: int
    ordered_global_node_ids_sha256: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class AlodCacheSimulation:
    policy: str
    capacity_rows: int | None
    capacity_bytes: int | None
    frame_count: int
    requested_rows: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    miss_rate: float
    loads: int
    evictions: int
    total_h2d_rows: int
    total_h2d_bytes: int
    maximum_occupancy_rows: int
    zero_transfer_frame_count: int

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


def characterize_request_trace(
    requests: Sequence[np.ndarray | Iterable[int]],
) -> tuple[AlodRequestTraceFrame, ...]:
    """Describe ordered requests without retaining a second full ID trace."""

    previous = np.empty(0, dtype=np.int64)
    cumulative = np.empty(0, dtype=np.int64)
    rows: list[AlodRequestTraceFrame] = []
    for frame_index, values in enumerate(requests):
        current = _ids(values)
        overlap = np.intersect1d(current, previous, assume_unique=True)
        new = np.setdiff1d(current, previous, assume_unique=True)
        discarded = np.setdiff1d(previous, current, assume_unique=True)
        cumulative = np.union1d(cumulative, current)
        rows.append(
            AlodRequestTraceFrame(
                frame_index=frame_index,
                requested_rows=int(current.size),
                overlap_with_previous_rows=int(overlap.size),
                new_rows=int(new.size),
                discarded_rows=int(discarded.size),
                cumulative_union_rows=int(cumulative.size),
                ordered_global_node_ids_sha256=hashlib.sha256(
                    current.tobytes()
                ).hexdigest(),
            )
        )
        previous = current
    return tuple(rows)


def _row_count(requests: Sequence[np.ndarray]) -> int:
    maximum = max((int(ids.max()) for ids in requests if ids.size), default=-1)
    return maximum + 1


def simulate_bounded_cache(
    requests: Sequence[np.ndarray | Iterable[int]],
    *,
    capacity_rows: int,
    property_bytes_per_row: int,
    global_row_count: int | None = None,
) -> AlodCacheSimulation:
    """Replay the production cache's protected-current-frame deterministic LRU."""

    normalized = [_ids(values) for values in requests]
    if capacity_rows <= 0:
        raise ValueError("capacity_rows must be positive")
    if property_bytes_per_row < 0:
        raise ValueError("property_bytes_per_row must be non-negative")
    maximum_request = max((ids.size for ids in normalized), default=0)
    if maximum_request > capacity_rows:
        raise ValueError(
            f"maximum frame request {maximum_request} exceeds cache capacity {capacity_rows}"
        )
    required_rows = _row_count(normalized)
    row_count = required_rows if global_row_count is None else int(global_row_count)
    if row_count < required_rows:
        raise ValueError("global_row_count does not cover the request trace")

    global_to_slot = np.full(row_count, -1, dtype=np.int64)
    slot_to_global = np.full(capacity_rows, -1, dtype=np.int64)
    last_used_frame = np.full(capacity_rows, -1, dtype=np.int64)
    hits = misses = evictions = zero_transfer_frames = maximum_occupancy = 0

    for frame_index, ids in enumerate(normalized):
        initial_slots = global_to_slot[ids]
        hit_mask = initial_slots >= 0
        frame_hits = int(hit_mask.sum())
        frame_misses = int(ids.size - frame_hits)
        protected = np.zeros(capacity_rows, dtype=bool)
        protected[initial_slots[hit_mask]] = True
        free_slots = np.flatnonzero(slot_to_global < 0)
        eviction_count = max(0, frame_misses - int(free_slots.size))
        eviction_slots = np.empty(0, dtype=np.int64)
        if eviction_count:
            candidates = np.flatnonzero((slot_to_global >= 0) & ~protected)
            if candidates.size < eviction_count:
                raise RuntimeError("no evictable slots remain for complete frame")
            order = np.lexsort((candidates, last_used_frame[candidates]))
            eviction_slots = candidates[order[:eviction_count]]
        target_slots = np.concatenate(
            (free_slots[: frame_misses - eviction_count], eviction_slots)
        ).astype(np.int64, copy=False)
        if target_slots.size != frame_misses:
            raise RuntimeError("cache replay did not produce one slot per miss")

        evicted_ids = slot_to_global[eviction_slots].copy()
        if evicted_ids.size:
            global_to_slot[evicted_ids] = -1
        miss_ids = ids[~hit_mask]
        slot_to_global[target_slots] = miss_ids
        global_to_slot[miss_ids] = target_slots
        ordered_slots = global_to_slot[ids]
        if np.any(ordered_slots < 0):
            raise RuntimeError("cache replay left a requested ID unmapped")
        last_used_frame[ordered_slots] = frame_index

        hits += frame_hits
        misses += frame_misses
        evictions += eviction_count
        zero_transfer_frames += int(frame_misses == 0)
        maximum_occupancy = max(
            maximum_occupancy, int(np.count_nonzero(slot_to_global >= 0))
        )

    requested = hits + misses
    hit_rate = hits / requested if requested else 1.0
    return AlodCacheSimulation(
        policy="bounded_cache",
        capacity_rows=int(capacity_rows),
        capacity_bytes=int(capacity_rows * property_bytes_per_row),
        frame_count=len(normalized),
        requested_rows=requested,
        cache_hits=hits,
        cache_misses=misses,
        hit_rate=hit_rate,
        miss_rate=1.0 - hit_rate,
        loads=misses,
        evictions=evictions,
        total_h2d_rows=misses,
        total_h2d_bytes=int(misses * property_bytes_per_row),
        maximum_occupancy_rows=maximum_occupancy,
        zero_transfer_frame_count=zero_transfer_frames,
    )


def simulate_previous_working_set(
    requests: Sequence[np.ndarray | Iterable[int]],
    *,
    property_bytes_per_row: int,
) -> AlodCacheSimulation:
    """Replay exact-row reuse from only the immediately preceding request."""

    normalized = [_ids(values) for values in requests]
    previous = np.empty(0, dtype=np.int64)
    hits = misses = zero_transfer_frames = maximum_occupancy = 0
    for ids in normalized:
        retained = int(np.intersect1d(ids, previous, assume_unique=True).size)
        frame_misses = int(ids.size - retained)
        hits += retained
        misses += frame_misses
        zero_transfer_frames += int(frame_misses == 0)
        maximum_occupancy = max(maximum_occupancy, int(ids.size))
        previous = ids
    requested = hits + misses
    hit_rate = hits / requested if requested else 1.0
    return AlodCacheSimulation(
        policy="previous_working_set",
        capacity_rows=None,
        capacity_bytes=None,
        frame_count=len(normalized),
        requested_rows=requested,
        cache_hits=hits,
        cache_misses=misses,
        hit_rate=hit_rate,
        miss_rate=1.0 - hit_rate,
        loads=misses,
        evictions=0,
        total_h2d_rows=misses,
        total_h2d_bytes=int(misses * property_bytes_per_row),
        maximum_occupancy_rows=maximum_occupancy,
        zero_transfer_frame_count=zero_transfer_frames,
    )


__all__ = [
    "AlodCacheSimulation",
    "AlodRequestTraceFrame",
    "characterize_request_trace",
    "simulate_bounded_cache",
    "simulate_previous_working_set",
]
