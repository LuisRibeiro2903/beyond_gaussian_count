"""Released ALoD viewer previous-working-set SPT reuse semantics.

This module reproduces only the temporal reuse decision exposed by the pinned
``hierarchy_viewer.py``. It is separate from the project's exact-row reuse and
from every bounded cache policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import AlodSelection, AlodSelectionTimings, AlodSptCut


@dataclass(frozen=True)
class AlodViewerReuseResult:
    selection: AlodSelection
    reusable_global_node_ids: np.ndarray
    requested_spts: int
    reused_spts: int
    loaded_spts: int
    reused_rows: int
    loaded_spt_rows: int


def apply_viewer_previous_reuse(
    current: AlodSelection,
    previous: AlodSelection | None,
    *,
    closer_ratio: float = 0.3,
    farther_ratio: float = 1.5,
) -> AlodViewerReuseResult:
    """Apply the pinned viewer's strict SPT-distance compatibility test.

    Newly cut SPTs are emitted first in current request order, followed by
    compatible reused SPTs in previous-working-set order. Upper hierarchy rows
    are never classified as reusable by this component.
    """

    if current.mode != "lod" or current.upper is None:
        raise ValueError("viewer previous reuse requires an LoD/HSPT selection")
    if closer_ratio < 0.0 or farther_ratio <= closer_ratio:
        raise ValueError("invalid viewer reuse distance-ratio bounds")

    previous_by_spt: dict[int, AlodSptCut] = {}
    previous_order: list[int] = []
    if previous is not None:
        for cut in previous.per_spt:
            previous_by_spt[int(cut.spt_id)] = cut
            previous_order.append(int(cut.spt_id))

    reusable: dict[int, AlodSptCut] = {}
    loaded: list[AlodSptCut] = []
    for cut in current.per_spt:
        old = previous_by_spt.get(int(cut.spt_id))
        if old is None or float(cut.distance) == 0.0:
            loaded.append(cut)
            continue
        ratio = float(old.distance) / float(cut.distance)
        if closer_ratio < ratio < farther_ratio:
            reusable[int(cut.spt_id)] = old
        else:
            loaded.append(cut)

    reused = [reusable[spt_id] for spt_id in previous_order if spt_id in reusable]
    ordered_cuts = tuple(loaded + reused)
    selected_spt = (
        np.ascontiguousarray(
            np.concatenate([cut.emitted_global_node_ids for cut in ordered_cuts]),
            dtype=np.int64,
        )
        if ordered_cuts
        else np.empty(0, dtype=np.int64)
    )
    upper_ids = np.ascontiguousarray(
        current.upper.upper_render_global_node_ids, dtype=np.int64
    )
    selected_hierarchy = np.ascontiguousarray(
        np.concatenate((upper_ids, selected_spt)), dtype=np.int64
    )
    skybox_rows = (
        current.raster_global_node_ids.size
        - current.selected_hierarchy_global_node_ids.size
    )
    raster = np.ascontiguousarray(
        np.concatenate((np.arange(skybox_rows, dtype=np.int64), selected_hierarchy)),
        dtype=np.int64,
    )
    reusable_ids = (
        np.ascontiguousarray(
            np.concatenate([cut.emitted_global_node_ids for cut in reused]),
            dtype=np.int64,
        )
        if reused
        else np.empty(0, dtype=np.int64)
    )
    selection = AlodSelection(
        camera_stable_id=current.camera_stable_id,
        mode=current.mode,
        frustum_enabled=current.frustum_enabled,
        upper=current.upper,
        per_spt=ordered_cuts,
        selected_spt_global_node_ids=selected_spt,
        selected_hierarchy_global_node_ids=selected_hierarchy,
        raster_global_node_ids=raster,
        timings=AlodSelectionTimings(
            camera_preparation_seconds=current.timings.camera_preparation_seconds,
            upper_traversal_seconds=current.timings.upper_traversal_seconds,
            spt_cut_seconds=current.timings.spt_cut_seconds,
            total_selection_seconds=current.timings.total_selection_seconds,
        ),
    )
    return AlodViewerReuseResult(
        selection=selection,
        reusable_global_node_ids=reusable_ids,
        requested_spts=len(current.per_spt),
        reused_spts=len(reused),
        loaded_spts=len(loaded),
        reused_rows=int(reusable_ids.size),
        loaded_spt_rows=int(sum(cut.emitted_global_node_ids.size for cut in loaded)),
    )


__all__ = ["AlodViewerReuseResult", "apply_viewer_previous_reuse"]
