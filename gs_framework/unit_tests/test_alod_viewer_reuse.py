import unittest

import numpy as np

from scene_runtime.alod.types import (
    AlodSelection,
    AlodSelectionTimings,
    AlodSptCut,
    AlodUpperSelection,
)
from scene_runtime.alod.viewer_reuse import apply_viewer_previous_reuse


def selection(cuts):
    empty = np.empty(0, dtype=np.int32)
    upper = AlodUpperSelection(
        coarse_cut_upper_local_ids=empty,
        coarse_cut_global_node_ids=empty,
        cut_leaf_upper_local_ids=empty,
        cut_leaf_global_node_ids=empty,
        upper_render_global_node_ids=np.asarray([900], dtype=np.int32),
        candidate_spt_ids=np.asarray([cut.spt_id for cut in cuts], dtype=np.int32),
        candidate_spt_upper_local_ids=empty,
        candidate_spt_root_global_node_ids=empty,
        candidate_spt_distances=np.asarray([cut.distance for cut in cuts], dtype=np.float32),
        frustum_planes=np.empty((4, 4), dtype=np.float32),
        frustum_rejected_upper_local_ids=empty,
    )
    spt = np.concatenate([cut.emitted_global_node_ids for cut in cuts])
    hierarchy = np.concatenate((np.asarray([900], dtype=np.int64), spt))
    return AlodSelection(
        camera_stable_id="camera",
        mode="lod",
        frustum_enabled=True,
        upper=upper,
        per_spt=tuple(cuts),
        selected_spt_global_node_ids=spt,
        selected_hierarchy_global_node_ids=hierarchy,
        raster_global_node_ids=np.concatenate((np.arange(3), hierarchy)),
        timings=AlodSelectionTimings(),
    )


def cut(spt_id, distance, ids):
    return AlodSptCut(
        spt_id=spt_id,
        root_global_node_id=100 + spt_id,
        distance=np.float32(distance),
        interval_size=len(ids),
        emitted_global_node_ids=np.asarray(ids, dtype=np.int64),
    )


class AlodViewerReuseTest(unittest.TestCase):
    def test_reuses_compatible_spt_cut_and_preserves_viewer_order(self):
        previous = selection([cut(2, 10, [20, 21]), cut(1, 10, [10, 11])])
        current = selection([cut(1, 11, [12]), cut(3, 10, [30])])
        result = apply_viewer_previous_reuse(current, previous)
        self.assertEqual(result.reused_spts, 1)
        self.assertEqual(result.loaded_spts, 1)
        self.assertEqual([value.spt_id for value in result.selection.per_spt], [3, 1])
        np.testing.assert_array_equal(
            result.selection.selected_hierarchy_global_node_ids,
            [900, 30, 10, 11],
        )
        np.testing.assert_array_equal(result.reusable_global_node_ids, [10, 11])

    def test_strict_ratio_boundaries_are_not_reused(self):
        previous = selection([cut(1, 3, [10]), cut(2, 15, [20])])
        current = selection([cut(1, 10, [11]), cut(2, 10, [21])])
        result = apply_viewer_previous_reuse(current, previous)
        self.assertEqual(result.reused_spts, 0)
        np.testing.assert_array_equal(
            result.selection.selected_spt_global_node_ids, [11, 21]
        )


if __name__ == "__main__":
    unittest.main()
