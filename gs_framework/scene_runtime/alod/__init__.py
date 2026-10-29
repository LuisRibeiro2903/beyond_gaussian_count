"""Production runtime for A LoD of Gaussians."""

from .cache_simulation import (
    AlodCacheSimulation,
    AlodRequestTraceFrame,
    characterize_request_trace,
    simulate_bounded_cache,
    simulate_previous_working_set,
)
from .camera import selection_camera_from_frame
from .hspt import build_production_hspt
from .resident import AlodResidentProperties
from .host import AlodHostProperties
from .host_scene import AlodHostScene
from .materialization import (
    AlodCompactCullingMetadata,
    AlodDeviceCullingMetadata,
    AlodMaterializationSelection,
    AlodMaterializationTimings,
    conservative_pinhole_materialization,
    fast_conservative_pinhole_materialization,
    identity_materialization,
    select_materialization,
    select_multiset_materialization,
    stable_unique_with_inverse,
)
from .residency import (
    AlodBoundedCache,
    AlodCacheCapacityError,
    AlodHostNoCache,
    AlodPreviousWorkingSetReuse,
    AlodViewerPreviousWorkingSetReuse,
)
from .selection import cut_spts, select_production, traverse_upper
from .types import (
    AlodHierarchySource,
    AlodHostResidencyConfig,
    AlodHspt,
    AlodRuntimeConfig,
    AlodSelection,
    AlodSelectionCamera,
)
from .viewer_reuse import AlodViewerReuseResult, apply_viewer_previous_reuse

__all__ = [
    "AlodHierarchySource",
    "AlodCacheSimulation",
    "AlodCompactCullingMetadata",
    "AlodDeviceCullingMetadata",
    "AlodHostProperties",
    "AlodHostResidencyConfig",
    "AlodHostScene",
    "AlodHspt",
    "AlodBoundedCache",
    "AlodCacheCapacityError",
    "AlodHostNoCache",
    "AlodPreviousWorkingSetReuse",
    "AlodViewerPreviousWorkingSetReuse",
    "AlodResidentProperties",
    "AlodRuntimeConfig",
    "AlodMaterializationSelection",
    "AlodMaterializationTimings",
    "AlodSelection",
    "AlodSelectionCamera",
    "AlodViewerReuseResult",
    "AlodRequestTraceFrame",
    "build_production_hspt",
    "characterize_request_trace",
    "cut_spts",
    "conservative_pinhole_materialization",
    "fast_conservative_pinhole_materialization",
    "identity_materialization",
    "select_production",
    "apply_viewer_previous_reuse",
    "selection_camera_from_frame",
    "select_materialization",
    "select_multiset_materialization",
    "stable_unique_with_inverse",
    "simulate_bounded_cache",
    "simulate_previous_working_set",
    "traverse_upper",
]
