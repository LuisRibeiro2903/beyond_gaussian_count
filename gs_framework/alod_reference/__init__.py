"""CPU-only correctness reference for ALoD HSPT construction and selection.

This package is independent of the official runtime, CUDA extensions, gsplat,
CityGaussian, rendering, residency, and oracle comparison tooling.
"""

from .hspt import build_hspt
from .frustum import (
    extract_four_frustum_planes,
    frustum_visible_spheres,
    scale_derived_bounds,
)
from .selection import cut_candidate_spts, select_reference, traverse_upper_hierarchy
from .types import (
    HSPT,
    HierarchyData,
    ReferenceCamera,
    ReferenceConfig,
    SelectionResult,
    SPTCutResult,
    UpperTraversalResult,
)

__all__ = [
    "HSPT",
    "HierarchyData",
    "ReferenceCamera",
    "ReferenceConfig",
    "SelectionResult",
    "SPTCutResult",
    "UpperTraversalResult",
    "build_hspt",
    "cut_candidate_spts",
    "extract_four_frustum_planes",
    "frustum_visible_spheres",
    "scale_derived_bounds",
    "select_reference",
    "traverse_upper_hierarchy",
]
