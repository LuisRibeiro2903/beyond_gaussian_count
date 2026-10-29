"""Read and validate project-owned official ALoD selection-oracle artifacts.

This package contains no hierarchy construction or selection implementation.
It only defines deterministic serialization, hashing, and validation helpers for
artifacts emitted by the pinned official runtime.
"""

from .format import (
    CAMERA_SCHEMA_VERSION,
    HSPT_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    PRE_RASTER_SCHEMA_VERSION,
    SPT_CUT_SCHEMA_VERSION,
    UPPER_TRAVERSAL_SCHEMA_VERSION,
    OracleValidationError,
    array_descriptor,
    canonical_json,
    canonicalize_global_ids,
    global_id_hash,
    load_expected_selected_global_ids,
    load_manifest,
    save_array_payload,
    sha256_file,
    validate_global_ids,
    validate_hspt_record,
    validate_hspt_payload_mappings,
    validate_manifest,
    validate_oracle_tree,
    validate_pre_raster_record,
    validate_record_payloads,
)

__all__ = [
    "CAMERA_SCHEMA_VERSION",
    "HSPT_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "PRE_RASTER_SCHEMA_VERSION",
    "SPT_CUT_SCHEMA_VERSION",
    "UPPER_TRAVERSAL_SCHEMA_VERSION",
    "OracleValidationError",
    "array_descriptor",
    "canonical_json",
    "canonicalize_global_ids",
    "global_id_hash",
    "load_expected_selected_global_ids",
    "load_manifest",
    "save_array_payload",
    "sha256_file",
    "validate_global_ids",
    "validate_hspt_record",
    "validate_hspt_payload_mappings",
    "validate_manifest",
    "validate_oracle_tree",
    "validate_pre_raster_record",
    "validate_record_payloads",
]
