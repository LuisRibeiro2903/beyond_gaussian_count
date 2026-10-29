# Excluded setup run

`representative_lod_gsplat_repeat_01_pre_rebuild/` preserves the first setup-time
result for audit only. It is excluded from all aggregate tables, figures, and
the validated 1,200-sample total because it ran before the unchanged-source
gsplat CUDA 12.9 `sm_120` rebuild and its exact loaded binary hash was not
preserved.

The formal `representative_lod/gsplat/repeat_01/` replacement was collected in
a fresh process after the rebuild and loaded the gsplat binary whose SHA-256 is
recorded in the campaign manifest.
