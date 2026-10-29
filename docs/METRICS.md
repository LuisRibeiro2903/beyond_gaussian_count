# Metric glossary

The saved records preserve their original field names. Use the definitions
below when reading results or running `benchmarks.benchmark_real_scene`.
Historical campaign tables can use more explicit names than the current
benchmark; their archived protocols remain authoritative for those runs.

## Timing and FPS

| Field | Meaning |
| --- | --- |
| `preparation_time_ms` | Synchronized wall time around `scene.prepare_frame()`: selection, materialization, residency operations and assembly used by that scene. |
| `frame_time_ms` | Wall time measured inside the renderer call. It excludes scene preparation, despite the historical name. The canonical CityGaussian campaign calls this `render_call_time_ms`. |
| `fps` | `1000 / frame_time_ms` for one frame: renderer-call throughput. |
| `end_to_end_frame_time_ms` | `preparation_time_ms + frame_time_ms`. Use this to include scene preparation when comparing execution policies. |
| `end_to_end_fps` | `1000 / end_to_end_frame_time_ms` for one frame. |
| `rasterization_time_ms` | CUDA event interval inside the renderer when `--log-gpu-events` is enabled. Otherwise unavailable. The canonical CityGaussian campaign calls this `cuda_rasterization_time_ms`. |

The CUDA event interval is nested within the renderer-call interval. Do not
add it to renderer-call time. These intervals do not include the complete
application loop: camera construction, some metric collection, image checks,
logging and startup fall outside the preparation-plus-render sum. They are
not measurements of interactive viewer FPS.

Summary fields `fps_from_mean_frame_time` and
`end_to_end_fps_from_mean_frame_time` divide 1000 by the corresponding mean
time. They are not the arithmetic mean of the per-frame FPS values. The
canonical CityGaussian table uses `headline_mean_end_to_end_frame_time_ms`
and `end_to_end_fps` for its aggregate. Preserve each campaign's aggregation
and repetition policy.

## Gaussian populations

| Field | Meaning |
| --- | --- |
| `total_gaussians` | Total Gaussian rows in the loaded scene asset, before per-camera selection. |
| `selected_gaussian_count` / `alod_semantic_selected_rows` | Rows selected by the scene's semantic selection policy, before the optional materialization filter. ALoD's total includes selected skybox rows. |
| `active_gaussians` / `alod_materialization_rows` | Rows assembled as renderer input after any materialization filter. ALoD preserves ordered duplicate occurrences. |
| `alod_unique_semantic_rows` / `alod_unique_materialization_rows` | Distinct stable global IDs in the corresponding ordered population. These counts can be smaller than the occurrence counts. |
| `backend_projection_survivor_count` / `visible_gaussians` | Backend projection survivors: packed Gaussian IDs, or positive-radii rows in unpacked mode. These are not a count of Gaussians proven to contribute to final pixels. |
| `rendered_gaussian_count` | Historical alias of `backend_projection_survivor_count` in the current benchmark. It does not measure a separate final-pixel population. |
| `resident_gaussian_count` / `resident_gaussians` | Rows retained by the scene's GPU residency policy, which can include rows unused by the current frame. It is neither allocated cache capacity nor a VRAM measurement. |

Packed survivor counting deduplicates backend input-row IDs. Duplicate stable
global IDs can still occupy different input rows. In ALoD-Ext, the cache can
store unique property rows while frame assembly reconstructs the ordered
occurrences. Keep those two populations separate when interpreting cache and
renderer counts.

## Memory and transfer

| Field | Meaning |
| --- | --- |
| `torch_current_allocated_bytes` / `torch_peak_allocated_bytes` | Live tensor allocations / peak allocations tracked by PyTorch at the recorded stage. |
| `torch_current_reserved_bytes` / `torch_peak_reserved_bytes` | Current / peak memory reserved by the PyTorch allocator. Reserved bytes can exceed live allocations. |
| `nvml_device_used_bytes` | Whole-device used VRAM, including allocations outside this process and PyTorch. |
| `nvml_process_used_bytes` | Process VRAM reported by NVML when available; inspect the accompanying accounting-support and reliability flags. |
| `process_ram_bytes` | Process resident set size (RSS), not system-wide RAM usage. |
| `alod_h2d_rows` / `alod_h2d_bytes` | Dynamic hierarchy-property rows / bytes transferred from host to GPU by the residency policy for that frame. |

Dynamic-property H2D excludes other transfer categories such as startup
skybox transfer and compact metadata or selected-ID transfer. It is not total
PCIe traffic. Likewise, logical property-storage bytes do not include all
renderer buffers, temporary tensors or allocator reservations.

Frame and stage memory peaks have explicit reset boundaries. Use
`scene_construction_peak`, `render_peak`, `whole_run_peak` and the archived
protocol to identify their scope. A per-frame peak is not a process-lifetime
peak. Missing or unreliable measurements must remain unavailable, not zero.

Warmup transfer and measured-frame transfer are separate in the saved ALoD
tables. A cold-inclusive total includes both; a steady-state total includes
only measured frames. Preserve that distinction when reporting cache reuse.

Implementation references: [frame metrics and summaries](../gs_framework/benchmarks/benchmark_real_scene.py),
[renderer timing and survivor counting](../gs_framework/scene_runtime/renderer.py),
and [ALoD population and residency accounting](../gs_framework/scene_runtime/alod/host_scene.py).
