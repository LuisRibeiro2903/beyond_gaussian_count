# Results and implementation map

These thirteen files are unchanged original records, not a newly curated set
of measurements. `companion.json` records each source path and hash in the
complete v1.0.0 record. Detailed protocols, per-frame traces, validation gates,
rejected runs and historical context remain available there.

| Thesis topic | Key records | Main implementation |
| --- | --- | --- |
| CityGaussian controlled ladder | [Timing](citygaussian/controlled_ladder.json), [counts](citygaussian/gaussian_counts.json), [memory](citygaussian/memory.json) | `gs_framework/scene_runtime/citygaussian.py` |
| Released runtime and backend attribution | [Official comparison](citygaussian/official_comparison.json), [backend timing](citygaussian/backend_timing.json), [attribution](citygaussian/backend_attribution.json) | `gs_framework/benchmarks/benchmark_real_scene.py`; exact official harnesses in the full record |
| ALoD source-faithful evaluation | [Rendering](alod/rendering.json), [frustum ablation](alod/frustum_ablation.json), [quality](alod/quality.json) | `gs_framework/alod_reference/`, `gs_framework/scene_runtime/alod/`, and oracle validation scripts |
| Resource-feasibility negative finding | [Uni10K preflight closure](alod/uni10k_negative_preflight.json) | Full-record dataset/preflight tools; no fine-training or performance result is inferred |
| ALoD-Exec on the formal MatrixCity hierarchy | [Formal comparison](alod_exec/matrixcity_formal.json) | `gs_framework/scene_runtime/alod/materialization.py` and `residency.py` |
| Final natural workloads | [Campus/Sci-Art completion](alod_exec/natural_completion.json) | Same runtime and benchmark; `tools/verify_natural_results.py` recomputes 876 saved statistics |
| Ordered-multiset synthetic control | [Synthetic closure](alod_exec/synthetic_multiset.json) | Selection, materialization and residency modules; synthetic construction and campaign tools in the full record |

The MatrixCity weak-LoD setting, the natural workloads and the synthetic
control are different experiments. Their results must retain their original
labels and comparison boundaries. Historical and final execution states must
also remain distinct. Keep quality, selection counts, backend survivors,
render timing, allocator memory and whole-device memory separate.

The full record includes the topology-correction diagnosis, oracle parity,
cache-pressure analysis, excluded/superseded measurements and paper/source
divergences. The relevant topology patch remains directly available under
`gs_framework/patches/alod/`. Fewer loose files do not change the evidence or
remove the limitations behind the dissertation's claims.
