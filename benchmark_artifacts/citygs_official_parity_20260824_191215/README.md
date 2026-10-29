# CityGaussian V1 official-runtime parity campaign

Campaign ID: `citygs_official_parity_20260824_191215`  
Status: **complete**  
Evidence class: `CITYGS-OFFICIAL-PARITY-*`

## Outcome

The released CityGaussian V1 runtime was executed headlessly through its direct PLY path with the exact three released LoD PLY representations, official `BlockedGaussian` cell construction, official projected-corner visibility, released strict-first-match Nyquist LoD, released active-set concatenation, and the original GraphDECO rasterizer. Camera parity passed and selected-set parity was exact for all 12 frozen oracle cameras. All six formal fresh processes completed.

This campaign is an external validation of the existing gsplat reproduction. It does not replace the canonical gsplat performance package `citygs_canonical_20260824_135615` and does not reproduce paper FPS.

## Official source call chain

`official_parity_benchmark.py` -> `GaussianModelLOD.load_ply()` -> `BlockedGaussian.__init__()` / `cell_divider()` -> `gaussian_renderer.render_lod()` -> `utils.large_utils.in_frustum()` -> released Nyquist `torch.max` first-match selection with coarsest fallback -> `BlockedGaussian.get_feats()` and `torch.cat()` -> `diff_gaussian_rasterization.GaussianRasterizer`.

The released `render_large_lod.py` reaches the same PLY path when `--load_vq` is absent (`LargeScene` -> `GaussianModelLOD.load_ply`). Its VQ path is separate (`--load_vq` / viewer YAML -> `GaussianModelLOD.load_vq()` -> `utils.vq_utils.load_vqgaussian()`). The interactive `viewer.py` YAML path loads VQ levels and adds Viser callbacks, resolution switching, image CPU transfer/encoding, and event-loop work, so it was not used for controlled timing.

The released `render_large_lod.py` has a synchronized whole-render timer but writes images inside the rendering loop. The interactive viewer has no suitable formal end-to-end timing boundary. This harness therefore instruments the released runtime decision points and excludes GUI, I/O, checks, and serialization.

## Validation

- Camera gate: pass. World-view, projection, and full-projection maximum matrix differences are exactly 0.0; derived float32 camera-center maximum difference is 1.91e-6.
- Semantic gate: 12/12 exact for visible cells, LoD indices/labels, per-cell and per-level selected counts, total selected count, nearest cell, and fallback.
- PLY parity: supported and used for the principal comparison; VQ was not used.
- Formal package: 6 fresh processes, 8,640 measured frames, exact trajectory order, finite RGB throughout.
- Cross-backend images: all finite; mean raw RGB absolute difference 0.00001932, mean clipped SSIM 0.99999680, mean coverage Jaccard 0.999999959. These are implementation-similarity metrics, not ground-truth quality.

## Performance result

Across three repetitions, official released-LoD end-to-end time is 52.516 ± 3.337 ms (19.04 FPS). Canonical gsplat `CG-LOD-RELEASE` is 15.673 ± 0.738 ms (63.80 FPS).

Official LoD selection is 1.644 ms, assembly is 7.873 ms, and render call is 42.999 ms. Canonical gsplat selection is 2.445 ms, assembly is 5.607 ms, and render call is 7.547 ms.

The same-asset flat control is 52.835 ms (18.93 FPS) for GraphDECO versus 11.360 ms (88.03 FPS) for gsplat. Both preparation and rendering differ in the full runtime, but the render-call difference accounts for most of the observed end-to-end gap, and the flat control independently supports a large backend contribution. No unmeasured kernel-level cause is asserted.

## Environment and compatibility

The isolated environment is `/home/luis/miniconda3/envs/citygs` with PyTorch 2.11.0+cu128 (compiled CUDA 12.8). Exact pinned extensions were rebuilt for `sm_120` with CUDA 12.9.86. CUDA 12.9 required two include-only compatibility patches: `<cstdint>` in GraphDECO `rasterizer_impl.h` and `<cfloat>` in `simple_knn.cu`. These do not change algorithms and are included in `environment/instrumentation.patch`.

## Evidence map

- `manifest.json`: complete identity, hashes, environment, timing, and run order.
- `validation/camera_parity.json`, `validation/selected_set_parity.json`, `validation/package_validation.json`: formal gates.
- `runs/`: raw frame JSONL and per-repeat summaries.
- `tables/official_config_summary.json`: official three-run aggregates.
- `tables/flat_backend_comparison.json`: same-asset backend control.
- `tables/released_lod_comparison.json`: full-runtime comparison and attribution.
- `tables/timing_decomposition.json`, `tables/memory_comparison.json`: phase and memory evidence.
- `comparisons/cross_backend_image_comparison.json`: 12-camera RGB/alpha metrics.
- `figures/`: thesis-ready comparisons.
- `environment/`: package list, CUDA/toolchain record, pinned commits, and complete patch.

## Boundaries and limitations

The official Camera/MiniCam is constructed before `render_lod`, matching released call boundaries; canonical project preparation constructs its official-camera representation inside `prepare_frame`. This retained architectural difference affects fine-grained selection timing. PyTorch allocator counters are not representation-neutral across backends, and whole-device memory is sampled after runs rather than continuously peaked. Official alpha is derived by a second non-timed constant-one render because this GraphDECO revision does not return alpha. No VQ, training, GUI FPS, paper camera path, or paper result reproduction is claimed.

Neither runtime was optimized. The pinned reference checkout `/home/luis/tese/CityGaussian`, the canonical gsplat source, trajectory, PLY assets, and dissertation files were not modified.
