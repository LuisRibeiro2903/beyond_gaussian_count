# CityGaussian backend microbenchmark

Campaign ID: `citygs_backend_microbench_20260824_213420`  
Status: **complete**  
Evidence class: `CITYGS-BACKEND-MICROBENCH-*`

## Outcome

B — the direct gap is materially smaller than the prior flat runtime gap. The previous high-level runtime gap cannot yet be attributed primarily to the rasterizer.

This is a direct same-input rasterizer comparison, separate from both the canonical gsplat CityGaussian campaign and the official-runtime parity campaign. It is not viewer FPS and does not reproduce paper numbers.

## Test cases and protocol

- `finest_flat`: 11,694,810 Gaussians from the exact finest released SH2 PLY, camera index 392.
- `representative_lod`: 3,823,434 preassembled Gaussians from the validated released visibility/Nyquist selector, camera index 392.
- Resolution: 1920×1080; black background; scale modifier 1; SH degree 2.
- Three fresh processes per backend/case, ten excluded warmup renders, 100 measured renders per process.
- Repeat ordering: GraphDECO→gsplat, gsplat→GraphDECO, GraphDECO→gsplat.
- Primary timing is CUDA-event `backend_gpu_time_ms`; secondary timing is synchronized `backend_synchronized_wall_time_ms`.

## Direct timing

| Case | Backend | Gaussian count | GPU mean ± process SD (ms) | Wall mean ± process SD (ms) | Backend throughput equivalent (s⁻¹) |
| --- | --- | ---: | ---: | ---: | ---: |
| `representative_lod` | graphdeco | 3,823,434 | 8.989 ± 0.229 | 9.627 ± 0.265 | 111.25 |
| `representative_lod` | gsplat | 3,823,434 | 6.652 ± 0.021 | 7.177 ± 0.089 | 150.33 |
| `finest_flat` | graphdeco | 11,694,810 | 10.769 ± 0.052 | 11.508 ± 0.087 | 92.86 |
| `finest_flat` | gsplat | 11,694,810 | 9.738 ± 0.267 | 10.410 ± 0.289 | 102.70 |

## Relative speed

| Case | GraphDECO GPU ms | gsplat GPU ms | GraphDECO / gsplat |
| --- | ---: | ---: | ---: |
| `representative_lod` | 8.989 | 6.652 | 1.35× |
| `finest_flat` | 10.769 | 9.738 | 1.11× |

## Input and image gate

| Case | Count/hash/camera equal | Raw RGB PSNR (dB) | Clipped SSIM | Mean raw RGB diff | Coverage Jaccard |
| --- | --- | ---: | ---: | ---: | ---: |
| `representative_lod` | yes | 77.461 | 0.999998093 | 9.1376132e-06 | 1.000000000 |
| `finest_flat` | yes | 77.973 | 0.999998271 | 9.1285128e-06 | 1.000000000 |

## Reconciliation with the earlier timing boundary

| Case | Backend | Earlier camera-392 CUDA boundary (ms) | Direct backend GPU time (ms) | Boundary minus direct (ms) |
| --- | --- | ---: | ---: | ---: |
| `representative_lod` | graphdeco | 10.507 | 8.989 | 1.518 |
| `representative_lod` | gsplat | 6.719 | 6.652 | 0.067 |
| `finest_flat` | graphdeco | 23.751 | 10.769 | 12.982 |
| `finest_flat` | gsplat | 10.029 | 9.738 | 0.292 |


The earlier official flat CUDA event began before the high-level `render()` function. Source inspection shows that boundary included creation/zeroing of the full N×3 screen-space tensor and per-frame `get_opacity` sigmoid, `get_scaling` exponentiation, `get_rotation` normalization, and `get_features` concatenation before `GaussianRasterizer.__call__`. The 11.69M camera-392 boundary is therefore 23.751 ms versus 10.769 ms for the direct preactivated/preconstructed call. By contrast, official adaptive-LoD instrumentation placed its CUDA event after active-set assembly and attribute extraction, immediately around `GaussianRasterizer`; its earlier 10.507 ms camera-392 value is correspondingly close to the direct 8.989 ms result. Canonical gsplat already held activated scene tensors and shows much smaller boundary-to-direct differences.

These paired values are not an exact subtraction-based component profile: the earlier samples were encountered during a moving-camera trajectory, whereas the microbenchmark repeatedly renders one warmed camera. They do, however, identify timing-boundary contamination in the flat official control and disprove treating its old CUDA field as a pure GraphDECO-kernel measurement. Camera-dependent tile/sort workload across the full trajectory is a remaining explanation for the old 40.968 ms official-LoD average and was not re-benchmarked here.


Every backend repeat re-hashed the same serialized activated tensors before upload. Camera validation confirms that the official transposed world-view matrix is exactly the transpose of the gsplat world-to-camera matrix and that both derive from the already validated parity conversion. Output RGB and alpha are finite. GraphDECO alpha is obtained by one extra untimed constant-one render because the pinned API does not return alpha.

## Exact logical-input mapping

| Logical attribute | gsplat input | GraphDECO input | Outside-timing conversion |
| --- | --- | --- | --- |
| Means / xyz | `means` N×3 | `means3D` N×3 | none after shared float32 tensor load |
| Scale | activated positive `scales` N×3 | activated positive `scales` N×3 | PLY log-scale exponentiation before serialization |
| Rotation | normalized wxyz `quats` N×4 | normalized wxyz `rotations` N×4 | normalization before serialization |
| Opacity | activated vector N | activated column N×1 | PLY sigmoid before serialization; GraphDECO column view only |
| SH | `colors` N×9×3, degree 2 | `shs` N×9×3, degree 2 | channel-major PLY rest coefficients mapped once before serialization |
| Camera | conventional w2c plus K | official transposed world-view and full projection | validated parity conversion before serialization |
| Image / background | 1920×1080, black | 1920×1080, black | static setup before timing |
| Scale modifier | implicit 1 | explicit 1 | static setup before timing |

Both backends internally perform covariance construction, SH evaluation, camera-space projection, frustum rejection, tile overlap generation, duplicated-instance sorting, and alpha compositing. The gsplat public result exposes positive radii, per-Gaussian tile intersections, and sorted intersection arrays. The pinned GraphDECO public result exposes radii but not tile/sort counts; obtaining those would require invasive API/kernel instrumentation, which this task forbids. Therefore this experiment isolates implementation-level rasterizer cost but cannot reduce the difference to identical internal work or kernel efficiency alone.

## Build verification

`environment/graphdeco_build.txt` records native `sm_120` cubins, no PTX fallback, `-DNDEBUG`, host `-O2`, CUDA `-gencode=arch=compute_120,code=sm_120`, and no `-G`. `environment/gsplat_build.txt` records native `sm_120` cubins plus compute_120 PTX, host/CUDA `-O3`, and no `-G`. GraphDECO remains pinned at `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d` with only the previously recorded `<cstdint>` include compatibility change. Neither backend or kernel was optimized for this campaign.

One initial gsplat setup result is retained under `validation/superseded_setup_runs/` but excluded from every table and validation total because it preceded the unchanged-source sm_120 rebuild and its exact loaded binary hash was not preserved. The formal replacement repeat and all other gsplat measurements loaded the binary hash recorded in `manifest.json`.

## Memory and thermal limits

Allocator and whole-device observations are in `tables/memory_observations.json`. PyTorch allocator values are secondary because custom-extension allocation semantics are not representation-neutral. NVML temperature/clock fields are retained when WSL exposed them; missing fields remain an environment limitation rather than being inferred.

## Evidence map

- `manifest.json`: identities, hashes, protocol, and interpretation.
- `cases/`: raw 100-sample JSONL, repeat summaries, provenance, and output arrays for all 12 fresh processes.
- `validation/*_input.json`: deterministic activated-tensor statistics/hashes, exact selection record, camera checks, and asset hashes.
- `validation/package_validation.json`: completeness and sanity gate.
- `tables/backend_timing.json`: within-process distributions and across-process mean/SD/CV.
- `tables/relative_speed.json`: direct GraphDECO/gsplat ratios.
- `tables/input_output_parity.json`: RGB, alpha, PSNR, SSIM, and coverage comparisons.
- `tables/internal_workload_diagnostics.json`: accessible backend diagnostic counts and limitations.
- `figures/`: direct GPU time and speed-ratio plots.

## Boundaries

Loading, selection, active-set assembly, concatenation, activation, camera construction, host-to-device transfer, rasterizer/static setup, warmup, diagnostics, alpha sanity rendering, image transfer, and serialization are excluded. The gsplat far-plane argument remains the canonical reproduction value (1e10), while the pinned GraphDECO projection helper stores 100; this is the already validated parity setup and the images pass the sanity gate. The direct operation still includes each public backend's Python/CUDA call chain and required output allocation. Internal rasterization workloads are implementation-dependent and only partially observable.

The pinned reference checkout `/home/luis/tese/CityGaussian`, Gaussian assets, trajectory, selection semantics, and thesis LaTeX were not modified.
