# ALoD Paper/Public-Source Comparison

PAPER-SOURCE-DIVERGENCE: no released rendering entry point exactly implements the paper-reported full bounded rendering-cache configuration.

| Mode | Paper Table 3, four reported scenes (ms) | This campaign | Fidelity / comparability |
| --- | --- | ---: | --- |
| Render Full | 48.1 / 68.3 / 47.1 / 83.2 | unavailable | No exact released rendering entry point; not fabricated. |
| w/o cache | 119.4 / 127.4 / 92.6 / 222.3 | 186.77 ms end-to-end | Closest public evaluator-style no-cache proxy; different hierarchy, cameras, training envelope, and GPU. |
| w/o frustum | 52.3 / 89.3 / 38.3 / 110.2 | 188.02 ms end-to-end | Source component ablation on no-cache proxy; paper row retained its unreleased cache, so numerical comparison is invalid. |
| HSPT cut | 31.9 / 45.7 / 31.3 / 36.5 | 36.00 ms | Exact released evaluator cut, diagnostic cameras; trend-only. |
| BFS cut | 47.8 / OOM / 40.0 / 53.7 | 91.54 ms | Minimal source-faithful reconstruction; no released BFS executable; trend-only. |

The local results must not be used as direct paper-number reproduction: assets, dataset scale, trained hierarchy, cameras, rendering harness, and GPU differ, and some paper configurations were not released.
