# Comparison Validity

Classify every proposed table, plot, or ranking before extracting numbers.

## Direct controlled comparison

The dataset/scene, asset or representation where relevant, camera identities
and order, resolution/intrinsics, renderer, settings, hardware/software,
warmup/repeats, timing boundary, and metric implementation are the same. The
changed stage is explicit and the parent configuration is identified.

This supports causal language only for the isolated change. It does not make
the result portable to other assets or hardware.

## Partially controlled comparison

Some important variables differ, but the observation remains informative.
List every known mismatch adjacent to the result and use qualified language.
Do not calculate a single “improvement” percentage unless the metric and
denominator remain meaningful.

## Paper-reference comparison

Project measurements are compared with paper-reported values. Put them in
separate labelled columns and document differences in dataset, trained asset,
camera path, renderer/backend, hardware, software, cache policy, preprocessing,
and training budget. This is normally reproducibility analysis, not an
apples-to-apples performance ranking.

## Non-comparable observation

Use this when differences in dataset, model/hierarchy scale, representation,
camera regime, renderer, cache/runtime mode, or training/preprocessing are too
large for a quantitative ranking. Discuss mechanisms or qualitative behavior
without implying a winner.

## Required comparison record

For each comparison, record:

- child experiment/configuration and parent/baseline;
- classification from the four categories above;
- controlled variables;
- differing or unknown variables;
- canonical evidence paths for both sides;
- allowed conclusion and prohibited overclaim.

The existing configuration ladder and detailed protocol remain authoritative:

- [`../gs_framework/docs/EXPERIMENT_LADDER.md`](../gs_framework/docs/EXPERIMENT_LADDER.md)
- [`../gs_framework/docs/BENCHMARK_PROTOCOL.md`](../gs_framework/docs/BENCHMARK_PROTOCOL.md)

In particular, CityGaussian's released-LoD and fused branches use different
representations, and the current CityGaussian aerial asset and ALoD mixed-scale
formal proxy do not support an unconditional cross-method FPS or quality
ranking.
