# Paper Implementation Task Template

Use this template for future faithful reproductions or integrations of published large-scale 3D Gaussian Splatting rendering/viewer methods.

## Objective

State the exact implementation goal in one paragraph.

## Thesis Relevance

Explain which thesis question or metric comparison this task supports.

## Current State

Summarize the relevant current repo state and cite files/classes/functions that will be touched.

## Paper And Reference Repository

- Paper title:
- Authors/year:
- Paper URL/DOI:
- Reference code URL and commit/tag:
- License notes:

## Reproduction Scope

Define which method components must be reproduced faithfully and which are out of scope.

## Training Scope

State whether training changes are required. If not, say explicitly that existing checkpoints/assets are used.

## Offline Construction Scope

List required asset conversion, preprocessing, metadata construction, or checkpoint preparation steps.

## Runtime Scope

Describe viewer/render-time behavior to implement and where it connects to `scene_runtime`, `GsplatRenderer`, or another renderer boundary.

## Explicitly Excluded Work

List tempting but excluded work, such as optimizer changes, unrelated refactors, CUDA rewrites, or dataset migration.

## Methodology Steps

Ordered implementation steps, each with expected files and validation.

## Mapping To Reference Code

Map paper/reference-code symbols to local modules/classes/functions.

## Permitted Deviations

Document allowed deviations and why they do not invalidate the reproduction.

## Required Inputs

List datasets, checkpoints, camera paths, external repositories, and local config required to run.

## Architecture Touchpoints

Identify exact current interfaces to use, such as `SceneLoader`, `GaussianScene.prepare_frame`, `ActiveSplats`, `RenderSettings`, `GsplatRenderer.render`, or metric logging.

## Invariants

List behavior that must remain true, such as flat rendering independence, no hidden dependency on the experimental block-based host-residency baseline, and no generated assets committed.

## Failure Cases

Describe expected failures, validation errors, unsupported assets, and graceful degradation.

## Performance Constraints

State target view sizes, frame budgets, VRAM limits, and metrics that must be collected.

## Acceptance Criteria

List concrete completion criteria, including visual and quantitative evidence.

## Required Tests

List unit, smoke, import, fixture, or regression tests required before completion.

## Required Benchmark Evidence

List benchmark commands, inputs, metrics, and comparison baselines. Mark any benchmark that depends on external assets.

## Required Completion Report

The final report must include:

- files changed;
- method components implemented;
- deviations from the paper/reference;
- tests and benchmark evidence;
- unresolved accuracy/performance risks;
- confirmation that generated data/checkpoints/logs were not committed.
