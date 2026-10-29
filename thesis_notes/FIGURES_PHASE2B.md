# Dissertation figures: Phase 2B

Completes the ALoD and ALoD-Exec architecture batch after Phase 2A. Four
original vector schematics explain Chapters 4 and 5. No training, rendering
experiment, GPU benchmark, performance plot, or numerical result was changed.

## Completed inventory

| Figure | Label | PDF under thesis_latex/images/alod/ | Printed / PDF page |
| --- | --- | --- | --- |
| 12: hierarchy and selection | fig:alod-hspt-selection | hspt_selection.pdf | 59 / 79 |
| 13: evaluator and viewer | fig:alod-released-execution | released_execution.pdf | 62 / 82 |
| 14: execution extension | fig:alod-exec-pipeline | exec_pipeline.pdf | 95 / 115 |
| 15: residency and occurrences | fig:alod-exec-residency-occurrences | residency_occurrences.pdf | 97 / 117 |

The two comment-only Chapter 4 architecture placeholders were replaced; they
had no existing figure labels. Chapter 5 gains the principal architecture
figure and one supporting diagram. The fourth is retained because distinguishing
physical properties from repeated renderer occurrences would overload Figure
14. Its policy comparison also makes the difference between previous-frame
and bounded reuse visible before the evaluation. All other labels remain.

The Phase 2A NewsGotT font, blue/orange palette, and line styles are reused
through the existing methodology_style.tex. Blue denotes released semantics
and shared data; green identifies ALoD-Exec operations; orange denotes the
backend. Darker blue boxes identify persistent data and dashed blue arrows
show its use. CPU/GPU boundaries are explicitly labelled. Labels are 9 pt,
headings 10 pt and supporting labels 8 pt at approximately 160 mm print width.
The hierarchy, camera distances, ellipses and letter IDs are illustrative,
not measurements or redrawings of published artwork.

## Primary source and semantic review

Read the local paper thesis_notes/papers/ALoD.pdf, particularly the hierarchy,
SPT/HSPT and caching sections and Figures 2--4. Captions cite existing keys
windisch2026alod and alodSource. The method card was used for discovery, not
as a substitute for the paper or source (its older HSPT expansion differs
from the current chapter). No bibliography entry was edited.

The generator verifies that these local released files exactly equal blobs
at LoDOfGaussians commit 73f547341547ae021ff036ad74fdf73a1240b450:

- scene/gaussian_model.py: binary hierarchy, volume split, sufficiently large
  SPTs, residual explicit subtrees, distance limits, descending upper-limit
  order and parent-derived upper-tree distance criterion.
- submodules/gaussianhierarchy/runtime_switching.cu: released binary-search
  interval, strict lower-limit test, and ordered nonzero compaction. The
  figure does not replace these details with an ideal octree or a conventional
  cut across fixed tree depths.
- eval_hierarchy.py: GPU selection, CPU property gather, full dynamic H2D,
  skybox prefix reuse, attribute activation and assembly.
- hierarchy_viewer.py: requested SPTs reuse stored cuts only for strict
  0.3 < stored_distance/current_distance < 1.5; retained distances refer to
  the cut's load distance, not necessarily the immediately preceding camera.
  Ordinary upper rows are freshly loaded. Unrequested SPT cuts are dropped.
  This can change the selected occurrence list relative to fresh evaluation.
- gaussian_renderer/__init__.py: shape/appearance formatting followed by
  packed gsplat projection and subsequent tile intersection/sorting/blending.

Project sources reviewed: scene_runtime/alod/{selection,hspt,host,host_scene,
materialization,residency,viewer_reuse}.py, scene_runtime/renderer.py, and
benchmarks/benchmark_real_scene.py. The key distinction is explicit:
Figure 13 depicts released GPU selection, whereas Figure 14 depicts the
project's CPU selector with the same validated semantics. The project prepares
activated host attributes at initialization. Its synchronized timing strip
in Figure 13 is project instrumentation, not a claim about original timers.
Preparation includes selection through assembled input; renderer time starts
after camera/background device preparation and ends after synchronization.
Nested CUDA timing is not another additive stage. Initialization, warm-up,
checks and writes remain outside the measured frame as already documented.

The final extension first builds stable unique semantic IDs and an inverse
on the CPU, sends unique IDs to the GPU, gathers persistent compact positions
and maximum scales, and applies the conservative geometric predicate. Retained
positions return for host lookup and occurrence-mask propagation. Residency
resolves unique dynamic IDs; missing full properties are gathered from the
host and transferred. GPU assembly gathers loaded/reused properties, combines
the separately resident skybox, and restores order and duplicate occurrences.
Packed backend survival follows materialization; it is not a synonym for
semantic selection or filtering. Skybox membership is fixed in the semantic
prefix but its materialized subset can vary after filtering. Its initial
upload is outside dynamic per-frame property H2D accounting.

Previous retains only exact properties from the preceding materialized set.
Bounded uses unique-ID slots and LRU replacement among unrequested rows when
needed; it fails if the current unique request exceeds capacity. Cache hits
still incur lookup, device gathering and assembly. The released no-cache
baseline continues to load duplicate occurrences separately. The letter-ID
example depicts occurrence reconstruction, not a new numerical experiment.

## Validation evidence and reproducibility

architecture_provenance.json hashes the paper, pinned released blobs, current
project sources, canonical checks, fonts, TeX sources, generator and outputs.
The generator checks the three final extension source hashes against
protocol_execution_002.json in
benchmark_artifacts/alod_synth_strong_lod_ext_multiset_closure_001. It checks
the existing package gate and correctness/regression results without rerunning
GPU tests. Scopes stay distinct:

- Run009: 16-camera official semantic oracle; 71-camera CPU/CUDA filter and
  rendering checks, from alod_ext_formal_run009_001/checks/correctness.json.
- Final synthetic multiset closure: all 25 cameras, including the seven
  duplicate-bearing cameras, and the dedicated duplicate gate.
- Post-change natural regression: Campus 24 and Sci-Art 21 cameras; canonical
  correctness and trace artifacts remain byte-identical, as recorded in the
  closure's natural_regression_summary.json.

No figure asserts that these gates cover the same cameras or policy sets.
The paper's bounded Render Full renderer remains unreconstructed and is not
illustrated as an implemented pipeline. No unverifiable architecture concept
was needed; unknown cache details stay in the existing source-boundary text.

From /home/luis/tese, using Python 3, Git and XeLaTeX with TikZ/fontspec/standalone:

```bash
python3 thesis_latex/scripts/generate_alod_architecture.py
python3 thesis_latex/scripts/generate_alod_architecture.py \
  --output output/figures_phase2b/reproduced
diff thesis_latex/images/alod/architecture_provenance.json \
  output/figures_phase2b/reproduced/architecture_provenance.json
```

Independent generation produced identical hashes for all four PDFs and the
entire provenance manifest. The fixed build date and fixed-length PDF trailer
IDs remove temporary-path metadata variation. Standalone builds reject
Overfull/Underfull, missing-character, font-substitution and LaTeX/PGF errors.

## Dissertation layout checks

Final dissertation.pdf: 145 A4 pages. Fresh auxiliary files were created under
output/figures_phase2b/build because the old root auxiliary file is truncated.
BibTeX regenerated the temporary bibliography; three final XeLaTeX passes
succeeded with stable references and no undefined citations/references,
missing characters or oversized-float warnings. BibTeX reports existing
undefined july strings and a duplicate CityGaussianV2 entry (exit 1), while
producing the bibliography used by the successful build. These source issues,
existing prose/table boxes, and the stmaryrd bold-font substitution remain
outside this batch.

All four final figure pages and neighboring text were rendered at 1800-pixel
page height and visually inspected. Figures remain readable at text width,
with complete captions, arrows, references and page numbers. The nearby
runtime-variants table now uses a float page rather than forcing a large
empty area before it. Its contents are unchanged. The adjacent Render Full
scope paragraph is kept together across the float pages. Short optional
captions keep the list of figures concise. Build scripts, logs, previews and
page mapping remain under output/figures_phase2b and are excluded from Git.
The compiled thesis is also an uncommitted build product.

## Remaining figure placeholders

The following eight Chapter 4/5 TODO blocks were outstanding at Phase 2B.
Phase 3A now completes the timing decomposition and normalized materialization
funnel, and adds natural-LoD distributions and final policy performance.
See [FIGURES_PHASE3A.md](FIGURES_PHASE3A.md) for the current six-placeholder
inventory, sources and final page mapping. The list below records the
Phase 2B state:

- Chapter 4: public proxy/training/hierarchy/validation pipeline.
- Chapter 4: faithful timing decomposition.
- Chapter 4: released viewer and order-dependent reuse evidence.
- Chapter 4: ground-truth/render/error image examples.
- Chapter 4: held-out and stress camera count/depth coverage.
- Chapter 4: synthetic-control scene/camera path and adaptive response.
- Chapter 5: cross-dataset normalized materialization funnels.
- Chapter 5: cache-capacity and transfer evidence.

Phase 1 and 2A CityGaussian assets and integration are unchanged. No remaining
performance plot or image-quality panel was produced in this batch.

## Focused commit files

- thesis_latex/chapters/ALoDReproductionAndEvaluation.tex
- thesis_latex/chapters/ALoDExec.tex
- thesis_latex/images/alod/hspt_selection.pdf
- thesis_latex/images/alod/released_execution.pdf
- thesis_latex/images/alod/exec_pipeline.pdf
- thesis_latex/images/alod/residency_occurrences.pdf
- thesis_latex/images/alod/source/architecture_style.tex
- thesis_latex/images/alod/source/hspt_selection.tex
- thesis_latex/images/alod/source/released_execution.tex
- thesis_latex/images/alod/source/exec_pipeline.tex
- thesis_latex/images/alod/source/residency_occurrences.tex
- thesis_latex/images/alod/architecture_provenance.json
- thesis_latex/scripts/generate_alod_architecture.py
- thesis_notes/FIGURES_PHASE2B.md
