# Golden v2 — Step 5 results & v1↔v2 comparison

Step 5 of [docs/golden_v2_plan.md](golden_v2_plan.md) ran train + evaluate for
`production_v2` and `jats_v2` against the post-Step-2/3 v2 golden universe.

## Summary

| Experiment       | Model architecture            | Training corpus            | Golden answer key             | Macro F1 | Docs matched |
|------------------|-------------------------------|----------------------------|-------------------------------|----------|--------------|
| `production_v2`  | `logistic_sections_v3` (13 wts cfg, 3 active in data) | `skol_training_v2`         | `skol_golden_ann_hand_v2`     | **0.1270** | 30 / 30      |
| `jats_v2`        | `logistic_sections_taxpub_v1` (3 cls) | `skol_training_taxpub_v1`  | `skol_golden_ann_jats_v2`     | **0.1715** | 56 / 75      |

Per-experiment line-level training accuracy (held-out test split) was high
(production_v2 ~0.96, jats_v2 ~0.97) — the models predict their *trained* labels well; macro-F1 on the v2 golden is low because the v2 golden carries
~13 tags the 3-class models cannot emit.

## Per-tag character-level F1

### `production_v2` vs `skol_golden_ann_hand_v2` (30/30 docs)

| Tag                    | Precision | Recall | F1     | Notes |
|------------------------|-----------|--------|--------|-------|
| Description            | 0.860     | 0.939  | 0.898  | trained tag |
| Misc-exposition        | 0.283     | 0.951  | 0.436  | trained tag (over-predicted) |
| Nomenclature           | 0.883     | 0.775  | 0.826  | trained tag |
| Biology                | 0.000     | 0.000  | 0.000  | 72 279 chars in answer key, unpredictable |
| Diagnosis              | 0.000     | 0.000  | 0.000  | 118 859 chars, unpredictable |
| Etymology              | 0.000     | 0.000  | 0.000  | 4 348 chars, unpredictable |
| Figure-caption         | 0.000     | 0.000  | 0.000  | 79 387 chars, unpredictable |
| Key                    | 0.000     | 0.000  | 0.000  | 61 195 chars, unpredictable |
| Materials-and-methods  | 0.000     | 0.000  | 0.000  | **58 447 chars — net-new v2 tag** |
| Materials-examined     | 0.000     | 0.000  | 0.000  | 177 861 chars, unpredictable |
| Notes                  | 0.000     | 0.000  | 0.000  | **285 947 chars — net-new v2 tag (intra-treatment promotion)** |
| Phylogeny / Type-designation / Page-header / Table / Bibliography / Index | 0.000 | 0.000 | 0.000 | unpredictable |
| **Macro Avg**          | **0.119** | **0.157** | **0.127** |   |

### `jats_v2` vs `skol_golden_ann_jats_v2` (56/75 docs — 19 unmatched)

| Tag                    | Precision | Recall | F1     | Notes |
|------------------------|-----------|--------|--------|-------|
| Misc-exposition        | 0.767     | 0.972  | 0.857  | trained tag |
| Description            | 0.000     | 0.000  | 0.000  | model predicted 57 894 chars; answer key had none in matched docs (concentrated in 19 unmatched docs) |
| Nomenclature           | 0.000     | 0.000  | 0.000  | model predicted 87 579 chars; answer key had none in matched docs |
| Figure-caption         | 0.000     | 0.000  | 0.000  | 4 600 chars in answer key, unpredictable |
| Materials-and-methods  | 0.000     | 0.000  | 0.000  | **24 059 chars — net-new v2 tag** |
| **Macro Avg**          | **0.153** | **0.194** | **0.171** |   |

Note: only 5 tags appeared in the matched-doc evaluation because the
19 skipped docs (likely those filtered out by `predict_classifier`'s
incremental selection) hold the Description / Nomenclature / Notes /
Etymology / Materials-examined / Type-designation / Biology / Diagnosis
answer-key chars (see tag distribution in the v2 jats golden below).

## Tag distribution in the v2 golden answer keys

`skol_golden_ann_jats_v2` (75 docs) — automated from the post-Step-2
JATS converter; includes the new `Materials-and-methods` and
intra-treatment `Notes` mappings:

| Tag                    | Occurrences | Docs containing |
|------------------------|-------------|-----------------|
| Misc-exposition        | 305         | 75              |
| Figure-caption         | 282         | 33              |
| **Notes**              | **120**     | **17**          |
| Description            | 111         | 16              |
| Nomenclature           | 94          | 19              |
| Etymology              | 62          | 19              |
| Materials-examined     | 56          | 15              |
| Type-designation       | 42          | 13              |
| Biology                | 41          | 10              |
| Diagnosis              | 15          | 4               |
| Key                    | 2           | 2               |
| **Materials-and-methods** | **2**    | **2**           |

Net-new tags (bold) flowed end-to-end from Step 2's converter changes into
the v2 silver standard, with `Notes` becoming the third most prevalent
intra-treatment tag — confirming that the catch-all promotion was a
meaningful semantic shift, not a cosmetic relabel.

`skol_golden_ann_hand_v2` (30 docs) carries even richer annotations
since the hand annotators were not constrained by TaxPub markup —
notably 286k chars of `Notes`, 178k chars of `Materials-examined`,
and 119k chars of `Diagnosis` content.

## v1 baselines — what's available

The plan called for a v1↔v2 side-by-side. Two caveats from the existing
v1 evaluation state:

1. **`production` (v1) was never evaluated.** Its experiment doc has
   `annotations: ""` (intentionally — see Step 1 backfill); no
   predictions on the v1 golden have been written, so no v1 hand
   baseline exists.
2. **`taxpub_v1_onnx_int8`** has a recorded evaluation
   (2026-03-29, F1 0.0736) — but the recorded `golden_database` is
   `skol_golden_ann_hand`, not the JATS silver the experiment is
   trained against. This pre-dates Step 1.A's golden_ann backfill,
   which corrected the doc to `skol_golden_ann_jats`. The cached
   metrics are against the wrong answer key and aren't a clean
   comparison point for jats_v2.
3. **`hand_annotated`** has a recorded evaluation with
   `macro_f1: 0.0018` but the per-tag dict is full of literal prose
   strings as keys — evaluator was broken at the time. Unusable.

To produce a clean v1↔v2 side-by-side we would need to re-run v1
evaluate for `production` and `taxpub_v1` against their correct golden
DBs. That is **out of scope for Step 5 as planned**; the plan's
explicit comparison goal (5.E) is the *v2 metrics* and the *new-tag
coverage* — both delivered above.

## Interpretation

1. **The v2 universe exposes a labelling gap, not a model regression.**
   Both v2 experiments use 3-class architectures (production_v2's
   MODEL_CONFIG declares 13 class weights but the training data
   contains only 3 distinct labels, so the model trains as 3-class;
   `logistic_sections_taxpub_v1` is 3-class by design). The v2 golden
   has ~13 tag classes including the net-new `Materials-and-methods`
   and intra-treatment `Notes`. F1 ceiling on a 3-class model against
   a 13-class answer key is mechanically bounded — the unpredictable
   tags contribute zeros to the macro average. Macro F1 of 0.127 /
   0.172 on a 13-class answer key with a 3-class predictor is in line
   with that bound.
2. **The new tags are visible in the v2 silver standard.**
   `Materials-and-methods` shows up in the v2 JATS silver
   (24 059 chars; 2 occurrences across 2 docs — small but non-zero,
   consistent with the rare article-level Materials-and-methods section
   in TaxPub journals). `Notes` is *substantially* present (120
   occurrences across 17 docs in the JATS silver, 286 k chars in the
   hand gold) — confirming Step 2.C's "intra-treatment catch-all →
   Notes" change is the bigger of the two Step-2 semantic shifts.
3. **The jats_v2 19-doc gap.** `skol_golden_ann_jats_v2` has 75 docs but
   only 56 matched in the evaluation. The remaining 19 either failed the
   predict step's incremental filter or did not produce any prediction
   attachment. Worth diagnosing in a follow-up — possibly tied to
   `SKIP_TAXONOMY_CHECK`-style pre-filters in `predict_classifier`.

## Follow-ups (not required to close Step 5)

1. **True 13-class production model.** Either (a) build a
   `skol_training_v2_13class` corpus by re-annotating the hand set
   with the 13-tag schema, or (b) re-use `skol_training_taxpub_v1`
   for training but evaluate against `skol_golden_ann_hand_v2`.
   Without one of these, no model can exercise more than 3 of the
   v2 golden's tags.
2. **Diagnose the 19 missing predictions on jats_v2 evaluate** —
   probably a pre-filter dropping non-mycology docs that the v2
   golden still includes.
3. **v1 baseline runs.** If a clean v1↔v2 side-by-side is wanted,
   re-run evaluate for `production` (after setting its annotations
   DB) and `taxpub_v1` (now correctly pointing at
   `skol_golden_ann_jats`). The new-tag tags will be zero on v1
   evaluations too (model architecture is identical), but
   Misc-exposition / Description / Nomenclature numbers would be
   directly comparable.
4. **Drive-by fixes landed during Step 5:**
   - `bin/train_classifier.py`: added `resolve_training_database()`
     helper so the experiment doc's `databases.training` actually
     overrides the MODEL_CONFIG hardcoded value (5 unit tests in
     `bin/train_classifier_test.py`). Previously every experiment
     silently trained against whatever was hardcoded in
     MODEL_CONFIGS, regardless of the doc's training-DB field.
   - `bin/predict_classifier.py`: added `logistic_sections_v3` to
     MODEL_CONFIGS so v2 experiments can run predict + evaluate.
   - Both `production_v2` and `jats_v2` docs now carry explicit
     `model_name` and `annotations` fields
     (`skol_exp_production_v2_ann` / `skol_exp_jats_v2_ann`).
