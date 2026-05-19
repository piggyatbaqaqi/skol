# Golden v2 plan — refresh the evaluation universe

## Background

The original golden set (`skol_golden`, `skol_golden_ann_hand`,
`skol_golden_ann_jats`) was built when:

- The label set was 12 tags (see `docs/taxon_to_treatment_plan.md` §1a
  for the original list).
- Hand annotation hadn't yet been redone on top of the post-merge
  `skol_ann_merged` corpus.
- The JATS converter mapped only ~10 sec-types.

Today:

- `ingestors/yedda_tags.py` defines 22 tags (20 active + 2 deprecated).
- `skol_training_v2` is the post-hand-annotation training set.
- `bin/jats_to_yedda.py` covers most but not all of the new tag set.

To take advantage of these changes we need a parallel "v2" golden
universe. To preserve scientific reproducibility we must also keep the
old experiments runnable verbatim against the old golden set. This
plan covers both: the v2 golden set itself, the JATS converter
extensions, and the experiments-framework extension that lets every
experiment record which golden set it is scored against.

## Goals

1. Three new golden databases (`skol_golden_v2`,
   `skol_golden_ann_hand_v2`, `skol_golden_ann_jats_v2`) built from
   the **same 105 source documents** the v1 set used, so v1↔v2
   metrics are directly comparable on the same articles.
2. JATS converter updates so the regenerated `*_ann_jats_v2` reflects
   the current 22-tag label set.
3. Experiment-schema extension: every experiment in `skol_experiments`
   carries its own `databases.golden` and `databases.golden_ann`,
   so any experiment can be re-run and re-evaluated independently and
   v1↔v2 experiments coexist.
4. Two new experiments — `production_v2` and `jats_v2` — that exercise
   the v2 golden set.
5. v1 experiments remain reproducible exactly as they ran before.

## Non-goals

- **No classifier for inline diagnostic prose.** Latin-description-as-
  diagnosis and diagnostic prose buried in Notes ("X sp. nov. differs
  from Y by …") are intrinsic granularity losses of the JATS path.
  We document the gap and leave a possible Phase-2 classifier as
  follow-up.
- **No INDEX / ToC-entry / FIX / Page-header derivation from JATS.**
  These are PDF-only structural artefacts or human-attention markers;
  the JATS-derived set will not produce them.
- **No deletion of v1 golden DBs.** They stay alongside v2 forever so
  the old experiments remain reproducible.

## Step 1 — Experiment-schema extension

This lands first so subsequent steps can record their target golden
DBs on each experiment.

### Schema

Two new optional fields per experiment doc, both under `databases`:

```json
{
  "databases": {
    "ingest": "...",
    "training": "...",
    "treatments": "...",
    "treatments_full": "...",
    "annotations": "...",
    "golden":      "skol_golden",            // plaintext + PDF + XML
    "golden_ann":  "skol_golden_ann_hand"    // answer-key .ann
  }
}
```

`golden` is the plaintext/PDF/XML DB used by `predict_classifier.py`'s
`--golden-db` (predict on every golden plaintext). `golden_ann` is the
answer-key `.ann` DB used by `evaluate_golden.py` to score those
predictions. Together they define the experiment's evaluation universe.

### Sub-steps

| # | Description | Status |
|---|---|---|
| 1.A | Backfill `databases.golden` + `databases.golden_ann` on every existing experiment doc (`production`, `hand_annotated`, `jats_v1`, `taxpub_v1`, `taxpub_v1_int8`, `taxpub_v1_onnx_int8`). Values per the table below. Script: `bin/backfill_experiment_golden_fields.py` (one-off, idempotent). | ⬜ Pending |
| 1.B | Extend `bin/env_config.py`: add the experiment-doc mapping rows `('golden', ['golden_db_name'])` and `('golden_ann', ['golden_ann_db_name'])`. Default values: `'skol_golden'` and `'skol_golden_ann_hand'` (the v1 names). Add both keys to the CLI args allowlist. | ⬜ Pending |
| 1.C | Rewire `bin/manage_experiment.py` evaluate step. Drop the hardcoded `"skol_golden"` and `"skol_golden_ann_hand"` literals at lines 493–505. Use `{golden_db_name}` and `{golden_ann_db_name}` placeholders (the existing `_apply()` substitution mechanism) so values flow from the resolved experiment config. | ⬜ Pending |
| 1.D | Tests: `bin/manage_experiment_test.py` (or extension thereof) covering (a) the resolved command includes the experiment's golden values, (b) an experiment lacking the new fields falls back to v1 defaults, (c) per-experiment override works. | ⬜ Pending |
| 1.E | Update `docs/experiments.md` and `docs/couchdbs.md` to document the new fields and per-experiment values. | ⬜ Pending |

### Per-experiment values for the backfill

| Experiment | golden | golden_ann | Why |
|---|---|---|---|
| `production` | `skol_golden` | `skol_golden_ann_hand` | v1 hand gold |
| `hand_annotated` | `skol_golden` | `skol_golden_ann_hand` | v1 hand gold |
| `jats_v1` | `skol_golden` | `skol_golden_ann_jats` | trained on JATS — score against JATS silver |
| `taxpub_v1` | `skol_golden` | `skol_golden_ann_jats` | same |
| `taxpub_v1_int8` | `skol_golden` | `skol_golden_ann_jats` | same |
| `taxpub_v1_onnx_int8` | `skol_golden` | `skol_golden_ann_jats` | same |

The `jats_v1` / `taxpub_v1*` rows currently evaluate (per the
hardcoded literal) against `skol_golden_ann_hand`. **This is a latent
inconsistency** — those experiments are trained on JATS but scored
against the hand standard. Backfilling them to `skol_golden_ann_jats`
restores the intended train/test pairing. Document this in the
backfill commit so the metric shift in the next evaluation run is
attributable.

## Step 2 — JATS converter extensions

### Tag gap

`ingestors/jats_to_yedda.py::sec_type_to_tag()` already maps 13 of the
22 tags. Gaps to fill for v2:

| Tag | Source | Action |
|---|---|---|
| `MATERIALS_AND_METHODS` | Article-level `<sec sec-type="materials-and-methods">` and variants (`methods`, `methodology`, `methods-and-materials`) | Extend `sec_type_to_tag()` *and* add an article-level section walker since this section lives outside `<tp:taxon-treatment>` |
| `NOTES` (catch-all in treatment) | Any prose block inside `<tp:taxon-treatment>` (or `<sec sec-type="taxon-treatment">`) without a recognised `sec-type` | Replace the "fall through to MISC_EXPOSITION" default for intra-treatment content with `NOTES`. Explicit `sec-type="notes"` stays in place. |
| `DIAGNOSIS` (TaxPub-only) | Explicit `<tp:treatment-sec sec-type="diagnosis">` (and `diagnosis-*` variants) — already mapped | No change — granularity gap documented under Known limitations |
| `INDEX`, `TOC` | _(not derived from JATS)_ | No change |
| `FIX`, `PAGE_HEADER` | _(PDF-only)_ | No change |

### Sub-steps

| # | Description | Status |
|---|---|---|
| 2.A | Extend `sec_type_to_tag()` with the `materials-and-methods` family (case-insensitive, hyphen/underscore tolerant). | ⬜ Pending |
| 2.B | Add article-level processing in `bin/jats_to_yedda.py` so non-treatment `<sec>` elements get their own `sec_type_to_tag()` pass instead of all collapsing to `MISC_EXPOSITION`. Treatment-level processing unchanged. | ⬜ Pending |
| 2.C | Change the intra-treatment fallback from `MISC_EXPOSITION` to `NOTES` in `process_treatment()` and `process_plain_jats_treatment()`. | ⬜ Pending |
| 2.D | Unit tests in `ingestors/jats_to_yedda_test.py` for each new case: explicit `materials-and-methods` sec-type, article-level vs intra-treatment scope, intra-treatment catch-all becomes Notes, no-regression on the existing 13 mappings. | ⬜ Pending |
| 2.E | Spot-check on 3 sample JATS files from `skol_dev` — count tag distributions before and after the change so we can characterise the impact size for the eventual v2 metrics shift. | ⬜ Pending |

## Step 3 — Curator v2 mode

### Design

`bin/curate_golden_dataset.py` currently writes hardcoded `skol_golden`
/ `skol_golden_ann_hand` / `skol_golden_ann_jats` DB names. For v2:

- Add `--version v1|v2` (or `--output-suffix _v2`) parameter that
  changes the output DB names.
- For `v2`, accept a `--reuse-ids-from skol_golden` flag that **inherits
  the exact 105 doc IDs** from v1 (no new stratified selection), so
  v1 and v2 sit on identical articles.
- Replace the source for hand-annotated `.ann` from `skol_training` to
  `skol_training_v2`.
- JATS-derived `.ann` regeneration uses the post-Step-2 converter.

### Sub-steps

| # | Description | Status |
|---|---|---|
| 3.A | Add `--version` and `--reuse-ids-from` flags to `bin/curate_golden_dataset.py`. Default `--version v1` so calling without flags is unchanged. | ⬜ Pending |
| 3.B | When `--version v2`, source hand `.ann` from `skol_training_v2` (override via `--hand-source-db`). | ⬜ Pending |
| 3.C | Tests for ID-inheritance: given a v1 `skol_golden`, `--reuse-ids-from skol_golden` produces exactly the same doc IDs in the v2 output (set equality). | ⬜ Pending |
| 3.D | Run the v2 curation against local dev CouchDB. Verify `skol_golden_v2` ⊆ same 105 article IDs as `skol_golden`. | ⬜ Pending |

## Step 4 — New experiment definitions

### Sub-steps

| # | Description | Status |
|---|---|---|
| 4.A | Create `production_v2` experiment doc. Inherits production's databases, points `training` at `skol_training_v2`, points `golden` / `golden_ann` at `skol_golden_v2` / `skol_golden_ann_hand_v2`. | ⬜ Pending |
| 4.B | Create `jats_v2` experiment doc. `training: skol_training_taxpub_v1` (unchanged unless we plan a v2 of that too), `golden_ann: skol_golden_ann_jats_v2`. | ⬜ Pending |
| 4.C | Optionally `jats_v2_int8` / `jats_v2_onnx_int8` to mirror the `taxpub_v1` family — only if you want quantised variants in the v2 comparison. | ⬜ Optional |
| 4.D | Update `docs/couchdbs.md` experiments table with the new rows. | ⬜ Pending |

## Step 5 — Pipeline runs

Train + evaluate each new experiment, compare against v1 metrics.

| # | Description | Status |
|---|---|---|
| 5.A | `python bin/manage_experiment.py runstep production_v2 train` | ⬜ Pending |
| 5.B | `python bin/manage_experiment.py runstep production_v2 evaluate` | ⬜ Pending |
| 5.C | `python bin/manage_experiment.py runstep jats_v2 train` | ⬜ Pending |
| 5.D | `python bin/manage_experiment.py runstep jats_v2 evaluate` | ⬜ Pending |
| 5.E | Comparison report: side-by-side v1 vs v2 metrics for each experiment, plus the new metrics for tags that didn't exist in v1 (`Materials-and-methods`, intra-treatment `Notes`). | ⬜ Pending |

## Known limitations of the v2 JATS silver standard

These are inherent granularity gaps that TaxPub markup cannot resolve.
They are why the hand-annotated set remains "gold" and the JATS-derived
set remains "silver":

1. **Latin description as diagnosis** — TaxPub authors sometimes tag a
   Latin diagnostic passage as `<tp:treatment-sec sec-type="description">`
   even when it functions as a diagnosis. The v2 JATS converter
   reproduces TaxPub's label verbatim, so these continue to be tagged
   `Description` in the silver set.
2. **Diagnostic prose in Notes** — Statements like "*Foo bar* sp. nov.
   differs from *Foo frob* by …" frequently sit inside a Notes section.
   TaxPub has no way to mark these inline; they will be tagged `Notes`
   in the silver set. Hand-annotators routinely promote them to
   `Diagnosis`.

### Phase 2 candidate (out of scope for v2)

A binary classifier trained on the hand-set's relabellings — wherever
the annotator promoted a Notes block to Diagnosis, that's a positive
example. With ~30 hand docs there should be 50–150 labelled blocks,
which is enough for a lightweight logistic-regression-on-TF-IDF or
sentence-BERT cosine baseline. The classifier would post-process the
JATS-derived `.ann` to up-promote candidate Notes → Diagnosis blocks.

## Workflow / sequencing

1. **Step 1 first** (experiment schema). It's the foundational
   refactor; everything downstream uses it.
2. **Step 2** (JATS converter). Adds the new tag coverage *before*
   the curator runs, so v2's JATS silver reflects the post-converter
   state.
3. **Step 3** (curator v2 mode + run). Produces `skol_golden_v2`,
   `skol_golden_ann_hand_v2`, `skol_golden_ann_jats_v2`.
4. **Step 4** (new experiment docs).
5. **Step 5** (pipeline runs + comparison).

Each step's commits are non-destructive of v1 state — old golden
DBs, old experiment docs, old training corpora all stay in place.
Rollback at any point is "stop and remove the new DBs and experiments".

## Rollback

- **Step 1**: revert the schema backfill (remove `databases.golden`
  and `databases.golden_ann` from each experiment doc); revert the
  `manage_experiment.py` / `env_config.py` code changes. v1
  experiments revert to using the hardcoded literals.
- **Step 2**: revert the JATS converter changes. Existing
  `skol_golden_ann_jats` (v1) is unaffected.
- **Step 3**: drop `skol_golden_v2`, `skol_golden_ann_hand_v2`,
  `skol_golden_ann_jats_v2`. v1 DBs unchanged.
- **Step 4**: remove the new experiment docs from `skol_experiments`.
- **Step 5**: nothing to roll back; pipeline runs are scoped to the
  new experiments' DBs.

## Open questions

None at time of writing — all decisions captured under "Goals" and
"Non-goals" above. New questions get added here as they surface.

## Progress

Sub-steps land one at a time, each leaving the test suite green and
v1 experiments fully reproducible.

(Empty for now — first commit will check off 1.A.)
