# Production v3 — full-19-tag baselines

## Background

After Step 5 of [golden_v2_plan.md](golden_v2_plan.md) we have two v2
golden answer keys (hand: 30 docs, JATS-silver: 75 docs) covering ~13
distinct semantic tags between them. The Step 5 report
([golden_v2_step5_report.md](golden_v2_step5_report.md)) showed that
both v2 experiments use 3-class predictors against a 13-class answer
key, so macro F1 is mechanically bounded.

This plan builds three **true-19-tag logistic-regression baselines**
to use as comparison floors for any future model work. Each baseline
is trained on a different slice of the available annotated corpus and
evaluated against both v2 golden DBs.

## Goals

1. A canonical 19-tag schema (drop `Holotype`/`Distribution` as
   deprecated, drop `FIX` as a workflow marker) documented in
   `ingestors/yedda_tags.py` so every downstream consumer agrees on
   the active label set.
2. A regenerated JATS training corpus (`skol_training_taxpub_v2`)
   using the post-Step-2 converter — so the JATS-only baseline gets
   the full ~13-tag coverage instead of the 8-tag pre-Step-2 set.
3. Three "no-golden" training DBs (option b from the planning
   discussion):
   - `skol_training_v2_no_golden` — hand corpus minus 30 hand-golden IDs
   - `skol_training_taxpub_v2_no_golden` — JATS corpus minus 19 JATS-golden IDs
   - `skol_training_v3_combined_no_golden` — union of the two above
4. Three new experiment docs (`production_v3_hand`,
   `production_v3_jats`, `production_v3_full`) using a fresh
   `logistic_sections_v3_19class` MODEL_CONFIG.
5. Train + evaluate each against **both** `skol_golden_ann_hand_v2`
   and `skol_golden_ann_jats_v2`, producing a 3×2 metrics grid.
6. Comparison report putting the new baselines next to the Step-5 v2
   numbers, plus per-tag coverage so future work can target the
   weakest classes.

## Non-goals

- **No new annotation effort.** This is purely about exercising the
  existing hand and JATS annotations with a true-multiclass model.
- **No SBERT/transformer baseline.** Logistic regression only, by
  design — it's the cheap floor that more expensive models must
  beat.
- **No retraining of the v2 experiments.** `production_v2` and
  `jats_v2` stay exactly as they are; v3 lives alongside.
- **No regeneration of golden DBs.** v2 golden universe is the
  evaluation target, unchanged.

## Corpus shape (confirmed)

| Set | Docs | Distinct tags | Notes |
|---|---|---|---|
| `skol_training_v2` (hand) | 190 | 18 | All hand annotations |
| `skol_training_taxpub_v1` (JATS, pre-Step-2) | 1 743 | 8 | Stale — uses old converter; includes deprecated `Holotype` |
| `skol_training_taxpub_v2` (JATS, post-Step-2) | 1 743 expected | ~13 | **To be generated in Step 0** |
| hand ∩ JATS | 0 | — | Disjoint — combined training is straightforward |
| hand ∩ `skol_golden_ann_hand_v2` | 30 | — | All 30 hand-golden IDs live in the hand training corpus — **must be excluded from training** (Step 2) |
| JATS ∩ `skol_golden_ann_jats_v2` | 19 | — | 19 of 75 JATS-golden IDs are in JATS training — **must be excluded from training** (Step 2). This is a pre-existing contamination in `skol_training_taxpub_v1` that inflated `jats_v2`'s Step-5 F1 (the model had seen ~1/3 of its matched eval docs during training). |

## 19-tag canonical schema

Drop from the 22-tag enum in `ingestors/yedda_tags.py`:

- `HOLOTYPE` — explicitly deprecated, folds into TYPE_DESIGNATION
- `DISTRIBUTION` — explicitly deprecated, folds into BIOLOGY
- `FIX` — workflow marker, not a semantic class

Keep all 19 remaining tags, including ones that don't actually appear
in either corpus (`New-combinations` has zero hand and zero JATS
occurrences but stays in the schema so the model can learn it later
without a schema bump). Trainer will simply learn a zero-prior weight
for that class.

## Step-by-step

### Step 0 — Regenerate JATS training corpus

Re-run `bin/jats_to_yedda.py` against the same 1 743 source JATS
articles used to populate `skol_training_taxpub_v1`, writing into a
new DB `skol_training_taxpub_v2`. This produces .ann files with the
post-Step-2 tag coverage (Materials-and-methods + intra-treatment
Notes).

| # | Description | Status |
|---|---|---|
| 0.A | Identify the 1 743 source doc IDs in `skol_training_taxpub_v1`. They originated from a `skol_dev` filter that selected JATS-format articles; verify the source query so we regenerate from the same set. | ✅ Done (2026-05-20) — `source_database: skol_dev` on every doc; filter is `xml_available && xml_format ∈ {jats,taxpub} && is_taxpub != False` (see `bin/jats_to_yedda.py::select_doc_ids`). |
| 0.B | Create `skol_training_taxpub_v2`. For each source doc, run the post-Step-2 `jats_to_yedda.py` → write the new `article.txt.ann` plus the doc body. | ✅ Done (2026-05-20) — added `--include-ids FILE` flag to `jats_to_yedda` (TDD, 6 tests in `bin/jats_to_yedda_test.py`); dumped the 1 743 IDs to `/tmp/taxpub_v1_ids.txt`; ran `python bin/jats_to_yedda.py --all --database skol_dev --include-ids /tmp/taxpub_v1_ids.txt --output-to couchdb --output-database skol_training_taxpub_v2`. 1 743 docs written, exit 0. |
| 0.C | Verify tag distribution shifted: pre-Step-2 had 8 tags; post-Step-2 should have ~13. | ✅ Done (2026-05-20) — **14 tags** (beat the estimate). Added: `Materials-examined` (5 633), `Biology` (2 991), `Type-designation` (2 438), `Diagnosis` (1 878), `Materials-and-methods` (136), `Bibliography` (8), `Phylogeny` (2). Dropped deprecated `Holotype` (7 427 → 0; folded into `Type-designation`). `Notes` grew 5 409 → 9 983 (intra-treatment catch-all promotion); `Misc-exposition` shrunk 19 475 → 11 120. |
| 0.D | Update `docs/couchdbs.md` with the new DB. | ✅ Done (2026-05-20) — `skol_training_taxpub_v2` row added under Training corpora. |

### Step 1 — 19-tag canonical-schema accessor

| # | Description | Status |
|---|---|---|
| 1.A | Add `Tag.active_19()` classmethod (or module-level `ACTIVE_TAGS_19` tuple) in `ingestors/yedda_tags.py` returning the 19-tag set with deprecation reasons documented. | ✅ Done (2026-05-20) — module-level `ACTIVE_TAGS_19: Tuple[Tag, ...]` and `DEPRECATED_TAGS: FrozenSet[Tag]` constants in [ingestors/yedda_tags.py](../ingestors/yedda_tags.py). |
| 1.B | TDD: tests pin the exact set; pin that `HOLOTYPE` / `DISTRIBUTION` / `FIX` are excluded; pin the count. | ✅ Done (2026-05-20) — `TestActiveTags19` (8 tests) in [ingestors/yedda_tags_test.py](../ingestors/yedda_tags_test.py): count anchor, exclusion of each deprecated tag, partition coverage, disjoint check, exact string anchor, ordered-tuple contract. |
| 1.C | Update `ingestors/jats_to_yedda.py` and `bin/train_classifier.py` MODEL_CONFIGS class_weights to source from `ACTIVE_TAGS_19` rather than hardcoded subsets. | ✅ Done (2026-05-20) — `JATS_EMIT_TAGS: FrozenSet[Tag]` constant added to `ingestors/jats_to_yedda.py` (4 tests in `TestJatsEmitTags`); `logistic_sections_v3` MODEL_CONFIG class_weights expanded from 13 → 19 keys matching `ACTIVE_TAGS_19` (placeholder weights 1.0; Step 3.B refines). v1 baselines `logistic_sections` and `logistic_sections_taxpub_v1` intentionally left at 3 classes for v1 comparison. Deprecated `Distribution` removed. |

### Step 2 — No-golden training DBs

Option (b) from the planning discussion: materialise filtered DBs so
the "exclude golden" decision is auditable and stable across runs.

| # | Description | Status |
|---|---|---|
| 2.A | New helper `bin/build_no_golden_training_db.py` with CLI: `--source DB --golden-ann DB --output DB`. Iterates source, skips any doc_id present in `--golden-ann`, copies the rest (doc + attachments) to `--output`. Idempotent (skip if output exists with matching doc count). | ✅ Done (2026-05-20) |
| 2.B | TDD: tests against in-memory fake DBs covering (a) basic filtering, (b) doc count equals source - golden, (c) attachments preserved, (d) idempotent on re-run. | ✅ Done (2026-05-20) — 7 tests in `bin/build_no_golden_training_db_test.py` + 5 tests in `bin/build_combined_training_db_test.py`, all green. |
| 2.C | Run for hand: `--source skol_training_v2 --golden-ann skol_golden_ann_hand_v2 --output skol_training_v2_no_golden` → 160 docs. | ✅ Done (2026-05-20) — `copied=160 skipped_golden=30 skipped_exists=0`. |
| 2.D | Run for JATS: `--source skol_training_taxpub_v2 --golden-ann skol_golden_ann_jats_v2 --output skol_training_taxpub_v2_no_golden` → 1 724 docs. | ✅ Done (2026-05-20) — `copied=1724 skipped_golden=19 skipped_exists=0`. |
| 2.E | Combined: new helper `bin/build_combined_training_db.py` that unions `skol_training_v2_no_golden` + `skol_training_taxpub_v2_no_golden` into `skol_training_v3_combined_no_golden` (1 884 docs; safe because hand ∩ JATS = 0). | ✅ Done (2026-05-20) — `copied=1884 skipped_exists=0`, zero ID collisions. |
| 2.F | Update `docs/couchdbs.md` with the three new DBs. | ✅ Done (2026-05-20). |

### Step 3 — 19-class MODEL_CONFIG class weights

Folded into Step 1.C: `logistic_sections_v3` was rewired to use the
full 19-key class_weights dict aligned with `ACTIVE_TAGS_19` (no new
MODEL_CONFIG entry needed — the existing `logistic_sections_v3` *is*
the 19-class baseline now). Remaining work is the weight-value
refinement.

| # | Description | Status |
|---|---|---|
| 3.A | ~~Add `logistic_sections_v3_19class` to MODEL_CONFIGS~~ | ✅ Folded into Step 1.C — `logistic_sections_v3` is now the 19-class baseline directly. |
| 3.B | Pre-compute inverse-frequency class weights from `skol_training_v3_combined_no_golden`'s tag distribution; replace the placeholder 1.0s in `logistic_sections_v3` with the computed values. | ✅ Done (2026-05-20) — formula: `weight = min(max_count / count, 10.0)` over per-tag annotation-block counts (131 543 total blocks across 1 884 docs). Misc-exposition gets 1.0 (most common); Diagnosis/Type-designation/Biology/Etymology get 4-7×; Phylogeny/Key/Materials-and-methods/ToC-entry capped at 10×. New-combinations has zero blocks in the corpus → 1.0 placeholder. Two new sanity tests in `bin/train_classifier_test.py` pin the ordering (misc ≤ everything; rare > common). |
| 3.C | TDD: pin the 19 class_weight keys against `ACTIVE_TAGS_19`. | ✅ Done in Step 1.C — `TestLogisticSectionsV3ClassWeightsCoverActive19` in `bin/train_classifier_test.py`. |

### Step 4 — Experiment docs ✅ Done (2026-05-20)

Three new docs in `skol_experiments`. `manage_experiment.py create`
doesn't expose `golden` / `golden_ann` / `redis_keys` — created
directly via a Python `couchdb` write, same pattern as Step 4 of
[golden_v2_plan.md](golden_v2_plan.md).

| Name | training | golden | golden_ann (primary) | annotations | model_name |
|---|---|---|---|---|---|
| `production_v3_hand`  | `skol_training_v2_no_golden`              | `skol_golden_v2` | `skol_golden_ann_hand_v2` | `skol_exp_production_v3_hand_ann` | `logistic_sections_v3` |
| `production_v3_jats`  | `skol_training_taxpub_v2_no_golden`       | `skol_golden_v2` | `skol_golden_ann_hand_v2` | `skol_exp_production_v3_jats_ann` | `logistic_sections_v3` |
| `production_v3_full`  | `skol_training_v3_combined_no_golden`     | `skol_golden_v2` | `skol_golden_ann_hand_v2` | `skol_exp_production_v3_full_ann` | `logistic_sections_v3` |

All three reference the same `logistic_sections_v3` MODEL_CONFIG
(architecture + 19-key class_weights from Step 3.B). They differ only
in training corpus. Redis keys carry a `:v3_{hand,jats,full}`
namespace so each model is independently addressable. `golden_ann`
set to the hand gold (richer answer key); the silver JATS eval is a
one-off Step-5.G invocation.

### Step 5 — Pipeline runs

| # | Description | Status |
|---|---|---|
| 5.A | `runstep production_v3_hand train` | ✅ Done (2026-05-20) — line F1 0.79 on 19-class test split (`collapse_labels=False` after [`f417c11`](../bin/train_classifier.py) fix). |
| 5.B | `runstep production_v3_hand evaluate` (against `skol_golden_ann_hand_v2`) | ✅ Done (2026-05-20) — character-level macro F1 **0.459** on hand gold. |
| 5.C | `runstep production_v3_jats train` | ✅ Done (2026-05-21) — line F1 0.95 on test split. |
| 5.D | `runstep production_v3_jats evaluate` | ✅ Done (2026-05-21) — character-level macro F1 **0.132** on hand gold. |
| 5.E | `runstep production_v3_full train` | ✅ Done (2026-05-22) — line F1 0.87; finished in ~30 min on 16-core Spark after `74ce0ab` defaults bump. |
| 5.F | `runstep production_v3_full evaluate` | ✅ Done (2026-05-22) — character-level macro F1 **0.326** on hand gold; lies between v3_hand and v3_jats, confirming the 9:1 JATS-dominated combined corpus drags performance below hand-only. |
| 5.G | ~~One-off `evaluate_golden.py` runs against `skol_golden_ann_jats_v2` for each of the three models.~~ | ❌ **Moot** — under the post-Step-7 architecture ([extraction_pipeline.md](extraction_pipeline.md)) JATS docs go through deterministic XML extraction, not the classifier. There's no production model path for the JATS silver gold to score; it becomes a converter QA artifact instead. |
| 5.H | ~~Re-evaluate `jats_v2` after pointing its `training` field at `skol_training_taxpub_v2_no_golden`.~~ | ❌ **Moot** — same reason; `jats_v2` and all v3 JATS-evaluation experiments are archived. |

### Step 6 — Comparison report — ✅ Done (2026-05-22)

Written at [docs/production_v3_report.md](production_v3_report.md).

Headline finding: **the cross-distribution-training hypothesis is
falsified.** v3_hand (160 hand docs) macro F1 **0.459**; v3_full
(1 884 combined docs) macro F1 **0.326**; v3_jats (1 724 JATS docs)
macro F1 **0.132**. Adding JATS data hurts PDF classification
monotonically. The 1×3 grid (one gold, three models) collapsed the
originally-planned 3×2 grid because Step 5.G/5.H became moot under
the new extraction architecture ([extraction_pipeline.md](extraction_pipeline.md)).

## Open questions

- **Should `Notes` get a class weight bump?** It's the most prevalent
  "interesting" tag in the JATS silver (120 occurrences) and
  third-largest in hand (3 726 chars). Default inverse-frequency
  would give it a moderate weight; we may want to bias higher
  since it's a target for future Phase-2 diagnosis-extraction work.
  **Resolved (2026-05-22):** v3_hand's Notes character-level F1 is
  0.518 (mid-range — sixth of 18 tags). No bump warranted at the
  v3 weighting; revisit when v4 metrics land.

## Workflow / sequencing

1. **Step 0 first** — without the regenerated JATS corpus the
   JATS-only baseline is meaningless.
2. **Steps 1–3 can land in one commit** — schema + build script +
   model config are all small and interlinked.
3. **Step 4 (docs) is trivial** once 1–3 land.
4. **Step 5 runs in sequence** — each train is single-digit minutes,
   each evaluate is 10–15 minutes (JATS-silver eval is the longest
   due to 75 docs).
5. **Step 6 report** — written from the metrics in the experiment
   docs (each evaluate saves to `evaluation` field).

## Rollback

- **Step 0**: drop `skol_training_taxpub_v2`. The v1 corpus stays.
- **Steps 1–3**: revert the code changes. v2 experiments unaffected.
- **Step 2**: drop the three new training DBs.
- **Step 4**: remove the three new experiment docs.
- **Step 5–6**: nothing to roll back; scoped to v3 DBs.

## Progress

Sub-steps land one at a time, each leaving the test suite green and
the v2 experiments (and the production v1 pipeline) fully
reproducible.
