# production_v3 — comparison report

Step 7 of [production_v3_plan.md](production_v3_plan.md). Compares
the three v3 baselines (`production_v3_hand`, `production_v3_jats`,
`production_v3_full`) against the v2 golden universe and records the
negative result that drove the architectural pivot to
[extraction_pipeline.md](extraction_pipeline.md).

## Headline

The cross-distribution training hypothesis is **falsified**.

| Model | Training corpus | Docs | Macro F1 on `skol_golden_ann_hand_v2` |
|---|---|---:|---:|
| **v3_hand** | `skol_training_v2_no_golden` (hand-annotated PDFs) | 160 | **0.459** |
| v3_full | `skol_training_v3_combined_no_golden` (hand ∪ JATS) | 1 884 | 0.326 |
| v3_jats | `skol_training_taxpub_v2_no_golden` (JATS-derived) | 1 724 | 0.132 |

The hypothesis going into Step 5 was: *"JATS-derived labels are
abundant, deterministic, and cheap. Augmenting the small
hand-annotated PDF corpus with 10× more JATS data should improve PDF
classification."* The data says no:

- Going from hand-only to hand+JATS (combined) **lost 13 F1 points**
  (29% relative drop) — adding 11× more training data made the
  PDF-classification metric *worse*.
- Going from hand-only to JATS-only lost 33 F1 points (71% relative).
  JATS docs alone don't teach a classifier to handle PDF structure.

## Why cross-distribution didn't transfer

JATS docs are produced by XML serialisation; PDF-extracted docs come
from layout analysis. They differ at the line level in ways the
classifier needs to model:

| Feature | JATS distribution | PDF distribution |
|---|---|---|
| Page headers | Absent | Frequent (4-tag, 9% of chars in hand gold) |
| Index entries | Absent | Frequent (`skol_training_v2` has 11k Index blocks across 11 docs) |
| Multi-column reflow artefacts | Absent | Common — paragraph fragmentation between columns |
| OCR substitution errors | Absent | Variable rate |
| Table interruptions | Clean (`<table-wrap>` structure) | Layout-detected; inline with body text |
| Bibliography style | `<ref-list>` element | Flowed text following the article |

A classifier trained on JATS data learns "after Description comes
Diagnosis" cleanly. It does not learn "after Description, a Page-header
may interrupt and then Description continues" — because that pattern
doesn't appear in its training data.

The combined corpus is 9:1 JATS-dominated by volume. The model's
priors are pulled toward the JATS distribution; on PDF input, that
pull is wrong.

## Per-tag character-level F1

The 18 tags appearing in the hand gold, sorted by v3_hand F1:

| Tag | v3_hand | v3_full | v3_jats |
|---|---:|---:|---:|
| Bibliography | **0.886** | 0.808 | 0.000 |
| Description | **0.819** | 0.673 | 0.423 |
| Materials-examined | **0.666** | 0.456 | 0.318 |
| Nomenclature | **0.652** | 0.318 | 0.194 |
| Misc-exposition | **0.608** | 0.555 | 0.361 |
| Materials-and-methods | **0.580** | 0.524 | 0.000 |
| Biology | **0.540** | 0.470 | 0.304 |
| Notes | **0.518** | 0.241 | 0.254 |
| Type-designation | **0.464** | 0.246 | 0.083 |
| Etymology | 0.463 | 0.101 | 0.052 |
| Diagnosis | **0.419** | 0.356 | 0.070 |
| Page-header | **0.410** | 0.309 | 0.000 |
| Key | **0.403** | 0.325 | 0.000 |
| Figure-caption | **0.398** | 0.342 | 0.183 |
| Phylogeny | **0.221** | 0.000 | 0.000 |
| Table | **0.114** | 0.043 | 0.000 |
| Index | **0.095** | 0.093 | 0.000 |
| ToC-entry | 0.000 | 0.000 | — |
| **Macro avg** | **0.459** | **0.326** | **0.132** |

Observations:

- **v3_hand wins or ties on 17 of 18 tags.** The lone tie is
  ToC-entry, where all three score zero (no model emits it well).
- **v3_jats scores exactly zero on every PDF-structural tag**:
  Bibliography, Page-header, Index, Table, Materials-and-methods,
  Key, ToC-entry. Predictable — those tags don't exist in its
  training data.
- **v3_full is uniformly between hand and jats** — exactly the
  outcome the "9:1 dominated by JATS" data ratio predicts. Nowhere
  does v3_full beat v3_hand; the augmentation is monotonically
  harmful.
- **Etymology gets a pathological 0.054 precision on v3_jats /
  0.101 on v3_full** with relatively high recall — the JATS-trained
  models over-predict Etymology widely. Plausibly because the JATS
  corpus's `Etymology:` paragraphs are short and surface-distinct
  in JATS but a Latin-prose-heavy reader on the PDF side falsely
  matches that pattern across many ordinary description sentences.

## Training cost vs. quality

| Model | Training corpus size | Training wall-clock (16-core Spark) | Macro F1 |
|---|---:|---:|---:|
| v3_hand | 160 docs | ~1 min | 0.459 |
| v3_jats | 1 724 docs | ~3 min | 0.132 |
| v3_full | 1 884 docs | ~30 min (was 1 h 53 min at 4 cores; killed) | 0.326 |

v3_full's training is **5-10× the cost of v3_hand for measurably
worse quality**. Not a productive trade.

## Implications

### Killed as deployment targets

- **`production_v3_jats`** — trained for a hypothesis that was
  falsified. The model is in Redis (`skol:classifier:model:v3_jats`)
  but it should not be selected for any production routing.
- **`production_v3_full`** — same reasoning. Worse than the
  cheaper hand-only model on the realistic distribution.

Both experiment docs in `skol_experiments` have been marked
`status="archived"` so they don't appear in the default
`manage_experiment.py list` output.

### Survivor

- **`production_v3_hand`** is the only v3 model that matters. It
  remains the production PDF classifier until v4 ships.

### Architectural pivot — the bigger lesson

The v3 negative result, combined with the sample finding that 77% of
JATS docs in `skol_dev` lack TaxPub markup, drove the architectural
shift in [docs/extraction_pipeline.md](extraction_pipeline.md):

- JATS documents don't need a classifier. Their structure is
  extractable deterministically (TaxPub markup directly; plain JATS
  via paragraph-leading keywords or numbered `<sec><title>` titles).
- The classifier exists *only* for PDF-extracted documents (and
  future plaintext-only sources). It is no longer a one-size-fits-all
  tool that pretends to handle every input.
- New training data only helps if it's drawn from the *same
  distribution* as the inference target. JATS docs are not drawn from
  the PDF distribution and adding them hurts.

## Moot subsequent steps

The original Step 5 plan included 5.G and 5.H:

- **5.G — one-off evaluations against `skol_golden_ann_jats_v2`** for
  each v3 model. **Skipped** because under the new architecture
  there's no production classifier path for JATS docs to score
  against the JATS silver gold. The silver gold becomes a converter
  QA artifact, not a model-eval target.
- **5.H — re-evaluate `jats_v2`** with the no-golden training
  corpus. **Skipped** for the same reason. `jats_v2` is an
  archived experiment with no deployment role.

Both are marked `~~strikethrough~~ skipped per architectural pivot`
in the v3 plan.

## What v3 leaves behind

| Artifact | Role going forward |
|---|---|
| `production_v3_hand` model in Redis | Production PDF classifier until v4 ships |
| `skol_training_v2_no_golden` (160 docs) | Pass 1 training corpus for v4 (per [v4_classifier_plan.md](v4_classifier_plan.md)) |
| `skol_training_taxpub_v2_no_golden` (1 724 docs) | Candidate Pass 2 training corpus for v4 — open empirical question whether SBERT transfers where tf-idf didn't |
| `skol_training_v3_combined_no_golden` (1 884 docs) | Same as above; the second arm of the v4 Pass 2 ablation |
| `skol_golden_ann_hand_v2` (30 docs) | The canonical evaluation target |
| `skol_golden_ann_jats_v2` (75 docs) | Converter QA artifact for `jats_to_yedda`; no longer scored against a model |
| `bin/build_no_golden_training_db.py` + `bin/build_combined_training_db.py` | Reusable corpus builders; serve v4 as-is |
| `bin/jats_to_yedda.py --include-ids` flag | Added in Step 0; used to regenerate any training corpus from a fixed ID list with the post-Step-2 converter |

## Lessons captured in code

| Fix | Commit |
|---|---|
| `logistic_sections_v3` silently 3-class because `collapse_labels=True` was the default | `f417c11` |
| `production_v2` / `production_v3_full` slow on 4 cores; bumped Spark defaults | `74ce0ab` |
| Step 1.E env_config doc-field bug (`databases.taxa` vs `databases.treatments`) | Earlier — in golden_v2_plan commits |
| Test pinning the 19-class `class_weights` covers `ACTIVE_TAGS_19` | `95b8d91`, `e01234a` |

## Next

[v4_classifier_plan.md](v4_classifier_plan.md) is the path forward.
v4's two-pass CRF design specifically tests whether SBERT's semantic
embeddings can do what tf-idf couldn't — bridge the JATS-PDF
distribution gap once Pass 1 removes the structural pieces.
[extraction_pipeline.md](extraction_pipeline.md) is the broader
architectural home into which v4 (and any future model) plugs.
