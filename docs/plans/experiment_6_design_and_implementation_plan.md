# SKoL Experiment 6: Layered Segmentation Pipeline — Design and Implementation Plan

## 1\. Problem statement

The current segmentation pipeline mixes several distinct abstraction layers into one CRF pass: page-layout recognition (headers, footers, captions, tables), article/treatment boundary detection, and line-level semantic labeling all influence the same transition matrix. This causes the model to treat label transitions and physical text boundaries as equivalent signals, producing spurious splits mid-paragraph and mid-sentence. Line granularity is currently the raw newline from PDF/XML extraction, which is unreliable on its own — OCR artifacts frequently produce one or two words per physical line.

The goal of Experiment 6 is to decompose this into independently trainable layers, each with a narrow responsibility, its own algorithm, and its own training signal.

## 2\. Proposed architecture

1. **Layout filter** — strips or tags page headers, footers, figure captions, and tables. Physical/layout knowledge only; no taxonomy content.  
2. **Article segmenter** — splits a source document (which may be a single born-digital article or a whole scanned issue/book) into individual articles, identifying front-matter regions (title, abstract, introduction, bibliography).  
3. **Treatment detector** — within a taxonomic article, locates treatment boundaries, informed by journal-specific house style. Handles cases like a new genus immediately followed by a new species, where each taxonomic level is a separate treatment.  
4. **Line labeler (CRF)** — assigns a section label (description, habitat, distribution, taxonomic citation, etc.) to each line within a single treatment. Existing SBERT \+ CRF architecture, retrained once layout noise is removed upstream.  
5. **Span grouper** — merges label-homogeneous line runs into coherent spans. Owns paragraph/sentence integrity as a hard constraint rather than an emergent property of CRF label transitions. Includes a learned line-join classifier and separate handling for keys/tables, which don't reduce to sentence extraction.

## 3\. Per-layer design

### 3.1 Layout filter

- **Algorithm**: lightweight per-line classifier (logistic regression or small MLP) on positional and repetition features. Supplemented by an unsupervised signal: near-identical text recurring at the same page position across consecutive pages is a strong header/footer indicator independent of any label.  
- **Training data**: reusable from the existing CRF training set — page headers, figure captions, and tables are already first-class labels there.  
- **Complexity estimate**: **2 units**

### 3.2 Article segmenter

- **Algorithm**: sequence classifier over line/paragraph units for front-matter categories (title, abstract, introduction, bibliography), plus boundary detection between articles for multi-article sources.  
- **Training data**: check first whether the corpus's ingestion metadata already encodes article boundaries for free — born-digital sources (Plazi, BHL, BioStor) are typically downloaded one article per file, which is usable boundary supervision with no new labeling. Older scanned issues/books will need new labels, since front-matter categories aren't captured by the current schema at all.  
- **Complexity estimate**: **5 units**

### 3.3 Treatment detector

- **Algorithm**: hybrid — a house-style-driven pattern matcher (using per-journal GGserver configs from Plazi) with an ML fallback classifier for out-of-config journals or ambiguous cases.  
- **Training data**: two sources. (1) Distant supervision from Plazi's TreatmentBank output for journals they've already processed — validate the detector against their existing treatment segmentation rather than hand-labeling from scratch. (2) For journals outside Plazi's coverage, new labeling informed by the accumulated style notes from Claude-API label validation, covering variant styles such as new-genus-then-new-species.  
- **Complexity estimate**: **8 units**

### 3.4 Line labeler (CRF)

- **Algorithm**: unchanged — SBERT embeddings with a CRF head (`torchcrf` or `sklearn-crfsuite`), regex/gazetteer features for reliable label locking (e.g. MycoBank number → taxonomic citation).  
- **Training data**: existing BIO-labeled data, largely reusable. Requires retraining once layout labels are removed from the transition matrix, since the label space and feature set both simplify.  
- **Complexity estimate**: **3 units**

### 3.5 Span grouper

- **Algorithm**: a per-boundary line-join classifier (join vs. break) rather than a single global heuristic. Features include line length relative to a local running median (not a fixed threshold), run-length of preceding short/long lines, terminal punctuation, mid-hyphenation, capitalization of the following line, and the CRF's own line label (so keys/tables get different join behavior than prose). Sentence segmentation tools (spaCy sentencizer, Punkt, pySBD) are only applied after reflow, and only to spans labeled as prose — keys and tables get a structure-aware joiner instead.  
- **Training data**: silver data first — where a raw line-broken version and a clean/reflowed version of the same text both exist (schema-constrained pipeline output, extracted description fields), align them to auto-derive join/break labels with no manual annotation. New labeling reserved for cases silver alignment can't resolve, particularly heavy-OCR documents with no clean counterpart.  
- **Complexity estimate**: **5 units**

## 4\. Data strategy

- **Silver vs. gold separation**: keep auto-derived (silver) and carefully annotated (gold) labels clearly separated in the training pipeline for the span grouper and treatment detector, to avoid compounding errors — especially since Mistral outputs are already reused as schema seeds elsewhere in the project.  
- **Evaluation**: extend the existing difficulty-stratification approach (description length, feature density, Latin phrasing, nesting load) to build held-out eval sets per new layer, so each layer's quality can be verified independently before it feeds the next.

## 5\. Complexity estimates

Unit definition: 1 complexity unit ≈ the effort to implement a simple ingest for one new journal (i.e., a straightforward house-style configuration case).

| Layer | Estimate (units) |
| :---- | :---- |
| Layout filter | 2 |
| Article segmenter | 5 |
| Treatment detector | 8 |
| Line labeler (CRF) retrain | 3 |
| Span grouper | 5 |
| Cross-layer eval framework | 3 |
| Pipeline integration / orchestration | 3 |
| **Total** | **\~29** |

These are engineering-judgment estimates, not measured effort — intended for relative sequencing and prioritization, not a committed schedule.

## 6\. Open questions / risks

- Does the corpus's ingestion metadata actually preserve article-level file boundaries for born-digital sources? This materially changes the article segmenter estimate if not.  
- How much of the corpus has a "clean" reflowed counterpart usable for span grouper silver-data alignment, versus requiring new labeling?  
- What fraction of target journals are covered by Plazi's TreatmentBank versus requiring net-new treatment boundary labeling?  
- Risk of error compounding if silver labels derived from one layer's output are used to train a downstream layer without a gold-labeled check.

## 7\. Suggested phasing

1. Layout filter (low complexity, unblocks CRF retrain)  
2. Line labeler retrain (validates that layout removal actually improves labeling quality)  
3. Span grouper (addresses the most visible current failure — paragraph/sentence splitting)  
4. Article segmenter  
5. Treatment detector (highest complexity; benefits from lessons learned in the article segmenter)  
6. Cross-layer eval framework and full pipeline integration, running in parallel with the above

---

## 8. Scope additions and ordering — 2026-09-10 planning session

Sections 1–7 describe the *segmentation* rewrite.  Four items were
added to Experiment 6's scope in the 2026-09-10 session, and the
ordering between them is not free.

### 8.1 Structural OCR — do this first, before §7's phase 1

**Replace text-only extraction with layout-aware OCR producing
markdown.**  Designed already in
[../structure_ocr.md](../structure_ocr.md) (2026-09-01): tool
evaluation, RTX 5090 / 24 GB VRAM optimisation, reference framework.

It belongs **before** the layout filter, not after, because it changes
that filter's input from lines to headed structure.  Two measurements
say how much it buys:

* §12.3.56 — **79.3 % of article-level section headers are buried
  mid-block** (280 of 353 in a 400-document sample), 240 of them inside
  `Misc-exposition`.  Markdown headings make the article segmenter's
  primary signal explicit instead of inferred.
* §12.3.58 — front matter is labelled by *register*, not structure, so
  an introduction reviewing a genus's morphology is labelled
  `Description`.  With headings present this is a lookup, not a
  classification.

**Record OCR provenance in the same change.**  §12.3.64 found the
ingest document carries bibliographic fields only — **no engine, no
confidence, no language**.  Nothing says how `article.txt` was produced
or how well.  That is cheap at ingest and impossible afterwards; it
would make "is this document worth processing" a stored field instead
of a heuristic argued from token shapes, and it is the precondition
[lotl-detector.md](lotl-detector.md) now depends on.

### 8.2 Multi-model pipeline on the Plazi GGServer pattern

Separation of concerns per §2's layers, with the GGServer workflow as
the model.  **Sequence it after 8.1**: building the separation against
unstructured text means building it twice.

### 8.3 Break `Biology` into `Habitat` / `Host` / `Distribution`

Listed by the operator as a lesser change.  **It is not lesser, because
of what it blocks.**  It changes the label set, and the label set is
the input to §8.4's supervised framework, to any fine-tuning corpus,
and to the 83 fixtures in `tests/fixtures/pathologies.json`.  **Build a
training set before this lands and you build it twice.**

The corpus already draws the distinctions:

| self-labelled sub-section | treatments | share |
|---|---:|---:|
| `Distribution:` | 4 196 | 5.1 % |
| `Habitat:` | 2 308 | 2.8 % |
| `Host:` / `Known host:` | 813 | 1.0 % |
| **host *and* distribution, separately labelled in one treatment** | **476** | 0.6 % |

§12.3.62 has the derivation; §12.3.52 corroborates independently
(4 536 `biology` fields contain the literal word `Distribution:`);
Trello #407 and §12.3.1 are the original argument.  **A fourth
candidate: `Substrate`** — distinct from host for saprobes, which is
most of this corpus.

This is also the destination of T6 in
[annotation-activity-split.md](annotation-activity-split.md): a label
string carrying a qualifier the schema should carry structurally
(`Asci in culture MEA`, `Squamules`) is the same defect at feature
level that `Biology` is at section level.  **T6's two traps transfer**:
`annotation_doc_id` has no deletion pass, so a re-run must target a new
candidate DB; and it must re-use the same round file with `--force`.

### 8.4 Supervised framework from accumulated labelling feedback

Last, because it consumes a label set that must be stable — i.e. after
8.3.  Inputs that exist today: rounds 1–7 in
`data/annotation_rounds/`, 2 396 hand-verified annotations in
`…_02_55_features_hand`, 25 882 canonical ones in
`…_02_57_features_canonical`, and 83 fixtures with `expected_flags` and
`known_missed_flags`.

### 8.5 Not breaking the UI and the experiment infrastructure

**The data is not the coupling — the label set is.**  Experiment 6 gets
its own `skol_exp_experiment6_*` databases and the Django experiment
selector already handles parallel experiments; `docs/couchdbs.md` is
updated per `CLAUDE.md` when they are created.

What breaks is everything keyed to the 19 tags: `_LABEL_TO_FIELD` and
`_LABEL_TO_SPANS_FIELD` in `treatment.py`, `ACTIVE_TAGS_19` in
`ingestors/yedda_tags.py`, `skol_classifier/v4/labels.py`'s three
projections, the brat converters, and the fixtures.

**So one piece of infrastructure comes before 8.3 and 8.2:** make the
label set a **versioned artifact** — tag list, label→field map, and
projection rules as a dated, loadable object rather than module
constants, with v4 and experiment 6 side by side.  It is small, and it
is what lets 8.3 land without a flag day.

**Bind the version to the model that uses it** (operator, 2026-09-10).
A global "current label set" reintroduces the coupling under a new
name.  Instead every model artifact — a CRF checkpoint, an SBERT
cache, a fine-tuned SLM, a prompt seed file — **declares the label-set
version it was built against**, and the pipeline refuses a mismatch
rather than silently projecting one vocabulary through a model trained
on another.  That makes the 8.3 split a *data* change: train a new
model against label set v6, leave the v4 model pinned to v4, and run
both.

Three places where the missing declaration has already cost something,
as the evidence that this is the right shape:

* `skol_exp_production_v4_03_00_treatments_structured` records no
  `model` field, so nothing on 1 285 documents says which model or
  which input granularity produced them — see M6 in
  [production-v5-execution.md](production-v5-execution.md), where that
  gap produced a wrong reading.
* The candidate annotations *do* carry `model: claude-opus-4-7` on all
  25 742, which is why the Claude side of the same question was
  answerable in one query.  The difference between those two databases
  is the whole argument.
* T6 in [annotation-activity-split.md](annotation-activity-split.md):
  `annotation_doc_id` has no deletion pass, so re-running a changed
  prompt into the same database yields the **union** of two
  vocabularies.  A declared version turns that silent corruption into
  a refusal.

### 8.6 Relationship to production_v5

There is no experiment 5.  `production_v5` in
[production-v5-execution.md](production-v5-execution.md) is the next
*extraction run of the current architecture*; Experiment 6 is a
*replacement architecture*.  Different axes, not successive versions.
**M4/M5 there are at risk of being overtaken** — if this pipeline
lands, a v5 run of the single-CRF architecture buys little.  M1–M3
(detector work on v4) and M6 (structured JSON) are unaffected and run
independently of everything here.

### 8.7 Revised phasing

Merging §7 with the above:

| # | step | note |
|---|---|---|
| 0 | Versioned label set | small; unblocks 3 and 5 without a flag day |
| 1 | **Structural OCR → markdown, with provenance** (8.1) | upstream of everything; retires two §12.3 classes |
| 2 | Layout filter | §7 phase 1, now against headed structure |
| 3 | **`Biology` split** (8.3) | before any training corpus is built |
| 4 | Line-labeler retrain | §7 phase 2, on the new label set |
| 5 | Span grouper | §7 phase 3 |
| 6 | Article segmenter | §7 phase 4; markdown headings make this cheap |
| 7 | Treatment detector | §7 phase 5 |
| 8 | Multi-model pipeline integration (8.2) | §7 phase 6 |
| 9 | Supervised framework (8.4) | last; needs 3 |

**Blocking neither, but blocking round 8 and M5:** T5 of
[annotation-activity-split.md](annotation-activity-split.md) — the
label review of round 7, 45 treatments, apparatus ready.  It is the
only outstanding item from that plan and it gates the next draw.

**Not in Experiment 6, and deliberately parallel to it:** M6's
structured-JSON work, the two user-visible increments recorded in the
production_v5 change log (surfacing `predicted_issues`, harvesting the
~2 757 documents from §12.3.61), and
[lotl-detector.md](lotl-detector.md), which is scheduled *after* this
plan because 8.1 supplies its precondition.

## 9. Cross-references

* [../structure_ocr.md](../structure_ocr.md) — 8.1's design
* [../schema_constrained_pipeline.md](../schema_constrained_pipeline.md) — taxon-agnostic Pass A/Pass B; M6's destination
* [../structured-form-schema.md](../structured-form-schema.md) — the JSON schema decisions
* [../data_quality_production_v4_model.md](../data_quality_production_v4_model.md) — §12.3, the evidence base
* [production-v5-execution.md](production-v5-execution.md) — M1–M3, M6
* [annotation-activity-split.md](annotation-activity-split.md) — complete; T6 lands in 8.3
* [clade-agnostic-detectors.md](clade-agnostic-detectors.md) — binding on 8.2's vocabularies
* [lotl-detector.md](lotl-detector.md) — after this plan

## Change log

* **2026-09-10** — §§8–9 added from the planning session: structural
  OCR with provenance, the Plazi-pattern pipeline, the `Biology` split
  and its sequencing constraint, the supervised framework, the
  versioned-label-set precondition, the production_v5 relationship, and
  a revised phasing that merges these with §7.
