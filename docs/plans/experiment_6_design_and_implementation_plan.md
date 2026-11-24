# SKoL Experiment 6: Layered Extraction — Design and Implementation Plan

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

Steps 1 and 4 are revised by §8.8 (JATS sources); duplicate
detection (§8.9) is added after steps 6 and 7; §8.10 discusses
reflow before labelling.  §8.11 adds step 3b (training-set
import and span transfer) and §8.12 the scan finder.

**Trello:** the "Layered Extraction" list holds one card per step,
titled `E6-<step> — <name> [<cu>]`: E6-0, 1a (#331), 1b, 2, 3 (#407),
3b, 3c (`Misc-exposition`), 3d (stratified rounds), 5a, 4, 5b, 6, 6b
(#405, document dedup), 7, 7b (treatment dedup), 8, eval, 9, and 12
(§8.12).  5a sits before 4 because the labeler's unit is the block.

**Not in Experiment 6, and deliberately parallel to it:** M6's
structured-JSON work, the two user-visible increments recorded in the
production_v5 change log (surfacing `predicted_issues`, harvesting the
~2 757 documents from §12.3.61), and
[lotl-detector.md](lotl-detector.md), which is scheduled *after* this
plan because 8.1 supplies its precondition.

### 8.8 JATS sources: one document model, two readers (accepted 2026-11-23)

**The gap** (operator, 2026-11-23): §8.1 makes markdown the input to
every layer, and §8.7 says nothing about where JATS XML enters.
Converting JATS to markdown to slot it in would throw away what the
publisher already states (article parts, section types, figures and
tables as elements, and for TaxPub the treatment boundaries) and then
ask each layer to infer it again.

**What happens today.**  Code references are from a read of the
current tree; the sample is a read of production, 2026-11-23.

* `plaintext_from_jats` (`ingestors/extract_plaintext.py:52-83`)
  keeps only `<body>` and strips all markup.  The front matter, the
  abstract, the back matter, every `sec-type` and every TaxPub element
  are lost.  PMC's `article.txt` is made this way.
* For Pensoft, `article.txt` comes from the PDF even when TaxPub XML
  is attached (`ingestors/pensoft.py:827`; `bin/extract_plaintext.py`
  prefers the PDF).
* Plain JATS gets no deterministic labelling at all.  `jats_to_yedda`,
  the G.1 sweep and the TaxPub inspector all require TaxPub markup.
  That is all of PMC, and 77 % of JATS documents
  (`docs/extraction_pipeline.md`).
* Even TaxPub's treatment boundaries are not used as boundaries.
  `treatment_assembler.py:60-102` writes the blocks back out as YEDDA,
  regroups them by Nomenclature paragraphs, and attaches the anchors
  by index.
* v4's `predict` step runs before `annotate_jats`
  (`bin/pipelines/v4_crf.py:47-64`).  Both write `article.txt.ann`
  into the same database with `--skip-existing`, so a TaxPub document
  that has a PDF keeps whichever step ran first.  In a sample of 30 of
  the 912 production ingest documents that carry both a PDF and TaxPub
  XML, 27 had the JATS-derived annotation and 3 had CRF labels on PDF
  text.
* The one attempt to use JATS for training, `production_v3_jats`,
  scored macro F1 0.132 against 0.459 for hand-trained
  (`docs/extraction_pipeline.md:7-14`).  It trained on passages and
  was applied to physical lines.

**Proposal.**

1. **A document model, not a text format, is the contract between
   layers.**  Blocks in reading order, each with:
   * a kind: heading, paragraph, list item, table, figure, caption, or
     key couplet;
   * its text and its section path (its heading ancestry);
   * its source span: page and box, or line range, for OCR; the
     element id or path for JATS;
   * zero or more **assertions**: a layer's output that the source
     states, with its origin.  Examples: "treatment starts here"
     (`taxpub`), "section label Description" (`jats-sec-type`),
     "running header" (`ocr-layout`).

   Markdown from §8.1 and JATS are two **readers** into the model.
   Markdown stays useful as one way to print it, for display and for
   prompts, but it is not the interchange format.
2. **Every layer has one contract.**  Read the model.  Where a block
   already carries an assertion for this layer, keep it, after a
   sanity check that logs any disagreement.  Elsewhere, infer.  That
   is how "works with markdown or JATS" gets written once, not once
   per layer per format.
3. **What each source asserts, layer by layer:**

   | layer | TaxPub | plain JATS | structural OCR |
   |---|---|---|---|
   | layout filter | nothing to filter | nothing to filter | infer |
   | article segmenter | asserted: one article per file; front, body, back and references are elements | asserted, as TaxPub | infer, helped by headings |
   | treatment detector | asserted (`tp:taxon-treatment`) | infer, from asserted sections and titles | infer |
   | block labeler (§3.4) | asserted (`treatment-sec@sec-type`) | infer | infer |
   | span grouper | paragraphs asserted; keys and tables still need the structure-aware joiner | as TaxPub | reflow, then join |

   The treatment detector and the labeler are the layers that must
   genuinely handle both formats, and plain JATS gains the most from
   them.
4. **The labeler's unit is the block, for every source.**  This is the
   lesson of `production_v3_jats`.  §8.1 already makes OCR output
   paragraph-level, so the two sources converge.  The step 4 retrain
   should be specified at block level, not line level.  Open question:
   for OCR, does reflow (the span grouper) move ahead of labelling?
   See §8.10.
5. **The JATS projections belong to the versioned label set (§8.5).**
   `sec_type_to_tag` and `JATS_EMIT_TAGS` map publisher vocabulary into
   SKOL labels, so they are a projection like v4's, and the `Biology`
   split (§8.3) changes them.
6. **Evaluation data for free.**  The 912 production documents that
   carry both a PDF and TaxPub XML let the OCR path run on the PDF and
   be scored against the XML's assertions, on the same documents: §4's
   per-layer evaluation sets for article segmentation, treatment
   detection and block labels, with no new labelling.  Publisher markup
   has its own errors, so this is silver, not gold (§4).
7. **Provenance (§8.1) records the reader.**  It says which source was
   read and which assertions came from it.  A document with both PDF
   and XML is read from the XML by default.

**Phasing.**  Step 1 of §8.7 becomes:
* **1a**, the document model and the markdown reader (§8.1);
* **1b**, the JATS reader.  It is small:
  `jats_xml_to_tagged_blocks` (`ingestors/jats_to_yedda.py:488-539`)
  already walks the right elements, and it would emit model blocks
  with assertions instead of YEDDA text.

Step 4 becomes the block-labeler retrain.  The suggested estimate
(Claude, 2026-11-23) is 3 units for 1b, in §5's units.

**Found while checking, outside Experiment 6:**
* the step-order collision above (Anomalies backlog,
  https://trello.com/c/jfd7fqGj);
* the dedup in `extract_treatments_to_couchdb.py:880-898`, which
  assumes both paths produce the same content-hash `_id` and so the
  same text.  When the `.ann` comes from PDF text they probably don't,
  which would leave a duplicate treatment.  This is inferred from the
  code and not verified.  It becomes part of §8.9;
* the index-matched anchors in the assembler (Anomalies backlog,
  https://trello.com/c/2WPRzA3R).

### 8.9 Duplicate detection is a stage (operator, 2026-11-23)

Duplicates must be detected at two levels, the document and the
treatment, and the choice between copies must use quality metrics.
This is a stage of the pipeline, not a side effect of how ids are
made.

**Why it has to be a stage.**  U3 in
`docs/data_quality_production_v4_model.md` (Trello #405) measured it
on `production_v4`, 2026-08-23:
* 36.7 % of ingest documents share a DOI with another.  98.3 % of
  those groups are a `crossref` PDF copy and a `pmc` JATS copy of the
  same article.
* 41.2 % of treatments come from an article ingested more than once.
  At least 7 389 (9.1 %) are provably redundant.
* The content-hash `_id` is not a dedup mechanism.  Byte-identical
  treatments from two ingests collapse and overwrite each other's
  provenance.  Near-identical ones never collapse: normalised-text
  hashing catches under 40 % of the provable cases.
* No source dominates.  "Prefer PMC" would discard the better
  extraction on 79.6 % of duplicated articles, so the choice needs a
  per-copy quality signal.
* DOI alone is blind to whole-volume scans with no DOI (the
  *Persoonia* volumes of #404).

**Design.**
1. **Document level, after the article segmenter (§8.7 step 6).**
   Whole volumes and issues must be split into articles before copies
   can be compared, so the unit is the article, not the ingest
   document.  Keys: DOI where present; otherwise a title, volume and
   page key, backed by a content similarity measure (shingled text)
   for documents with neither.
2. **Treatment level, after the treatment detector (step 7).**
   Within a duplicated article, pair the treatments by nomenclature
   (name, author, MycoBank number) and text similarity.  Across
   articles, the same treatment can also be republished, as in a
   monograph quoting a protologue.  Those are kept, but linked, not
   merged.
3. **Choose by quality, keep everything.**  Each duplicate cluster
   records all its copies and names one **preferred** copy.  The
   choice is made from per-copy quality, not from a source ranking.
   The signals are `assess_treatments` verdicts (`clean`, `flags`,
   `usable_text`), description length and coverage, OCR provenance
   (§8.1), and whether the source asserts the structure (§8.8).
   Non-preferred copies are kept with a pointer to the preferred one,
   so provenance survives and the choice can be re-made when the
   metrics change.  The cluster records the metric version it was
   decided under, as `checks_version` does.
4. **Source identity belongs in the treatment id**, so two ingests
   never silently overwrite each other.  Sameness is the cluster's
   job, not the hash's.
5. **The JATS/PDF pair is the main case, and §8.8 makes it
   tractable.**  With both copies read into one document model, the
   JATS copy's assertions and the PDF copy's inferred layers can be
   compared block by block.  That is the same comparison §8.8 uses for
   evaluation.

**Phasing.**  Document-level dedup goes after step 6, treatment-level
after step 7, and both before step 8, so the integrated pipeline
serves preferred copies only.  This depends on the quality
assessment (`bin/assess_treatments.py`) running on Experiment 6
output.  The suggested estimate (Claude, 2026-11-23) is 5 units, in
§5's units, for both levels.

### 8.10 Span grouping before and after labelling (proposed 2026-11-23)

The operator asked whether there is a compelling reason for reflow to
happen after labelling.  There is not.

**The only reason in the plan** is §3.5's use of the line label as a
join feature, so that keys and tables join differently from prose.
That distinction is a block *kind*, though, not a section label, and
it is visible from layout: couplet numbering, dot leaders, columns.
Against reflowing late: labelling physical lines is what produces
§1's spurious mid-paragraph splits, and §8.8 makes the labeler's unit
the block, which must exist before labelling.

**Proposal: two passes, with different criteria.**

1. **Before labelling: typographic grouping.**  It rebuilds blocks
   from lines using hyphenation, line length against a running median,
   terminal punctuation, capitalisation, indentation, and page and
   column breaks.  It assigns block kinds (prose, key, table, list).
   * JATS asserts all of this, so the pass is a no-op there.
   * For structural OCR, which already emits paragraphs, it is a repair
     pass: join paragraphs split by page or column breaks or by
     figures, and split ones the OCR ran together.
   * **Training data:** the 912 PDF-plus-TaxPub documents (§8.8,
     item 6) give silver join/break labels for free, by aligning the
     XML's paragraphs with the PDF's lines.  This replaces §3.5's
     silver source.
2. **After labelling: semantic grouping.**  It merges consecutive
   blocks with the same label into one span, such as a description
   running over several paragraphs.  It applies sentence segmentation
   to prose spans only, as §3.5 already says.

**Open choice: what happens to a block that changes label part-way
through?**  An inline sub-section head is the common case: the corpus
self-labels `Distribution:` in 5.1 % of treatments and `Habitat:` in
2.8 % (§8.3), often run into the same paragraph.  Either:
* **(a)** pass 1 also splits at run-in heads, which are typographic
  (a bold or italic lead word, then a colon), so every block is
  label-homogeneous and pass 2 only merges; or
* **(b)** the labeler labels sentences within a block, and pass 2
  splits where the label changes.

(a) keeps the labeler simple and puts a regex-friendly cue in the
typographic pass.  (b) catches changes that have no head.  The
recommendation is (a) first, measuring how many label changes it
leaves unsplit.

### 8.11 Models and training data by layer (2026-11-23)

**The pattern** (operator, 2026-11-23): Claude labels a training set,
a person spot-checks it, and a local model is trained on the result.
`bin/llm_relabel.py` already runs this as Claude → staging database →
brat review → upload.  It fits where the labels need judgement and no
source states them.  Where JATS or TaxPub states them (§8.8), the
source supplies the labels.  Those are silver too, and get the same
spot-check (§4).

| Layer | Model | Status | Training labels |
|---|---|---|---|
| Structural OCR (§8.1) | layout-aware OCR | off the shelf (`structure_ocr.md`); not trained here | evaluated only, on the 912 PDF + TaxPub pairs |
| Markdown and JATS readers (§8.8) | none | deterministic | — |
| Layout filter (§3.1) | small classifier plus a repetition heuristic | new, small; may shrink to almost nothing if OCR output is clean | the hand set's `Page-header` (5 443 spans) and `Figure` (451) |
| Typographic grouping (§8.10 pass 1) | join/break classifier, block kind, run-in-head split | new | silver from aligning the 912 XML/PDF pairs; Claude only for damaged OCR with no XML copy |
| Article segmenter (§3.2) | sequence classifier: article boundaries and article parts | new | JATS `front`/`body`/`back` (asserted); **Claude** for whole-volume scans, where nothing is asserted |
| Document dedup (§8.9) | matching rules plus a similarity threshold | rules | **Claude** judges a sample of near-duplicate pairs to set the threshold |
| Treatment detector (§3.3) | house-style pattern matcher with an ML fallback | new; the largest | TaxPub `tp:taxon-treatment` (asserted); **Claude** for plain JATS and OCR journals |
| Treatment dedup (§8.9) | name, author, MycoBank number, text similarity | rules | **Claude** judges a sample of pairs, including linked-not-merged republications |
| Block labeler (§3.4) | SBERT + CRF | **existing, retrained** on blocks and the new label set | the adjusted hand set (below); TaxPub `sec-type` (asserted); **Claude** relabelling via `llm_relabel.py` |
| Semantic grouping (§8.10 pass 2) | merge rules plus an off-the-shelf sentence splitter | no training, unless §8.10 option (b) | (b) needs sentence-level labels, from Claude |
| Quality checks (`assess_treatments`) | rule checks | existing | none; they feed §8.9's choice of copy |

Three layers get new trained models (article segmenter, treatment
detector, typographic grouping), one is retrained (block labeler), and
the layout filter may barely need one.

#### Adjusting the original training set

The hand set is 190 YEDDA files, 160 for training and 30 for gold
(`skol_golden_ann_hand_v2`): 169 Mycotaxon (mostly volumes 117–119,
2011–12), 20 Persoonia (volumes 16–19) and 1 Mycologia.

1. **Transfer the spans onto the new blocks.**  The YEDDA labels are
   already on blocks, not lines (operator), so the change of unit is
   small.  The transfer is still needed: the new readers produce
   different text and different block boundaries from the PDF
   extraction the spans were drawn on, so each span is carried to the
   new blocks by text alignment.  A new block covered by two old labels
   goes to review, not to a guess; that is usually §8.10's run-in head.
2. **Split the labels by layer, in a label store (proposed).**  One
   sequence today holds three layers' labels:
   * `Page-header`, `Figure`, `Section-header`: layout filter and block
     kinds;
   * `Title`, `Author`, `Institution`, `Key-words`, `Bibliography`,
     `Contents`: article segmenter;
   * the starts of `Nomenclature` runs: treatment detector boundaries;
   * labels inside a treatment: block labeler.

   *Library over raw YEDDA, or split files?*  Neither alone.  Import
   the YEDDA files once into the document model (§8.8) as assertions
   with origin `hand-v2` and a layer each.  The YEDDA files are then a
   frozen archive, and the import is a tested, repeatable function of
   them: the stability of reading raw files.  After the import, each
   layer's labels are edited in the store, one layer and one version
   at a time: the editability of split files, without split files
   drifting apart.  brat review exports one layer's view.
3. **Eliminate `Misc-exposition` (operator: if feasible).**  It is
   28 482 of the hand set's ~40 800 spans, so the CRF mostly learns
   "not a treatment".  It is a catch-all because the old label set had
   no names for what it holds.  The replacement:
   * outside treatments, the article segmenter's parts: abstract,
     introduction, methods, results, discussion, acknowledgements,
     bibliography, appendix;
   * inside treatments, named sections, including commentary
     (`Notes`, `Remarks`, `Discussion`) and `Specimens examined`;
   * a residual `Unclassified`, which is a QC signal and is never
     trained on.

   **Feasibility is a measurement:** Claude labels a sample of the
   28 482 spans with these labels, a person spot-checks it, and the
   label is dropped if the residual is small.  Sample size is set for
   95 % confidence on the residual rate.
4. **The `Biology` split (§8.3), approved.**  The hand set's label is
   `Habitat-distribution` (139 spans).  Claude relabels it into
   `Habitat` / `Host` / `Distribution` / `Substrate`, a person reviews
   in brat, and the result is a new label-set version (§8.7 step 0),
   written to a new database per T6.
5. **Vary by time first, then by house style (operator).**  The hand
   set covers about two decades of two journals.  Claude-labelled
   rounds are drawn stratified by decade, then by house style (journal
   or series):
   * the mykoweb literature, already ingested
     (`ingestors/local_mykoweb_literature.py`; 219 works in
     `/data/skol/www/mykoweb.com/systematics/literature`, from 1874
     on), and the 307 protologues in `…/CAF/protologue`;
   * early Mycotaxon and Persoonia, Annales Mycologici, and the NY State
     Museum reports, already in the corpus.

   *Systema Mycologicum* (Latin, 1821) is out of scope.  Many old works
   are also on the rescan list (`data/ocr_rescan_targets.md`).  Label
   them after re-OCR where that is cheap; otherwise span transfer
   (item 1) carries the labels to the re-read text.
   `production_v3_jats` failed by training on passages and applying to
   lines.  At block level that mismatch is gone, so whether
   TaxPub-derived labels now help is re-tested, not assumed.
6. **Gold is gold (operator).**  The 30 gold files are transferred
   too, and a person reviews every transferred block.  Claude never
   labels a document it is scored against.
7. **Keep the three origins apart:** hand gold, publisher-asserted,
   and Claude-labelled.  Every label records its origin (item 2's
   store), and each spot-check is sized for 95 % confidence.

**Data note.**  The hand set holds some labels that look like typos or
notes, not labels: `m` (2 spans), `AnnotatorComment` (1), `Comment`
(1), and the singletons `Common-name` (1) and `Contents` (3).  The
import (item 2) sends them to review instead of creating labels.

**Phasing.**  A new §8.7 step **3b**, after the `Biology` split and
before the block-labeler retrain: the label store and import, span
transfer, and the `Misc-exposition` measurement.  The time-stratified
Claude rounds start once the label set is settled, and feed steps 4,
6 and 7.  Suggested estimates (Claude, 2026-11-23), in §5's units:
3 for 3b, 3 for the `Misc-exposition` replacement if the measurement
supports it, and 5 for the stratified rounds.

### 8.12 Finding better scans from OCR quality (proposed 2026-11-23)

**Operator, 2026-11-23:** use OCR quality metrics to pick documents
that would benefit from an independent scan.  The Internet Archive
and Google Books hold several scans of many public-domain works.

**Inputs that exist.**
* `treatments_to_structured.ocr_damage` (rejoin, substitution and
  intra-word rates) and the ranked works in
  `data/ocr_rescan_targets.md`, with its caution: rank by per-article
  tails, not volume medians.
* §8.1's provenance and per-word engine confidence, once recorded.

**The tool.**
1. Rank works by measured damage and keep those with no better copy.
2. Distinguish a bad OCR pass from a bad scan, using image resolution
   and engine confidence.  A bad pass over good images (Mycotaxon
   3–31, 300 ppi) needs re-OCR, not a new scan.  A bad scan (Persoonia
   1–19, 150 ppi) needs another source.
3. For the second kind, search for other scans: Internet Archive by
   title, author, year and BHL item; Google Books.  Fetch candidates
   and score a sample of pages with the same metrics.
4. Ingest the winner as **another copy**.  §8.9's duplicate clusters
   then choose the preferred copy by quality, so a new scan never
   silently replaces an old one.  This is why it follows §8.9 (#405).

Trello: https://trello.com/c/8N63rz8g (Backlog until the Layered
extraction list exists).

## 9. Cross-references

* [../structure_ocr.md](../structure_ocr.md) — 8.1's design
* [../extraction_pipeline.md](../extraction_pipeline.md) — today's JATS/PDF dispatch; 8.8's starting point
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
* **2026-11-23** — §8.8 proposed: JATS sources enter through a shared
  document model with per-source assertions, not through markdown.
  Revises §8.7 steps 1 and 4.  Accepted by the operator the same day,
  and §8.9 added: duplicate detection at document and treatment level,
  chosen by quality metrics, as a pipeline stage.
  §8.10 proposed: span grouping in two passes, typographic before
  labelling and semantic after.  Renamed "Layered extraction" by the
  operator (formerly "Layered Segmentation Pipeline", and "Plazi-style
  extraction" in conversation); the Trello list uses the same name.
  §8.11 added: models and training data by layer, and the adjustments
  to the hand-annotated training set (label store, span transfer,
  `Misc-exposition` replacement, time-stratified rounds).  §8.12
  proposed: finding better scans from OCR quality.  Steps carded on
  the Layered Extraction Trello list.
