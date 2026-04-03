# Annotation Next Steps

This document summarises the current state of the annotation infrastructure
and lists the remaining work in priority order.

---

## What is complete

### Training corpus alignment (this sprint)

- **`bin/seed_dev_from_training.py`** — seeded 8 `skol_training` documents
  that lacked `skol_dev_id` into `skol_dev` (PDF copied, `article.txt`
  extracted via `PDFSectionExtractor`, `skol_dev_id` back-filled).  One doc
  was skipped (no PDF attachment).

- **`bin/fix_staging_yedda.py`** — applied to all 190 `skol_staging` documents:
  - Fixed malformed YEDDA blocks caused by missing `*]` delimiters and OCR
    noise (`[@` sequences from scanned footnote markers or botanical abbreviations).
    The fixer now iterates until no `[@` remains in any block text.
  - Recovered `--- PDF Page N Label L ---` page markers from `skol_dev`
    `article.txt` for all documents that have a `skol_dev_id`.

- **brat files regenerated** — 190 `.txt`/`.ann` pairs in
  `brat/data/skol/` are clean (no `[@` noise).

### Infrastructure (earlier sessions)

| Component | Location |
|---|---|
| 12-tag YEDDA label set | `ingestors/yedda_tags.py` |
| JATS sec-type → 12-tag mapping | `ingestors/jats_to_yedda.py` |
| Automatic label migration (Tier 1) | `bin/migrate_labels.py` |
| LLM-assisted relabeling (Tier 2) | `bin/llm_relabel.py` |
| YEDDA ↔ brat converters | `bin/yedda_to_brat.py`, `bin/brat_to_yedda.py` |
| Annotation upload | `bin/upload_annotation.py` |
| Span utilities | `ingestors/gnfinder_client.py`, `ingestors/gnparser_client.py`, `ingestors/particle_detector.py`, `ingestors/spans.py` |
| Span annotation script | `bin/annotate_spans.py` (pipeline step `annotate_spans`) |
| Treatment assembly (12-tag) | `taxon.py` — `group_paragraphs()` redesign, flat section fields |
| Multi-section embeddings | `bin/embed_taxa.py` — primary, distribution, biology embeddings |
| Experiment pipeline | `bin/manage_experiment.py` — `runnext`, `runstep`, `resetstep`, `skipstep` |
| Evaluation (12-tag aware) | `bin/evaluate_golden.py` — `--collapse-tags` flag |

---

## Remaining work

### 1. Relabel training corpus to 12-tag scheme

The 190 `skol_training` documents still carry the old 8-tag labels.
`llm_relabel.py` upgrades them using the Claude API, writing the result to
a staging database for human review before touching canonical data.

```bash
# Estimate token cost first
python bin/llm_relabel.py --database skol_training --estimate

# Run relabeling → writes to skol_training_llm_stage
python bin/llm_relabel.py --database skol_training
```

### 2. Human brat review

Convert the staging database to brat, review in the brat UI, then
round-trip the corrected annotations back to CouchDB.

```bash
# Export to brat
python bin/yedda_to_brat.py \
    --staging-db skol_training_llm_stage \
    --output-dir /path/to/brat/data/skol_review/

# After annotating in brat:
python bin/brat_to_yedda.py article.txt article.ann > article.txt.ann
python bin/upload_annotation.py DOC_ID article.txt.ann \
    --database skol_training_llm_stage
```

Brat is configured for the `skol` entity set in `brat/data/skol/`.
The `annotation.conf` and `visual.conf` there list all 12 tags.

### 3. Retrain the classifier

Once the relabeled corpus is reviewed:

```bash
# Add logistic_sections_v3 config in bin/train_classifier.py (12-class,
# inverse-frequency class weights from new label distribution), then:

python bin/manage_experiment.py resetstep taxpub_v1_onnx_int8 train
python bin/manage_experiment.py runstep  taxpub_v1_onnx_int8 train
python bin/manage_experiment.py runstep  taxpub_v1_onnx_int8 evaluate --force
```

Evaluation will now show rows for Diagnosis, Distribution,
Materials-examined, Type-designation, Biology, and the other new tags.

### 4. Regenerate taxa in CouchDB

`taxon.py` was redesigned for the 12-tag state machine and carries new flat
section fields (`diagnosis`, `distribution`, `materials_examined`, etc.) and
section-specific span tracking.  The CouchDB taxa records need to be
re-extracted to pick up these fields.

```bash
python bin/extract_taxa_to_couchdb.py \
    --experiment taxpub_v1_onnx_int8 --force
```

The document IDs will change (the SHA-256 hash now covers all section texts).

### 5. Run span annotation

`annotate_spans.py` is ready and integrated as the `annotate_spans` pipeline
step.  It requires gnfinder and gnparser services reachable from the server.

```bash
# Check service URLs in env_config / .skol_env, then:
python bin/manage_experiment.py runstep taxpub_v1_onnx_int8 annotate_spans
```

Span records are written as `article.spans.json` to the annotations database.

### 6. Django search integration (Phase 3)

Once span records exist in sufficient volume:

- Overlay `<mark class="entity entity-{label}">` highlights in the Source
  Context Viewer for TaxonName, Author, MB-number, DOI, and
  Fungarium-code spans.
- Add a `react-select` pulldown for treatment-section search
  (Description/Distribution/Biology) in `django/search/views.py`.
- Add external cross-links: GBIF for TaxonName, MycoBank for MB-number,
  doi.org for DOI.

Files: `django/search/views.py`, `django/search/templates/search/`,
`django/frontend/src/`.

### 7. Span-enhanced classifier (deferred — Phase 4)

After span corpus coverage is measurable, add span density features to
`SkolClassifierV2`: TaxonName count, MB-number presence, SP_NOV annotation.
Strong prior: MB-number → likely Nomenclature; dense names+authors →
Nomenclature or Type-designation.

---

## Quick-reference: brat annotation round-trip

```
skol_staging[doc_id].article.txt.ann   (YEDDA, canonical)
        │
        ▼  bin/yedda_to_brat.py --staging-db skol_staging --output-dir brat/
brat/data/skol/{doc_id}.txt + .ann     (brat standoff — annotate here)
        │
        ▼  bin/brat_to_yedda.py {doc_id}.txt {doc_id}.ann
article.txt.ann                        (YEDDA, updated)
        │
        ▼  bin/upload_annotation.py {doc_id} article.txt.ann
skol_staging[doc_id].article.txt.ann   (updated in CouchDB)
```

YEDDA is the canonical storage format.  brat is the annotation UI only.
The round-trip is lossless for section-level tags.
