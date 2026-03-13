# Golden Dataset and Experiment Framework

## Context

The SKOL classifier trains on 190 hand-annotated Mycotaxon/Persoonia documents in `skol_training`, predicts annotations for 30K+ articles in `skol_dev`, and extracts taxa through a multi-stage pipeline. We now auto-generate YEDDA from JATS XML and BioC JSON, but have no way to evaluate annotation quality or compare methods. We need:

1. A **Golden Dataset** — a curated, balanced evaluation set with ground-truth annotations
2. An **Experiment Framework** — named configurations that tie together databases, Redis keys, and classifier models, allowing systematic comparison against the golden dataset
3. A **centralized plaintext extraction** module so `predict_classifier` no longer extracts text from PDFs directly
4. Eventually, **removal of BioC support** once JATS proves superior

## Database Naming

Golden databases drop the `_dev` suffix (no non-dev version planned):

| Database | Purpose |
|---|---|
| `skol_golden` | Curated articles with plaintext — union of all golden sources |
| `skol_golden_ann_hand` | Hand-annotated YEDDA (gold standard, from skol_training) |
| `skol_golden_ann_jats` | JATS-derived YEDDA (silver standard, from TaxPub markup) |
| `skol_golden_ann_bioc` | BioC-derived YEDDA (baseline, to measure before removing bioc) |
| `skol_experiments` | Experiment registry — one document per named experiment |

Experiment databases follow: `skol_exp_{name}_{stage}` (e.g., `skol_exp_jats_v1_taxa`).

---

## Phase 1: Plaintext Extraction Module

**Goal**: Centralize all text extraction. `predict_classifier` will require plaintext (`article.txt`) to already exist rather than extracting from PDFs.

### New file: `ingestors/extract_plaintext.py`

Functions (reusing existing code):

- **`plaintext_from_pdf(pdf_bytes)`** — Wrap `PDFSectionExtractor.pdf_to_text()` from `pdf_section_extractor.py`. Returns text with `--- PDF Page N Label L ---` markers preserved for section-mode compatibility.

- **`plaintext_from_jats(xml_string)`** — Reuse `extract_text()` from `ingestors/jats_to_yedda.py` on the JATS `<body>` element. Preserve section breaks with blank lines.

- **`plaintext_from_bioc(bioc_json)`** — Extract passage text from `bioc_json[0]["documents"][0]["passages"]`. Reuse `clean_passage_text()` from `ingestors/bioc_to_yedda.py`.

- **`plaintext_from_efetch(pmcid, api_key=None)`** — Download plaintext from NCBI E-utilities: `efetch.fcgi?db=pmc&id={pmcid}&retmode=text`. Use `RateLimitedHttpClient`. Respect NCBI rate limits (3 rps without API key, 10 with).

### New file: `ingestors/extract_plaintext_test.py`

Tests for each function with fixture data.

### New file: `bin/extract_plaintext.py` (CLI)

Batch extract plaintext for documents in a CouchDB database and save as `article.txt` attachments:

```
bin/extract_plaintext.py --database skol_dev --source pdf --skip-existing
bin/extract_plaintext.py --database skol_dev --source jats --doc-id abc123
bin/extract_plaintext.py --database skol_dev --source efetch --limit 50
bin/extract_plaintext.py --database skol_dev --source auto  # try each source in priority order
```

Source priority for `--source auto`: efetch (PMC articles) > downloaded txt > PDF > JATS > BioC.

### Modify: `ingestors/pmc.py`

Add `--download-text` flag. When set, after downloading JATS XML, also fetch plaintext via E-utilities `efetch` with `retmode=text` and store as `article.txt` attachment. This is the preferred plaintext source for PMC articles.

### Modify: `ingestors/pensoft.py`

Add `--download-text` flag. When set, after downloading the PDF, call `plaintext_from_pdf()` to extract text and store as `article.txt` attachment. Pensoft already downloads PDFs, so we extract plaintext from the PDF rather than from JATS XML.

### Modify: `bin/predict_classifier.py`

Remove PDF extraction from the classifier pipeline. In section mode, require `article.txt` to exist (previously extracted by `bin/extract_plaintext.py`). The `PDFSectionExtractor` stays for parsing section structure from text, but `pdf_to_text()` is no longer called by the classifier. Add clear error message when `article.txt` is missing: "Run bin/extract_plaintext first".

### Modify: `debian/skol.cron`

Add `extract_plaintext` step before `predict_classifier` in the cron pipeline.

### Critical existing code to reuse
- `pdf_section_extractor.py:pdf_to_text()` — PDF→text via PyMuPDF
- `pdf_section_extractor.py:parse_text_to_sections()` — section parsing (unchanged)
- `ingestors/jats_to_yedda.py:extract_text()` — JATS recursive text extraction
- `ingestors/bioc_to_yedda.py:clean_passage_text()` — BioC text cleanup
- `ingestors/rate_limited_client.py:RateLimitedHttpClient` — rate-limited HTTP

---

## Phase 2: Golden Dataset Curation

**Goal**: Create a curated, balanced evaluation set. Run once after testing.

### New file: `bin/curate_golden_dataset.py`

**Step 1 — Select hand-annotated documents from `skol_training`**:
- Parse each `article.txt.ann`, count distinct YEDDA tag types
- Require >= N tags (configurable `--min-tags`, default 4) for "full complement"
- All 190 current docs have no `source` field (hand-annotated)

**Step 2 — Obtain plaintext** (priority order, NEVER from YEDDA):
1. Downloaded `article.txt` in skol_dev (if exists)
2. PDF-extracted text (call `plaintext_from_pdf()` on `article.pdf`)
3. JATS-derived text (call `plaintext_from_jats()` on `article.xml`)
4. BioC-derived text (call `plaintext_from_bioc()` on `bioc_json`)
5. E-utilities efetch (for PMC articles with pmcid)

**Step 3 — Select PMC+Pensoft JATS articles from `skol_dev`**:
- Filter: `xml_available: true` and `xml_format: "jats"`
- NOT already in the hand-annotated selection
- Generate JATS-derived YEDDA using `jats_xml_to_yedda()`
- Also generate BioC-derived YEDDA where `bioc_json_available: true`
- Configurable limit: `--jats-limit N`

**Balance recommendation**: If ~150 hand-annotated docs qualify, include ~50-75 JATS articles (~1:2 to 1:3 ratio of JATS to hand). The golden set should be ~200-225 documents total — large enough to be statistically meaningful, small enough to curate carefully. Recommend `--jats-limit 75` as default. Ensure JATS articles span multiple journals (MycoKeys, IMA Fungus, etc.) for diversity.

**Step 4 — Populate golden databases**:
- `skol_golden`: All selected articles with metadata + `article.txt` + `article.pdf` (if available) + `article.xml` (if available). This is the UNION of all sources.
- `skol_golden_ann_hand`: Hand-annotated `.txt.ann` for docs from skol_training
- `skol_golden_ann_jats`: JATS-derived `.txt.ann` for docs with JATS XML
- `skol_golden_ann_bioc`: BioC-derived `.txt.ann` for docs with BioC JSON

Each golden doc tracks provenance:
```python
doc['golden_sources'] = {
    'hand_annotated': True/False,
    'jats_available': True/False,
    'bioc_available': True/False,
    'has_pdf': True/False,
    'pmcid': 'PMC...' or None,
    'plaintext_source': 'efetch' | 'pdf' | 'jats' | 'bioc',
}
```

CLI:
```
bin/curate_golden_dataset.py --min-tags 4 --jats-limit 75 --dry-run
bin/curate_golden_dataset.py --all --verbosity 2
```

---

## Phase 3: Experiment Framework

**Goal**: A named-experiment system that ties together databases, Redis keys, and notes. Stored in `skol_experiments` CouchDB database.

### Experiment document schema

```json
{
    "_id": "production",
    "notes": "Current production pipeline: logistic regression on hand-annotated training data",
    "status": "deployed",
    "databases": {
        "ingest": "skol_dev",
        "training": "skol_training",
        "taxa": "skol_taxa_dev",
        "taxa_full": "skol_taxa_full_dev"
    },
    "redis_keys": {
        "classifier_model": "skol:classifier:model:logistic_sections_v2.0",
        "embedding": "skol:embedding:v1.1",
        "menus": "skol:ui:menus_latest"
    },
    "evaluation": null,
    "created_at": "...",
    "updated_at": "..."
}
```

Status values: `draft`, `testing`, `evaluated`, `deployed`, `archived`.

An experimental run:
```json
{
    "_id": "jats_training_v1",
    "notes": "Test whether JATS-derived annotations produce better taxa than hand-annotated training",
    "status": "evaluated",
    "databases": {
        "ingest": "skol_dev",
        "training": "skol_golden_ann_jats",
        "taxa": "skol_exp_jats_v1_taxa",
        "taxa_full": "skol_exp_jats_v1_taxa_full"
    },
    "redis_keys": {
        "classifier_model": "skol:classifier:model:jats_training_v1",
        "embedding": "skol:embedding:jats_training_v1",
        "menus": "skol:ui:menus_jats_training_v1"
    },
    "evaluation": {
        "macro_f1": 0.82,
        "per_tag": { "Nomenclature": 0.95, "Description": 0.78, ... },
        "taxa_yield": 1234,
        "nomenclature_match_rate": 0.91,
        "evaluated_at": "..."
    }
}
```

### New file: `bin/manage_experiment.py`

CRUD operations for experiments:

```
bin/manage_experiment.py create --name jats_v1 --notes "..." --training-db skol_golden_ann_jats
bin/manage_experiment.py list
bin/manage_experiment.py show production
bin/manage_experiment.py run jats_v1 --steps train,predict,extract-taxa,embed,build-menus
bin/manage_experiment.py evaluate jats_v1 --golden skol_golden
bin/manage_experiment.py deploy jats_v1   # promote to production
bin/manage_experiment.py archive old_experiment
```

The `run` subcommand assembles pipeline components:

| Step | Script | Reads from experiment | Writes to experiment |
|---|---|---|---|
| `train` | `train_classifier.py` | `databases.training` | `redis_keys.classifier_model` |
| `predict` | `predict_classifier.py` | `databases.ingest` + classifier model | `.ann` attachments |
| `extract-taxa` | `extract_taxa_to_couchdb.py` | `databases.ingest` | `databases.taxa` |
| `embed` | `embed_taxa.py` | `databases.taxa` | `redis_keys.embedding` |
| `build-menus` | `build_vocab_tree.py` | `databases.taxa_full` | `redis_keys.menus` |
| `evaluate` | `evaluate_golden.py` | predictions + golden | `evaluation` field |

The `deploy` subcommand updates the `production` experiment record to point at this experiment's databases and Redis keys. It also updates `skol:ui:menus_latest` to alias the experiment's menus.

### Modify: `bin/env_config.py`

Add:
```python
'experiment_name': _get_env('EXPERIMENT_NAME', 'production'),
'experiments_database': _get_env('EXPERIMENTS_DATABASE', 'skol_experiments'),
```

### Modify: `bin/rebuild_redis.py`

Add experiment awareness: `--experiment NAME` flag reads experiment config from `skol_experiments` and passes appropriate database/Redis key arguments to each component.

---

## Phase 4: Evaluation Framework

**Goal**: Compare annotation methods quantitatively against the golden dataset.

### New file: `bin/evaluate_golden.py`

**Metrics** (per-tag and macro-averaged):

1. **Tag-level P/R/F1**: Parse YEDDA into blocks `(text, tag)`. Align predicted vs. ground truth by text overlap.
2. **Token-level IoU**: Per-character tag agreement.
3. **Confusion matrix**: Tag-by-tag misclassification counts.
4. **Taxa comparison**: Yield, nomenclature match rate, spurious rate, description overlap.

**Comparison pairs** (for each experimental classifier):
- Classifier vs. all of Golden (skol_golden union)
- Classifier vs. hand annotations (gold standard)
- Classifier vs. JATS annotations (silver standard)
- JATS vs. hand annotations (measures JATS quality)

**Output**: Markdown report + JSON stored in experiment's `evaluation` field.

### New file: `bin/evaluate_golden_test.py`

Tests with synthetic YEDDA fixtures.

### Reuse
- `yedda_parser/__init__.py:parse_yedda_string()` — parse YEDDA format
- `ingestors/bioc_to_yedda.py:Tag` enum — canonical tag names

---

## Phase 5: Django Experiment Integration

**Goal**: Experiment pulldown in Django that switches search, taxa, and menus.

### Modify: `django/search/models.py`

Add to `UserSettings`:
```python
default_experiment = models.CharField(max_length=100, default='production')
```

### Modify: `django/search/serializers.py`

Add `default_experiment` to `UserSettingsSerializer`.

### Modify: `django/search/views.py`

Key views that need experiment awareness:
- `SearchView` — use experiment's embedding key
- `TaxaInfoView` — use experiment's taxa DB
- `EmbeddingListView` — filter to experiment's embedding
- `BuildVocabTreeView` — use experiment's taxa_full DB and menus key
- `sources_view()` in `skolweb/urls.py` — use experiment's ingest DB

Pattern: read user's `default_experiment`, load experiment config from `skol_experiments`, use experiment's database/Redis key names.

### New API endpoint: `GET /api/experiments/`

List available experiments (name, notes, status) for the pulldown.

### Modify: `django/templates/index.html`

Add experiment `<select>` dropdown in settings menu (after "Search Settings" header, before embedding selector). Changing it calls `saveSetting('default_experiment', value)` and reloads the page.

---

## Phase 6: Existing Script Modifications

### `bin/bioc_to_yedda.py` (line 106) and `bin/jats_to_yedda.py` (line 97)

Parameterize `source_database` to use the actual `--database` argument value instead of hardcoding `"skol_dev"`.

### `bin/predict_classifier.py`

Add `--output-database` option: when set, write predicted `.ann` as documents in the specified database (matching bioc_to_yedda/jats_to_yedda format with `source`, `source_database`, `article.txt.ann` attachment) instead of as attachments on source documents.

### `debian/postinst` and `debian/postinst.template`

Add new scripts to symlink loop: `extract_plaintext`, `curate_golden_dataset`, `manage_experiment`, `evaluate_golden`.

---

## Phase 7: BioC Removal (execute last)

**Goal**: Remove all BioC support after golden dataset evaluation confirms JATS is superior.

### Steps
1. Remove `bioc_json` and `bioc_json_available` fields from all `skol_dev` documents
2. Remove `ingestors/bioc_to_yedda.py` and `ingestors/bioc_to_yedda_test.py`
3. Remove `bin/bioc_to_yedda.py`
4. Remove `--download-bioc-json` from PMC ingestor (`ingestors/pmc.py`)
5. Remove BioC-related tests from `ingestors/pmc_test.py`
6. Remove `skol_golden_ann_bioc` database
7. Remove bioc cron entries from `debian/skol.cron`
8. Update `debian/postinst` to remove `bioc_to_yedda` symlink
9. Remove `plaintext_from_bioc()` from `ingestors/extract_plaintext.py`

### Prerequisite
Run golden dataset evaluation (Phase 4) first and confirm JATS annotations meet or exceed BioC quality on all metrics.

---

## Files Summary

### New files
| File | Purpose |
|---|---|
| `ingestors/extract_plaintext.py` | Plaintext extraction from PDF, JATS, BioC, efetch |
| `ingestors/extract_plaintext_test.py` | Tests |
| `bin/extract_plaintext.py` | CLI for batch plaintext extraction |
| `bin/curate_golden_dataset.py` | One-shot golden dataset curation |
| `bin/manage_experiment.py` | Experiment CRUD and pipeline orchestration |
| `bin/evaluate_golden.py` | Evaluation metrics and comparison |
| `bin/evaluate_golden_test.py` | Tests |

### Modified files
| File | Change |
|---|---|
| `ingestors/pmc.py` | Add `--download-text` (efetch plaintext) |
| `ingestors/pensoft.py` | Add `--download-text` (JATS→plaintext) |
| `bin/predict_classifier.py` | Require plaintext; add `--output-database` |
| `bin/bioc_to_yedda.py` | Parameterize `source_database` |
| `bin/jats_to_yedda.py` | Parameterize `source_database` |
| `bin/env_config.py` | Add `experiment_name`, `experiments_database` |
| `bin/rebuild_redis.py` | Add `--experiment` flag |
| `debian/skol.cron` | Add `extract_plaintext` before `predict_classifier` |
| `debian/postinst{,.template}` | Add new script symlinks |
| `django/search/models.py` | Add `default_experiment` to UserSettings |
| `django/search/serializers.py` | Add `default_experiment` field |
| `django/search/views.py` | Experiment-aware search, taxa, embeddings |
| `django/search/urls.py` | Add `/api/experiments/` endpoint |
| `django/templates/index.html` | Experiment pulldown in settings |

---

## Verification

1. **Phase 1**: `pytest ingestors/extract_plaintext_test.py -v` passes. Run `bin/extract_plaintext.py --database skol_dev --source pdf --limit 5` and verify `article.txt` attachments appear.
2. **Phase 2**: Run `bin/curate_golden_dataset.py --min-tags 4 --jats-limit 75 --dry-run`. Verify document counts and balance. Then run without `--dry-run`. Verify all golden databases exist and have correct documents.
3. **Phase 3**: `bin/manage_experiment.py create --name test1 --notes "test"`. Verify document in `skol_experiments`. Run `bin/manage_experiment.py list`.
4. **Phase 4**: `bin/evaluate_golden.py --experiment production --golden skol_golden`. Verify report output.
5. **Phase 5**: Django migration, verify experiment pulldown appears in settings, switching changes search results.
6. **Phase 6**: Verify parameterized `source_database` and `--output-database` work correctly.
7. **Phase 7**: Only after evaluation confirms JATS >= BioC. Run removal steps and verify no bioc references remain.

## Implementation Order

Phases 1-2-3-4 should be implemented sequentially (each builds on prior). Phase 5 (Django) can proceed in parallel with Phase 4. Phase 6 modifications are prerequisites needed during Phases 1-3. Phase 7 executes only after evaluation data from Phase 4 is reviewed.
