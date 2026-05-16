# Renaming "Taxon" → "Treatment"

## Background

A *taxon* is the organism/group (e.g., *Amanita muscaria*). A *treatment* is
the passage in a publication that describes, names, or revises it. The codebase
stores treatments. Calling them taxa is a category error — one taxon has many
treatments across many papers.

The rename aligns with the domain vocabulary used by Plazi, TaxPub, gnfinder,
GBIF, and the broader biodiversity informatics community.

## Pros

- **Scientifically correct.** Removes a category error that will cause confusion
  once cross-linking between treatments (multiple papers naming the same taxon)
  is implemented.
- **Aligns with external systems.** Every system being integrated (gnfinder,
  gnparser, Plazi, GBIF) uses "treatment" for this concept.
- **Clarifies the two-layer model.** Layer 1 annotates treatment sections; Layer
  2 detects taxon names *within* treatments. Currently "taxon" refers to both
  the container and one of its contents.

## Cons / Risks

**Scope is large.** Every layer of the stack is affected:

| Layer | What changes |
|---|---|
| Python classes | `Taxon` → `Treatment`, `TaxonExtractor` → `TreatmentExtractor` |
| Source files | `taxon.py`, `taxon_test.py`, `extract_taxa_to_couchdb.py`, `embed_taxa.py` |
| CouchDB DB names | `skol_taxa_dev` → `skol_treatments_dev`, `skol_exp_NAME_taxa` → `skol_exp_NAME_treatments` |
| CouchDB field names | `taxon` text field (concatenated Nomenclature); `taxa_db` references |
| Redis keys | `skol:taxa:embedding:{id}` → `skol:treatments:embedding:{id}` |
| Django views | `TaxaInfoView`, `PDFFromTaxaView`, `SourceContextView`; `taxa_db` query param |
| Django API | Response fields referencing "taxa" |
| Frontend | React components, API calls, display labels |
| Config / env vars | `TAXON_DB_NAME`, experiment schema `databases.taxa` |
| Experiment docs | Every live experiment document in CouchDB has `databases.taxa` hardcoded |
| Documentation | `docs/experiments.md`, `docs/api-reference.md` |

**Existing CouchDB data requires migration.** CouchDB has no rename operation —
databases must be copied and deleted. The experiment documents in CouchDB have
`databases.taxa` hardcoded and need a migration script.

**Partial rename is worse than no rename.** Mixing `TreatmentExtractor` with
`taxon_db_name` is more confusing than consistent-but-wrong terminology.

## Recommended approach (when ready)

Do this as a dedicated refactor, not piecemeal alongside other work.

### Step 1 — Rename Python symbols (pure refactor)

Rename classes, variables, and source files. No behavior change; verify with
the existing test suite.

- `taxon.py` → `treatment.py`; `Taxon` → `Treatment`; `group_paragraphs` stays
- `taxon_test.py` → `treatment_test.py`
- `extract_taxa_to_couchdb.py` → `extract_treatments_to_couchdb.py`;
  `TaxonExtractor` → `TreatmentExtractor`
- `embed_taxa.py` → `embed_treatments.py`
- Django views: `TaxaInfoView` → `TreatmentsInfoView`, etc.
- Config keys: `taxon_db_name` → `treatments_db_name`, `TAXON_DB_NAME` →
  `TREATMENTS_DB_NAME`

### Step 2 — Keep CouchDB DB names stable temporarily

Decouple the code rename from the data migration. Map the new config key
`treatments_db_name` to the existing `skol_taxa_dev` / `skol_exp_NAME_taxa`
database names until the data migration is scheduled.

### Step 3 — Migrate CouchDB DB names (schedule separately)

When the next full `extract_treatments` pipeline pass runs:

1. Create new-named databases (`skol_treatments_dev`, etc.).
2. Re-extract into the new databases.
3. Flip config to point at new names.
4. Write a migration script to update `databases.taxa` → `databases.treatments`
   in all live experiment documents.
5. Deprecate and eventually delete the old `*_taxa` databases.

## Progress

Sub-steps are landed one at a time so the test suite stays green between
commits.

### Step 1 — Python symbol rename

| Sub-step | Description | Status |
|---|---|---|
| 1.A | `taxon.py` → `treatment.py`; `class Taxon` → `Treatment`; update all importers (`finder.py`, `taxon_clusterer.py`, `taxa_json_translator.py`, `tests/test_ingest_field.py`, `tests/test_couchdb_file.py`, `couchdb_file.py`, `examples/extract_taxa_from_couchdb.py`, `bin/extract_taxa_to_couchdb.py`, `setup.py`). Docs: `docs/TAXON_PIPELINE_README.md`, `docs/EXTRACTING_TAXON_OBJECTS.md`. | ✅ Done |
| 1.B | `TaxonExtractor` → `TreatmentExtractor` (file `bin/extract_taxa_to_couchdb.py` kept; symbol only). Also updated Spark `appName("SKOL Taxon Extractor")` → `"SKOL Treatment Extractor"`, the docstring `appName("TaxonExtractor")` example, the `[TaxonExtractor]` debug log prefixes, the docstring/log references in `taxa_json_translator.py`, the test importer `tests/test_load_taxa.py`, and example app names (`DistributedTaxonExtractor` → `DistributedTreatmentExtractor`, etc.) in `examples/extract_taxa_from_couchdb.py`. Docs: `TEST_LOAD_TAXA.md`, `TAXON_LOAD_METHOD.md`, `TAXA_JSON_TRANSLATOR.md`, `TAXA_JSON_TRANSLATOR_SUMMARY.md`, `COUCHDB_INTEGRATION_SUMMARY.md`, `TAXA_TRANSLATION_QUICKSTART.md`, `EXTRACTING_TAXON_OBJECTS.md`, `VERBOSITY_CENTRALIZED.md`, `QUICKSTART_COUCHDB.md`, `SESSION_SUMMARY.md`, `TAXA_ROUNDTRIP_EXAMPLE.md`, `TAXA_ID_JOIN_FIX.md`, `couchdb_file_README.md`. | ✅ Done |
| 1.C | Rename `bin/extract_taxa_to_couchdb.py` → `bin/extract_treatments_to_couchdb.py` (+ `_test.py` + `with_skol` symlink). 105 substitutions across ~30 files: `setup.py`, `pyproject.toml`, `taxa_json_translator.py`, `tests/test_load_taxa.py`, `examples/example_taxa_translation.py`, two `fixes/*.py`, `bin/env_config.py`, `bin/manage_experiment.py` (subprocess invocation), `django/search/views.py` (comment), `skol_classifier/couchdb_io.py` (comment), `README.md`, plus 19 `docs/*.md` files. The CLI entry-point name `skol-extract-taxa` is **intentionally kept** for user-facing backward compatibility (only its target module reference changes). | ✅ Done |
| 1.D | Rename `bin/embed_taxa.py` → `bin/embed_treatments.py` (+ `_test.py` + `with_skol` symlink). 50 substitutions across 20 files: `bin/rebuild_redis.py` (per CLAUDE.md rule), `bin/manage_experiment.py` (subprocess call), `django/search/views.py` (subprocess invocation + log strings), `django/skolweb/settings.py` (comment), `django/RECREATE_EMBEDDINGS.sh`, `django/test_api.sh`, `django/QUICKSTART.md`, `django/README.md`, `pyproject.toml`, `setup.py`, `README.md`, `COMPATIBILITY_MODULE.md`, plus 6 `docs/*.md` files. **No Redis key prefix changes** — the plan's table said `skol:taxa:embedding:{id}` → `skol:treatments:embedding:{id}`, but the actual code uses generic `skol:embedding:v1.1` (the build lock is `skol:build:embedding:lock`); neither contains taxon vocabulary. Also fixed pre-existing 1.A leftover in `pyproject.toml` `[tool.setuptools] py-modules` list (`"taxon"` → `"treatment"`). Cross-repo fix: `dr-drafts-mycosearch/src/data.py` was directly `import taxon` (a latent 1.A breakage); changed to `import treatment as taxon` to keep the single `taxon.group_paragraphs(...)` call site working with minimal churn. | ✅ Done |
| 1.E | Rename `bin/taxa_to_json.py` → `bin/treatments_to_json.py` (+ `with_skol` symlink). No `_test.py` exists for this script. 27 substitutions across 4 files: `README.md`, `bin/env_config.py`, `docs/work-skipping-options.md`, plus the renamed script itself. Also renamed the file's single internal function `translate_taxa_to_json` → `translate_treatments_to_json` (only called once inside the same module). | ✅ Done |
| 1.F | Django views: `TaxaInfoView` → `TreatmentsInfoView`, `PDFFromTaxaView` → `PDFFromTreatmentsView`. Touched: `django/search/views.py` (class definitions), `django/search/urls.py` (imports + `path()` references), `docs/GOLDEN_DATASET_AND_EXPERIMENTS.md` (one reference). **Kept**: the URL path `/api/taxa/{id}/`, the URL `name='taxa-info'`/`'taxa-pdf'` (used for Django `reverse()`), the `taxa_id` path parameter, and `docs/api-reference.md` (URL paths only, no class names). These are user-facing API identifiers — renaming would break clients without a deprecation path. No React/frontend references exist for the view class names. | ✅ Done |
| 1.G | Config keys: `taxon_db_name` → `treatments_db_name`; env `TAXON_DB_NAME` → `TREATMENTS_DB_NAME`. The new key still maps to the existing `skol_taxa_dev` DB name (Step 2 compat). | ⬜ Pending |

### Deferred — persisted data (Step 3 territory)

These appear in Python code but encode CouchDB / Spark field names that
downstream consumers read.  They are intentionally **not** renamed in Step 1:

- The dict key `"taxon"` returned by `Treatment.as_row()` (the concatenated
  Nomenclature text) — still emitted to CouchDB / Spark schemas under that
  name.
- Spark `StructField("taxon", StringType(), False)` in
  `bin/extract_taxa_to_couchdb.py`, `taxa_json_translator.py`,
  `bin/taxa_to_json.py`.
- DataFrame `.select("taxon", ...)` references in `taxa_json_translator.py`,
  `bin/taxa_to_json.py`, `examples/example_taxa_translation.py`,
  `tests/test_load_taxa.py`.

These migrate together with the CouchDB DB rename in Step 3.
