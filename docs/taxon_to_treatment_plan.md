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

Run this **twice**: first against the dev environment (local CouchDB), then
against prod via SSH to `synoptickeyof.life`.  Validate end-to-end on dev
before scheduling prod — the prod pass is the irreversible one.

The per-environment checklist:

1. Create new-named databases (`skol_treatments_dev` for dev,
   `skol_treatments` and `skol_exp_NAME_treatments` for prod).
2. Re-extract into the new databases via a full `extract_treatments`
   pipeline pass.
3. Flip config to point at new names: change the default in
   `bin/env_config.py` and `django/skolweb/settings.py` from
   `'skol_taxa_dev'` → `'skol_treatments_dev'` (and the prod equivalent);
   remove the `TAXON_DB_NAME` env-var fallback and the
   `TAXON_DB_NAME = TREATMENTS_DB_NAME` Django alias once Step 2 compat
   is no longer needed.
4. Write a migration script to update `databases.taxa` →
   `databases.treatments` in all live experiment documents (shared
   between dev and prod runs).
5. Migrate the persisted strings listed in "Deferred — persisted data"
   below: the `"taxon"` field returned by `Treatment.as_row()`, the
   Spark `StructField("taxon", ...)` schemas, the Redis-stats dict key
   `'taxa_db_name'`.
6. Deprecate and eventually delete the old `*_taxa` databases on each
   server (after a holding period to allow rollback).

Run mechanics:

- **Dev pass**: run locally against `http://localhost:5984` first.  Use
  this pass to verify the migration script, validate the re-extracted
  data against `skol_taxa_dev` (sample compares), and shake out any
  remaining missed references.
- **Prod pass**: once dev is signed off, run via SSH to
  `synoptickeyof.life`.  Use the same migration script and config
  changes; the env-var/Django alias removals from step (3) should land
  in the code only after **both** environments have been migrated.

## Progress

Sub-steps are landed one at a time so the test suite stays green between
commits.

### Step 1 — Python symbol rename — ✅ Complete

All seven sub-steps below landed across 8 commits in `skol` plus 1 cross-repo
fix in `dr-drafts-mycosearch`.  Test suite went from 60/6 baseline (before
rename) to 75/6 after 1.D made `bin/embed_treatments_test.py` collectable;
no rename introduced a new failure.

| Sub-step | Description | Status |
|---|---|---|
| 1.A | `taxon.py` → `treatment.py`; `class Taxon` → `Treatment`; update all importers (`finder.py`, `taxon_clusterer.py`, `taxa_json_translator.py`, `tests/test_ingest_field.py`, `tests/test_couchdb_file.py`, `couchdb_file.py`, `examples/extract_taxa_from_couchdb.py`, `bin/extract_taxa_to_couchdb.py`, `setup.py`). Docs: `docs/TAXON_PIPELINE_README.md`, `docs/EXTRACTING_TAXON_OBJECTS.md`. | ✅ Done |
| 1.B | `TaxonExtractor` → `TreatmentExtractor` (file `bin/extract_taxa_to_couchdb.py` kept; symbol only). Also updated Spark `appName("SKOL Taxon Extractor")` → `"SKOL Treatment Extractor"`, the docstring `appName("TaxonExtractor")` example, the `[TaxonExtractor]` debug log prefixes, the docstring/log references in `taxa_json_translator.py`, the test importer `tests/test_load_taxa.py`, and example app names (`DistributedTaxonExtractor` → `DistributedTreatmentExtractor`, etc.) in `examples/extract_taxa_from_couchdb.py`. Docs: `TEST_LOAD_TAXA.md`, `TAXON_LOAD_METHOD.md`, `TAXA_JSON_TRANSLATOR.md`, `TAXA_JSON_TRANSLATOR_SUMMARY.md`, `COUCHDB_INTEGRATION_SUMMARY.md`, `TAXA_TRANSLATION_QUICKSTART.md`, `EXTRACTING_TAXON_OBJECTS.md`, `VERBOSITY_CENTRALIZED.md`, `QUICKSTART_COUCHDB.md`, `SESSION_SUMMARY.md`, `TAXA_ROUNDTRIP_EXAMPLE.md`, `TAXA_ID_JOIN_FIX.md`, `couchdb_file_README.md`. | ✅ Done |
| 1.C | Rename `bin/extract_taxa_to_couchdb.py` → `bin/extract_treatments_to_couchdb.py` (+ `_test.py` + `with_skol` symlink). 105 substitutions across ~30 files: `setup.py`, `pyproject.toml`, `taxa_json_translator.py`, `tests/test_load_taxa.py`, `examples/example_taxa_translation.py`, two `fixes/*.py`, `bin/env_config.py`, `bin/manage_experiment.py` (subprocess invocation), `django/search/views.py` (comment), `skol_classifier/couchdb_io.py` (comment), `README.md`, plus 19 `docs/*.md` files. The CLI entry-point name `skol-extract-taxa` is **intentionally kept** for user-facing backward compatibility (only its target module reference changes). | ✅ Done |
| 1.D | Rename `bin/embed_taxa.py` → `bin/embed_treatments.py` (+ `_test.py` + `with_skol` symlink). 50 substitutions across 20 files: `bin/rebuild_redis.py` (per CLAUDE.md rule), `bin/manage_experiment.py` (subprocess call), `django/search/views.py` (subprocess invocation + log strings), `django/skolweb/settings.py` (comment), `django/RECREATE_EMBEDDINGS.sh`, `django/test_api.sh`, `django/QUICKSTART.md`, `django/README.md`, `pyproject.toml`, `setup.py`, `README.md`, `COMPATIBILITY_MODULE.md`, plus 6 `docs/*.md` files. **No Redis key prefix changes** — the plan's table said `skol:taxa:embedding:{id}` → `skol:treatments:embedding:{id}`, but the actual code uses generic `skol:embedding:v1.1` (the build lock is `skol:build:embedding:lock`); neither contains taxon vocabulary. Also fixed pre-existing 1.A leftover in `pyproject.toml` `[tool.setuptools] py-modules` list (`"taxon"` → `"treatment"`). Cross-repo fix: `dr-drafts-mycosearch/src/data.py` was directly `import taxon` (a latent 1.A breakage); changed to `import treatment as taxon` to keep the single `taxon.group_paragraphs(...)` call site working with minimal churn. | ✅ Done |
| 1.E | Rename `bin/taxa_to_json.py` → `bin/treatments_to_json.py` (+ `with_skol` symlink). No `_test.py` exists for this script. 27 substitutions across 4 files: `README.md`, `bin/env_config.py`, `docs/work-skipping-options.md`, plus the renamed script itself. Also renamed the file's single internal function `translate_taxa_to_json` → `translate_treatments_to_json` (only called once inside the same module). | ✅ Done |
| 1.F | Django views: `TaxaInfoView` → `TreatmentsInfoView`, `PDFFromTaxaView` → `PDFFromTreatmentsView`. Touched: `django/search/views.py` (class definitions), `django/search/urls.py` (imports + `path()` references), `docs/GOLDEN_DATASET_AND_EXPERIMENTS.md` (one reference). **Kept**: the URL path `/api/taxa/{id}/`, the URL `name='taxa-info'`/`'taxa-pdf'` (used for Django `reverse()`), the `taxa_id` path parameter, and `docs/api-reference.md` (URL paths only, no class names). These are user-facing API identifiers — renaming would break clients without a deprecation path. No React/frontend references exist for the view class names. | ✅ Done |
| 1.G | Config keys: `taxon_db_name` → `treatments_db_name`; env `TAXON_DB_NAME` → `TREATMENTS_DB_NAME`. Env lookup chain in `bin/env_config.py` is `TREATMENTS_DB_NAME` → `TAXON_DB_NAME` (deprecated fallback) → default `'skol_taxa_dev'`; `django/skolweb/settings.py` mirrors that chain and additionally retains `TAXON_DB_NAME = TREATMENTS_DB_NAME` as an alias for any unmigrated callers (Step 2 compat). 42 substitutions of `taxon_db_name` across 14 files (`taxa_json_translator.py`, `bin/build_vocab_tree.py`, `bin/build_sources_stats.py`, `bin/embed_treatments.py`, `bin/extract_treatments_to_couchdb.py`, `tests/test_load_taxa.py`, `examples/example_taxa_translation.py`, plus 7 `docs/*.md`); plus surgical edits to `django/skolweb/settings.py` (new var + alias + env chain), `django/skolweb/urls.py` (`settings.TAXA_DB_NAME` → `TREATMENTS_DB_NAME` and local var rename), `django/search/views.py` (subprocess env-var name), `bin/build_sources_stats.py` (local var rename, help text), `bin/extract_treatments_to_couchdb.py` (argparse error message). **Kept**: the persisted Redis-stats dict key `'taxa_db_name'` and the user-facing query param `?taxa_db=` (serialized / public-API contracts — Step 3 territory). The underlying CouchDB database name still defaults to `skol_taxa_dev` until Step 3. | ✅ Done |

### Deferred — persisted data (Step 3 territory)

These appear in Python code but encode CouchDB / Spark field names that
downstream consumers read.  They are intentionally **not** renamed in Step 1:

- The dict key `"taxon"` returned by `Treatment.as_row()` (the concatenated
  Nomenclature text) — still emitted to CouchDB / Spark schemas under that
  name.
- Spark `StructField("taxon", StringType(), False)` in
  `bin/extract_treatments_to_couchdb.py`, `taxa_json_translator.py`,
  `bin/treatments_to_json.py`.
- DataFrame `.select("taxon", ...)` references in `taxa_json_translator.py`,
  `bin/treatments_to_json.py`, `examples/example_taxa_translation.py`,
  `tests/test_load_taxa.py`.
- Redis-stats dict key `'taxa_db_name'` written by `bin/build_sources_stats.py`.
- URL query param `?taxa_db=<id>` and JSON response key `'taxa_db'` in
  `django/search/views.py`.
- URL path `/api/taxa/{id}/` and Django route names `'taxa-info'` /
  `'taxa-pdf'`.
- `experiment.databases.taxa` field in stored experiment documents (the
  `databases.taxa` → `treatments_db_name` mapping in `bin/env_config.py`
  reads from this stored field).
- Env-var deprecation fallback `TAXON_DB_NAME` and Django setting alias
  `TAXON_DB_NAME = TREATMENTS_DB_NAME`.
- The default DB name `'skol_taxa_dev'` in `bin/env_config.py` and
  `django/skolweb/settings.py`.

These migrate together with the CouchDB DB rename in Step 3.

### Step 2 — Code/data decoupling — ✅ Complete

Absorbed into 1.G's design.  No separate work was needed: the new
`treatments_db_name` config key defaults to `'skol_taxa_dev'`, the env-var
lookup chain accepts `TAXON_DB_NAME` as a deprecated fallback, and
`django/skolweb/settings.py` keeps `TAXON_DB_NAME = TREATMENTS_DB_NAME` as
an alias.  Existing CouchDB databases (`skol_taxa_dev`,
`skol_exp_NAME_taxa`) and unmigrated `.skol_env` files continue working
unchanged.

### Step 3 — Pending operational scheduling

Two passes — dev first, then prod.  See the Step 3 section above for the
full per-pass checklist.

| Pass | Target | Mechanism | Status |
|---|---|---|---|
| 3-dev | Local CouchDB at `http://localhost:5984` | `bin/migrate_taxa_to_treatments.py --execute` | ✅ Data migrated (DB + doc + experiment-doc rewrites). Code-side cutover still pending — see below. |
| 3-prod | `https://synoptickeyof.life:5984` | SSH to skol@synoptickeyof.life | ⬜ Not started |

3-dev outcome (run 2026-05-16):

- `skol_taxa_dev` (25,420 docs), `skol_taxa_full_dev` (12,302),
  `skol_taxa_taxpub_v1_dev` (43,046) replicated to their `*_treatments_*`
  counterparts in ~35s total via `_replicate`.
- All 80,768 docs in the new DBs had their persisted `taxon` field
  renamed to `treatment` via bulk-doc updates.
- All 6 docs in `skol_experiments` had `databases.taxa` →
  `databases.treatments` (and `taxa_full` → `treatments_full`) with
  their values rewritten via the same underscore-component rule —
  including non-local prod-only references, so the docs are
  prod-replication-ready.
- Originals untouched; rollback = `curl -X DELETE` each new DB.

3-dev code-side cutover (next):

- Flip default in `bin/env_config.py` (`treatments_db_name` default
  `'skol_taxa_dev'` → `'skol_treatments_dev'`) and the matching
  Django setting.
- Change Spark consumers (`taxa_json_translator.py`,
  `bin/extract_treatments_to_couchdb.py`, `bin/treatments_to_json.py`)
  to read the `treatment` field from documents instead of `taxon`,
  and to write `treatment` in their output schemas.
- Re-point any Django views / pipelines at the new DB defaults via
  experiment docs (already done — experiment docs now refer to
  `databases.treatments`).
- Validate via spot-check + a re-run of the relevant pipelines
  against the new DBs.

Mid-run fix that landed during 3-dev: couchdb-python's
`server.replicate()` helper builds short-form URLs with a `http://any`
placeholder hostname that fails Erlang DNS resolution.  The migration
script was updated to POST directly to `_replicate` with full source
and target URLs plus explicit `auth.basic` blocks for each side.  The
same fix will apply to 3-prod.

Sequencing notes:

- Run **3-dev first** so the migration script and re-extracted data can
  be validated against the existing `skol_taxa_dev` (sample compares,
  spot checks).  Treat dev as the rehearsal — prod is irreversible.
- The config-default flip (`'skol_taxa_dev'` → `'skol_treatments_dev'`)
  and the removal of the `TAXON_DB_NAME` env fallback / Django alias
  should land in the code only after **both** environments have been
  migrated, so neither is mid-flight when a code deploy goes out.
- A migration script for live experiment documents (`databases.taxa` →
  `databases.treatments`) is shared between the two passes; write it
  once during 3-dev, re-run it during 3-prod.
- The "Deferred — persisted data" list above is the full set of
  strings, dict keys, Spark schemas, and stored experiment-doc fields
  that move atomically with this rename.
