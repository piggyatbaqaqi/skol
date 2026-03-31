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
