# CouchDB Databases

A reference for every CouchDB database in the SKOL local CouchDB instance —
where it comes from, what it's for, and which other databases / experiments
it is tied to.  Inventory is as of 2026-05-19 from `localhost:5984`; doc
counts and sizes shift over time but the role of each DB is durable.

Whenever a new database is added, update the relevant section below.  (See
`CLAUDE.md` — `Whenever we create a new CouchDB database, please update
docs/couchdbs.md`.)

## At a glance

| DB | Docs | Size | Role |
|---|---:|---:|---|
| **Source / ingest** | | | |
| `skol_dev` | 30,967 | 79.7 GB | Primary ingest database. Article metadata plus `article.txt` and `article.pdf` attachments. `article.txt` files are extracted from `article.pdf` either directly or via OCR. Some articles have only `article.xml` which are JATS XML TaxPub.|
| **Training corpora** | | | |
| `skol_training` | 190 | 6.8 GB | Hand-curated training set. Each doc carries `article.txt`, `article.pdf`, and a hand-edited `article.txt.ann`. `skol_dev_id` cross-links to the matching ingest doc. |
| `skol_training_taxpub_v1` | 1,743 | 156.6 MB | JATS/TaxPub-derived training corpus for the `taxpub_v1*` family of experiments. One `article.txt.ann` per doc, plus bibliographic metadata (`doi`, `title`, `authors`). |
| `skol_training_llm_stage` | 180 | 5.3 MB | Staging area for `bin/llm_relabel.py`. Source DB recorded in `source_db`, change list in `changes.json` alongside the relabelled `article.txt.ann`. |
| **Annotation review pipeline** (training-doc lineage) | | | |
| `skol_ann_reviewed` | 190 | 13.5 MB | Human-reviewed YEDDA annotations. Output of the brat round-trip on `skol_training`. |
| `skol_ann_fixed` | 190 | 22.4 MB | Output of `fixes/fix_missing_yedda.py` — gap-filling and page-marker recovery on the `_reviewed` `.ann`s. |
| `skol_ann_merged` | 190 | 980.5 MB | Output of `fixes/merge_yedda.py` followed by `bin/enrich_ann_merged.py`. Merges human-reviewed labels onto the new OCR text from `skol_training`, then enriches with Crossref publication metadata + `is_golden` flag + copied `article.txt`/`article.pdf` attachments. |
| `skol_staging` | 190 | 23.3 MB | Output of `bin/llm_relabel.py` running against `skol_training`. Same shape as `skol_training_llm_stage`. |
| **Golden / evaluation** | | | |
| `skol_golden` | 105 | 1.0 GB | 105 curated articles + `article.txt`/`article.pdf` (and `article.xml` where JATS exists). The union of all golden sources. Built by `bin/curate_golden_dataset.py`. |
| `skol_golden_ann_hand` | 30 | 4.6 MB | 30 hand-annotated `.txt.ann` (gold standard). Sourced from `skol_training`. |
| `skol_golden_ann_jats` | 75 | 6.7 MB | 75 JATS-derived `.txt.ann` (silver standard). Sourced from articles with TaxPub markup. |
| **Treatments (post-Step-3 rename)** | | | |
| `skol_treatments_dev` | 25,420 | 190.5 MB | Primary treatments database. Output of `bin/extract_treatments_to_couchdb.py` against `skol_dev`. Step-3-dev rename of `skol_taxa_dev`; each document's nomenclature field renamed `taxon` → `treatment`. |
| `skol_treatments_full_dev` | 12,302 | 51.9 MB | Treatments enriched with `json_annotated` (structured JSON via `bin/treatments_to_json.py`). Step-3-dev rename of `skol_taxa_full_dev`. |
| `skol_treatments_taxpub_v1_dev` | 43,046 | 261.9 MB | Treatments for the `taxpub_v1*` experiments. Step-3-dev rename of `skol_taxa_taxpub_v1_dev`. |
| **Legacy taxa (pre-rename; retained for rollback)** | | | |
| `skol_taxa_dev` | 25,420 | 160.1 MB | Pre-rename original of `skol_treatments_dev`. Kept until Step 3-prod completes (per `docs/taxon_to_treatment_plan.md`). |
| `skol_taxa_full_dev` | 12,302 | 55.0 MB | Pre-rename original of `skol_treatments_full_dev`. |
| `skol_taxa_taxpub_v1_dev` | 43,046 | 254.5 MB | Pre-rename original of `skol_treatments_taxpub_v1_dev`. |
| `skol_taxa_migration_dev` | 23,009 | 16.2 MB | Feb-2026 dedup mapping table: `old_id → new_id` produced by `fixes/migrate_taxa_ids.py` when taxa IDs moved from provenance-based to content-based hashes. Historical artefact; not a current data store. Excluded by the Step-3 `migrate_taxa_to_treatments.py` denylist. |
| **Experiment-specific annotations** | | | |
| `skol_exp_taxpub_v1_ann` | 18,794 | 1.3 GB | `.ann` (`article.pdf.ann`) plus ingest metadata for the `taxpub_v1*` experiment family. Referenced from `skol_experiments` as `databases.annotations`. |
| `skol_exp_hand_ann` | 105 | 2.8 MB | _One `article.txt.ann` per doc; 105 entries matches `skol_golden`'s size. **INVESTIGATE**: confirm this is the hand-annotated annotations DB for the `hand_annotated` experiment (the experiment's `databases.annotations` field is missing, so this guess can't be confirmed automatically)._ |
| **Django app data** | | | |
| `skol_collections_dev` | 3,979 | 2.8 MB | User-created collections (specimens/observations) synced from the Django app. Each doc has `type: collection` plus the same `treatment`/`description`/`ingest` shape as taxa so unified search works. Written by `django/search/couchdb_sync.py`. |
| `skol_collections_history_dev` | 4,153 | 3.2 MB | Append-only change history for collections (`change_type`, `changed_at`, prior `name`/`nomenclature`/`description`/`owner`). |
| `skol_comments_dev` | 6 | 0.1 MB | Threaded comments on collections (`body`, `author`, `path`, `parent_path`, `edit_history`, soft-delete + hidden flags). |
| **Configuration** | | | |
| `skol_experiments` | 6 | 0.4 MB | Experiment configuration documents — wires together ingest/training/treatments/annotations DBs and Redis keys per experiment. See "Experiments" below. |

## Experiments

`skol_experiments` is the canonical wiring diagram. Each experiment document
points at the databases and Redis keys that step scripts (under
`bin/manage_experiment.py runstep <name> <step>`) read and write.

| Experiment | ingest | training | treatments | treatments_full | annotations | embedding key |
|---|---|---|---|---|---|---|
| `production` | `skol_dev` | `skol_training` | `skol_treatments_dev` | `skol_treatments_full_dev` | `""` _(explicit empty — falls back to ingest)_ | `skol:embedding:v1.1` |
| `taxpub_v1` | `skol_dev` | `skol_training_taxpub_v1` | `skol_treatments_taxpub_v1_dev` | `skol_exp_taxpub_v1_treatments_full` | `skol_exp_taxpub_v1_ann` | `skol:embedding:taxpub_v1` |
| `taxpub_v1_int8` | `skol_dev` | `skol_training_taxpub_v1` | `skol_treatments_taxpub_v1_dev` | `skol_exp_taxpub_v1_int8_treatments_full` | `skol_exp_taxpub_v1_ann` | `skol:embedding:taxpub_v1_int8` |
| `taxpub_v1_onnx_int8` | `skol_dev` | `skol_training_taxpub_v1` | `skol_treatments_taxpub_v1_dev` | `skol_exp_taxpub_v1_onnx_int8_treatments_full` | `skol_exp_taxpub_v1_ann` | `skol:embedding:taxpub_v1_onnx` |
| `jats_v1` | `skol_dev` | `skol_golden_ann_jats` | `skol_exp_jats_v1_treatments` ⚠ | `skol_exp_jats_v1_treatments_full` ⚠ | _(not set)_ | `skol:embedding:jats_v1` |
| `hand_annotated` | `skol_golden` | `skol_training` | `skol_exp_hand_annotated_treatments` ⚠ | `skol_exp_hand_annotated_treatments_full` ⚠ | _(not set)_ | `skol:embedding:hand_annotated` |

⚠ — Names ending in `_treatments` / `_treatments_full` are post-rename
values that the Step-3 migration script wrote into `skol_experiments`.
**INVESTIGATE**: the local CouchDB does **not** yet hold matching databases
for these experiments (no `skol_exp_jats_v1_treatments`, no
`skol_exp_hand_annotated_treatments`, etc.). Those will only exist after
the experiments are re-run; until then those experiment configs are valid
on paper but cannot be executed end-to-end.

## Lineage graph

```
                 ┌──────────────────────────────┐
                 │  skol_dev (primary ingest)    │
                 └──┬─────────────┬─────────────┘
                    │ extract     │ seed
                    │ treatments  │ (one-way)
                    ▼             │
   skol_treatments_dev            │
   skol_treatments_full_dev       │
   skol_treatments_taxpub_v1_dev  │
                                  │
                          ┌───────┴────────┐
                          │ skol_training   │
                          │ skol_training_  │
                          │   taxpub_v1     │
                          └───────┬────────┘
                                  │ review (brat round-trip)
                                  ▼
                          skol_ann_reviewed
                                  │
                                  │ fix_missing_yedda
                                  ▼
                          skol_ann_fixed
                                  │
                                  │ merge_yedda + enrich_ann_merged
                                  ▼
                          skol_ann_merged
                                  │
                                  │ llm_relabel
                                  ▼
                          skol_staging
                          skol_training_llm_stage

   skol_golden  ─┬─  skol_golden_ann_hand
                └─  skol_golden_ann_jats   (built by curate_golden_dataset)
```

## Open investigation items

These are databases or experiment slots where I could not determine the
intent from code, docs, or doc shape alone:

1. **`skol_exp_hand_ann`** — 105 docs, one `article.txt.ann` each. The
   `hand_annotated` experiment in `skol_experiments` has no
   `databases.annotations` field, so the link is inferred by name + size
   match with `skol_golden` (also 105). **TODO: confirm the linkage and
   add `databases.annotations: skol_exp_hand_ann` to the experiment doc.**

2. **`jats_v1` and `hand_annotated` experiment treatments DBs** — The
   experiment docs reference `skol_exp_jats_v1_treatments`,
   `skol_exp_jats_v1_treatments_full`, `skol_exp_hand_annotated_treatments`,
   `skol_exp_hand_annotated_treatments_full` but those databases don't
   exist locally. Are those experiments archived, or pending an
   `extract_treatments_to_couchdb` run that hasn't happened yet?

3. **Production-experiment annotations** — _Resolved 2026-05-19_:
   `production` now carries `databases.annotations: ""` (explicit empty,
   matching the `ANNOTATIONS_DB_NAME` default in `bin/env_config.py`),
   with a `notes` field marking it as a placeholder until a 'best'
   experiment's databases are promoted into the slot. The context viewer's
   fallback to `ingest_db` (`skol_dev`) is the intentional behaviour while
   no specific annotations DB is chosen.
