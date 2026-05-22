# v3_hand full-pipeline buildout

Working plan to take `production_v3_hand` from a tested model in
Redis to a deployed pipeline that produces Treatments and SBERT
embeddings for every doc in `skol_dev`. Three sequential phases,
each shippable and reversible. Production deployment is a separate
later step.

## Background

After [docs/production_v3_report.md](production_v3_report.md),
`production_v3_hand` is the only v3 model worth deploying — macro F1
0.459 on the v2 hand gold, four-of-nine classes scoring above 0.5.
What's missing for a useful deployment:

1. **YEDDA coverage of skol_dev.** Today only a tiny subset of
   `skol_dev` docs have predicted `.ann` attachments. We need full
   coverage so Treatments can be assembled across the whole corpus.
2. **Treatment extraction across skol_dev.** Existing
   `skol_treatments_dev` (25 420 records) is v1-era; needs
   regenerating with the v3 labels.
3. **SBERT embeddings on the new Treatments.** Production search
   reads `skol:taxa:embedding:{doc_id}` keys. Treatments without
   embeddings are invisible to similarity search.

The architectural decision in
[docs/extraction_pipeline.md](extraction_pipeline.md) — JATS docs
go through deterministic XML extraction, not the classifier —
makes the YEDDA-coverage step gate on the extraction-pipeline
dispatcher being in place. So Phase A builds the dispatcher; Phase
B runs the sweep; Phase C embeds. Total wall-clock estimate:
~3-6 hours including TDD iterations.

## Goals

1. Land Commit 1 of the
   [extraction_pipeline.md](extraction_pipeline.md) migration
   sequence: dispatcher scaffold + `taxpub_treatment_extractor` +
   `classifier_logistic_v3` (wrapping the existing predict flow).
2. Use that dispatcher to generate complete Treatment coverage of
   `skol_dev`.
3. Embed every Treatment with SBERT for production search.

## Non-goals

- **Not yet shipping Commits 2-5** of the extraction-pipeline
  migration (keyword labeler, v4 CRFs, OCR sub-pipeline, markdown
  labeling). Those are subsequent commits with their own scope.
- **Not yet deploying to production.** Production deployment is a
  separate step after dev verification + operator sign-off.
- **Not regenerating training corpora.** v3_hand stays as-shipped.

## Phase A — Extraction-pipeline dispatcher (Commit 1)

The minimum-viable dispatcher that subsumes today's monolithic
`predict → YEDDA → extract_treatments` flow without changing
external behavior. After Phase A, taxpub JATS docs go through the
XML reader; everything else goes through `production_v3_hand`.

### Package layout

```
skol_classifier/extraction/
  __init__.py                          — exports + bootstrap
  catalog/                             — ngautonml catalog (Apache-2.0)
    __init__.py
    catalog.py
    memory_catalog.py
    catalog_element_mixin.py
  interfaces.py                        — Inspector / Component /
                                         ComponentInstance ABCs
  state.py                             — PipelineState
  dispatcher.py                        — Dispatcher class
  inspectors/
    __init__.py
    attachments.py                     — has_xml, has_pdf, has_plaintext
    xml_root.py                        — xml_format
    taxpub_markup.py                   — has_taxpub_markup
    plaintext_signal.py                — has_taxonomic_signal
  components/
    __init__.py
    taxpub_treatment_extractor.py
    classifier_logistic_v3.py
    treatment_assembler.py
```

### Sub-steps

| # | Description | Status |
|---|---|---|
| A.1 | Copy ngautonml's `catalog.py`, `memory_catalog.py`, `catalog_element_mixin.py` into `skol_classifier/extraction/catalog/`. Apache-2.0 copyright header preserved. Drop `..wrangler.constants.CATALOG_IGNORE` and `..wrangler.logger` imports — substitute small skol-local equivalents. Tests: 6 existing-style tests for `MemoryCatalog` (register, name lookup, tag lookup, autoload, etc.). | ⬜ |
| A.2 | `interfaces.py` — `Inspector(CatalogElementMixin, ABC)` + `Component(CatalogElementMixin, ABC)` + `ComponentInstance(ABC)` + subtype tags (`TextProducer` / `EntityDetector` / `SectionLabeler` / `Assembler`). TDD: 4 tests asserting that concrete subclasses can be registered + their declared tags surface correctly. | ⬜ |
| A.3 | `state.py` — `PipelineState` with `doc`, `props`, `_label_contributions`, `_span_contributions`, `_attachment_cache`, plus helpers (`add_section_labels`, `add_spans`, `get_attachment`). TDD: 5 tests covering deterministic-first label merge + specific-over-general span merge + attachment caching. | ⬜ |
| A.4 | Inspectors (4 files in `inspectors/`). TDD: 1 happy-path + 1 missing-precondition test per inspector = 8 tests. | ⬜ |
| A.5 | `components/taxpub_treatment_extractor.py` — wraps `ingestors.jats_to_yedda.jats_xml_to_tagged_blocks`. TDD: feeds a known taxpub XML fixture, asserts the same TaggedBlock list as calling jats_to_yedda directly. | ⬜ |
| A.6 | `components/classifier_logistic_v3.py` — wraps the existing `predict_classifier.py` + `.ann` read flow. TDD: feeds a doc whose `.ann` is pre-seeded; asserts the TaggedBlock list matches the existing parser output. | ⬜ |
| A.7 | `components/treatment_assembler.py` — wraps `taxon.py::group_paragraphs()`. TDD: known TaggedBlock list → expected Treatment list. | ⬜ |
| A.8 | `dispatcher.py` — runs inspectors → selects components → topo-sort → executes → merges → calls assembler. TDD: 3 end-to-end tests covering a taxpub-JATS doc, a plaintext-only doc, and an unrecognised doc (empty result). | ⬜ |
| A.9 | `bin/extract_treatments_to_couchdb.py` — refactor to call the dispatcher. The existing entry-point CLI flags (`--experiment`, `--doc-id`, `--skip-existing`, `--force`, `--dry-run`) are preserved. TDD: re-run the existing `extract_treatments_to_couchdb_test.py` cases through the dispatcher; assert identical Treatment outputs. | ⬜ |
| A.10 | Verification on `skol_golden_v2` (105 docs). Compare the dispatcher's Treatment output against the existing pipeline's output. Field-equality on at least: nomenclature text, each section text, span counts per section. | ⬜ |

### Worked Commit-1 scope

Concrete file count: **~17 new files + 1 modified**. Estimated
~1 500 lines of code (≈ 60% tests, 40% implementation). The
catalog copy is ~400 lines; everything else is new and small.

### Acceptance criteria

After Phase A:

- All existing `extract_treatments_to_couchdb_test.py` tests pass
  against the dispatcher.
- Running `bin/extract_treatments_to_couchdb.py` on a JATS doc
  produces the same Treatment record as today.
- Running it on a plaintext-only doc with a pre-seeded `.ann`
  produces the same Treatment record as today.
- A.10's golden-DB verification has 0 mismatches (field-equality on
  ~105 docs).

## Phase B — Full skol_dev YEDDA + Treatment buildout

With the dispatcher in place, sweep `skol_dev` to produce
Treatments for every taxonomically-relevant doc.

### Sub-steps

| # | Description | Status |
|---|---|---|
| B.1 | Create the **v3_hand-specific output DBs** so every pipeline step writes to its own namespace (none share with the unrelated dev defaults): `skol_treatments_v3_dev` (treatments — replaces today's `skol_treatments_dev` in the experiment doc), `skol_treatments_full_v3_dev` (full-context treatments — replaces today's `skol_treatments_full_dev`). The `databases.annotations` field on `production_v3_hand` is already v3-specific (`skol_exp_production_v3_hand_ann`); no change there. Update the `production_v3_hand` experiment doc with the new `treatments` + `treatments_full` values **before** running B.3, since `extract_treatments_to_couchdb.py` reads them from there. Both DBs are created lazily by CouchDB on first write; nothing to pre-allocate. | ⬜ |
| B.2 | Predict YEDDA for all skol_dev docs lacking `.ann`. The dispatcher decides per-doc whether predict_classifier runs (PDF/plaintext) or whether the XML reader runs (taxpub). For docs that already have `article.txt.ann` from earlier experiment runs, skip via `--skip-existing` (existing flag). | ⬜ |
| B.3 | Run `bin/extract_treatments_to_couchdb.py --database skol_dev --output-database skol_treatments_v3_dev` to extract Treatments through the dispatcher. Expected ~10-50 k Treatments depending on how many of the ~28 k PDF docs and ~2 500 taxpub docs are taxonomic. | ⬜ |
| B.4 | Spot-check: for 10 random Treatments, inspect that all flat section fields populated correctly. For 5 JATS-source Treatments, verify the XML-reader path produced the same section text as the previous `jats_to_yedda` output. | ⬜ |
| B.5 | Update `docs/couchdbs.md` with the new `skol_treatments_v3_dev` + `skol_treatments_full_v3_dev` rows, and update the experiments table to reflect `production_v3_hand`'s new `treatments` / `treatments_full` pointers. | ⬜ |

## Phase C — SBERT embedding

Embed Treatments and write to Redis so the Django search reads them.

### Sub-steps

| # | Description | Status |
|---|---|---|
| C.1 | Pick the embedding text. Per [treatment_architecture.md §Phase 5](treatment_architecture.md): `description + "\n\n" + diagnosis` (with diagnosis falling back to empty if absent). | ⬜ |
| C.2 | Choose the Redis key namespace. The `production_v3_hand` experiment doc declares `redis_keys.embedding`; whatever it says becomes the prefix. Inspect + confirm before running. | ⬜ |
| C.3 | Run `bin/embed_treatments.py --experiment production_v3_hand` (existing script). It already iterates the target DB, computes SBERT embeddings, writes to Redis. Should pick up the new `skol_treatments_v3_dev` automatically once B.1's choice is wired into the experiment doc. | ⬜ |
| C.4 | Smoke-test the Django search UI: query a known taxonomic phrase (e.g. "Pardosa moesta sp. nov."), confirm reasonable matches surface. | ⬜ |

## Phase E — Scheduled cron jobs for ongoing operation

`debian/skol.cron` already schedules the v1 production pipeline
(train_classifier, predict_classifier, extract_taxa_to_couchdb,
embed_taxa).  Add parallel v3_hand-experiment jobs so the pipeline
keeps producing fresh treatments + embeddings as new docs land in
`skol_dev` via the daily ingestion jobs.

| # | Description | Status |
|---|---|---|
| E.1 | Pick cadence + clock slots that don't collide with the v1 jobs.  v1 train at 00:30, predict at 00:00, extract at 04:00, embed at 06:00 — schedule v3_hand later in the day, e.g. predict 12:00, extract 14:00, embed 16:00, train weekly Sunday 03:30.  Training stays infrequent because the v3_hand training corpus is fixed; predict/extract/embed are daily to track new ingest. | ⬜ |
| E.2 | Add jobs to `debian/skol.cron`: `manage_experiment.py runstep production_v3_hand train` (weekly), `predict` (daily), `extract_taxa` (daily, after predict), `embed` (daily, after extract).  Each line uses the canonical `runstep` invocation so the experiment doc's databases / Redis keys / model name flow through. | ⬜ |
| E.3 | Verify on dev: bump the dev cron entries (or run the commands by hand at the scheduled times) and confirm new treatments + embeddings appear in `skol_treatments_v3_dev` and the production_v3_hand Redis namespace within a few hours. | ⬜ |
| E.4 | Document the new cron entries in `docs/experiments.md` so future operators know which jobs maintain each experiment. | ⬜ |

Lives in v3_buildout because it's how the work in Phase B (one-time
sweep) becomes sustainable.  Without E the buildout decays — new
docs ingest into `skol_dev` but no Treatments / embeddings get
produced.

## Phase F — `skol-gnservices` Debian package

Ships gnfinder + gnparser as a sibling Debian package built from the
same repo as `skol.deb`.  Required for an automated v3_hand (and
eventual v4) production install — currently the binaries live in
`~/bin/` on dev and would have to be installed by hand on prod.
Closes that gap before any production rollout.

Same-repo, separate-packaging-dir layout:

```
skol/
  packaging/skol-gnservices/
    debian/
      control                       ← hand-written, not stdeb-generated
      rules
      copyright
      skol-gnservices.install
      gnfinder.service              ← systemd unit for gnfinder -p 9080
      gnparser.service              ← systemd unit for gnparser -p 9081
      postinst                      ← create skol-gn user, enable units
      prerm
    VERSION                          ← tracks upstream gnfinder/gnparser tags
    fetch_binaries.sh                ← curls release tarballs from
                                       github.com/gnames during build
  Makefile                           ← gains `make deb-gnservices` target
```

Two binary packages from one source repo because the gnservices
release cadence (follows upstream gnfinder/gnparser) is independent
from skol's release cadence — different changelogs + different
versions, but one place to PR cross-cutting config changes (e.g.
shifting ports between env_config and the systemd unit at once).

| # | Description | Status |
|---|---|---|
| F.1 | Create `packaging/skol-gnservices/` skeleton: `debian/control` declaring `Package: skol-gnservices`, `Architecture: amd64`, `Depends: adduser, systemd`; `debian/rules` using `dh $@`; `debian/copyright` listing Apache-2.0 for both upstream binaries; a `VERSION` file pinning the upstream tags. | ⬜ |
| F.2 | Write `fetch_binaries.sh` that pulls the gnfinder + gnparser release tarballs from github.com/gnames during the package build and unpacks the binaries to a staging area `debian/skol-gnservices/opt/skol-gnservices/bin/`. | ⬜ |
| F.3 | Add `gnfinder.service` and `gnparser.service` systemd units (each starting the binary with the agreed-upon ports + run-as user).  Place them in `debian/skol-gnservices/lib/systemd/system/`. | ⬜ |
| F.4 | Add `debian/postinst` to create a system user `skol-gn`, enable both units, start them.  Mirror existing skol postinst patterns. | ⬜ |
| F.5 | Add `Makefile` target `deb-gnservices` that invokes `dpkg-buildpackage -us -uc -b` from `packaging/skol-gnservices/`. | ⬜ |
| F.6 | Update `stdeb.cfg`'s `Depends:` (or `Recommends:`) line on the main `skol` package to include `skol-gnservices`. | ⬜ |
| F.7 | Build both `.deb`s locally on dev; install via `dpkg -i`; verify `gnfinder` and `gnparser` services come up on the expected ports and that the existing `ingestors/gnfinder_client.py` + `gnparser_client.py` clients can talk to them. | ⬜ |
| F.8 | Update `docs/experiments.md` and `docs/v4_classifier_plan.md` to document the install + the deb layout. | ⬜ |

Phases E and F together gate Phase D (production deployment) —
without them, prod gets neither scheduled extraction jobs nor the
gn-services the future v4 pipeline depends on.

## Phase D — Production deployment (separately scheduled)

Out of scope for this buildout; treated as a follow-up that
operator drives. Sketch of what's needed when ready:

- Apply `0020`-`0023` Django migrations on prod's database.
- Replicate `skol_training_v2`, `skol_training_taxpub_v2`,
  `skol_training_v2_no_golden`, `skol_training_v3_combined_no_golden`,
  `skol_experiments`, all five v2 golden DBs from dev to prod.
- Train `production_v3_hand` on prod (model lives in Redis;
  re-run `bin/train_classifier.py` after the training corpus
  replicates).
- Run Phase B + C on prod's `skol_dev`.
- Smoke-test prod search.
- Per [memory: prod_spark_sizing.md] — set `SPARK_CORES`,
  `SPARK_DRIVER_MEMORY`, `SPARK_EXECUTOR_MEMORY`, `NUM_WORKERS`
  in `/home/skol/.skol_env` before any Spark step. Prod is
  smaller than dev.

## Compute budget

Estimated wall-clock on the dev 24-core / 96 GB / 24 GB-GPU host:

| Phase | Wall-clock |
|---|---|
| Phase A (build + tests) | ~3-4 hours of focused implementation + iteration; tests run in seconds. |
| Phase B.2 (predict YEDDA on ~28 k PDF docs at 16-core Spark) | ~30-50 minutes |
| Phase B.3 (extract treatments — fast, ~1 ms each) | ~2-5 minutes |
| Phase C.3 (SBERT embedding ~10-50 k Treatments at GPU batch 512) | ~10-30 minutes |
| **Total** | **~4-5.5 hours** including Phase A's TDD iteration |

User budget per the previous "compute budget acceptable" sign-off.

## Decisions locked in

- **B.1 target DBs:** new `skol_treatments_v3_dev` +
  `skol_treatments_full_v3_dev`. v1-era dev DBs are kept untouched
  for rollback.
- **C.4 smoke-test query:** Django search for a known taxonomic
  phrase (e.g. "Pardosa moesta sp. nov.") returning reasonable
  matches counts as Phase-C acceptance.

## Open questions

- **B.2 already-classified docs.** Some `skol_dev` docs already
  have `.ann` from previous experiment runs (e.g. `taxpub_v1_*`).
  Do we re-predict to ensure consistency, or skip-existing? Lean
  re-predict — older `.ann`s use older models. The dispatcher
  decides which model to use, not the existing file. (Confirm
  before B.2.)
- **C.3 embedding model variant.** v1 uses `all-mpnet-base-v2`
  per `embedding_name` in env_config. Stay with that, or upgrade?
  Lean: stay. Upgrade is a separate experiment.

## Rollback

- **Phase A:** revert the `skol_classifier/extraction/` package and
  the `extract_treatments_to_couchdb.py` refactor. The old
  monolithic flow returns. Existing Treatments unaffected.
- **Phase B:** drop `skol_treatments_v3_dev` (if option (b)
  chosen). Existing `skol_treatments_dev` untouched. Or restore
  pre-overwrite snapshot if option (a) was chosen.
- **Phase C:** delete the Redis embedding keys for the v3
  namespace. Search degrades to "no results"; no data loss.

## Progress

Sub-step status table will be updated in-place as commits land.
