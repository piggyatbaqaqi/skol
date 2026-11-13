# The experiment record is the canonical source

*Started 2026-11-13, after `production_v4_1` was deployed to
production.  Three steps; each ships independently and each is
test-first (tests approved, committed with strict xfail, then
implemented).*

## Context

Deploying `production_v4_1` (`manage_experiment deploy`) exposed
three places where production does not do what its `production`
experiment record says:

1. **Menus.**  Three things disagreed about `redis_keys.menus`:
   * `manage_experiment create` writes the **tree key**
     (`skol:ui:menus_<name>`), and `build_vocab_tree --experiment`
     writes the tree there;
   * the web views read it as a **pointer** (a key whose value is
     the tree key), defaulting to `skol:ui:menus_latest`;
   * `build_vocab_tree` always repoints the shared
     `skol:ui:menus_latest`, from any build.

   The deploy put a tree key into the record, so `/api/vocab-tree/`
   returned 404 with the tree JSON as the "key".  Repointing the
   record at `menus_latest` did not hold: the 22:00 cron
   (`build_vocab_tree --db=skol_treatments_full_dev`) had already
   moved that pointer to a tree built from the old database.  The
   stopgap in place since 2026-11-13: a dedicated pointer
   `skol:ui:menus_production_v4_1_pointer` →
   `skol:ui:menus_production_v4_1`, named by both the `production`
   and `production_v4_1` records.
2. **Deploy copies only `databases` and `redis_keys`.**  The
   `production` record still says `pipeline: v3_logistic` with v3
   `pipeline_state`, although it now serves v4.1 (`v4_crf`).
3. **Cron ignores the record.**  The nightly jobs in
   `debian/skol.cron` run without `--experiment`, so they read and
   write the pre-experiment databases (`skol_treatments_dev`,
   `skol_treatments_full_dev`, `skol:embedding:v1.1`) that
   production no longer serves.  The v3/v1 jobs
   (`train_classifier`, `predict_classifier`,
   `extract_treatments_to_couchdb`) cannot simply take
   `--experiment production`: `predict_classifier` would write v3
   logistic predictions into the v4 annotations database.

## Principle

The `production` experiment record — its `databases`, `redis_keys`
and `pipeline` — is the only description of what production
serves and how it is maintained.  No shared pointer, crontab line
or script default may override it.

## Step 1: menus come only from the record

`redis_keys.menus` names the Redis key holding the tree.  There is
no `skol:ui:menus_latest`.

* **Views** (`django/search/views.py`): `VocabTreeView`,
  `VocabTreeChildrenView` and `BuildVocabTreeView` (GET and POST)
  read the tree from the user's experiment's `redis_keys.menus`.
  An explicit `?version=` still selects `skol:ui:menus_<version>`.
  A missing `menus` key or a missing tree is a 404 naming the
  experiment and key — no fallback to another tree.
  `VocabTreeChildrenView` no longer ignores the experiment.
* **API-triggered build**: `POST /api/vocab-tree/build/` runs
  `build_vocab_tree --experiment <name>`, not `--db X`, so it
  rebuilds the experiment's tree in place.
* **`build_vocab_tree`**: `save_to_redis` never writes a pointer.
  With `--experiment` it writes the record's `menus` key; without,
  it writes `skol:ui:menus_<version>` and changes nothing else.
  `load_from_redis` with no version reads `skol:ui:menus_latest`
  today; it will require a version or key instead.
* **`VocabTreeVersionsView`** keeps listing tree keys; its
  `latest` field goes.
* **`bin/rebuild_redis.py`**: drop `skol:ui:menus_latest` from the
  `vocab_tree` key list.
* **One-time fix** (`fixes/`): for every `skol_experiments` record
  whose `menus` value is a pointer (the stored value is a key name,
  not tree JSON), rewrite `menus` to the pointer's target.  Then
  delete the pointer keys (`skol:ui:menus_latest`,
  `skol:ui:menus_production_v4_1_pointer`) that no record still
  names.  `--dry-run` previews (the shared flag, as in the re-key
  fix); production writes are confirmed by the operator.
* **Docs**: `docs/api-reference.md` (calls it a "menus pointer
  key"), `docs/api-vocab-tree.md`,
  `docs/GOLDEN_DATASET_AND_EXPERIMENTS.md`.

Order on production: install the deb with the new views, then run
the fix.  The stopgap pointer keeps the old views working until
then.

## Step 2: deploy copies the pipeline

`manage_experiment deploy` also copies `pipeline` and
`pipeline_state` into the `production` record, so
`manage_experiment runstep production <step>` runs the deployed
pipeline's steps.  After the release, re-run
`manage_experiment deploy production_v4_1` on production
(operator-confirmed): it is idempotent for `databases` and
`redis_keys`, which are the values already there.

Re-running deploy also rewrites `redis_keys.menus` back to the
tree key, so it must follow Step 1's views on production — or the
menus outage recurs.  It also drops `golden`/`golden_ann` again
(they are not in the `production_v4_1` record); either add them to
`production_v4_1` first or restore them after, as was done on
2026-11-13.

## Step 3: cron runs the record's pipeline

Every experiment-dependent job becomes
`manage_experiment runstep production <step>`.  The step's script
and flags then come from the record's pipeline (`v4_crf` today),
not from the crontab, and the next deploy changes cron behaviour
with no crontab edit.  `runstep` reruns a step each time; the
`production_v3_hand` jobs already work this way.

| Today (`debian/skol.cron`) | Becomes |
|---|---|
| 00:00 `predict_classifier …` | `runstep production predict` |
| 00:30 `train_classifier` | removed — v4 training is deliberate, not nightly |
| 04:00 `extract_treatments_to_couchdb` | `runstep production extract_treatments` |
| (none) | 05:00 `runstep production assess_treatments` |
| 06:00 `embed_treatments --expire None --force` | `runstep production embed` |
| 07:00 `build_sources_stats` | `runstep production build_sources_stats` |
| 22:00 `treatments_to_json …` | `runstep production treatments_to_json` |
| 22:00 (Sun/Tue/Thu/Sat) `build_vocab_tree --db=skol_treatments_full_dev` | `runstep production build_vocab` |

Every `extract_treatments` job is followed later the same day by
`assess_treatments` for the same experiment, when its pipeline has
that step: the verdicts in `databases.treatment_quality` then cover
each night's new treatments.  It is incremental (it skips verdicts
whose `checks_version` matches the current detectors), so a normal
night assesses only the new treatments; the first run after a deb
that changes a detector reassesses all of them.
`production_v3_hand` is exempt: `v3_logistic` has no
`assess_treatments` step and its record has no `treatment_quality`
database.

Unchanged: `ingest`, `extract_plaintext`, `jats_to_yedda` and
`backfill_plazi_uuids` (they read and write the shared `skol_dev`
ingest database; `backfill_plazi_uuids` stamps Plazi document UUIDs
onto ingested articles, not onto treatments), `daily_report`,
`prod_smoke_check`, `manage_fungaria`, the Redis health check, and
the `production_v3_hand` / `verify_spans` lines, which already name
their experiment.

A test (`tests/skol_cron_test.py`) parses `debian/skol.cron` and
asserts the table above, the assess-after-extract rule, and that no
line runs one of the experiment-dependent scripts directly.

Ships in the same deb as the dr-drafts-mycosearch embed fix
(`15454bf`) and the strict `build_vocab_tree` CLI (`a4261dd`), and
needs Step 2 on production first: without it, `runstep production`
runs the v3 pipeline.

### Open questions

* **Shared annotations database.**  `production_v4_1` uses
  `skol_exp_production_v4_01_00_ann_combined`, shared with
  `production_v4`.  Nightly `predict` adds predictions for new
  documents there, which both experiments then see.  Same
  classifier, so consistent, but worth knowing.

## Rollout on production

The three steps ship in one deb, and the deb installs the new
crontab, so do these in one sitting, each production write confirmed
by the operator:

1. Install the skol deb (with the dr-drafts-mycosearch wheel that
   carries the embed fix).  From here until step 2, `/vocab-tree/`
   fails: the new views read `redis_keys.menus` as the tree key, and
   the records still name the stopgap pointer.
2. `fixes/fix_menus_pointer_records.py --dry-run`, then without
   `--dry-run`.  Expected on production: `production` and
   `production_v4_1` go from `skol:ui:menus_production_v4_1_pointer`
   to `skol:ui:menus_production_v4_1`; the stopgap pointer and
   `skol:ui:menus_latest` are deleted.  Check `/vocab-tree/` serves
   the v4.1 tree.
3. `manage_experiment deploy production_v4_1`, so the `production`
   record carries `pipeline: v4_crf`.  It must happen before 00:00,
   when the first `runstep production predict` runs.  Then restore
   `golden`/`golden_ann` in `production` (or add them to
   `production_v4_1` before deploying).
4. The next morning: check the `production-*.log` files from the
   00:00–07:00 jobs.

## Out of scope

* `redis_keys.classifier_model` in `production_v4_1` names
  `skol:classifier:model:production_v4_1`, which does not exist.
  Only the v3 `predict_classifier`/`train_classifier` read it, and
  Step 3 removes them from cron.

## Change log

* **2026-11-13** — Implemented: tests `a1bade7`; Step 1 `055c64c`;
  Step 2 `00e3be1`; Step 3 `45c37a2`.  Not yet rolled out (see
  "Rollout on production").

* **2026-11-13** — Decided: `assess_treatments` runs after every
  `extract_treatments` job (05:00 for `production`);
  `backfill_plazi_uuids` stays with the ingest jobs.

* **2026-11-13** — Plan written after the `production_v4_1`
  deploy and the menus outage it caused.
