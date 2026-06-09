# SKOL database naming cleanup — deferred plan

Status: **deferred** until the per-family pipeline restructure
(commit 5dde6d4) settles into production_v4 use.  The restructure
pass this builds on lives in `~/.claude/plans/cozy-forging-locket.md`
(working-copy plan, kept outside the repo since it tracks
in-flight implementation rather than a durable design).

## Context

The CouchDB database namespace grew organically across v2 / v3 /
v4 iterations.  Names use multiple conventions and don't sort in
pipeline order, which makes operator triage harder than it
should be:

| DB | Convention | Pipeline role |
|---|---|---|
| `skol_dev` | global, role-named | ingest |
| `skol_training` | global, role-named | training |
| `skol_training_v3_combined_no_golden` | versioned, ad-hoc | training |
| `skol_golden`, `skol_golden_v2` | global, versioned | eval (plaintext) |
| `skol_golden_ann_hand`, `…_v2` | global, role-named, versioned | eval (answer-key) |
| `skol_exp_<X>_ann` | per-experiment | annotations |
| `skol_exp_<X>_taxa` | per-experiment | extracted treatments |
| `skol_exp_<X>_taxa_full` | per-experiment | LLM-annotated treatments |
| `skol_experiments` | registry | — |
| `skol_treatments_v3_dev` | legacy ad-hoc | treatments (pre-experiment-doc) |

This pass renames them under a single convention with two
goals:

1. **Predictable prefix** so `_all_dbs` listings group logically.
2. **Sort order matches pipeline order** so an operator scanning
   the list sees `01_ingest` before `02_training` before
   `03_annotations`, etc.

## Eval / production split — new convention from 2026-06-09

Carried over from the pipeline-restructure discussion.  Decision
made: **eval pipelines must NOT write to production DBs**.
Eval-vs-production parity (same models, same pipeline steps)
applies to *code*, not *data*.  When a pipeline step processes a
truncated dataset (golden set, ablation hold-out, etc.) it
writes to an `_eval`-suffixed DB.

The convention (using the new role names settled later in
the plan):

| Role | Production DB | Eval DB |
|---|---|---|
| Annotations | `skol_exp_<X>_01_00_ann` | `skol_exp_<X>_01_00_ann_eval` |
| Prose treatments | `skol_exp_<X>_02_00_treatments_prose` | `skol_exp_<X>_02_00_treatments_prose_eval` |
| Structured treatments | `skol_exp_<X>_03_00_treatments_structured` | `skol_exp_<X>_03_00_treatments_structured_eval` |

Eval DBs sort alphabetically immediately after their
production counterparts (the trailing `_eval` puts them right
next to the production DB they're paired with).  Confirmed
2026-06-09 as the desired listing order.

The variable substitution layer
([bin/pipelines/base.py:build_variables](../bin/pipelines/base.py))
already exposes `{eval_ann_db}`.  Today it defaults to
`{annotations_db}` — change that default to
`{annotations_db}_eval` so the split is the framework's default
behaviour rather than something each pipeline step has to
remember.  Operators wanting the legacy shared-DB behaviour can
explicitly set `eval_annotations_db_name` on the experiment doc.

Rationale (from 2026-06-09 conversation):

- **Eval idempotency.**  Re-running eval should produce eval, not
  silently mutate production.
- **Production safety.**  A bad eval run can't poison the docs the
  search UI reads.
- **Cleanup discipline.**  Archiving an experiment is one
  `archive_experiment` call that drops both `_ann` and
  `_ann_eval`; today it'd have to manually distinguish "is this
  doc a production prediction or an eval prediction?"

Trade-offs accepted (full discussion in conversation log):

- Database proliferation roughly doubles per experiment.
- Cross-DB comparison (production vs eval on the same doc)
  becomes a two-query operation instead of one.
- Storage cost grows ~linearly with experiment count × eval
  corpus size.  For the 105-doc golden set: ~500 MB/experiment;
  negligible at current scale.

## Naming convention — settled 2026-06-09

Two distinct shapes, one for per-experiment data, one for
shared / global data.  Sort order is "experiment first, stage
second" — operators care that all DBs for ONE experiment
cluster together; the relative order of `production_v3_hand`
vs `production_v4` in the listing doesn't matter.

### Per-experiment DBs

```
skol_exp_<EXPERIMENT>_<STAGE>_<ROLE>[_eval]
```

- `<EXPERIMENT>` — the experiment doc's `_id` (e.g.
  `production_v4`, `production_v3_hand`).
- `<STAGE>` — decimal-numeric ordering tag matching the
  pipeline-step write order: `01_00` annotations, `02_00`
  prose treatments, `03_00` structured treatments.
  Underscore separator (CouchDB DB names disallow `.`).
  Insertion-friendly: a future step whose output sorts
  between annotations and prose treatments gets `01_50`
  without renumbering existing DBs.
- `<ROLE>` — the data shape: `ann` (YEDDA-tagged annotations
  from predict), `treatments_prose` (extracted treatment text
  blocks — Nomenclature, Description, Etymology …), or
  `treatments_structured` (LLM-annotated JSON with named
  fields per treatment).  Today's `taxa` / `taxa_full` are
  retired — see Role-naming cleanup section below.
- `_eval` (optional suffix) — present iff the DB carries
  predictions over a truncated dataset (golden set / ablation
  hold-out), distinct from production data.  See the
  Eval / production split section above.

Example listing for `production_v4`:

```
skol_exp_production_v4_01_00_ann
skol_exp_production_v4_01_00_ann_eval
skol_exp_production_v4_02_00_treatments_prose
skol_exp_production_v4_02_00_treatments_prose_eval
skol_exp_production_v4_03_00_treatments_structured
skol_exp_production_v4_03_00_treatments_structured_eval
```

All six group together in `_all_dbs`; within the group, stage
order is preserved.

### Shared / global DBs

Shared DBs follow role-named conventions with no stage prefix —
they're outside any per-experiment ordering by design:

```
skol_dev                          # ingest corpus, shared
skol_training                     # default training corpus
skol_training_v3_combined_no_golden
skol_training_v3_hand
skol_training_v3_jats
skol_training_v2_no_golden
skol_golden_v1                    # eval plaintext, v1
skol_golden_v2                    # eval plaintext, v2
skol_golden_ann_hand_v1           # eval answer-key, hand-annotated, v1
skol_golden_ann_hand_v2
skol_golden_ann_jats
skol_golden_ann_bioc
skol_experiments                  # the experiment-doc registry
```

These sort outside the `skol_exp_*` grouping naturally
(`skol_d…` and `skol_g…` and `skol_t…` < `skol_exp_…` < …) so
no marker is needed to distinguish them visually.

### Combined listing

```
skol_dev
skol_exp_production_v3_hand_01_00_ann
skol_exp_production_v3_hand_02_00_treatments_prose
skol_exp_production_v3_hand_03_00_treatments_structured
skol_exp_production_v3_jats_01_00_ann
skol_exp_production_v3_jats_02_00_treatments_prose
skol_exp_production_v4_01_00_ann
skol_exp_production_v4_01_00_ann_eval
skol_exp_production_v4_02_00_treatments_prose
skol_exp_production_v4_02_00_treatments_prose_eval
skol_exp_production_v4_03_00_treatments_structured
skol_exp_production_v4_03_00_treatments_structured_eval
skol_experiments
skol_golden_ann_hand_v2
skol_golden_v2
skol_training_v3_combined_no_golden
```

All per-experiment data clusters; eval DBs sort directly next
to their production counterparts within each group; shared data
lives at the top and bottom of the list around the
`skol_exp_*` block.

### Role-naming cleanup — settled 2026-06-09

`taxa` / `taxa_full` are historical: when the pipeline was
primarily taxonomic-name parsing, the role name reflected that.
The actual data shape is *treatments* (Nomenclature +
Description + Etymology + … blocks for one organism each), and
the codebase has been steadily renaming in that direction:
`bin/extract_treatments_to_couchdb.py`, `bin/treatments_to_json.py`,
`bin/embed_treatments.py` all already use the term.  This pass
finishes the rename.

Two tiers:

- `treatments_prose` — text blocks of natural language
  (Nomenclature, Description, Etymology, …) extracted from
  predict's YEDDA output by `extract_treatments_to_couchdb`.
- `treatments_structured` — LLM-annotated JSON with named
  fields per treatment, output of `treatments_to_json`.

Symmetric `_prose` + `_structured` naming makes the
relationship between tiers obvious: same `treatments` root,
different transformation.  The `_full` suffix (which suggested
"more complete" but actually meant "more processed") is gone.

## Scope — what gets renamed

| Old name | New name |
|---|---|
| `skol_dev` | unchanged (shared) |
| `skol_training` | unchanged (shared) |
| `skol_training_v3_combined_no_golden` | unchanged (shared) |
| `skol_training_v3_hand` | unchanged (shared) |
| `skol_training_v3_jats` | unchanged (shared) |
| `skol_training_v2_no_golden` | unchanged (shared) |
| `skol_golden` | `skol_golden_v1` (version disambiguation) |
| `skol_golden_v2` | unchanged |
| `skol_golden_ann_hand` | `skol_golden_ann_hand_v1` (version disambiguation) |
| `skol_golden_ann_hand_v2` | unchanged |
| `skol_golden_ann_jats` | unchanged |
| `skol_golden_ann_bioc` | unchanged |
| `skol_exp_<X>_ann` | `skol_exp_<X>_01_00_ann` |
| `skol_exp_<X>_ann_combined` | `skol_exp_<X>_01_00_ann_combined` (or fold `combined` into the experiment name; see open question 5) |
| `skol_exp_<X>_taxa` | `skol_exp_<X>_02_00_treatments_prose` |
| `skol_exp_<X>_taxa_full` | `skol_exp_<X>_03_00_treatments_structured` |
| (new) `skol_exp_<X>_01_00_ann_eval` | eval predictions land here, separate from production |
| (new) `skol_exp_<X>_02_00_treatments_prose_eval` | as above |
| (new) `skol_exp_<X>_03_00_treatments_structured_eval` | as above |
| `skol_experiments` | unchanged |
| `skol_treatments_v3_dev` | drop after migration (legacy) |
| `skol_treatments_v3_jats` etc. | drop after migration (legacy) |

Most of the rename pass is **adding** per-experiment stage tags
to the experiment-specific DBs.  The shared / global DBs mostly
stay the same.  This makes the migration cheaper than the
original plan suggested.

### Surfaces beyond DB names

Renaming `taxa` / `taxa_full` → `treatments_prose` /
`treatments_structured` propagates through several
non-CouchDB surfaces.  All must be migrated in the same pass
so the experiment doc, the pipeline modules, and the running
code stay consistent.

1. **Experiment-doc fields** —
   [bin/manage_experiment.py:_default_experiment](../bin/manage_experiment.py)
   currently builds `databases.taxa` / `databases.taxa_full`.
   These rename to `databases.treatments_prose` /
   `databases.treatments_structured`.  Plus the migration
   touches every existing doc in `skol_experiments`.
2. **env_config mapping** —
   [bin/env_config.py:_apply_experiment](../bin/env_config.py)
   has mapping rows for `databases.taxa` →
   `treatments_database` / `taxa_full` → `treatments_full_database`.
   These flip to `treatments_prose_database` /
   `treatments_structured_database` (and the legacy keys retire).
3. **Per-family pipeline modules** —
   [bin/pipelines/v3_logistic.py](../bin/pipelines/v3_logistic.py)
   and [bin/pipelines/v4_crf.py](../bin/pipelines/v4_crf.py)
   both have an `extract_taxa` step running
   `bin/extract_treatments_to_couchdb.py`.  Rename the step to
   `extract_treatments` to match the script name + new DB role.
   (The lazy-repair in `_ensure_pipeline` will leave the old
   `extract_taxa` step entry on existing experiment docs as a
   fossil; the migration script needs to delete it.)
4. **Pipeline variables** —
   [bin/pipelines/base.py:build_variables](../bin/pipelines/base.py)
   exposes `{annotations_db}`.  Add `{treatments_prose_db}` /
   `{treatments_structured_db}` for steps that read either
   tier explicitly (today's `embed`, `treatments_to_json`,
   `annotate_spans`, `build_vocab`, `build_sources_stats` all
   reach the tiers indirectly via `--experiment {name}` → env_config
   resolution, so most pipeline modules don't need updates;
   only the eval-variant steps need explicit variable substitution).
5. **Cron** — [debian/skol.cron](../debian/skol.cron) references
   the `extract_taxa` step name at line 84.  Update to
   `extract_treatments`.  Any other step-name references stay
   the same.
6. **Django code** — search UI, REST endpoints, and templates
   reference `databases.taxa_full` for treatment display
   (haven't inventoried specific files yet; need a grep pass
   over `django/search/` and `django/skolweb/` before execution).
7. **Per-experiment script flags** —
   `bin/treatments_to_json.py`, `bin/embed_treatments.py`,
   `bin/build_vocab_tree.py` etc. read the relevant DB names
   from env_config so they pick up the rename automatically.
   But any ad-hoc shell scripts or CLI invocations hard-coding
   `--source-db skol_exp_X_taxa_full` need updating.
8. **docs/couchdbs.md, docs/api-reference.md** — both reference
   the old role names.
9. **rebuild_redis** — if any Redis keys derive from the role
   names (e.g. embedding-cache keys), the rename touches there
   too.  Audit [bin/rebuild_redis.py](../bin/rebuild_redis.py)
   during execution.

This is the work the user signed up for on 2026-06-09 ("the
deeper scope is expected and a price I want to pay soon").

Plus: every experiment doc's `databases.annotations`,
`databases.taxa`, `databases.taxa_full` fields get rewritten to
the new names.  Add the eval counterparts at the same time.

## Migration approach

CouchDB doesn't support in-place database rename.  Two paths:

### Path A: replicate + delete

```bash
for old in $(curl ...skol_experiments); do
    new=$(translate "$old")
    curl -X POST $URL/_replicate -d '{"source": "old", "target": "new", "create_target": true}'
    # Verify doc counts match, then:
    curl -X DELETE $URL/$old
done
```

Pros: well-tested CouchDB primitive; the replicator handles
attachments correctly (modulo the `_bulk_get` workaround already
landed in [bin/replicate_dbs.py](../bin/replicate_dbs.py)).
Cons: temporary 2× disk footprint while old+new coexist.
Production has 18 TB available — not a constraint.

### Path B: stop services, rename Erlang files on disk, restart

Pros: no replication overhead.
Cons: requires service downtime; CouchDB stores docs in opaque
binary B-trees that don't tolerate filename mismatches.  Higher
risk of corruption.  Not worth the savings.

**Recommendation**: Path A.

## Execution sequence

1. Generate the rename map for the current state of production
   ([fixes/rename_dbs.py](../fixes/rename_dbs.py) — new
   one-shot script following CLAUDE.md's `fixes/` convention).
2. Replicate every old → new via `bin/replicate_dbs.py`.  Per the
   replication learnings from 2026-06: include
   `--no-bulk-get` until the upstream CouchDB multipart bug
   ships a fix.
3. For each experiment doc, rewrite the `databases.*` fields to
   point at the new names AND add the `_eval` counterparts.
4. Update the per-family pipeline modules
   ([bin/pipelines/v3_logistic.py](../bin/pipelines/v3_logistic.py),
   [bin/pipelines/v4_crf.py](../bin/pipelines/v4_crf.py)) so
   every truncated-dataset step writes via `{eval_*_db}`
   variables.
5. Update [bin/pipelines/base.py:build_variables](../bin/pipelines/base.py)
   so the default for `eval_ann_db` becomes
   `{annotations_db}_eval` instead of `{annotations_db}`.
6. Update [docs/couchdbs.md](../docs/couchdbs.md) per
   CLAUDE.md's "new CouchDB DB" rule.
7. After 2 weeks of stable operation against the new names, drop
   the old DBs.

## Out of scope

- Renaming non-CouchDB stores (Redis keys keep their current
  `skol:embedding:<X>` / `skol:classifier:model:<X>` shape — they
  already sort by namespace).
- Renaming Django URL paths / Search endpoints.
- Cross-tenant separation (production / staging / dev all share
  one CouchDB cluster today; this is fine for current scale).

## Open questions

1. ~~**Prefix scheme**~~ — settled 2026-06-09: per-experiment
   stage tag `<EXPERIMENT>_<STAGE_NUM>_<ROLE>` (e.g.
   `skol_exp_production_v4_01_00_ann`); shared DBs role-named
   with no stage tag.
2. **Eval-DB lifecycle** — auto-delete after N days?  Or
   long-lived?  Tied to whether you trust the `_eval`-suffix
   convention enough to prune.
3. **One DB rename pass or staged?** — All-at-once is simpler but
   the migration window is longer; staged (training first, then
   golden, then experiments) means more total work but smaller
   blast radius per step.
4. **Legacy DB pruning** — `skol_treatments_v3_dev` and friends:
   drop, archive, or leave behind?
5. **Model-variant tags in DB name vs experiment name** — today's
   ``skol_exp_production_v4_ann_combined`` carries the
   ``_combined`` corpus-variant tag at the DB level.  Two paths:
   (a) Keep at DB level — rename to
   ``skol_exp_production_v4_01_00_ann_combined``.  Simple
   migration; experiment-doc shape unchanged.  Downside: the
   variant lives at the DB-name level, not in the
   experiment-doc schema, so operators inspecting an experiment
   doc don't see which corpus variant its outputs were built
   from.
   (b) Fold into experiment name — rename the EXPERIMENT to
   ``production_v4_combined`` so the doc-name carries the
   variant tag, then DB name becomes the clean
   ``skol_exp_production_v4_combined_01_00_ann``.  Bigger
   migration (touches the experiment doc + every Redis key
   referencing it + the cron entries naming it), but the
   resulting schema is cleaner.
