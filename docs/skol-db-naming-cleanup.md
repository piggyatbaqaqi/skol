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

### Why `_eval` stays at the DB level, not in the experiment name

Considered 2026-06-09: should `_eval` be folded into the
experiment name (e.g. `production_v4` and `production_v4_eval`
as separate experiment docs), symmetric with the Option B
treatment of `_combined` (a model-corpus variant)?

**Decision: keep `_eval` as a DB-level suffix.**  The reasoning
draws a principled line between two kinds of "variant":

| Property | `_combined` (variant) | `_eval` (measurement) |
|---|---|---|
| Different model? | Yes | No |
| Different production outputs? | Yes | No (same `.ann`, different doc set) |
| Belongs in a research write-up as a separate experiment? | Yes | No — eval IS the measurement |

`_combined` is a different experimental condition (different
training corpus → different model → different outputs);
folding it into the experiment name is appropriate.  `_eval`
is the same experiment evaluated against a smaller dataset
with known answer keys; folding it into the experiment name
would force a conceptual split between production and its
measurement.  Costs of doing so: paired experiment docs to
maintain, redis-key sharing convention to define, awkward
`train` step semantics on the eval doc ("which model does it
train?"), and the rhetorical question "production_v4 achieved
F1=X — but the F1 is on the production_v4_eval doc, and the
production_v4 doc has F1=null".

### Deferred follow-on: UI synthetic-selector for eval browsing

The DB-level approach leaves one operator concern open: the
search-UI experiment-selector shows `production_v4` but
operators may want to browse the 105 golden-set treatments
separately from the 30k production treatments.

**Deferred mini-project (post-rename)**: extend the Django
experiment-selector to compute a synthetic "production_v4
(eval)" entry that points at the same experiment doc but with
a `view: eval` flag the view layer reads.  When eval-view is
selected, all DB-name resolutions in the search code swap from
production DBs (e.g. `..._03_00_treatments_structured`) to
their `_eval` siblings (`..._03_00_treatments_structured_eval`).

Estimated scope: ~20 lines in `django/search/` view code +
selector template, plus a convention that any per-experiment-doc
UI element accepts a `?view=eval` query param.  Single place
to maintain the convention.

Sequenced after this rename pass lands, so the
`{eval_*_db}` variable substitution layer is already in place.
Captured here so we don't forget the operator-convenience hook
when we get there.

**Open caveat**: if eval ever becomes writable through the UI
(hand-corrections of eval predictions that should propagate to
the eval DB), the synthetic-selector approach needs the writer
code to also honor the `view` flag, with non-trivial
write-to-wrong-DB risk.  Today eval is read-only from the UI,
so this isn't a current concern — flag it if/when the
hand-correction interface gets extended to eval data.

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

## Decisions

All four open questions resolved 2026-06-09.

### 1. Prefix scheme

Per-experiment stage tag `<EXPERIMENT>_<STAGE_NUM>_<ROLE>` (e.g.
`skol_exp_production_v4_01_00_ann`); shared DBs role-named with
no stage tag.

### 2. Eval-DB lifecycle — keep long-lived, add freshness check

Eval DBs persist alongside production DBs, NOT auto-deleted.
Recompute on a 105-doc golden set is cheap, but the debug value
of having exact predictions saved (grep "which doc did treatment
X come from") outweighs the storage cost.

**Mitigation against stale-data-as-truth syndrome**: add a
freshness check to
[bin/evaluate_golden.py](../bin/evaluate_golden.py) — refuse to
run `score_golden` unless the eval predictions were generated
against the current model Redis key (compare a timestamp /
model-key fingerprint stamped onto the eval DB on
`predict_golden`).  Require `--force` to override.  ~10 lines.

Without this, an F1 number you cite today could be from a
predict commit 3 months stale — exactly the failure mode the
eval-DB split was supposed to prevent.

### 3. Migration — one all-at-once pass, single session

Staged-across-sessions actually increases hallucination risk
because it forces inter-session memory reconstruction
("where are we in the migration?") — the failure mode I'm most
prone to.  All-at-once in a single dedicated session with full
context loaded is lower-risk *if*:

1. **Exhaustive grep first**: build a checklist of every match
   for the old names (`taxa`, `taxa_full`, `_full` in
   DB context, `extract_taxa` step name, `databases.taxa`,
   `treatments_full_database` config key, etc.) before
   changing any code.
2. **Incremental commits within the session**, one logical
   surface per commit so PR review stays tractable:
   pipeline modules → env_config → experiment-doc fields →
   evaluate_golden freshness check → Django → cron → docs.
3. **Test suite + grep regression** at the end as a safety
   net.
4. **Q5 experiment-doc rename committed LAST and separately**
   — that touches the doc `_id`, Redis keys, cron-references,
   and is the highest-blast-radius single surface.

### 4. Legacy DB pruning

Drop after migration: `skol_treatments_v3_dev`,
`skol_treatments_v3_jats`, and any other ad-hoc legacy DBs
that predate the experiment-doc-driven naming.  No archive —
they're reproducible from the original ingest + experiment
doc if ever needed.

### 5. Model-variant tags → fold into experiment name

`production_v4` with DB `skol_exp_production_v4_ann_combined`
becomes experiment `production_v4_combined` with DB
`skol_exp_production_v4_combined_01_00_ann`.

Rationale: the experiment doc is what shows up in the
search-UI experiment selector, so the variant tag being part
of the experiment identity (not a hidden DB-name suffix) means
operators see "I'm looking at production_v4_combined" rather
than "I'm looking at production_v4 but somehow this is the
combined variant."

**Extra migration burden**: the doc `_id` is the experiment
key, so renaming is not in-place.  Touches:

- CouchDB: copy doc to new `_id`, delete the old (no
  in-place doc rename in CouchDB).
- Per-experiment DBs: rename incorporates the variant into
  the slug, same migration path as everything else.
- Redis keys derived from experiment name:
  `skol:embedding:production_v4` → `skol:embedding:production_v4_combined`,
  same for `skol:classifier:model:*` if pinned per-experiment,
  same for `skol:ui:menus_*`.
- Cron entries referencing the experiment name
  ([debian/skol.cron](../debian/skol.cron) does reference
  experiment names).
- The pipeline-state field on the doc (preserved as part of
  the copy).

This is the highest-blast-radius single change in the plan;
commit it last and separately so a problem here doesn't
contaminate the DB-rename commits.
