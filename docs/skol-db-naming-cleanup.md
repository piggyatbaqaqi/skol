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

The convention is:

| Role | Production DB | Eval DB |
|---|---|---|
| Annotations | `skol_exp_<X>_ann` | `skol_exp_<X>_ann_eval` |
| Taxa | `skol_exp_<X>_taxa` | `skol_exp_<X>_taxa_eval` |
| Taxa full | `skol_exp_<X>_taxa_full` | `skol_exp_<X>_taxa_full_eval` |

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

## Naming-convention options for the rename pass

Three styles, in decreasing order of operator-readability:

### Option A — decimal-numeric prefix (RECOMMENDED, user-proposed 2026-06-09)

```
skol_01_00_ingest_dev
skol_02_00_training
skol_02_00_training_v3_combined_no_golden
skol_02_00_training_v3_hand
skol_02_00_training_v3_jats
skol_02_50_training_v2_no_golden   # inserted later; no renumbering
skol_03_00_golden_v1
skol_03_00_golden_ann_hand_v1
skol_03_00_golden_v2
skol_03_00_golden_ann_hand_v2
skol_03_00_golden_ann_jats
skol_03_00_golden_ann_bioc
skol_04_00_exp_<X>_ann
skol_04_00_exp_<X>_ann_eval
skol_05_00_exp_<X>_taxa
skol_05_00_exp_<X>_taxa_eval
skol_05_00_exp_<X>_taxa_full
skol_05_00_exp_<X>_taxa_full_eval
skol_99_00_experiments
```

Sort order: lexicographic gives `01_00 < 01_50 < 02_00 < 02_50
< 10_00 < 99_00`, matching pipeline order.

**Insertion property**: a new stage between 2 and 3 gets
`02_50`; between 2 and 2_50 gets `02_25`.  No renumbering churn.

**CouchDB constraint** noted: DB names must match
`[a-z][a-z0-9_$()+/-]*` — periods are not allowed.  The
underscore separator (`01_00` not `01.00`) keeps the design
property within the CouchDB-legal character set.

Pros: self-documenting, sorts correctly, extensible without
renaming.
Cons: longer names than alphabetic; slight cognitive load
mapping numbers to roles until the mapping is internalised.

### Option B — alphabetic prefix matching pipeline step order

```
skol_a_ingest_dev
skol_b_training_v3_combined_no_golden
skol_c_golden_v2
skol_c_golden_ann_hand_v2
skol_d_exp_<X>_ann
skol_d_exp_<X>_ann_eval
skol_e_exp_<X>_taxa
skol_e_exp_<X>_taxa_eval
skol_zz_experiments
```

Pros: sorts in pipeline order; cheap to insert a new stage
("between b and c, use ba_…").  No re-renumbering churn.
Cons: less self-documenting than the role name itself; operator
has to learn the letter-to-role mapping.

### Option C — role-named, no ordering prefix

```
skol_ingest_dev
skol_training_v3_combined_no_golden
skol_golden_v2
skol_golden_ann_hand_v2
skol_exp_<X>_ann
skol_exp_<X>_ann_eval
skol_exp_<X>_taxa
skol_exp_<X>_taxa_eval
skol_experiments
```

Pros: self-documenting, no prefix-numbering decisions.
Cons: doesn't satisfy the "sort in process order" goal — the
list still scrambles `experiments` between `exp_X` entries.

**Recommendation**: Option A.  Self-documenting + sortable +
extensible without renames.  Settled 2026-06-09.

## Scope — what gets renamed

| Old name | New name (Option A — decimal-numeric) |
|---|---|
| `skol_dev` | `skol_01_00_ingest_dev` |
| `skol_training` | `skol_02_00_training` |
| `skol_training_v3_combined_no_golden` | `skol_02_00_training_v3_combined_no_golden` |
| `skol_training_v3_hand` | `skol_02_00_training_v3_hand` |
| `skol_training_v3_jats` | `skol_02_00_training_v3_jats` |
| `skol_training_v2_no_golden` | `skol_02_00_training_v2_no_golden` |
| `skol_golden` | `skol_03_00_golden_v1` |
| `skol_golden_v2` | `skol_03_00_golden_v2` |
| `skol_golden_ann_hand` | `skol_03_00_golden_ann_hand_v1` |
| `skol_golden_ann_hand_v2` | `skol_03_00_golden_ann_hand_v2` |
| `skol_golden_ann_jats` | `skol_03_00_golden_ann_jats` |
| `skol_golden_ann_bioc` | `skol_03_00_golden_ann_bioc` |
| `skol_exp_<X>_ann*` | `skol_04_00_exp_<X>_ann*` |
| `skol_exp_<X>_taxa*` | `skol_05_00_exp_<X>_taxa*` |
| `skol_experiments` | `skol_99_00_experiments` |
| `skol_treatments_v3_dev` | drop after migration (legacy) |
| `skol_treatments_v3_jats` etc. | drop after migration (legacy) |

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

1. ~~**Prefix scheme**~~ — settled 2026-06-09: Option A
   (decimal-numeric, e.g. `skol_01_00_ingest_dev`).
2. **Eval-DB lifecycle** — auto-delete after N days?  Or
   long-lived?  Tied to whether you trust the `_eval`-suffix
   convention enough to prune.
3. **One DB rename pass or staged?** — All-at-once is simpler but
   the migration window is longer; staged (training first, then
   golden, then experiments) means more total work but smaller
   blast radius per step.
4. **Legacy DB pruning** — `skol_treatments_v3_dev` and friends:
   drop, archive, or leave behind?
