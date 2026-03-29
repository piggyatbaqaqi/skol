# Managing and Running Experiments

Experiments track classifier configurations, databases, and Redis keys as
named documents in the `skol_experiments` CouchDB database. The `--experiment`
flag, available on every bin script, resolves these values automatically.

## Quick start

```bash
# Create an experiment and run the full pipeline automatically
python bin/manage_experiment.py create --name my_exp \
  --notes "Logistic regression with augmented training data" \
  --training-db skol_training_augmented \
  --model-name logistic_sections_taxpub_v1

# Run each step in sequence (repeat until done)
python bin/manage_experiment.py runnext my_exp

# Or run the whole pipeline step by step manually
python bin/manage_experiment.py pipeline my_exp   # check status at any time
python bin/manage_experiment.py show my_exp       # review results
```

## Experiment document schema

Each experiment is a CouchDB document in `skol_experiments`:

```json
{
  "_id": "my_exp",
  "model_name": "logistic_sections_taxpub_v1",
  "notes": "Description of the experiment",
  "status": "draft",
  "databases": {
    "ingest":       "skol_dev",
    "training":     "skol_training",
    "annotations":  "skol_exp_my_exp_ann",
    "taxa":         "skol_exp_my_exp_taxa",
    "taxa_full":    "skol_exp_my_exp_taxa_full"
  },
  "redis_keys": {
    "classifier_model": "skol:classifier:model:my_exp",
    "embedding":        "skol:embedding:my_exp",
    "menus":            "skol:ui:menus_my_exp"
  },
  "evaluation": null,
  "pipeline": {
    "current_step": 0,
    "steps": [
      {"name": "train",         "status": "pending", "started_at": null, "completed_at": null},
      {"name": "predict",       "status": "pending", "started_at": null, "completed_at": null},
      {"name": "annotate_jats", "status": "pending", "started_at": null, "completed_at": null},
      {"name": "extract_taxa",  "status": "pending", "started_at": null, "completed_at": null},
      {"name": "embed",         "status": "pending", "started_at": null, "completed_at": null},
      {"name": "evaluate",      "status": "pending", "started_at": null, "completed_at": null},
      {"name": "build_vocab",   "status": "pending", "started_at": null, "completed_at": null}
    ]
  },
  "created_at": "2026-03-14T...",
  "updated_at": "2026-03-14T..."
}
```

**Status lifecycle**: `draft` &rarr; `testing` &rarr; `evaluated` &rarr; `deployed` &rarr; `archived`

**`model_name`** names the entry in `MODEL_CONFIGS` (in `bin/train_classifier.py`) to use for training
and prediction. When `--experiment` is passed to any script, this value is used as the default for
`--model` so you don't need to repeat it on every command.

## How `--experiment` resolves config

When you pass `--experiment NAME` to any bin script, `get_env_config()` loads
the experiment document and overrides the following config keys:

| Experiment field | Config key(s) | Default (no experiment) |
|---|---|---|
| `databases.ingest` | `ingest_db_name` | `skol_dev` |
| `databases.training` | `training_database` | `skol_training` |
| `databases.annotations` | `annotations_db_name` | `skol_training` |
| `databases.taxa` | `taxon_db_name`, `source_db` | `skol_taxa_dev` |
| `databases.taxa_full` | `dest_db` | `skol_taxa_full_dev` |
| `redis_keys.classifier_model` | `classifier_model_key` | _(built from model_version)_ |
| `redis_keys.embedding` | `embedding_name` | `skol:embedding:v1.1` |
| `redis_keys.menus` | `menus_key` | _(none)_ |
| `model_name` | `model_name` | _(none — must pass `--model` explicitly)_ |

**Priority** (highest to lowest):

1. CLI arguments (e.g., `--ingest-db-name bar`)
2. Experiment values
3. Environment variables
4. `/home/skol/.skol_env` file
5. Hardcoded defaults

CLI args always win. `--experiment my_exp --ingest-db-name custom_db` uses
`custom_db` for ingest but still resolves the Redis key and other databases
from the experiment.

## Managing experiments

All commands go through `bin/manage_experiment.py`:

```bash
# Create
python bin/manage_experiment.py create --name NAME \
  [--notes "..."] [--model-name MODEL] \
  [--training-db DB] [--ingest-db DB]

# List all experiments
python bin/manage_experiment.py list

# Show full document (JSON)
python bin/manage_experiment.py show NAME

# Update fields
python bin/manage_experiment.py update NAME \
  [--notes "..."] [--status STATUS] [--model-name MODEL] \
  [--training-db DB] [--ingest-db DB]

# Archive (cannot archive "production")
python bin/manage_experiment.py archive NAME

# Deploy: copies databases + redis_keys to the "production" record
python bin/manage_experiment.py deploy NAME
```

Valid status values: `draft`, `testing`, `evaluated`, `deployed`, `archived`.

## Pipeline orchestration

Every experiment has a built-in pipeline that runs the full workflow in order.
Use these subcommands instead of invoking the individual scripts directly.

### Step order and dependencies

```
1. train          ─┐
2. predict         │  sequential: each requires the previous to be
3. annotate_jats   │  completed or skipped before it can run
4. extract_taxa    │
5. embed          ─┘
6. evaluate    ─┐  independent: both require steps 1-5 done,
7. build_vocab ─┘  but can run in either order
```

- `predict` runs the ML classifier on non-TaxPub documents (`is_taxpub=False`)
- `annotate_jats` runs `jats_to_yedda` on TaxPub documents (`is_taxpub=True`)
- Both write to `databases.annotations`; both must complete before `extract_taxa`

### `pipeline` — show status

```bash
python bin/manage_experiment.py pipeline NAME
```

Prints a status table with start/completion timestamps. Warns about failed
steps and steps that have been `running` for more than an hour (possible stall).

### `runnext` — run the next pending step

```bash
python bin/manage_experiment.py runnext NAME [--force]
```

Finds the first step with `status=pending`, checks that its dependencies are
met, then runs it.  Exits non-zero if the step fails.  Use `--force` to pass
`--force` through to the underlying script (replacing `--skip-existing`).

```bash
# Iterate until the whole pipeline is done:
while python bin/manage_experiment.py runnext my_exp; do :; done
```

### `runstep` — run a named step (no dependency check)

```bash
python bin/manage_experiment.py runstep NAME STEP[,STEP,...] [--force]
```

Runs one or more named steps in the order given, bypassing dependency checks.
Use this to re-run a specific step or run `evaluate` and `build_vocab` in
parallel in separate terminals.

```bash
python bin/manage_experiment.py runstep my_exp evaluate
python bin/manage_experiment.py runstep my_exp extract_taxa,embed
python bin/manage_experiment.py runstep my_exp evaluate,build_vocab
```

### `resetstep` — mark steps as pending

```bash
python bin/manage_experiment.py resetstep NAME STEP[,STEP,...]
```

Resets one or more steps to `pending` and clears their timestamps.
Use this to retry a failed step or force re-execution of a completed step.

```bash
python bin/manage_experiment.py resetstep my_exp extract_taxa
python bin/manage_experiment.py resetstep my_exp extract_taxa,embed
```

### `skipstep` — mark steps as skipped

```bash
python bin/manage_experiment.py skipstep NAME STEP[,STEP,...]
```

Marks steps as `skipped`.  Skipped steps count as done for dependency
purposes.  Use this when a step is not applicable for a particular experiment
(e.g., `annotate_jats` when there are no TaxPub documents, or `build_vocab`
when a vocabulary tree is not needed).

```bash
python bin/manage_experiment.py skipstep my_exp annotate_jats
python bin/manage_experiment.py skipstep my_exp build_vocab
```

### Step status values

| Status | Meaning |
|---|---|
| `pending` | Not yet run |
| `running` | Currently executing (set at subprocess launch) |
| `completed` | Finished with exit code 0 |
| `failed` | Finished with non-zero exit code |
| `skipped` | Explicitly skipped; treated as done for dependency purposes |

### What each step runs

| Step | Script | Key flags |
|---|---|---|
| `train` | `train_classifier.py` | `--experiment NAME --force` |
| `predict` | `predict_classifier.py` | `--experiment NAME --incremental --skip-existing` |
| `annotate_jats` | `jats_to_yedda.py` | `--experiment NAME --all --taxpub-only --output-to couchdb --skip-existing` |
| `extract_taxa` | `extract_taxa_to_couchdb.py` | `--experiment NAME --skip-existing` |
| `embed` | `embed_taxa.py` | `--experiment NAME --force` |
| `evaluate` | `predict_classifier.py` then `evaluate_golden.py` | predict: `--golden-db skol_golden --skip-existing`; evaluate: `--golden-db skol_golden_ann_hand --plaintext-db skol_golden --save-to-experiment` |
| `build_vocab` | `build_vocab_tree.py` | `--experiment NAME` |

The `evaluate` step runs two commands in sequence: it first predicts on the
105-document golden set (`skol_golden`), then scores those predictions against
the 30 hand-annotated gold standards (`skol_golden_ann_hand`).

`--force` on `runnext`/`runstep` replaces `--skip-existing` with `--force` in
every command of the step.

## Training a classifier

```bash
python bin/train_classifier.py --experiment NAME --force
```

`--model` defaults to `experiment.model_name` when `--experiment` is set;
pass it explicitly only to override.

| Flag | Purpose |
|---|---|
| `--model MODEL` | Model config name; defaults to `experiment.model_name` or `logistic_sections` |
| `--experiment NAME` | Resolves training DB, Redis key, and model name from experiment |
| `--force` | Train even if a model already exists in Redis |
| `--read-text` | Use `article.txt` attachments (not PDF extraction) |
| `--expire HH:MM:SS` | Set Redis TTL (`none` = never expires) |
| `--dry-run` | Preview without training |

The trained model is saved to the Redis key from
`experiment.redis_keys.classifier_model` (e.g.,
`skol:classifier:model:my_exp`). Without `--experiment`, the key is built as
`skol:classifier:model:{model}_{model_version}`.

## Running predictions

```bash
python bin/predict_classifier.py --model logistic_sections \
  --experiment NAME --force --read-text
```

| Flag | Purpose |
|---|---|
| `--experiment NAME` | Resolves ingest DB and Redis model key |
| `--force` | Re-predict even if `.ann` attachments exist |
| `--skip-existing` | Skip documents that already have `.ann` |
| `--incremental` | Save after each batch (crash-resistant) |
| `--taxonomy-filter` | Only predict on documents with taxonomy abbreviations |
| `--skip-golden` | Skip documents marked as golden dataset members |
| `--include-taxpub` | Also predict on `is_taxpub=True` documents (skipped by default) |
| `--golden-db DB` | Restrict to doc IDs in DB and force `--read-text` (for golden-set evaluation) |
| `--limit N` | Process at most N documents |
| `--doc-id ID1,ID2` | Process only specific document IDs |
| `--read-text` | Use `article.txt` attachments (not PDF extraction) |
| `--dry-run` | Preview without saving |

Predictions are saved as `.ann` attachments (YEDDA format) on each CouchDB
document in the experiment's ingest database.

## Evaluating against the golden dataset

```bash
mkdir -p golden/NAME
python bin/evaluate_golden.py --experiment NAME \
  --golden-db skol_golden_ann_hand \
  --save-to-experiment --output golden/NAME/hand_annotated.json -vv
```

| Flag | Purpose |
|---|---|
| `--experiment NAME` | Use experiment's ingest DB as the predicted source |
| `--predicted-db DB` | Alternative: specify predicted DB directly (mutually exclusive with `--experiment`) |
| `--golden-db DB` | Database with ground-truth `.txt.ann` annotations (required) |
| `--save-to-experiment` | Write metrics back to the experiment document |
| `--output FILE` | Write detailed JSON results to a file |
| `-v` / `-vv` | Increase verbosity (`-vv` shows confusion matrix) |
| `-q` | Suppress output |

**Metrics reported**:
- **Block-level**: per-tag precision, recall, F1 (exact text match of YEDDA blocks)
- **Token-level IoU**: character-level intersection-over-union per tag, plus micro average
- **Macro F1**: averaged across all tags

When `--save-to-experiment` is used, the experiment document is updated with
evaluation results and its status changes to `evaluated`.

## Golden databases

Golden databases are for **evaluation only** — never use them as training data.

| Database | Contents |
|---|---|
| `skol_golden` | 105 curated articles with `article.txt` plaintext |
| `skol_golden_ann_hand` | 30 hand-annotated `.txt.ann` (gold standard) |
| `skol_golden_ann_jats` | 75 JATS-derived `.txt.ann` (silver standard) |

These are created by `bin/curate_golden_dataset.py --all`. See
`docs/GOLDEN_DATASET_AND_EXPERIMENTS.md` for the curation methodology.

## Output directory convention

All artifacts for an experiment live under `golden/<experiment_name>/`:

```
golden/
  hand_annotated/
    train_classifier.log
    hand_annotated.json
  jats_baseline/
    train_classifier.log
    hand_annotated.json
```

## End-to-end example: comparing two training sources

```bash
# Experiment 1: baseline (hand-annotated training data)
python bin/manage_experiment.py create --name hand_baseline \
  --notes "Logistic regression on hand-annotated training" \
  --model-name logistic_sections
python bin/manage_experiment.py skipstep hand_baseline annotate_jats,extract_taxa,embed,build_vocab
while python bin/manage_experiment.py runnext hand_baseline; do :; done

# Experiment 2: TaxPub-augmented training data
python bin/manage_experiment.py create --name taxpub_v2 \
  --notes "Logistic regression on JATS/TaxPub-augmented training" \
  --training-db skol_training_taxpub_v1 \
  --model-name logistic_sections_taxpub_v1
while python bin/manage_experiment.py runnext taxpub_v2; do :; done

# Compare results
python bin/manage_experiment.py list
python bin/manage_experiment.py show hand_baseline
python bin/manage_experiment.py show taxpub_v2
```

## Environment variables

These can be set in `/home/skol/.skol_env` or exported in the shell.
`--experiment` overrides the database and Redis key values below.

| Variable | Default | Purpose |
|---|---|---|
| `EXPERIMENT_NAME` | _(empty)_ | Same as `--experiment` |
| `COUCHDB_URL` | `http://localhost:5984` | CouchDB server URL |
| `COUCHDB_USER` | `admin` | CouchDB username |
| `COUCHDB_PASSWORD` | `SU2orange!` | CouchDB password |
| `INGEST_DB_NAME` | `skol_dev` | Default ingest database |
| `TRAINING_DATABASE` | `skol_training` | Default training database |
| `MODEL_VERSION` | `v2.0` | Appended to Redis key when no experiment is set |
| `REDIS_HOST` | `localhost` | Redis server host |
| `REDIS_PORT` | `6380` | Redis server port |
| `REDIS_USERNAME` | `admin` | Redis username |
| `REDIS_PASSWORD` | _(empty)_ | Redis password |
| `REDIS_TLS` | `false` | Use TLS for Redis connections |
