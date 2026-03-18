# Managing and Running Experiments

Experiments track classifier configurations, databases, and Redis keys as
named documents in the `skol_experiments` CouchDB database. The `--experiment`
flag, available on every bin script, resolves these values automatically.

## Quick start

```bash
# Create an experiment
python bin/manage_experiment.py create --name my_exp \
  --notes "Logistic regression with augmented training data" \
  --training-db skol_training_augmented

# Train
python bin/train_classifier.py --model logistic_sections \
  --experiment my_exp --force --read-text --expire none

# Predict on the golden evaluation set
python bin/predict_classifier.py --model logistic_sections \
  --experiment my_exp --force --read-text

# Evaluate against hand annotations
mkdir -p golden/my_exp
python bin/evaluate_golden.py --experiment my_exp \
  --golden-db skol_golden_ann_hand \
  --save-to-experiment --output golden/my_exp/hand_annotated.json

# Review
python bin/manage_experiment.py show my_exp
```

## Experiment document schema

Each experiment is a CouchDB document in `skol_experiments`:

```json
{
  "_id": "my_exp",
  "notes": "Description of the experiment",
  "status": "draft",
  "databases": {
    "ingest":    "skol_dev",
    "training":  "skol_training",
    "taxa":      "skol_exp_my_exp_taxa",
    "taxa_full": "skol_exp_my_exp_taxa_full"
  },
  "redis_keys": {
    "classifier_model": "skol:classifier:model:my_exp",
    "embedding":        "skol:embedding:my_exp",
    "menus":            "skol:ui:menus_my_exp"
  },
  "evaluation": null,
  "created_at": "2026-03-14T...",
  "updated_at": "2026-03-14T..."
}
```

**Status lifecycle**: `draft` &rarr; `testing` &rarr; `evaluated` &rarr; `deployed` &rarr; `archived`

## How `--experiment` resolves config

When you pass `--experiment NAME` to any bin script, `get_env_config()` loads
the experiment document and overrides the following config keys:

| Experiment field | Config key(s) | Default (no experiment) |
|---|---|---|
| `databases.ingest` | `ingest_db_name` | `skol_dev` |
| `databases.training` | `training_database` | `skol_training` |
| `databases.taxa` | `taxon_db_name`, `source_db` | `skol_taxa_dev` |
| `databases.taxa_full` | `dest_db` | `skol_taxa_full_dev` |
| `redis_keys.classifier_model` | `classifier_model_key` | _(built from model_version)_ |
| `redis_keys.embedding` | `embedding_name` | `skol:embedding:v1.1` |

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
  [--notes "..."] [--training-db DB] [--ingest-db DB]

# List all experiments
python bin/manage_experiment.py list

# Show full document (JSON)
python bin/manage_experiment.py show NAME

# Update fields
python bin/manage_experiment.py update NAME \
  [--notes "..."] [--status STATUS] [--training-db DB] [--ingest-db DB]

# Archive (cannot archive "production")
python bin/manage_experiment.py archive NAME

# Deploy: copies databases + redis_keys to the "production" record
python bin/manage_experiment.py deploy NAME
```

Valid status values: `draft`, `testing`, `evaluated`, `deployed`, `archived`.

## Training a classifier

```bash
python bin/train_classifier.py --model logistic_sections \
  --experiment NAME --force --read-text --expire none
```

| Flag | Purpose |
|---|---|
| `--model MODEL` | Model config name (currently only `logistic_sections`) |
| `--experiment NAME` | Resolves training DB and Redis key from experiment |
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

Training always uses a dedicated training database (e.g., `skol_training` or a
variant). The golden databases are only used for evaluation — as the ingest
target for predictions and as the ground-truth source for scoring.

```bash
# Experiment 1: baseline on standard hand-annotated training data
python bin/manage_experiment.py create --name hand_baseline \
  --notes "Logistic regression on hand-annotated training" \
  --ingest-db skol_golden
mkdir -p golden/hand_baseline
python bin/train_classifier.py --model logistic_sections \
  --experiment hand_baseline --force --read-text --expire none
python bin/predict_classifier.py --model logistic_sections \
  --experiment hand_baseline --force --read-text
python bin/evaluate_golden.py --experiment hand_baseline \
  --golden-db skol_golden_ann_hand \
  --save-to-experiment --output golden/hand_baseline/hand_annotated.json

# Experiment 2: train on JATS-augmented training data
python bin/manage_experiment.py create --name jats_augmented \
  --notes "Logistic regression on JATS-augmented training" \
  --training-db skol_training_jats --ingest-db skol_golden
mkdir -p golden/jats_augmented
python bin/train_classifier.py --model logistic_sections \
  --experiment jats_augmented --force --read-text --expire none
python bin/predict_classifier.py --model logistic_sections \
  --experiment jats_augmented --force --read-text
python bin/evaluate_golden.py --experiment jats_augmented \
  --golden-db skol_golden_ann_hand \
  --save-to-experiment --output golden/jats_augmented/hand_annotated.json

# Compare
python bin/manage_experiment.py list
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
