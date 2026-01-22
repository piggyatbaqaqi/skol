# Work-Skipping, Partial Computation, and Force-Recomputation Options

This document summarizes the current state of work-skipping options across SKOL pipeline programs and provides recommendations for unification.

## Current State by Program

### 1. ingest.py - Data Ingestion

**Purpose:** Ingest bibliographic data from web sources into CouchDB

| Option | Description |
|--------|-------------|
| `--publication KEY` | Process only specific publication source |
| `--all` | Process all predefined sources |
| `--blocked` | Include sources marked as blocked |

**Skip Behavior:**
- Source blocking mechanism (sources marked `blocked=True` are skipped unless `--blocked`)
- No explicit skip-existing; relies on CouchDB document ID for idempotency

**Missing Options:** `--dry-run`, `--skip-existing`, `--force`, `--limit`

---

### 2. train_classifier.py - Model Training

**Purpose:** Train text classification models and save to Redis

| Option | Description |
|--------|-------------|
| `--read-text` | Read from .txt attachment instead of PDF |
| `--save-text {eager,lazy}` | Text attachment handling mode |
| `--expire HH:MM:SS` | Redis key expiration time |

**Skip Behavior:** None - always retrains

**Missing Options:** `--dry-run`, `--skip-existing`, `--force`

---

### 3. predict_classifier.py - Model Inference

**Purpose:** Apply trained classifier to documents

| Option | Description |
|--------|-------------|
| `--read-text` | Read from .txt attachment instead of PDF |
| `--save-text {eager,lazy}` | Text attachment handling mode |

**Skip Behavior:**
- Checks if model exists in Redis (exits if missing)
- No per-document skipping; processes all matching documents

**Missing Options:** `--dry-run`, `--skip-existing`, `--force`, `--limit`, `--doc-id`

---

### 4. extract_taxa_to_couchdb.py - Taxon Extraction

**Purpose:** Extract Taxon objects from annotated documents

| Option | Description |
|--------|-------------|
| `--doc-id DOC_ID` | Process only specific document |

**Skip Behavior:**
- Uses idempotent composite keys: `taxon_{hash(doc_id, url, line_number)}`
- Update-or-insert logic prevents duplicates
- Skips taxa without nomenclature

**Missing Options:** `--dry-run`, `--skip-existing`, `--force`, `--limit`

---

### 5. embed_taxa.py - Compute Embeddings

**Purpose:** Compute sBERT embeddings and save to Redis

| Option | Description |
|--------|-------------|
| `--force` | Recompute even if embeddings exist |
| `--expire {SECONDS\|None}` | Redis key expiration |

**Skip Behavior:**
- **Checks if Redis key exists** - skips if found (unless `--force`)

**Missing Options:** `--dry-run`, `--limit`

---

### 6. taxa_to_json.py - Taxa Translation

**Purpose:** Translate taxa descriptions to structured JSON using Mistral

| Option | Description |
|--------|-------------|
| `--skip-existing` | Skip records already in destination database |
| `--incremental` | Save each record immediately (crash-resistant) |
| `--dry-run` | Preview without saving |
| `--limit N` | Process only first N records |

**Skip Behavior:**
- `--skip-existing`: Queries destination DB, filters out existing IDs

**Missing Options:** `--force`, `--doc-id`

---

### 7. regenerate_from_pdf.py (fixes/)

**Purpose:** Regenerate .txt from PDF and update .ann markers

| Option | Description |
|--------|-------------|
| `--doc-id DOC_ID` | Process only specific document |
| `--dry-run` | Preview without saving |
| `--skip-txt` | Skip regenerating .txt (only update .ann) |
| `--skip-ann` | Skip updating .ann (only regenerate .txt) |

**Skip Behavior:**
- Skips documents without PDF attachment
- Implicit eager mode (always regenerates)

**Missing Options:** `--skip-existing`, `--force`, `--limit`

---

### 8. regenerate_txt_with_pages.py (fixes/)

**Purpose:** Regenerate .txt attachments with page markers

| Option | Description |
|--------|-------------|
| `--doc-id DOC_ID` | Process only specific document |
| `--dry-run` | Preview without saving |
| `--pattern PATTERN` | Filter by attachment pattern |

**Skip Behavior:**
- Implicit eager mode (always regenerates)

**Missing Options:** `--skip-existing`, `--force`, `--limit`

---

## Summary Matrix

| Program | --dry-run | --skip-existing | --force | --limit | --doc-id | --incremental |
|---------|:---------:|:---------------:|:-------:|:-------:|:--------:|:-------------:|
| ingest.py | - | - | - | - | - | - |
| train_classifier.py | - | - | - | - | - | - |
| predict_classifier.py | - | - | - | - | - | - |
| extract_taxa_to_couchdb.py | - | (idempotent) | - | - | YES | - |
| embed_taxa.py | - | (auto) | YES | - | - | - |
| taxa_to_json.py | YES | YES | - | YES | - | YES |
| regenerate_from_pdf.py | YES | - | - | - | YES | - |
| regenerate_txt_with_pages.py | YES | - | - | - | YES | - |

---

## Identified Patterns

### Pattern 1: Idempotent Processing
Used by `extract_taxa_to_couchdb.py`:
- Deterministic output IDs based on input hash
- Safe to re-run; duplicates impossible
- Update-or-insert logic

### Pattern 2: Conditional Skip on Output Existence
Used by `embed_taxa.py`:
- Check if output exists before computing
- Skip by default, recompute with `--force`

### Pattern 3: Explicit Skip-Existing
Used by `taxa_to_json.py`:
- Query destination for existing record IDs
- Filter source records before processing
- Useful for crash recovery

### Pattern 4: Text Attachment Handling
Used by `train_classifier.py`, `predict_classifier.py`:
- `--read-text`: Read from .txt instead of PDF
- `--save-text eager`: Always save/replace
- `--save-text lazy`: Save only if missing

### Pattern 5: Incremental Processing
Used by `taxa_to_json.py`:
- Save each record immediately
- Crash-resistant; progress preserved

---

## Recommendations for Unified Options

### Proposed Standard Options

All programs should support these options where applicable:

```
Work Control:
  --dry-run           Preview what would be done without making changes
  --skip-existing     Skip records/documents that already have output
  --force             Process even if output already exists (overrides --skip-existing)

Partial Processing:
  --limit N           Process at most N records
  --doc-id ID         Process only the specified document ID
  --pattern PATTERN   Filter by document/attachment pattern

Incremental Mode:
  --incremental       Save each record as it completes (crash-resistant)

Text Handling (for PDF-based pipelines):
  --read-text         Read from .txt attachment instead of converting PDF
  --save-text MODE    Text attachment strategy: eager|lazy|none
```

### Option Semantics

#### --dry-run
- Show what would be processed without side effects
- Print counts, sample output, validation results
- Exit code 0 on success preview

#### --skip-existing
- Query output destination for existing records
- Filter out records that already have output
- For CouchDB: check if document/attachment exists
- For Redis: check if key exists
- Useful for crash recovery and incremental runs

#### --force
- Process all records regardless of existing output
- Overwrite/replace existing output
- Takes precedence over `--skip-existing`
- Use case: Reprocess after code changes

#### --limit N
- Process at most N records
- Applied after `--skip-existing` filtering
- Useful for testing and debugging

#### --doc-id ID
- Process only the specified document
- Mutually exclusive with `--limit` (or `--limit` ignored)

#### --incremental
- Save/commit each record immediately after processing
- Progress preserved on crash
- Default: batch mode (process all, then save all)

### Implicit vs Explicit Behavior

**Current inconsistency:**
- `embed_taxa.py`: Implicit skip (auto-checks Redis)
- `extract_taxa_to_couchdb.py`: Idempotent (safe to rerun)
- `predict_classifier.py`: No skip (always processes all)

**Recommendation:**
Programs should be **explicit** about their skip behavior:
1. Default: Process all (no implicit skip)
2. `--skip-existing`: Enable skip behavior
3. `--force`: Override skip, force reprocessing

Exception: Idempotent programs (like `extract_taxa_to_couchdb.py`) don't need skip options since re-running is safe and produces the same output.

### Implementation Priority

1. **High Priority** (frequently re-run, long-running):
   - `predict_classifier.py`: Add `--dry-run`, `--skip-existing`, `--doc-id`, `--limit`
   - `train_classifier.py`: Add `--dry-run`, `--force` (for Redis key overwrite)

2. **Medium Priority** (occasionally re-run):
   - `ingest.py`: Add `--dry-run`, `--limit`
   - `extract_taxa_to_couchdb.py`: Add `--dry-run`, `--limit`

3. **Low Priority** (fixes/, already have good options):
   - `regenerate_from_pdf.py`: Add `--limit`
   - `regenerate_txt_with_pages.py`: Add `--limit`

### Checking for Existing Output

Each program needs to define what "existing output" means:

| Program | Output | How to Check "Exists" |
|---------|--------|----------------------|
| ingest.py | CouchDB document | `doc_id in db` |
| train_classifier.py | Redis model | `redis.exists(key)` |
| predict_classifier.py | .ann attachment | `'.ann' in doc['_attachments']` |
| extract_taxa_to_couchdb.py | CouchDB taxon doc | `taxon_id in db` (already idempotent) |
| embed_taxa.py | Redis embedding | `redis.exists(key)` (already implemented) |
| taxa_to_json.py | CouchDB taxa_full doc | `doc_id in dest_db` (already implemented) |
| regenerate_*.py | .txt attachment | `'.txt' in doc['_attachments']` |

---

## Code Locations for Implementation

### predict_classifier.py
- Add skip logic in `predict_and_save()` around line 175
- Check for `.ann` attachment existence before processing document
- Filter documents using Spark before `classifier.predict()`

### train_classifier.py
- Add force check in `train_and_save()` around line 89
- Check `redis_client.exists(classifier_model_name)`
- Skip if exists and not `--force`

### ingest.py
- Add dry-run mode in ingestion loop
- Show what would be ingested without calling `db.save()`

### extract_taxa_to_couchdb.py
- Add dry-run mode in `process_document()`
- Show extracted taxa without saving to database

---

## Notes on Idempotency

Programs that are already idempotent may not need `--skip-existing`:

1. **extract_taxa_to_couchdb.py**: Uses deterministic hash-based IDs
   - Re-running produces same taxa with same IDs
   - Existing documents updated (not duplicated)
   - `--skip-existing` would only save time, not correctness

2. **ingest.py**: Uses source-derived document IDs
   - Re-ingesting same source updates existing documents
   - Generally safe to re-run

For these programs, `--skip-existing` is an **optimization**, not a correctness requirement.
