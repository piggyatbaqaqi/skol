# Treatment-Complete Annotation Architecture

## Context

The current classifier uses a simplified 8-tag label set that collapses the full taxonomic
treatment structure. Evaluation uses an even coarser 3-class scheme (Nomenclature /
Description / Misc-exposition). This under-represents the standardized sections of a
taxonomic treatment (Diagnosis, Distribution, Materials-examined, Type-designation, Biology)
and provides no mechanism for marking sub-section entities like taxon names, authorship
strings, DOIs, or MycoBank numbers.

The goal is a two-layer annotation model:
- **Layer 1 (YEDDA .ann)**: Section-level treatment structure — expanded to 12 labels
  aligned with TaxPub/Plazi ontology. Trained with the existing ML classifier.
- **Layer 2 (.spans.json)**: Character-offset entity spans — taxon names (gnfinder REST
  API), authorship (gnparser REST API), and structured particles (regex). Populated by a
  new pipeline step.

The two layers are complementary: Layer 1 answers "what kind of section is this?";
Layer 2 answers "where are the specific named entities?". Together they enable richer
search, cross-linking to GBIF/CoL/MycoBank, and span-feature-enhanced classification.

The previous plan (pipeline orchestration) is **fully implemented** — all Phase 1–3
subcommands (runnext, runstep, resetstep, skipstep, pipeline) are live, documented in
`docs/experiments.md`, and committed to main.

---

## Architecture Overview

```
JATS XML / plaintext
        │
        ├─► jats_to_yedda.py ──► article.txt.ann    (Layer 1: 12-tag YEDDA)
        │
        └─► annotate_spans.py ─► article.spans.json  (Layer 2: entity spans)
                 │
                 ├── gnfinder REST API  → TaxonName spans (+ nomenclatural annotation)
                 ├── gnparser REST API  → Author/Year spans (trailing context of hits)
                 └── particle_detector  → DOI, MB-number, Page-ref, CollectionCode spans
```

Span record schema:
```json
{
  "version": "1",
  "doc_id": "abc123",
  "source_attachment": "article.txt",
  "spans": [
    {
      "start": 1024, "end": 1038, "label": "TaxonName",
      "text": "Pardosa moesta", "source": "gnfinder", "confidence": 0.99,
      "metadata": {"canonical": "Pardosa moesta", "cardinality": 2,
                   "annot_nomen": "sp. nov.", "annot_nomen_type": "SP_NOV"}
    },
    {
      "start": 1039, "end": 1051, "label": "Author",
      "text": "Banks, 1892", "source": "gnparser"
    },
    {
      "start": 500, "end": 512, "label": "MB-number",
      "text": "MB 123456", "source": "regex"
    }
  ]
}
```

---

## Phase 1 — Expanded Label Set

**Goal**: Add 4 new section-level tags, split Holotype into two more precise tags,
and update all downstream code that uses the tag enum.

### 1a. `ingestors/yedda_tags.py`

Replace 8-tag enum with 12 tags:

```python
class Tag(str, Enum):
    NOMENCLATURE       = "Nomenclature"
    DESCRIPTION        = "Description"
    DIAGNOSIS          = "Diagnosis"           # NEW: differential diagnosis
    ETYMOLOGY          = "Etymology"
    DISTRIBUTION       = "Distribution"        # NEW: geographic range
    MATERIALS_EXAMINED = "Materials-examined"  # replaces Holotype (multiple specimens)
    TYPE_DESIGNATION   = "Type-designation"    # NEW: holotype/lectotype line (split from Holotype)
    BIOLOGY            = "Biology"             # NEW: ecology / host / habitat
    NOTES              = "Notes"
    KEY                = "Key"
    FIGURE_CAPTION     = "Figure-caption"
    MISC_EXPOSITION    = "Misc-exposition"
```

Keep `Holotype = "Holotype"` as a deprecated alias so existing .ann files remain readable.

### 1b. `ingestors/jats_to_yedda.py` — `sec_type_to_tag()`

Updated mappings (additions only; existing mappings unchanged unless noted):

| sec-type value(s) | New tag |
|---|---|
| `diagnosis` | `Diagnosis` |
| `distribution`, `habitat`, `habitat-distribution` | `Distribution` |
| `holotype`, `type material`, `type species`, `type genus` | `Type-designation` |
| `material`, `materials examined`, `specimens examined` | `Materials-examined` |
| `biology`, `ecology`, `host` | `Biology` |
| morphological subtypes (fruiting body, spores, …) | `Description` (unchanged) |

### 1c. `bin/evaluate_golden.py`

- Remove the hardcoded `_COLLAPSE_MAP` that forces 3-class evaluation.
- Add optional `--collapse-tags SRC:DST,...` flag for custom collapsing (e.g. `--collapse-tags Diagnosis:Description`).
- Default: evaluate all tags as-is.

### 1d. `bin/train_classifier.py`

- Add a `logistic_sections_v3` config (12-class, uses `skol_training` with all sources).
- After training data regeneration, recalculate class weights using inverse-frequency
  of the new label distribution.

### 1e. Regeneration and retraining sequence

```bash
python bin/curate_golden_dataset.py --all --no-mark -v   # skol_golden_ann_jats
python bin/jats_to_yedda.py --all --taxpub-only \
    --output-to couchdb --output-database skol_training_taxpub_v1 \
    --experiment taxpub_v1_onnx_int8 --force              # training corpus
python bin/manage_experiment.py resetstep taxpub_v1_onnx_int8 train
python bin/manage_experiment.py runstep taxpub_v1_onnx_int8 train
python bin/manage_experiment.py runstep taxpub_v1_onnx_int8 evaluate --force
```

### Files (Phase 1)

| File | Change |
|---|---|
| `ingestors/yedda_tags.py` | Add 4 new tags; deprecate Holotype |
| `ingestors/jats_to_yedda.py` | Update `sec_type_to_tag()` |
| `bin/evaluate_golden.py` | Remove hardcoded collapse; add `--collapse-tags` |
| `bin/train_classifier.py` | Add 12-class model config |
| `ingestors/yedda_tags_test.py` | Tests for new tags + backward compat |
| `ingestors/jats_to_yedda_test.py` | Tests for new sec-type mappings |

---

## Phase 2 — Span Layer

### 2a. `ingestors/gnfinder_client.py`  *(new)*

Thin wrapper around `https://finder.globalnames.org/api/v1/find` (or configurable URL).

```python
@dataclass
class NameSpan:
    start: int; end: int; verbatim: str; canonical: str
    cardinality: int; odds_log10: float
    annot_nomen: str; annot_nomen_type: str   # "SP_NOV", "COMB_NOV", "NO_ANNOT"

def find_names(text: str, gnfinder_url: str = _DEFAULT_URL,
               verify: bool = False) -> List[NameSpan]: ...
```

Retry with exponential backoff; configurable timeout. POST body: `{"text": "..."}`.

### 2b. `ingestors/gnparser_client.py`  *(new)*

Wrapper around `https://parser.globalnames.org/api/v1` (or configurable URL).

```python
@dataclass
class ParsedAuthorship:
    verbatim: str        # "Banks, 1892"
    offset_in_input: int # characters from start of the string passed in
    length: int
    year: str
    authors: List[str]

def parse_authorship_after_name(
    text: str,           # window: text[name_end : name_end + 80]
    gnparser_url: str = _DEFAULT_URL,
) -> Optional[ParsedAuthorship]: ...
```

### 2c. `ingestors/particle_detector.py`  *(new)*

Mostly regex-based, but fungarium codes are loaded from the live Redis registry
(populated by `bin/manage_fungaria.py download`, refreshed by `bin/rebuild_redis.py`).

```python
PATTERNS: Dict[str, re.Pattern] = {
    "DOI":      re.compile(r'\b(10\.\d{4,}/\S+)', re.IGNORECASE),
    "MB-number": re.compile(r'\bMB\s*(\d{5,7})\b|\bMycoBank\s+#?\s*(\d{5,7})\b'),
    "Page-ref": re.compile(r'\b(?:p\.|pp\.)\s*(\d+(?:[-–]\d+)?)'),
    "GBIF-ID":  re.compile(r'\bGBIF[:\s]+(\d{7,})\b', re.IGNORECASE),
}

class FungariumDetector:
    """Builds detection pattern dynamically from Redis fungaria registry."""

    def __init__(self, redis_client: redis.Redis):
        # Load codes from Redis key skol:fungaria (set by manage_fungaria.py).
        # Also merge data/personal_fungaria.json for local collections.
        # Sort longest-first so e.g. "DUKE" matches before "DU".
        codes = _load_codes(redis_client)
        pattern = r'\b(' + '|'.join(re.escape(c) for c in codes) + r')[\s:]+([A-Z]?\d[\w./\-]+)'
        self.re = re.compile(pattern)

    def detect(self, text: str,
               section_label: Optional[str] = None) -> List[Span]:
        # Confidence 0.9 inside Materials-examined; 0.6 elsewhere.
        confidence = 0.9 if section_label == 'Materials-examined' else 0.6
        ...

def detect_particles(text: str,
                     redis_client: Optional[redis.Redis] = None,
                     section_label: Optional[str] = None) -> List[Span]:
    """Run all detectors; section_label boosts fungarium confidence."""
    ...
```

**Context-aware invocation from `annotate_spans.py`**: when iterating over a
document, pass the YEDDA section label of the current passage to
`detect_particles()`. Spans detected inside `Materials-examined` get
`confidence=0.9`; elsewhere `confidence=0.6`. This lets downstream consumers
filter by confidence if needed.

**No new Redis key needed** — `skol:fungaria` already exists and is maintained by
the existing `rebuild_redis.py` / `manage_fungaria.py` infrastructure.

### 2d. `ingestors/spans.py`  *(new)*

Canonical data class and (de)serialization:

```python
@dataclass
class Span:
    start: int; end: int; label: str; text: str; source: str
    confidence: float = 1.0; metadata: dict = field(default_factory=dict)

def spans_to_json(spans, doc_id, source_attachment) -> str: ...
def spans_from_json(json_str: str) -> List[Span]: ...
def spans_to_bio(text: str, spans: List[Span]) -> List[Tuple[str, str]]: ...
def resolve_conflicts(spans: List[Span]) -> List[Span]:
    """When spans overlap, keep the shorter (more specific) one."""
```

### 2e. `bin/annotate_spans.py`  *(new)*

```
Flags: --experiment, --database, --doc-id, --source {all,gnfinder,regex},
       --gnfinder-url, --gnparser-url, --skip-existing, --force, --dry-run,
       --limit N, --verbosity

Algorithm per document:
  1. Fetch article.txt from CouchDB (or article.txt from golden DB if --golden-db)
  2. Parse article.txt.ann to get section-label → char-range mapping
     (so particle detector receives the section context for each passage)
  3. gnfinder → TaxonName spans (whole document)
  4. For each TaxonName, gnparser on text[name.end:name.end+80] → Author spans
  5. For each labeled section passage, particle_detector(passage, section_label)
     → DOI, MB-number, Page-ref, Fungarium-code spans
     (Fungarium-code gets confidence=0.9 inside Materials-examined, 0.6 elsewhere)
  6. resolve_conflicts(all spans)
  7. spans_to_json → write article.spans.json to annotations DB
```

### 2f. Pipeline integration in `manage_experiment.py`

- Add `"annotate_spans"` to `_PIPELINE_STEPS` after `"embed"` (index 5).
- Update `_SEQUENTIAL_COUNT` to 6.
- Add to `_build_step_commands`:
  ```python
  "annotate_spans": [sys.executable, str(_BIN_DIR / "annotate_spans.py"),
                     "--experiment", "{name}", "--skip-existing"],
  ```
- Add `databases.spans` field to `_default_experiment()` (defaults to same as
  `databases.annotations`).
- Update `docs/experiments.md` pipeline step table.

### Files (Phase 2)

| File | Change |
|---|---|
| `ingestors/gnfinder_client.py` | New |
| `ingestors/gnfinder_client_test.py` | New |
| `ingestors/gnparser_client.py` | New |
| `ingestors/gnparser_client_test.py` | New |
| `ingestors/particle_detector.py` | New |
| `ingestors/particle_detector_test.py` | New |
| `ingestors/spans.py` | New |
| `ingestors/spans_test.py` | New |
| `bin/annotate_spans.py` | New |
| `bin/manage_experiment.py` | Add `annotate_spans` step; spans DB field |
| `docs/experiments.md` | Update pipeline step table |

---

## Phase 3 — Django / Search Integration

### 3a. `django/search/views.py`

- After fetching `.ann`, also try `article.spans.json`.
- Overlay entity highlights as `<mark class="entity entity-{label}">` spans.
- Add tooltip with label + source + (for TaxonName) canonical name.

### 3b. Treatment section search pulldown

Replace the current "Description" search field label with a `react-select` pulldown
(per the project's UI convention) letting the user choose which section's embedding
to query against:

| Option | Redis key queried | Shown when |
|---|---|---|
| Description (default) | `skol:taxa:embedding:{id}` (primary) | Always |
| Distribution | `skol:taxa:embedding:{id}:distribution` | Always |
| Biology | `skol:taxa:embedding:{id}:biology` | When ≥1 treatment has biology embedding |

Treatments lacking an embedding for the selected section are excluded from results
(not ranked last — excluded entirely, since a missing embedding means the section
is absent, not just empty).

The query-side change is in `django/search/views.py`: accept a `section` parameter
(`primary` / `distribution` / `biology`), construct the corresponding Redis key
pattern, and proceed as today.

### 3c. New search facets / entity display

- Badge per recognized entity type in search result cards.
- Filter: nomenclatural annotation type (SP_NOV / COMB_NOV / NO_ANNOT).

### 3d. External cross-links from entity spans

| Label | Link target |
|---|---|
| `TaxonName` | `https://www.gbif.org/species/search?q={canonical}` |
| `MB-number` | `https://www.mycobank.org/page/Name%20details%20page/{number}` |
| `DOI` | `https://doi.org/{doi}` |

### Files (Phase 3)

| File | Change |
|---|---|
| `django/search/views.py` | Section pulldown query routing; read spans.json; overlay entity highlights |
| `django/frontend/src/` | `react-select` treatment section pulldown component |
| `django/search/templates/search/` | Entity badge CSS + tooltip markup |
| `docs/api-reference.md` | Document `section` query parameter and new endpoints |

---

## Phase 4 — Span-Enhanced Section Classifier *(deferred)*

Once Phase 2 spans populate a meaningful fraction of the corpus:

- Add span density features to `SkolClassifierV2`: count of TaxonName / Author /
  MB-number spans per passage; presence of SP_NOV annotation.
- Strong priors: MB-number present → very likely Nomenclature; dense TaxonName +
  Author spans → Nomenclature or Type-designation.
- New model config in `bin/train_classifier.py` that accepts spans as optional input.

Deferred: no code changes until Phase 2 corpus coverage is measurable.

---

## Phase 5 — Treatment Assembly (`taxon.py` + downstream)

### Current algorithm and its problem

`group_paragraphs()` in `taxon.py` is a two-state machine:
*collect Nomenclatures → collect Descriptions*. Any block with a label other than
Nomenclature or Description increments a gap counter; after 6 misses the treatment is
abandoned. With the expanded label set, Etymology / Holotype / Notes already in the
corpus cause silent treatment breakage. The new labels (Diagnosis, Distribution,
Materials-examined, Type-designation, Biology) make this worse.

### Redesigned state machine

Two states; termination on Nomenclature or Misc-exposition gap only:

```
INITIAL:     waiting for first Nomenclature
IN_TREATMENT: collecting section blocks
  ─ Nomenclature:    yield current treatment; start new one with this block
  ─ any treatment-section label (Description, Diagnosis, Etymology,
    Distribution, Materials-examined, Type-designation, Biology, Notes,
    Key, Figure-caption):  add to current treatment; reset gap counter
  ─ Misc-exposition: increment gap counter; if > MISC_GAP_LIMIT yield + reset
  ─ document boundary change: yield + reset (unchanged from current)
```

`MISC_GAP_LIMIT` replaces `LONG_GAP`; same idea but only ticks on Misc-exposition,
not on treatment-section labels. Proposed default: 4 (tunable).

**Treatment-section labels** (all keep the treatment open):
`Description`, `Diagnosis`, `Etymology`, `Distribution`,
`Materials-examined`, `Type-designation`, `Biology`,
`Notes`, `Key`, `Figure-caption`

**Treatment-terminating**: `Nomenclature` (triggers yield + new treatment)

**Treatment-gap content**: `Misc-exposition` (ticks gap counter)

### `Taxon` class changes (`taxon.py`)

Internal storage: `sections: Dict[str, List[Paragraph]]` keyed by label name.
Nomenclature blocks stored separately as before (they identify the treatment).

`as_row()` output — **flat fields** (user preference) plus backward-compat aliases:

```python
{
    # Backward compatibility (unchanged field names)
    'taxon':       str,   # concatenated Nomenclature text
    'description': str,   # concatenated Description text (None if absent)

    # New flat section fields (None if that section absent from this treatment)
    'diagnosis':         str | None,
    'etymology':         str | None,
    'distribution':      str | None,
    'materials_examined': str | None,
    'type_designation':  str | None,
    'biology':           str | None,
    'notes':             str | None,
    'key':               str | None,
    'figure_captions':   str | None,   # concatenated figure caption blocks

    # Span tracking — extended to cover all section types
    'nomenclature_spans':        List[Dict],  # unchanged
    'description_spans':         List[Dict],  # unchanged
    'diagnosis_spans':           List[Dict],
    'etymology_spans':           List[Dict],
    'distribution_spans':        List[Dict],
    'materials_examined_spans':  List[Dict],
    'type_designation_spans':    List[Dict],
    'biology_spans':             List[Dict],
    'notes_spans':               List[Dict],

    # Existing metadata fields unchanged
    'ingest': ..., 'line_number': ..., 'paragraph_number': ...,
    'pdf_page': ..., 'attachment_name': ..., 'json_annotated': ...,

    # Document ID — updated hash
    '_id': str,  # taxon_<sha256(all sections in canonical order)>
}
```

### Document ID hash update (`extract_taxa_to_couchdb.py`)

Current: `sha256(taxon + ":" + description)`

New: `sha256(taxon + ":" + description + ":" + diagnosis + ":" + etymology + ...)`
using all section texts in a fixed canonical order (empty string for absent sections).
This is a breaking change — all existing taxon records get new IDs on regeneration.
Acceptable since `extract_taxa` is a re-runnable pipeline step.

### `embed_taxa.py` — embedding strategy

**Targeted hybrid**: embed only the sections that are semantically coherent and
useful for similarity search. Not all sections warrant a vector.

| Section | Embed | Redis key suffix | Rationale |
|---|---|---|---|
| Description + Diagnosis | Yes — primary | _(none, backward compat)_ | Most morphologically informative; concatenation is coherent |
| Distribution | Yes — secondary | `:distribution` | Major search axis; short, self-contained |
| Biology | Yes — optional | `:biology` | Ecology/host data; add once corpus coverage is confirmed |
| Materials-examined | No | — | Structured specimen data; serves keyword/regex search, not sBERT |
| Etymology, Notes, Type-designation, Figure-captions, Key | No | — | Too short, heterogeneous, or structured |

**Key naming** (backward-compatible):
- Primary: `skol:taxa:embedding:{doc_id}` (existing key, same semantics as today)
- Distribution: `skol:taxa:embedding:{doc_id}:distribution`
- Biology: `skol:taxa:embedding:{doc_id}:biology`

**embed_taxa.py changes**:
- Compute primary embedding from `description + "\n\n" + diagnosis` (fall back to
  `description` alone if `diagnosis` absent)
- If `distribution` field non-empty, compute and store distribution embedding
- If `biology` field non-empty, compute and store biology embedding
- Add `--section {primary,distribution,biology,all}` flag to allow re-embedding
  a single section without reprocessing the whole corpus

**Storage**: ~2–3× current footprint; manageable with existing Redis infrastructure.

### Django search layer

`django/search/views.py` currently uses only `nomenclature_spans` and
`description_spans` for Source Context Viewer highlighting. Extend to render
all `*_spans` fields with type-specific CSS classes so each treatment section
is visually distinguishable in the context viewer.

### Files (Phase 5)

| File | Change |
|---|---|
| `taxon.py` | Redesign `group_paragraphs()` state machine; extend `Taxon` class |
| `taxon_test.py` | Add test cases: multi-section treatment, gap behavior, new labels |
| `bin/extract_taxa_to_couchdb.py` | Updated output schema; new hash function |
| `bin/embed_taxa.py` | Embed description + diagnosis |
| `django/search/views.py` | Render all section spans in context viewer |

---

## Phase 6 — Relabeling Strategy

### Problem

The existing hand-labeled training data (`skol_training`) and golden datasets use the
old 8-tag scheme. Before the 12-tag classifier can be trained or evaluated, the corpora
must be relabeled. Fully manual relabeling of ~1700+ documents is not practical.

### Approach: Three-tier relabeling

**Tier 1 — Automatic migration scripts** (zero human effort, high confidence):

| Source signal | Rule | New tag |
|---|---|---|
| JATS `sec-type` attribute | `jats_to_yedda.sec_type_to_tag()` mapping | Direct (already implemented) |
| `[@text#Holotype*]` block | Split: short (≤2 lines) → `Type-designation`; longer → `Materials-examined` | Automatic |
| Header keywords | "Distribution", "Habitat", "Range" → `Distribution`; "Diagnosis" → `Diagnosis`; "Biology", "Ecology", "Host" → `Biology` | Regex on first sentence |
| DOI / MB-number present (from particle_detector) | Boosts confidence that block is `Nomenclature` or `Type-designation` | Soft signal; flag for review |

Implement as `bin/migrate_labels.py --experiment NAME [--dry-run]`. Writes updated
`.ann` attachments; records change counts per document.

**Tier 2 — LLM-assisted pre-labeling** (handles ambiguous and novel blocks):

For blocks that Tier 1 cannot confidently relabel (remaining Misc-exposition blocks,
unlabeled plain-text sections), use the Claude API to propose a label given:
- The block text
- The preceding 2 blocks for context
- The current 12-tag taxonomy with definitions

Output: a JSONL file of `{doc_id, block_index, proposed_label, confidence, rationale}`
for human review. Not applied automatically — feeds the brat review queue.

**Tier 3 — brat for human review** (precision annotation; long-term scalable UI):

brat is used as the annotation *tool*, not as the storage format. YEDDA remains
canonical. The workflow is:

```
article.txt.ann (YEDDA)
    │
    ▼ bin/yedda_to_brat.py
article.txt + article.ann (brat standoff)
    │
    ▼ human annotates in brat
article.ann (updated brat standoff)
    │
    ▼ bin/brat_to_yedda.py
article.txt.ann (YEDDA, updated)
    │
    ▼ couchdb attachment upload
```

Converters: `bin/yedda_to_brat.py` and `bin/brat_to_yedda.py`.
The round-trip is lossless for section-level YEDDA tags (brat spans cover the same
character ranges as YEDDA inline delimiters).

### Why not migrate entirely to brat storage?

1. **Standoff requires two-file synchronization.** brat `.ann` stores character offsets
   into a separate `.txt` file. Any OCR correction or text update silently breaks the
   offset file. YEDDA's inline format is self-contained — annotation and text are the
   same attachment.

2. **The entire pipeline speaks YEDDA.** `jats_to_yedda.py`, `predict_classifier.py`,
   `extract_taxa_to_couchdb.py`, `taxon.py`, `evaluate_golden.py`, and the Django viewer
   all read/write `[@...#Tag*]`. Migrating storage would be a multi-month flag day with
   no new capability.

3. **CouchDB attachment model maps naturally to YEDDA.** One attachment `article.txt.ann`
   per document. brat standoff would require two synchronized attachments.

4. **Layer 2 spans (`.spans.json`) already use character offsets.** Sub-section entity
   annotation — the use case where brat's character-offset model excels — is handled by
   the spans layer. brat's main advantage is already covered without changing Layer 1
   storage.

5. **brat as UI, not storage, scales to all-of-biology.** The bottleneck for biology
   expansion will be annotator throughput, not storage format. Using brat as a UI layer
   over YEDDA storage gives both: brat's excellent multi-label span interface for
   annotators, and YEDDA's simple, scriptable, git-diffable format for the pipeline.

### Files (Phase 6)

| File | Change |
|---|---|
| `bin/migrate_labels.py` | New: Tier 1 automatic migration |
| `bin/migrate_labels_test.py` | New |
| `bin/yedda_to_brat.py` | New: export for brat annotation |
| `bin/brat_to_yedda.py` | New: import from brat annotation |
| `bin/llm_prelabel.py` | New: Tier 2 LLM-assisted pre-labeling (Claude API) |

---

## Implementation Order

```
Phase 1  (label set + jats_to_yedda + evaluate_golden)
    → Phase 6     (relabeling: Tier 1 automatic migration + brat converters)
    → Phase 5     (taxon.py treatment assembly: depends on new labels being live)
    → Phase 2a-d  (span utilities — can develop in parallel with Phase 1/5)
    → Phase 2e-f  (annotate_spans script + pipeline step)
    → Phase 3     (Django: requires spans in DB + Phase 5 span fields)
    → Phase 4     (span features in classifier: deferred)
```

Phase 1 and Phase 5 are tightly coupled — Phase 5's state machine depends on the
new label enum values being defined. Phase 6 Tier 1 migration runs immediately after
Phase 1 to produce a relabeled training corpus before the classifier is retrained.
Run Phase 1 and 6 sequentially within the same PR or back-to-back commits.

---

## Verification

### Phase 1
```bash
python -m pytest ingestors/yedda_tags_test.py ingestors/jats_to_yedda_test.py -v
python bin/curate_golden_dataset.py --all --no-mark -v
python bin/evaluate_golden.py --predicted-db skol_golden_ann_jats \
    --golden-db skol_golden_ann_hand --plaintext-db skol_golden
# Metrics table should show Diagnosis, Distribution, Materials-examined rows
```

### Phase 2
```bash
python -m pytest ingestors/gnfinder_client_test.py \
    ingestors/gnparser_client_test.py ingestors/particle_detector_test.py -v
python bin/annotate_spans.py --doc-id <id> --database skol_dev --dry-run -v
# Inspect article.spans.json in Fauxton; verify taxon name offsets are correct
```

### Phase 3
```bash
python manage.py runserver  # (from django/)
# Search for a taxon; verify entity highlights appear with correct labels and links
```

### Phase 5
```bash
python -m pytest taxon_test.py -v
# Spot-check: find a JATS doc with Etymology + Distribution sections in skol_dev
# Run extract_taxa_to_couchdb for that doc; confirm etymology/distribution fields populated
python bin/extract_taxa_to_couchdb.py --experiment taxpub_v1_onnx_int8 \
    --doc-id <id_with_multi_section_treatment> --verbosity 2
# Check CouchDB record has flat section fields and all *_spans populated
```

---

## Complexity Estimates

Unit: 1 point = "Use Claude Code to write an ingestor for a new website"
(HTTP fetcher, format parser, CouchDB storage, tests, pipeline integration).

| Phase | Points | Rationale |
|---|---|---|
| **Phase 1** — Expanded label set | **1** | Localized changes: enum additions, dict updates, one new CLI flag, backward-compat deprecation alias. Well-specified, low integration risk. |
| **Phase 2** — Span layer | **5** | Five new modules (gnfinder client, gnparser client, particle detector, spans, annotate_spans), two external REST APIs with retry logic, Redis-backed fungarium pattern builder, section-context-aware orchestration, pipeline integration. Highest new-code surface area. |
| **Phase 3** — Django / search | **3** | Two subsystems: Django query routing + entity highlight overlay (backend) and react-select treatment section pulldown (frontend). Patterns are familiar but frontend/backend must stay in sync. |
| **Phase 5** — Treatment assembly | **3** | State machine redesign is the hard part — subtle correctness requirements. Taxon class extension is mechanical but large. embed_taxa multi-key strategy adds Redis complexity. Three separate downstream files. |
| **Phase 6** — Relabeling | **2** | Four scripts with clear specs: Tier 1 migration (regex+rules), two lossless format converters, one LLM client. Lower integration risk than Phase 2 but more than Phase 1. |
| **Phase 4** — Span-enhanced classifier | *deferred* | No estimate until Phase 2 corpus coverage is measurable. |
| **Total** | **14** | |

Phase 2 is the scheduling risk — if it slips, Phase 3 and the annotate_spans pipeline step slip with it.
Phases 1, 5, and 6 are relatively independent and could be parallelized across iterations.
