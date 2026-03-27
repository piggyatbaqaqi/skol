# Latin Mycological Text Classifier

## Context

The SKOL classifier (3-class: Nomenclature, Description, Misc-exposition) handles modern (1980s–2000s) English-with-Latin articles well. The goal is to extend it to classify **entirely Latin** historical literature — starting with Fries's *Systema mycologicum* (1821–1832) and similar 19th-century works. No annotated training data for these fully-Latin texts exists yet.

Existing assets:
- 260 Diagnosis annotations (pure Latin) + 5,891 Description annotations (Latin-heavy) in current training data
- Suffix features already capture Latin morphology (-ae, -is, -um, -orum)
- *Systema mycologicum* and a few other important works available to the user
- Experiment framework for managing training runs

## Phase 0: Data Acquisition

**Goal**: Get historical Latin works into CouchDB as searchable text.

1. **Ingest from BHL/Internet Archive** — extend `ingestors/internet_archive.py` to accept a list of known item identifiers (the `_ingest_item` method already handles single items).
   - New script: `bin/ingest_historical.py` — ingests by IA identifier into `skol_latin_corpus`
   - Priority texts: *Systema mycologicum* (3 vols), *Elenchus Fungorum*, Persoon's *Synopsis Methodica Fungorum*

2. **OCR quality assessment** — 19th-century typefaces (blackletter/Fraktur) produce poor OCR.
   - New script: `bin/assess_ocr_quality.py` — character statistics, common error patterns (ſ→s, rn→m), correction mappings
   - BHL may already have corrected OCR for some texts

3. **Plaintext extraction** — may need `ingestors/djvu_xml.py` for djvu.xml format if existing extractors don't apply.

**New CouchDB databases**: `skol_latin_corpus`, `skol_latin_ann_llm`, `skol_latin_ann_hand`, `skol_latin_golden`

## Phase 1: LLM Annotation Bootstrap

**Goal**: Generate initial YEDDA annotations using an LLM, with English translations for hand-checking.

New script: `bin/annotate_with_llm.py`

**Two-pass approach per species entry**:
- **Pass 1**: LLM segments text into blocks and assigns labels (Nomenclature, Description, Misc-exposition). Few-shot examples drawn from the existing 260 Diagnosis annotations.
- **Pass 2**: LLM provides English translation of each labeled block.

**Output**: Two CouchDB attachments per document:
- `.txt.ann` — YEDDA annotation (stored in `skol_latin_ann_llm`)
- `.txt.ann.en` — English translation keyed to each block

**Entry boundary detection**: Use nomenclatural patterns (numbered genus + species + author abbreviation). Extend `NOMENCLATURE_RE` in `preprocessing.py` for older citation formats (no year numbers, abbreviations like "Fr.", "Pers.", "Bull.").

**Chunking**: Process one species entry at a time. *Systema mycologicum* entries follow extremely regular structure:
```
1234. AGARICUS campestris.     → Nomenclature
Pileo carnoso convexo...       → Description
Hab. in pratis...              → Description
Obs. Species notissima...      → Misc-exposition
```

## Phase 2: Hand Correction and Golden Dataset

**Goal**: Produce verified annotations for evaluation and training.

1. **Select ~50–100 species entries** from *Systema mycologicum* Vol. 1, spanning different genera and structural complexity.
2. **Correct LLM annotations** using a text editor, with `.txt.ann.en` translations as reference.
3. **Curate golden dataset** — new script `bin/curate_latin_golden.py` (follows pattern of `bin/curate_golden_dataset.py`), populating `skol_latin_golden` and `skol_latin_ann_hand`.
4. **Measure inter-annotator agreement** if multiple annotators participate — use existing `bin/evaluate_golden.py`.

## Phase 3: Model Training

**Goal**: Train a combined classifier that handles both modern English and historical Latin.

**Strategy: domain adaptation through combined training, not a separate model.**

### Feature engineering adjustments

| Change | File | Detail |
|--------|------|--------|
| Expand suffix vocab | `skol_classifier/feature_extraction.py` | 400 → 600–800 to capture richer Latin case endings |
| Expand word vocab | same | 3600 → 5000 for Latin vocabulary |
| Language indicator feature | same, new pipeline stage | Binary feature: line predominantly Latin vs English (heuristic: absence of English function words + high frequency of Latin suffixes) |
| Latin nomenclature patterns | `skol_classifier/preprocessing.py` | Extend `NOMENCLATURE_RE` for pre-1900 citation formats |

### Optional: 4-class scheme

Consider preserving Diagnosis as a separate class instead of collapsing it to Description. In Latin texts, Diagnosis (comparative: "differt a...") is structurally distinct from Description (purely morphological).

- Modify `collapse_labels()` in `preprocessing.py` to accept a `collapse_mode` parameter: `"3-class"` (current default) vs `"4-class"`
- Add Diagnosis to `Tag` enum in `ingestors/yedda_tags.py` if not already present

### Training data

- Merge `skol_training` (190 modern articles) + `skol_latin_ann_hand` (50–100 Latin entries)
- Use `extraction_mode='line'` for Latin texts (no section headers in historical works)
- Rebalance class weights — Latin texts are almost entirely Nomenclature + Description with very little Misc-exposition

### Experiment config

Register as `bin/manage_experiment.py create --name latin_v1` with new `MODEL_CONFIGS` entry in `bin/train_classifier.py`.

## Phase 4: Evaluation

1. **Latin-specific evaluation**: `bin/evaluate_golden.py --golden-db skol_latin_ann_hand`
2. **Cross-domain regression check**: Run combined model on existing English golden dataset to ensure no degradation
3. **Annotation efficiency metric**: Compare LLM pre-annotation + hand-correction time vs annotation from scratch

## Phase 5: Integration

No new code needed if the experiment framework is in place:
- Latin model appears as a selectable experiment in Django UI
- `bin/predict_classifier.py --experiment latin_v1` classifies Latin works
- Taxa from *Systema mycologicum* become searchable in the existing search infrastructure

## Sequencing

```
Phase 0 (Data Acquisition)
    ↓
Phase 1 (LLM Annotation) → Phase 2 (Hand Correction + Golden)
                                  ↓
                            Phase 3 (Model Training)
                                  ↓
                            Phase 4 (Evaluation)
                                  ↓
                            Phase 5 (Integration)
```

Critical path: Phases 0–1. Phase 3 requires ≥50 hand-corrected entries from Phase 2.

## Key Risks

| Risk | Mitigation |
|------|------------|
| Poor OCR on 19th-century typefaces | Phase 0 assessment; BHL corrected OCR; manual transcription for key works |
| LLM annotation quality | Two-pass with translation; start with structurally regular entries; measure vs hand annotations |
| Combined model degrades English performance | Evaluate on English golden before/after; class weights; ensemble fallback |
| Insufficient Latin training data | *Systema mycologicum* has 5000+ entries; 100 annotated may suffice given structural regularity |

## Key Files

| File | Role |
|------|------|
| `skol_classifier/preprocessing.py` | `SuffixTransformer`, `NOMENCLATURE_RE`, `collapse_labels()` — need Latin extensions |
| `skol_classifier/feature_extraction.py` | Feature pipeline — language indicator, expanded vocab |
| `ingestors/internet_archive.py` | Base for historical text acquisition |
| `ingestors/yedda_tags.py` | Tag enum — Diagnosis tag if 4-class scheme |
| `bin/train_classifier.py` | Model config table — Latin experiment configs |
