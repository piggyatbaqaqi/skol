# LOTL detector — replace `_latin_ratio` with a real language detector

*Trello: https://trello.com/c/1rNUBhdR/395-lotl-detector*
*Created 2026-07-05.  Planning doc only — not scheduled.*

## Motivation

`treatments_to_structured.triage_signals._latin_ratio` is a
hand-curated heuristic:

  * ~40-entry `_LATIN_VOCAB` (ascomycete-biased)
  * Latin suffix list (`-us`, `-a`, `-um`, `-orum`,
    `-arum`, `-ibus`, `-atis`, `-osis`, `-alis`, `-ata`,
    `-atum`, `-ans`, `-ens`, `-ceae`)
  * Fraction-of-tokens over 3-char words

Known limitations:

1. **Vocabulary bias**: taxon_67cc93d2 (slime mold) does
   not fire even though the operator identifies a Latin
   diagnosis block.  Slime mold anatomical Latin
   (`capillitium`, `columella`, `peridiolum`,
   `aethalium`, …) isn't in `_LATIN_VOCAB`.  Any clade
   we haven't hand-covered has the same risk.
2. **Suffix false positives**: many English words end
   in `-a` (data, formula), `-um` (drum, alum),
   `-us` (bus, plus), `-is` (this, his).  The current
   `len(tok) > len(suf) + 1` gate is brittle.
3. **Latin-only framing**: our corpus is
   fungi-taxonomic but there are significant German and
   Czech mycological literatures the operator
   references.  Long-term SKOL will need to handle
   Latin-Other-Than-English (LOTE) content beyond the
   ICBN-Latin case.
4. **Not fungi-agnostic**: SKOL's stated design goal is
   fungi-agnostic — currently violated by the
   ascomycete-biased vocabulary.  Enriching for slime
   molds (2026-07-05 stopgap) makes this worse before
   we make it better.

Migrating to a trained language identification model
addresses all four cleanly.

## Scope

Replace the current suffix + vocab heuristic with a
real language ID library, then generalize the
"Latin vs English" detectors to
"Latin vs LOTL" (Language Other Than Latin).

**In scope**:

  * Evaluate candidate libraries (fasttext, cld3,
    langdetect) against our real corpus.
  * Wrap chosen library behind a stable interface.
  * Replace `_latin_ratio` with the wrapper.
  * Generalize `latin_block_count` and
    `latin_between_english` to consume language
    classifications rather than a Latin-only boolean —
    a Latin paragraph sandwiched by non-Latin
    (regardless of which specific non-Latin language)
    still fires the merge signal.
  * Deploy: deb-package updates, model file
    distribution, sync with tsqali + puchpuchobs +
    synoptickeyof.life.
  * Regression tests against known Latin / English /
    other cases.

**Out of scope**:

  * Multi-language search-product features (searching
    German content in an English UI, etc.) — orthogonal.
  * Section classifier retraining to consume language
    features — that's part of M3/M4 in the v5 plan.

## Point estimate

Total: **~20 hours (16-30 hours range)** — 2-3 days of
focused work.

| Phase | Hours | Description |
|-------|-------|-------------|
| Evaluation | 4-8 | Assemble labeled corpus (known Latin blocks from taxon_572d470e, taxon_9ecad903, taxon_67cc93d2 + English from §0.5 poster-children + German/Czech samples if available); run fasttext + cld3 + langdetect; measure precision/recall/F1 by class + model size + load time + inference latency; pick winner. |
| Integration | 6-10 | Replace `_latin_ratio` with library wrapper.  Generalize `latin_block_count` + `latin_between_english` to LOTL framing.  Update tests — model loading changes fixture cost from ~0.06s to ~1s+. |
| Packaging + deploy | 4-8 | Update deb package per CLAUDE.md wheel/deb sync rule.  Model file distribution decision (check-in vs download-on-install).  Deploy to tsqali + puchpuchobs; verify; deploy to synoptickeyof.life. |
| Regression + validation | 2-4 | Rerun triage CSV.  Compare flag counts before/after.  Verify §0.5 poster-children don't regress.  Verify taxon_67cc93d2 now fires. |

Uncertainty pushing past 30 hours:

  * **Latin classification quality on OCR-noisy content**
    turns out low → may need fine-tuning or a custom
    classifier trained on our corpus.
  * **Model file distribution** — check-in adds ~1 MB to
    the repo (quantized fasttext) or ~126 MB (full).
    Download-on-install introduces network dependency at
    deploy time.
  * **Native code dependencies** — fasttext and cld3
    are C++ with Python bindings; wheel/deb sync gets
    fiddlier.  langdetect is pure-Python, simpler
    packaging.

## Candidate library comparison

| Library | License | Latin? | Model size | Native? | Notes |
|---------|---------|--------|-----------|---------|-------|
| **fasttext** (`lid.176`) | MIT | Yes | 1 MB (quantized) / 126 MB (full) | C++ | 176 langs.  Facebook Research.  Well-tested. |
| **cld3** (Google) | Apache 2.0 | Yes | ~500 KB | C++ | Neural n-gram.  Similar accuracy to fasttext on short text. |
| **langdetect** | Apache 2.0 | Yes | bundled | Pure Python | Slower.  Fewer langs but Latin included.  Simpler packaging. |
| **langid.py** | BSD | Yes | bundled | Pure Python | ~97 langs.  Reasonable accuracy. |

Empirical decision on real corpus.  Bias toward whatever
scores best on the OCR-noisy Latin cases we currently
miss, weighted by packaging simplicity if scores are
close.

## Concerns to weigh before starting

  * **Feature scope creep**: this is a full week's
    detour from M1's post-hoc-detector focus.  Only
    worth doing if the current heuristic's blind spots
    accumulate materially, OR if M3's segment classifier
    would meaningfully benefit from consuming
    language-ID as an input feature.
  * **Reproducibility across environments**: pinned
    model version becomes a versioned artifact.  Rerun
    the triage CSV 6 months from now with a different
    model release → potentially different classification.
    Solvable, but new class of concern.
  * **Marginal current benefit is small**: heuristic
    catches taxon_9ecad903 (Group B verified) and
    multi-Latin cases via `latin_block_count`.  Known
    miss is one treatment: taxon_67cc93d2 (slime mold).
    Unknown: how many more silent misses are lurking.
    The 2026-07-05 vocab-enrichment stopgap likely
    closes taxon_67cc93d2 without library migration.

## Stopgaps in place

* **2026-07-05 slime mold vocab enrichment** — added
  slime-mold-relevant Latin terms to `_LATIN_VOCAB`.
  Does NOT fully close taxon_67cc93d2, because that
  treatment's Latin content sits in a bilingual
  paragraph (Latin diagnosis followed by English
  description with no blank-line separator — the
  paragraph is roughly 50% Latin, 50% English, scoring
  0.259 vs the 0.35 threshold).  Adding vocab can
  raise the ratio slightly but not past the paragraph-
  boundary structural issue.  Enrichment still
  benefits other slime-mold and general-mycological
  treatments where the Latin block does live in its
  own paragraph — worth doing as a general robustness
  step, just not sufficient to close 67cc93d2.

## Additional structural gap (identified 2026-07-05)

**Paragraph-boundary limitation of `_latin_ratio`**:
`latin_block_count` and `latin_between_english` both
score at the paragraph level (delimited by
`\n\s*\n`).  When Latin and English content share a
paragraph — as in taxon_67cc93d2, where the Latin
diagnosis runs directly into the English description
without a blank line — the detector sees one
mixed-language paragraph rather than a Latin block.

Structural fixes to consider alongside the language-
detector migration:

  * **Sentence-level scoring** — split into sentences,
    score each, look for Latin sentences among
    English.  More granular; handles the
    intra-paragraph case cleanly.
  * **Sliding window** — score fixed-size N-character
    windows.  Simple; robust to punctuation-heavy or
    unpunctuated text.
  * **Both, cascaded** — start at paragraph level, if
    a paragraph scores ambiguously (say 0.20–0.45),
    fall back to sentence-level scoring inside it.

Whichever library wins the comparison should be
plumbed at a granularity finer than the paragraph
for this reason.

## When to schedule

Not before M1 completes.  Candidates:

  * **After M2** — if M2 review passes surface
    additional Latin-heuristic misses beyond
    taxon_67cc93d2, the case strengthens.  Do it
    before M3 so segment classifier can consume
    language features.
  * **After M3** — if M3's segment classifier bootstraps
    fine without language ID, defer until Track A's
    hand-review accumulates German / Czech content
    that justifies LOTE support beyond the
    "make the taxonomic Latin detector more robust"
    scope.
  * **After a segment-classifier retrain shows language
    ID is a valuable input feature** — most durable
    trigger; ties the investment to a measured M4/M5
    quality gain.

## Cross-references

  * [docs/plans/production-v5-execution.md](production-v5-execution.md) — the plan this defers from
  * [docs/data_quality_production_v4_model.md](../data_quality_production_v4_model.md) — evidence catalogue (§6 idea #1, taxon_67cc93d2)
  * [treatments_to_structured/triage_signals.py](../../treatments_to_structured/triage_signals.py) — `_latin_ratio`, `latin_block_count`, `latin_between_english`
