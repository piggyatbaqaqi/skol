# Clade-agnostic detector watchlists

*Planning doc.  Started 2026-07-05 while designing M2 Group B
(structural-anatomy-doubling detector).  Not scheduled — captured
so the fungi-specific vocabulary we're shipping in M2/M2 doesn't
become architectural debt.*

## Motivation

M2 Group A's `count_repeated_section_headers` and M2 Group B's
`count_repeated_structural_anatomy` both ship with hand-curated
watchlists that are **fungi-specific**:

  * Group A: Observations, Illustration, Etymology, Habitat,
    Type, Holotype (section-header keywords).
  * Group B: Basidiocarp, Ascomata, Perithecia, Apothecia,
    Sporocarp, Aethalia, Conidiomata, Thallus (top-level
    fruiting-body terms).

The **fungi-agnostic design goal** for skol is violated locally
by both.  A plant-taxonomic corpus would need `Leaf`, `Stem`,
`Root`, `Petal`, `Sepal`; an insect corpus would need `Thorax`,
`Abdomen`, `Elytra`.  Nothing in the current detector code
would let us switch corpora without hand-editing the
watchlists.

## Approaches — from cheapest to most powerful

### 1. Bootstrap from `features_hand` labels (recommended default)

**Idea**: the reviewer-verified feature-label DB IS the
structural-anatomy watchlist.  Aggregate label frequency across
`features_hand` docs; the top-N labels by occurrence are the
watchlist for this corpus's clade.

For our current fungi corpus, this would auto-derive:
Pileus, Stipe, Ascomata, Basidiocarps, Conidia, Ascospores,
Basidia, Paraphyses, etc.

For a plant corpus with the same tooling and reviewer-labeled
golden data, it would auto-derive Leaf, Stem, Root, etc.
**Zero clade-specific code**.

**Advantages**:
  * Composes with Track A's hand-review flow — every reviewed
    treatment sharpens the derived watchlist.
  * Ranks by empirical relevance, not editor guess.
  * Handles synonyms and canonical forms via the existing
    `docs/feature_label_canonicalization.json` drift map.

**Disadvantages**:
  * Cold-start problem: needs enough reviewed treatments to
    rank labels reliably.  Heaps' Law dependency (see the v5
    execution plan §Continuous Track A).
  * The label set is species-anatomy focused; won't include
    section-header keywords like Observations or Etymology.
    Complementary but not a full replacement for Group A's
    watchlist.

**Implementation shape**:
  ```python
  def load_structural_anatomy_watchlist(
      features_hand_db, min_treatments: int = 5,
  ) -> Set[str]:
      """Aggregate feature_label counts by number of treatments
      the label appears in; return labels above the threshold."""
      ...
  ```

  Cached on first use, invalidated when reviewer count crosses
  a re-derivation threshold.

### 2. Corpus-driven paragraph-start noun extraction

**Idea**: split every treatment's description into paragraphs;
count first-word frequency across the corpus; the top-N
noun-shaped tokens are candidates.

**Advantages**:
  * Works with zero golden data — corpus-only.
  * Fully unsupervised.

**Disadvantages**:
  * Noisier than #1 — surfaces prose words alongside anatomy.
  * Requires a noun-shape filter (POS tag, or a morphology
    heuristic like "starts capitalized, ends in typical Latin
    or English noun endings").
  * No filtering for anatomy-specific meaning.

**Fallback for cold-start**: use #2 when there isn't enough
golden data for #1 yet.

### 3. Segment classifier output (M3+ payoff)

**Idea**: once M3's segment classifier exists, its per-paragraph
section labels give us the ranking signal.  Labels the classifier
assigns to "Description" or "Diagnosis" paragraphs, aggregated
across the corpus, define the structural anatomy vocabulary.

This is #1 but at the pipeline-input level (classifier output)
rather than the reviewer-output level (features_hand).

**Advantages**:
  * Runs on the full 45k corpus, not just the ~250 reviewed
    treatments.
  * Composes with #1 — use both, take the union.

**Disadvantages**:
  * Depends on M3 landing.
  * Segment classifier quality bounds detector quality.

### 4. Ontology lookup

**Idea**: pull term lists from external ontologies (Plant Ontology,
Mammal Anatomy Ontology, Hymenoptera Anatomy Ontology, etc.).

**Advantages**:
  * Rich, curated, community-maintained term sets.
  * Cross-clade coverage available today.

**Disadvantages**:
  * Requires ontology download + maintenance per clade.
  * Ignores what the corpus actually contains — may include
    terms never used in our literature.
  * Version-pinning becomes an ops concern (see LOTL plan for
    similar concern with fasttext model versions).

Deferred unless #1 + #3 prove insufficient.

### 5. TF-IDF against a background corpus

**Idea**: words that are common in taxonomic treatments but
uncommon in general prose (e.g., Wikipedia's non-taxonomic
articles) are candidates.

**Advantages**:
  * Language-agnostic (unlike ontologies).
  * Runs on corpus alone.

**Disadvantages**:
  * Surfaces both anatomy AND taxonomic conventions
    ("gen. nov.", "sp. nov.", "type", "syntype").
  * Noisy — needs post-filtering.

Interesting for exploratory analysis but not a primary
detector-watchlist source.

### 6. LLM prompt-based inference

**Idea**: give Claude ~10 treatment excerpts and ask for the
top-level anatomical structures.

**Advantages**:
  * Works on any clade, any corpus, zero setup.

**Disadvantages**:
  * Adds inference cost.
  * Stochastic (different results across runs).
  * Not naturally part of the detector data flow.

Useful for one-off corpus exploration; not for production
detector configuration.

## Recommendation

**Ship the hand-curated fungi watchlists in M2** as an
acceptable stopgap — they are the fastest path to closing
documented pathology cases, and the fixture-based
regression tests prevent them from becoming silent
technical debt.

**Migrate to approach #1 (features_hand aggregation) when**
either happens first:

  * Track A's reviewed-treatment count reaches ~150 —
    enough to derive a stable watchlist by frequency.
  * We start ingesting a non-fungal corpus and need the
    detectors to still work.

**Add approach #3 (M3 segment classifier) as a
supplement** when M3 lands.  The union of #1 and #3
gives the strongest signal.

**Approaches #4-#6 remain optional** — reach for them if
specific clades need coverage before Track A reaches
critical mass.

## Interaction with related plans

  * [`production-v5-execution.md`](production-v5-execution.md)
    — Track A (hand-review) is the pacing constraint on
    approach #1.  M3 segment classifier enables approach #3.
  * [`lotl-detector.md`](lotl-detector.md) — parallel
    fungi-agnostic-goal violation: hand-curated Latin
    vocabulary.  Both this doc and LOTL point at the same
    underlying issue: hand-curation is fast to ship but
    scales poorly across clades.

## Cross-references

  * [`treatments_to_structured/triage_signals.py`](../../treatments_to_structured/triage_signals.py)
    — `_SECTION_HEADER_WATCHLIST` (Group A) and
    `_STRUCTURAL_ANATOMY_WATCHLIST` (Group B, this commit).
  * [`docs/feature_label_canonicalization.json`](../feature_label_canonicalization.json)
    — drift map for the labels that approach #1 would aggregate.

## Change log

  * **2026-07-05** — initial draft during M2 Group B design.
    Recorded the six approaches, recommended #1 as the
    default with M3-driven #3 as the compound signal.
    Deferred until Track A reaches ~150 reviewed treatments
    OR a non-fungal corpus lands.
