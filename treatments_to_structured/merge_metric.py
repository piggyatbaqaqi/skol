"""Multi-species-merge detection heuristic for Treatment docs.

Empirical intuition: real single-species taxonomic treatments
concentrate technical vocabulary within specific sections (one
Pileus block, one Stipe block, one Spores block, ...), so each
anatomical term appears a handful of times.  Multi-species merged
treatments repeat the same section structure N times, so each
anatomical term gets N-fold boosted.

Metric ``n_terms_above_k``: count of distinct non-stop-word terms
in description+diagnosis that appear at least K times.  Higher
values indicate more likely-merged treatments.

Validated 2026-07-01 in
``notebooks/heaps_law_analysis.ipynb`` against the 56-treatment
sample: threshold k=5, boundary >= 10 separates cleanly (above:
avg 48.6 annotations, max 161; below: avg 9.5 annotations, max
26).  The metric also caught taxon_2b793602153da... — a real
merge that Claude only produced 19 annotations for — which pure
annotation-count filtering would have missed.

See:
  * bin/select_for_annotation --exclude-suspected-merges (consumer)
  * treatments_to_structured/status.STATUS_SKIPPED_MERGE_SUSPECT
    (persistence)
  * docs/data_quality_production_v4_model.md §6 (motivation)
"""

import re
from collections import Counter
from typing import Any, Dict, Set


# Compact English stop-word list.  Inlined to avoid the NLTK
# dependency for a tool that runs at CLI startup.  Coverage is
# sufficient for the metric's precision needs; if false positives
# emerge in production, extend here (or move to a data file).
_ENGLISH_STOP_WORDS: Set[str] = {
    'the', 'of', 'and', 'in', 'to', 'or', 'with', 'on', 'at',
    'by', 'an', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'not', 'no', 'this', 'that', 'these', 'those', 'it',
    'its', 'for', 'from', 'up', 'down', 'into', 'out', 'over',
    'under', 'after', 'before', 'then', 'than', 'so', 'such',
    'which', 'when', 'where', 'who', 'how', 'what', 'why',
    'some', 'any', 'all', 'each', 'every', 'one', 'two', 'three',
    'four', 'five', 'more', 'most', 'less', 'least', 'very',
    'much', 'many', 'few', 'also', 'both', 'either', 'neither',
    'if', 'but', 'however', 'although', 'though', 'while',
    'have', 'has', 'had', 'having', 'may', 'might', 'can',
    'could', 'would', 'should', 'shall', 'will', 'do', 'does',
    'did', 'done', 'there', 'their', 'them', 'they', 'he',
    'she', 'his', 'her', 'him',
}

# Domain-specific structural noise that would inflate the metric
# without signaling a merge (units, generic dimensional descriptors,
# cell/wall vocabulary that appears throughout ANY treatment).
_MYCOLOGY_NOISE_WORDS: Set[str] = {
    'mm', 'cm', 'um', 'nm', 'ul', 'kpa',
    'wide', 'long', 'high', 'thick', 'diam', 'diameter',
    'size', 'cell', 'cells', 'wall', 'walls',
    'base', 'apex', 'surface',
}

STOP_WORDS: Set[str] = _ENGLISH_STOP_WORDS | _MYCOLOGY_NOISE_WORDS

# Tokenizer: sequences of 3+ ASCII letters, lowercase-folded at
# match time.  3+ drops most common short words already;
# lowercase ensures 'Pileus' and 'pileus' aggregate.
_TOKEN_RE = re.compile(r'[a-zA-Z]{3,}')


def n_terms_above_k(text: str, k: int = 5) -> int:
    """Count distinct non-stop-word terms appearing at least
    ``k`` times in ``text``.

    Args:
        text: The text to analyze (description + diagnosis
            joined by whitespace).
        k: Minimum count for a term to be included.  Default 5,
            calibrated against the 2026-07-01 sample.

    Returns:
        Number of distinct terms hitting the threshold.  Higher
        indicates more repeated section structure (likely
        multi-species merge).
    """
    if not text:
        return 0
    tokens = _TOKEN_RE.findall(text.lower())
    counts = Counter(t for t in tokens if t not in STOP_WORDS)
    return sum(1 for c in counts.values() if c >= k)


def treatment_merge_metric(
    treatment: Dict[str, Any],
    k: int = 5,
) -> int:
    """Compute the merge-detection metric for a Treatment doc.

    Reads ``description`` and ``diagnosis`` fields (either may be
    None or absent), joins with a space, and calls
    ``n_terms_above_k``.  Other fields are intentionally ignored
    — merges surface most cleanly in the description; other
    fields like `key` and `figure_captions` have different
    distributional properties.
    """
    parts = []
    for field in ('description', 'diagnosis'):
        val = treatment.get(field)
        if val:
            parts.append(val)
    text = ' '.join(parts)
    return n_terms_above_k(text, k=k)


#: The merge-suspect cutoff, and the last tier of CLAUDE.md's priority
#: order -- CLI, then MERGE_THRESHOLD, then the config file, then this.
#: ``bin/env_config`` imports it as its own fallback so the number
#: exists once.
#:
#: Raised 10 -> 15 on 2026-08-26.  10 came from a 56-treatment sample
#: in 2026-07-01; measured against 30 hand verdicts it ran at 51.7 %
#: precision and wrongly excluded ~3 111 of the 7 632 treatments it
#: flagged.  15 is the F1 optimum on those cases (68.2 -> 73.3).  See
#: docs/data_quality_production_v4_model.md section 6.1.
DEFAULT_MERGE_THRESHOLD = 15


def is_suspected_merge(
    treatment: Dict[str, Any],
    threshold: int = DEFAULT_MERGE_THRESHOLD,
    k: int = 5,
) -> bool:
    """Predicate: is this Treatment doc a suspected multi-species
    merge?

    Threshold default 15 is measured; see DEFAULT_MERGE_THRESHOLD.
    The superseded 10 came from the 2026-07-01
    56-treatment sample.  Above threshold: 11 treatments with
    avg 48.6 annotations.  Below: 45 treatments with avg 9.5.
    """
    return treatment_merge_metric(treatment, k=k) >= threshold


__all__ = (
    'STOP_WORDS',
    'n_terms_above_k',
    'treatment_merge_metric',
    'is_suspected_merge',
)
