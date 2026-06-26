"""Complexity scoring for Treatment docs.

Higher score = richer prose worth annotating during the Phase 1
bootstrap pass.  Used by ``bin/select_for_annotation`` to pick a
sample for the Claude-API annotator.

The score is a comparative signal — there is no canonical
threshold.  Calibration happens by inspecting scored samples and
adjusting the weighted-combo coefficients in this module.  See
docs/schema_constrained_pipeline.md §10.4 deliverable (1).
"""

import re
from typing import Any, Dict


# Seed gazetteer of fungal feature names.  Deliberately small —
# Pass A induction (later phase) replaces this with a corpus-derived
# controlled vocabulary.
_FEATURE_KEYWORDS = (
    # Macro features
    'pileus', 'cap',
    'lamellae', 'gills',
    'stipe', 'stem',
    'context', 'flesh',
    'volva', 'annulus', 'veil', 'ring',
    'rhizomorphs',
    'odour', 'odor', 'taste',
    # Micro features
    'spores', 'spore',
    'basidia',
    'cystidia',
    'hyphae', 'hypha',
    'mycelia', 'mycelium',
    'conidia',
    'conidiophores',
    'phialides',
    'chlamydospores',
    'asci', 'ascus',
    'sporangia',
)

# Word-bounded so 'ring' doesn't false-match 'spring' / 'training',
# 'stem' doesn't false-match 'system', etc.
_FEATURE_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(kw) for kw in _FEATURE_KEYWORDS) + r')\b'
)

# Captures a numeric value followed by a size / temperature unit.
# Won't catch every parenthetical-range pattern (the Trichoderma
# notes field carries things like ``(4.9-)5.6-7.8(-8.8) μm`` where
# the unit is far from the leading digits), but registers at least
# one hit per such block, which is enough signal for comparative
# complexity scoring.  Calibrate by inspecting scored samples.
_MEASUREMENT_RE = re.compile(
    r'\d+(?:\.\d+)?\s*(?:mm|cm|nm|µm|μm|um|µ|μ|°c|°f)',
)

# Word counter for the (a) signal.  Matches alphanumeric runs.
# Numbers count as words — they're real signal in feature-rich
# descriptions ("3-5 cm wide" has three contentful word-like tokens).
_WORD_RE = re.compile(r'\b\w+\b')

# Weights chosen to satisfy the §10.6 test orderings with comfortable
# margins.  Comparative orderings are what matters; absolute values
# aren't meaningful.  Tune by inspection during calibration.
_WORD_WEIGHT = 1.0
_FEATURE_WEIGHT = 5.0
_MEASUREMENT_WEIGHT = 5.0


def complexity_score(treatment: Dict[str, Any]) -> float:
    """Score a Treatment doc by prose richness.

    First-cut definition (per the §10 design): weighted combination
    of (a) total prose word count across description + diagnosis,
    (b) DISTINCT feature-keyword hits from a small seed gazetteer
    (pileus, lamellae, stipe, ...), (c) measurement-pattern count
    (numbers followed by mm / cm / µm / nm / °C / °F).

    Args:
        treatment: A Treatment document with ``description`` and/or
            ``diagnosis`` string fields.  Either may be ``None``
            (CouchDB null) or absent — treated as empty.

    Returns:
        Non-negative float.  An empty / all-null treatment scores
        0.0.  Comparative semantics only.
    """
    description = treatment.get('description') or ''
    diagnosis = treatment.get('diagnosis') or ''
    text = f'{description}\n\n{diagnosis}'.strip().lower()
    if not text:
        return 0.0
    word_count = len(_WORD_RE.findall(text))
    feature_hits = len(set(_FEATURE_RE.findall(text)))
    measurement_hits = len(_MEASUREMENT_RE.findall(text))
    return float(
        _WORD_WEIGHT * word_count
        + _FEATURE_WEIGHT * feature_hits
        + _MEASUREMENT_WEIGHT * measurement_hits
    )
