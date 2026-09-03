#!/usr/bin/env python3
"""Rule-shaped feature-label canonicalization.

``docs/feature_label_canonicalization.json`` is a hand-maintained
map: one entry per drift form, added by eyeballing each bootstrap
run.  It has 50 entries against a candidate vocabulary of 1 060
labels, and it cannot keep up — 58 % of the round-5 label types are
singletons nobody has reviewed.

Some drift is **rule-shaped**, though: a family of forms that vary in
one systematic way, where the whole family can be settled once
instead of entry by entry.  This module holds those rules.  It does
not replace the map — the map records decisions that have no rule,
and it wins wherever both apply.

**Two rules, and the second is deliberately not a merge.**

``canonical_morph`` normalises the head noun of the sexual/asexual
family (``morph`` / ``stage`` / ``state`` / ``phase``, glued or
spaced) onto the established canonical.  It never touches the
modifier: ``Sexual morph`` and ``Asexual morph`` score 0.96 string
similarity, the highest in the corpus, and
``docs/feature_label_non_synonyms.md`` names them the most dangerous
pair on record.

The growth-condition rule that used to live here moved to
``canonical_annotation.split_condition`` on 2026-09-03, which names
its dimensions (``medium``, a list, and ``condition``) instead of
returning one opaque string.  The field it fed was called ``context``,
which collided with the mycological sense of the word — *pileal
context* is the flesh — and is retired.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

_MAP_PATH = (
    Path(__file__).resolve().parent.parent
    / 'docs' / 'feature_label_canonicalization.json'
)

# Bare modifier + head noun, and nothing else.  The trailing anchor is
# load-bearing: `Synasexual morph conidia` and `Anamorph Stromata` name
# structures *of* a morph, not the morph, and must not be rewritten.
_MORPH_RE = re.compile(
    r'^\s*(a?sexual)[\s_-]*(morph|stage|state|phase)\s*$',
    re.IGNORECASE,
)

# Growth-condition qualifiers.  Media are named by short all-caps
# codes (MEA, OA, PDA, PCA, SNA, CMA, DG18, V8) or spelled out as
# agar; `in culture` may carry a medium of its own.
_CONTEXT_RE = re.compile(
    r'\s+(?:'
    r'on\s+(?P<on>[A-Z][A-Z0-9]{1,5}|[a-z]+\s+agar|agar)'
    r'|in\s+(?P<in_>culture(?:\s+[A-Z][A-Z0-9]{1,5})?|vitro|situ)'
    r')\s*$'
)


def load_canonicalization(
        path: Optional[Path] = None) -> Dict[str, Tuple[str, ...]]:
    """Read the hand map, dropping its ``_comment`` / ``_note`` keys.

    **Values are paths** as of 2026-09-03, not label strings:
    ``('Pileus', 'context', 'microscopic')`` is the pileus, its context
    (flesh), viewed microscopically.  A one-step path is an ordinary
    rename.  Consumers that need only a label call
    :func:`canonicalize`, which returns the head.
    """
    with (path or _MAP_PATH).open(encoding='utf-8') as handle:
        raw = json.load(handle)
    return {
        key: tuple(value) if isinstance(value, list) else (value,)
        for key, value in raw.items() if not key.startswith('_')
    }


def canonical_morph(label: str) -> Optional[str]:
    """Canonical form for the sexual/asexual family, else ``None``.

    ``None`` means "not in this family" — the caller keeps the label
    as it stands.  The modifier is preserved exactly; only the head
    noun and its spacing are normalised.
    """
    match = _MORPH_RE.match(label)
    if not match:
        return None
    modifier = match.group(1).lower()
    return f'{modifier.capitalize()} morph'


def canonical_path(
        label: str,
        mapping: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> Tuple[str, ...]:
    """Canonical **path** for a label: the hand map first, then rules.

    The map wins because it records decisions a human made about
    specific forms; rules only see what it has no opinion on.  An
    unmapped label with no rule of its own is a one-step path, which is
    the honest statement that nothing is known about its depth.
    """
    if mapping is None:
        mapping = load_canonicalization()
    if label in mapping:
        return mapping[label]
    return (canonical_morph(label) or label,)


def canonicalize(
        label: str,
        mapping: Optional[Dict[str, Tuple[str, ...]]] = None) -> str:
    """Canonical **label** — the head of :func:`canonical_path`.

    Kept as a string on purpose: ``heaps.labels_by_treatment`` takes a
    canonicalizer that returns one label, and a vocabulary curve counts
    features, not paths.  Callers that need the depth ask for the path.
    """
    return canonical_path(label, mapping)[0]
