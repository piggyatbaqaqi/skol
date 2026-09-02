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

``split_medium_context`` decomposes ``Colony on MEA`` into
``('Colony', 'MEA')``.  It exists because the obvious rule — strip
the qualifier, merge into the base label — is exactly what the
non-synonyms doc forbids: *"the medium is the entire point of the
observation … never collapse this family"*.  The same doc names the
fix, which is what this function implements: *"a separate `context`
field, not a longer label"*.  The writers call it to populate that
field; ``canonicalize`` does **not**, because it returns a label and
a label that dropped its medium would have lost the observation.
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
        path: Optional[Path] = None) -> Dict[str, str]:
    """Read the hand map, dropping its ``_comment`` / ``_note`` keys."""
    with (path or _MAP_PATH).open(encoding='utf-8') as handle:
        raw = json.load(handle)
    return {k: v for k, v in raw.items() if not k.startswith('_')}


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


def split_medium_context(label: str) -> Tuple[str, Optional[str]]:
    """Split a growth-condition qualifier off a label.

    Returns ``(base_label, context)`` with ``context`` ``None`` when
    the label carries no condition.  **The context is returned, never
    discarded** — two media yield two different pairs, so this cannot
    collapse the family the non-synonyms doc protects.
    """
    match = _CONTEXT_RE.search(label)
    if not match:
        return label, None
    context = match.group('on') or match.group('in_')
    # `in vitro` / `in situ` are fixed Latin phrases; the bare second
    # word is not a term anyone uses, and a stored context of `vitro`
    # reads as a parsing bug.  Media keep just their code.
    if context in ('vitro', 'situ'):
        context = f'in {context}'
    return label[:match.start()].rstrip(), context


def canonicalize(
        label: str,
        mapping: Optional[Dict[str, str]] = None) -> str:
    """Canonical label: the hand map first, then the rules.

    The map wins because it records decisions a human made about
    specific forms.  Rules only see what the map has no opinion on.
    ``split_medium_context`` is deliberately not applied here.
    """
    if mapping is None:
        mapping = load_canonicalization()
    if label in mapping:
        return mapping[label]
    return canonical_morph(label) or label
