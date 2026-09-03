#!/usr/bin/env python3
"""Raw bootstrap annotations to canonical ones.

A bootstrap label often carries more than a feature name.
``Ascomata height`` is the feature ``Ascomata`` plus the sub-attribute
``height``, whose *value* the structured pass extracts from the span.
``Colony on MEA`` is ``Colony`` with ``medium`` = ``MEA``.  ``Gamma and
beta conidia`` is two features in one string, and its span says both
are absent.  This module takes those apart.

**Deterministic by decision, 2026-09-03.**  The tempting alternative —
telling the annotator not to do it — cannot work: the prompt carries
nine seed labels and instructs the model to *invent* names for
anything else, so it has no way to know a label is new, and its rule 3
already asks for one feature per span.  Everything here is a rule with
a guard, and every guard exists because its absence would invent
structure rather than find it.

**Order is part of the design** (see :func:`canonicalize_label`).

**Absence is a value, never a suppression.**  Knowing a structure was
looked for and not found is diagnostic, so an absence span yields the
labels *plus* ``presence='absent'`` — strictly more information than
the raw compound carried.
"""

import collections
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Container,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

# Growth media are short upper-case codes.  Spelled-out forms
# ("oatmeal agar") are normalised by the agar branch.
_MEDIUM_CODE = r'[A-Z][A-Z0-9]{1,5}'

# `on MEA`, `on OA and PCA`, `on oatmeal agar`, or a bare trailing code
# ("Colony MEA" -- the annotator elides the preposition).
_ON_MEDIA = re.compile(
    r'\s+(?:on\s+)?(?P<media>' + _MEDIUM_CODE +
    r'(?:\s*(?:,|and|or)\s*' + _MEDIUM_CODE + r')*)\s*$'
)
_ON_AGAR = re.compile(r'\s+on\s+(?P<agar>[a-z]+\s+agar|agar)\s*$', re.I)

# The condition dimension: where the observation was made, as opposed
# to on what.  `in culture` may carry a medium of its own.
_CONDITION = re.compile(
    r'\s+in\s+(?P<cond>culture|vitro|vivo)'
    r'(?:\s+(?P<media>' + _MEDIUM_CODE + r'))?\s*$',
    re.I,
)
_ON_HOST = re.compile(r'\s+(?:on|in)\s+(?P<cond>host|situ|nature)\s*$', re.I)

_ABSENT = re.compile(
    r'\b(?:not\s+(?:observed|seen|found|produced|present|detected)'
    r'|absent|none\s+(?:seen|observed|found)|lacking)\b',
    re.IGNORECASE,
)

_SPLIT = re.compile(r'\s*(?:\band\b|\bor\b|,)\s*', re.IGNORECASE)
_ELIDED = re.compile(r'^(\w+)-\s*(?:and|or)\s+(\w+)$', re.IGNORECASE)


@dataclass(frozen=True)
class CanonicalLabel:
    """A path into the attribute tree, plus the dimensions taken out of
    the label's name.

    ``path`` is the position ``build_vocab_tree`` navigates —
    ``('Ascomata', 'height')``, or ``('Peridium', 'hyphae', 'width')``
    when a branch runs deeper.  Depth 1 is still a path; a consumer
    walking paths should not need a special case at the root.

    ``media`` is a tuple because ``Colony on OA and PCA`` is one
    observation on two media.  ``transforms`` records which rules
    fired, so a derived record can be audited against the raw one.
    """

    path: Tuple[str, ...]
    media: Tuple[str, ...] = ()
    condition: Optional[str] = None
    transforms: Tuple[str, ...] = field(default=())

    @property
    def label(self) -> str:
        """The top-level feature — the only part that keys a record."""
        return self.path[0]


def _resolve(text: str, index: Mapping[str, str]) -> Optional[str]:
    """Look ``text`` up allowing the corpus's plural drift.

    ``Ascoma``/``Ascomata`` and ``Colonies``/``Colony`` both occur; a
    strict lookup would refuse pairs the vocabulary really does hold.
    """
    key = text.strip().lower()
    if not key:
        return None
    for candidate in (key, key + 's', key.rstrip('s'),
                      re.sub(r'a$', 'ae', key), re.sub(r'um$', 'a', key)):
        if candidate in index:
            return index[candidate]
    return None


def fold_case(label: str, known: Mapping[str, str]) -> str:
    """Fold onto an existing label differing only in case.

    **Never invents.**  A label with no case-variant in the vocabulary
    comes back exactly as it went in — this rule consolidates, it does
    not normalise for its own sake.
    """
    return known.get(label.lower(), label)


def split_condition(
        label: str) -> Tuple[str, Tuple[str, ...], Optional[str]]:
    """Take the growth-condition dimensions out of a label.

    Returns ``(base, media, condition)``.  Two *named* dimensions
    rather than one opaque string: ``Asci in culture MEA`` is
    ``condition='in culture'`` **and** ``medium=('MEA',)``, which the
    single ``context`` field this replaces could only store as the
    uninterpretable ``'culture MEA'``.

    The field was renamed because ``context`` is a mycological term of
    art — ``pileal context`` is the flesh, and ``schemas/pileus.json``
    already uses ``context_color`` in that sense.
    """
    base = label.strip()
    media: Tuple[str, ...] = ()
    condition: Optional[str] = None

    match = _CONDITION.search(base)
    if match:
        word = match.group('cond').lower()
        condition = f'in {word}'
        if match.group('media'):
            media = (match.group('media'),)
        return base[:match.start()].rstrip(), media, condition

    match = _ON_HOST.search(base)
    if match:
        return (base[:match.start()].rstrip(), media,
                f"{'in' if match.group('cond').lower() in ('situ',) else 'on'}"
                f" {match.group('cond').lower()}")

    match = _ON_AGAR.search(base)
    if match:
        return base[:match.start()].rstrip(), (match.group('agar'),), None

    match = _ON_MEDIA.search(base)
    if match:
        codes = tuple(
            code for code in re.split(r'\s*(?:,|and|or)\s*',
                                      match.group('media'))
            if code
        )
        return base[:match.start()].rstrip(), codes, None

    return base, media, condition


def strip_sub_attribute(
    label: str,
    established: Mapping[str, str],
) -> Tuple[str, Optional[str]]:
    """Split ``<feature> <sub-attribute>`` when the head is a feature.

    **The guard is that the head must be *established*.**  Without it,
    ``Biofilm Architecture`` strips onto ``Biofilm`` — a label as rare
    as the one being stripped — and the hierarchy is invented rather
    than found.  It also protects the hyphal-system family:
    ``Generative`` is not a feature, so ``Generative hyphae`` survives
    whole, which ``docs/feature_label_non_synonyms.md`` requires.
    """
    parts = label.split()
    if len(parts) < 2:
        return label, None
    head = ' '.join(parts[:-1])
    resolved = established.get(head.lower())
    if resolved is None or resolved == label:
        return label, None
    return resolved, parts[-1].lower()


def split_compound(
    label: str,
    known: Mapping[str, str],
) -> Optional[List[str]]:
    """Split ``X and Y`` into its features, or ``None`` to refuse.

    Three shapes: two whole labels (``Basidia and cheilocystidia``), a
    shared trailing noun (``Gamma and beta conidia``), and an elided
    prefix (``Micro- and macropycnidia``).

    **Every part must resolve to a known label.**  Refusing is the
    common case and the safe one: splitting on faith would mint labels
    instead of consolidating them.  ``Mega- and microconidia`` refuses
    because ``Megaconidia`` is not in the vocabulary.
    """
    text = label.strip()

    elided = _ELIDED.match(text)
    if elided:
        prefix, tail = elided.group(1), elided.group(2)
        for cut in range(2, len(tail)):
            noun = tail[cut:]
            first, second = (_resolve(prefix + noun, known),
                             _resolve(tail, known))
            if first and second:
                return [first, second]
        return None

    parts = [p for p in _SPLIT.split(text) if p.strip()]
    if len(parts) < 2:
        return None

    whole = [_resolve(p, known) for p in parts]
    if all(whole):
        return [w for w in whole if w]

    # Shared trailing noun: the last part carries it, the earlier ones
    # borrow it.  "Gamma and beta conidia" -> Gamma conidia, Beta conidia.
    tail_words = parts[-1].split()
    if len(tail_words) >= 2:
        noun = tail_words[-1]
        borrowed = [
            _resolve(p if len(p.split()) > 1 else f'{p} {noun}', known)
            for p in parts[:-1]
        ]
        last = _resolve(parts[-1], known)
        if last and all(borrowed):
            return [b for b in borrowed if b] + [last]
    return None


def presence_from_span(text: str) -> Optional[str]:
    """``'absent'`` when the span says the structure was not found.

    ``None`` means "nothing stated", not "present": presence is the
    default and recording it would put a redundant key on every
    annotation in the corpus.
    """
    return 'absent' if _ABSENT.search(text or '') else None


def canonicalize_label(
    label: str,
    *,
    known: Mapping[str, str],
    established: Mapping[str, str],
    protected: Container[str] = frozenset(),
) -> List[CanonicalLabel]:
    """Apply the rules in order; return one label or several.

    **The order is load-bearing.**

    0. ``protected`` first — a label that is itself a hand-map target
       is a decision a human already made, and is returned whole.
       Without this, ``Partial veil microscopic`` survives only because
       ``Partial veil`` happens to sit below the support guard, which
       is an accident rather than a design.
    1. ``fold_case`` next, so ``Colony Reverse`` becomes
       ``Colony reverse`` and can then be recognised as a
       sub-attribute; folding last would miss it.
    2. ``split_condition`` next, so ``Colony on OA and PCA`` has its
       ``and`` consumed before the compound splitter sees it —
       otherwise one observation on two media becomes two features.
    3. ``split_compound``, then
    4. ``strip_sub_attribute`` on each resulting part.
    """
    key = label.strip().lower()
    if key in protected:
        return [CanonicalLabel(path=(known.get(key, label.strip()),))]

    transforms: List[str] = []

    folded = fold_case(label, known)
    if folded != label:
        transforms.append('case_fold')

    base, media, condition = split_condition(folded)
    if media or condition:
        transforms.append('condition')

    parts = split_compound(base, known)
    if parts:
        transforms.append('compound')
    else:
        parts = [base]

    out: List[CanonicalLabel] = []
    for part in parts:
        head, sub = strip_sub_attribute(part, established)
        marks = tuple(transforms + (['sub_attribute'] if sub else []))
        out.append(CanonicalLabel(
            path=(head, sub) if sub else (head,), media=media,
            condition=condition, transforms=marks,
        ))
    return out


_PASSTHROUGH = (
    'field', 'start', 'end', 'source_text', 'source_spans',
    'treatment_id', 'doc_id', 'model', 'created_at', 'reviewed_at',
    'reviewer', 'reviewer_action', 'round', 'round_file',
    'round_provenance',
)


def canonical_records(
    annotation: Mapping[str, Any],
    *,
    known: Mapping[str, str],
    established: Mapping[str, str],
    protected: Container[str] = frozenset(),
    source_db: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """One raw annotation to one or more canonical records.

    The ``_id`` keeps the candidate DB's scheme
    (``<treatment_id>:<feature_label>:<start>``) so the derived DB can
    be joined to the raw one, and ``raw_label`` records what the
    annotator actually emitted — the traceability that justified
    deriving a database instead of mutating the candidate one.

    Keys are omitted rather than set to ``None`` when nothing applies,
    following the convention already in ``brat_ingest``.
    """
    label = str(annotation.get('feature_label') or '').strip()
    if not label:
        return []

    out: List[Dict[str, Any]] = []
    presence = presence_from_span(str(annotation.get('source_text') or ''))
    for canonical in canonicalize_label(
            label, known=known, established=established,
            protected=protected):
        record: Dict[str, Any] = {
            key: annotation[key]
            for key in _PASSTHROUGH if key in annotation
        }
        record['_id'] = (
            f"{annotation.get('treatment_id')}:{canonical.label}:"
            f"{int(annotation.get('start', 0))}"
        )
        record['feature_label'] = canonical.label
        record['attribute_path'] = list(canonical.path)
        if source_db:
            record['source_db'] = source_db
        if canonical.label != label:
            record['raw_label'] = label
        if canonical.media:
            record['medium'] = list(canonical.media)
        if canonical.condition:
            record['condition'] = canonical.condition
        if presence:
            record['presence'] = presence
        if canonical.transforms:
            record['transforms'] = list(canonical.transforms)
        out.append(record)
    return out


def vocabulary_index(
    annotations: Any,
    canonicalizer: Any = None,
    min_df: int = 1,
) -> Dict[str, str]:
    """Lower-cased label index over annotations, filtered by support.

    ``min_df=1`` gives the ``known`` index, ``min_df=5`` the
    ``established`` one that guards :func:`strip_sub_attribute`.
    Support is *treatment* frequency, not occurrence count — the same
    unit ``corpus_vocabulary`` uses, and for the same reason: forty
    repeats inside one document are one piece of evidence.
    """
    by_treatment: Dict[str, set] = collections.defaultdict(set)
    for ann in annotations:
        tid = ann.get('treatment_id')
        raw = ann.get('feature_label')
        if not tid or not raw:
            continue
        text = str(raw)
        if callable(canonicalizer):
            text = canonicalizer(text)
        elif canonicalizer is not None:
            text = canonicalizer.get(text, text)
        by_treatment[str(tid)].add(text)
    counts: 'collections.Counter[str]' = collections.Counter(
        label for labels in by_treatment.values() for label in labels
    )

    # **The frequent spelling wins.**  Building this as
    # ``{label.lower(): label for ...}`` lets the last variant iterated
    # win, which is arbitrary -- and measured on the real corpus it put
    # the *rarer* spelling in charge of 33 keys, so ``fold_case`` folded
    # 550 occurrences of `Conidiogenous cells` onto one occurrence of
    # `Conidiogenous Cells`.  Ties break lexicographically so the index
    # is reproducible run to run.
    index: Dict[str, str] = {}
    for label, count in sorted(counts.items()):
        if count < min_df:
            continue
        key = label.lower()
        incumbent = index.get(key)
        if incumbent is None or count > counts[incumbent]:
            index[key] = label
    return index
