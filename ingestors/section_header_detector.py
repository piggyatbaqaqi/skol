"""Detect whole-line section headers in article plaintext.

Used by v4 (per docs/v4_classifier_plan.md §1.C) as a Step-2 feature:
the layout CRF receives a per-line ``section_header_flag`` derived
from these spans.  The detector emits ``Span(label='section-header')``
objects so downstream code can treat the result uniformly with the
gnfinder / particle_detector / page_header_detector outputs.

Vocabulary mirrors the section synonyms recognised by
``ingestors/jats_to_yedda.py:sec_type_to_tag`` plus the v4-plan hints
("Taxonomy", "Systematics", "Taxonomic treatment").  Each detected
header carries metadata::

    {
        'canonical':  '<lowercased section name as matched>',
        'yedda_hint': '<Tag.X.value the section maps to>',
    }

so consumers can group case variants by ``canonical`` and access the
intended YEDDA tag without re-parsing the text.

The detector is intentionally conservative on body-text false
positives: line-anchored (``re.MULTILINE``) and bounded by trailing
punctuation only — any word after the matcher means it's body text,
not a header.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, NamedTuple, Tuple

from ingestors.spans import Span
from ingestors.yedda_tags import Tag


class SectionEntry(NamedTuple):
    """One row of the vocabulary table.

    ``matcher``: a regex source.  By default it's the inner phrase
    that gets wrapped in the leading-numbering / trailing-punct
    template (see ``_TEMPLATED_LINE_RE`` below).  When ``is_full_regex``
    is True the entry is compiled as-is — useful for entries that
    need variable trailing content (e.g. ``A new species of <X>``).
    ``hint``: the YEDDA-tag value the detector emits as
    ``metadata['yedda_hint']`` for any line matching this entry.
    """
    matcher: str
    hint: str
    is_full_regex: bool = False


_NOMENCLATURE = Tag.NOMENCLATURE.value


# Vocabulary table.  Ordered from most-specific to least-specific so
# that alternation prefers multi-word patterns (e.g. "materials and
# methods" before "methods").
#
# Entries default to the templated path: ^numbering? body trailing? $
# where ``body`` is the matcher and trailing/numbering are added by
# the module-level regex.  Set ``is_full_regex=True`` and write the
# anchors yourself when you need custom line shape (e.g. variable
# trailing content for a wildcard phrase).
_SECTIONS: Tuple[SectionEntry, ...] = (
    # Materials-examined synonyms (specific phrases first)
    SectionEntry(r"materials?\s+examined", Tag.MATERIALS_EXAMINED.value),
    SectionEntry(r"specimens?\s+examined", Tag.MATERIALS_EXAMINED.value),
    SectionEntry(r"examined\s+material", Tag.MATERIALS_EXAMINED.value),
    # Materials-and-methods synonyms
    SectionEntry(r"materials?\s+and\s+methods?",
                 Tag.MATERIALS_AND_METHODS.value),
    SectionEntry(r"methodology", Tag.MATERIALS_AND_METHODS.value),
    SectionEntry(r"methods", Tag.MATERIALS_AND_METHODS.value),
    # Type-designation synonyms
    SectionEntry(r"type\s+material", Tag.TYPE_DESIGNATION.value),
    SectionEntry(r"type\s+designation", Tag.TYPE_DESIGNATION.value),
    SectionEntry(r"holotype", Tag.TYPE_DESIGNATION.value),
    SectionEntry(r"paratype", Tag.TYPE_DESIGNATION.value),
    SectionEntry(r"isotype", Tag.TYPE_DESIGNATION.value),
    # Bibliography synonyms
    SectionEntry(r"references\s+cited", Tag.BIBLIOGRAPHY.value),
    SectionEntry(r"literature\s+cited", Tag.BIBLIOGRAPHY.value),
    SectionEntry(r"references", Tag.BIBLIOGRAPHY.value),
    SectionEntry(r"bibliography", Tag.BIBLIOGRAPHY.value),
    # Phylogeny synonyms
    SectionEntry(r"phylogenetic\s+analys(?:i|e)s", Tag.PHYLOGENY.value),
    SectionEntry(r"phylogenetic\s+analysis", Tag.PHYLOGENY.value),
    SectionEntry(r"phylogeny", Tag.PHYLOGENY.value),
    # Key synonyms
    SectionEntry(
        r"key\s+to\s+(?:species|genera|taxa|the\s+species)",
        Tag.KEY.value,
    ),
    # New-combinations synonyms
    SectionEntry(r"new\s+combinations?", Tag.NEW_COMBINATIONS.value),
    # Taxonomic-section-start phrases (user-supplied real-world
    # headers).  All hint Nomenclature: treatments downstream of
    # these headers typically begin with a nomenclatural act.
    SectionEntry(r"nomenclator\s+and\s+taxonomic\s+description",
                 _NOMENCLATURE),
    SectionEntry(r"descriptions?\s+of\s+the\s+species", _NOMENCLATURE),
    SectionEntry(r"taxonomic\s+description", _NOMENCLATURE),
    SectionEntry(r"taxonomic\s+revision", _NOMENCLATURE),
    SectionEntry(r"taxonomic\s+treatment", _NOMENCLATURE),
    SectionEntry(r"taxonomic\s+part", _NOMENCLATURE),
    SectionEntry(r"descriptive\s+part", _NOMENCLATURE),
    SectionEntry(r"the\s+species", _NOMENCLATURE),
    SectionEntry(r"taxa\s+studied", _NOMENCLATURE),
    SectionEntry(r"species\s+recorded", _NOMENCLATURE),
    # "A new species" with optional " of <binomial>" — variable
    # trailing content forces the full-regex override.
    SectionEntry(
        r"^[ \t]*a\s+new\s+species(?:\s+of\s+\S.*)?[ \t]*$",
        _NOMENCLATURE,
        is_full_regex=True,
    ),
    # Single-word sections that map to specific YEDDA tags
    SectionEntry(r"etymology", Tag.ETYMOLOGY.value),
    SectionEntry(r"descriptions?", Tag.DESCRIPTION.value),
    SectionEntry(r"diagnos(?:is|es)", Tag.DIAGNOSIS.value),
    SectionEntry(r"biology", Tag.BIOLOGY.value),
    SectionEntry(r"ecology", Tag.BIOLOGY.value),
    SectionEntry(r"habitat", Tag.BIOLOGY.value),
    # Notes block.  Per user rule: Discussion / Remarks / Comments /
    # Notes all default to Notes; downstream CRFs disambiguate when
    # the body content is clearly Diagnosis.
    SectionEntry(r"notes", Tag.NOTES.value),
    SectionEntry(r"remarks", Tag.NOTES.value),
    SectionEntry(r"comments", Tag.NOTES.value),
    SectionEntry(r"discussion", Tag.NOTES.value),
    SectionEntry(r"distribution", Tag.DISTRIBUTION.value),
    # Article-level sections that don't map to a specific
    # treatment-content YEDDA tag — they fall through to
    # Misc-exposition (matching jats_to_yedda's behaviour).
    SectionEntry(r"introduction", Tag.MISC_EXPOSITION.value),
    SectionEntry(r"abstract", Tag.MISC_EXPOSITION.value),
    SectionEntry(r"results?", Tag.MISC_EXPOSITION.value),
    SectionEntry(r"conclusions?", Tag.MISC_EXPOSITION.value),
    SectionEntry(r"acknowledg(?:e)?ments?", Tag.MISC_EXPOSITION.value),
    SectionEntry(r"appendix|appendices", Tag.MISC_EXPOSITION.value),
    SectionEntry(r"taxonomy", Tag.MISC_EXPOSITION.value),
    SectionEntry(r"systematics", Tag.MISC_EXPOSITION.value),
)


# Optional leading numbering: up to 4 digits / Roman numerals plus a
# separator (. ) : -).  Anchored to line start.
_NUMBERING = r'^[ \t]*(?:[\divxIVX]{1,4}[.):\-\s]+\s*)?'

# Optional trailing punctuation + line end.  ``.``, ``:``, ``-``, em
# and en dashes — but nothing else, so a word after the matcher
# pushes the line back into body-text territory.
_TRAILING = r'[ \t]*[.:\-—–]*[ \t]*$'


_TEMPLATED_ENTRIES = tuple(e for e in _SECTIONS if not e.is_full_regex)
_FULL_REGEX_ENTRIES = tuple(e for e in _SECTIONS if e.is_full_regex)

_TEMPLATED_LINE_RE = re.compile(
    _NUMBERING
    + r'(' + r'|'.join(e.matcher for e in _TEMPLATED_ENTRIES) + r')'
    + _TRAILING,
    re.MULTILINE | re.IGNORECASE,
)

_FULL_REGEX_COMPILED: Tuple[Tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(e.matcher, re.MULTILINE | re.IGNORECASE), e.hint)
    for e in _FULL_REGEX_ENTRIES
)

# Per-templated-entry hint lookup: anchored to the matcher's own
# phrase so we can recover the hint from the captured group.
_TEMPLATED_HINT_LOOKUP: Tuple[Tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(r'^' + e.matcher + r'$', re.IGNORECASE), e.hint)
    for e in _TEMPLATED_ENTRIES
)


def _lookup_templated_hint(matched_phrase: str) -> str:
    """Return the YEDDA-tag hint for a captured templated phrase.

    Walks ``_TEMPLATED_HINT_LOOKUP`` in declaration order so the
    most-specific (multi-word) entries match first.
    """
    candidate = matched_phrase.strip().lower()
    for pat, hint in _TEMPLATED_HINT_LOOKUP:
        if pat.match(candidate):
            return hint
    return Tag.MISC_EXPOSITION.value


def detect_section_headers(text: str) -> List[Span]:
    """Find whole-line section headers in ``text``.

    Returns a list of :class:`Span` with ``label='section-header'``,
    ``source='regex'``, and ``metadata={'canonical', 'yedda_hint'}``.

    Two regex passes per call:

    1. The templated alternation — fast, single-pattern scan over
       every entry that uses the default leading-numbering + trailing-
       punct line shape (the vast majority).
    2. The full-regex entries — one ``finditer`` per entry, for the
       rare patterns (e.g. wildcards) that need custom line shape.

    Lines matched by either pass become section-header spans.
    """
    if not text:
        return []
    spans: List[Span] = []
    seen_ranges: List[Tuple[int, int]] = []

    # Pass 1: templated entries.
    for m in _TEMPLATED_LINE_RE.finditer(text):
        phrase = m.group(1)
        canonical = ' '.join(phrase.lower().split())
        metadata: Dict[str, Any] = {
            'canonical': canonical,
            'yedda_hint': _lookup_templated_hint(phrase),
        }
        spans.append(Span(
            start=m.start(), end=m.end(),
            label='section-header',
            text=text[m.start():m.end()],
            source='regex',
            metadata=metadata,
        ))
        seen_ranges.append((m.start(), m.end()))

    # Pass 2: full-regex entries.  Skip matches that fully overlap
    # an already-emitted templated span (defensive — keeps the per-
    # line "one span" invariant).
    for pat, hint in _FULL_REGEX_COMPILED:
        for m in pat.finditer(text):
            if any(
                seen_start <= m.start() and m.end() <= seen_end
                for seen_start, seen_end in seen_ranges
            ):
                continue
            matched_text = text[m.start():m.end()]
            canonical = ' '.join(matched_text.strip().lower().split())
            spans.append(Span(
                start=m.start(), end=m.end(),
                label='section-header',
                text=matched_text,
                source='regex',
                metadata={'canonical': canonical, 'yedda_hint': hint},
            ))

    # Final sort by start offset so callers see spans in doc order.
    spans.sort(key=lambda s: s.start)
    return spans
