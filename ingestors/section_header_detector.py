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
from typing import Any, Dict, List, Tuple

from ingestors.spans import Span
from ingestors.yedda_tags import Tag


# Vocabulary table: (regex-source matcher, YEDDA-tag value).  Ordered
# from most-specific to least-specific so that alternation prefers
# multi-word patterns (e.g. "materials and methods" before "methods").
_SECTIONS: Tuple[Tuple[str, str], ...] = (
    # Materials-examined synonyms (specific phrases first)
    (r"materials?\s+examined", Tag.MATERIALS_EXAMINED.value),
    (r"specimens?\s+examined", Tag.MATERIALS_EXAMINED.value),
    (r"examined\s+material", Tag.MATERIALS_EXAMINED.value),
    # Materials-and-methods synonyms
    (r"materials?\s+and\s+methods?", Tag.MATERIALS_AND_METHODS.value),
    (r"methodology", Tag.MATERIALS_AND_METHODS.value),
    (r"methods", Tag.MATERIALS_AND_METHODS.value),
    # Type-designation synonyms
    (r"type\s+material", Tag.TYPE_DESIGNATION.value),
    (r"type\s+designation", Tag.TYPE_DESIGNATION.value),
    (r"holotype", Tag.TYPE_DESIGNATION.value),
    (r"paratype", Tag.TYPE_DESIGNATION.value),
    (r"isotype", Tag.TYPE_DESIGNATION.value),
    # Bibliography synonyms
    (r"references\s+cited", Tag.BIBLIOGRAPHY.value),
    (r"literature\s+cited", Tag.BIBLIOGRAPHY.value),
    (r"references", Tag.BIBLIOGRAPHY.value),
    (r"bibliography", Tag.BIBLIOGRAPHY.value),
    # Phylogeny synonyms
    (r"phylogenetic\s+analys(?:i|e)s", Tag.PHYLOGENY.value),
    (r"phylogenetic\s+analysis", Tag.PHYLOGENY.value),
    (r"phylogeny", Tag.PHYLOGENY.value),
    # Key synonyms
    (r"key\s+to\s+(?:species|genera|taxa|the\s+species)",
     Tag.KEY.value),
    # New-combinations synonyms
    (r"new\s+combinations?", Tag.NEW_COMBINATIONS.value),
    (r"taxonomic\s+treatment", Tag.MISC_EXPOSITION.value),
    # Single-word sections that map to specific YEDDA tags
    (r"etymology", Tag.ETYMOLOGY.value),
    (r"descriptions?", Tag.DESCRIPTION.value),
    (r"diagnos(?:is|es)", Tag.DIAGNOSIS.value),
    (r"biology", Tag.BIOLOGY.value),
    (r"ecology", Tag.BIOLOGY.value),
    (r"habitat", Tag.BIOLOGY.value),
    (r"notes", Tag.NOTES.value),
    (r"remarks", Tag.NOTES.value),
    (r"comments", Tag.NOTES.value),
    (r"distribution", Tag.DISTRIBUTION.value),
    # Article-level sections that don't map to a specific
    # treatment-content YEDDA tag — they fall through to
    # Misc-exposition (matching jats_to_yedda's behaviour).
    (r"introduction", Tag.MISC_EXPOSITION.value),
    (r"abstract", Tag.MISC_EXPOSITION.value),
    (r"discussion", Tag.MISC_EXPOSITION.value),
    (r"results?", Tag.MISC_EXPOSITION.value),
    (r"conclusions?", Tag.MISC_EXPOSITION.value),
    (r"acknowledg(?:e)?ments?", Tag.MISC_EXPOSITION.value),
    (r"appendix|appendices", Tag.MISC_EXPOSITION.value),
    (r"taxonomy", Tag.MISC_EXPOSITION.value),
    (r"systematics", Tag.MISC_EXPOSITION.value),
)


# Optional leading numbering: up to 4 digits / Roman numerals plus a
# separator (. ) : -).  Anchored to line start.
_NUMBERING = r'^[ \t]*(?:[\divxIVX]{1,4}[.):\-\s]+\s*)?'

# Optional trailing punctuation + line end.  ``.``, ``:``, ``-``, em
# and en dashes — but nothing else, so a word after the matcher
# pushes the line back into body-text territory.
_TRAILING = r'[ \t]*[.:\-—–]*[ \t]*$'

_ALTERNATION = r'|'.join(m for m, _ in _SECTIONS)

_LINE_RE = re.compile(
    _NUMBERING + r'(' + _ALTERNATION + r')' + _TRAILING,
    re.MULTILINE | re.IGNORECASE,
)


def _build_hint_lookup() -> Dict[str, str]:
    """Build a {compiled matcher pattern: yedda_hint} lookup that
    ``detect_section_headers`` uses to recover the canonical tag
    after a successful overall match.
    """
    return {
        re.compile(m, re.IGNORECASE).pattern: hint
        for m, hint in _SECTIONS
    }


_HINT_BY_PATTERN = _build_hint_lookup()
_COMPILED_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(r'^' + m + r'$', re.IGNORECASE), hint)
    for m, hint in _SECTIONS
)


def _lookup_yedda_hint(matched_phrase: str) -> str:
    """Return the YEDDA-tag hint for the captured phrase.

    Walks the _SECTIONS table in declaration order so the
    most-specific (multi-word) entries match first — same behaviour
    the main regex already gives us, but applied to the captured
    substring rather than the original line.
    """
    candidate = matched_phrase.strip().lower()
    for pat, hint in _COMPILED_PATTERNS:
        if pat.match(candidate):
            return hint
    return Tag.MISC_EXPOSITION.value


def detect_section_headers(text: str) -> List[Span]:
    """Find whole-line section headers in ``text``.

    Returns a list of :class:`Span` with ``label='section-header'``,
    ``source='regex'``, and ``metadata={'canonical', 'yedda_hint'}``.

    Match rules:

    1. Optional leading numbering (digit or Roman numeral, up to 4
       characters, followed by ``.``/``)``/``:``/``-``/whitespace).
    2. One of the vocabulary phrases (case-insensitive).
    3. Optional trailing punctuation (``.``/``:``/``-``/em-dash/
       en-dash) and end-of-line.

    Lines that contain additional words after the matcher don't
    match — they're body text, not headers.
    """
    if not text:
        return []
    spans: List[Span] = []
    for m in _LINE_RE.finditer(text):
        phrase = m.group(1)
        canonical = ' '.join(phrase.lower().split())
        metadata: Dict[str, Any] = {
            'canonical': canonical,
            'yedda_hint': _lookup_yedda_hint(phrase),
        }
        spans.append(Span(
            start=m.start(),
            end=m.end(),
            label='section-header',
            text=text[m.start():m.end()],
            source='regex',
            metadata=metadata,
        ))
    return spans
