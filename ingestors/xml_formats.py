#!/usr/bin/env python3
"""XML format identity for ingested articles — one registry.

**"Is this JATS?" had three answers.**  `fixes/backfill_jats_flags`
wrote ``is_jats = xml_fmt in ('jats', 'taxpub')``; four sites in
``bin/extract_plaintext`` and ``bin/curate_golden_dataset`` tested
``xml_format == 'jats'``, which silently excludes TaxPub; and a test
in ``pensoft_test`` asserted that a TaxPub document detects as
``'jats'``.  Every one of them was restating a fact about the format
*taxonomy* at a call site.

This module owns that taxonomy.  :data:`FORMATS` is the registry;
:func:`detect` names the format of a document header;
:func:`is_jats_family`, :func:`is_plain_jats` and :func:`is_taxpub`
answer membership.  Callers ask, they do not restate.

**Adding a format is one entry.**  BITS (JATS for books) and NISO STS
(JATS for standards) are the obvious next ones: each would be an
``XmlFormat`` with ``jats_family=True`` and its own header matcher,
inserted by specificity.  Nothing else in the codebase should need to
change, because nothing else knows the membership rule.

**Order is specificity order.**  :func:`detect` returns the first
match, so a profile must precede the format it profiles — TaxPub *is*
JATS, and a TaxPub header carries JATS markers too.  Pinned by
``test_more_specific_formats_come_first``.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

JATS = 'jats'
TAXPUB = 'taxpub'

# Only the head of the document is classified.  A marker in the body
# must not retro-classify the document, and reading further costs more
# for no signal.
_HEADER_BYTES = 2000

# Matches <tp:taxon-treatment> and sec-type="taxon-treatment" alike.
_TAXPUB_TREATMENT_MARKER = b'taxon-treatment'


def _is_taxpub_header(header: str) -> bool:
    """The TaxPub namespace, which no plain JATS document carries."""
    return 'www.plazi.org/taxpub' in header


def _is_jats_header(header: str) -> bool:
    """JATS by declaration, by its predecessor's name, or by shape.

    Three rules, preserved verbatim from
    ``PensoftIngestor._detect_xml_format``:

    * ``JATS`` is matched **case-sensitively** -- it appears in the
      formal public identifier.  Lower-casing it would newly match any
      URL containing "jats", which several TaxPub DTD locations do.
    * ``journalpublishing`` is matched case-insensitively; it names the
      pre-JATS NLM DTD.
    * Otherwise an ``<article`` root with a DTD or a namespace is taken
      as JATS.  This is a shape heuristic, not a declaration, and it is
      last for that reason.
    """
    lowered = header.lower()
    if 'JATS' in header:
        return True
    if 'journalpublishing' in lowered:
        return True
    return '<article' in header and (
        'dtd' in lowered or 'xmlns' in header
    )


@dataclass(frozen=True)
class XmlFormat:
    """One recognised XML format.

    ``jats_family`` is the membership answer callers used to restate
    for themselves; ``matches`` decides detection from the header.
    """

    name: str
    jats_family: bool
    matches: Callable[[str], bool]


# Ordered by specificity: detection returns the first match, so a
# profile of a format must precede the format it profiles.
FORMATS: Tuple[XmlFormat, ...] = (
    XmlFormat(name=TAXPUB, jats_family=True, matches=_is_taxpub_header),
    XmlFormat(name=JATS, jats_family=True, matches=_is_jats_header),
)

_BY_NAME = {fmt.name: fmt for fmt in FORMATS}


def detect(content: bytes) -> Optional[str]:
    """Name of the format ``content`` is in, or ``None``.

    ``None`` means "not a format we recognise", which is a routine
    answer -- ingest downloads XML that turns out to be a landing page
    or a publisher's own schema.
    """
    header = content[:_HEADER_BYTES].decode('utf-8', errors='ignore')
    for fmt in FORMATS:
        if fmt.matches(header):
            return fmt.name
    return None


def is_jats_family(xml_format: Optional[str]) -> bool:
    """True when ``xml_format`` names a JATS-family format.

    **This is the single encoding of that rule.**  Unknown and missing
    names are False rather than guessed at: callers read
    ``doc.get('xml_format')``, which is absent on every document
    ingested before format detection existed.
    """
    fmt = _BY_NAME.get(xml_format or '')
    return fmt is not None and fmt.jats_family


def is_plain_jats(xml_format: Optional[str]) -> bool:
    """True for JATS and nothing else — **not** TaxPub.

    Distinct from :func:`is_jats_family` on purpose.  Some callers do
    mean the narrow thing (a converter that only handles unprofiled
    JATS); they should say so by name rather than by writing
    ``== 'jats'`` and leaving the reader to guess whether TaxPub was
    considered.
    """
    return xml_format == JATS


def is_taxpub(xml_format: Optional[str]) -> bool:
    """True for TaxPub exactly."""
    return xml_format == TAXPUB


def has_taxpub_treatments(content: bytes) -> bool:
    """True when the document body carries taxon-treatment markup.

    One substring covers both spellings: Pensoft's
    ``<tp:taxon-treatment>`` elements and the JATS
    ``sec-type="taxon-treatment"`` pattern that ``d309fe9`` broadened
    detection to.

    Unlike :func:`detect`, the **whole** document is searched.
    Treatments live in the body, arbitrarily far in, so a header
    window would miss them.
    """
    return _TAXPUB_TREATMENT_MARKER in content


def is_taxpub_document(
    xml_format: Optional[str],
    content: Optional[bytes] = None,
) -> bool:
    """True when this document should carry the ``is_taxpub`` flag.

    Two sources, either sufficient: the format *declares* TaxPub, or
    the body carries treatment markup while declaring something else.

    ``content`` is optional because not every caller has fetched the
    attachment.  **Absent content is not evidence**: omitting it gives
    the answer the format name alone supports, never a ``False`` that
    reads as "checked the body and found nothing".
    """
    if is_taxpub(xml_format):
        return True
    return content is not None and has_taxpub_treatments(content)
