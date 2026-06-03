"""Regex-based particle detector for structured entities in taxonomic text.

Detects:
- DOI references
- MycoBank (MB) numbers
- Page references (p. / pp.)
- GBIF identifiers
- ISSN numbers
- Iconography-header phrases ("Selected icons", "Iconography", etc.)
- Author-footnote markers (``1)``, ``2)``, … at line start — author contact/affiliation footnotes)
- Fungarium collection codes (loaded from Redis + personal_fungaria.json)

Fungarium codes are loaded from the ``skol:fungaria`` Redis key maintained
by ``bin/manage_fungaria.py``.  If Redis is unavailable the detector falls
back to ``data/personal_fungaria.json`` only.

Detected spans are returned as :class:`ingestors.spans.Span` objects with
``source="regex"``.  Fungarium spans inside a ``Materials-examined`` section
receive ``confidence=0.9``; elsewhere ``confidence=0.6``.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestors.spans import Span

# ---------------------------------------------------------------------------
# Static regex patterns
# ---------------------------------------------------------------------------

_PATTERNS: Dict[str, re.Pattern] = {  # type: ignore[type-arg]
    "DOI": re.compile(r'\b(10\.\d{4,}/\S+)', re.IGNORECASE),
    "MB-number": re.compile(
        r'\bMB\s*(\d{5,7})\b|\bMycoBank\s+#?\s*(\d{5,7})\b',
        re.IGNORECASE,
    ),
    "Page-ref": re.compile(r'\b(?:p\.|pp\.)\s*(\d+(?:[-\u2013]\d+)?)'),
    "GBIF-ID": re.compile(r'\bGBIF[:\s]+(\d{7,})\b', re.IGNORECASE),
    "ISSN": re.compile(r'\bISSN\s*:?\s*(\d{4}-\d{3}[\dX])\b', re.IGNORECASE),
    # CBS culture-collection accession numbers (Westerdijk Institute).
    # Two formats observed in our corpus (24 681 hits across 1 044
    # docs): old dotted ``CBS 513.77`` (3 digits + period + 2-4 digits)
    # and modern ``CBS 136259`` (5-7 contiguous digits).  Either may
    # carry a trailing ``T`` / ``t`` for the type strain.  Composite
    # citations like ``CBS 144700/AP 6516 T`` match the CBS portion
    # only; the AP code is left for FungariumDetector if registered.
    # Lookup URL: https://wi.knaw.nl/fungal_table  (SPA \u2014 paste the
    # accession into the on-site search form).
    "CBS-number": re.compile(
        r'\bCBS\s+(\d{3}\.\d{2,4}|\d{5,7})(?:\s*[Tt])?\b',
        re.IGNORECASE,
    ),
    # Author affiliation / contact footnotes at the bottom of the first page
    # of a journal article.  The OCR'd text typically looks like:
    #   "1) Department of Botany, University of ..."
    #   "2) E-mail: author@example.com"
    # These are structurally distinct from treatment content and can be used
    # to detect the transition out of article-header material.
    "Author-footnote": re.compile(r'^[1-9]\)', re.MULTILINE),
    # Iconography section headers in taxonomic treatments.  These appear
    # inside Bibliography-labeled blocks but mark iconographic reference
    # lists that are part of the taxonomic treatment, not the main
    # bibliography.  Matching the header phrase allows downstream consumers
    # to distinguish the two without changing the Layer 1 label.
    "Iconography-header": re.compile(
        r'\b(selected\s+icons?'
        r'|selected\s+iconograph(?:y|ies)'
        r'|selected\s+illustrations?'
        r'|iconograph(?:y|ies))\s*[.:\-\u2013]?',
        re.IGNORECASE,
    ),
    # Synthetic page-boundary marker emitted by the skol PDF
    # extractor.  Two forms observed in the corpus:
    #   ``--- PDF Page 2 Label 2 ---`` (both numbers; usually equal)
    #   ``--- PDF Page 12 ---``        (page only)
    # MULTILINE anchors the regex to line starts so an inline mention
    # in body text doesn't match.  Detection lets v4's layout CRF
    # consume the marker as a particle (Step 2 ``particles[12]``
    # feature) and rescues page_header_detector's sequence fit on
    # docs where natural page-number runs are absent or weak \u2014 see
    # v4 plan \u00a71.B "page_header_detector.py" recommendation.
    "PDF-page-marker": re.compile(
        r'^---\s*PDF\s+Page\s+(\d+)'
        r'(?:\s+Label\s+(\d+))?\s*---\s*$',
        re.MULTILINE | re.IGNORECASE,
    ),
}

# Path to the personal fungaria JSON (relative to this module's package dir)
_PERSONAL_FUNGARIA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "personal_fungaria.json"
)

_MATERIALS_EXAMINED_LABEL = "Materials-examined"
_FUNGARIUM_CONFIDENCE_IN_SECTION = 0.9
_FUNGARIUM_CONFIDENCE_ELSEWHERE = 0.6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fungaria_codes(
    redis_client: Optional[Any] = None,
) -> List[str]:
    """Return herbarium codes from Redis and/or personal_fungaria.json.

    Codes are sorted longest-first so that longer codes match before
    shorter ones in the compiled regex (e.g. "DUKE" before "DU").

    Args:
        redis_client: Live ``redis.Redis`` instance, or ``None`` to skip
            the Redis lookup.

    Returns:
        Deduplicated list of herbarium code strings, longest first.
    """
    codes: set = set()

    # 1. Redis registry
    if redis_client is not None:
        try:
            raw = redis_client.get("skol:fungaria")
            if raw:
                data = json.loads(raw)
                for code in (data.get("institutions") or {}):
                    if code:
                        codes.add(code)
        except Exception:
            pass  # Redis unavailable — silently fall back

    # 2. Personal fungaria JSON
    if _PERSONAL_FUNGARIA_PATH.exists():
        try:
            entries = json.loads(_PERSONAL_FUNGARIA_PATH.read_text(encoding="utf-8"))
            for entry in entries:
                code = entry.get("code", "")
                if code:
                    codes.add(code)
        except Exception:
            pass

    return sorted(codes, key=len, reverse=True)


def _build_fungarium_pattern(codes: List[str]) -> Optional[re.Pattern]:  # type: ignore[type-arg]
    """Compile a fungarium detection regex from a list of codes.

    Returns ``None`` if *codes* is empty.

    Args:
        codes: Herbarium codes sorted longest-first.

    Returns:
        Compiled regex or ``None``.
    """
    if not codes:
        return None
    alternatives = "|".join(re.escape(c) for c in codes)
    return re.compile(
        r'\b(' + alternatives + r')[\s:]+([A-Z]?\d[\w./\-]+)',
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class FungariumDetector:
    """Detects herbarium voucher citations using a dynamically built regex.

    The pattern is compiled once from the union of codes in the Redis
    ``skol:fungaria`` key and ``data/personal_fungaria.json``.

    Args:
        redis_client: Live ``redis.Redis`` instance (may be ``None``).
    """

    def __init__(self, redis_client: Optional[Any] = None) -> None:
        codes = _load_fungaria_codes(redis_client)
        self._pattern: Optional[re.Pattern] = _build_fungarium_pattern(codes)  # type: ignore[type-arg]

    def detect(
        self,
        text: str,
        section_label: Optional[str] = None,
    ) -> List[Span]:
        """Find herbarium voucher citations in *text*.

        Args:
            text: Text to search.
            section_label: YEDDA section label for the current passage.
                Spans inside ``Materials-examined`` get ``confidence=0.9``;
                all others get ``confidence=0.6``.

        Returns:
            List of :class:`Span` objects for detected voucher citations.
        """
        if self._pattern is None:
            return []
        confidence = (
            _FUNGARIUM_CONFIDENCE_IN_SECTION
            if section_label == _MATERIALS_EXAMINED_LABEL
            else _FUNGARIUM_CONFIDENCE_ELSEWHERE
        )
        spans: List[Span] = []
        for m in self._pattern.finditer(text):
            spans.append(
                Span(
                    start=m.start(),
                    end=m.end(),
                    label="Fungarium-code",
                    text=m.group(0),
                    source="regex",
                    confidence=confidence,
                    metadata={"code": m.group(1), "accession": m.group(2)},
                )
            )
        return spans


def detect_particles(
    text: str,
    redis_client: Optional[Any] = None,
    section_label: Optional[str] = None,
) -> List[Span]:
    """Detect all structured particles in *text*.

    Runs the static regex patterns (DOI, MB-number, Page-ref, GBIF-ID)
    and the dynamic fungarium detector, then returns all spans combined.

    The spans are **not** conflict-resolved here; call
    :func:`ingestors.spans.resolve_conflicts` if needed.

    Args:
        text: Plaintext passage to search.
        redis_client: Redis client for loading fungarium codes (may be
            ``None`` to use personal_fungaria.json only).
        section_label: YEDDA section label, forwarded to
            :class:`FungariumDetector` to set confidence.

    Returns:
        Unsorted list of :class:`Span` objects for all detected particles.
    """
    spans: List[Span] = []

    for label, pattern in _PATTERNS.items():
        for m in pattern.finditer(text):
            metadata: Dict[str, Any] = {}
            # Per-label metadata extraction.  Two cases so far; if a
            # third lands, factor into a per-pattern hook map.
            if label == "CBS-number" and m.lastindex:
                metadata["accession"] = m.group(1)
            elif label == "PDF-page-marker":
                metadata["page_number"] = int(m.group(1))
                if m.group(2):
                    metadata["label_number"] = int(m.group(2))
            spans.append(
                Span(
                    start=m.start(),
                    end=m.end(),
                    label=label,
                    text=m.group(0),
                    source="regex",
                    metadata=metadata,
                )
            )

    detector = FungariumDetector(redis_client)
    spans.extend(detector.detect(text, section_label=section_label))

    return spans
