"""Canonical span data model for Layer 2 entity annotation.

Spans record character-offset entity ranges within article plaintext.
They are produced by gnfinder (taxon names), gnparser (authorship strings),
and particle_detector (DOIs, MycoBank numbers, etc.) and stored as
``article.spans.json`` CouchDB attachments.

JSON envelope schema (version "1")::

    {
      "version": "1",
      "doc_id": "abc123",
      "source_attachment": "article.txt",
      "spans": [
        {
          "start": 1024, "end": 1038,
          "label": "TaxonName",
          "text": "Pardosa moesta",
          "source": "gnfinder",
          "confidence": 0.99,
          "metadata": {
            "canonical": "Pardosa moesta",
            "cardinality": 2,
            "annot_nomen": "sp. nov.",
            "annot_nomen_type": "SP_NOV"
          }
        }
      ]
    }
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple


_JSON_VERSION = "1"


@dataclass
class Span:
    """A character-offset entity span within article plaintext.

    Attributes:
        start: Inclusive start character offset.
        end: Exclusive end character offset.
        label: Entity type label (e.g. "TaxonName", "Author", "DOI").
        text: Verbatim text slice ``article_text[start:end]``.
        source: Producer identifier ("gnfinder", "gnparser", "regex").
        confidence: Detection confidence in [0.0, 1.0]; default 1.0.
        metadata: Optional source-specific key/value payload.
    """

    start: int
    end: int
    label: str
    text: str
    source: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"end ({self.end}) must be > start ({self.start})"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )

    @property
    def length(self) -> int:
        """Number of characters covered by this span."""
        return self.end - self.start

    def overlaps(self, other: "Span") -> bool:
        """Return True if this span overlaps *other* (partial or full)."""
        return self.start < other.end and other.start < self.end


def spans_to_json(
    spans: List[Span],
    doc_id: str,
    source_attachment: str,
) -> str:
    """Serialize spans to a JSON string in the canonical envelope format.

    Args:
        spans: List of Span objects to serialize.
        doc_id: CouchDB document ``_id`` these spans belong to.
        source_attachment: Name of the plaintext attachment the offsets
            index into (typically ``"article.txt"``).

    Returns:
        JSON string with ``"version"``, ``"doc_id"``,
        ``"source_attachment"``, and ``"spans"`` keys.
    """
    envelope: Dict[str, Any] = {
        "version": _JSON_VERSION,
        "doc_id": doc_id,
        "source_attachment": source_attachment,
        "spans": [asdict(s) for s in spans],
    }
    return json.dumps(envelope, ensure_ascii=False)


def spans_from_json(json_str: str) -> List[Span]:
    """Deserialize spans from a JSON envelope string.

    Args:
        json_str: JSON string produced by :func:`spans_to_json`.

    Returns:
        List of :class:`Span` objects.

    Raises:
        KeyError: If the envelope is missing required keys.
        ValueError: If individual span fields fail validation.
    """
    envelope = json.loads(json_str)
    result: List[Span] = []
    for raw in envelope["spans"]:
        result.append(
            Span(
                start=int(raw["start"]),
                end=int(raw["end"]),
                label=raw["label"],
                text=raw["text"],
                source=raw["source"],
                confidence=float(raw.get("confidence", 1.0)),
                metadata=raw.get("metadata") or {},
            )
        )
    return result


def spans_to_bio(
    text: str,
    spans: List[Span],
) -> List[Tuple[str, str]]:
    """Convert spans to a BIO-tagged token list using whitespace tokenization.

    Tokens are produced by splitting *text* on whitespace.  Each token
    receives a BIO tag derived from the first span (by start offset) that
    covers the token's character range.

    Args:
        text: The full plaintext string the span offsets index into.
        spans: List of :class:`Span` objects (need not be sorted).

    Returns:
        List of ``(token, bio_tag)`` tuples where *bio_tag* is one of
        ``"O"``, ``"B-{label}"``, or ``"I-{label}"``.
    """
    sorted_spans = sorted(spans, key=lambda s: s.start)
    tokens: List[Tuple[str, str]] = []
    char_pos = 0
    for token in text.split():
        # locate token in original string (accounts for multi-space gaps)
        token_start = text.index(token, char_pos)
        token_end = token_start + len(token)
        char_pos = token_end

        tag = "O"
        for span in sorted_spans:
            if span.end <= token_start:
                continue
            if span.start >= token_end:
                break
            # token overlaps this span
            if token_start >= span.start:
                if token_start == span.start:
                    tag = f"B-{span.label}"
                else:
                    tag = f"I-{span.label}"
            else:
                tag = f"I-{span.label}"
            break
        tokens.append((token, tag))
    return tokens


def resolve_conflicts(spans: List[Span]) -> List[Span]:
    """Remove overlapping spans, keeping the shorter (more specific) one.

    When two spans overlap, the one with the smaller character length is
    retained.  If they are equal length, the one with the higher
    confidence is kept; ties are broken by lower start offset.

    The algorithm uses a greedy sweep: spans are sorted by start offset,
    then length ascending, then confidence descending so that in a single
    linear pass the preferred span is always encountered first.

    Args:
        spans: Unsorted list of :class:`Span` objects.

    Returns:
        Conflict-free list of spans in start-offset order.
    """
    # Sort: length asc, confidence desc — shorter / higher-confidence spans first
    candidates = sorted(
        spans,
        key=lambda s: (s.length, -s.confidence),
    )
    accepted: List[Span] = []
    for candidate in candidates:
        if any(candidate.overlaps(kept) for kept in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda s: s.start)
