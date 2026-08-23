"""The single supported way to turn a span's offsets into text.

Span character and line offsets are indexed to the treatment's
``attachment_name`` — normally ``article.txt.ann``, the
YEDDA-annotated file — inside the treatment's ``annotations_db``.
They are **not** offsets into ``article.txt``, and they are not in
the database ``ingest.db_name`` names.

That distinction is easy to get wrong and, worse, fails quietly:
reading the offsets against ``article.txt`` returns plausible prose
from elsewhere in the document rather than an error.  A whole
false "86 % of spans are mislocated" finding was built on exactly
that mistake — see §16 in
``docs/data_quality_production_v4_model.md``.

Two defences live here:

* :func:`coordinate_space` reads ``annotations_db`` and refuses to
  guess.  It never falls back to ``ingest.db_name``, which points at
  the raw-input database that holds ``article.txt`` and
  ``article.pdf`` but not the annotated file.  The *attachment name*
  is a different matter: that set is small and closed, so the stored
  name is tried first and ``FALLBACK_ATTACHMENTS`` after it — v3_hand
  treatments say ``article.txt.ann`` while the v3 classifier wrote
  ``article.pdf.ann``.  A wrong guess is caught by the fingerprint.
* :func:`span_head` / :func:`verify_head` implement a fingerprint
  stored on the span (``Span.head``).  Resolving compares it, so a
  wrong attachment, a stale offset or a re-extraction that moved the
  text all raise instead of returning something believable.

Everything that needs span text should call :func:`resolve_span`
rather than assembling its own attachment URL.

One deliberate exception: ``django/search/views.py`` keeps its own
``_collect_ann_db_candidates()`` probe order, because it must serve
older taxa documents that predate ``annotations_db`` — which this
module rejects by design rather than guessing around.  New code
should use this module.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

# Long enough to be distinctive, short enough that storing one per
# span is cheap.
HEAD_LENGTH = 40

# Tried in order after the treatment's own ``attachment_name``.  The
# DATABASE is never guessed — that is the mistake this module exists
# to prevent — but the attachment name is a small closed set, and
# with a `head` fingerprint a wrong choice is caught rather than
# silently accepted.  v3_hand needs this: its treatments say
# ``article.txt.ann`` while the v3 classifier actually wrote
# ``article.pdf.ann``.
FALLBACK_ATTACHMENTS = ('article.pdf.ann', 'article.txt.ann')


class SpanResolutionError(RuntimeError):
    """A span could not be resolved, or resolved to the wrong text."""


@dataclasses.dataclass(frozen=True)
class CoordinateSpace:
    """The (database, document, attachment) a span's offsets index."""

    db: str
    doc_id: str
    attachment: str

    def __str__(self) -> str:
        return f'{self.db}/{self.doc_id}/{self.attachment}'


def coordinate_space(treatment: Dict[str, Any]) -> CoordinateSpace:
    """Where this treatment's span offsets are measured.

    Raises:
        SpanResolutionError: if ``annotations_db``,
            ``attachment_name`` or the ingest document id is missing.
            Deliberately does not fall back to ``ingest.db_name``:
            that database holds ``article.txt``, not the annotated
            file, so falling back would silently resolve against the
            wrong coordinate space.
    """
    db = treatment.get('annotations_db')
    if not db:
        raise SpanResolutionError(
            f"treatment {treatment.get('_id')!r} has no 'annotations_db'; "
            f"refusing to guess the coordinate space "
            f"(ingest.db_name is the raw-input database and does not "
            f"hold the annotated attachment)"
        )
    attachment = treatment.get('attachment_name')
    if not attachment:
        raise SpanResolutionError(
            f"treatment {treatment.get('_id')!r} has no 'attachment_name'"
        )
    doc_id = (treatment.get('ingest') or {}).get('_id')
    if not doc_id:
        raise SpanResolutionError(
            f"treatment {treatment.get('_id')!r} has no ingest._id"
        )
    return CoordinateSpace(db=db, doc_id=doc_id, attachment=attachment)


def span_head(text: str) -> str:
    """A short whitespace-collapsed fingerprint of ``text``.

    Whitespace is collapsed so the fingerprint survives line
    rewrapping, which changes where newlines fall without changing
    the words.
    """
    return ' '.join(text.split())[:HEAD_LENGTH]


def verify_head(stored: Optional[str], actual: str) -> None:
    """Check a stored fingerprint against freshly resolved text.

    ``stored`` of ``None`` is tolerated — spans written before
    fingerprints existed must keep resolving.

    Raises:
        SpanResolutionError: on mismatch, quoting both values so the
            caller can see *what* it resolved to.
    """
    if stored is None:
        return
    if span_head(actual).startswith(stored[:HEAD_LENGTH]):
        return
    raise SpanResolutionError(
        f"span head mismatch: expected {stored!r}, "
        f"resolved to {span_head(actual)!r} — the offsets were almost "
        f"certainly read against the wrong attachment or are stale"
    )


def _candidate_attachments(space: CoordinateSpace) -> List[str]:
    """The stored attachment name first, then the known alternatives."""
    names = [space.attachment]
    names.extend(n for n in FALLBACK_ATTACHMENTS if n not in names)
    return names


def _attachment_text(space: CoordinateSpace, server: Any) -> str:
    if space.db not in server:
        raise SpanResolutionError(f"database {space.db!r} not found")
    db = server[space.db]
    tried = _candidate_attachments(space)
    for name in tried:
        try:
            blob = db.get_attachment(space.doc_id, name)
        except Exception as exc:                   # noqa: BLE001
            raise SpanResolutionError(
                f"cannot read {space.db}/{space.doc_id}/{name}: {exc}"
            ) from exc
        if blob is not None:
            decoded: str = blob.read().decode('utf-8', errors='replace')
            return decoded
    raise SpanResolutionError(
        f"no annotated attachment on {space.db}/{space.doc_id}; "
        f"tried {', '.join(tried)}"
    )


def _offsets(span: Dict[str, Any], space: 'CoordinateSpace',
             length: int) -> Tuple[int, int]:
    """Read and bounds-check a span's character offsets.

    Some stored spans carry offsets as strings rather than ints —
    taxon_09b97d5f's ``diagnosis_spans`` are one example — so they
    are coerced rather than trusted.
    """
    raw_start, raw_end = span.get('start_char'), span.get('end_char')
    if raw_start is None or raw_end is None:
        raise SpanResolutionError(f"span has no character offsets: {span!r}")
    try:
        start, end = int(raw_start), int(raw_end)
    except (TypeError, ValueError) as exc:
        raise SpanResolutionError(
            f"span has non-numeric character offsets: "
            f"start_char={raw_start!r} end_char={raw_end!r}"
        ) from exc
    if start < 0 or end > length or start > end:
        raise SpanResolutionError(
            f"span [{start}:{end}] falls outside {space} ({length} chars)"
        )
    return start, end


def resolve_span(
    treatment: Dict[str, Any],
    span: Dict[str, Any],
    server: Any,
) -> str:
    """Return the source text a span covers.

    Args:
        treatment: The treatments_prose document the span came from.
        span: One entry of a ``*_spans`` list.
        server: An open ``couchdb.Server``.

    Raises:
        SpanResolutionError: if the coordinate space is unknown, the
            attachment is missing, the offsets fall outside the file,
            or a stored ``head`` disagrees with the resolved text.
    """
    space = coordinate_space(treatment)
    text = _attachment_text(space, server)
    start, end = _offsets(span, space, len(text))
    resolved = text[start:end]
    verify_head(span.get('head'), resolved)
    return resolved


def resolve_spans(
    treatment: Dict[str, Any],
    spans: List[Dict[str, Any]],
    server: Any,
) -> List[str]:
    """Resolve several spans, reading the attachment once."""
    if not spans:
        return []
    space = coordinate_space(treatment)
    text = _attachment_text(space, server)
    out: List[str] = []
    for span in spans:
        start, end = _offsets(span, space, len(text))
        resolved = text[start:end]
        verify_head(span.get('head'), resolved)
        out.append(resolved)
    return out
