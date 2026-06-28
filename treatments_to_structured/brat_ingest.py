"""Pure-logic helpers for the Phase 1 brat-review ingestion.

Phase 1 deliverable 6 of treatments_to_structured.  See
docs/schema_constrained_pipeline.md §10.4.

Reads a reviewer's edited brat ``.ann`` file (parsed via
``brat_render.parse_brat_ann``) and computes the diff against the
candidate-DB annotations Claude originally produced.  Three
outcomes per annotation:

  * ``kept`` — the reviewer accepted the bootstrap span exactly
    (same field-relative offsets, same feature_label).
  * ``added`` — the reviewer introduced a span the bootstrap
    didn't have (new T-line in the .ann).
  * ``deleted`` — the bootstrap had a span the reviewer removed
    (candidate annotation has no matching T-line).

Boundary changes and label changes ARE delete+add under this
scheme, since the durable ``_id`` is
``<treatment_id>:<feature_label>:<start>`` — any change of either
field produces a different ``_id`` and thus a different annotation
identity.  Future enhancement: heuristic matching to identify
likely "edits" (same field/offsets, different label, etc.) for
operator reporting.  Phase 1 keeps the diff strict.

The CLI (``bin/brat_ingest.py``) wires this with CouchDB I/O.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AnnotationKey:
    """The identity tuple used to match reviewed vs candidate
    annotations.  Two annotations with the same key are considered
    "the same annotation" — anything else is delete+add."""

    feature_label: str
    field: str
    start: int
    end: int


def annotation_key(ann: Dict[str, Any]) -> AnnotationKey:
    """Extract the identity tuple from an annotation dict.

    Works on both candidate-DB annotation docs and parsed brat
    ``.ann`` results from ``brat_render.parse_brat_ann`` — both
    carry the same four fields.
    """
    return AnnotationKey(
        feature_label=ann['feature_label'],
        field=ann['field'],
        start=int(ann['start']),
        end=int(ann['end']),
    )


@dataclass
class DiffResult:
    """Result of diffing a reviewer's brat .ann against the
    candidate-DB annotations for one treatment.

    Each list carries the full annotation dicts (not just keys), so
    downstream code can write the kept+added set to the reviewed DB
    without re-fetching anything.
    """

    kept: List[Dict[str, Any]] = field(default_factory=list)
    added: List[Dict[str, Any]] = field(default_factory=list)
    deleted: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Compact one-line summary for stderr reporting."""
        return (
            f'kept={len(self.kept)} '
            f'added={len(self.added)} '
            f'deleted={len(self.deleted)}'
        )


def diff_annotations(
    reviewed_anns: List[Dict[str, Any]],
    candidate_anns: List[Dict[str, Any]],
) -> DiffResult:
    """Diff a reviewer's brat .ann annotations against the
    candidate-DB annotations.

    Args:
        reviewed_anns: From ``brat_render.parse_brat_ann`` on the
            reviewer's edited .ann file.  Each dict has
            ``feature_label``, ``field``, ``start``, ``end``,
            ``source_text``, ``source_spans``.
        candidate_anns: From the candidate DB for one treatment.
            Same shape plus ``treatment_id``, ``doc_id``, ``model``,
            ``created_at``, ``_id``, ``_rev``.

    Returns:
        ``DiffResult`` with kept/added/deleted lists.  The reviewed
        dicts in ``kept`` and ``added`` carry the reviewer's
        ``source_text`` / ``source_spans`` (re-derived by
        parse_brat_ann from the synth doc); ``deleted`` dicts are
        the original candidate docs untouched.
    """
    candidate_by_key: Dict[AnnotationKey, Dict[str, Any]] = {
        annotation_key(c): c for c in candidate_anns
    }
    reviewed_keys: set = set()
    result = DiffResult()
    for r in reviewed_anns:
        k = annotation_key(r)
        reviewed_keys.add(k)
        if k in candidate_by_key:
            result.kept.append(r)
        else:
            result.added.append(r)
    for k, c in candidate_by_key.items():
        if k not in reviewed_keys:
            result.deleted.append(c)
    return result


def make_reviewed_doc(
    ann: Dict[str, Any],
    treatment_id: str,
    doc_id: str,
    reviewer: str,
    reviewed_at: str,
    action: str,
    *,
    candidate_match: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a reviewed-DB annotation doc from a kept-or-added
    annotation.

    Args:
        ann: Annotation dict (kept = from parse_brat_ann; added =
            from parse_brat_ann).
        treatment_id: For populating the doc field.  parse_brat_ann
            doesn't know the treatment, so the caller supplies it.
        doc_id: Source ingest doc's ``_id``.  Same reason.
        reviewer: Free-form identifier (user@host, name, etc.).
        reviewed_at: ISO-8601 timestamp.
        action: ``'kept'`` or ``'added'``.
        candidate_match: For ``kept``, the original candidate doc
            so its ``model`` and ``created_at`` (provenance of the
            bootstrap) flow through.  None for ``added`` (no
            bootstrap provenance to preserve).

    Returns:
        Dict ready for CouchDB save.  ``_id`` is built from the
        annotation's identity tuple — same scheme as the candidate
        DB, so the operator can join the two DBs by ``_id`` if
        they want side-by-side comparison.  ``_rev`` is NOT set;
        caller is responsible for merge-with-rev semantics.
    """
    if action not in {'kept', 'added'}:
        raise ValueError(
            f"action must be 'kept' or 'added'; got {action!r}"
        )
    doc_id_str = (
        f"{treatment_id}:{ann['feature_label']}:{int(ann['start'])}"
    )
    if action == 'kept' and candidate_match is not None:
        bootstrap_model = candidate_match.get('model')
        bootstrap_created_at = candidate_match.get('created_at')
    else:
        bootstrap_model = None
        bootstrap_created_at = None
    return {
        '_id': doc_id_str,
        'feature_label': ann['feature_label'],
        'field': ann['field'],
        'start': int(ann['start']),
        'end': int(ann['end']),
        'source_text': ann.get('source_text', ''),
        'source_spans': ann.get('source_spans', []),
        'treatment_id': treatment_id,
        'doc_id': doc_id,
        'model': bootstrap_model,
        'created_at': bootstrap_created_at,
        'reviewed_at': reviewed_at,
        'reviewer': reviewer,
        'reviewer_action': action,
    }


def treatment_id_from_ann_filename(path: str) -> str:
    """Convention: brat .ann file is named ``<treatment_id>.ann``.

    Used by the CLI's --ann-dir mode to discover which treatment
    each file belongs to without making the operator pass --doc-id
    alongside.
    """
    import os
    base = os.path.basename(path)
    if not base.endswith('.ann'):
        raise ValueError(
            f"expected .ann extension; got {path!r}"
        )
    return base[:-len('.ann')]


__all__ = (
    'AnnotationKey',
    'annotation_key',
    'DiffResult',
    'diff_annotations',
    'make_reviewed_doc',
    'treatment_id_from_ann_filename',
)
