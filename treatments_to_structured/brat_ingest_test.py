"""Tests for treatments_to_structured.brat_ingest."""

from typing import Any, Dict

import pytest

from treatments_to_structured.brat_ingest import (
    AnnotationKey,
    DiffResult,
    annotation_key,
    diff_annotations,
    make_reviewed_doc,
    treatment_id_from_ann_filename,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ann(
    feature_label: str,
    start: int,
    end: int,
    *,
    field: str = 'description',
    extras: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Minimal annotation dict with the four identity fields plus
    optional extras (model, source_text, etc.)."""
    out: Dict[str, Any] = {
        'feature_label': feature_label,
        'field': field,
        'start': start,
        'end': end,
        'source_text': f'<{feature_label} text>',
        'source_spans': [{'start': start + 100, 'end': end + 100}],
    }
    if extras:
        out.update(extras)
    return out


# ---------------------------------------------------------------------------
# annotation_key
# ---------------------------------------------------------------------------


class TestAnnotationKey:
    def test_extracts_four_identity_fields(self) -> None:
        ann = _ann('Pileus', 48, 253)
        k = annotation_key(ann)
        assert k == AnnotationKey(
            feature_label='Pileus', field='description',
            start=48, end=253,
        )

    def test_coerces_string_offsets_to_int(self) -> None:
        """Defensive: some older candidate docs carry offsets as
        strings.  The diff must treat 48 and '48' as identical."""
        ann = {
            'feature_label': 'Pileus', 'field': 'description',
            'start': '48', 'end': '253',
        }
        k = annotation_key(ann)
        assert k.start == 48 and k.end == 253

    def test_key_is_hashable(self) -> None:
        """AnnotationKey must be usable as a dict key for the
        diff's O(N+M) lookup."""
        ann = _ann('Pileus', 48, 253)
        d = {annotation_key(ann): 'x'}
        assert d[annotation_key(ann)] == 'x'

    def test_label_difference_yields_different_keys(self) -> None:
        """Label drift across re-runs (the real 2026-06-28 case
        of 'Universal veil microstructure' vs '(microscopic)')
        produces distinct keys.  Two annotations at the same
        offsets but with different labels are NOT the same."""
        a = annotation_key(
            _ann('Universal veil microstructure on pileus', 5709, 6866),
        )
        b = annotation_key(
            _ann('Universal veil (microscopic, on pileus)', 5709, 6866),
        )
        assert a != b


# ---------------------------------------------------------------------------
# diff_annotations
# ---------------------------------------------------------------------------


class TestDiffAnnotations:
    """The three-way categorization that drives the reviewed-DB write."""

    def test_empty_inputs(self) -> None:
        result = diff_annotations([], [])
        assert result.kept == []
        assert result.added == []
        assert result.deleted == []

    def test_all_kept_when_reviewer_made_no_changes(self) -> None:
        candidates = [
            _ann('Pileus', 48, 253),
            _ann('Stipe', 451, 600),
        ]
        # Reviewer's brat output identical (parse_brat_ann would
        # produce equivalent dicts from the unmodified .ann).
        reviewed = [
            _ann('Pileus', 48, 253),
            _ann('Stipe', 451, 600),
        ]
        result = diff_annotations(reviewed, candidates)
        assert len(result.kept) == 2
        assert result.added == []
        assert result.deleted == []

    def test_reviewer_added_new_annotation(self) -> None:
        candidates = [_ann('Pileus', 48, 253)]
        reviewed = [
            _ann('Pileus', 48, 253),
            _ann('Annulus', 700, 720),  # new
        ]
        result = diff_annotations(reviewed, candidates)
        assert len(result.kept) == 1
        assert len(result.added) == 1
        assert result.added[0]['feature_label'] == 'Annulus'
        assert result.deleted == []

    def test_reviewer_deleted_annotation(self) -> None:
        candidates = [
            _ann('Pileus', 48, 253),
            _ann('Hallucination', 5000, 5010),  # reviewer rejects
        ]
        reviewed = [_ann('Pileus', 48, 253)]
        result = diff_annotations(reviewed, candidates)
        assert len(result.kept) == 1
        assert result.added == []
        assert len(result.deleted) == 1
        assert result.deleted[0]['feature_label'] == 'Hallucination'

    def test_boundary_edit_is_delete_plus_add(self) -> None:
        """Reviewer adjusted span boundaries — under our identity
        scheme (label + field + start + end), this looks like a
        delete of the original span and an add of the new span.
        Future heuristic could classify as 'edit'; Phase 1 stays
        strict."""
        candidates = [_ann('Pileus', 48, 253)]
        reviewed = [_ann('Pileus', 48, 280)]  # end moved
        result = diff_annotations(reviewed, candidates)
        assert result.kept == []
        assert len(result.added) == 1
        assert result.added[0]['end'] == 280
        assert len(result.deleted) == 1
        assert result.deleted[0]['end'] == 253

    def test_label_edit_is_delete_plus_add(self) -> None:
        """Reviewer renamed feature_label — also delete+add under
        the strict identity scheme."""
        candidates = [_ann('Spores', 792, 966)]
        reviewed = [_ann('Basidiospores', 792, 966)]
        result = diff_annotations(reviewed, candidates)
        assert result.kept == []
        assert len(result.added) == 1
        assert result.added[0]['feature_label'] == 'Basidiospores'
        assert len(result.deleted) == 1
        assert result.deleted[0]['feature_label'] == 'Spores'

    def test_multiple_same_label_different_offsets(self) -> None:
        """Three Lamellae annotations at different offsets in the
        same treatment (the Murrill multi-species case from
        taxon_22346900...).  Each is a distinct identity, each is
        independently kept / added / deleted."""
        candidates = [
            _ann('Lamellae', 344, 399),
            _ann('Lamellae', 1029, 1133),
            _ann('Lamellae', 1850, 1945),
        ]
        # Reviewer accepted first, deleted second, kept third
        reviewed = [
            _ann('Lamellae', 344, 399),
            _ann('Lamellae', 1850, 1945),
        ]
        result = diff_annotations(reviewed, candidates)
        assert len(result.kept) == 2
        assert result.added == []
        assert len(result.deleted) == 1
        assert result.deleted[0]['start'] == 1029

    def test_summary_string(self) -> None:
        candidates = [_ann('Pileus', 48, 253), _ann('Veil', 800, 900)]
        reviewed = [_ann('Pileus', 48, 253), _ann('Annulus', 700, 720)]
        result = diff_annotations(reviewed, candidates)
        summary = result.summary()
        assert 'kept=1' in summary
        assert 'added=1' in summary
        assert 'deleted=1' in summary


# ---------------------------------------------------------------------------
# make_reviewed_doc
# ---------------------------------------------------------------------------


class TestMakeReviewedDoc:
    """Build the doc that goes into the reviewed DB."""

    def test_kept_preserves_bootstrap_provenance(self) -> None:
        """For 'kept', the reviewed doc carries the original
        Claude model + created_at as provenance fields — so
        downstream consumers can compute reviewer/Claude agreement
        without joining DBs."""
        reviewed_ann = _ann('Pileus', 48, 253)
        candidate = _ann('Pileus', 48, 253, extras={
            'model': 'claude-opus-4-7',
            'created_at': '2026-06-28T20:46:16.174358+00:00',
        })
        doc = make_reviewed_doc(
            reviewed_ann,
            treatment_id='taxon_abc',
            doc_id='ingest_xyz',
            reviewer='operator@host',
            reviewed_at='2026-06-29T10:00:00+00:00',
            action='kept',
            candidate_match=candidate,
        )
        assert doc['model'] == 'claude-opus-4-7'
        assert (
            doc['created_at']
            == '2026-06-28T20:46:16.174358+00:00'
        )
        assert doc['reviewer'] == 'operator@host'
        assert (
            doc['reviewed_at'] == '2026-06-29T10:00:00+00:00'
        )
        assert doc['reviewer_action'] == 'kept'

    def test_added_has_no_bootstrap_provenance(self) -> None:
        """For 'added', the reviewer introduced the span — there
        was no Claude run that produced it, so model + created_at
        are null."""
        reviewed_ann = _ann('Annulus', 700, 720)
        doc = make_reviewed_doc(
            reviewed_ann,
            treatment_id='taxon_abc',
            doc_id='ingest_xyz',
            reviewer='operator@host',
            reviewed_at='2026-06-29T10:00:00+00:00',
            action='added',
            candidate_match=None,
        )
        assert doc['model'] is None
        assert doc['created_at'] is None
        assert doc['reviewer_action'] == 'added'

    def test_id_matches_candidate_db_scheme(self) -> None:
        """Same _id scheme as the candidate DB — operator can
        join the two DBs by _id to compare bootstrap vs reviewed."""
        reviewed_ann = _ann('Pileus', 48, 253)
        doc = make_reviewed_doc(
            reviewed_ann, treatment_id='taxon_abc',
            doc_id='d', reviewer='r', reviewed_at='t',
            action='added',
        )
        assert doc['_id'] == 'taxon_abc:Pileus:48'

    def test_no_rev_in_returned_doc(self) -> None:
        """Caller is responsible for merging _rev for overwrite —
        same contract as status.make_status_doc."""
        reviewed_ann = _ann('Pileus', 48, 253)
        doc = make_reviewed_doc(
            reviewed_ann, treatment_id='taxon_abc',
            doc_id='d', reviewer='r', reviewed_at='t',
            action='added',
        )
        assert '_rev' not in doc

    def test_invalid_action_raises(self) -> None:
        """Typo guard: only 'kept' and 'added' produce docs.
        'deleted' annotations don't appear in the reviewed DB."""
        with pytest.raises(ValueError):
            make_reviewed_doc(
                _ann('Pileus', 48, 253),
                treatment_id='taxon_abc',
                doc_id='d', reviewer='r', reviewed_at='t',
                action='deleted',
            )

    def test_source_text_and_spans_from_reviewer(self) -> None:
        """The reviewed doc's source_text and source_spans come
        from the reviewer's annotation (parse_brat_ann re-derived
        them from the synth doc), NOT from the candidate.  This
        matters for boundary edits — the reviewer's new boundaries
        produce new source_text."""
        reviewed_ann = _ann('Pileus', 48, 280)
        reviewed_ann['source_text'] = 'Pileus ... extended.'
        candidate = _ann('Pileus', 48, 253, extras={
            'source_text': 'Pileus ... original.',
            'model': 'claude-opus-4-7',
        })
        doc = make_reviewed_doc(
            reviewed_ann, treatment_id='taxon_abc',
            doc_id='d', reviewer='r', reviewed_at='t',
            action='added',  # boundary change → add
            candidate_match=None,
        )
        assert doc['source_text'] == 'Pileus ... extended.'

    def test_offsets_coerced_to_int(self) -> None:
        """Defensive: string offsets get normalized."""
        reviewed_ann = {
            'feature_label': 'Pileus', 'field': 'description',
            'start': '48', 'end': '253',
            'source_text': 'x', 'source_spans': [],
        }
        doc = make_reviewed_doc(
            reviewed_ann, treatment_id='taxon_abc',
            doc_id='d', reviewer='r', reviewed_at='t',
            action='added',
        )
        assert doc['start'] == 48
        assert doc['end'] == 253
        assert isinstance(doc['start'], int)


# ---------------------------------------------------------------------------
# treatment_id_from_ann_filename
# ---------------------------------------------------------------------------


class TestTreatmentIdFromAnnFilename:
    """Filename convention: <treatment_id>.ann."""

    def test_basename_with_directory(self) -> None:
        tid = treatment_id_from_ann_filename(
            '/some/path/to/taxon_841d5cbed.ann',
        )
        assert tid == 'taxon_841d5cbed'

    def test_just_filename(self) -> None:
        tid = treatment_id_from_ann_filename('taxon_abc.ann')
        assert tid == 'taxon_abc'

    def test_full_hash_treatment_id(self) -> None:
        """Real treatment IDs are 64+ chars; the helper doesn't
        truncate or normalize."""
        full = 'taxon_841d5cbed697b1882ba6b0f044556d801ae2df2f698fcc72c7a52bcb2349ce44'
        tid = treatment_id_from_ann_filename(f'{full}.ann')
        assert tid == full

    def test_missing_extension_raises(self) -> None:
        with pytest.raises(ValueError) as exc:
            treatment_id_from_ann_filename('taxon_abc.txt')
        assert '.ann' in str(exc.value)
