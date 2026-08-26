"""Tests for bin/brat_export CLI helpers.

CouchDB and filesystem-touching paths are exercised through small
fakes; the live test (in a commit message) covers end-to-end.
"""

import io
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brat_export import (  # type: ignore[import]  # noqa: E402
    collect_entity_types,
    fetch_anns_for_treatment,
    list_annotated_treatment_ids,
    fetch_reviewer_touched_ids,
    render_annotation_conf,
    select_treatment_ids,
)


# ---------------------------------------------------------------------------
# CouchDB fakes (smaller than bin/brat_ingest_test.py's — only need
# rows-of-_id for list_annotated, and rows-of-doc for fetch_anns)
# ---------------------------------------------------------------------------


class _FakeRow:
    """Mimics the couchdb-python row shape: an .id attribute and an
    optional .doc attribute (only populated when include_docs=True
    is passed)."""

    def __init__(self, _id: str, doc: Dict[str, Any] = None) -> None:
        self.id = _id
        self.doc = doc


class _FakeView:
    def __init__(self, rows: List[_FakeRow]) -> None:
        self.rows = rows


class _FakeAnnDb:
    """A CouchDB-like stand-in keyed by _id, returning rows that
    behave like couchdb-python's view rows."""

    def __init__(self, docs: List[Dict[str, Any]]) -> None:
        self.docs = docs

    def view(self, _name: str, **kwargs: Any) -> _FakeView:
        # _all_docs without bounds → every doc (used by
        # list_annotated_treatment_ids).
        if 'startkey' not in kwargs:
            return _FakeView(
                rows=[
                    _FakeRow(d.get('_id', ''),
                             doc=d if kwargs.get('include_docs')
                             else None)
                    for d in self.docs
                ],
            )
        # Bounded scan (used by fetch_anns_for_treatment).
        startkey = kwargs['startkey']
        endkey = kwargs['endkey']
        rows = []
        for d in self.docs:
            _id = d.get('_id', '')
            if startkey <= _id <= endkey:
                rows.append(_FakeRow(_id, doc=d))
        return _FakeView(rows)


# ---------------------------------------------------------------------------
# list_annotated_treatment_ids
# ---------------------------------------------------------------------------


class TestListAnnotatedTreatmentIds:
    """Distinct treatment IDs from annotation _ids of shape
    <tid>:<label>:<offset>."""

    def test_distinct_ids_sorted(self) -> None:
        db = _FakeAnnDb([
            {'_id': 'taxon_b:Pileus:0'},
            {'_id': 'taxon_a:Pileus:0'},
            {'_id': 'taxon_a:Stipe:100'},  # repeat tid
            {'_id': 'taxon_c:Lamellae:5'},
        ])
        ids = list_annotated_treatment_ids(db)
        assert ids == ['taxon_a', 'taxon_b', 'taxon_c']

    def test_skips_ids_without_colon(self) -> None:
        """A _design doc or design view appears with no colon —
        must not be treated as a treatment ID."""
        db = _FakeAnnDb([
            {'_id': 'taxon_a:Pileus:0'},
            {'_id': '_design/blah'},
            {'_id': ''},
        ])
        ids = list_annotated_treatment_ids(db)
        assert ids == ['taxon_a']

    def test_empty_db(self) -> None:
        db = _FakeAnnDb([])
        assert list_annotated_treatment_ids(db) == []


# ---------------------------------------------------------------------------
# fetch_anns_for_treatment
# ---------------------------------------------------------------------------


class TestFetchAnnsForTreatment:
    def test_returns_only_matching_treatment_docs(self) -> None:
        db = _FakeAnnDb([
            {'_id': 'taxon_a:Pileus:0',
             'feature_label': 'Pileus'},
            {'_id': 'taxon_a:Stipe:100',
             'feature_label': 'Stipe'},
            {'_id': 'taxon_b:Pileus:0',
             'feature_label': 'Pileus'},
        ])
        result = fetch_anns_for_treatment(db, 'taxon_a')
        ids = sorted(d['_id'] for d in result)
        assert ids == ['taxon_a:Pileus:0', 'taxon_a:Stipe:100']

    def test_empty_when_no_match(self) -> None:
        db = _FakeAnnDb([{'_id': 'taxon_a:Pileus:0'}])
        assert fetch_anns_for_treatment(db, 'taxon_z') == []


# ---------------------------------------------------------------------------
# collect_entity_types
# ---------------------------------------------------------------------------


class TestCollectEntityTypes:
    """Build the annotation.conf entity-type list from observed
    feature_labels.  Wire form is space → underscore (matches
    annotations_to_brat / parse_brat_ann)."""

    def test_single_treatment_simple_labels(self) -> None:
        anns = {'taxon_a': [
            {'feature_label': 'Pileus'},
            {'feature_label': 'Stipe'},
        ]}
        types = collect_entity_types(anns)
        assert types == ['Pileus', 'Stipe']

    def test_multi_word_labels_underscore_wire_form(self) -> None:
        """Brat type tokens can't contain whitespace; annotations_to_
        brat substitutes spaces with underscores on the wire.
        annotation.conf must match the wire form."""
        anns = {'taxon_a': [
            {'feature_label': 'Basal mycelium'},
            {'feature_label': 'Universal veil on pileus'},
        ]}
        types = collect_entity_types(anns)
        assert types == [
            'Basal_mycelium', 'Universal_veil_on_pileus',
        ]

    def test_deduplicates_across_treatments(self) -> None:
        anns = {
            'taxon_a': [
                {'feature_label': 'Pileus'},
                {'feature_label': 'Stipe'},
            ],
            'taxon_b': [
                {'feature_label': 'Pileus'},  # dup
                {'feature_label': 'Lamellae'},
            ],
        }
        types = collect_entity_types(anns)
        assert types == ['Lamellae', 'Pileus', 'Stipe']

    def test_skips_empty_labels(self) -> None:
        """Defensive: a malformed doc with empty/None label
        shouldn't crash the export."""
        anns = {'taxon_a': [
            {'feature_label': ''},
            {'feature_label': None},
            {'feature_label': 'Pileus'},
        ]}
        types = collect_entity_types(anns)
        assert types == ['Pileus']

    def test_empty_input(self) -> None:
        assert collect_entity_types({}) == []


# ---------------------------------------------------------------------------
# render_annotation_conf
# ---------------------------------------------------------------------------


class TestRenderAnnotationConf:
    """The brat annotation.conf format."""

    def test_has_entities_header(self) -> None:
        conf = render_annotation_conf(['Pileus', 'Stipe'])
        assert '[entities]' in conf

    def test_lists_types(self) -> None:
        conf = render_annotation_conf(['Pileus', 'Stipe'])
        # One type per line after the [entities] header.
        lines = conf.splitlines()
        idx = lines.index('[entities]')
        assert lines[idx + 1:idx + 3] == ['Pileus', 'Stipe']

    def test_provenance_comment_present(self) -> None:
        """The header comment captures provenance so the operator
        knows which export produced this conf."""
        conf = render_annotation_conf(['Pileus'])
        assert '# brat annotation config' in conf
        assert 'brat_export' in conf

    def test_empty_types_emits_placeholder_section(self) -> None:
        """Brat requires [entities] to exist; emit it even when
        empty so operator can extend manually."""
        conf = render_annotation_conf([])
        assert '[entities]' in conf
        assert 'no entity types found' in conf

    def test_trailing_newline(self) -> None:
        """POSIX text-file convention: end with a newline."""
        conf = render_annotation_conf(['Pileus'])
        assert conf.endswith('\n')

    def test_emits_empty_relations_events_attributes(self) -> None:
        """Brat warns at load time if any of [relations] /
        [events] / [attributes] is missing — 'Project
        configuration: missing section ... may be wrong'.
        Phase 1 only uses entities, but we emit empty sections
        anyway so brat reads the config as 'intentionally
        entity-only' rather than 'incomplete'."""
        conf = render_annotation_conf(['Pileus'])
        assert '[relations]' in conf
        assert '[events]' in conf
        assert '[attributes]' in conf


# ---------------------------------------------------------------------------
# select_treatment_ids
# ---------------------------------------------------------------------------


class TestSelectTreatmentIds:
    def test_no_filter_returns_all_annotated(self) -> None:
        db = _FakeAnnDb([
            {'_id': 'taxon_a:Pileus:0'},
            {'_id': 'taxon_b:Pileus:0'},
        ])
        result = select_treatment_ids(db, None)
        assert result == ['taxon_a', 'taxon_b']

    def test_filter_restricts(self) -> None:
        db = _FakeAnnDb([
            {'_id': 'taxon_a:Pileus:0'},
            {'_id': 'taxon_b:Pileus:0'},
            {'_id': 'taxon_c:Pileus:0'},
        ])
        result = select_treatment_ids(db, ['taxon_a', 'taxon_c'])
        assert result == ['taxon_a', 'taxon_c']

    def test_filter_preserves_input_order(self) -> None:
        """Operator order is meaningful — the export-then-review
        order might be intentional (e.g., low-complexity-first to
        warm up)."""
        db = _FakeAnnDb([
            {'_id': 'taxon_a:Pileus:0'},
            {'_id': 'taxon_b:Pileus:0'},
            {'_id': 'taxon_c:Pileus:0'},
        ])
        result = select_treatment_ids(db, ['taxon_c', 'taxon_a'])
        assert result == ['taxon_c', 'taxon_a']

    def test_filter_with_unknown_id_raises(self) -> None:
        """Operator typo'd a doc ID — fail loudly rather than
        silently producing fewer files than expected."""
        db = _FakeAnnDb([{'_id': 'taxon_a:Pileus:0'}])
        with pytest.raises(ValueError) as exc:
            select_treatment_ids(db, ['taxon_a', 'taxon_xyz'])
        assert 'taxon_xyz' in str(exc.value)

    def test_no_filter_empty_db_raises(self) -> None:
        """No annotations at all → nothing to do.  Surface as
        error rather than producing an empty directory the
        operator might mistake for success."""
        db = _FakeAnnDb([])
        with pytest.raises(ValueError) as exc:
            select_treatment_ids(db, None)
        assert 'no annotations' in str(exc.value)


# ---------------------------------------------------------------------------
# fetch_reviewer_touched_ids — reviewed-even-if-empty detection
# via features_status.reviewer_action (written by brat_ingest).
# ---------------------------------------------------------------------------


class TestFetchReviewerTouchedIds:
    """Treatments that brat_ingest touched, including the case
    where the reviewer rejected every annotation (features_hand
    ends up empty for them).  Complements
    list_annotated_treatment_ids on features_hand — either
    presence in features_hand OR presence of reviewer_action in
    features_status counts as 'reviewed'."""

    def test_only_reviewer_action_docs_returned(self) -> None:
        docs = [
            {
                '_id': 'taxon_reviewed_empty',
                'treatment_id': 'taxon_reviewed_empty',
                'status': 'success',
                'annotation_count': 0,
                'reviewer_action': {
                    'reviewer': 'operator@host',
                    'reviewed_at': '2026-07-01T00:00:00Z',
                    'kept_count': 0, 'added_count': 0,
                    'deleted_count': 1,
                },
            },
            {
                '_id': 'taxon_reviewed_with_kept',
                'treatment_id': 'taxon_reviewed_with_kept',
                'status': 'success',
                'annotation_count': 5,
                'reviewer_action': {
                    'reviewer': 'operator@host',
                    'reviewed_at': '2026-07-01T00:00:00Z',
                    'kept_count': 5, 'added_count': 0,
                    'deleted_count': 0,
                },
            },
            {
                # No reviewer_action — bootstrapped but not
                # reviewed.  Must NOT appear.
                '_id': 'taxon_not_reviewed',
                'treatment_id': 'taxon_not_reviewed',
                'status': 'success',
                'annotation_count': 8,
            },
            {
                # skipped_merge_suspect — also not reviewed.
                '_id': 'taxon_skipped',
                'treatment_id': 'taxon_skipped',
                'status': 'skipped_merge_suspect',
            },
        ]
        db = _FakeAnnDb(docs)
        assert fetch_reviewer_touched_ids(db) == {
            'taxon_reviewed_empty',
            'taxon_reviewed_with_kept',
        }

    def test_reviewed_empty_case_detected(self) -> None:
        """The headline benefit: a treatment that was reviewed
        and had every annotation rejected (features_hand empty
        for it) still counts as reviewed via reviewer_action."""
        db = _FakeAnnDb([{
            '_id': 'taxon_empty',
            'treatment_id': 'taxon_empty',
            'status': 'success',
            'annotation_count': 1,  # bootstrap produced 1
            'reviewer_action': {
                'kept_count': 0,
                'added_count': 0,
                'deleted_count': 1,  # reviewer rejected the 1
                'reviewer': 'r', 'reviewed_at': 't',
            },
        }])
        assert 'taxon_empty' in fetch_reviewer_touched_ids(db)

    def test_skips_none_doc_rows(self) -> None:
        """Defensive: deleted-doc rows have doc=None; skip."""
        class _RowNone:
            id = 'taxon_deleted'
            doc = None

        class _StubDb:
            def view(self, *_, **__):
                class _V:
                    rows = [_RowNone()]
                return _V()
        assert fetch_reviewer_touched_ids(_StubDb()) == set()

    def test_empty_db(self) -> None:
        assert fetch_reviewer_touched_ids(_FakeAnnDb([])) == set()

    def test_falls_back_to_row_id_when_treatment_id_missing(
        self,
    ) -> None:
        """Defensive: if a status doc is missing the
        ``treatment_id`` field (shouldn't happen but defensive
        against malformed data), fall back to the row _id."""
        db = _FakeAnnDb([{
            '_id': 'taxon_a',
            'reviewer_action': {'kept_count': 0},
        }])
        assert fetch_reviewer_touched_ids(db) == {'taxon_a'}


class _FakeTreatmentsDb:
    """Stand-in for treatments_prose: membership is all we need to
    tell a typo'd ID from a treatment that simply produced no
    annotations."""

    def __init__(self, ids: List[str]) -> None:
        self._ids = set(ids)

    def __contains__(self, key: object) -> bool:
        return key in self._ids


class TestSelectTreatmentIdsSkipUnannotated:
    """`--doc-id` hard-fails when any ID lacks annotations, to catch
    typos.  But a treatment Claude legitimately returned no spans for
    is indistinguishable from a typo by that test, so one empty
    treatment poisons a whole round's export.  --skip-unannotated
    separates the two cases using the treatments DB: an ID that IS a
    treatment but has no annotations is dropped with a warning; an ID
    that is not a treatment at all still raises."""

    def test_default_still_raises_on_empty_treatment(self) -> None:
        """Unchanged behaviour without the flag."""
        db = _FakeAnnDb([{'_id': 'taxon_a:Pileus:0'}])
        treatments = _FakeTreatmentsDb(['taxon_a', 'taxon_empty'])
        with pytest.raises(ValueError):
            select_treatment_ids(
                db, ['taxon_a', 'taxon_empty'],
                treatments_db=treatments,
            )

    def test_skips_empty_treatment(self) -> None:
        db = _FakeAnnDb([{'_id': 'taxon_a:Pileus:0'}])
        treatments = _FakeTreatmentsDb(['taxon_a', 'taxon_empty'])
        warn = io.StringIO()
        result = select_treatment_ids(
            db, ['taxon_a', 'taxon_empty'],
            skip_unannotated=True, treatments_db=treatments,
            warn_stream=warn,
        )
        assert result == ['taxon_a']
        assert 'taxon_empty' in warn.getvalue()

    def test_still_raises_on_a_real_typo(self) -> None:
        """The guard's actual purpose survives the flag: an ID that
        is not a treatment at all is a typo, not an empty result."""
        db = _FakeAnnDb([{'_id': 'taxon_a:Pileus:0'}])
        treatments = _FakeTreatmentsDb(['taxon_a', 'taxon_empty'])
        with pytest.raises(ValueError) as exc:
            select_treatment_ids(
                db, ['taxon_a', 'taxon_typo'],
                skip_unannotated=True, treatments_db=treatments,
                warn_stream=io.StringIO(),
            )
        assert 'taxon_typo' in str(exc.value)

    def test_typo_reported_even_alongside_an_empty(self) -> None:
        """A typo must not be masked by a legitimately empty ID in
        the same batch, and the message must name only the typo."""
        db = _FakeAnnDb([{'_id': 'taxon_a:Pileus:0'}])
        treatments = _FakeTreatmentsDb(['taxon_a', 'taxon_empty'])
        with pytest.raises(ValueError) as exc:
            select_treatment_ids(
                db, ['taxon_a', 'taxon_empty', 'taxon_typo'],
                skip_unannotated=True, treatments_db=treatments,
                warn_stream=io.StringIO(),
            )
        msg = str(exc.value)
        assert 'taxon_typo' in msg
        assert 'taxon_empty' not in msg

    def test_preserves_order_of_survivors(self) -> None:
        db = _FakeAnnDb([
            {'_id': 'taxon_a:Pileus:0'},
            {'_id': 'taxon_c:Pileus:0'},
        ])
        treatments = _FakeTreatmentsDb(['taxon_a', 'taxon_b', 'taxon_c'])
        result = select_treatment_ids(
            db, ['taxon_c', 'taxon_b', 'taxon_a'],
            skip_unannotated=True, treatments_db=treatments,
            warn_stream=io.StringIO(),
        )
        assert result == ['taxon_c', 'taxon_a']

    def test_all_empty_raises_rather_than_exporting_nothing(
        self,
    ) -> None:
        """An empty export directory looks like success.  Fail."""
        db = _FakeAnnDb([{'_id': 'taxon_z:Pileus:0'}])
        treatments = _FakeTreatmentsDb(['taxon_a', 'taxon_b'])
        with pytest.raises(ValueError) as exc:
            select_treatment_ids(
                db, ['taxon_a', 'taxon_b'],
                skip_unannotated=True, treatments_db=treatments,
                warn_stream=io.StringIO(),
            )
        assert 'no annotations' in str(exc.value)

    def test_without_treatments_db_cannot_distinguish(self) -> None:
        """Refuse to guess: skipping blindly would silently swallow
        the typos the guard exists to catch."""
        db = _FakeAnnDb([{'_id': 'taxon_a:Pileus:0'}])
        with pytest.raises(ValueError) as exc:
            select_treatment_ids(
                db, ['taxon_a', 'taxon_empty'],
                skip_unannotated=True, treatments_db=None,
                warn_stream=io.StringIO(),
            )
        assert 'treatments' in str(exc.value).lower()


class TestSelectTreatmentIdsAllowUnannotated:
    """`--skip-unannotated` drops what has no annotations;
    `--allow-unannotated` **keeps** it and exports an empty `.ann`.

    They answer opposite questions and both are right sometimes.
    Skipping suits a round where the annotations are the point and one
    empty treatment would abort the export.  Keeping suits review of
    what Claude *missed*, where a treatment it returned nothing for is
    the most informative case in the sample.

    T5 forced this.  Of round 5's first 50, **ten have prose and zero
    annotations** — and all ten were genuinely asked: 17 output tokens
    each, `stop_reason: end_turn`, i.e. Claude answering "no features
    here" rather than failing.  That is 20 % of the review sample, and
    `--skip-unannotated` would hide exactly the treatments that decide
    the recall figure, making recall look better than it is.

    Rejected once, in T3a, for a reason that held then: brat's only
    advantage over `bin/treatment_dossier` is *editing*, and merge
    suspects had nothing to edit.  Here there is — the labels Claude
    missed.
    """

    def test_unannotated_ids_are_kept(self) -> None:
        got = select_treatment_ids(
            _FakeAnnDb([{'_id': 'taxon_a:Pileus:0'}]),
            ['taxon_a', 'taxon_b'], allow_unannotated=True,
            treatments_db=_FakeTreatmentsDb(['taxon_a', 'taxon_b']),
            warn_stream=io.StringIO(),
        )
        assert got == ['taxon_a', 'taxon_b']

    def test_works_when_nothing_is_annotated(self) -> None:
        """`--skip-unannotated` raises 'nothing to export' here.  For a
        recall review that is the interesting case, not an error.
        """
        got = select_treatment_ids(
            _FakeAnnDb([]), ['taxon_a', 'taxon_b'],
            allow_unannotated=True,
            treatments_db=_FakeTreatmentsDb(['taxon_a', 'taxon_b']),
            warn_stream=io.StringIO(),
        )
        assert got == ['taxon_a', 'taxon_b']

    def test_order_is_preserved(self) -> None:
        """The round file's order is the draw order, and T5's
        'first 50' rule depends on it surviving the export.
        """
        got = select_treatment_ids(
            _FakeAnnDb([{'_id': 'taxon_b:Pileus:0'}]),
            ['taxon_c', 'taxon_a', 'taxon_b'], allow_unannotated=True,
            treatments_db=_FakeTreatmentsDb(['taxon_a', 'taxon_b',
                                             'taxon_c']),
            warn_stream=io.StringIO(),
        )
        assert got == ['taxon_c', 'taxon_a', 'taxon_b']

    def test_typos_are_still_caught(self) -> None:
        """The relaxation must not surrender the original guarantee.
        An id absent from treatments_prose is a typo in either mode,
        and silently exporting nothing for it is how a review set
        quietly comes up short.
        """
        with pytest.raises(ValueError, match='typo'):
            select_treatment_ids(
                _FakeAnnDb([{'_id': 'taxon_a:Pileus:0'}]),
                ['taxon_a', 'taxon_nope'], allow_unannotated=True,
                treatments_db=_FakeTreatmentsDb(['taxon_a']),
                warn_stream=io.StringIO(),
            )

    def test_needs_the_treatments_db_to_tell_them_apart(self) -> None:
        with pytest.raises(ValueError, match='treatments_prose'):
            select_treatment_ids(
                _FakeAnnDb([]), ['taxon_a'], allow_unannotated=True,
                treatments_db=None, warn_stream=io.StringIO(),
            )

    def test_it_reports_what_it_kept(self) -> None:
        """Silence would leave the operator unable to tell an empty
        `.ann` from a broken export.
        """
        warn = io.StringIO()
        select_treatment_ids(
            _FakeAnnDb([]), ['taxon_a'], allow_unannotated=True,
            treatments_db=_FakeTreatmentsDb(['taxon_a']),
            warn_stream=warn,
        )
        assert 'taxon_a' in warn.getvalue()

    def test_the_two_flags_are_mutually_exclusive(self) -> None:
        """Skip and keep cannot both be honoured; asking for both is an
        operator error, not something to settle by precedence.
        """
        with pytest.raises(ValueError, match='mutually exclusive'):
            select_treatment_ids(
                _FakeAnnDb([]), ['taxon_a'],
                skip_unannotated=True, allow_unannotated=True,
                treatments_db=_FakeTreatmentsDb(['taxon_a']),
                warn_stream=io.StringIO(),
            )
