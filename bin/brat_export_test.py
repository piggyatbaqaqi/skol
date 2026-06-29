"""Tests for bin/brat_export CLI helpers.

CouchDB and filesystem-touching paths are exercised through small
fakes; the live test (in a commit message) covers end-to-end.
"""

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
