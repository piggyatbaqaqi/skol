#!/usr/bin/env python3
"""Tests for ``fixes/retire_merge_skips``.

This deletes CouchDB documents, so the tests are mostly about what it
must refuse to do.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retire_merge_skips import (  # type: ignore[import]  # noqa: E402
    STATUS_SKIPPED,
    classify,
    load_snapshot,
    retire,
)

pytestmark = pytest.mark.xfail(
    raises=NotImplementedError, strict=True,
    reason='retire: implementation follows test confirmation',
)


def _skip(tid: str, score: int) -> dict:
    return {'_id': tid, 'treatment_id': tid, 'status': STATUS_SKIPPED,
            'metric_value': score, 'threshold': 10,
            'metric_name': 'n_terms_above_5'}


class _FakeRow:
    def __init__(self, doc: dict) -> None:
        self.id, self.doc = doc['_id'], doc


class _FakeStatusDb:
    def __init__(self, docs: list) -> None:
        self.docs = {d['_id']: d for d in docs}
        self.deleted: list = []

    def view(self, _name: str, **_kw: object) -> object:
        rows = [_FakeRow(d) for d in self.docs.values()]

        class _V:
            def __init__(self, r: list) -> None:
                self.rows = r

            def __iter__(self) -> object:
                return iter(self.rows)
        return _V(rows)

    def delete(self, doc: dict) -> None:
        self.deleted.append(doc['_id'])
        del self.docs[doc['_id']]


class TestLoadSnapshot:
    def test_reads_id_and_score(self, tmp_path: Path) -> None:
        p = tmp_path / 'snap.tsv'
        p.write_text('treatment_id\tn_terms_above_5\n'
                     'taxon_a\t12\ntaxon_b\t47\n', encoding='utf-8')
        assert load_snapshot(p) == {'taxon_a': 12, 'taxon_b': 47}

    def test_a_missing_snapshot_is_fatal(self, tmp_path: Path) -> None:
        """Running without it would delete scores recorded nowhere
        else -- the exact loss plan F3 exists to prevent.
        """
        with pytest.raises(FileNotFoundError):
            load_snapshot(tmp_path / 'nope.tsv')


class TestClassify:
    def test_below_the_new_threshold_retires(self) -> None:
        assert classify(_skip('taxon_a', 12), 15,
                        {'taxon_a': 12}) == 'retire'

    def test_at_the_threshold_is_kept(self) -> None:
        """`>=` is the filter's own comparison; a doc scoring exactly
        the threshold is still a suspect.
        """
        assert classify(_skip('taxon_a', 15), 15,
                        {'taxon_a': 15}) == 'keep'

    def test_above_the_threshold_is_kept(self) -> None:
        assert classify(_skip('taxon_a', 47), 15,
                        {'taxon_a': 47}) == 'keep'

    def test_a_doc_missing_from_the_snapshot_is_refused(self) -> None:
        """Not skipped, not deleted: refused.  Its score would survive
        nowhere.
        """
        assert classify(_skip('taxon_a', 12), 15, {}) == 'unsnapshotted'

    def test_a_non_skip_doc_is_left_alone(self) -> None:
        """The status DB also holds success/partial/error docs for
        every annotated treatment.  Touching those would destroy the
        annotation record.
        """
        doc = {'_id': 'taxon_a', 'status': 'success',
               'annotation_count': 12}
        assert classify(doc, 15, {'taxon_a': 12}) == 'not-a-skip'

    def test_the_snapshot_wins_over_the_doc(self) -> None:
        """The snapshot was taken before anything could overwrite the
        score; the doc's own copy may since have been rewritten.
        """
        assert classify(_skip('taxon_a', 99), 15,
                        {'taxon_a': 12}) == 'retire'


class TestRetire:
    def test_deletes_only_what_is_below_the_threshold(self) -> None:
        db = _FakeStatusDb([_skip('taxon_a', 12), _skip('taxon_b', 47)])
        snap = {'taxon_a': 12, 'taxon_b': 47}
        n, kept, refused = retire(db, 15, snap, dry_run=False)
        assert (n, kept, refused) == (1, 1, [])
        assert db.deleted == ['taxon_a']

    def test_dry_run_deletes_nothing(self) -> None:
        """2 112 documents is not a number to delete on a first try."""
        db = _FakeStatusDb([_skip('taxon_a', 12)])
        n, _, _ = retire(db, 15, {'taxon_a': 12}, dry_run=True)
        assert n == 1
        assert db.deleted == []

    def test_annotation_status_docs_are_untouched(self) -> None:
        db = _FakeStatusDb([
            _skip('taxon_a', 12),
            {'_id': 'taxon_b', 'status': 'success', 'annotation_count': 9},
        ])
        retire(db, 15, {'taxon_a': 12, 'taxon_b': 9}, dry_run=False)
        assert db.deleted == ['taxon_a']
        assert 'taxon_b' in db.docs

    def test_unsnapshotted_docs_are_refused_not_deleted(self) -> None:
        db = _FakeStatusDb([_skip('taxon_a', 12), _skip('taxon_b', 12)])
        n, _, refused = retire(db, 15, {'taxon_a': 12}, dry_run=False)
        assert n == 1
        assert refused == ['taxon_b']
        assert db.deleted == ['taxon_a']

    def test_rerunning_retires_nothing(self) -> None:
        db = _FakeStatusDb([_skip('taxon_b', 47)])
        assert retire(db, 15, {'taxon_b': 47}, dry_run=False)[0] == 0
