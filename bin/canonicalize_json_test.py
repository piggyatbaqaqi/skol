"""Tests for bin/canonicalize_json.py.

Proposal 2a of docs/plans/vocab-tree-narrowing.md (Trello AZeJMCR3).

Run with: python -m pytest bin/canonicalize_json_test.py -v
"""
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from canonicalize_json import (  # type: ignore[import]  # noqa: E402
    RECORD_FIELD,
    PinError,
    apply_updates,
    build_parser,
    check_pin,
    pin_record,
    plan_updates,
)
from treatments_to_structured.vocab_canonical import (  # noqa: E402
    source_hash,
)

PENDING = pytest.mark.xfail(
    strict=True, reason='canonicalize_json is a stub until P2a lands')

SYNONYMS = {'cap': ('pileus', 'Pileus')}
SPELLINGS = {'pileus': 'Pileus', 'color': 'Color'}
VERSION = {'synonyms': 'a' * 40, 'spellings': 'b' * 40}
OTHER = {'synonyms': 'c' * 40, 'spellings': 'b' * 40}


def _stale(doc_id: str = 't1') -> Dict[str, Any]:
    return {'_id': doc_id, '_rev': '1-x', 'title': 'kept',
            'json_annotated': {'Cap': {'colour': ['brown']}}}


def _fresh(doc_id: str = 't2') -> Dict[str, Any]:
    annotated = {'Pileus': {'Color': ['brown']}}
    return {'_id': doc_id, '_rev': '1-y',
            'json_annotated': annotated,
            'json_canonical': annotated, 'canonical_edits': [],
            'canonical_map_version': dict(VERSION),
            'canonical_source': source_hash(annotated)}


class FakeDB:
    def __init__(self) -> None:
        self.batches: List[List[Dict[str, Any]]] = []

    def update(self, docs: List[Dict[str, Any]]
               ) -> List[Tuple[bool, str, str]]:
        self.batches.append(docs)
        return [(True, d['_id'], '2-z') for d in docs]


def test_record_field() -> None:
    assert RECORD_FIELD == 'vocab_canonical'


@PENDING
class TestParser:
    def test_pin_flag(self) -> None:
        args = build_parser().parse_args(['--experiment', 'x', '--pin'])
        assert args.pin and args.experiment_name == 'x'

    def test_common_flags(self) -> None:
        args = build_parser().parse_args(['--dry-run', '--force'])
        assert args.dry_run and args.force

    def test_strict(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(['--pinn'])


@PENDING
class TestPin:
    def test_pin_record(self) -> None:
        record = {'_id': 'production', 'pipeline': 'v4_crf'}
        pinned = pin_record(record, VERSION)
        assert pinned[RECORD_FIELD] == VERSION
        assert pinned['pipeline'] == 'v4_crf'
        assert RECORD_FIELD not in record

    def test_match(self) -> None:
        assert check_pin({RECORD_FIELD: dict(VERSION)}, VERSION) == VERSION

    def test_no_pin_says_how_to_set_one(self) -> None:
        with pytest.raises(PinError, match='--pin'):
            check_pin({'_id': 'production'}, VERSION)

    def test_mismatch_names_both_versions(self) -> None:
        with pytest.raises(PinError) as caught:
            check_pin({RECORD_FIELD: dict(OTHER)}, VERSION)
        assert 'a' * 40 in str(caught.value)
        assert 'c' * 40 in str(caught.value)


@PENDING
class TestPlanUpdates:
    def run(self, docs: List[Dict[str, Any]], force: bool = False) -> Any:
        return plan_updates(docs, SYNONYMS, SPELLINGS, VERSION, force=force)

    def test_stale_is_updated_and_fresh_is_skipped(self) -> None:
        updates, counts = self.run([_stale(), _fresh()])
        assert [d['_id'] for d in updates] == ['t1']
        assert counts == {'checked': 2, 'updated': 1, 'fresh': 1,
                          'unusable': 0}

    def test_update_sets_the_canonical_fields(self) -> None:
        (doc,), _ = self.run([_stale()])
        assert doc['json_canonical'] == {'Pileus': {'Color': ['brown']}}
        assert [e['rule'] for e in doc['canonical_edits']] == [
            'synonym', 'spelling']
        assert doc['canonical_map_version'] == VERSION

    def test_everything_else_is_kept(self) -> None:
        stale = _stale()
        (doc,), _ = self.run([stale])
        assert doc['json_annotated'] == stale['json_annotated']
        assert (doc['_id'], doc['_rev'], doc['title']) == (
            't1', '1-x', 'kept')

    def test_input_is_not_modified(self) -> None:
        stale = _stale()
        before = copy.deepcopy(stale)
        self.run([stale])
        assert stale == before

    def test_other_version_is_stale(self) -> None:
        doc = _fresh()
        doc['canonical_map_version'] = dict(OTHER)
        updates, _ = self.run([doc])
        assert len(updates) == 1

    def test_unusable(self) -> None:
        updates, counts = self.run([{'_id': 't3'},
                                    {'_id': 't4', 'json_annotated': 'x'}])
        assert updates == []
        assert counts['unusable'] == 2

    def test_design_documents_are_skipped(self) -> None:
        _, counts = self.run([{'_id': '_design/x', 'json_annotated': {}}])
        assert counts['checked'] == 0

    def test_force_rewrites_fresh(self) -> None:
        updates, counts = self.run([_fresh()], force=True)
        assert len(updates) == 1
        assert counts['updated'] == 1


@PENDING
class TestApplyUpdates:
    def test_batches(self) -> None:
        db = FakeDB()
        docs = [_stale(f't{i}') for i in range(5)]
        assert apply_updates(db, docs, batch=2) == 5
        assert [len(b) for b in db.batches] == [2, 2, 1]

    def test_nothing_to_do(self) -> None:
        db = FakeDB()
        assert apply_updates(db, []) == 0
        assert db.batches == []
