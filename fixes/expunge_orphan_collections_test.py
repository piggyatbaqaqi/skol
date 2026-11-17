"""Tests for fixes/expunge_orphan_collections.py.

Django tests synced their Collections into the developer's local
CouchDB until 2026-11-17 (docs/django-test-isolation.md).  A collection
document is an orphan when its collection_id is in no Django database
that still exists: test databases are destroyed after each run.  The
dev database (django/db.sqlite3) is the list of live collections.
Owner names cannot decide it: 'niemand' and 'jsmith' are both fixture
users and real dev accounts.
"""
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import expunge_orphan_collections as eoc  # type: ignore[import]  # noqa: E402

COLLECTIONS = 'skol_collections_dev'
HISTORY = 'skol_collections_history_dev'


class Row:
    def __init__(self, doc: Dict[str, Any]) -> None:
        self.id = doc['_id']
        self.doc = doc
        self.value = {'rev': doc['_rev']}


class FakeDb:
    """The couchdb.Database calls the fix uses: _all_docs and update."""

    def __init__(self, docs: List[Dict[str, Any]]) -> None:
        self.docs = {d['_id']: dict(d, _rev='1-a') for d in docs}
        self.updates: List[List[Dict[str, Any]]] = []

    def view(self, name: str, **kwargs: Any) -> Iterator[Row]:
        assert name == '_all_docs'
        return iter([Row(d) for d in self.docs.values()])

    def update(self, docs: List[Dict[str, Any]]) -> List[Any]:
        self.updates.append(docs)
        for d in docs:
            assert d.get('_deleted') is True
            assert self.docs[d['_id']]['_rev'] == d['_rev']
            del self.docs[d['_id']]
        return [(True, d['_id'], '2-b') for d in docs]


def _coll(cid: int, owner: str) -> Dict[str, Any]:
    return {'_id': f'collection_{cid}', 'collection_id': cid,
            'owner': {'user_id': 1, 'username': owner}}


def _dbs() -> Dict[str, FakeDb]:
    return {
        COLLECTIONS: FakeDb([
            _coll(111111111, 'niemand'),      # live
            _coll(222222222, 'niemand'),      # orphan: hand-check copy
            _coll(333333333, 'testuser'),     # orphan
            _coll(444444444, 'jsmith'),       # orphan
            {'_id': '_design/collections', 'views': {}},
        ]),
        HISTORY: FakeDb([
            {'_id': 'collection_111111111_20261101T000000'},
            {'_id': 'collection_333333333_20261101T000000'},
            {'_id': 'collection_333333333_20261102T000000'},
        ]),
    }


LIVE = {111111111}


class TestPlan:
    def test_live_ids_kept_others_orphaned(self) -> None:
        plan = eoc.make_plan(_dbs(), LIVE)
        assert sorted(i for i, _ in plan.orphans[COLLECTIONS]) == [
            'collection_222222222', 'collection_333333333',
            'collection_444444444']
        assert plan.kept[COLLECTIONS] == 1

    def test_history_follows_its_collection(self) -> None:
        plan = eoc.make_plan(_dbs(), LIVE)
        assert sorted(i for i, _ in plan.orphans[HISTORY]) == [
            'collection_333333333_20261101T000000',
            'collection_333333333_20261102T000000']
        assert plan.kept[HISTORY] == 1

    def test_design_docs_never_planned(self) -> None:
        plan = eoc.make_plan(_dbs(), LIVE)
        assert all(not i.startswith('_design/')
                   for ids in plan.orphans.values() for i, _ in ids)

    def test_collection_id_of(self) -> None:
        assert eoc.collection_id_of('collection_333333333') == 333333333
        assert eoc.collection_id_of(
            'collection_333333333_20261101T000000') == 333333333
        assert eoc.collection_id_of('_design/collections') is None
        assert eoc.collection_id_of('something_else') is None


class TestDryRunAndApply:
    def test_dry_run_deletes_nothing_and_reports_owners(self) -> None:
        dbs = _dbs()
        plan = eoc.make_plan(dbs, LIVE)
        text = eoc.report(plan)
        assert all(not db.updates for db in dbs.values())
        assert 'niemand: 1' in text
        assert 'testuser: 1' in text
        assert 'jsmith: 1' in text
        assert f'{COLLECTIONS}: 3 orphaned, 1 kept' in text
        assert f'{HISTORY}: 2 orphaned, 1 kept' in text

    def test_apply_deletes_exactly_the_orphans(self) -> None:
        dbs = _dbs()
        plan = eoc.make_plan(dbs, LIVE)
        assert eoc.apply_plan(plan, dbs, batch=2) == 5
        assert sorted(dbs[COLLECTIONS].docs) == [
            '_design/collections', 'collection_111111111']
        assert sorted(dbs[HISTORY].docs) == [
            'collection_111111111_20261101T000000']
        # Batched: three collection deletions in batches of two.
        assert [len(b) for b in dbs[COLLECTIONS].updates] == [2, 1]


class TestLiveIdsMustBeTrustworthy:
    def _django_db(self, tmp_path: Path, ids: List[int]) -> str:
        path = tmp_path / 'db.sqlite3'
        con = sqlite3.connect(path)
        con.execute('create table search_collection (collection_id integer)')
        con.executemany('insert into search_collection values (?)',
                        [(i,) for i in ids])
        con.commit()
        con.close()
        return str(path)

    def test_reads_the_collection_ids(self, tmp_path: Path) -> None:
        path = self._django_db(tmp_path, [111111111, 555555555])
        assert eoc.load_live_ids(path) == {111111111, 555555555}

    def test_empty_database_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(eoc.RefusedError):
            eoc.load_live_ids(self._django_db(tmp_path, []))

    def test_missing_database_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(eoc.RefusedError):
            eoc.load_live_ids(str(tmp_path / 'absent.sqlite3'))

    def test_database_without_the_table_refuses(self, tmp_path: Path) -> None:
        path = tmp_path / 'other.sqlite3'
        sqlite3.connect(path).close()
        with pytest.raises(eoc.RefusedError):
            eoc.load_live_ids(str(path))


class TestParser:
    def test_defaults_are_a_dry_run_against_the_dev_db(self) -> None:
        args = eoc.build_parser().parse_args([])
        assert args.apply is False
        assert args.django_db == eoc.DEFAULT_DJANGO_DB

    def test_apply_and_common_flags(self) -> None:
        args = eoc.build_parser().parse_args(
            ['--apply', '--django-db', 'x.sqlite3', '--verbosity', '2'])
        assert (args.apply, args.django_db, args.verbosity) == (
            True, 'x.sqlite3', 2)

    def test_rejects_unknown_flags(self) -> None:
        parser = eoc.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['--aply'])
