"""Tests for fixes/fix_menus_pointer_records.py.

Run with: python -m pytest fixes/fix_menus_pointer_records_test.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from fix_menus_pointer_records import (  # noqa: E402
    MenusFix,
    apply_menus_fix,
    delete_keys,
    is_pointer,
    plan_menus_fix,
    pointer_keys_to_delete,
)

TREE = json.dumps({'version': 'v', 'tree': {'spores': {}}})
V41 = 'skol:ui:menus_production_v4_1'
STOPGAP = 'skol:ui:menus_production_v4_1_pointer'
LATEST = 'skol:ui:menus_latest'


class FakeRedis:
    def __init__(self, data: Dict[str, str]) -> None:
        self.data = dict(data)

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)


class FakeDb:
    def __init__(self, docs: List[Dict[str, Any]]) -> None:
        self.docs = {d['_id']: d for d in docs}
        self.saved: List[Dict[str, Any]] = []

    def __getitem__(self, key: str) -> Dict[str, Any]:
        return self.docs[key]

    def save(self, doc: Dict[str, Any]) -> Any:
        self.saved.append(doc)
        self.docs[doc['_id']] = doc
        return doc['_id'], '2-x'


def record(doc_id: str, menus: Optional[str]) -> Dict[str, Any]:
    keys = {} if menus is None else {'menus': menus}
    return {'_id': doc_id, 'redis_keys': keys}


def production_like() -> FakeRedis:
    """Production on 2026-11-13: the stopgap pointer, menus_latest
    pointing at a cron-built tree, and two tree keys."""
    return FakeRedis({
        V41: TREE,
        'skol:ui:menus_2026_11_12_22_00': TREE,
        STOPGAP: V41,
        LATEST: 'skol:ui:menus_2026_11_12_22_00',
    })


RECORDS = [
    record('production', STOPGAP),
    record('production_v4_1', STOPGAP),
    record('production_v3_hand', 'skol:ui:menus_production_v3_hand'),
    record('old_exp', LATEST),
    record('no_menus', None),
]


class TestIsPointer:
    def test_key_name_is_a_pointer(self) -> None:
        assert is_pointer(V41)

    def test_tree_json_is_not(self) -> None:
        assert not is_pointer(TREE)

    def test_missing_is_not(self) -> None:
        assert not is_pointer(None)


class TestPlanMenusFix:
    def test_pointer_records_get_their_target(self) -> None:
        fixes = plan_menus_fix(RECORDS, production_like())
        assert sorted(fixes, key=lambda f: f.doc_id) == [
            MenusFix('old_exp', LATEST, 'skol:ui:menus_2026_11_12_22_00'),
            MenusFix('production', STOPGAP, V41),
            MenusFix('production_v4_1', STOPGAP, V41),
        ]

    def test_tree_keys_missing_keys_and_no_menus_left_alone(self) -> None:
        ids = {f.doc_id for f in plan_menus_fix(RECORDS, production_like())}
        # production_v3_hand's key holds nothing; no_menus has no key.
        assert ids.isdisjoint({'production_v3_hand', 'no_menus'})


class TestPointerKeysToDelete:
    def test_old_pointers_and_menus_latest(self) -> None:
        r = production_like()
        fixes = plan_menus_fix(RECORDS, r)
        assert sorted(pointer_keys_to_delete(r, fixes, RECORDS)) == [
            LATEST, STOPGAP]

    def test_key_still_named_after_the_fix_is_kept(self) -> None:
        """A record that names a pointer we could not resolve (its
        target is gone) keeps naming it, so it is not deleted."""
        r = FakeRedis({STOPGAP: 'skol:ui:menus_gone', V41: TREE})
        records = [record('a', STOPGAP)]
        fixes = plan_menus_fix(records, r)
        assert STOPGAP not in pointer_keys_to_delete(r, fixes, records)


class TestApplyMenusFix:
    def test_rewrites_only_menus(self) -> None:
        db = FakeDb([{'_id': 'production', 'status': 'deployed',
                      'redis_keys': {'menus': STOPGAP,
                                     'embedding': 'skol:embedding:x'}}])
        n = apply_menus_fix(db, [MenusFix('production', STOPGAP, V41)])
        assert n == 1
        doc = db.docs['production']
        assert doc['redis_keys'] == {'menus': V41,
                                     'embedding': 'skol:embedding:x'}
        assert doc['status'] == 'deployed'


class ClusterFakeRedis(FakeRedis):
    """Rejects a multi-key DELETE, as a Redis Cluster does when the
    keys hash to different slots (what happened on tsqali, 2026-11-13)."""

    def delete(self, *keys: str) -> int:
        if len(keys) > 1:
            raise RuntimeError("CROSSSLOT Keys in request don't hash to "
                               "the same slot")
        return int(self.data.pop(keys[0], None) is not None)


@pytest.mark.xfail(strict=True, reason="delete_keys not written")
class TestDeleteKeys:
    def test_deletes_each_key_on_a_cluster(self) -> None:
        r = ClusterFakeRedis({STOPGAP: V41, LATEST: 'x', V41: TREE})
        assert delete_keys(r, [STOPGAP, LATEST]) == 2
        assert set(r.data) == {V41}

    def test_missing_key_is_not_counted(self) -> None:
        r = ClusterFakeRedis({LATEST: 'x'})
        assert delete_keys(r, [STOPGAP, LATEST]) == 1
