"""Tests for fixes/rekey_structured_treatments.py.

Run with: python -m pytest fixes/rekey_structured_treatments_test.py -v
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rekey_structured_treatments import (  # noqa: E402
    apply_rekey,
    legacy_structured_id,
    plan_rekey,
    target_dbs,
)
from treatments_json_translator import build_llm_input_text  # noqa: E402
from treatments_to_structured.modal_inference import (  # noqa: E402
    taxon_doc_id,
)


def prose(doc_id: str, description: Optional[str] = 'Pileus convex.',
          diagnosis: Optional[str] = None,
          treatment: str = 'Russula x') -> Dict[str, Any]:
    return {'_id': doc_id, 'treatment': treatment,
            'description': description, 'diagnosis': diagnosis,
            'etymology': f'etymology of {doc_id}'}


def structured(doc_id: str, **extra: Any) -> Dict[str, Any]:
    return {'_id': doc_id, '_rev': '1-x', 'treatment': 'Russula x',
            'json_annotated': {'pileus': {'shape': ['convex']}},
            'json_valid': True, 'provenance': {'backend': 'modal'}, **extra}


class FakeDB:
    """The slice of couchdb.Database the fix uses."""

    def __init__(self, docs: List[Dict[str, Any]],
                 fail_saving: Optional[Set[str]] = None) -> None:
        self.docs = {d['_id']: dict(d) for d in docs}
        self.fail_saving = fail_saving or set()
        self.saved: List[Dict[str, Any]] = []
        self.deleted: List[str] = []

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.docs

    def __iter__(self) -> Iterator[str]:
        return iter(list(self.docs))

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        return dict(self.docs[doc_id])

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return dict(self.docs[doc_id]) if doc_id in self.docs else None

    def save(self, doc: Dict[str, Any]) -> None:
        if doc['_id'] in self.fail_saving:
            raise RuntimeError(f'conflict on {doc["_id"]}')
        if '_rev' in doc:
            raise AssertionError('a new copy must not carry a _rev')
        self.saved.append(dict(doc))
        self.docs[doc['_id']] = dict(doc, _rev='1-new')

    def delete(self, doc: Dict[str, Any]) -> None:
        self.deleted.append(doc['_id'])
        del self.docs[doc['_id']]


def quiet(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class TestLegacyStructuredId:

    def test_matches_how_treatments_to_json_named_it(self) -> None:
        doc = prose('p1', description='Pileus convex.',
                    diagnosis='Differs from R. y.')
        assert legacy_structured_id(doc) == taxon_doc_id(
            'Russula x', build_llm_input_text(doc))

    def test_ignores_sections_the_prose_id_includes(self) -> None:
        """The prose _id hashes all eleven sections; the old structured
        name only the treatment and description-plus-diagnosis text."""
        a, b = prose('p1'), prose('p2')
        a['etymology'], b['etymology'] = 'one', 'other'
        assert legacy_structured_id(a) == legacy_structured_id(b)


class TestPlanRekey:

    def test_one_to_one(self) -> None:
        p = prose('taxon_p1')
        old = legacy_structured_id(p)
        plan = plan_rekey([p], {old})
        assert plan['copies'] == [(old, 'taxon_p1')]
        assert plan['retire'] == {old}
        assert plan['already'] == [] and plan['missing'] == []

    def test_identical_text_gets_a_copy_per_prose_treatment(self) -> None:
        a, b = prose('taxon_a'), prose('taxon_b')
        old = legacy_structured_id(a)
        assert old == legacy_structured_id(b)
        plan = plan_rekey([a, b], {old})
        assert sorted(plan['copies']) == [(old, 'taxon_a'), (old, 'taxon_b')]
        assert plan['retire'] == {old}

    def test_already_named_by_prose_id(self) -> None:
        p = prose('taxon_p1')
        plan = plan_rekey([p], {'taxon_p1', legacy_structured_id(p)})
        assert plan['already'] == ['taxon_p1']
        assert plan['copies'] == []
        assert plan['retire'] == {legacy_structured_id(p)}

    def test_untranslatable_is_not_missing(self) -> None:
        plan = plan_rekey([prose('taxon_e', description=''),
                           prose('taxon_n', description='... --')], set())
        assert plan['untranslatable'] == 2
        assert plan['missing'] == [] and plan['copies'] == []

    def test_translatable_without_json_is_missing(self) -> None:
        plan = plan_rekey([prose('taxon_p1')], set())
        assert plan['missing'] == ['taxon_p1']

    def test_unrelated_structured_docs_are_left_alone(self) -> None:
        p = prose('taxon_p1')
        plan = plan_rekey([p], {legacy_structured_id(p), 'taxon_unknown'})
        assert 'taxon_unknown' not in plan['retire']


class TestApplyRekey:

    def test_copies_content_then_retires_old_names(self) -> None:
        a, b = prose('taxon_a'), prose('taxon_b')
        old = legacy_structured_id(a)
        db = FakeDB([structured(old)])
        counts = quiet(apply_rekey, db, plan_rekey([a, b], {old}))
        assert counts == {'copied': 2, 'failed': 0, 'deleted': 1}
        assert sorted(db.docs) == ['taxon_a', 'taxon_b']
        for new in ('taxon_a', 'taxon_b'):
            copy = db.docs[new]
            assert copy['json_annotated'] == {'pileus': {'shape': ['convex']}}
            assert copy['provenance'] == {'backend': 'modal'}

    def test_copies_happen_before_any_delete(self) -> None:
        p = prose('taxon_p1')
        old = legacy_structured_id(p)
        order: List[str] = []

        class Recording(FakeDB):
            def save(self, doc: Dict[str, Any]) -> None:
                order.append('save')
                super().save(doc)

            def delete(self, doc: Dict[str, Any]) -> None:
                order.append('delete')
                super().delete(doc)

        db = Recording([structured(old)])
        quiet(apply_rekey, db, plan_rekey([p], {old}))
        assert order == ['save', 'delete']

    def test_a_failed_copy_deletes_nothing(self) -> None:
        a, b = prose('taxon_a'), prose('taxon_b')
        old = legacy_structured_id(a)
        db = FakeDB([structured(old)], fail_saving={'taxon_b'})
        counts = quiet(apply_rekey, db, plan_rekey([a, b], {old}))
        assert counts == {'copied': 1, 'failed': 1, 'deleted': 0}
        assert old in db.docs

    def test_dry_run_writes_nothing(self) -> None:
        p = prose('taxon_p1')
        old = legacy_structured_id(p)
        db = FakeDB([structured(old)])
        counts = quiet(apply_rekey, db, plan_rekey([p], {old}), dry_run=True)
        assert counts == {'copied': 1, 'failed': 0, 'deleted': 1}
        assert db.saved == [] and db.deleted == []

    def test_rerun_has_nothing_to_do(self) -> None:
        p = prose('taxon_p1')
        old = legacy_structured_id(p)
        db = FakeDB([structured(old)])
        quiet(apply_rekey, db, plan_rekey([p], {old}))
        again = plan_rekey([p], set(db.docs))
        assert again['copies'] == [] and again['retire'] == set()
        assert again['already'] == ['taxon_p1']


class TestTargetDbs:

    CONFIG = {
        'treatments_prose_db_name':
            'skol_exp_production_v4_1_02_00_treatments_prose',
        'treatments_structured_db_name':
            'skol_exp_production_v4_1_03_00_treatments_structured',
    }

    def test_from_the_experiment(self) -> None:
        assert target_dbs('production_v4_1', self.CONFIG) == {
            'prose': 'skol_exp_production_v4_1_02_00_treatments_prose',
            'structured':
                'skol_exp_production_v4_1_03_00_treatments_structured',
        }

    def test_refuses_without_an_experiment(self) -> None:
        with pytest.raises(ValueError):
            target_dbs(None, self.CONFIG)
