"""Tests for bin/assess_treatments.py.

The detector verdicts are checked against the same ground truth as
tests/pathologies_test.py: tests/fixtures/pathologies.json's
poster_children (must be clean) and pathologies (must carry exactly
their expected flags).

Run with: python -m pytest bin/assess_treatments_test.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from assess_treatments import (  # type: ignore[import]  # noqa: E402
    DETECTOR_SOURCES,
    assess,
    assess_all,
    checks_version,
    needs_assessment,
    quality_db_name,
    verdict_doc,
)

FIXTURE = json.loads(
    (Path(__file__).resolve().parent.parent / 'tests' / 'fixtures'
     / 'pathologies.json').read_text())
ANCHOR = [{'kind': 'pdf', 'page': '1', 'label': '1'}]
NOW = '2026-10-21T00:00:00+00:00'


def fixture_treatment(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Treatment doc for a fixture entry, built as pathologies_test does."""
    return {
        'description': entry.get('description', ''),
        'diagnosis': entry.get('diagnosis', ''),
        'synthetic_nomenclature': entry.get('synthetic_nomenclature', False),
        'description_spans': entry.get('description_spans', []),
        'source_anchors': entry.get('source_anchors', ANCHOR),
    }


def prefixes(flags: List[str]) -> set:
    return {flag.split('=')[0] for flag in flags}


def treatment(doc_id: str, description: str = 'Pileus 40-60 mm, convex.',
              diagnosis: str = '') -> Dict[str, Any]:
    return {'_id': doc_id, 'description': description,
            'diagnosis': diagnosis, 'source_anchors': ANCHOR}


class FakeDB:
    """The slice of couchdb.Database assess_all uses."""

    def __init__(self, docs: Optional[List[Dict[str, Any]]] = None) -> None:
        self.docs = {d['_id']: d for d in docs or []}
        self.saved: List[Dict[str, Any]] = []

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.docs

    def __iter__(self) -> Iterator[str]:
        return iter(list(self.docs))

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.docs.get(doc_id)

    def save(self, doc: Dict[str, Any]) -> None:
        self.saved.append(dict(doc))
        self.docs[doc['_id']] = dict(doc)


# ---------------------------------------------------------------------------
# checks_version
# ---------------------------------------------------------------------------


class TestChecksVersion:

    def test_covers_the_detector_sources(self) -> None:
        names = {p.name for p in DETECTOR_SOURCES}
        assert names == {'triage_signals.py', 'merge_metric.py'}
        assert all(p.exists() for p in DETECTOR_SOURCES)

    def test_is_a_stable_hex_string(self) -> None:
        version = checks_version()
        assert version == checks_version()
        int(version, 16)

    def test_changes_when_a_detector_changes(self, tmp_path: Path) -> None:
        a, b = tmp_path / 'a.py', tmp_path / 'b.py'
        a.write_text('x = 1\n')
        b.write_text('y = 2\n')
        before = checks_version((a, b))
        b.write_text('y = 3\n')
        assert checks_version((a, b)) != before


# ---------------------------------------------------------------------------
# assess — against the fixture ground truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'entry', FIXTURE['poster_children'],
    ids=[e['id'][:22] for e in FIXTURE['poster_children']])
def test_poster_children_are_clean(entry: Dict[str, Any]) -> None:
    verdict = assess(fixture_treatment(entry), authored_binomial=False)
    assert verdict == {'clean': True, 'flags': [], 'usable_text': True,
                       'gn_evaluated': True}


@pytest.mark.parametrize(
    'entry', [e for e in FIXTURE['pathologies'] if e.get('expected_flags')],
    ids=[e['id'][:22] for e in FIXTURE['pathologies']
         if e.get('expected_flags')])
def test_pathologies_carry_their_expected_flags(
        entry: Dict[str, Any]) -> None:
    verdict = assess(
        fixture_treatment(entry),
        authored_binomial=entry.get('authored_binomial_in_desc', False))
    assert verdict['clean'] is False
    assert prefixes(verdict['flags']) == set(entry['expected_flags'])


class TestAssessGates:

    def poster_child(self) -> Dict[str, Any]:
        return fixture_treatment(FIXTURE['poster_children'][0])

    def test_unevaluated_gn_check_is_not_clean(self) -> None:
        """A service outage must not pass treatments as clean."""
        verdict = assess(self.poster_child(), authored_binomial=None)
        assert verdict['clean'] is False
        assert verdict['gn_evaluated'] is False
        assert verdict['flags'] == []

    def test_text_free_treatment_is_not_clean(self) -> None:
        """An empty treatment fires no detector at all."""
        verdict = assess(treatment('t', description='', diagnosis=''),
                         authored_binomial=False)
        assert verdict['flags'] == []
        assert verdict['usable_text'] is False
        assert verdict['clean'] is False

    def test_diagnosis_alone_is_usable_text(self) -> None:
        verdict = assess(
            treatment('t', description='', diagnosis='Differs from X '
                      'by its larger basidiospores.'),
            authored_binomial=False)
        assert verdict['usable_text'] is True

    def test_flags_keep_their_values(self) -> None:
        merged = next(e for e in FIXTURE['pathologies']
                      if '§6:merge_metric' in e.get('expected_flags', []))
        verdict = assess(fixture_treatment(merged), authored_binomial=False)
        metric = [f for f in verdict['flags']
                  if f.startswith('§6:merge_metric')]
        assert metric and '=' in metric[0]


# ---------------------------------------------------------------------------
# verdict_doc / needs_assessment / quality_db_name
# ---------------------------------------------------------------------------


class TestVerdictDoc:

    def test_shape(self) -> None:
        verdict = {'clean': False, 'flags': ['§10:mid_sentence'],
                   'usable_text': True, 'gn_evaluated': True}
        assert verdict_doc('taxon_a', verdict, 'v1', NOW) == {
            '_id': 'taxon_a', 'clean': False,
            'flags': ['§10:mid_sentence'], 'usable_text': True,
            'gn_evaluated': True, 'checks_version': 'v1',
            'checked_at': NOW,
        }


class TestNeedsAssessment:

    def test_missing_verdict(self) -> None:
        assert needs_assessment(None, 'v2') is True

    def test_current_verdict_is_skipped(self) -> None:
        assert needs_assessment({'checks_version': 'v2'}, 'v2') is False

    def test_stale_verdict(self) -> None:
        assert needs_assessment({'checks_version': 'v1'}, 'v2') is True

    def test_force(self) -> None:
        assert needs_assessment({'checks_version': 'v2'}, 'v2',
                                force=True) is True


class TestQualityDbName:

    def test_from_experiment_record(self) -> None:
        name = 'skol_exp_production_v4_1_02_70_treatment_quality'
        assert quality_db_name({'treatment_quality_db_name': name}) == name

    def test_refuses_without_one(self) -> None:
        """No dev default: the verdicts belong to an experiment."""
        for config in ({}, {'treatment_quality_db_name': ''}):
            with pytest.raises(ValueError):
                quality_db_name(config)


# ---------------------------------------------------------------------------
# assess_all
# ---------------------------------------------------------------------------


class TestAssessAll:

    def test_saves_one_verdict_per_treatment(self) -> None:
        db = FakeDB()
        counts = assess_all(
            [treatment('a'), treatment('empty', description='')],
            db, gn_check=lambda text: False, version='v1', checked_at=NOW)
        assert set(db.docs) == {'a', 'empty'}
        assert db.docs['empty']['clean'] is False
        assert db.docs['a']['checks_version'] == 'v1'
        assert counts['scanned'] == 2 and counts['assessed'] == 2

    def test_counts_clean(self) -> None:
        clean = dict(fixture_treatment(FIXTURE['poster_children'][0]),
                     _id='pc')
        counts = assess_all([clean, treatment('empty', description='')],
                            FakeDB(), gn_check=lambda text: False,
                            version='v1', checked_at=NOW)
        assert counts['clean'] == 1

    def test_gn_check_sees_the_description(self) -> None:
        seen: List[str] = []

        def gn(text: str) -> Optional[bool]:
            seen.append(text)
            return False

        assess_all([treatment('a', description='Pileus convex.')],
                   FakeDB(), gn_check=gn, version='v1', checked_at=NOW)
        assert seen == ['Pileus convex.']

    def test_skips_current_and_reassesses_stale(self) -> None:
        db = FakeDB([
            {'_id': 'cur', '_rev': '1-a', 'checks_version': 'v2'},
            {'_id': 'old', '_rev': '1-b', 'checks_version': 'v1'},
        ])
        counts = assess_all([treatment('cur'), treatment('old')], db,
                            gn_check=lambda text: False, version='v2',
                            checked_at=NOW)
        assert [d['_id'] for d in db.saved] == ['old']
        assert counts['skipped'] == 1 and counts['assessed'] == 1

    def test_update_keeps_the_revision(self) -> None:
        db = FakeDB([{'_id': 'old', '_rev': '3-c', 'checks_version': 'v1'}])
        assess_all([treatment('old')], db, gn_check=lambda text: False,
                   version='v2', checked_at=NOW)
        assert db.saved[0]['_rev'] == '3-c'

    def test_force_reassesses_current(self) -> None:
        db = FakeDB([{'_id': 'cur', '_rev': '1-a', 'checks_version': 'v2'}])
        assess_all([treatment('cur')], db, gn_check=lambda text: False,
                   version='v2', checked_at=NOW, force=True)
        assert [d['_id'] for d in db.saved] == ['cur']

    def test_dry_run_saves_nothing(self) -> None:
        db = FakeDB()
        counts = assess_all([treatment('a')], db,
                            gn_check=lambda text: False, version='v1',
                            checked_at=NOW, dry_run=True)
        assert db.saved == []
        assert counts['assessed'] == 1

    def test_skips_design_docs(self) -> None:
        db = FakeDB()
        counts = assess_all([treatment('_design/views'), treatment('a')],
                            db, gn_check=lambda text: False, version='v1',
                            checked_at=NOW)
        assert set(db.docs) == {'a'}
        assert counts['scanned'] == 1
