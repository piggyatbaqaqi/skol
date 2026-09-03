#!/usr/bin/env python3
"""Tests for ``bin/build_canonical_labels.py``.

The script is thin I/O over
``treatments_to_structured.canonical_annotation``; what needs testing
here is the fan-out and the tally, because the tally is what the
operator reads to decide whether the pass did what the plan predicted
(99 sub-attribute, 48 compound, 36 case-fold, 100 condition).
"""

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_canonical_labels import (  # type: ignore[import]  # noqa: E402
    canonicalize_all,
    orphaned_ids,
)


KNOWN = {
    'colony': 'Colony',
    'colony reverse': 'Colony reverse',
    'ascomata': 'Ascomata',
    'gamma conidia': 'Gamma conidia',
    'beta conidia': 'Beta conidia',
    'partial veil': 'Partial veil',
    'partial veil microscopic': 'Partial veil microscopic',
}
ESTABLISHED = {
    'colony': 'Colony',
    'ascomata': 'Ascomata',
    'gamma conidia': 'Gamma conidia',
    'beta conidia': 'Beta conidia',
}
PROTECTED = frozenset({'partial veil microscopic'})
SOURCE = 'skol_exp_production_v4_02_50_features_candidate'


def _ann(label: str, tid: str = 't1', start: int = 0,
         text: str = 'Colonies reaching 40 mm.') -> Dict[str, Any]:
    return {
        'feature_label': label, 'treatment_id': tid, 'start': start,
        'end': start + 20, 'field': 'description', 'source_text': text,
    }


def _run(annotations):
    return canonicalize_all(
        annotations, known=KNOWN, established=ESTABLISHED,
        protected=PROTECTED, source_db=SOURCE)


class TestCanonicalizeAll:
    def test_counts_inputs_and_outputs_separately(self) -> None:
        """A compound turns one raw annotation into two records, so the
        two counts are not interchangeable and the report must not
        conflate them."""
        records, tally = _run([
            _ann('Ascomata'),
            _ann('Gamma and beta conidia', start=40),
        ])
        assert tally['raw'] == 2
        assert tally['records'] == 3
        assert len(records) == 3

    @pytest.mark.parametrize('label,key', [
        ('Colony Reverse', 'case_fold'),
        ('Colony on MEA', 'condition'),
        ('Gamma and beta conidia', 'compound'),
        ('Ascomata height', 'sub_attribute'),
    ])
    def test_each_rule_is_tallied_by_name(
            self, label: str, key: str) -> None:
        _, tally = _run([_ann(label)])
        assert tally[key] >= 1

    def test_untouched_labels_are_tallied_as_unchanged(self) -> None:
        _, tally = _run([_ann('Ascomata')])
        assert tally['unchanged'] == 1
        assert tally['case_fold'] == 0

    def test_protected_targets_are_tallied_and_left_whole(self) -> None:
        records, tally = _run([_ann('Partial veil microscopic')])
        assert tally['protected'] == 1
        assert records[0]['feature_label'] == 'Partial veil microscopic'
        assert records[0]['attribute_path'] == ['Partial veil microscopic']

    def test_an_unusable_annotation_is_dropped_and_counted(self) -> None:
        """The pass walks a whole database and must not die on one bad
        row, but it must not hide it either."""
        _, tally = _run([_ann('   '), _ann('Ascomata')])
        assert tally['dropped'] == 1
        assert tally['records'] == 1

    def test_records_carry_their_source_for_the_diff(self) -> None:
        records, _ = _run([_ann('Colony on MEA')])
        assert records[0]['source_db'] == SOURCE
        assert records[0]['raw_label'] == 'Colony on MEA'

    def test_duplicate_ids_are_collapsed(self) -> None:
        """Two raw labels that canonicalize to the same label on the
        same span produce one document, not a conflict.  `Colonies` and
        `Colony` on one span is exactly that case."""
        records, tally = _run([
            _ann('Colony', start=0), _ann('COLONY', start=0),
        ])
        assert len(records) == 1
        assert tally['duplicate'] == 1


@pytest.mark.xfail(strict=True, reason='orphaned_ids is unwritten')
class TestOrphanedIds:
    """A derived DB must own its contents.

    The first build wrote `Conidiogenous Cells`; the fix made the
    frequent spelling win, so the second build wrote
    `Conidiogenous cells` under a different `_id` and left 961 stale
    documents behind — both spellings live in the DB, and a reader
    cannot tell which is current.  A rebuild has to delete what it did
    not write.

    This is safe here precisely *because* the DB is derived: nothing is
    lost that `bin/build_canonical_labels` cannot reproduce from the
    candidate DB.
    """

    def test_stale_documents_are_named_for_deletion(self) -> None:
        assert orphaned_ids(
            existing={'a', 'b', 'stale'}, produced={'a', 'b'},
        ) == ['stale']

    def test_a_clean_rebuild_deletes_nothing(self) -> None:
        assert orphaned_ids(existing={'a'}, produced={'a', 'b'}) == []

    def test_design_documents_are_never_deleted(self) -> None:
        """Views belong to whoever created them, not to the build."""
        assert orphaned_ids(
            existing={'_design/by_label', 'stale'}, produced=set(),
        ) == ['stale']

    def test_the_order_is_deterministic(self) -> None:
        assert orphaned_ids(
            existing={'c', 'a', 'b'}, produced=set()) == ['a', 'b', 'c']
