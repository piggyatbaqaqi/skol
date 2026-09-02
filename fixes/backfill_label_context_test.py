#!/usr/bin/env python3
"""Tests for ``fixes.backfill_label_context``.

``context_for`` is the whole decision the script makes; ``main`` is a
CouchDB walk around it.  These exercise the decision, including the
cases where the answer is "leave the doc alone" -- which is what keeps
the backfill idempotent and keeps it from writing nulls into 8 943
docs that have no growth condition.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location(
    'backfill_label_context',
    _ROOT / 'fixes' / 'backfill_label_context.py',
)
assert _spec is not None and _spec.loader is not None
backfill_label_context = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill_label_context)

context_for = backfill_label_context.context_for


def _doc(**fields: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        '_id': 'taxon_a:Colony:0',
        'feature_label': 'Colony',
        'field': 'description',
        'start': 0,
        'end': 10,
    }
    base.update(fields)
    return base


class TestContextFor:
    @pytest.mark.parametrize('label,context', [
        ('Colony on MEA', 'MEA'),
        ('Colonies on SNA', 'SNA'),
        ('Culture on DG18', 'DG18'),
        ('Conidia in culture', 'culture'),
        ('Asci in culture V8', 'culture V8'),
        ('Conidia in vitro', 'in vitro'),
    ])
    def test_conditions_are_recognised(
            self, label: str, context: str) -> None:
        assert context_for(_doc(feature_label=label)) == context

    @pytest.mark.parametrize('label', [
        'Colony', 'Pileus', 'Sexual morph', 'Conidiogenous cells',
    ])
    def test_labels_without_a_condition_return_none(
            self, label: str) -> None:
        """``None`` means leave the doc alone.  8 943 of 9 068
        candidate docs take this branch, and writing ``None`` into
        them would touch every one for nothing."""
        assert context_for(_doc(feature_label=label)) is None

    def test_missing_label_returns_none(self) -> None:
        doc = _doc()
        del doc['feature_label']
        assert context_for(doc) is None

    @pytest.mark.parametrize('label', ['', '   ', None, 42])
    def test_unusable_labels_return_none(self, label: Any) -> None:
        """A malformed doc is skipped, not crashed on: the backfill
        walks whole databases and must not die on one bad row."""
        assert context_for(_doc(feature_label=label)) is None

    def test_is_idempotent_on_an_already_backfilled_doc(self) -> None:
        """The script compares this answer with what the doc already
        carries, so re-running writes nothing."""
        doc = _doc(feature_label='Colony on MEA', context='MEA')
        assert context_for(doc) == doc['context']
