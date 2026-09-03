#!/usr/bin/env python3
"""Tests for ``fixes.backfill_display_labels``.

The one-shot for annotations written before the parse-time capture
existed.  ``recover_display_label`` supplies the evidence; this script
chooses between the candidates a label collects across its several
spans, and that choice is what needs pinning.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location(
    'backfill_display_labels',
    _ROOT / 'fixes' / 'backfill_display_labels.py',
)
assert _spec is not None and _spec.loader is not None
backfill_display_labels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill_display_labels)

choose_display_labels = backfill_display_labels.choose_display_labels


def _ann(label: str, text: str, doc_id: str = 'd1') -> Dict[str, Any]:
    return {'_id': doc_id, 'feature_label': label, 'source_text': text}


class TestChooseDisplayLabels:
    def test_a_single_witness_is_enough(self) -> None:
        got = choose_display_labels([
            _ann('Spore print', 'Spore-print white to cream.'),
        ])
        assert got == {'Spore print': 'Spore-print'}

    def test_the_most_witnessed_form_wins(self) -> None:
        """`Clamp connections` collects more than one candidate across
        the corpus.  The corpus votes rather than the iteration order
        deciding -- the mistake vocabulary_index made."""
        got = choose_display_labels(
            [_ann('Clamp connections', 'Clamp-connections present.')] * 3
            + [_ann('Clamp connections', 'Clamp/connections present.')]
        )
        assert got == {'Clamp connections': 'Clamp-connections'}

    def test_a_tie_breaks_deterministically(self) -> None:
        first = choose_display_labels([
            _ann('Clamp connections', 'Clamp-connections here.'),
            _ann('Clamp connections', 'Clamp/connections here.'),
        ])
        second = choose_display_labels([
            _ann('Clamp connections', 'Clamp/connections here.'),
            _ann('Clamp connections', 'Clamp-connections here.'),
        ])
        assert first == second

    def test_labels_with_nothing_to_recover_are_absent(self) -> None:
        got = choose_display_labels([
            _ann('Pileus brown', 'Pileus brown 3 cm.'),
            _ann('Basidiospores', 'Colonies on MEA.'),
        ])
        assert got == {}

    def test_an_annotation_already_carrying_one_is_skipped(self) -> None:
        """Re-running must not churn, and must not overrule a form the
        annotator captured at parse time -- that one saw the original,
        while this one is inferring from a span."""
        ann = _ann('Spore print', 'Spore-print white.')
        ann['display_label'] = 'Spore print (deposit)'
        assert choose_display_labels([ann]) == {}
