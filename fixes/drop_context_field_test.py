#!/usr/bin/env python3
"""Tests for ``fixes.drop_context_field``."""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location(
    'drop_context_field', _ROOT / 'fixes' / 'drop_context_field.py')
assert _spec is not None and _spec.loader is not None
drop_context_field = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drop_context_field)

documents_to_strip = drop_context_field.documents_to_strip


def _doc(**over: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        '_id': 'taxon_a:Colony:0', '_rev': '1-abc',
        'feature_label': 'Colony on MEA', 'start': 0,
    }
    base.update(over)
    return base


class TestDocumentsToStrip:
    def test_a_document_with_the_field_loses_it(self) -> None:
        got = documents_to_strip([_doc(context='MEA')])
        assert len(got) == 1
        assert 'context' not in got[0]

    def test_everything_else_survives(self) -> None:
        """Including _rev, without which the write is rejected."""
        got = documents_to_strip([_doc(context='MEA')])
        assert got[0]['_id'] == 'taxon_a:Colony:0'
        assert got[0]['_rev'] == '1-abc'
        assert got[0]['feature_label'] == 'Colony on MEA'

    def test_a_document_without_the_field_is_untouched(self) -> None:
        """What makes a re-run free."""
        assert documents_to_strip([_doc()]) == []

    def test_the_input_is_not_mutated(self) -> None:
        """The caller may still need the original -- for a report, or
        because the write failed."""
        original = _doc(context='MEA')
        documents_to_strip([original])
        assert original['context'] == 'MEA'
