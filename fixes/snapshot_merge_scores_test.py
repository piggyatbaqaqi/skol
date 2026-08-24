#!/usr/bin/env python3
"""Tests for ``fixes/snapshot_merge_scores.py``.

The snapshot exists because annotating a merge suspect **destroys**
its evidence: ``bin/llm_annotate_features`` replaces the whole status
doc, wiping ``metrics.n_terms_above_5``, and once that is gone
``fetch_prior_merge_skip_ids`` no longer recognises the treatment as
a suspect — so a later ``select_for_annotation
--exclude-suspected-merges`` silently re-admits it into the
annotatable pool.  See ``docs/plans/annotation-activity-split.md``
(F3).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'bin'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snapshot_merge_scores import (  # type: ignore[import]  # noqa: E402
    STATUS_MERGE_SUSPECT,
    format_tsv,
    iter_merge_scores,
    parse_metrics,
)


class _FakeRow:
    def __init__(self, doc_id: str, doc: Optional[Dict[str, Any]]) -> None:
        self.id = doc_id
        self.doc = doc


class _FakeDb:
    """Duck-typed CouchDB stand-in exposing only ``view``."""

    def __init__(self, docs: Dict[str, Dict[str, Any]]) -> None:
        self._docs = docs

    def view(self, _name: str, **_kw: Any) -> List[_FakeRow]:
        return [_FakeRow(k, v) for k, v in self._docs.items()]


@pytest.mark.xfail(strict=True, reason="skeleton: implementation pending")
class TestParseMetrics:
    """``metrics`` is sometimes a dict and sometimes its repr."""

    def test_dict_passes_through(self) -> None:
        assert parse_metrics({'n_terms_above_5': 11}) == {
            'n_terms_above_5': 11}

    def test_python_repr_string_is_parsed(self) -> None:
        """Live status docs store the repr, single quotes and all."""
        raw = "{'n_terms_above_5': 47, 'merge_threshold': 10}"
        assert parse_metrics(raw)['n_terms_above_5'] == 47

    def test_json_string_is_parsed(self) -> None:
        assert parse_metrics('{"n_terms_above_5": 12}')[
            'n_terms_above_5'] == 12

    def test_none_and_garbage_give_empty(self) -> None:
        assert parse_metrics(None) == {}
        assert parse_metrics('not a mapping at all') == {}
        assert parse_metrics('[1, 2, 3]') == {}


@pytest.mark.xfail(strict=True, reason="skeleton: implementation pending")
class TestIterMergeScores:
    """Only merge suspects, and never silently lose one."""

    def _db(self) -> _FakeDb:
        return _FakeDb({
            'taxon_a': {'status': STATUS_MERGE_SUSPECT,
                        'metrics': {'n_terms_above_5': 11}},
            'taxon_b': {'status': STATUS_MERGE_SUSPECT,
                        'metrics': "{'n_terms_above_5': 915}"},
            'taxon_c': {'status': 'success',
                        'metrics': {'n_terms_above_5': 3}},
            '_design/x': None,
        })

    def test_selects_only_merge_suspects(self) -> None:
        rows = list(iter_merge_scores(self._db()))
        assert [r[0] for r in rows] == ['taxon_a', 'taxon_b']

    def test_scores_are_ints(self) -> None:
        rows = dict((r[0], r[1]) for r in iter_merge_scores(self._db()))
        assert rows == {'taxon_a': 11, 'taxon_b': 915}

    def test_suspect_missing_its_score_is_reported_not_dropped(self) -> None:
        """A suspect with no score must surface as None, not vanish.

        Silently dropping it would understate the population and make
        the snapshot an incomplete restore.
        """
        db = _FakeDb({'taxon_x': {'status': STATUS_MERGE_SUSPECT}})
        assert list(iter_merge_scores(db)) == [('taxon_x', None)]


@pytest.mark.xfail(strict=True, reason="skeleton: implementation pending")
class TestFormatTsv:
    """Stable, sorted, greppable output."""

    def test_header_then_rows_sorted_by_score_descending(self) -> None:
        out = format_tsv([('taxon_a', 11), ('taxon_b', 915)]).splitlines()
        assert out[0] == 'treatment_id\tn_terms_above_5'
        assert out[1].split('\t') == ['taxon_b', '915']
        assert out[2].split('\t') == ['taxon_a', '11']

    def test_missing_score_renders_empty_and_sorts_last(self) -> None:
        out = format_tsv([('taxon_x', None), ('taxon_a', 11)]).splitlines()
        assert out[1].split('\t')[0] == 'taxon_a'
        assert out[2] == 'taxon_x\t'

    def test_trailing_newline(self) -> None:
        assert format_tsv([('taxon_a', 1)]).endswith('\n')

    def test_empty_input_still_has_header(self) -> None:
        assert format_tsv([]) == 'treatment_id\tn_terms_above_5\n'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
