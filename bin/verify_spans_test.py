"""Tests for bin/verify_spans."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_spans import (  # noqa: E402
    SpanCheck,
    check_treatment,
    meets_threshold,
    summarise,
)

_ANN = 'aaaa[@Stroma thin, whitish or yellow#Description*]bbbb'
_START, _END = 4, 40

_TREATMENT = {
    '_id': 'taxon_abc',
    'annotations_db': 'ann_db',
    'attachment_name': 'article.txt.ann',
    'ingest': {'_id': 'src1', 'db_name': 'skol_dev'},
    'description_spans': [{'start_char': _START, 'end_char': _END}],
}


class _Blob:
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text.encode('utf-8')


class _Db:
    def __init__(self, docs):
        self._docs = docs

    def get_attachment(self, doc_id, name):
        blob = self._docs.get(doc_id, {}).get(name)
        return _Blob(blob) if blob is not None else None


class _Server:
    def __init__(self, dbs):
        self._dbs = dbs

    def __contains__(self, name):
        return name in self._dbs

    def __getitem__(self, name):
        return _Db(self._dbs[name])


def _server():
    return _Server({'ann_db': {'src1': {'article.txt.ann': _ANN}}})


class TestCheckTreatment:
    def test_resolvable_span_passes(self) -> None:
        results = check_treatment(_TREATMENT, _server())
        assert [r.ok for r in results] == [True]

    def test_reports_the_field_and_index(self) -> None:
        r = check_treatment(_TREATMENT, _server())[0]
        assert r.field == 'description_spans' and r.index == 0

    def test_head_mismatch_fails_with_a_reason(self) -> None:
        doc = dict(_TREATMENT, description_spans=[
            {'start_char': _START, 'end_char': _END, 'head': 'Beyma van FH'},
        ])
        r = check_treatment(doc, _server())[0]
        assert not r.ok and 'mismatch' in r.reason

    def test_matching_head_passes(self) -> None:
        doc = dict(_TREATMENT, description_spans=[
            {'start_char': _START, 'end_char': _END,
             'head': '[@Stroma thin, whitish'},
        ])
        assert check_treatment(doc, _server())[0].ok

    def test_missing_annotations_db_fails_cleanly(self) -> None:
        doc = dict(_TREATMENT)
        del doc['annotations_db']
        r = check_treatment(doc, _server())[0]
        assert not r.ok and 'annotations_db' in r.reason

    def test_treatment_with_no_spans_yields_nothing(self) -> None:
        assert check_treatment({'_id': 'x'}, _server()) == []

    def test_checks_every_span_field(self) -> None:
        doc = dict(_TREATMENT, diagnosis_spans=[
            {'start_char': _START, 'end_char': _END},
        ])
        fields = {r.field for r in check_treatment(doc, _server())}
        assert fields == {'description_spans', 'diagnosis_spans'}


class TestSummarise:
    def test_counts_and_rate(self) -> None:
        checks = [
            SpanCheck('t', 'description_spans', 0, True, ''),
            SpanCheck('t', 'description_spans', 1, False, 'boom'),
        ]
        total, ok, rate = summarise(checks)
        assert (total, ok) == (2, 1) and abs(rate - 50.0) < 1e-6

    def test_empty_is_not_a_division_error(self) -> None:
        assert summarise([]) == (0, 0, 0.0)


class TestPassRateGate:
    """v3_hand has a real, known gap: some source documents carry no
    annotated attachment at all, so ~5.6 % of its spans cannot
    resolve.  A nightly job that always fails is one people learn to
    ignore, so the gate is a floor that still catches regressions.
    """

    def test_meets_threshold(self) -> None:
        checks = [SpanCheck('t', 'f', 0, True, ''),
                  SpanCheck('t', 'f', 1, False, 'x')]
        assert meets_threshold(checks, 50.0)

    def test_below_threshold(self) -> None:
        checks = [SpanCheck('t', 'f', 0, True, ''),
                  SpanCheck('t', 'f', 1, False, 'x')]
        assert not meets_threshold(checks, 50.1)

    def test_default_demands_everything(self) -> None:
        checks = [SpanCheck('t', 'f', 0, False, 'x')]
        assert not meets_threshold(checks, 100.0)

    def test_no_spans_is_not_a_pass(self) -> None:
        """An empty sample must not read as success."""
        assert not meets_threshold([], 100.0)
