"""Tests for span_resolver.

The failure this module exists to prevent: reading a span's offsets
against the wrong attachment returns *plausible* text rather than an
error.  Resolution must therefore be single-path and self-verifying.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from span_resolver import (  # noqa: E402
    SpanResolutionError,
    coordinate_space,
    resolve_span,
    span_head,
    verify_head,
)

_ANN = (
    'line one\n'
    '[@Stroma thin, whitish or yellow, hyphal or subiculum-like.#Description*]\n'
    'trailing\n'
)
_START = _ANN.index('[@Stroma')
_END = _START + 40

_TREATMENT = {
    '_id': 'taxon_abc',
    'annotations_db': 'skol_exp_x_ann_combined',
    'attachment_name': 'article.txt.ann',
    'ingest': {'_id': 'src1', 'db_name': 'skol_dev'},
}


class _FakeServer:
    """Minimal CouchDB stand-in: {db_name: {doc_id: {att: text}}}."""

    def __init__(self, dbs):
        self._dbs = dbs

    def __contains__(self, name):
        return name in self._dbs

    def __getitem__(self, name):
        if name not in self._dbs:
            raise KeyError(name)
        return _FakeDb(self._dbs[name])


class _FakeDb:
    def __init__(self, docs):
        self._docs = docs

    def get_attachment(self, doc_id, name):
        blob = self._docs.get(doc_id, {}).get(name)
        if blob is None:
            return None
        return _FakeBlob(blob)


class _FakeBlob:
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text.encode('utf-8')


def _server():
    return _FakeServer({
        'skol_exp_x_ann_combined': {'src1': {'article.txt.ann': _ANN}},
        # The decoy: the raw DB holds a DIFFERENT file at the same
        # offsets, which is exactly how a wrong lookup stays silent.
        'skol_dev': {'src1': {'article.txt': 'x' * 400}},
    })


class TestCoordinateSpace:
    """Which (db, doc, attachment) a treatment's offsets belong to."""

    def test_reads_annotations_db_not_ingest_db(self) -> None:
        space = coordinate_space(_TREATMENT)
        assert space.db == 'skol_exp_x_ann_combined'
        assert space.attachment == 'article.txt.ann'
        assert space.doc_id == 'src1'

    def test_missing_annotations_db_raises(self) -> None:
        doc = dict(_TREATMENT)
        del doc['annotations_db']
        with pytest.raises(SpanResolutionError) as exc:
            coordinate_space(doc)
        assert 'annotations_db' in str(exc.value)

    def test_ingest_db_name_is_never_used_as_a_fallback(self) -> None:
        """skol_dev holds article.txt, not the annotated file; silently
        falling back to it is the whole bug."""
        doc = dict(_TREATMENT)
        del doc['annotations_db']
        with pytest.raises(SpanResolutionError) as exc:
            coordinate_space(doc)
        assert 'skol_dev' not in str(exc.value).replace(
            'ingest.db_name', '')


class TestSpanHead:
    """The fingerprint stored with a span."""

    def test_is_a_prefix_of_the_span_text(self) -> None:
        head = span_head(_ANN[_START:_END])
        assert _ANN[_START:_END].startswith(head)

    def test_collapses_whitespace(self) -> None:
        assert span_head('a  b\n c') == 'a b c'

    def test_bounded_length(self) -> None:
        assert len(span_head('z' * 500)) <= 40

    def test_empty_text_yields_empty_head(self) -> None:
        assert span_head('') == ''


class TestVerifyHead:
    def test_match_passes(self) -> None:
        assert verify_head('Stroma thin, whitish', 'Stroma thin, whitish') is None

    def test_absent_head_is_tolerated(self) -> None:
        """Spans written before fingerprints existed must still
        resolve; absence is not a mismatch."""
        assert verify_head(None, 'anything') is None

    def test_mismatch_raises_with_both_values(self) -> None:
        with pytest.raises(SpanResolutionError) as exc:
            verify_head('Stroma thin', 'Beyma van FH (1938)')
        msg = str(exc.value)
        assert 'Stroma thin' in msg and 'Beyma van FH' in msg


class TestResolveSpan:
    def test_returns_the_text_at_the_offsets(self) -> None:
        span = {'start_char': _START, 'end_char': _END}
        assert resolve_span(_TREATMENT, span, _server()) == _ANN[_START:_END]

    def test_verifies_a_stored_head(self) -> None:
        span = {'start_char': _START, 'end_char': _END,
                'head': span_head(_ANN[_START:_END])}
        assert resolve_span(_TREATMENT, span, _server()) == _ANN[_START:_END]

    def test_wrong_offsets_are_caught_by_the_head(self) -> None:
        """The case that motivated this module."""
        span = {'start_char': 0, 'end_char': 8, 'head': 'Stroma thin, whitish'}
        with pytest.raises(SpanResolutionError):
            resolve_span(_TREATMENT, span, _server())

    def test_missing_attachment_raises(self) -> None:
        doc = dict(_TREATMENT, attachment_name='article.nope.ann')
        with pytest.raises(SpanResolutionError) as exc:
            resolve_span(doc, {'start_char': 0, 'end_char': 4}, _server())
        assert 'article.nope.ann' in str(exc.value)

    def test_missing_database_raises(self) -> None:
        doc = dict(_TREATMENT, annotations_db='skol_gone')
        with pytest.raises(SpanResolutionError) as exc:
            resolve_span(doc, {'start_char': 0, 'end_char': 4}, _server())
        assert 'skol_gone' in str(exc.value)

    def test_offsets_past_end_of_file_raise(self) -> None:
        span = {'start_char': 10 ** 6, 'end_char': 10 ** 6 + 10}
        with pytest.raises(SpanResolutionError):
            resolve_span(_TREATMENT, span, _server())
