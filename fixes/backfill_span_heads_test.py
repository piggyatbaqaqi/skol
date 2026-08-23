"""Tests for fixes/backfill_span_heads."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_span_heads import (  # noqa: E402
    AttachmentCache,
    backfill_treatment,
)
from span_resolver import SpanResolutionError  # noqa: E402


_XFAIL = pytest.mark.xfail(
    reason="2026-08-21: backfill_span_heads not implemented yet",
    strict=True,
)

_ANN = 'aaaa[@Stroma thin, whitish or yellow#Description*]bbbbcccc'


class _Blob:
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text.encode('utf-8')


class _Db:
    def __init__(self, docs, counter):
        self._docs = docs
        self._counter = counter

    def get_attachment(self, doc_id, name):
        self._counter.append((doc_id, name))
        blob = self._docs.get(doc_id, {}).get(name)
        return _Blob(blob) if blob is not None else None


class _Server:
    def __init__(self, dbs, counter):
        self._dbs = dbs
        self._counter = counter

    def __contains__(self, name):
        return name in self._dbs

    def __getitem__(self, name):
        return _Db(self._dbs[name], self._counter)


def _fixture():
    reads = []
    server = _Server({'ann_db': {'src1': {'article.txt.ann': _ANN}}}, reads)
    treatment = {
        '_id': 'taxon_abc',
        'annotations_db': 'ann_db',
        'attachment_name': 'article.txt.ann',
        'ingest': {'_id': 'src1', 'db_name': 'skol_dev'},
        'description_spans': [{'start_char': 4, 'end_char': 24}],
        'diagnosis_spans': [{'start_char': 24, 'end_char': 40}],
    }
    return server, treatment, reads


@_XFAIL
class TestBackfillTreatment:
    def test_sets_head_on_every_span(self) -> None:
        server, treatment, _ = _fixture()
        cache = AttachmentCache(server)
        assert backfill_treatment(treatment, cache) == 2
        assert 'head' in treatment['description_spans'][0]
        assert 'head' in treatment['diagnosis_spans'][0]

    def test_head_matches_the_span_text(self) -> None:
        server, treatment, _ = _fixture()
        backfill_treatment(treatment, AttachmentCache(server))
        assert treatment['description_spans'][0]['head'] == _ANN[4:24]

    def test_returns_zero_when_nothing_changes(self) -> None:
        """Idempotent: a second pass rewrites nothing."""
        server, treatment, _ = _fixture()
        cache = AttachmentCache(server)
        backfill_treatment(treatment, cache)
        assert backfill_treatment(treatment, cache) == 0

    def test_existing_head_is_not_overwritten(self) -> None:
        server, treatment, _ = _fixture()
        treatment['description_spans'][0]['head'] = 'PRESET'
        backfill_treatment(treatment, AttachmentCache(server))
        assert treatment['description_spans'][0]['head'] == 'PRESET'

    def test_treatment_without_spans_is_a_no_op(self) -> None:
        server, _, _ = _fixture()
        assert backfill_treatment({'_id': 'x'}, AttachmentCache(server)) == 0

    def test_unresolvable_treatment_raises(self) -> None:
        """The caller decides whether to skip; the backfill does not
        silently write a wrong fingerprint."""
        server, treatment, _ = _fixture()
        del treatment['annotations_db']
        with pytest.raises(SpanResolutionError):
            backfill_treatment(treatment, AttachmentCache(server))


@_XFAIL
class TestAttachmentCache:
    def test_reads_each_attachment_once(self) -> None:
        """81k treatments share far fewer source documents; re-reading
        a 200 KB attachment per treatment is the whole cost."""
        server, treatment, reads = _fixture()
        cache = AttachmentCache(server)
        second = dict(treatment, _id='taxon_def')
        backfill_treatment(treatment, cache)
        backfill_treatment(second, cache)
        assert len(reads) == 1

    def test_evicts_beyond_its_limit(self) -> None:
        server, _, reads = _fixture()
        cache = AttachmentCache(server, max_entries=1)
        for _ in range(2):
            cache.text('ann_db', 'src1', 'article.txt.ann')
            cache.clear()
        assert len(reads) == 2
