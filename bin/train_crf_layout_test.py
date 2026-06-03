"""Tests for bin/train_crf_layout.py — v4 layout CRF trainer."""
from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skol_classifier.v4.crf_layout import (  # noqa: E402
    LABEL_TO_INDEX,
)

from train_crf_layout import (  # type: ignore[import]  # noqa: E402
    inverse_frequency_weights,
    make_sbert_lookup,
    split_docs,
    train_one_run,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRedis:
    """Bytes-only stand-in for a redis.Redis client."""

    def __init__(self) -> None:
        self.store: Dict[bytes, bytes] = {}
        self.set_calls: List[str] = []

    def get(self, key) -> Optional[bytes]:
        key_bytes = key.encode() if isinstance(key, str) else key
        return self.store.get(key_bytes)

    def set(self, key, value) -> None:
        key_bytes = key.encode() if isinstance(key, str) else key
        self.store[key_bytes] = value
        self.set_calls.append(key_bytes.decode())


class FakeAttachment:
    """File-like wrapper that ``couchdb.Database.get_attachment``
    returns."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body


class FakeCouchDb:
    """Minimal stand-in for couchdb.Database: iteration, item access,
    get_attachment.  Each doc dict carries an ``_attachments`` key
    holding raw bytes for each named attachment."""

    def __init__(self, docs: Dict[str, Dict[str, Any]]) -> None:
        self.docs = docs

    def __iter__(self):
        return iter(self.docs)

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        return self.docs[doc_id]

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.docs

    def get_attachment(self, doc_id: str, name: str):
        atts = self.docs.get(doc_id, {}).get('_attachments') or {}
        if name in atts:
            return FakeAttachment(atts[name]['data'])
        return None


def _synth_doc(
    doc_id: str,
    *,
    ann_text: str,
    spans_json_bytes: bytes,
    page_headers_json_bytes: bytes,
    plaintext: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a synthetic doc carrying the four attachments the trainer
    reads: article.txt.ann, article.spans.v4.json,
    article.page-headers.json, and optionally article.txt."""
    atts = {
        'article.txt.ann': ann_text.encode('utf-8'),
        'article.spans.v4.json': spans_json_bytes,
        'article.page-headers.json': page_headers_json_bytes,
    }
    if plaintext is not None:
        atts['article.txt'] = plaintext.encode('utf-8')
    return {
        '_id': doc_id, '_rev': '1-aaa',
        '_attachments': {
            name: {'content_type': 'text/plain', 'length': len(data),
                   'data': data}
            for name, data in atts.items()
        },
    }


# ---------------------------------------------------------------------------
# 1. split_docs
# ---------------------------------------------------------------------------


class TestSplitDocs(unittest.TestCase):
    """Stratified-by-doc-length split: returns (train_ids, dev_ids).
    Idempotent under seed; partition is exact."""

    def _docs(self, n=100):
        # Variable doc lengths so the quartile-stratified shuffle
        # actually has something to balance.
        return [(f'doc_{i:03d}', 100 + (i * 37) % 5000) for i in range(n)]

    def test_dev_fraction_proportion(self):
        train, dev = split_docs(
            self._docs(100), dev_fraction=0.2, seed=42,
        )
        self.assertEqual(len(train) + len(dev), 100)
        # ~20 docs in dev; allow ±1 for quartile rounding.
        self.assertGreaterEqual(len(dev), 19)
        self.assertLessEqual(len(dev), 21)

    def test_same_seed_yields_same_split(self):
        docs = self._docs(100)
        a_train, a_dev = split_docs(docs, dev_fraction=0.2, seed=42)
        b_train, b_dev = split_docs(docs, dev_fraction=0.2, seed=42)
        self.assertEqual(a_train, b_train)
        self.assertEqual(a_dev, b_dev)

    def test_different_seed_different_split(self):
        docs = self._docs(100)
        _, dev_a = split_docs(docs, dev_fraction=0.2, seed=1)
        _, dev_b = split_docs(docs, dev_fraction=0.2, seed=2)
        self.assertNotEqual(set(dev_a), set(dev_b))


# ---------------------------------------------------------------------------
# 2. inverse_frequency_weights
# ---------------------------------------------------------------------------


class TestInverseFrequencyWeights(unittest.TestCase):
    """Class-weight derivation: rare labels weighted higher; unseen
    labels weighted 1.0 (defensive, not penalised)."""

    def test_balanced_corpus_yields_uniform_weights(self):
        # Equal counts across all 8 labels → all weights == 1.0.
        counts = np.array([100] * 8, dtype=np.float64)
        w = inverse_frequency_weights(counts)
        self.assertEqual(w.shape, (8,))
        self.assertTrue(np.allclose(w, 1.0))

    def test_rare_class_gets_higher_weight(self):
        # Other dominates; Page-header rare.
        counts = np.zeros(8, dtype=np.float64)
        counts[LABEL_TO_INDEX['Other']] = 1000
        counts[LABEL_TO_INDEX['Page-header']] = 10
        w = inverse_frequency_weights(counts)
        self.assertGreater(
            w[LABEL_TO_INDEX['Page-header']],
            w[LABEL_TO_INDEX['Other']] * 50,
        )

    def test_unseen_class_gets_weight_one(self):
        # Index never appears → weight = 1.0.
        counts = np.zeros(8, dtype=np.float64)
        counts[LABEL_TO_INDEX['Other']] = 1000
        counts[LABEL_TO_INDEX['Page-header']] = 50
        w = inverse_frequency_weights(counts)
        self.assertEqual(float(w[LABEL_TO_INDEX['Index']]), 1.0)
        self.assertEqual(float(w[LABEL_TO_INDEX['ToC-entry']]), 1.0)


# ---------------------------------------------------------------------------
# 3. SBERT lookup factory
# ---------------------------------------------------------------------------


class TestMakeSbertLookup(unittest.TestCase):
    """The closure constructs ``skol:sbert:<model_tag>:<sha256>``
    keys, decodes bytes as float32, and returns None on miss."""

    def test_hit_returns_decoded_vector(self):
        r = FakeRedis()
        vec = np.full(768, 0.5, dtype=np.float32)
        line = 'Boletus edulis Bull.'
        key = (
            'skol:sbert:mpnet:'
            + hashlib.sha256(line.encode('utf-8')).hexdigest()
        )
        r.store[key.encode()] = vec.tobytes()

        lookup = make_sbert_lookup(r, model_tag='mpnet', dim=768)
        result = lookup(line)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, vec)

    def test_miss_returns_none(self):
        lookup = make_sbert_lookup(FakeRedis(), 'mpnet', 768)
        self.assertIsNone(lookup('unseen line'))

    def test_model_tag_in_key(self):
        """Switching model_tag changes the key namespace."""
        r = FakeRedis()
        vec = np.zeros(384, dtype=np.float32)
        line = 'x'
        key = (
            'skol:sbert:minilm:'
            + hashlib.sha256(line.encode('utf-8')).hexdigest()
        )
        r.store[key.encode()] = vec.tobytes()
        lookup = make_sbert_lookup(r, model_tag='minilm', dim=384)
        self.assertIsNotNone(lookup(line))


# ---------------------------------------------------------------------------
# 4. Training loop integration
# ---------------------------------------------------------------------------


def _empty_spans_json(plaintext: str) -> bytes:
    return (
        b'{"schema_version":"1","source_attachment":"article.txt",'
        b'"doc_id":"d","spans":[]}'
    )


def _empty_page_headers_json(plaintext: str) -> bytes:
    n_lines = len(plaintext.split('\n'))
    return (
        b'{"schema_version":"1","n_lines":'
        + str(n_lines).encode()
        + b',"regions":[],"per_line_confidence":'
        + ('[' + ','.join(['0.0'] * n_lines) + ']').encode()
        + b',"sequence_fit":null,"alternation_score":0.0}'
    )


def _three_doc_corpus() -> Dict[str, Dict[str, Any]]:
    """Three synthetic docs: two with all required attachments, one
    missing article.spans.v4.json (gets skipped)."""
    docs = {}

    # Doc 1: header line + body
    pt = 'Page top header\nBoletus edulis Bull.\nA description line.\n'
    ann = (
        '[@Page top header#Page-header*]'
        '[@Boletus edulis Bull.#Nomenclature*]'
        '[@A description line.#Description*]'
    )
    docs['d1'] = _synth_doc(
        'd1', ann_text=ann, plaintext=pt,
        spans_json_bytes=_empty_spans_json(pt),
        page_headers_json_bytes=_empty_page_headers_json(pt),
    )

    # Doc 2: header + body
    pt2 = 'Vol 5 page 12\nMore description text.\n'
    ann2 = (
        '[@Vol 5 page 12#Page-header*]'
        '[@More description text.#Description*]'
    )
    docs['d2'] = _synth_doc(
        'd2', ann_text=ann2, plaintext=pt2,
        spans_json_bytes=_empty_spans_json(pt2),
        page_headers_json_bytes=_empty_page_headers_json(pt2),
    )

    # Doc 3: missing article.spans.v4.json — must be skipped.
    pt3 = 'Some body text\n'
    ann3 = '[@Some body text#Description*]'
    doc3 = _synth_doc(
        'd3', ann_text=ann3, plaintext=pt3,
        spans_json_bytes=_empty_spans_json(pt3),
        page_headers_json_bytes=_empty_page_headers_json(pt3),
    )
    # Drop spans attachment.
    del doc3['_attachments']['article.spans.v4.json']
    docs['d3'] = doc3
    return docs


class TestTrainingLoopIntegration(unittest.TestCase):
    """End-to-end trainer with mocked CouchDB + mocked Redis + mocked
    SBERT lookup.  Two epochs, three synthetic docs, no live
    services."""

    def setUp(self):
        self.db = FakeCouchDb(_three_doc_corpus())
        self.redis = FakeRedis()

    def _zero_sbert(self, _line):
        # Stand in for a SBERT cache hit — every line has a zero
        # vector, so the model trains on the non-SBERT features.
        return np.zeros(768, dtype=np.float32)

    def test_trains_and_saves(self):
        counts = train_one_run(
            self.db, self.redis,
            doc_ids=['d1', 'd2', 'd3'],
            sbert_lookup=self._zero_sbert,
            epochs=2, lr=0.01, seed=42,
            dev_fraction=0.0,           # tiny corpus: skip dev split
            device='cpu',
            redis_key='skol:classifier:model:v4_layout',
            meta_key='skol:classifier:model:v4_layout:meta',
            dry_run=False,
        )
        # d3 should be skipped (missing spans attachment).
        self.assertEqual(counts['trained_docs'], 2)
        self.assertEqual(counts['skipped_no_spans'], 1)
        # Bundle written to both keys.
        self.assertIn('skol:classifier:model:v4_layout',
                      self.redis.set_calls)
        self.assertIn('skol:classifier:model:v4_layout:meta',
                      self.redis.set_calls)

    def test_dry_run_does_not_write_redis(self):
        counts = train_one_run(
            self.db, self.redis,
            doc_ids=['d1', 'd2'],
            sbert_lookup=self._zero_sbert,
            epochs=1, lr=0.01, seed=42,
            dev_fraction=0.0,
            device='cpu',
            redis_key='skol:classifier:model:v4_layout',
            meta_key='skol:classifier:model:v4_layout:meta',
            dry_run=True,
        )
        self.assertEqual(counts['trained_docs'], 2)
        self.assertEqual(self.redis.set_calls, [])

    def test_skip_existing_short_circuits(self):
        """Both Redis keys already populated → trainer skips work."""
        self.redis.store[b'skol:classifier:model:v4_layout'] = b'existing'
        self.redis.store[b'skol:classifier:model:v4_layout:meta'] = b'{}'
        counts = train_one_run(
            self.db, self.redis,
            doc_ids=['d1', 'd2'],
            sbert_lookup=self._zero_sbert,
            epochs=1, lr=0.01, seed=42,
            dev_fraction=0.0,
            device='cpu',
            redis_key='skol:classifier:model:v4_layout',
            meta_key='skol:classifier:model:v4_layout:meta',
            dry_run=False,
            skip_existing=True,
        )
        self.assertEqual(counts['trained_docs'], 0)
        self.assertEqual(counts.get('short_circuited'), 1)
        # No new writes (the existing bytes are still there).
        self.assertEqual(self.redis.set_calls, [])

    def test_missing_attachment_doc_skipped(self):
        """Docs without article.spans.v4.json don't crash the run."""
        counts = train_one_run(
            self.db, self.redis,
            doc_ids=['d3'],          # only the broken doc
            sbert_lookup=self._zero_sbert,
            epochs=1, lr=0.01, seed=42,
            dev_fraction=0.0,
            device='cpu',
            redis_key='skol:classifier:model:v4_layout',
            meta_key='skol:classifier:model:v4_layout:meta',
            dry_run=True,
        )
        self.assertEqual(counts['trained_docs'], 0)
        self.assertEqual(counts['skipped_no_spans'], 1)


if __name__ == '__main__':
    unittest.main()
