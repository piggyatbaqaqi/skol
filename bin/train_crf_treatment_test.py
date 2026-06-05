"""Tests for bin/train_crf_treatment.py — v4 Pass-2 CRF trainer.

Parallel to bin/train_crf_layout_test.py; the only really novel test
class is TestPass2Mask which exercises the layout-line filtering.
"""
from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skol_classifier.v4.crf_treatment import (  # noqa: E402
    LABEL_TO_INDEX,
)

from train_crf_treatment import (  # type: ignore[import]  # noqa: E402
    _prepare_doc_pass2,
    inverse_frequency_weights,
    make_sbert_lookup,
    split_docs,
    train_one_run,
)


# ---------------------------------------------------------------------------
# Fakes (same shape as train_crf_layout_test)
# ---------------------------------------------------------------------------


class FakeRedis:
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
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body


class FakeCouchDb:
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


def _empty_spans_json() -> bytes:
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


# ---------------------------------------------------------------------------
# 1. split_docs (duplicated logic, duplicated test surface)
# ---------------------------------------------------------------------------


class TestSplitDocs(unittest.TestCase):

    def _docs(self, n=100):
        return [(f'doc_{i:03d}', 100 + (i * 37) % 5000) for i in range(n)]

    def test_dev_fraction_proportion(self):
        train, dev = split_docs(
            self._docs(100), dev_fraction=0.2, seed=42,
        )
        self.assertEqual(len(train) + len(dev), 100)
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

    def test_balanced_corpus_yields_uniform_weights(self):
        counts = np.array([100] * 12, dtype=np.float64)
        w = inverse_frequency_weights(counts)
        self.assertEqual(w.shape, (12,))
        self.assertTrue(np.allclose(w, 1.0))

    def test_rare_class_gets_higher_weight(self):
        counts = np.zeros(12, dtype=np.float64)
        counts[LABEL_TO_INDEX['Misc-exposition']] = 1000
        counts[LABEL_TO_INDEX['Phylogeny']] = 10
        w = inverse_frequency_weights(counts)
        self.assertGreater(
            w[LABEL_TO_INDEX['Phylogeny']],
            w[LABEL_TO_INDEX['Misc-exposition']] * 50,
        )

    def test_unseen_class_gets_weight_one(self):
        counts = np.zeros(12, dtype=np.float64)
        counts[LABEL_TO_INDEX['Misc-exposition']] = 1000
        counts[LABEL_TO_INDEX['Description']] = 50
        w = inverse_frequency_weights(counts)
        # Etymology never appears.
        self.assertEqual(float(w[LABEL_TO_INDEX['Etymology']]), 1.0)


# ---------------------------------------------------------------------------
# 3. make_sbert_lookup
# ---------------------------------------------------------------------------


class TestMakeSbertLookup(unittest.TestCase):

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
        np.testing.assert_array_equal(lookup(line), vec)

    def test_miss_returns_none(self):
        lookup = make_sbert_lookup(FakeRedis(), 'mpnet', 768)
        self.assertIsNone(lookup('unseen line'))


# ---------------------------------------------------------------------------
# 4. Pass-2 mask filters Pass-1 layout lines
# ---------------------------------------------------------------------------


class TestPass2Mask(unittest.TestCase):
    """The critical Pass-2 contract: lines whose Pass-1 oracle label
    is a layout tag are removed from the per-doc tensors before the
    CRF sees them.  pytorch-crf's mask only supports contiguous
    valid positions, so we filter to a content subsequence."""

    def _zero_sbert(self, _line):
        return np.zeros(768, dtype=np.float32)

    def test_pass1_layout_lines_are_filtered(self):
        """3-line doc: [Page-header, Description, Bibliography].
        After Pass-2 filtering only line 1 (the Description body)
        should reach the CRF tensors.  No trailing newline so the
        synthetic empty line isn't part of the count."""
        plaintext = 'Top header\nBody description.\nLit cited.'
        ann = (
            '[@Top header#Page-header*]'
            '[@Body description.#Description*]'
            '[@Lit cited.#Bibliography*]'
        )
        doc = _synth_doc(
            'd', ann_text=ann, plaintext=plaintext,
            spans_json_bytes=_empty_spans_json(),
            page_headers_json_bytes=_empty_page_headers_json(plaintext),
        )
        db = FakeCouchDb({'d': doc})

        prepared = _prepare_doc_pass2(
            db, 'd', sbert_lookup=self._zero_sbert,
        )
        self.assertIsNotNone(prepared)
        features, labels = prepared

        # Only one line survives Pass-1 filtering: the Description.
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(labels.shape[0], 1)
        self.assertEqual(
            int(labels[0]), LABEL_TO_INDEX['Description'],
        )

    def test_all_layout_doc_skipped(self):
        """A doc where every line is a layout tag has nothing left
        after Pass-1 filtering — _prepare_doc_pass2 must return None
        rather than feed an empty sequence to pytorch-crf."""
        plaintext = 'Top header\nFooter'
        ann = (
            '[@Top header#Page-header*]'
            '[@Footer#Page-header*]'
        )
        doc = _synth_doc(
            'd', ann_text=ann, plaintext=plaintext,
            spans_json_bytes=_empty_spans_json(),
            page_headers_json_bytes=_empty_page_headers_json(plaintext),
        )
        db = FakeCouchDb({'d': doc})
        self.assertIsNone(
            _prepare_doc_pass2(
                db, 'd', sbert_lookup=self._zero_sbert,
            ),
        )


# ---------------------------------------------------------------------------
# 5. Step 7.δ: --use-predicted-layout (exposure-bias mode)
# ---------------------------------------------------------------------------


class _StubLayoutCRF:
    """LayoutCRF stand-in.  ``decode()`` returns
    ``[stub_indices]`` regardless of features so the test asserts
    flow of control, not model behaviour."""

    def __init__(self, stub_indices):
        self.stub_indices = list(stub_indices)
        self.feature_dim = 791
        self.calls = 0

    def to(self, _device):
        return self

    def eval(self):
        return self

    def decode(self, _features, _mask):
        self.calls += 1
        return [list(self.stub_indices)]


class TestPrepareDocPass2WithPredictedLayout(unittest.TestCase):
    """Step 7.δ extends ``_prepare_doc_pass2`` with a
    ``use_predicted_layout=True`` mode: the per-line layout sequence
    comes from running the trained Pass-1 CRF on the doc's features
    instead of from ``build_label_sequence``.  Trains Pass-2 on
    sequences that match what it'll actually see at inference time.
    """

    def _zero_sbert(self, _line):
        return np.zeros(768, dtype=np.float32)

    def test_uses_layout_crf_decode_when_flag_set(self):
        """3-line doc with annotations that label every line as
        Description (Pass-1 oracle → 'Other' for all 3).  Predicted
        Pass-1 says line 0 is Page-header (idx 0), so only lines 1+2
        survive.  Assertion: pass2 features have 2 rows, not 3."""
        from skol_classifier.v4.crf_layout import (
            OTHER_INDEX as LAYOUT_OTHER_INDEX,
        )
        plaintext = 'Header line\nBody one\nBody two'
        ann = (
            '[@Header line#Description*]'
            '[@Body one#Description*]'
            '[@Body two#Description*]'
        )
        doc = _synth_doc(
            'd', ann_text=ann, plaintext=plaintext,
            spans_json_bytes=_empty_spans_json(),
            page_headers_json_bytes=_empty_page_headers_json(plaintext),
        )
        db = FakeCouchDb({'d': doc})
        # Stub predicts: [Page-header(0), Other(7), Other(7)]
        stub = _StubLayoutCRF([0, LAYOUT_OTHER_INDEX, LAYOUT_OTHER_INDEX])

        prepared = _prepare_doc_pass2(
            db, 'd', sbert_lookup=self._zero_sbert,
            use_predicted_layout=True, layout_crf=stub,
            device='cpu',
        )
        self.assertIsNotNone(prepared)
        features, labels = prepared
        # Only 2 lines survive (the two predicted Other lines),
        # NOT 3 (which would be the oracle Description-everywhere
        # answer).  This is the load-bearing assertion: the
        # predicted layout decides the mask, not the YEDDA blocks.
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(labels.shape[0], 2)
        self.assertEqual(stub.calls, 1)

    def test_oracle_path_unchanged_when_flag_unset(self):
        """When ``use_predicted_layout=False`` (the default), the
        original oracle path is used and the layout CRF stub MUST
        NOT be called."""
        plaintext = 'Body line'
        ann = '[@Body line#Description*]'
        doc = _synth_doc(
            'd', ann_text=ann, plaintext=plaintext,
            spans_json_bytes=_empty_spans_json(),
            page_headers_json_bytes=_empty_page_headers_json(plaintext),
        )
        db = FakeCouchDb({'d': doc})
        stub = _StubLayoutCRF([0])  # would mark line as layout if used

        prepared = _prepare_doc_pass2(
            db, 'd', sbert_lookup=self._zero_sbert,
            use_predicted_layout=False, layout_crf=stub,
        )
        self.assertIsNotNone(prepared)
        features, _ = prepared
        # Oracle says it's Description (non-layout) — 1 row.
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(stub.calls, 0)

    def test_layout_crf_required_when_flag_set(self):
        """Passing use_predicted_layout=True without supplying a
        layout_crf is an operator error and must raise clearly."""
        plaintext = 'Body line'
        ann = '[@Body line#Description*]'
        doc = _synth_doc(
            'd', ann_text=ann, plaintext=plaintext,
            spans_json_bytes=_empty_spans_json(),
            page_headers_json_bytes=_empty_page_headers_json(plaintext),
        )
        db = FakeCouchDb({'d': doc})
        with self.assertRaises(ValueError) as cm:
            _prepare_doc_pass2(
                db, 'd', sbert_lookup=self._zero_sbert,
                use_predicted_layout=True, layout_crf=None,
            )
        self.assertIn('layout_crf', str(cm.exception).lower())


# ---------------------------------------------------------------------------
# 5. Training loop integration
# ---------------------------------------------------------------------------


def _three_doc_corpus() -> Dict[str, Dict[str, Any]]:
    docs = {}

    # Doc 1: layout + content
    pt = 'Top header\nBoletus edulis Bull.\nA description line.\n'
    ann = (
        '[@Top header#Page-header*]'
        '[@Boletus edulis Bull.#Nomenclature*]'
        '[@A description line.#Description*]'
    )
    docs['d1'] = _synth_doc(
        'd1', ann_text=ann, plaintext=pt,
        spans_json_bytes=_empty_spans_json(),
        page_headers_json_bytes=_empty_page_headers_json(pt),
    )

    pt2 = 'Boletus rubellus.\nLight brown specimens.\n'
    ann2 = (
        '[@Boletus rubellus.#Nomenclature*]'
        '[@Light brown specimens.#Description*]'
    )
    docs['d2'] = _synth_doc(
        'd2', ann_text=ann2, plaintext=pt2,
        spans_json_bytes=_empty_spans_json(),
        page_headers_json_bytes=_empty_page_headers_json(pt2),
    )

    # Doc 3: missing spans attachment.
    pt3 = 'Some body text\n'
    ann3 = '[@Some body text#Description*]'
    doc3 = _synth_doc(
        'd3', ann_text=ann3, plaintext=pt3,
        spans_json_bytes=_empty_spans_json(),
        page_headers_json_bytes=_empty_page_headers_json(pt3),
    )
    del doc3['_attachments']['article.spans.v4.json']
    docs['d3'] = doc3
    return docs


class TestTrainingLoopIntegration(unittest.TestCase):

    def setUp(self):
        self.db = FakeCouchDb(_three_doc_corpus())
        self.redis = FakeRedis()

    def _zero_sbert(self, _line):
        return np.zeros(768, dtype=np.float32)

    def test_trains_and_saves(self):
        counts = train_one_run(
            self.db, self.redis,
            doc_ids=['d1', 'd2', 'd3'],
            sbert_lookup=self._zero_sbert,
            epochs=2, lr=0.01, seed=42,
            dev_fraction=0.0,
            device='cpu',
            redis_key='skol:classifier:model:v4_treatment',
            meta_key='skol:classifier:model:v4_treatment:meta',
            dry_run=False,
        )
        self.assertEqual(counts['trained_docs'], 2)
        self.assertEqual(counts['skipped_no_spans'], 1)
        self.assertIn('skol:classifier:model:v4_treatment',
                      self.redis.set_calls)
        self.assertIn('skol:classifier:model:v4_treatment:meta',
                      self.redis.set_calls)

    def test_dry_run_does_not_write_redis(self):
        counts = train_one_run(
            self.db, self.redis,
            doc_ids=['d1', 'd2'],
            sbert_lookup=self._zero_sbert,
            epochs=1, lr=0.01, seed=42,
            dev_fraction=0.0,
            device='cpu',
            redis_key='skol:classifier:model:v4_treatment',
            meta_key='skol:classifier:model:v4_treatment:meta',
            dry_run=True,
        )
        self.assertEqual(counts['trained_docs'], 2)
        self.assertEqual(self.redis.set_calls, [])

    def test_skip_existing_short_circuits(self):
        self.redis.store[b'skol:classifier:model:v4_treatment'] = b'x'
        self.redis.store[b'skol:classifier:model:v4_treatment:meta'] = b'{}'
        counts = train_one_run(
            self.db, self.redis,
            doc_ids=['d1', 'd2'],
            sbert_lookup=self._zero_sbert,
            epochs=1, lr=0.01, seed=42,
            dev_fraction=0.0,
            device='cpu',
            redis_key='skol:classifier:model:v4_treatment',
            meta_key='skol:classifier:model:v4_treatment:meta',
            dry_run=False,
            skip_existing=True,
        )
        self.assertEqual(counts['trained_docs'], 0)
        self.assertEqual(counts.get('short_circuited'), 1)
        self.assertEqual(self.redis.set_calls, [])

    def test_missing_attachment_doc_skipped(self):
        counts = train_one_run(
            self.db, self.redis,
            doc_ids=['d3'],
            sbert_lookup=self._zero_sbert,
            epochs=1, lr=0.01, seed=42,
            dev_fraction=0.0,
            device='cpu',
            redis_key='skol:classifier:model:v4_treatment',
            meta_key='skol:classifier:model:v4_treatment:meta',
            dry_run=True,
        )
        self.assertEqual(counts['trained_docs'], 0)
        self.assertEqual(counts['skipped_no_spans'], 1)


if __name__ == '__main__':
    unittest.main()
