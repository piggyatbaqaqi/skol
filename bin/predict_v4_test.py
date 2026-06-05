"""Tests for bin/predict_v4.py — v4 end-to-end prediction CLI.

The model decoding contract belongs to
``skol_classifier/v4/predictor_test.py``.  These tests stub
``predict_doc`` out and focus on the CLI plumbing:

* skip-existing / force / dry-run / limit semantics
* missing-attachment skip behaviour
* output-DB write semantics (FakeCouchDb spy)
* env_config experiment-doc → Redis-key resolution
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import predict_v4  # noqa: E402


# ---------------------------------------------------------------------------
# Fake CouchDB (mirrors bin/train_crf_layout_test.py:60-82)
# ---------------------------------------------------------------------------


class FakeAttachment:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body


class FakeCouchDb:
    """Minimal stand-in for couchdb.Database supporting iteration,
    item access, attachment read, and ``put_attachment`` spy.
    Records all writes so tests can assert what was saved."""

    def __init__(self, docs: Dict[str, Dict[str, Any]]) -> None:
        self.docs = docs
        # (doc_id, name, body, content_type) tuples in call order.
        self.puts: List[Tuple[str, str, bytes, str]] = []

    def __iter__(self):
        return iter(self.docs)

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        return self.docs[doc_id]

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.docs

    def view(self, _view_name: str):
        """``_all_docs`` view — yields objects with ``.id`` attribute."""
        class Row:
            def __init__(self, _id: str) -> None:
                self.id = _id
        for doc_id in self.docs:
            yield Row(doc_id)

    def get_attachment(self, doc_id: str, name: str):
        atts = self.docs.get(doc_id, {}).get('_attachments') or {}
        if name in atts:
            return FakeAttachment(atts[name]['data'])
        return None

    def put_attachment(
        self, doc: Dict[str, Any], data: bytes,
        *, filename: str, content_type: str = 'text/plain',
    ) -> None:
        doc_id = doc['_id']
        self.puts.append((doc_id, filename, data, content_type))
        # Mirror what real couchdb does: register the attachment so
        # subsequent reads/skip-existing checks see it.
        atts = self.docs[doc_id].setdefault('_attachments', {})
        atts[filename] = {
            'content_type': content_type,
            'length': len(data),
            'data': data,
        }


# ---------------------------------------------------------------------------
# Doc synthesis helpers
# ---------------------------------------------------------------------------


_DEFAULT_PLAINTEXT = 'Line one\nLine two\nLine three'
_DEFAULT_SPANS = json.dumps({
    'version': '1', 'doc_id': 'unused',
    'source_attachment': 'article.txt', 'spans': [],
}).encode('utf-8')
_DEFAULT_PAGE_HEADERS = json.dumps({
    'version': 1, 'per_line_confidence': [0.0, 0.0, 0.0],
}).encode('utf-8')


def _synth_doc(
    doc_id: str,
    *,
    plaintext: Optional[bytes] = None,
    spans_json: Optional[bytes] = _DEFAULT_SPANS,
    page_headers_json: Optional[bytes] = _DEFAULT_PAGE_HEADERS,
    existing_ann: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Build a doc carrying the attachments predict_v4 reads."""
    atts: Dict[str, Dict[str, Any]] = {}
    if plaintext is None:
        plaintext = _DEFAULT_PLAINTEXT.encode('utf-8')
    atts['article.txt'] = {
        'content_type': 'text/plain',
        'length': len(plaintext),
        'data': plaintext,
    }
    if spans_json is not None:
        atts['article.spans.v4.json'] = {
            'content_type': 'application/json',
            'length': len(spans_json),
            'data': spans_json,
        }
    if page_headers_json is not None:
        atts['article.page-headers.json'] = {
            'content_type': 'application/json',
            'length': len(page_headers_json),
            'data': page_headers_json,
        }
    if existing_ann is not None:
        atts['article.txt.ann'] = {
            'content_type': 'text/plain',
            'length': len(existing_ann),
            'data': existing_ann,
        }
    return {
        '_id': doc_id, '_rev': '1-aaa',
        '_attachments': atts,
    }


def _stub_predict(plaintext: str, *_args, **_kwargs):
    """Stub for predictor.predict_doc — bypasses model load + decode.
    Returns (per_line_tags, ann_text) for whatever number of lines
    plaintext.split('\\n') yields."""
    lines = plaintext.split('\n')
    tags = ['Description'] * len(lines)
    blocks = [f'[@{ln}#Description*]' for ln in lines if ln.strip()]
    ann_text = '\n\n'.join(blocks) + ('\n' if blocks else '')
    return tags, ann_text


# ---------------------------------------------------------------------------
# 1. skip-existing semantics
# ---------------------------------------------------------------------------


class TestSkipExisting(unittest.TestCase):

    def test_skip_existing_skips_doc_with_ann(self):
        """skip_existing checks the OUTPUT DB for an existing
        article.txt.ann.  A source-side .ann (the hand-annotated
        golden ground truth) is unrelated."""
        input_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})
        output_db = FakeCouchDb({
            'doc_a': _synth_doc(
                'doc_a', existing_ann=b'[@old#Description*]\n',
            ),
        })

        with mock.patch(
            'predict_v4.predict_doc', side_effect=_stub_predict,
        ):
            counts = predict_v4.predict_all(
                input_db, output_db,
                layout_crf=mock.MagicMock(),
                treatment_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=True, force=False,
                dry_run=False, limit=None,
                verbosity=0,
            )
        self.assertEqual(counts['skipped_existing'], 1)
        self.assertEqual(counts['predicted'], 0)
        self.assertEqual(output_db.puts, [])

    def test_force_overrides_skip_existing(self):
        """When both skip_existing=True and force=True, force wins."""
        input_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})
        output_db = FakeCouchDb({
            'doc_a': _synth_doc(
                'doc_a', existing_ann=b'[@old#Description*]\n',
            ),
        })

        with mock.patch(
            'predict_v4.predict_doc', side_effect=_stub_predict,
        ):
            counts = predict_v4.predict_all(
                input_db, output_db,
                layout_crf=mock.MagicMock(),
                treatment_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=True, force=True,
                dry_run=False, limit=None,
                verbosity=0,
            )
        self.assertEqual(counts['skipped_existing'], 0)
        self.assertEqual(counts['predicted'], 1)
        self.assertEqual(len(output_db.puts), 1)
        self.assertEqual(output_db.puts[0][1], 'article.txt.ann')


# ---------------------------------------------------------------------------
# 2. missing-attachment handling
# ---------------------------------------------------------------------------


class TestMissingAttachments(unittest.TestCase):

    def test_missing_spans_skips_with_warning(self):
        input_db = FakeCouchDb({
            'doc_a': _synth_doc('doc_a', spans_json=None),
        })
        output_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})

        with mock.patch(
            'predict_v4.predict_doc', side_effect=_stub_predict,
        ) as stub:
            counts = predict_v4.predict_all(
                input_db, output_db,
                layout_crf=mock.MagicMock(),
                treatment_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=False, force=False,
                dry_run=False, limit=None,
                verbosity=0,
            )
        self.assertEqual(counts['skipped_no_attachments'], 1)
        self.assertEqual(counts['predicted'], 0)
        stub.assert_not_called()
        self.assertEqual(output_db.puts, [])

    def test_missing_page_headers_skips_with_warning(self):
        input_db = FakeCouchDb({
            'doc_a': _synth_doc('doc_a', page_headers_json=None),
        })
        output_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})

        with mock.patch(
            'predict_v4.predict_doc', side_effect=_stub_predict,
        ):
            counts = predict_v4.predict_all(
                input_db, output_db,
                layout_crf=mock.MagicMock(),
                treatment_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=False, force=False,
                dry_run=False, limit=None,
                verbosity=0,
            )
        self.assertEqual(counts['skipped_no_attachments'], 1)
        self.assertEqual(counts['predicted'], 0)

    def test_missing_plaintext_skips_with_warning(self):
        """No article.txt and no PDF and no .ann to strip — skip."""
        input_db = FakeCouchDb({
            'doc_a': _synth_doc(
                'doc_a',
                plaintext=b'',  # placeholder; remove below
            ),
        })
        # Remove the plaintext attachment so all three fallback
        # sources are absent.
        del input_db.docs['doc_a']['_attachments']['article.txt']
        output_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})

        with mock.patch(
            'predict_v4.predict_doc', side_effect=_stub_predict,
        ):
            counts = predict_v4.predict_all(
                input_db, output_db,
                layout_crf=mock.MagicMock(),
                treatment_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=False, force=False,
                dry_run=False, limit=None,
                verbosity=0,
            )
        self.assertEqual(counts['skipped_no_plaintext'], 1)
        self.assertEqual(counts['predicted'], 0)


# ---------------------------------------------------------------------------
# 3. write path + dry-run + limit
# ---------------------------------------------------------------------------


class TestWritePath(unittest.TestCase):

    def test_writes_ann_to_output_db(self):
        input_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})
        output_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})

        with mock.patch(
            'predict_v4.predict_doc', side_effect=_stub_predict,
        ):
            counts = predict_v4.predict_all(
                input_db, output_db,
                layout_crf=mock.MagicMock(),
                treatment_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=False, force=False,
                dry_run=False, limit=None,
                verbosity=0,
            )
        self.assertEqual(counts['predicted'], 1)
        self.assertEqual(len(output_db.puts), 1)
        doc_id, name, body, content_type = output_db.puts[0]
        self.assertEqual(doc_id, 'doc_a')
        self.assertEqual(name, 'article.txt.ann')
        self.assertIn(b'Description', body)
        self.assertEqual(content_type, 'text/plain')

    def test_dry_run_does_not_write(self):
        input_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})
        output_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})

        with mock.patch(
            'predict_v4.predict_doc', side_effect=_stub_predict,
        ) as stub:
            counts = predict_v4.predict_all(
                input_db, output_db,
                layout_crf=mock.MagicMock(),
                treatment_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=False, force=False,
                dry_run=True, limit=None,
                verbosity=0,
            )
        # Predict still runs (so dry-run can report what WOULD be
        # written) but nothing reaches the output DB.
        stub.assert_called()
        self.assertEqual(counts['predicted'], 1)
        self.assertEqual(output_db.puts, [])

    def test_limit_stops_after_n_docs(self):
        input_db = FakeCouchDb({
            f'doc_{i}': _synth_doc(f'doc_{i}') for i in range(5)
        })
        output_db = FakeCouchDb({
            f'doc_{i}': _synth_doc(f'doc_{i}') for i in range(5)
        })

        with mock.patch(
            'predict_v4.predict_doc', side_effect=_stub_predict,
        ):
            counts = predict_v4.predict_all(
                input_db, output_db,
                layout_crf=mock.MagicMock(),
                treatment_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=False, force=False,
                dry_run=False, limit=2,
                verbosity=0,
            )
        self.assertEqual(counts['predicted'], 2)
        self.assertEqual(len(output_db.puts), 2)

    def test_design_docs_are_skipped(self):
        """Doc IDs starting with '_' (design docs) never reach the
        predictor."""
        input_db = FakeCouchDb({
            '_design/keep_me_out': _synth_doc('_design/keep_me_out'),
            'doc_a': _synth_doc('doc_a'),
        })
        output_db = FakeCouchDb({
            '_design/keep_me_out': _synth_doc('_design/keep_me_out'),
            'doc_a': _synth_doc('doc_a'),
        })
        with mock.patch(
            'predict_v4.predict_doc', side_effect=_stub_predict,
        ) as stub:
            predict_v4.predict_all(
                input_db, output_db,
                layout_crf=mock.MagicMock(),
                treatment_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=False, force=False,
                dry_run=False, limit=None,
                verbosity=0,
            )
        # Predictor never sees the design doc.
        self.assertEqual(stub.call_count, 1)
        self.assertEqual(len(output_db.puts), 1)
        self.assertEqual(output_db.puts[0][0], 'doc_a')


# ---------------------------------------------------------------------------
# 4. Redis-key resolution from experiment doc
# ---------------------------------------------------------------------------


class TestRedisKeyResolution(unittest.TestCase):
    """``resolve_redis_keys`` is the small CLI helper that pulls the
    pass1 / pass2 Redis keys out of the env_config dict, falling
    back to the trainer defaults if either is absent.  The dict
    layout matches what env_config produces after applying an
    experiment doc with ``redis_keys.classifier_model_pass1`` etc."""

    def test_resolved_from_experiment(self):
        config = {
            'classifier_model_key_pass1': 'skol:custom:v4_layout_hand',
            'classifier_model_key_pass2': 'skol:custom:v4_pass2_combined',
        }
        keys = predict_v4.resolve_redis_keys(config)
        self.assertEqual(
            keys['pass1_state'], 'skol:custom:v4_layout_hand',
        )
        self.assertEqual(
            keys['pass1_meta'], 'skol:custom:v4_layout_hand:meta',
        )
        self.assertEqual(
            keys['pass2_state'], 'skol:custom:v4_pass2_combined',
        )
        self.assertEqual(
            keys['pass2_meta'], 'skol:custom:v4_pass2_combined:meta',
        )

    def test_falls_back_to_default_keys_when_experiment_silent(self):
        config: Dict[str, Any] = {
            'classifier_model_key_pass1': '',
            'classifier_model_key_pass2': '',
        }
        keys = predict_v4.resolve_redis_keys(config)
        self.assertEqual(
            keys['pass1_state'], 'skol:classifier:model:v4_layout',
        )
        self.assertEqual(
            keys['pass2_state'], 'skol:classifier:model:v4_treatment',
        )

    def test_missing_keys_default_too(self):
        keys = predict_v4.resolve_redis_keys({})
        self.assertEqual(
            keys['pass1_state'], 'skol:classifier:model:v4_layout',
        )
        self.assertEqual(
            keys['pass2_state'], 'skol:classifier:model:v4_treatment',
        )


# ---------------------------------------------------------------------------
# 5. _iter_doc_ids
# ---------------------------------------------------------------------------


class TestIterDocIds(unittest.TestCase):

    def test_skips_design_docs(self):
        db = FakeCouchDb({
            '_design/foo': _synth_doc('_design/foo'),
            'doc_b': _synth_doc('doc_b'),
            'doc_c': _synth_doc('doc_c'),
        })
        ids = list(predict_v4._iter_doc_ids(db))
        self.assertEqual(ids, ['doc_b', 'doc_c'])

    def test_limit_caps_iteration(self):
        db = FakeCouchDb({f'd{i}': _synth_doc(f'd{i}') for i in range(5)})
        ids = list(predict_v4._iter_doc_ids(db, limit=3))
        self.assertEqual(len(ids), 3)


# ---------------------------------------------------------------------------
# Step 6.F: --single-crf-key dispatch + predict_all_single
# ---------------------------------------------------------------------------


def _stub_predict_single(plaintext: str, *_args, **_kwargs):
    """Stub for ``predict_doc_single`` — mirrors ``_stub_predict``."""
    lines = plaintext.split('\n')
    tags = ['Description'] * len(lines)
    blocks = [f'[@{ln}#Description*]' for ln in lines if ln.strip()]
    ann_text = '\n\n'.join(blocks) + ('\n' if blocks else '')
    return tags, ann_text


class TestSingleCRFMode(unittest.TestCase):
    """Step 6.F adds an end-to-end single-CRF inference path to
    ``predict_v4``.  These tests exercise the new
    ``predict_all_single`` loop the same way ``TestWritePath``
    exercises the two-pass ``predict_all``."""

    def test_writes_ann_to_output_db_in_single_mode(self):
        input_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})
        output_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})

        with mock.patch(
            'predict_v4.predict_doc_single',
            side_effect=_stub_predict_single,
        ):
            counts = predict_v4.predict_all_single(
                input_db, output_db,
                single_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=False, force=False,
                dry_run=False, limit=None,
                verbosity=0,
            )
        self.assertEqual(counts['predicted'], 1)
        self.assertEqual(len(output_db.puts), 1)
        self.assertEqual(output_db.puts[0][1], 'article.txt.ann')

    def test_skip_existing_skips_doc_with_ann_in_single_mode(self):
        input_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})
        output_db = FakeCouchDb({
            'doc_a': _synth_doc(
                'doc_a', existing_ann=b'[@old#Description*]\n',
            ),
        })

        with mock.patch(
            'predict_v4.predict_doc_single',
            side_effect=_stub_predict_single,
        ):
            counts = predict_v4.predict_all_single(
                input_db, output_db,
                single_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=True, force=False,
                dry_run=False, limit=None,
                verbosity=0,
            )
        self.assertEqual(counts['skipped_existing'], 1)
        self.assertEqual(counts['predicted'], 0)
        self.assertEqual(output_db.puts, [])

    def test_missing_spans_skips_with_warning_in_single_mode(self):
        input_db = FakeCouchDb({
            'doc_a': _synth_doc('doc_a', spans_json=None),
        })
        output_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})

        with mock.patch(
            'predict_v4.predict_doc_single',
            side_effect=_stub_predict_single,
        ) as stub:
            counts = predict_v4.predict_all_single(
                input_db, output_db,
                single_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=False, force=False,
                dry_run=False, limit=None,
                verbosity=0,
            )
        self.assertEqual(counts['skipped_no_attachments'], 1)
        self.assertEqual(counts['predicted'], 0)
        stub.assert_not_called()

    def test_dry_run_does_not_write_in_single_mode(self):
        input_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})
        output_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})

        with mock.patch(
            'predict_v4.predict_doc_single',
            side_effect=_stub_predict_single,
        ) as stub:
            counts = predict_v4.predict_all_single(
                input_db, output_db,
                single_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=False, force=False,
                dry_run=True, limit=None,
                verbosity=0,
            )
        stub.assert_called()
        self.assertEqual(counts['predicted'], 1)
        self.assertEqual(output_db.puts, [])

    def test_limit_stops_after_n_docs_in_single_mode(self):
        input_db = FakeCouchDb({
            f'doc_{i}': _synth_doc(f'doc_{i}') for i in range(5)
        })
        output_db = FakeCouchDb({
            f'doc_{i}': _synth_doc(f'doc_{i}') for i in range(5)
        })

        with mock.patch(
            'predict_v4.predict_doc_single',
            side_effect=_stub_predict_single,
        ):
            counts = predict_v4.predict_all_single(
                input_db, output_db,
                single_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=False, force=False,
                dry_run=False, limit=2,
                verbosity=0,
            )
        self.assertEqual(counts['predicted'], 2)
        self.assertEqual(len(output_db.puts), 2)


if __name__ == '__main__':
    unittest.main()
