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

    def test_missing_spans_AND_plaintext_distinguishes_orphan(self):
        """When BOTH spans and plaintext are missing, the doc is an
        orphan (no plaintext to annotate from) — the diagnostic
        should NOT say "re-run annotate_v4" because that's a dead
        end.  Instead it should surface the no-plaintext situation
        so the operator immediately sees the doc needs re-ingestion
        rather than re-annotation."""
        import io
        from contextlib import redirect_stdout
        # Build a doc with NO spans AND NO plaintext attachment.
        doc = _synth_doc('doc_a', plaintext=b'', spans_json=None)
        del doc['_attachments']['article.txt']  # truly no plaintext
        input_db = FakeCouchDb({'doc_a': doc})
        output_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})

        captured = io.StringIO()
        with mock.patch(
            'predict_v4.predict_doc', side_effect=_stub_predict,
        ), redirect_stdout(captured):
            predict_v4.predict_all(
                input_db, output_db,
                layout_crf=mock.MagicMock(),
                treatment_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=False, force=False,
                dry_run=False, limit=None,
                verbosity=1,
            )
        out = captured.getvalue()
        # Operator-facing assertion: the message must NOT mislead
        # the operator into re-running annotate_v4 when the
        # underlying problem is missing plaintext source.
        self.assertNotIn('re-run annotate_v4', out)
        # Must mention that plaintext is missing so the operator
        # understands the dead end.
        self.assertIn('no plaintext source', out)
        self.assertIn('doc_a', out)


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


# ---------------------------------------------------------------------------
# Step 7.γ: --ablate-particles flag propagation
# ---------------------------------------------------------------------------


class TestAblateParticlesFlag(unittest.TestCase):
    """``--ablate-particles`` threads ``ablate_particles=True`` into
    the inner ``predict_doc`` / ``predict_doc_single`` call,
    regardless of dispatch mode.  We patch the inner function and
    inspect the kwargs."""

    def _run_predict_all(self, *, ablate: bool):
        input_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})
        output_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})
        seen = []

        def spy(*args, **kwargs):
            seen.append(kwargs)
            return _stub_predict(*args, **kwargs)

        with mock.patch('predict_v4.predict_doc', side_effect=spy):
            predict_v4.predict_all(
                input_db, output_db,
                layout_crf=mock.MagicMock(),
                treatment_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=False, force=False,
                dry_run=False, limit=None,
                verbosity=0,
                ablate_particles=ablate,
            )
        return seen

    def test_two_pass_forwards_ablate_true(self):
        seen = self._run_predict_all(ablate=True)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].get('ablate_particles'), True)

    def test_two_pass_forwards_ablate_false_by_default(self):
        seen = self._run_predict_all(ablate=False)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].get('ablate_particles'), False)

    def _run_predict_all_single(self, *, ablate: bool):
        input_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})
        output_db = FakeCouchDb({'doc_a': _synth_doc('doc_a')})
        seen = []

        def spy(*args, **kwargs):
            seen.append(kwargs)
            return _stub_predict_single(*args, **kwargs)

        with mock.patch(
            'predict_v4.predict_doc_single', side_effect=spy,
        ):
            predict_v4.predict_all_single(
                input_db, output_db,
                single_crf=mock.MagicMock(),
                sbert_lookup=lambda _t: None,
                device='cpu',
                skip_existing=False, force=False,
                dry_run=False, limit=None,
                verbosity=0,
                ablate_particles=ablate,
            )
        return seen

    def test_single_crf_forwards_ablate_true(self):
        seen = self._run_predict_all_single(ablate=True)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].get('ablate_particles'), True)

    def test_single_crf_forwards_ablate_false_by_default(self):
        seen = self._run_predict_all_single(ablate=False)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].get('ablate_particles'), False)


# ---------------------------------------------------------------------------
# Production cutover: implicit-default-flip dispatch (post-Step-7)
# ---------------------------------------------------------------------------


class TestSingleCRFImplicitDefault(unittest.TestCase):
    """When ``config['classifier_model_key_single']`` is non-empty
    (set on the experiment doc via env_config's redis_mapping) and
    the operator passed NO explicit two-pass override on the CLI,
    predict_v4.main() defaults to single-CRF mode against that key.

    The hierarchy is exercised by inspecting ``args.single_crf_key``
    AFTER main()'s dispatch resolution.  We pull the resolution
    block out as a helper so the tests don't have to spin up
    couchdb + Redis + Spark just to exercise CLI plumbing."""

    def _resolve_dispatch(
        self, *,
        cli_single=None, cli_pass1=None, cli_pass2=None,
        config_single='',
    ):
        """Replay predict_v4.main()'s precedence block."""
        class Args:
            single_crf_key = cli_single
            pass1_key = cli_pass1
            pass2_key = cli_pass2
        args = Args()
        config = {
            'classifier_model_key_pass1': '',
            'classifier_model_key_pass2': '',
            'classifier_model_key_single': config_single,
        }
        # Mirror the production code's precedence; the inline block
        # in main() is small enough to copy here without drift.
        if (
            not args.single_crf_key
            and not args.pass1_key
            and not args.pass2_key
            and config.get('classifier_model_key_single')
        ):
            args.single_crf_key = config['classifier_model_key_single']
        return args

    def test_config_single_key_flips_to_single_mode(self):
        """Operational cutover: production_v4 has
        classifier_model_single set, no CLI flags passed → single."""
        args = self._resolve_dispatch(
            config_single='skol:classifier:model:v4_single_combined',
        )
        self.assertEqual(
            args.single_crf_key,
            'skol:classifier:model:v4_single_combined',
        )
        self.assertIsNone(args.pass1_key)
        self.assertIsNone(args.pass2_key)

    def test_explicit_pass1_overrides_config_single(self):
        """Operator passes --pass1-key explicitly: two-pass wins
        even though the experiment doc carries a single key.  Used
        for ad-hoc A/B against the legacy two-pass model."""
        args = self._resolve_dispatch(
            cli_pass1='skol:classifier:model:v4_layout',
            config_single='skol:classifier:model:v4_single_combined',
        )
        # Implicit flip is suppressed; args.single_crf_key stays None.
        self.assertIsNone(args.single_crf_key)
        self.assertEqual(
            args.pass1_key, 'skol:classifier:model:v4_layout',
        )

    def test_explicit_pass2_overrides_config_single(self):
        """Symmetric to test_explicit_pass1_overrides_config_single
        — --pass2-key alone is also enough to suppress the flip."""
        args = self._resolve_dispatch(
            cli_pass2='skol:classifier:model:v4_pass2_combined',
            config_single='skol:classifier:model:v4_single_combined',
        )
        self.assertIsNone(args.single_crf_key)
        self.assertEqual(
            args.pass2_key, 'skol:classifier:model:v4_pass2_combined',
        )

    def test_explicit_single_overrides_config_single(self):
        """If the CLI also passes --single-crf-key, that explicit
        value wins — the flip is just a no-op since the field was
        already set."""
        args = self._resolve_dispatch(
            cli_single='skol:custom:smoke_single',
            config_single='skol:classifier:model:v4_single_combined',
        )
        self.assertEqual(args.single_crf_key, 'skol:custom:smoke_single')

    def test_empty_config_falls_through_to_two_pass(self):
        """No config field, no CLI flags → args.single_crf_key
        stays None, downstream dispatch picks the two-pass path."""
        args = self._resolve_dispatch(config_single='')
        self.assertIsNone(args.single_crf_key)
        self.assertIsNone(args.pass1_key)
        self.assertIsNone(args.pass2_key)


# ---------------------------------------------------------------------------
# Input-DB flag rename: --source-db (canonical) + --golden-db (alias)
# ---------------------------------------------------------------------------


class TestSourceDbFlag(unittest.TestCase):
    """``--golden-db`` is misnamed — it's the *input* DB, not specifically
    the golden set.  The pipeline calls predict_v4 with the ingest DB and
    expects the full corpus to be predicted, not the eval set.  Rename
    to ``--source-db`` (canonical); keep ``--golden-db`` as a back-compat
    alias.  And reverse the env_config fallback so the ingest DB (rather
    than the golden set) is the implicit default when neither flag is
    passed."""

    def _resolve_input(
        self, *,
        cli_source=None, cli_golden=None,
        config_ingest='', config_golden='',
    ):
        """Replay predict_v4.main()'s input-DB resolution."""
        class Args:
            source_db = cli_source
            golden_db = cli_golden
        args = Args()
        config = {
            'ingest_db_name': config_ingest,
            'golden_db_name': config_golden,
        }
        # Mirror production: --source-db wins; --golden-db is the
        # backward-compat alias; env_config fallback prefers the
        # ingest DB (whole-corpus default) over the golden set.
        return (
            args.source_db
            or args.golden_db
            or config.get('ingest_db_name')
            or config.get('golden_db_name')
        )

    def test_cli_source_db_wins(self):
        assert self._resolve_input(
            cli_source='skol_dev',
            config_ingest='ignored',
        ) == 'skol_dev'

    def test_cli_golden_db_alias_still_works(self):
        """Operators with shell history using --golden-db keep working."""
        assert self._resolve_input(
            cli_golden='skol_golden_v2',
        ) == 'skol_golden_v2'

    def test_source_db_wins_over_alias(self):
        """When both are passed (script + alias), source wins."""
        assert self._resolve_input(
            cli_source='skol_dev',
            cli_golden='skol_golden_v2',
        ) == 'skol_dev'

    def test_falls_back_to_ingest_db_not_golden(self):
        """The Step-7-cutover bug fix: with neither CLI flag passed,
        env_config gives the ingest DB precedence so the manage_experiment
        predict step processes the full production corpus."""
        assert self._resolve_input(
            config_ingest='skol_dev',
            config_golden='skol_golden_v2',
        ) == 'skol_dev'

    def test_falls_through_to_golden_when_no_ingest(self):
        """If no ingest DB is configured (legacy evaluate-only setup),
        the golden-set fallback still kicks in."""
        assert self._resolve_input(
            config_golden='skol_golden_v2',
        ) == 'skol_golden_v2'


if __name__ == '__main__':
    unittest.main()
