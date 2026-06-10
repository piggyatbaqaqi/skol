"""Tests for bin/embed_lines.py — per-line SBERT cache writer.

Pure-helper tests (sha256 key derivation, line iteration, plaintext source
chain) need no infrastructure. Integration tests touch live Redis at
localhost:6379 under the namespace ``skol:sbert:test:<uuid>:`` and clean
up in tearDown; they are skipped if Redis is not reachable.

The SBERT encoder is replaced with a deterministic fake in every test so
no model weights are loaded.
"""
from __future__ import annotations

import hashlib
import sys
import unittest
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import create_redis_client  # noqa: E402

from embed_lines import (  # type: ignore[import]  # noqa: E402
    LineEmbedder,
    _acquire_lock,
    _release_lock,
    _resolve_skip_existing,
    iter_unique_lines,
    load_plaintext,
    process_doc,
    redis_key,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _redis_available() -> bool:
    try:
        r = create_redis_client(decode_responses=False)
        r.ping()
        return True
    except Exception:
        return False


class FakeDb:
    """In-memory stand-in for a couchdb.Database supporting get_attachment."""

    def __init__(self, docs: Dict[str, Dict[str, bytes]]) -> None:
        self._docs = docs

    def get_attachment(self, doc_id: str, name: str) -> Optional[bytes]:
        return self._docs.get(doc_id, {}).get(name)


class FakeEncoder:
    """Deterministic stand-in for SentenceTransformer.encode.

    Same input -> same vector, different input -> different vector
    (with overwhelming probability). Records call_count so tests can
    assert the encoder wasn't re-invoked on cached lines.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.call_count = 0
        self.last_batch: List[str] = []

    def __call__(self, lines: List[str]) -> np.ndarray:
        self.call_count += 1
        self.last_batch = list(lines)
        out = np.zeros((len(lines), self.dim), dtype=np.float32)
        for i, line in enumerate(lines):
            out[i, 0] = float(len(line) % 10)
            out[i, 1] = float(sum(ord(c) for c in line[:8]) % 100)
        return out


# Sentinel vector used in --force tests; first slot is 99.0, well outside
# the FakeEncoder's range (which puts len%10 in slot 0).
_SENTINEL_VEC = np.full(768, 99.0, dtype=np.float32)


def _make_redis_namespace() -> str:
    """Per-test prefix so parallel test runs and crashes don't collide."""
    return f'skol:sbert:test:{uuid.uuid4().hex[:8]}:'


# ---------------------------------------------------------------------------
# 1. Pure helpers — redis_key()
# ---------------------------------------------------------------------------


class TestRedisKeyDerivation(unittest.TestCase):
    """sha256 key derivation must be stable so downstream consumers
    (Step 2 feature assembler) can compute keys independently."""

    def test_sha256_key_derivation_mpnet(self):
        h = hashlib.sha256(b'Hello world').hexdigest()
        self.assertEqual(
            redis_key('Hello world', 'mpnet'),
            f'skol:sbert:mpnet:{h}',
        )

    def test_sha256_key_derivation_minilm(self):
        h = hashlib.sha256(b'Hello world').hexdigest()
        self.assertEqual(
            redis_key('Hello world', 'minilm'),
            f'skol:sbert:minilm:{h}',
        )

    def test_sha256_unicode_utf8_encoded(self):
        line = 'Cintractiella from Micronesia — π'
        h = hashlib.sha256(line.encode('utf-8')).hexdigest()
        self.assertEqual(
            redis_key(line, 'mpnet'),
            f'skol:sbert:mpnet:{h}',
        )

    def test_custom_prefix(self):
        h = hashlib.sha256(b'foo').hexdigest()
        self.assertEqual(
            redis_key('foo', 'mpnet', prefix='skol:sbert:test:'),
            f'skol:sbert:test:mpnet:{h}',
        )


# ---------------------------------------------------------------------------
# 2. Pure helpers — iter_unique_lines()
# ---------------------------------------------------------------------------


class TestIterUniqueLines(unittest.TestCase):
    """Splits plaintext on '\\n' (matching line_mode.py), drops empty /
    whitespace-only lines, dedupes in first-seen order."""

    def test_skips_empty_and_whitespace_only(self):
        plaintext = 'first line\n\n   \n\t\nsecond line\n'
        self.assertEqual(
            list(iter_unique_lines(plaintext)),
            ['first line', 'second line'],
        )

    def test_dedupes_first_seen(self):
        plaintext = 'a\nb\na\nc\nb\n'
        self.assertEqual(
            list(iter_unique_lines(plaintext)),
            ['a', 'b', 'c'],
        )

    def test_preserves_internal_whitespace(self):
        """Lines keep their internal whitespace verbatim — we hash the
        line as-is so Step 2 can match the same line text exactly."""
        plaintext = '  leading spaces\ntrailing spaces  \n'
        self.assertEqual(
            list(iter_unique_lines(plaintext)),
            ['  leading spaces', 'trailing spaces  '],
        )

    def test_empty_string_yields_nothing(self):
        self.assertEqual(list(iter_unique_lines('')), [])


# ---------------------------------------------------------------------------
# 3. Pure helpers — load_plaintext() and its 3-path fallback chain
# ---------------------------------------------------------------------------


class TestLoadPlaintext(unittest.TestCase):
    """3-path fallback chain:
    article.txt  ->  article.pdf  ->  article.txt.ann"""

    def test_path_precedence_txt_wins(self):
        """When all three attachments are present, article.txt wins; the
        PDF and YEDDA helpers are not consulted."""
        db = FakeDb({'doc1': {
            'article.txt': b'plain text from txt',
            'article.pdf': b'%PDF-1.4 dummy bytes',
            'article.txt.ann': b'[@yedda block#Nomenclature*]',
        }})
        with mock.patch('embed_lines.plaintext_from_pdf') as mock_pdf, \
             mock.patch('embed_lines.plaintext_from_yedda') as mock_yedda:
            text, source = load_plaintext(db, 'doc1')
        self.assertEqual(text, 'plain text from txt')
        self.assertEqual(source, 'article.txt')
        mock_pdf.assert_not_called()
        mock_yedda.assert_not_called()

    def test_pdf_path_when_no_txt(self):
        """No article.txt but article.pdf present -> PDF helper called.
        Covers the ~160 hand-annotated PDF docs in the training corpus."""
        db = FakeDb({'doc1': {
            'article.pdf': b'%PDF-1.4 fake',
            'article.txt.ann': b'[@block#Description*]',
        }})
        with mock.patch('embed_lines.plaintext_from_pdf',
                        return_value='extracted from pdf') as mock_pdf, \
             mock.patch('embed_lines.plaintext_from_yedda') as mock_yedda:
            text, source = load_plaintext(db, 'doc1')
        self.assertEqual(text, 'extracted from pdf')
        self.assertEqual(source, 'article.pdf')
        mock_pdf.assert_called_once_with(b'%PDF-1.4 fake')
        mock_yedda.assert_not_called()

    def test_yedda_fallback_when_only_ann(self):
        """Only article.txt.ann present -> YEDDA helper called.
        Covers the ~1724 JATS-derived docs that have only the .ann file."""
        db = FakeDb({'doc1': {
            'article.txt.ann':
                b'[@first block#Nomenclature*][@second#Description*]',
        }})
        with mock.patch('embed_lines.plaintext_from_yedda',
                        return_value='first block\n\nsecond') as mock_yedda:
            text, source = load_plaintext(db, 'doc1')
        self.assertEqual(text, 'first block\n\nsecond')
        self.assertEqual(source, 'article.txt.ann')
        mock_yedda.assert_called_once()

    def test_missing_returns_none(self):
        """Doc with no recognized attachment yields (None, 'missing')."""
        db = FakeDb({'doc1': {}})
        text, source = load_plaintext(db, 'doc1')
        self.assertIsNone(text)
        self.assertEqual(source, 'missing')


# ---------------------------------------------------------------------------
# 4. Pure helper — _resolve_skip_existing()
# ---------------------------------------------------------------------------


class TestResolveSkipExisting(unittest.TestCase):
    """Idempotent-by-default semantics for the cache builder:
    skip_existing is True unless ``--force`` is set.  env_config's
    own ``skip_existing`` field is ignored because it hardcodes False
    as the env-var default — trusting it would re-embed every cached
    line on every run (which is what happened in the first 1884-doc
    pass: cached_hits stayed at 0)."""

    def test_default_no_force_is_skip(self):
        """Empty / None / explicit-False force => skip cached lines."""
        self.assertTrue(_resolve_skip_existing({}))
        self.assertTrue(_resolve_skip_existing({'force': None}))
        self.assertTrue(_resolve_skip_existing({'force': False}))

    def test_force_overrides_to_no_skip(self):
        """--force re-embeds everything regardless of cache state."""
        self.assertFalse(_resolve_skip_existing({'force': True}))

    def test_skip_existing_field_is_ignored(self):
        """env_config's own skip_existing value never affects the
        decision — only --force matters."""
        self.assertTrue(_resolve_skip_existing({'skip_existing': False}))
        self.assertTrue(_resolve_skip_existing({'skip_existing': True}))
        self.assertFalse(
            _resolve_skip_existing({'skip_existing': True, 'force': True}),
        )
        self.assertTrue(
            _resolve_skip_existing({'skip_existing': False, 'force': False}),
        )


# ---------------------------------------------------------------------------
# 5. Integration — process_doc + LineEmbedder against live Redis
# ---------------------------------------------------------------------------


@unittest.skipUnless(_redis_available(), 'Requires Redis (env_config)')
class TestProcessDocIdempotent(unittest.TestCase):
    """Re-running on the same doc must not re-invoke the encoder.
    This also covers the silent-recompute failure mode behind the
    `redis lost an embedding` memory note."""

    def setUp(self):
        self.prefix = _make_redis_namespace()
        self.r = create_redis_client(decode_responses=False)
        self.encoder = FakeEncoder(dim=768)
        self.embedder = LineEmbedder(
            model_tag='mpnet',
            redis_client=self.r,
            dim=768,
            encoder=self.encoder,
            key_prefix=self.prefix,
        )

    def tearDown(self):
        for key in self.r.scan_iter(match=f'{self.prefix}*'):
            self.r.delete(key)

    def test_idempotent_rerun_skips_encoder(self):
        db = FakeDb({'doc1': {
            'article.txt': b'line one\nline two\nline one',
        }})
        counts1 = process_doc(db, 'doc1', self.embedder,
                              skip_existing=True, force=False)
        self.assertEqual(counts1['unique'], 2)
        self.assertEqual(counts1['embedded'], 2)
        self.assertEqual(self.encoder.call_count, 1)

        counts2 = process_doc(db, 'doc1', self.embedder,
                              skip_existing=True, force=False)
        self.assertEqual(counts2['cached_hits'], 2)
        self.assertEqual(counts2['embedded'], 0)
        self.assertEqual(
            self.encoder.call_count, 1,
            'Encoder must not be invoked when all lines are already cached',
        )


@unittest.skipUnless(_redis_available(), 'Requires Redis (env_config)')
class TestProcessDocForce(unittest.TestCase):
    """--force replaces any pre-existing cached vector."""

    def setUp(self):
        self.prefix = _make_redis_namespace()
        self.r = create_redis_client(decode_responses=False)
        self.encoder = FakeEncoder(dim=768)
        self.embedder = LineEmbedder(
            model_tag='mpnet',
            redis_client=self.r,
            dim=768,
            encoder=self.encoder,
            key_prefix=self.prefix,
        )

    def tearDown(self):
        for key in self.r.scan_iter(match=f'{self.prefix}*'):
            self.r.delete(key)

    def test_force_overwrites_sentinel(self):
        line = 'a real line'
        key = redis_key(line, 'mpnet', prefix=self.prefix)
        self.r.set(key, _SENTINEL_VEC.tobytes())

        db = FakeDb({'doc1': {'article.txt': line.encode('utf-8')}})
        process_doc(db, 'doc1', self.embedder,
                    skip_existing=True, force=False)
        self.assertEqual(
            self.r.get(key), _SENTINEL_VEC.tobytes(),
            'Without --force, sentinel must survive',
        )

        process_doc(db, 'doc1', self.embedder,
                    skip_existing=False, force=True)
        after = np.frombuffer(self.r.get(key), dtype=np.float32)
        self.assertNotEqual(
            float(after[0]), float(_SENTINEL_VEC[0]),
            'With --force, encoder output must overwrite the sentinel',
        )


@unittest.skipUnless(_redis_available(), 'Requires Redis (env_config)')
class TestProcessDocEmptyLines(unittest.TestCase):
    """Empty / whitespace-only lines must never produce a Redis key."""

    def setUp(self):
        self.prefix = _make_redis_namespace()
        self.r = create_redis_client(decode_responses=False)
        self.encoder = FakeEncoder(dim=768)
        self.embedder = LineEmbedder(
            model_tag='mpnet',
            redis_client=self.r,
            dim=768,
            encoder=self.encoder,
            key_prefix=self.prefix,
        )

    def tearDown(self):
        for key in self.r.scan_iter(match=f'{self.prefix}*'):
            self.r.delete(key)

    def test_empty_lines_not_written(self):
        plaintext = 'real line\n\n   \n\t\nanother real line\n'
        db = FakeDb({'doc1': {'article.txt': plaintext.encode('utf-8')}})
        counts = process_doc(db, 'doc1', self.embedder,
                             skip_existing=True, force=False)
        self.assertEqual(counts['unique'], 2)
        self.assertEqual(counts['embedded'], 2)
        for bad in ('', '   ', '\t'):
            bad_key = redis_key(bad, 'mpnet', prefix=self.prefix)
            self.assertFalse(
                self.r.exists(bad_key),
                f'Empty/whitespace line {bad!r} produced a key',
            )


@unittest.skipUnless(_redis_available(), 'Requires Redis (env_config)')
class TestModelNamespaceIsolation(unittest.TestCase):
    """Same line under different models -> different keys, different
    vectors. Confirms the dual-model design doesn't collide."""

    def setUp(self):
        self.prefix = _make_redis_namespace()
        self.r = create_redis_client(decode_responses=False)

    def tearDown(self):
        for key in self.r.scan_iter(match=f'{self.prefix}*'):
            self.r.delete(key)

    def test_distinct_keys_distinct_vectors(self):
        line = 'shared line text'
        mpnet_embedder = LineEmbedder(
            model_tag='mpnet', redis_client=self.r, dim=768,
            encoder=FakeEncoder(dim=768), key_prefix=self.prefix,
        )
        minilm_embedder = LineEmbedder(
            model_tag='minilm', redis_client=self.r, dim=384,
            encoder=FakeEncoder(dim=384), key_prefix=self.prefix,
        )

        db = FakeDb({'doc1': {'article.txt': line.encode('utf-8')}})
        process_doc(db, 'doc1', mpnet_embedder,
                    skip_existing=True, force=False)
        process_doc(db, 'doc1', minilm_embedder,
                    skip_existing=True, force=False)

        mpnet_key = redis_key(line, 'mpnet', prefix=self.prefix)
        minilm_key = redis_key(line, 'minilm', prefix=self.prefix)
        self.assertNotEqual(mpnet_key, minilm_key)

        mpnet_vec = np.frombuffer(
            self.r.get(mpnet_key), dtype=np.float32)
        minilm_vec = np.frombuffer(
            self.r.get(minilm_key), dtype=np.float32)
        self.assertEqual(mpnet_vec.shape, (768,))
        self.assertEqual(minilm_vec.shape, (384,))


@unittest.skipUnless(_redis_available(), 'Requires Redis (env_config)')
class TestProcessDocPdfFallback(unittest.TestCase):
    """Path #2 of the fallback chain: article.pdf when no article.txt."""

    def setUp(self):
        self.prefix = _make_redis_namespace()
        self.r = create_redis_client(decode_responses=False)
        self.encoder = FakeEncoder(dim=768)
        self.embedder = LineEmbedder(
            model_tag='mpnet', redis_client=self.r, dim=768,
            encoder=self.encoder, key_prefix=self.prefix,
        )

    def tearDown(self):
        for key in self.r.scan_iter(match=f'{self.prefix}*'):
            self.r.delete(key)

    def test_pdf_fallback_embeds_extracted_lines(self):
        db = FakeDb({'doc1': {
            'article.pdf': b'%PDF-1.4 fake',
            'article.txt.ann': b'[@should not be used#Notes*]',
        }})
        with mock.patch('embed_lines.plaintext_from_pdf',
                        return_value='line A\nline B\n'):
            counts = process_doc(db, 'doc1', self.embedder,
                                 skip_existing=True, force=False)
        self.assertEqual(counts['unique'], 2)
        self.assertEqual(counts['embedded'], 2)
        for line in ('line A', 'line B'):
            key = redis_key(line, 'mpnet', prefix=self.prefix)
            self.assertTrue(self.r.exists(key))


@unittest.skipUnless(_redis_available(), 'Requires Redis (env_config)')
class TestProcessDocYeddaFallback(unittest.TestCase):
    """Path #3 of the fallback chain: article.txt.ann only.  Covers
    the ~1724 JATS-derived docs in skol_training_v3_combined_no_golden."""

    def setUp(self):
        self.prefix = _make_redis_namespace()
        self.r = create_redis_client(decode_responses=False)
        self.encoder = FakeEncoder(dim=768)
        self.embedder = LineEmbedder(
            model_tag='mpnet', redis_client=self.r, dim=768,
            encoder=self.encoder, key_prefix=self.prefix,
        )

    def tearDown(self):
        for key in self.r.scan_iter(match=f'{self.prefix}*'):
            self.r.delete(key)

    def test_yedda_fallback_embeds_stripped_lines(self):
        db = FakeDb({'doc1': {
            'article.txt.ann': b'[@some block#Nomenclature*]',
        }})
        with mock.patch(
            'embed_lines.plaintext_from_yedda',
            return_value='yedda line one\nyedda line two\n',
        ):
            counts = process_doc(db, 'doc1', self.embedder,
                                 skip_existing=True, force=False)
        self.assertEqual(counts['unique'], 2)
        self.assertEqual(counts['embedded'], 2)
        for line in ('yedda line one', 'yedda line two'):
            key = redis_key(line, 'mpnet', prefix=self.prefix)
            self.assertTrue(self.r.exists(key))


# ---------------------------------------------------------------------------
# Build lock — value carries hostname:pid for "is the holder dead?" probes
# ---------------------------------------------------------------------------


class TestBuildLockHolderIdentity(unittest.TestCase):
    """The build-lock value carries ``<hostname>:<pid>`` so an
    operator who sees a stuck lock can immediately tell which
    process to check (``ps -p <pid>`` / ``/proc/<pid>/cmdline``)
    instead of having to scan all hosts.  Lock SEMANTICS are
    unchanged — only the value content changes; nothing in the
    codebase reads the value today."""

    def setUp(self):
        self.r = create_redis_client(decode_responses=False)
        # Distinct key per test so the suite is order-independent.
        self.lock_key = (
            f'skol:build:sbert:test-{uuid.uuid4().hex[:8]}:lock'
        )

    def tearDown(self):
        self.r.delete(self.lock_key)

    def test_acquire_stores_hostname_and_pid(self):
        import os, socket
        ok = _acquire_lock(self.r, self.lock_key, verbosity=0)
        self.assertTrue(ok)
        value = self.r.get(self.lock_key)
        self.assertIsNotNone(value)
        decoded = value.decode()
        expected = f'{socket.gethostname()}:{os.getpid()}'
        self.assertEqual(decoded, expected)

    def test_acquire_collision_reports_existing_holder(self):
        """When a second invocation collides, the diagnostic
        prints WHO currently holds the lock — that's the whole
        point of the value change."""
        import io
        from contextlib import redirect_stdout
        self.assertTrue(
            _acquire_lock(self.r, self.lock_key, verbosity=1),
        )
        existing_value = self.r.get(self.lock_key).decode()
        # Simulate a second invocation hitting the held lock.
        captured = io.StringIO()
        with redirect_stdout(captured):
            ok = _acquire_lock(self.r, self.lock_key, verbosity=1)
        self.assertFalse(ok)
        out = captured.getvalue()
        # The collision message must surface WHO holds it so an
        # operator can decide whether the holder is still alive.
        self.assertIn(existing_value, out)
        self.assertIn(self.lock_key, out)

    def test_release_deletes_value(self):
        _acquire_lock(self.r, self.lock_key, verbosity=0)
        self.assertIsNotNone(self.r.get(self.lock_key))
        _release_lock(self.r, self.lock_key, verbosity=0)
        self.assertIsNone(self.r.get(self.lock_key))

    def test_value_is_not_legacy_building_string(self):
        """Regression: the old value was the literal ``b'building'``.
        This test pins the new shape so a future refactor doesn't
        silently revert."""
        _acquire_lock(self.r, self.lock_key, verbosity=0)
        value = self.r.get(self.lock_key)
        self.assertNotEqual(value, b'building')


if __name__ == '__main__':
    unittest.main()
