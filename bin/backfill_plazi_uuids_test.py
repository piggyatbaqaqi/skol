"""Tests for fixes/backfill_plazi_uuids.py."""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_plazi_uuids import (  # type: ignore[import]  # noqa: E402
    compute_plazi_update,
    iter_doi_docs,
    parse_plazi_response,
    process_doc,
    query_plazi,
    save_with_retry,
    should_skip,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso(dt: Optional[datetime] = None) -> str:
    """Match the format the script writes: '...Z'-suffix UTC ISO 8601."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


class FakeResponse:
    def __init__(self, status_code: int, body: Any = None) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        return self._body


class FakeHttpClient:
    """Stand-in for RateLimitedHttpClient.get()."""

    def __init__(
        self,
        responses: Optional[Dict[str, FakeResponse]] = None,
        default: Optional[FakeResponse] = None,
    ) -> None:
        self.responses: Dict[str, FakeResponse] = responses or {}
        self.default = default or FakeResponse(200, [])
        self.urls_seen: List[str] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.urls_seen.append(url)
        return self.responses.get(url, self.default)


# ---------------------------------------------------------------------------
# parse_plazi_response
# ---------------------------------------------------------------------------


class TestParsePlaziResponse(unittest.TestCase):
    """Convert the Plazi API JSON array to (uuids, lnk_dois) tuple."""

    def test_with_hits(self):
        body = [
            {'DocCount': 1, 'DocUuid': 'AAAA', 'LnkDoi': '10.5281/zenodo.1'},
            {'DocCount': 1, 'DocUuid': 'BBBB', 'LnkDoi': '10.5281/zenodo.2'},
        ]
        uuids, lnk_dois = parse_plazi_response(body)
        self.assertEqual(uuids, ['AAAA', 'BBBB'])
        self.assertEqual(lnk_dois, ['10.5281/zenodo.1', '10.5281/zenodo.2'])

    def test_empty_array(self):
        uuids, lnk_dois = parse_plazi_response([])
        self.assertEqual(uuids, [])
        self.assertEqual(lnk_dois, [])

    def test_entry_without_docuuid_is_skipped(self):
        """Defensive: API shape drift shouldn't crash us."""
        body = [
            {'DocCount': 1, 'LnkDoi': '10.5281/zenodo.x'},  # no DocUuid
            {'DocCount': 1, 'DocUuid': 'CCCC', 'LnkDoi': '10.5281/zenodo.y'},
        ]
        uuids, lnk_dois = parse_plazi_response(body)
        self.assertEqual(uuids, ['CCCC'])
        self.assertEqual(lnk_dois, ['10.5281/zenodo.y'])

    def test_entry_without_lnkdoi_keeps_uuid(self):
        """LnkDoi can legitimately be absent for some treatments."""
        body = [{'DocCount': 1, 'DocUuid': 'DDDD'}]
        uuids, lnk_dois = parse_plazi_response(body)
        self.assertEqual(uuids, ['DDDD'])
        self.assertEqual(lnk_dois, [''])


# ---------------------------------------------------------------------------
# compute_plazi_update
# ---------------------------------------------------------------------------


class TestComputePlaziUpdate(unittest.TestCase):
    """Build the dict that goes at ``doc['plazi']``."""

    def test_with_hits(self):
        body = [{'DocCount': 1, 'DocUuid': 'AAAA', 'LnkDoi': '10.5281/x'}]
        now = '2026-06-03T12:00:00Z'
        update = compute_plazi_update(body, now)
        self.assertEqual(update['uuids'], ['AAAA'])
        self.assertEqual(update['lnk_dois'], ['10.5281/x'])
        self.assertEqual(update['looked_up_at'], now)
        self.assertEqual(update['source'], 'plazi:GgSrvApi:v1')

    def test_empty_still_stamps(self):
        """An empty Plazi hit still produces a fully-formed plazi
        dict — proves the doc was checked.  Idempotent re-runs depend
        on the ``looked_up_at`` being present."""
        update = compute_plazi_update([], '2026-06-03T12:00:00Z')
        self.assertEqual(update['uuids'], [])
        self.assertEqual(update['lnk_dois'], [])
        self.assertEqual(update['looked_up_at'], '2026-06-03T12:00:00Z')
        self.assertEqual(update['source'], 'plazi:GgSrvApi:v1')


# ---------------------------------------------------------------------------
# should_skip
# ---------------------------------------------------------------------------


class TestShouldSkip(unittest.TestCase):
    """Per CLAUDE.md rule 11, default behaviour is idempotent: docs
    already-checked recently are skipped.  ``--force`` overrides;
    ``--re-check-after-days N`` re-queries entries older than N
    days."""

    NOW = '2026-06-03T12:00:00Z'

    def test_no_plazi_field_proceed(self):
        self.assertFalse(should_skip(
            {'doi': '10.1/x'},
            force=False, re_check_after_days=365, now_iso=self.NOW,
        ))

    def test_recent_lookup_skip(self):
        recent = (datetime(2026, 6, 1, tzinfo=timezone.utc)
                  .strftime('%Y-%m-%dT%H:%M:%SZ'))
        self.assertTrue(should_skip(
            {'doi': '10.1/x', 'plazi': {'looked_up_at': recent}},
            force=False, re_check_after_days=365, now_iso=self.NOW,
        ))

    def test_stale_lookup_proceed(self):
        """Older than re_check_after_days → re-query."""
        old = (datetime(2025, 1, 1, tzinfo=timezone.utc)
               .strftime('%Y-%m-%dT%H:%M:%SZ'))
        self.assertFalse(should_skip(
            {'doi': '10.1/x', 'plazi': {'looked_up_at': old}},
            force=False, re_check_after_days=365, now_iso=self.NOW,
        ))

    def test_force_overrides_recent(self):
        recent = (datetime(2026, 6, 1, tzinfo=timezone.utc)
                  .strftime('%Y-%m-%dT%H:%M:%SZ'))
        self.assertFalse(should_skip(
            {'doi': '10.1/x', 'plazi': {'looked_up_at': recent}},
            force=True, re_check_after_days=365, now_iso=self.NOW,
        ))

    def test_malformed_plazi_field_proceed(self):
        """Defensive: a broken / non-dict ``plazi`` field shouldn't
        be treated as a successful lookup."""
        self.assertFalse(should_skip(
            {'doi': '10.1/x', 'plazi': 'garbage'},
            force=False, re_check_after_days=365, now_iso=self.NOW,
        ))


# ---------------------------------------------------------------------------
# query_plazi
# ---------------------------------------------------------------------------


class TestQueryPlazi(unittest.TestCase):
    """The HTTP layer.  We exercise it through a fake RateLimitedHttpClient
    so the tests don't hit Plazi for real."""

    PLAZI_URL = 'https://api.plazi.org/GgSrvApi/v1'

    def test_returns_array_on_200(self):
        body = [{'DocCount': 1, 'DocUuid': 'AAAA', 'LnkDoi': '10.5281/x'}]
        client = FakeHttpClient(default=FakeResponse(200, body))
        result = query_plazi(
            '10.3897/mycokeys.99.107606',
            plazi_url=self.PLAZI_URL,
            http_client=client,
        )
        self.assertEqual(result, body)

    def test_returns_none_on_non_200(self):
        client = FakeHttpClient(default=FakeResponse(503, None))
        result = query_plazi(
            '10.3897/mycokeys.99.107606',
            plazi_url=self.PLAZI_URL,
            http_client=client,
        )
        self.assertIsNone(result)

    def test_constructs_correct_url(self):
        """DOI is URL-encoded, format=json is set, the path matches
        the OpenAPI spec exactly (``/Treatments/searchByDOI``)."""
        client = FakeHttpClient(default=FakeResponse(200, []))
        query_plazi(
            '10.5281/zenodo.7105224',
            plazi_url=self.PLAZI_URL,
            http_client=client,
        )
        self.assertEqual(len(client.urls_seen), 1)
        url = client.urls_seen[0]
        self.assertIn('/Treatments/searchByDOI', url)
        self.assertIn('format=json', url)
        # Slashes in the DOI must be percent-encoded — without it the
        # Plazi router parses ``10.5281`` as the DOI and ``zenodo.X``
        # as part of the path.
        self.assertIn('DOI=10.5281%2Fzenodo.7105224', url)

    def test_trims_trailing_slash_on_plazi_url(self):
        """A trailing slash on ``--plazi-url`` shouldn't produce a
        ``//Treatments/...`` URL."""
        client = FakeHttpClient(default=FakeResponse(200, []))
        query_plazi(
            '10.1/x',
            plazi_url=self.PLAZI_URL + '/',
            http_client=client,
        )
        self.assertNotIn('//Treatments', client.urls_seen[0])


# ---------------------------------------------------------------------------
# iter_doi_docs
# ---------------------------------------------------------------------------


class TestIterDoiDocs(unittest.TestCase):
    """Filters CouchDB docs to the ones eligible for Plazi lookup:
    has a non-empty ``doi`` field and isn't a design doc."""

    def test_skips_design_docs_and_missing_doi(self):
        docs = {
            '_design/foo': {'_id': '_design/foo'},
            'doc-with-doi': {'_id': 'doc-with-doi', 'doi': '10.1/x'},
            'doc-no-doi': {'_id': 'doc-no-doi'},
            'doc-empty-doi': {'_id': 'doc-empty-doi', 'doi': ''},
        }
        db = FakeCouchDb(docs)
        ids = [d['_id'] for d in iter_doi_docs(db)]
        self.assertEqual(ids, ['doc-with-doi'])


# ---------------------------------------------------------------------------
# process_doc — integration
# ---------------------------------------------------------------------------


class FakeCouchDb:
    """Minimal stand-in for couchdb.Database supporting ``iter()`` and
    ``__getitem__`` and ``.save(doc)``."""

    def __init__(self, docs: Dict[str, Dict[str, Any]]) -> None:
        self.docs = docs
        self.saves: List[Dict[str, Any]] = []

    def __iter__(self):
        return iter(self.docs)

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        return self.docs[doc_id]

    def save(self, doc: Dict[str, Any]) -> None:
        self.saves.append({'_id': doc['_id'], **doc})


class TestProcessDocIntegration(unittest.TestCase):
    """End-to-end: a stub CouchDB + a fake HTTP client, run through
    ``process_doc``, assert the doc was correctly updated."""

    def setUp(self):
        self.now = '2026-06-03T12:00:00Z'

    def test_hit_writes_plazi_field(self):
        body = [{'DocCount': 1, 'DocUuid': 'AAAA', 'LnkDoi': '10.5281/x'}]
        client = FakeHttpClient(default=FakeResponse(200, body))
        doc = {'_id': 'd1', 'doi': '10.3897/mycokeys.99.107606'}
        result = process_doc(
            doc, http_client=client,
            plazi_url='https://api.plazi.org/GgSrvApi/v1',
            now_iso=self.now, dry_run=False,
        )
        self.assertEqual(result, 'updated')
        self.assertIn('plazi', doc)
        self.assertEqual(doc['plazi']['uuids'], ['AAAA'])
        self.assertEqual(doc['plazi']['looked_up_at'], self.now)

    def test_miss_still_stamps(self):
        client = FakeHttpClient(default=FakeResponse(200, []))
        doc = {'_id': 'd1', 'doi': '10.3390/jof11090688'}
        result = process_doc(
            doc, http_client=client,
            plazi_url='https://api.plazi.org/GgSrvApi/v1',
            now_iso=self.now, dry_run=False,
        )
        self.assertEqual(result, 'updated')
        self.assertEqual(doc['plazi']['uuids'], [])
        self.assertEqual(doc['plazi']['looked_up_at'], self.now)

    def test_dry_run_skips_save(self):
        body = [{'DocCount': 1, 'DocUuid': 'AAAA', 'LnkDoi': '10.5281/x'}]
        client = FakeHttpClient(default=FakeResponse(200, body))
        doc = {'_id': 'd1', 'doi': '10.3897/mycokeys.99.107606'}
        result = process_doc(
            doc, http_client=client,
            plazi_url='https://api.plazi.org/GgSrvApi/v1',
            now_iso=self.now, dry_run=True,
        )
        self.assertEqual(result, 'dry_run')
        # plazi field IS computed (so verbose logs can show what
        # would change) but the doc itself isn't expected to persist.
        self.assertIn('plazi', doc)

    def test_http_failure_returns_skip(self):
        client = FakeHttpClient(default=FakeResponse(503, None))
        doc = {'_id': 'd1', 'doi': '10.3897/mycokeys.99.107606'}
        result = process_doc(
            doc, http_client=client,
            plazi_url='https://api.plazi.org/GgSrvApi/v1',
            now_iso=self.now, dry_run=False,
        )
        self.assertEqual(result, 'http_failure')
        # No plazi stamp on http failure — next run will retry.
        self.assertNotIn('plazi', doc)


class FlakyDb:
    """CouchDB stand-in whose ``.save()`` fails for the first
    ``fail_count`` calls then succeeds.  Records every attempt."""

    def __init__(self, fail_count: int = 0,
                 exc: Optional[Exception] = None) -> None:
        self.fail_count = fail_count
        self.attempts: List[Dict[str, Any]] = []
        self.exc = exc or RuntimeError('simulated transient failure')

    def save(self, doc: Dict[str, Any]) -> None:
        self.attempts.append({'_id': doc['_id'], **doc})
        if len(self.attempts) <= self.fail_count:
            raise self.exc


class TestSaveWithRetry(unittest.TestCase):
    """The live Plazi backfill against skol_dev returned 7 transient
    413 'document_too_large' errors on docs that direct-PUT cleanly
    moments later — same docs save fine on the next attempt.  The
    one-retry path catches these without waiting a full day for the
    cron to come back round.  Tests use a no-op sleep callable so we
    don't wait in CI."""

    def _no_sleep(self, _seconds: float) -> None:
        return None

    def test_first_attempt_succeeds(self):
        db = FlakyDb(fail_count=0)
        ok = save_with_retry(
            db, {'_id': 'd1'}, sleep_seconds=0.5,
            max_attempts=2, sleep_fn=self._no_sleep,
        )
        self.assertTrue(ok)
        self.assertEqual(len(db.attempts), 1)

    def test_retries_once_then_succeeds(self):
        """Transient failure on attempt 1 should be retried, second
        attempt succeeds."""
        db = FlakyDb(fail_count=1)
        sleeps: List[float] = []
        ok = save_with_retry(
            db, {'_id': 'd1'}, sleep_seconds=0.5,
            max_attempts=2, sleep_fn=sleeps.append,
        )
        self.assertTrue(ok)
        self.assertEqual(len(db.attempts), 2)
        self.assertEqual(sleeps, [0.5])

    def test_persistent_failure_counts_as_save_failure(self):
        """Two failures in a row → caller treats this as a real
        save failure (no stamp)."""
        db = FlakyDb(fail_count=2)
        ok = save_with_retry(
            db, {'_id': 'd1'}, sleep_seconds=0.5,
            max_attempts=2, sleep_fn=self._no_sleep,
        )
        self.assertFalse(ok)
        self.assertEqual(len(db.attempts), 2)

    def test_max_attempts_one_is_no_retry(self):
        """Caller can opt out of retry by passing max_attempts=1."""
        db = FlakyDb(fail_count=1)
        ok = save_with_retry(
            db, {'_id': 'd1'}, sleep_seconds=0.5,
            max_attempts=1, sleep_fn=self._no_sleep,
        )
        self.assertFalse(ok)
        self.assertEqual(len(db.attempts), 1)

    def test_logs_exception_on_failure(self):
        """Exhausted retries must surface the underlying exception
        so the operator can tell a permanent 413 from a transient
        503 (the original silent-retry made the live debugging of
        the 43 MB doc_too_large issue much harder than necessary)."""
        db = FlakyDb(
            fail_count=2,
            exc=RuntimeError('document_too_large: foo'),
        )
        captured: List[str] = []
        ok = save_with_retry(
            db, {'_id': 'd1'}, sleep_seconds=0.5,
            max_attempts=2, sleep_fn=self._no_sleep,
            on_error=captured.append,
        )
        self.assertFalse(ok)
        self.assertEqual(len(captured), 2)
        self.assertTrue(
            all('document_too_large' in line for line in captured),
            captured,
        )

    def test_on_error_skipped_when_attempt_succeeds(self):
        """Successful attempts must not call on_error."""
        db = FlakyDb(fail_count=0)
        captured: List[str] = []
        ok = save_with_retry(
            db, {'_id': 'd1'}, sleep_seconds=0.5,
            max_attempts=2, sleep_fn=self._no_sleep,
            on_error=captured.append,
        )
        self.assertTrue(ok)
        self.assertEqual(captured, [])

    def test_on_error_called_for_each_retried_attempt(self):
        """Two failures + a success would still log the first
        failure, so the operator can see transient errors as they
        happen rather than only when retries exhaust."""
        db = FlakyDb(
            fail_count=1,
            exc=RuntimeError('temporary glitch'),
        )
        captured: List[str] = []
        ok = save_with_retry(
            db, {'_id': 'd1'}, sleep_seconds=0.5,
            max_attempts=2, sleep_fn=self._no_sleep,
            on_error=captured.append,
        )
        self.assertTrue(ok)
        self.assertEqual(len(captured), 1)
        self.assertIn('temporary glitch', captured[0])


# ---------------------------------------------------------------------------
# Plazi response size guard
# ---------------------------------------------------------------------------


class TestPlaziOversizeGuard(unittest.TestCase):
    """The live Plazi API returned ~700 000 entries (66 MB) for
    DOIs it has no real match on — clear server-side misbehavior.
    Stamping that into the doc balloons it to 43 MB and trips
    CouchDB's document_too_large.  ``query_plazi`` rejects responses
    larger than a sanity cap so the affected docs are simply skipped
    (no save, no failure that retries forever)."""

    def test_oversize_response_returns_none(self):
        from backfill_plazi_uuids import query_plazi, _MAX_PLAZI_ENTRIES

        class FakeResp:
            status_code = 200
            def json(self) -> List[Dict[str, str]]:
                # Just over the cap.
                return [
                    {'DocUuid': f'u{i}', 'LnkDoi': '10.x/y'}
                    for i in range(_MAX_PLAZI_ENTRIES + 1)
                ]

        class FakeClient:
            def get(self, _url: str) -> 'FakeResp':
                return FakeResp()

        body = query_plazi(
            '10.foo/bar', plazi_url='http://x', http_client=FakeClient(),
        )
        self.assertIsNone(body)

    def test_at_cap_size_still_passes(self):
        from backfill_plazi_uuids import query_plazi, _MAX_PLAZI_ENTRIES

        class FakeResp:
            status_code = 200
            def json(self) -> List[Dict[str, str]]:
                return [
                    {'DocUuid': f'u{i}', 'LnkDoi': '10.x/y'}
                    for i in range(_MAX_PLAZI_ENTRIES)
                ]

        class FakeClient:
            def get(self, _url: str) -> 'FakeResp':
                return FakeResp()

        body = query_plazi(
            '10.foo/bar', plazi_url='http://x', http_client=FakeClient(),
        )
        self.assertIsNotNone(body)
        self.assertEqual(len(body), _MAX_PLAZI_ENTRIES)


if __name__ == '__main__':
    unittest.main()
