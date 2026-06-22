"""Tests for fixes/backfill_plazi_uuids.py."""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_plazi_uuids import (  # type: ignore[import]  # noqa: E402
    FailureRecord,
    PlaziResult,
    _SOURCE_TAG,
    build_search_url,
    compute_plazi_error,
    compute_plazi_update,
    format_failure_log_line,
    format_heartbeat,
    is_heartbeat_tick,
    is_sticky_reason,
    iter_doi_docs,
    parse_failure_log_line,
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
    """Convert the Plazi srsStats data list to (uuids, dois) tuple.

    Post-2026-06-10 migration: the endpoint is ``srsStats/stats``
    and entries carry ``PubLnkArticleDoi`` (the DOI of the article
    each treatment was *extracted from*), not the legacy
    ``LnkDoi`` (which was the DOI the treatment merely linked to).
    The doc-level field name ``lnk_dois`` is preserved for storage
    backward-compat — every doc previously written carries that
    key, and renaming would require a full corpus migration."""

    def test_with_hits(self):
        body = [
            {'DocCount': 1, 'DocUuid': 'AAAA',
             'PubLnkArticleDoi': '10.5281/zenodo.1',
             'TaxName': 'Genus species A'},
            {'DocCount': 1, 'DocUuid': 'BBBB',
             'PubLnkArticleDoi': '10.5281/zenodo.2',
             'TaxName': 'Genus species B'},
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
            {'DocCount': 1, 'PubLnkArticleDoi': '10.5281/zenodo.x'},
            {'DocCount': 1, 'DocUuid': 'CCCC',
             'PubLnkArticleDoi': '10.5281/zenodo.y'},
        ]
        uuids, lnk_dois = parse_plazi_response(body)
        self.assertEqual(uuids, ['CCCC'])
        self.assertEqual(lnk_dois, ['10.5281/zenodo.y'])

    def test_entry_without_articledoi_keeps_uuid(self):
        """PubLnkArticleDoi can legitimately be absent for some
        treatments — keep the UUID, blank slot in the parallel
        list so the two stay index-aligned."""
        body = [{'DocCount': 1, 'DocUuid': 'DDDD'}]
        uuids, lnk_dois = parse_plazi_response(body)
        self.assertEqual(uuids, ['DDDD'])
        self.assertEqual(lnk_dois, [''])

    def test_legacy_lnkdoi_field_ignored(self):
        """Regression: pre-migration responses carried ``LnkDoi``
        (treatment-references-this-DOI semantics).  After the
        srsStats migration we want ``PubLnkArticleDoi`` only —
        ``LnkDoi`` would silently bring back the wrong semantics
        if we accepted it as a fallback."""
        body = [
            {'DocCount': 1, 'DocUuid': 'EEEE',
             'LnkDoi': '10.5281/zenodo.legacy'},  # should be ignored
        ]
        uuids, lnk_dois = parse_plazi_response(body)
        self.assertEqual(uuids, ['EEEE'])
        # No PubLnkArticleDoi field present → blank slot.
        self.assertEqual(lnk_dois, [''])


# ---------------------------------------------------------------------------
# compute_plazi_update
# ---------------------------------------------------------------------------


class TestComputePlaziUpdate(unittest.TestCase):
    """Build the dict that goes at ``doc['plazi']``."""

    def test_with_hits(self):
        body = [
            {'DocCount': 1, 'DocUuid': 'AAAA',
             'PubLnkArticleDoi': '10.5281/x'},
        ]
        now = '2026-06-03T12:00:00Z'
        update = compute_plazi_update(body, now)
        self.assertEqual(update['uuids'], ['AAAA'])
        self.assertEqual(update['lnk_dois'], ['10.5281/x'])
        self.assertEqual(update['looked_up_at'], now)
        self.assertEqual(update['source'], 'plazi:GgServer:srsStats:v1')

    def test_empty_still_stamps(self):
        """An empty Plazi hit still produces a fully-formed plazi
        dict — proves the doc was checked.  Idempotent re-runs depend
        on the ``looked_up_at`` being present."""
        update = compute_plazi_update([], '2026-06-03T12:00:00Z')
        self.assertEqual(update['uuids'], [])
        self.assertEqual(update['lnk_dois'], [])
        self.assertEqual(update['looked_up_at'], '2026-06-03T12:00:00Z')
        self.assertEqual(update['source'], 'plazi:GgServer:srsStats:v1')


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
            {'doi': '10.1/x',
             'plazi': {'looked_up_at': recent, 'source': _SOURCE_TAG}},
            force=False, re_check_after_days=365, now_iso=self.NOW,
        ))

    def test_stale_lookup_proceed(self):
        """Older than re_check_after_days → re-query."""
        old = (datetime(2025, 1, 1, tzinfo=timezone.utc)
               .strftime('%Y-%m-%dT%H:%M:%SZ'))
        self.assertFalse(should_skip(
            {'doi': '10.1/x',
             'plazi': {'looked_up_at': old, 'source': _SOURCE_TAG}},
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

    def _failed_doc(self, failed_at: str) -> Dict[str, Any]:
        return {
            'doi': '10.1/x',
            'plazi': {
                'error': {'reason': 'http_status', 'detail': '500',
                          'url': 'http://x'},
                'failed_at': failed_at,
                'source': 'plazi:GgServer:srsStats:v1',
            },
        }

    def test_recent_sticky_failure_skip(self):
        """A doc stamped with a recent sticky failure is backed off
        (the weak block) so restarts don't re-hit it."""
        recent = (datetime(2026, 6, 1, tzinfo=timezone.utc)
                  .strftime('%Y-%m-%dT%H:%M:%SZ'))
        self.assertTrue(should_skip(
            self._failed_doc(recent),
            force=False, re_check_after_days=365,
            retry_failed_after_days=7, now_iso=self.NOW,
        ))

    def test_stale_sticky_failure_proceed(self):
        """Past the retry-failed window → re-query."""
        old = (datetime(2026, 5, 1, tzinfo=timezone.utc)
               .strftime('%Y-%m-%dT%H:%M:%SZ'))
        self.assertFalse(should_skip(
            self._failed_doc(old),
            force=False, re_check_after_days=365,
            retry_failed_after_days=7, now_iso=self.NOW,
        ))

    def test_force_overrides_sticky_failure(self):
        recent = (datetime(2026, 6, 1, tzinfo=timezone.utc)
                  .strftime('%Y-%m-%dT%H:%M:%SZ'))
        self.assertFalse(should_skip(
            self._failed_doc(recent),
            force=True, re_check_after_days=365,
            retry_failed_after_days=7, now_iso=self.NOW,
        ))

    def test_failure_stamp_without_failed_at_proceed(self):
        """Defensive: an error record missing ``failed_at`` is not a
        valid backoff marker, so we re-query rather than block forever."""
        self.assertFalse(should_skip(
            {'doi': '10.1/x', 'plazi': {'error': {'reason': 'runaway'}}},
            force=False, re_check_after_days=365,
            retry_failed_after_days=7, now_iso=self.NOW,
        ))

    def test_old_source_lookup_is_requeried(self):
        """Migration: a recent success stamp from the old searchByDOI
        source tag carries stale/wrong data and must be re-queried under
        the new srsStats endpoint — not skipped on freshness alone."""
        recent = (datetime(2026, 6, 1, tzinfo=timezone.utc)
                  .strftime('%Y-%m-%dT%H:%M:%SZ'))
        self.assertFalse(should_skip(
            {'doi': '10.1/x', 'plazi': {
                'looked_up_at': recent, 'source': 'plazi:GgSrvApi:v1'}},
            force=False, re_check_after_days=365, now_iso=self.NOW,
        ))

    def test_missing_source_lookup_is_requeried(self):
        """A stamp with no source tag predates the convention; re-query."""
        recent = (datetime(2026, 6, 1, tzinfo=timezone.utc)
                  .strftime('%Y-%m-%dT%H:%M:%SZ'))
        self.assertFalse(should_skip(
            {'doi': '10.1/x', 'plazi': {'looked_up_at': recent}},
            force=False, re_check_after_days=365, now_iso=self.NOW,
        ))

    def test_old_source_failure_is_requeried(self):
        """A recent sticky-failure stamp from the old source tag is
        re-queried too, not held by the weak block."""
        recent = (datetime(2026, 6, 1, tzinfo=timezone.utc)
                  .strftime('%Y-%m-%dT%H:%M:%SZ'))
        self.assertFalse(should_skip(
            {'doi': '10.1/x', 'plazi': {
                'error': {'reason': 'http_status'},
                'failed_at': recent, 'source': 'plazi:GgSrvApi:v1'}},
            force=False, re_check_after_days=365,
            retry_failed_after_days=7, now_iso=self.NOW,
        ))


# ---------------------------------------------------------------------------
# query_plazi
# ---------------------------------------------------------------------------


class RaisingHttpClient:
    """Stand-in whose ``get()`` raises, simulating a pre-response
    network failure (ReadTimeout / ConnectionError / DNS)."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def get(self, url: str, **kwargs: Any) -> Any:
        raise self.exc


class TestQueryPlazi(unittest.TestCase):
    """The HTTP layer.  We exercise it through a fake RateLimitedHttpClient
    so the tests don't hit Plazi for real.  ``query_plazi`` now returns a
    ``PlaziResult`` carrying either the parsed ``body`` (reason is None) or
    a structured failure ``reason``/``detail``, plus the exact ``url``."""

    PLAZI_URL = 'https://tb.plazi.org/GgServer'

    def test_success_returns_body_and_no_reason(self):
        """srsStats returns {data: [...rows...]}; query_plazi unwraps
        to the inner list so downstream code can iterate uniformly."""
        rows = [
            {'DocCount': 1, 'DocUuid': 'AAAA',
             'PubLnkArticleDoi': '10.5281/x',
             'TaxName': 'Genus species'},
        ]
        body = {'data': rows}
        client = FakeHttpClient(default=FakeResponse(200, body))
        result = query_plazi(
            '10.3897/mycokeys.99.107606',
            plazi_url=self.PLAZI_URL, http_client=client,
        )
        self.assertIsInstance(result, PlaziResult)
        self.assertEqual(result.body, rows)
        self.assertIsNone(result.reason)
        self.assertIn('/srsStats/stats', result.url)

    def test_non_200_reports_http_status_with_code(self):
        client = FakeHttpClient(default=FakeResponse(503, None))
        result = query_plazi(
            '10.3897/mycokeys.99.107606',
            plazi_url=self.PLAZI_URL, http_client=client,
        )
        self.assertIsNone(result.body)
        self.assertEqual(result.reason, 'http_status')
        self.assertEqual(result.detail, '503')

    def test_request_exception_reports_request_error(self):
        """A pre-response exception is the one *transient* reason —
        detail carries the exception class for the bug report."""
        client = RaisingHttpClient(TimeoutError('read timed out'))
        result = query_plazi(
            '10.1/x', plazi_url=self.PLAZI_URL, http_client=client,
        )
        self.assertIsNone(result.body)
        self.assertEqual(result.reason, 'request_error')
        self.assertEqual(result.detail, 'TimeoutError')

    def test_bad_json_reports_bad_json(self):
        class BadJson(FakeResponse):
            def json(self) -> Any:
                raise ValueError('no json could be decoded')

        client = FakeHttpClient(default=BadJson(200, None))
        result = query_plazi(
            '10.1/x', plazi_url=self.PLAZI_URL, http_client=client,
        )
        self.assertIsNone(result.body)
        self.assertEqual(result.reason, 'bad_json')

    def test_non_dict_body_reports_not_list(self):
        """srsStats wraps the rows in ``{data: [...]}``.  A response
        that isn't a dict (or whose ``data`` isn't a list) carries
        the legacy ``not_list`` reason for caller-disposition
        compatibility."""
        client = FakeHttpClient(default=FakeResponse(200, 'plain text'))
        result = query_plazi(
            '10.1/x', plazi_url=self.PLAZI_URL, http_client=client,
        )
        self.assertIsNone(result.body)
        self.assertEqual(result.reason, 'not_list')

    def test_data_key_missing_treated_as_empty(self):
        """Defensive: a dict without ``data`` is treated as
        zero-hit, not an error.  Plazi sometimes returns an empty
        response shape; the caller writes ``uuids=[]`` and moves
        on."""
        client = FakeHttpClient(
            default=FakeResponse(200, {'message': 'no data'})
        )
        result = query_plazi(
            '10.1/x', plazi_url=self.PLAZI_URL, http_client=client,
        )
        self.assertEqual(result.body, [])
        self.assertIsNone(result.reason)

    def test_constructs_correct_url(self):
        """Post-2026-06-10: the endpoint is ``/srsStats/stats``,
        the DOI is wrapped in URL-encoded double-quotes so Plazi's
        parser does exact match (without quotes the dash-as-range
        bug fires on any DOI containing ``-``).  The output and
        grouping fields project enough for the parser to extract
        UUIDs."""
        client = FakeHttpClient(
            default=FakeResponse(200, {'data': []})
        )
        query_plazi(
            '10.5281/zenodo.7105224',
            plazi_url=self.PLAZI_URL, http_client=client,
        )
        self.assertEqual(len(client.urls_seen), 1)
        url = client.urls_seen[0]
        self.assertIn('/srsStats/stats', url)
        self.assertIn('format=JSON', url)
        # FP-pubLnk.articleDoi=%22<encoded-doi>%22 — the quotes
        # are part of the value, both encoded.
        self.assertIn(
            'FP-pubLnk.articleDoi=%2210.5281%2Fzenodo.7105224%22',
            url,
        )
        # outputFields + groupingFields project the columns the
        # parser reads.
        self.assertIn('outputFields=doc.uuid', url)
        self.assertIn('groupingFields=doc.uuid', url)


class TestBuildSearchUrl(unittest.TestCase):
    """The URL builder is shared by query_plazi and the bug-report tool,
    so it's a named helper.  Post-2026-06-10 migration: srsStats/stats
    endpoint with FP-pubLnk.articleDoi filter and quoted DOI."""

    PLAZI_URL = 'https://tb.plazi.org/GgServer'

    def test_uses_srsstats_endpoint(self):
        url = build_search_url('10.5281/zenodo.7105224', self.PLAZI_URL)
        self.assertIn('/srsStats/stats', url)
        self.assertNotIn('/Treatments/searchByDOI', url)

    def test_quotes_doi_for_exact_match(self):
        """Quoting defeats Plazi's dash-as-range bug.  On the
        srsStats endpoint the quote-strip works correctly (unlike
        the legacy searchByDOI endpoint, where quoted DOIs returned
        0 even for known-good no-dash DOIs because that endpoint
        matched ``LnkDoi`` rather than ``pubLnk.articleDoi``)."""
        url = build_search_url('10.7717/peerj-cs.2153', self.PLAZI_URL)
        # %22 around the URL-encoded DOI value.
        self.assertIn(
            'FP-pubLnk.articleDoi=%2210.7717%2Fpeerj-cs.2153%22',
            url,
        )

    def test_quotes_doi_containing_dash_regression(self):
        """The case that motivated the migration: 10.3852/11-180
        triggered the dash-as-range parser on searchByDOI returning
        ~700 k entries; on srsStats with quoting it exact-matches."""
        url = build_search_url('10.3852/11-180', self.PLAZI_URL)
        self.assertIn(
            'FP-pubLnk.articleDoi=%2210.3852%2F11-180%22', url,
        )

    def test_sets_format_json(self):
        url = build_search_url('10.5281/zenodo.7105224', self.PLAZI_URL)
        self.assertIn('format=JSON', url)

    def test_projects_output_and_grouping_fields(self):
        """The stats endpoint requires explicit output + grouping
        field projection.  We project the minimum the parser needs:
        doc.uuid + pubLnk.articleDoi + tax.name."""
        url = build_search_url('10.5281/zenodo.7105224', self.PLAZI_URL)
        self.assertIn('outputFields=doc.uuid', url)
        self.assertIn('groupingFields=doc.uuid', url)
        self.assertIn('pubLnk.articleDoi', url)
        self.assertIn('tax.name', url)

    def test_trims_trailing_slash(self):
        url = build_search_url('10.1/x', self.PLAZI_URL + '/')
        self.assertNotIn('//srsStats', url)


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
        rows = [
            {'DocCount': 1, 'DocUuid': 'AAAA',
             'PubLnkArticleDoi': '10.3897/mycokeys.99.107606',
             'TaxName': 'Genus species'},
        ]
        client = FakeHttpClient(default=FakeResponse(200, {'data': rows}))
        doc = {'_id': 'd1', 'doi': '10.3897/mycokeys.99.107606'}
        status, result = process_doc(
            doc, http_client=client,
            plazi_url='https://tb.plazi.org/GgServer',
            now_iso=self.now, dry_run=False,
        )
        self.assertEqual(status, 'updated')
        self.assertIsInstance(result, PlaziResult)
        self.assertIn('plazi', doc)
        self.assertEqual(doc['plazi']['uuids'], ['AAAA'])
        self.assertEqual(doc['plazi']['looked_up_at'], self.now)

    def test_miss_still_stamps(self):
        client = FakeHttpClient(default=FakeResponse(200, {'data': []}))
        doc = {'_id': 'd1', 'doi': '10.3390/jof11090688'}
        status, _ = process_doc(
            doc, http_client=client,
            plazi_url='https://tb.plazi.org/GgServer',
            now_iso=self.now, dry_run=False,
        )
        self.assertEqual(status, 'updated')
        self.assertEqual(doc['plazi']['uuids'], [])
        self.assertEqual(doc['plazi']['looked_up_at'], self.now)

    def test_dry_run_skips_save(self):
        rows = [
            {'DocCount': 1, 'DocUuid': 'AAAA',
             'PubLnkArticleDoi': '10.3897/mycokeys.99.107606'},
        ]
        client = FakeHttpClient(default=FakeResponse(200, {'data': rows}))
        doc = {'_id': 'd1', 'doi': '10.3897/mycokeys.99.107606'}
        status, _ = process_doc(
            doc, http_client=client,
            plazi_url='https://tb.plazi.org/GgServer',
            now_iso=self.now, dry_run=True,
        )
        self.assertEqual(status, 'dry_run')
        # plazi field IS computed (so verbose logs can show what
        # would change) but the doc itself isn't expected to persist.
        self.assertIn('plazi', doc)

    def test_sticky_failure_stamps_error(self):
        """A server-engaged failure (here a 503) is sticky: it stamps an
        error record so should_skip can back the doc off, and returns the
        PlaziResult so the caller can log reproduction info."""
        client = FakeHttpClient(default=FakeResponse(503, None))
        doc = {'_id': 'd1', 'doi': '10.3897/mycokeys.99.107606'}
        status, result = process_doc(
            doc, http_client=client,
            plazi_url='https://tb.plazi.org/GgServer',
            now_iso=self.now, dry_run=False,
        )
        self.assertEqual(status, 'sticky_failure')
        self.assertEqual(result.reason, 'http_status')
        self.assertEqual(doc['plazi']['error']['reason'], 'http_status')
        self.assertEqual(doc['plazi']['error']['detail'], '503')
        self.assertEqual(doc['plazi']['failed_at'], self.now)
        # An error stamp must never look like a successful empty lookup.
        self.assertNotIn('looked_up_at', doc['plazi'])

    def test_transient_failure_leaves_doc_unstamped(self):
        """A pre-response failure (request_error) is transient: no stamp,
        so the doc is retried on the very next run."""
        client = RaisingHttpClient(ConnectionError('refused'))
        doc = {'_id': 'd1', 'doi': '10.3897/mycokeys.99.107606'}
        status, result = process_doc(
            doc, http_client=client,
            plazi_url='https://tb.plazi.org/GgServer',
            now_iso=self.now, dry_run=False,
        )
        self.assertEqual(status, 'transient_failure')
        self.assertEqual(result.reason, 'request_error')
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

    def test_oversize_response_is_runaway_with_count(self):
        from backfill_plazi_uuids import _MAX_PLAZI_ENTRIES
        over = _MAX_PLAZI_ENTRIES + 1
        rows = [
            {'DocUuid': f'u{i}', 'PubLnkArticleDoi': '10.x/y'}
            for i in range(over)
        ]
        client = FakeHttpClient(default=FakeResponse(200, {'data': rows}))
        result = query_plazi(
            '10.foo/bar', plazi_url='http://x', http_client=client,
        )
        self.assertIsNone(result.body)
        self.assertEqual(result.reason, 'runaway')
        # detail carries the entry count so the bug report can cite it.
        self.assertEqual(result.detail, str(over))

    def test_at_cap_size_still_passes(self):
        from backfill_plazi_uuids import _MAX_PLAZI_ENTRIES
        rows = [
            {'DocUuid': f'u{i}', 'PubLnkArticleDoi': '10.x/y'}
            for i in range(_MAX_PLAZI_ENTRIES)
        ]
        client = FakeHttpClient(default=FakeResponse(200, {'data': rows}))
        result = query_plazi(
            '10.foo/bar', plazi_url='http://x', http_client=client,
        )
        self.assertIsNone(result.reason)
        self.assertEqual(len(result.body), _MAX_PLAZI_ENTRIES)


# ---------------------------------------------------------------------------
# Failure classification + error stamp + log line
# ---------------------------------------------------------------------------


class TestIsStickyReason(unittest.TestCase):
    """'Server-engaged = sticky': any failure where Plazi returned an
    HTTP response is backed off; only a pre-response request_error is
    transient and retried every run."""

    def test_request_error_is_transient(self):
        self.assertFalse(is_sticky_reason('request_error'))

    def test_server_engaged_reasons_are_sticky(self):
        for reason in ('http_status', 'bad_json', 'not_list', 'runaway'):
            with self.subTest(reason=reason):
                self.assertTrue(is_sticky_reason(reason))


class TestComputePlaziError(unittest.TestCase):
    """The error stamp written to ``doc['plazi']`` for sticky failures.
    It must be distinguishable from a successful empty lookup (no
    ``looked_up_at``/``uuids``) and carry enough to reproduce."""

    def test_shape(self):
        result = PlaziResult(
            body=None, reason='http_status', detail='500',
            url='https://tb.plazi.org/GgServer/srsStats/stats'
                '?FP-pubLnk.articleDoi=%2210.1%2Fx%22&format=JSON',
        )
        stamp = compute_plazi_error(result, '2026-06-05T12:00:00Z')
        self.assertEqual(stamp['error']['reason'], 'http_status')
        self.assertEqual(stamp['error']['detail'], '500')
        self.assertEqual(stamp['error']['url'], result.url)
        self.assertEqual(stamp['failed_at'], '2026-06-05T12:00:00Z')
        self.assertEqual(stamp['source'], 'plazi:GgServer:srsStats:v1')
        self.assertNotIn('looked_up_at', stamp)
        self.assertNotIn('uuids', stamp)


class TestFailureLogLine(unittest.TestCase):
    """The failure log line must carry everything the bug-report tool
    needs to reproduce (id, reason, detail, doi, url), and format/parse
    must round-trip so the tool can read back what the backfill wrote."""

    def test_format_contains_repro_fields(self):
        result = PlaziResult(
            body=None, reason='runaway', detail='703118',
            url='https://api.plazi.org/x?DOI=10.1%2Fx&format=json',
        )
        line = format_failure_log_line('doc-42', result, '10.1/x')
        self.assertIn('doc-42', line)
        self.assertIn('reason=runaway', line)
        self.assertIn('detail=703118', line)
        self.assertIn('doi=10.1/x', line)
        self.assertIn(result.url, line)

    def test_round_trip(self):
        result = PlaziResult(
            body=None, reason='http_status', detail='500',
            url='https://api.plazi.org/x?DOI=10.1%2Fx&format=json',
        )
        line = format_failure_log_line('doc-42', result, '10.1/x')
        rec = parse_failure_log_line(line)
        self.assertIsInstance(rec, FailureRecord)
        self.assertEqual(rec.doc_id, 'doc-42')
        self.assertEqual(rec.reason, 'http_status')
        self.assertEqual(rec.detail, '500')
        self.assertEqual(rec.doi, '10.1/x')
        self.assertEqual(rec.url, result.url)

    def test_round_trip_tolerates_leading_indent(self):
        """The backfill prints failures with a two-space indent; the
        parser must still recognise them."""
        result = PlaziResult(
            body=None, reason='bad_json', detail='None',
            url='https://api.plazi.org/x?DOI=10.1%2Fx&format=json',
        )
        line = '  ' + format_failure_log_line('d1', result, '10.1/x')
        rec = parse_failure_log_line(line)
        self.assertEqual(rec.doc_id, 'd1')
        self.assertEqual(rec.reason, 'bad_json')

    def test_non_failure_line_returns_none(self):
        self.assertIsNone(parse_failure_log_line(
            '  d1: doi=\'10.1/x\' -> 0 uuid(s)'))
        self.assertIsNone(parse_failure_log_line('random log noise'))


# ---------------------------------------------------------------------------
# Heartbeat (progress line for long, mostly-skipped runs)
# ---------------------------------------------------------------------------


class TestHeartbeat(unittest.TestCase):
    """A long run over an already-stamped DB skips almost everything and
    prints nothing per-doc, so it can look frozen. A heartbeat every N
    scanned docs shows it advancing."""

    def test_tick_on_multiples_only(self):
        self.assertTrue(is_heartbeat_tick(500, 500))
        self.assertTrue(is_heartbeat_tick(1000, 500))
        self.assertFalse(is_heartbeat_tick(499, 500))
        self.assertFalse(is_heartbeat_tick(501, 500))

    def test_no_tick_at_zero(self):
        """scanned == 0 must not fire (0 % N == 0)."""
        self.assertFalse(is_heartbeat_tick(0, 500))

    def test_interval_zero_disables(self):
        self.assertFalse(is_heartbeat_tick(500, 0))

    def test_format_shows_scanned_skipped_queried(self):
        counts = {'scanned': 500, 'skipped_fresh': 480, 'queried': 20}
        line = format_heartbeat(counts)
        self.assertIn('500', line)
        self.assertIn('480', line)
        self.assertIn('20', line)


if __name__ == '__main__':
    unittest.main()
