"""Tests for bin/plazi_bug_report.py.

The tool turns logged Plazi ``searchByDOI`` failures (written by
backfill_plazi_uuids) into a single Plazi-policy-compliant bug report:
one issue body plus a ``failure.csv`` of all cases, with an embedded
script that replays every row of the CSV.  Per
https://github.com/plazi/community a machine/batch audit must be one
issue with a CSV, never individual cases.
"""
from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_plazi_uuids import (  # type: ignore[import]  # noqa: E402
    FailureRecord,
    PlaziResult,
    format_failure_log_line,
)
from plazi_bug_report import (  # type: ignore[import]  # noqa: E402
    EXPECTED,
    failure_csv,
    find_failures,
    issue_body,
    live_observe,
    main,
    observed_for,
    repro_script,
    retrieval_instructions,
)

_SITE = 'https://synoptickeyof.life'


def _rec(doc_id: str = 'd1', reason: str = 'http_status',
         detail: str = '500', doi: str = '10.1/x',
         url: str = 'https://api.plazi.org/v1/Treatments/searchByDOI'
                    '?DOI=10.1%2Fx&format=json') -> FailureRecord:
    return FailureRecord(
        doc_id=doc_id, reason=reason, detail=detail, doi=doi, url=url)


# ---------------------------------------------------------------------------
# find_failures
# ---------------------------------------------------------------------------


class TestFindFailures(unittest.TestCase):
    """Pull the requested ids out of a log file, reusing the backfill's
    own log format so the URLs match exactly what was queried."""

    def _log_line(self, doc_id: str, reason: str, detail: str,
                  doi: str) -> str:
        result = PlaziResult(
            body=None, reason=reason, detail=detail,
            url=f'https://api.plazi.org/v1/x?DOI={doi}&format=json')
        return '  ' + format_failure_log_line(doc_id, result, doi)

    def test_selects_requested_ids_in_order(self):
        lines = [
            self._log_line('a', 'http_status', '500', '10.1/a'),
            'some unrelated noise',
            self._log_line('b', 'runaway', '703118', '10.1/b'),
            self._log_line('c', 'bad_json', 'None', '10.1/c'),
        ]
        found, missing = find_failures(lines, ['b', 'a'])
        self.assertEqual([r.doc_id for r in found], ['b', 'a'])
        self.assertEqual(missing, [])

    def test_reports_missing_ids(self):
        lines = [self._log_line('a', 'http_status', '500', '10.1/a')]
        found, missing = find_failures(lines, ['a', 'zzz'])
        self.assertEqual([r.doc_id for r in found], ['a'])
        self.assertEqual(missing, ['zzz'])

    def test_last_occurrence_wins(self):
        """A doc that failed, was retried, and failed differently should
        report its most recent failure."""
        lines = [
            self._log_line('a', 'request_error', 'ReadTimeout', '10.1/a'),
            self._log_line('a', 'http_status', '500', '10.1/a'),
        ]
        found, _ = find_failures(lines, ['a'])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].reason, 'http_status')


# ---------------------------------------------------------------------------
# expected / observed prose
# ---------------------------------------------------------------------------


class TestExpectedObserved(unittest.TestCase):
    def test_expected_is_the_contract(self):
        self.assertIn('200', EXPECTED)
        self.assertIn('array', EXPECTED.lower())

    def test_observed_http_status(self):
        obs = observed_for(_rec(reason='http_status', detail='500'))
        self.assertIn('500', obs)

    def test_observed_runaway_cites_count_and_cap(self):
        obs = observed_for(_rec(reason='runaway', detail='703118'))
        self.assertIn('703118', obs)
        self.assertIn('100', obs)

    def test_observed_request_error_names_exception(self):
        obs = observed_for(_rec(reason='request_error', detail='ReadTimeout'))
        self.assertIn('ReadTimeout', obs)

    def test_observed_bad_json(self):
        obs = observed_for(_rec(reason='bad_json', detail='None'))
        self.assertIn('JSON', obs)


# ---------------------------------------------------------------------------
# failure_csv
# ---------------------------------------------------------------------------


class TestFailureCsv(unittest.TestCase):
    EXPECTED_HEADER = [
        'doc_id', 'doi', 'reason', 'detail', 'url', 'expected', 'observed',
    ]

    def test_header_and_row(self):
        text = failure_csv([_rec()])
        rows = list(csv.reader(io.StringIO(text)))
        self.assertEqual(rows[0], self.EXPECTED_HEADER)
        self.assertEqual(rows[1][0], 'd1')
        self.assertEqual(rows[1][1], '10.1/x')
        self.assertEqual(rows[1][2], 'http_status')

    def test_fields_with_commas_are_quoted(self):
        """observed text contains commas; round-tripping through the csv
        reader must recover the exact field."""
        text = failure_csv([_rec(reason='runaway', detail='703118')])
        parsed = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(parsed[0]['reason'], 'runaway')
        self.assertIn('703118', parsed[0]['observed'])

    def test_live_observed_override(self):
        text = failure_csv([_rec()], observed={'d1': 'HTTP 500 (live re-run)'})
        parsed = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(parsed[0]['observed'], 'HTTP 500 (live re-run)')


# ---------------------------------------------------------------------------
# repro_script + issue_body
# ---------------------------------------------------------------------------


class TestReproScript(unittest.TestCase):
    def test_reads_csv_and_curls(self):
        script = repro_script('failure.csv', user_agent='skol-x/1.0')
        self.assertIn('failure.csv', script)
        self.assertIn('curl', script)
        self.assertIn('skol-x/1.0', script)


class TestRetrievalInstructions(unittest.TestCase):
    """The report tells a reader how to pull the source skol document for
    each failing id via the Django attachment API at synoptickeyof.life
    (GET /api/pdf/<db>/<doc_id>/), not CouchDB directly."""

    def test_curls_each_doc_by_id(self):
        records = [_rec(doc_id='a', doi='10.1/a'),
                   _rec(doc_id='b', doi='10.1/b')]
        text = retrieval_instructions(
            records, site_url=_SITE, source_db='skol')
        self.assertIn('synoptickeyof.life', text)
        self.assertIn('/api/pdf/skol/a/', text)
        self.assertIn('/api/pdf/skol/b/', text)
        self.assertIn('curl', text)

    def test_uses_django_api_without_credentials(self):
        """The Django API proxies CouchDB server-side, so the report needs
        no credentials and must not leak the CouchDB host/port."""
        text = retrieval_instructions(
            [_rec(doc_id='a')], site_url=_SITE, source_db='skol')
        self.assertIn('/api/pdf/', text)
        self.assertNotIn('COUCHDB', text)
        self.assertNotIn('5984', text)


class TestIssueBody(unittest.TestCase):
    def test_contains_summary_repro_and_cases(self):
        records = [
            _rec(doc_id='a', reason='http_status', detail='500',
                 doi='10.1/a'),
            _rec(doc_id='b', reason='runaway', detail='703118',
                 doi='10.1/b'),
        ]
        body = issue_body(
            records, csv_filename='failure.csv',
            script_filename='reproduce.sh', user_agent='skol-x/1.0',
            site_url=_SITE, source_db='skol')
        # Mentions both DOIs / the CSV / a reproduction step.
        self.assertIn('10.1/a', body)
        self.assertIn('10.1/b', body)
        self.assertIn('failure.csv', body)
        self.assertIn('reproduce.sh', body)
        # Per-reason grouping so the Plazi team sees the failure mix.
        self.assertIn('http_status', body)
        self.assertIn('runaway', body)
        # Expected-vs-observed framing.
        self.assertIn('Expected', body)
        self.assertIn('Observed', body)
        # Source-document retrieval via the Django API at synoptickeyof.life.
        self.assertIn('synoptickeyof.life', body)
        self.assertIn('/api/pdf/skol/a/', body)


# ---------------------------------------------------------------------------
# live_observe
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _FakeClient:
    def __init__(self, resp: Any) -> None:
        self.resp = resp
        self.seen: List[str] = []

    def get(self, url: str, **kwargs: Any) -> Any:
        self.seen.append(url)
        return self.resp


class TestLiveObserve(unittest.TestCase):
    def test_reports_status_and_snippet(self):
        client = _FakeClient(_FakeResp(500, 'Internal Server Error'))
        obs = live_observe(_rec(), client)
        self.assertEqual(len(client.seen), 1)
        self.assertIn('500', obs)

    def test_handles_request_exception(self):
        class Boom:
            def get(self, _url: str, **_kw: Any) -> Any:
                raise TimeoutError('read timed out')

        obs = live_observe(_rec(), Boom())
        self.assertIn('TimeoutError', obs)


# ---------------------------------------------------------------------------
# main — integration (no network)
# ---------------------------------------------------------------------------


class TestMainIntegration(unittest.TestCase):
    def _write_log(self, path: Path) -> None:
        lines = []
        for doc_id, reason, detail, doi in [
            ('a', 'http_status', '500', '10.1/a'),
            ('b', 'runaway', '703118', '10.1/b'),
        ]:
            result = PlaziResult(
                body=None, reason=reason, detail=detail,
                url=f'https://api.plazi.org/v1/x?DOI={doi}&format=json')
            lines.append(format_failure_log_line(doc_id, result, doi))
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def test_writes_csv_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            log = tmpdir / 'run.log'
            self._write_log(log)
            out = tmpdir / 'out'
            rc = main([
                'a', 'b', '--log', str(log), '--out-dir', str(out),
            ])
            self.assertEqual(rc, 0)
            csv_path = out / 'failure.csv'
            self.assertTrue(csv_path.exists())
            parsed = list(csv.DictReader(
                io.StringIO(csv_path.read_text(encoding='utf-8'))))
            self.assertEqual({r['doc_id'] for r in parsed}, {'a', 'b'})
            # A report markdown is written too.
            reports = list(out.glob('*.md'))
            self.assertEqual(len(reports), 1)
            report_text = reports[0].read_text()
            self.assertIn('failure.csv', report_text)
            # Retrieval instructions default to production CouchDB.
            self.assertIn('synoptickeyof.life', report_text)

    def test_missing_id_is_a_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            log = tmpdir / 'run.log'
            self._write_log(log)
            rc = main([
                'a', 'nope', '--log', str(log),
                '--out-dir', str(tmpdir / 'out'),
            ])
            # Still writes a report for the found ids, but signals that
            # some requested ids weren't in the log.
            self.assertEqual(rc, 2)


if __name__ == '__main__':
    unittest.main()
