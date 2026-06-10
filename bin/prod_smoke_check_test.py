"""Tests for bin/prod_smoke_check.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prod_smoke_check import (  # type: ignore[import]  # noqa: E402
    DEFAULT_CHECKS,
    Check,
    CheckResult,
    all_ok,
    evaluate,
    format_report,
    main,
    run_checks,
)


class FakeResponse:
    def __init__(self, status_code: int, content_type: str = '') -> None:
        self.status_code = status_code
        self.headers = {'Content-Type': content_type}


class FakeSession:
    """Records GETs; maps url -> FakeResponse (or an Exception to raise)."""

    def __init__(self, by_url: Optional[Dict[str, Any]] = None,
                 default: Any = None) -> None:
        self.by_url = by_url or {}
        self.default = default or FakeResponse(200, 'text/html')
        self.urls: List[str] = []

    def get(self, url: str, **kw: Any) -> Any:
        self.urls.append(url)
        resp = self.by_url.get(url, self.default)
        if isinstance(resp, Exception):
            raise resp
        return resp


class TestEvaluate(unittest.TestCase):
    def test_status_match_passes(self):
        c = Check('x', '/x', (200,))
        self.assertTrue(evaluate(c, status=200, content_type='text/html').ok)

    def test_status_mismatch_fails(self):
        c = Check('admin', '/skol/admin/login/', (200,))
        r = evaluate(c, status=404, content_type='text/html')
        self.assertFalse(r.ok)
        self.assertIn('404', r.detail)

    def test_content_type_substring_required(self):
        c = Check('css', '/skol/static/x.css', (200,), 'text/css')
        self.assertFalse(
            evaluate(c, status=200, content_type='text/html').ok)
        self.assertTrue(
            evaluate(c, status=200,
                     content_type='text/css; charset=utf-8').ok)

    def test_error_fails(self):
        c = Check('x', '/x', (200,))
        r = evaluate(c, status=None, content_type=None, error='timeout')
        self.assertFalse(r.ok)
        self.assertIn('timeout', r.detail)

    def test_multiple_acceptable_statuses(self):
        c = Check('admin', '/skol/admin/', (200, 302))
        self.assertTrue(
            evaluate(c, status=302, content_type='text/html').ok)


class TestRunChecks(unittest.TestCase):
    def test_builds_urls_and_evaluates(self):
        checks = [
            Check('admin', '/skol/admin/login/', (200,)),
            Check('css', '/skol/static/admin/css/base.css',
                  (200,), 'text/css'),
        ]
        sess = FakeSession(by_url={
            'https://h/skol/admin/login/': FakeResponse(200, 'text/html'),
            'https://h/skol/static/admin/css/base.css':
                FakeResponse(200, 'text/css'),
        })
        results = run_checks(checks, base_url='https://h', http=sess)
        self.assertTrue(all_ok(results))
        self.assertEqual(sess.urls, [
            'https://h/skol/admin/login/',
            'https://h/skol/static/admin/css/base.css',
        ])

    def test_base_url_trailing_slash_trimmed(self):
        sess = FakeSession()
        run_checks([Check('x', '/x', (200,))],
                   base_url='https://h/', http=sess)
        self.assertEqual(sess.urls, ['https://h/x'])

    def test_request_exception_becomes_failure(self):
        sess = FakeSession(by_url={'https://h/x': RuntimeError('boom')})
        results = run_checks([Check('x', '/x', (200,))],
                             base_url='https://h', http=sess)
        self.assertFalse(all_ok(results))
        self.assertIn('boom', results[0].detail)


class TestDefaults(unittest.TestCase):
    """The defaults must cover the three regressions we actually hit:
    Django reachable under /skol, admin static served by the Alias, and a
    favicon at the root."""

    def test_covers_the_three_regressions(self):
        paths = {c.path for c in DEFAULT_CHECKS}
        self.assertIn('/skol/admin/login/', paths)
        self.assertTrue(
            any('static' in p and p.endswith('.css') for p in paths))
        self.assertIn('/favicon.ico', paths)

    def test_brat_route_is_a_routing_guard(self):
        """/brat proxies to an on-demand backend (:8001). The probe
        guards routing, not service uptime: 200 (up) and 503 (Apache
        reached brat but the backend is down) both prove it is NOT
        falling through to the CouchDB catch-all, which would 404."""
        brat = next(c for c in DEFAULT_CHECKS if c.path == '/brat/')
        self.assertIn(200, brat.expect_status)
        self.assertIn(503, brat.expect_status)
        self.assertNotIn(404, brat.expect_status)


class TestReportAndExit(unittest.TestCase):
    def test_format_lists_pass_and_fail(self):
        results = [
            CheckResult(Check('a', '/a', (200,)), True, 'status 200'),
            CheckResult(Check('b', '/b', (200,)), False, 'status 404'),
        ]
        out = format_report(results)
        self.assertIn('a', out)
        self.assertIn('b', out)
        self.assertIn('PASS', out)
        self.assertIn('FAIL', out)

    def test_all_ok(self):
        self.assertTrue(
            all_ok([CheckResult(Check('a', '/a', (200,)), True, '')]))
        self.assertFalse(
            all_ok([CheckResult(Check('a', '/a', (200,)), False, '')]))


class TestMain(unittest.TestCase):
    def test_exit_zero_when_all_pass(self):
        sess = FakeSession(default=FakeResponse(200, 'text/css'))
        self.assertEqual(main(['--base-url', 'https://h'], http=sess), 0)

    def test_exit_nonzero_on_failure(self):
        sess = FakeSession(default=FakeResponse(404, 'text/html'))
        self.assertEqual(main(['--base-url', 'https://h'], http=sess), 1)


if __name__ == '__main__':
    unittest.main()
