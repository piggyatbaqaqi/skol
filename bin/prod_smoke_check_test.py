"""Tests for bin/prod_smoke_check.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

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
    def __init__(self, status_code: int, content_type: str = '',
                 text: str = '') -> None:
        self.status_code = status_code
        self.headers = {'Content-Type': content_type}
        self.text = text


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


_XFAIL_BODY_SUBSTRING = pytest.mark.xfail(
    reason=(
        "2026-08-08: Check.expect_body_substring / evaluate(body=...) "
        "not yet implemented; lands in the follow-up commit."
    ),
    strict=True,
)


class TestEvaluateBodySubstring(unittest.TestCase):
    """A content-type alone cannot tell the landing page from an Apache
    error page or a blank index.html — both are text/html.  So a check
    may also demand a marker string in the body."""

    @_XFAIL_BODY_SUBSTRING
    def test_body_substring_present_passes(self):
        c = Check('root', '/', (200,), 'text/html',
                  expect_body_substring='Synoptic Key')
        r = evaluate(c, status=200, content_type='text/html',
                     body='<title>Coming soon: Synoptic Key Of Life!</title>')
        self.assertTrue(r.ok)

    @_XFAIL_BODY_SUBSTRING
    def test_body_substring_absent_fails(self):
        c = Check('root', '/', (200,), 'text/html',
                  expect_body_substring='Synoptic Key')
        r = evaluate(c, status=200, content_type='text/html',
                     body='<h1>Apache2 Ubuntu Default Page</h1>')
        self.assertFalse(r.ok)
        self.assertIn('Synoptic Key', r.detail)

    @_XFAIL_BODY_SUBSTRING
    def test_body_substring_is_case_insensitive(self):
        """index.html capitalises it 'Synoptic Key Of Life'; a copy-edit
        to 'of' must not turn the smoke check red."""
        c = Check('root', '/', (200,), 'text/html',
                  expect_body_substring='synoptic key')
        r = evaluate(c, status=200, content_type='text/html',
                     body='SYNOPTIC KEY OF LIFE')
        self.assertTrue(r.ok)

    @_XFAIL_BODY_SUBSTRING
    def test_missing_body_fails_when_substring_demanded(self):
        c = Check('root', '/', (200,), 'text/html',
                  expect_body_substring='Synoptic Key')
        r = evaluate(c, status=200, content_type='text/html', body=None)
        self.assertFalse(r.ok)

    @_XFAIL_BODY_SUBSTRING
    def test_body_ignored_when_not_demanded(self):
        """Checks that do not set expect_body_substring keep passing
        regardless of body — no retrofitting of the other routes."""
        c = Check('favicon', '/favicon.ico', (200,))
        self.assertTrue(
            evaluate(c, status=200, content_type='image/png',
                     body='anything at all').ok)


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


_XFAIL_BARE_ROOT = pytest.mark.xfail(
    reason=(
        "2026-08-08: bare '/' landing-page check not yet in "
        "DEFAULT_CHECKS; implementation lands in the follow-up commit."
    ),
    strict=True,
)


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

    @_XFAIL_BARE_ROOT
    def test_covers_bare_root(self):
        """Bare '/' must serve the DocumentRoot landing page
        (/var/www/skol/index.html), not the CouchDB catch-all.  The
        2026-06-10 <Location /> block that fixed the June 9 outage
        swallowed the document root as collateral damage, and this
        script's blind spot let it hide for two months."""
        self.assertIn('/', {c.path for c in DEFAULT_CHECKS})

    @_XFAIL_BARE_ROOT
    def test_bare_root_demands_html_because_status_cannot_tell(self):
        """CouchDB's welcome banner answers '/' with 200 too, so status
        alone cannot detect the regression.  Content-type is the only
        signal that separates the landing page (text/html) from the
        proxied CouchDB welcome (application/json)."""
        root = next(c for c in DEFAULT_CHECKS if c.path == '/')
        self.assertEqual(root.expect_status, (200,))
        self.assertIsNotNone(root.expect_content_type)
        self.assertIn('text/html', root.expect_content_type or '')
        self.assertTrue(root.expect_body_substring)


# The two responses prod can actually give for '/': the landing page
# Apache serves from DocumentRoot, and the CouchDB welcome banner that
# the <Location /> catch-all substituted for it on 2026-06-10.
_LANDING_BODY = (
    '<!DOCTYPE html><html lang="en"><head>'
    '<title>Coming soon: Synoptic Key Of Life!</title></head></html>'
)
_COUCHDB_WELCOME_BODY = (
    '{"couchdb":"Welcome","version":"3.4.3","git_sha":"e12b967d7"}'
)


class TestBareRootDiscrimination(unittest.TestCase):
    """The bare-root check must actually distinguish the landing page
    from the CouchDB welcome blob that replaced it."""

    @_XFAIL_BARE_ROOT
    def test_couchdb_welcome_fails_the_root_check(self):
        root = next(c for c in DEFAULT_CHECKS if c.path == '/')
        # What prod really returned on 2026-08-08: 200 + CouchDB JSON.
        result = evaluate(root, status=200, content_type='application/json',
                          body=_COUCHDB_WELCOME_BODY)
        self.assertFalse(result.ok)

    @_XFAIL_BARE_ROOT
    def test_landing_page_passes_the_root_check(self):
        root = next(c for c in DEFAULT_CHECKS if c.path == '/')
        result = evaluate(root, status=200,
                          content_type='text/html; charset=UTF-8',
                          body=_LANDING_BODY)
        self.assertTrue(result.ok)

    @_XFAIL_BARE_ROOT
    def test_html_error_page_fails_the_root_check(self):
        """The reason we match the body too: a stray Apache error page
        is text/html and would sail past a content-type-only check."""
        root = next(c for c in DEFAULT_CHECKS if c.path == '/')
        result = evaluate(root, status=200, content_type='text/html',
                          body='<h1>Apache2 Ubuntu Default Page</h1>')
        self.assertFalse(result.ok)

    @_XFAIL_BARE_ROOT
    def test_run_checks_threads_the_body_through(self):
        """evaluate() only sees a body if run_checks passes resp.text."""
        root = next(c for c in DEFAULT_CHECKS if c.path == '/')
        sess = FakeSession(by_url={
            'https://h/': FakeResponse(200, 'text/html',
                                       _COUCHDB_WELCOME_BODY),
        })
        results = run_checks([root], base_url='https://h', http=sess)
        self.assertFalse(all_ok(results))


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


def _satisfying_session(base_url: str = 'https://h') -> FakeSession:
    """A FakeSession answering every DEFAULT_CHECKS url with a response
    that check accepts.  Derived from the checks themselves so adding a
    check with a different content-type does not silently break this."""
    return FakeSession(by_url={
        f'{base_url}{c.path}': FakeResponse(
            c.expect_status[0],
            c.expect_content_type or 'text/html',
            getattr(c, 'expect_body_substring', None) or 'ok')
        for c in DEFAULT_CHECKS
    })


class TestMain(unittest.TestCase):
    def test_exit_zero_when_all_pass(self):
        sess = _satisfying_session()
        self.assertEqual(main(['--base-url', 'https://h'], http=sess), 0)

    def test_exit_nonzero_on_failure(self):
        sess = FakeSession(default=FakeResponse(404, 'text/html'))
        self.assertEqual(main(['--base-url', 'https://h'], http=sess), 1)


if __name__ == '__main__':
    unittest.main()
