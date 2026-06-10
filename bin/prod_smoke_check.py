#!/usr/bin/env python3
"""Smoke-check the production reverse-proxy routing.

Motivated by 2026-06-09: an unattended ``apache2`` security upgrade
fully restarted Apache, which activated a ``mod_proxy`` precedence change
and silently rerouted ``/skol/*`` from Django to the ``<Location />`` →
CouchDB catch-all — breaking the admin, its static files, and the
favicon, with no config change on our part.

This script curls the handful of URLs whose routing those regressions
broke and exits non-zero if any is wrong, so a cron/timer catches the
next surprise in minutes instead of by hand.  Run it after the daily
unattended-upgrade window (Apache restarts ~06:14).

Usage::

    bin/prod_smoke_check.py                 # checks synoptickeyof.life
    bin/prod_smoke_check.py --quiet         # print only on failure (cron)
    bin/prod_smoke_check.py --base-url https://staging.example
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # type: ignore[import-untyped]  # noqa: E402


_DEFAULT_BASE_URL = 'https://synoptickeyof.life'
_DEFAULT_TIMEOUT = 10


@dataclass(frozen=True)
class Check:
    name: str
    path: str
    expect_status: Tuple[int, ...]
    expect_content_type: Optional[str] = None


@dataclass(frozen=True)
class CheckResult:
    check: Check
    ok: bool
    detail: str


# The three routes whose proxy precedence the apache upgrade broke, each
# tied to a specific <Location> escape from the CouchDB catch-all:
#   /skol/...          -> Django   (else CouchDB "Database does not exist")
#   /skol/static/...   -> Alias    (else proxied to Django, which 404s)
#   /favicon.ico       -> Alias    (else CouchDB 404)
DEFAULT_CHECKS: List[Check] = [
    # 200 distinguishes Django from CouchDB's 404 "Database does not
    # exist"; no content-type needed here.
    Check('django-admin', '/skol/admin/login/', (200,)),
    # The content-type is the real signal: a proxied-to-Django 404 would
    # be text/html, so demand text/css to prove the Alias served the file.
    Check('django-static-css', '/skol/static/admin/css/base.css',
          (200,), 'text/css'),
    Check('favicon', '/favicon.ico', (200,)),
    # brat proxies to an on-demand backend (:8001). 200 = up, 503 =
    # Apache reached brat but the backend is down — both prove routing
    # didn't fall through to CouchDB (which would 404). Routing guard,
    # not an uptime check.
    Check('brat', '/brat/', (200, 503)),
]


def evaluate(
    check: Check,
    status: Optional[int],
    content_type: Optional[str],
    error: Optional[str] = None,
) -> CheckResult:
    """Decide pass/fail for one check from the observed response (pure)."""
    if error is not None:
        return CheckResult(check, False, f'request error: {error}')
    if status not in check.expect_status:
        return CheckResult(
            check, False,
            f'status {status}, expected {list(check.expect_status)}')
    if (check.expect_content_type
            and check.expect_content_type not in (content_type or '')):
        return CheckResult(
            check, False,
            f'content-type {content_type!r} lacks '
            f'{check.expect_content_type!r}')
    detail = f'status {status}'
    if content_type:
        detail += f' ({content_type})'
    return CheckResult(check, True, detail)


def run_checks(
    checks: List[Check], *,
    base_url: str, http: Any,
    verify: bool = True, timeout: int = _DEFAULT_TIMEOUT,
) -> List[CheckResult]:
    """Fetch each check's URL and evaluate it.  A request that raises
    becomes a failed CheckResult rather than aborting the run."""
    base = base_url.rstrip('/')
    results: List[CheckResult] = []
    for check in checks:
        url = f'{base}{check.path}'
        try:
            resp = http.get(url, timeout=timeout, verify=verify)
        except Exception as exc:  # noqa: BLE001 — any failure is a failure
            results.append(evaluate(
                check, None, None, error=f'{type(exc).__name__}: {exc}'))
            continue
        headers = getattr(resp, 'headers', {}) or {}
        content_type = headers.get('Content-Type', '')
        results.append(evaluate(check, resp.status_code, content_type))
    return results


def all_ok(results: List[CheckResult]) -> bool:
    return all(r.ok for r in results)


def format_report(results: List[CheckResult]) -> str:
    lines: List[str] = []
    for r in results:
        tag = 'PASS' if r.ok else 'FAIL'
        lines.append(f'[{tag}] {r.check.name:<18} {r.check.path:<40} '
                     f'{r.detail}')
    n_fail = sum(1 for r in results if not r.ok)
    lines.append(
        f'{len(results) - n_fail}/{len(results)} passed'
        + ('' if n_fail == 0 else f' — {n_fail} FAILED'))
    return '\n'.join(lines)


def main(argv: Optional[List[str]] = None, *, http: Any = None) -> int:
    parser = argparse.ArgumentParser(
        description='Smoke-check production /skol reverse-proxy routing.',
    )
    parser.add_argument(
        '--base-url', default=_DEFAULT_BASE_URL,
        help=f'Site root to check (default: {_DEFAULT_BASE_URL}).',
    )
    parser.add_argument(
        '--insecure', action='store_true',
        help='Skip TLS certificate verification.',
    )
    parser.add_argument(
        '--timeout', type=int, default=_DEFAULT_TIMEOUT,
        help=f'Per-request timeout in seconds (default: {_DEFAULT_TIMEOUT}).',
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Print nothing when everything passes (for cron — any output '
             'then signals a problem).',
    )
    args = parser.parse_args(argv)

    session = http if http is not None else requests.Session()
    results = run_checks(
        DEFAULT_CHECKS, base_url=args.base_url, http=session,
        verify=not args.insecure, timeout=args.timeout)
    ok = all_ok(results)
    if not (ok and args.quiet):
        stream = sys.stdout if ok else sys.stderr
        print(format_report(results), file=stream)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
