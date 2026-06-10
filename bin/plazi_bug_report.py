#!/usr/bin/env python3
"""Turn logged Plazi ``searchByDOI`` failures into one bug report.

``backfill_plazi_uuids`` logs each failed lookup as a greppable
``PLAZI_FAILURE`` line carrying the doc id, failure reason, detail, DOI,
and exact URL.  This tool reads those back for one or more doc ids and
emits a single, Plazi-policy-compliant report:

  - ``failure.csv`` — every case in one CSV (per
    https://github.com/plazi/community, machine/batch audits must be one
    issue with a CSV, never individual cases);
  - a ``reproduce.sh`` script that replays every row of the CSV; and
  - a Markdown issue body with the expected-vs-observed framing, a
    per-failure-mode grouping, and instructions for retrieving the
    source skol document for each id from production CouchDB.

Usage:
    plazi_bug_report.py <id> [<id> ...] --log run.log [--out-dir DIR]
        [--live] [--couchdb-url URL] [--source-db NAME]

``--live`` re-runs each request to capture the current response as the
'observed' value; without it, 'observed' is derived from the log.

Exit status is 2 when some requested ids were not found in the log (the
report is still written for the ids that were), else 0.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import (
    Any, Dict, Iterable, List, Optional, Sequence, Tuple,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_plazi_uuids import (  # type: ignore[import]  # noqa: E402
    FailureRecord,
    _USER_AGENT,
    is_sticky_reason,
    parse_failure_log_line,
)


_DEFAULT_SITE_URL = 'https://synoptickeyof.life/skol'
_DEFAULT_SOURCE_DB = 'skol'
_DEFAULT_SCRIPT_NAME = 'reproduce.sh'
_CSV_NAME = 'failure.csv'
_REPORT_NAME = 'plazi-bug-report.md'

CSV_HEADER = [
    'doc_id', 'doi', 'reason', 'detail', 'url', 'expected', 'observed',
]

# The contract every searchByDOI call is expected to honour.
EXPECTED = (
    'HTTP 200 with a JSON array of at most 100 treatment objects (each '
    'with DocUuid / LnkDoi), or an empty array [] for a DOI that has no '
    'Plazi treatments.'
)

# A bash script that replays every row of the CSV.  Placeholders are
# substituted by ``repro_script`` (a raw template keeps the backslashes
# and ${} / %{} literal for bash and curl).
_REPRO_TEMPLATE = r'''#!/usr/bin/env bash
# Replay every Plazi searchByDOI failure listed in __CSV__.
# Usage: ./__SCRIPT__ [__CSV__]
set -uo pipefail
CSV="${1:-__CSV__}"
UA="__UA__"
python3 - "$CSV" <<'PY' | while IFS=$'\t' read -r doc_id doi url; do
import csv
import sys
with open(sys.argv[1], newline='') as fh:
    for row in csv.DictReader(fh):
        print('\t'.join((row['doc_id'], row['doi'], row['url'])))
PY
    echo "== ${doc_id} (${doi})"
    curl -sS -A "$UA" \
        -w '\nHTTP %{http_code}  %{size_download} bytes\n' "$url" | head -c 600
    echo
    echo '---'
done
'''


def _index_failures(lines: Iterable[str]) -> Dict[str, FailureRecord]:
    """Map doc id -> its most-recent failure from log ``lines``.  A doc
    that failed more than once (retried then failed again) keeps its last
    line; first-appearance order is preserved (dict insertion order)."""
    latest: Dict[str, FailureRecord] = {}
    for line in lines:
        rec = parse_failure_log_line(line)
        if rec is not None:
            latest[rec.doc_id] = rec
    return latest


def find_failures(
    lines: Iterable[str],
    identifiers: Sequence[str],
) -> Tuple[List[FailureRecord], List[str]]:
    """Recover the failures for ``identifiers`` from log ``lines``.

    Returns ``(found, missing)``.  ``found`` follows the order of
    ``identifiers`` (most-recent line per id); ``missing`` lists ids with
    no failure line at all.
    """
    latest = _index_failures(lines)
    found: List[FailureRecord] = []
    missing: List[str] = []
    for ident in identifiers:
        rec = latest.get(ident)
        if rec is None:
            missing.append(ident)
        else:
            found.append(rec)
    return found, missing


def all_failures(lines: Iterable[str]) -> List[FailureRecord]:
    """Every distinct failure in the log (most-recent line per id),
    first-appearance order — the whole-log mode used when no ids are
    named."""
    return list(_index_failures(lines).values())


def filter_failures(
    records: Sequence[FailureRecord],
    *,
    include_transient: bool = False,
    reasons: Optional[Sequence[str]] = None,
) -> List[FailureRecord]:
    """Scope a set of failures for a Plazi-facing report.

    - ``reasons`` (if given) restricts to exactly those failure reasons —
      handy for one report per failure class.
    - otherwise transient ``request_error`` failures (our own
      timeout / connection / DNS blips, not Plazi's fault) are dropped
      unless ``include_transient`` is set.
    """
    if reasons:
        wanted = set(reasons)
        return [r for r in records if r.reason in wanted]
    if include_transient:
        return list(records)
    return [r for r in records if is_sticky_reason(r.reason)]


def observed_for(record: FailureRecord) -> str:
    """A human sentence describing what Plazi actually returned, from the
    logged reason/detail."""
    reason, detail = record.reason, record.detail
    if reason == 'http_status':
        return f'HTTP {detail} (non-200 response).'
    if reason == 'runaway':
        return (
            f'HTTP 200 with {detail} entries — far exceeds the 100-entry '
            f'sanity cap; this looks like the full Plazi index rather '
            f'than a match for this DOI.'
        )
    if reason == 'request_error':
        return (
            f'No HTTP response — the request raised {detail} '
            f'(timeout / connection / DNS).'
        )
    if reason == 'bad_json':
        return 'HTTP 200, but the response body was not valid JSON.'
    if reason == 'not_list':
        return 'HTTP 200 with JSON that is not an array.'
    return f'{reason} (detail={detail}).'


def failure_csv(
    records: Sequence[FailureRecord],
    observed: Optional[Dict[str, str]] = None,
) -> str:
    """Render the cases as CSV text (header + one row per case).  The
    ``observed`` map (doc_id -> text, e.g. from ``--live``) overrides the
    log-derived observation."""
    overrides = observed or {}
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_HEADER)
    for rec in records:
        obs = overrides.get(rec.doc_id) or observed_for(rec)
        writer.writerow([
            rec.doc_id, rec.doi, rec.reason, rec.detail, rec.url,
            EXPECTED, obs,
        ])
    return buf.getvalue()


def repro_script(csv_filename: str, user_agent: str,
                 script_name: str = _DEFAULT_SCRIPT_NAME) -> str:
    """A self-contained bash script that curls every row of the CSV."""
    return (
        _REPRO_TEMPLATE
        .replace('__CSV__', csv_filename)
        .replace('__SCRIPT__', script_name)
        .replace('__UA__', user_agent)
    )


def retrieval_instructions(
    records: Sequence[FailureRecord],
    *,
    site_url: str,
    source_db: str,
) -> str:
    """Markdown: how to pull each failing doc's source attachment via the
    Django attachment API (``GET /api/pdf/<db>/<doc_id>/``).  The API
    proxies CouchDB server-side, so no credentials are needed or exposed.
    """
    base = site_url.rstrip('/')
    out: List[str] = [
        'Retrieve the source document attachment for any case via the '
        'Django API (no credentials needed; it proxies CouchDB '
        'server-side). Append a name (e.g. `article.txt`) to fetch a '
        'specific attachment; the default is `article.pdf`:',
        '',
        '```bash',
    ]
    out.extend(
        f'curl -sS {base}/api/pdf/{source_db}/{rec.doc_id}/ '
        f'-o {rec.doc_id}.pdf'
        for rec in records
    )
    out.append('```')
    return '\n'.join(out)


def issue_body(
    records: Sequence[FailureRecord],
    *,
    csv_filename: str,
    script_filename: str,
    user_agent: str,
    site_url: str,
    source_db: str,
) -> str:
    """Assemble the Markdown issue body for the whole batch."""
    by_reason: Dict[str, List[FailureRecord]] = {}
    for rec in records:
        by_reason.setdefault(rec.reason, []).append(rec)

    out: List[str] = [
        '# Plazi `searchByDOI` returns unexpected responses for a batch '
        'of DOIs',
        '',
        f'Reporting {len(records)} DOI lookup(s) against '
        '`/Treatments/searchByDOI` that did not return the expected '
        'result. Per the community guidance these are aggregated here '
        f'with a CSV (`{csv_filename}`) rather than filed individually.',
        '',
        '## Expected',
        '',
        EXPECTED,
        '',
        '## Observed (by failure mode)',
        '',
    ]
    for reason in sorted(by_reason):
        recs = by_reason[reason]
        out.append(f'### `{reason}` — {len(recs)} case(s)')
        out.append('')
        out.append(observed_for(recs[0]))
        out.append('')
        for rec in recs:
            out.append(
                f'- `{rec.doi}` (doc `{rec.doc_id}`) — '
                f'[exact query]({rec.url})'
            )
        out.append('')

    out += [
        '## Reproduce all cases',
        '',
        f'`{csv_filename}` lists every case (columns: '
        f'{", ".join(CSV_HEADER)}). The script `{script_filename}` '
        'replays each row:',
        '',
        '```bash',
        f'./{script_filename} {csv_filename}',
        '```',
        '',
    ]
    if records:
        out += [
            'A single case by hand:',
            '',
            '```bash',
            f'curl -sS -A "{user_agent}" "{records[0].url}"',
            '```',
            '',
        ]
    out += [
        '## Source documents',
        '',
        retrieval_instructions(
            records, site_url=site_url, source_db=source_db),
        '',
    ]
    return '\n'.join(out)


def live_observe(record: FailureRecord, http_client: Any) -> str:
    """Re-run one request now and describe the current response (status +
    a short body snippet).  Names the exception if the request raises."""
    try:
        resp = http_client.get(record.url)
    except Exception as exc:  # noqa: BLE001 — report whatever it raised
        return f'No HTTP response — {type(exc).__name__}: {exc}'
    snippet = (getattr(resp, 'text', '') or '')[:200].replace('\n', ' ')
    return f'HTTP {resp.status_code}; body[:200]={snippet!r}'


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Build one Plazi bug report (issue + failure.csv + '
                    'repro script) from logged searchByDOI failures.',
    )
    parser.add_argument(
        'identifiers', nargs='*',
        help='skol document _id(s) to include. Omit to report every '
             'failure in the log (whole-log mode).',
    )
    parser.add_argument(
        '--log', required=True,
        help='Backfill log file to scan for PLAZI_FAILURE lines.',
    )
    parser.add_argument(
        '--include-transient', action='store_true',
        help='Whole-log mode: also include transient request_error '
             '(timeout/connection/DNS) failures, which are usually our '
             "side rather than Plazi's. Ignored when ids are named.",
    )
    parser.add_argument(
        '--reason', nargs='*', default=None,
        help='Whole-log mode: restrict to these failure reasons (e.g. '
             '`--reason runaway` for one report per class). Ignored when '
             'ids are named.',
    )
    parser.add_argument(
        '--out-dir', default='.',
        help='Directory for failure.csv, the repro script, and the '
             'report (default: current directory).',
    )
    parser.add_argument(
        '--site-url', default=_DEFAULT_SITE_URL,
        help=f'skol site base URL for the Django attachment API used in '
             f'retrieval instructions (default: {_DEFAULT_SITE_URL}).',
    )
    parser.add_argument(
        '--source-db', default=_DEFAULT_SOURCE_DB,
        help=f'CouchDB database holding the docs '
             f'(default: {_DEFAULT_SOURCE_DB}).',
    )
    parser.add_argument(
        '--user-agent', default=_USER_AGENT,
        help='User-Agent for the repro script / --live re-runs.',
    )
    parser.add_argument(
        '--script-name', default=_DEFAULT_SCRIPT_NAME,
        help=f'Name of the repro script (default: {_DEFAULT_SCRIPT_NAME}).',
    )
    parser.add_argument(
        '--live', action='store_true',
        help='Re-run each request now to capture the current response as '
             "the 'observed' value.",
    )
    args = parser.parse_args(argv)

    lines = Path(args.log).read_text(encoding='utf-8').splitlines()
    if args.identifiers:
        found, missing = find_failures(lines, args.identifiers)
    else:
        found = filter_failures(
            all_failures(lines),
            include_transient=args.include_transient,
            reasons=args.reason,
        )
        missing = []

    if not found:
        if args.identifiers:
            print(
                'No matching PLAZI_FAILURE lines for: '
                + ', '.join(args.identifiers),
                file=sys.stderr,
            )
            return 2
        print('No failures found in the log; nothing to report.',
              file=sys.stderr)
        return 0

    observed: Optional[Dict[str, str]] = None
    if args.live:
        from ingestors.rate_limited_client import (  # noqa: E402
            RateLimitedHttpClient,
        )
        client = RateLimitedHttpClient(
            user_agent=args.user_agent, verbosity=0)
        observed = {
            rec.doc_id: live_observe(rec, client) for rec in found
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _CSV_NAME).write_text(
        failure_csv(found, observed=observed), encoding='utf-8')
    (out_dir / args.script_name).write_text(
        repro_script(_CSV_NAME, args.user_agent, args.script_name),
        encoding='utf-8')
    (out_dir / _REPORT_NAME).write_text(
        issue_body(
            found, csv_filename=_CSV_NAME,
            script_filename=args.script_name, user_agent=args.user_agent,
            site_url=args.site_url, source_db=args.source_db),
        encoding='utf-8')

    print(
        f'Wrote {out_dir / _CSV_NAME}, {out_dir / args.script_name}, '
        f'{out_dir / _REPORT_NAME} ({len(found)} case(s)).',
        file=sys.stderr,
    )
    for ident in missing:
        print(f'  (no log entry for requested id: {ident})', file=sys.stderr)
    return 2 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
