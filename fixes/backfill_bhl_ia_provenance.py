#!/usr/bin/env python3
"""Backfill IA scan provenance onto BHL volume documents stored before
BhlIngestor recorded it (commit 07e2b08): the six Lloyd volumes in
skol_dev, 2026-11-23.

Adds ``ia_contributor``, ``ia_scanning_center``, ``ia_scan_date``,
``ia_image_count`` and ``ia_pdf_md5`` from archive.org's metadata API.
For a volume stored whole (no ``withheld_pages``), the stored
article.pdf's CouchDB md5 digest is compared with IA's; mismatches are
reported, not acted on.  Idempotent: documents that already carry any
``ia_*`` field are skipped.  One metadata request per document, rate
limited; no PDF is downloaded.

Usage: fixes/backfill_bhl_ia_provenance.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import base64
import binascii
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

IA_FIELDS = ('ia_contributor', 'ia_scanning_center', 'ia_scan_date',
             'ia_image_count', 'ia_pdf_md5')


def needs_provenance(doc: Dict[str, Any]) -> bool:
    """A BHL volume with an IA identifier and no ``ia_*`` field yet."""
    if doc.get('source') != 'bhl':
        return False
    if not str(doc.get('ia_identifier') or '').strip():
        return False
    return not any(field in doc for field in IA_FIELDS)


def attachment_md5_hex(doc: Dict[str, Any],
                       name: str = 'article.pdf') -> Optional[str]:
    """Hex md5 of an attachment from its CouchDB ``md5-<base64>``
    digest, or None if absent or not an md5 digest."""
    attachment = (doc.get('_attachments') or {}).get(name) or {}
    digest = str(attachment.get('digest') or '')
    if not digest.startswith('md5-'):
        return None
    try:
        return base64.b64decode(digest[4:], validate=True).hex()
    except (binascii.Error, ValueError):
        return None


PAGE_SIZE = 100


def bhl_docs(db: Any) -> Iterable[Dict[str, Any]]:
    """Every ``source: bhl`` document, by Mango query, a page at a time:
    _find returns 25 documents unless told otherwise, and couchdb-python
    does not hand back the bookmark, so page with limit/skip (stable
    here: ``source`` is never changed)."""
    skip = 0
    while True:
        page = list(db.find({'selector': {'source': 'bhl'},
                             'limit': PAGE_SIZE, 'skip': skip}))
        yield from page
        if len(page) < PAGE_SIZE:
            return
        skip += len(page)


def backfill(
    db: Any,
    fetch: Callable[[str], Dict[str, Any]],
    dry_run: bool = False,
    limit: Optional[int] = None,
    verbosity: int = 1,
) -> Dict[str, int]:
    """
    Fill provenance on every BHL document that needs it.

    ``fetch(identifier)`` returns the ``ia_*`` fields (``{}`` on
    failure).  Returns counts: ``scanned``, ``eligible``, ``written``,
    ``lookup_failed``, ``md5_mismatch``.
    """
    from ingestors.timestamps import set_timestamps

    counts = {'scanned': 0, 'eligible': 0, 'written': 0,
              'lookup_failed': 0, 'md5_mismatch': 0}
    for doc in bhl_docs(db):
        counts['scanned'] += 1
        if not needs_provenance(doc):
            continue
        counts['eligible'] += 1
        if limit is not None and counts['written'] >= limit:
            if verbosity >= 1:
                print(f'  stop: hit --limit {limit}')
            break
        identifier = str(doc['ia_identifier']).strip()
        update = fetch(identifier)
        if not update:
            counts['lookup_failed'] += 1
            if verbosity >= 1:
                print(f'  {doc["_id"]}: IA lookup failed for {identifier}',
                      file=sys.stderr)
            continue
        expected = update.get('ia_pdf_md5')
        if expected and not doc.get('withheld_pages'):
            stored = attachment_md5_hex(doc)
            if stored is not None and stored != expected:
                counts['md5_mismatch'] += 1
                if verbosity >= 1:
                    print(f'  {doc["_id"]}: stored article.pdf md5 {stored}'
                          f' != IA {expected} ({identifier})',
                          file=sys.stderr)
        if verbosity >= 1:
            tag = '(DRY RUN) ' if dry_run else ''
            print(f'  {tag}{doc["_id"]} {identifier}: + '
                  f'{", ".join(sorted(update))}')
        if not dry_run:
            doc.update(update)
            set_timestamps(doc)
            db.save(doc)
        counts['written'] += 1
    return counts


def main() -> int:
    """CLI entry point."""
    import argparse
    from urllib.robotparser import RobotFileParser

    import couchdb

    from env_config import common_parser, get_env_config
    from ingestors.bhl import BhlIngestor, parse_ia_metadata
    from ingestors.rate_limited_client import RateLimitedHttpClient

    parser = argparse.ArgumentParser(
        description='Backfill IA scan provenance (ia_* fields) onto BHL '
                    'volume documents stored before BhlIngestor recorded '
                    'it.',
        parents=[common_parser()],
    )
    args = parser.parse_args()
    config = get_env_config(cli_args=args)
    verbosity = int(config.get('verbosity') or 1)

    robots = RobotFileParser('https://archive.org/robots.txt')
    robots.read()
    client = RateLimitedHttpClient(
        user_agent='synoptickeyof.life', robot_parser=robots,
        verbosity=verbosity, rate_limit_min_ms=1000, rate_limit_max_ms=3000)

    def fetch(identifier: str) -> Dict[str, Any]:
        url = f'{BhlIngestor.IA_METADATA}/{identifier}'
        try:
            response = client.get(url)
            if response.status_code != 200:
                return {}
            return parse_ia_metadata(response.json(), identifier)
        except Exception:
            return {}

    server = couchdb.Server(config['couchdb_url'])
    if config.get('couchdb_username') and config.get('couchdb_password'):
        server.resource.credentials = (config['couchdb_username'],
                                       config['couchdb_password'])
    db_name = (config.get('ingest_db_name') or config.get('couchdb_database')
               or 'skol_dev')
    if db_name not in server:
        print(f'error: database {db_name!r} not found', file=sys.stderr)
        return 2
    dry_run = bool(config.get('dry_run'))
    if verbosity >= 1:
        print(f'Target DB: {db_name} {"(DRY RUN)" if dry_run else ""}')
    counts = backfill(server[db_name], fetch, dry_run=dry_run,
                      limit=config.get('limit'), verbosity=verbosity)
    print()
    for key in ('scanned', 'eligible', 'written', 'lookup_failed',
                'md5_mismatch'):
        print(f'{key}: {counts[key]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
