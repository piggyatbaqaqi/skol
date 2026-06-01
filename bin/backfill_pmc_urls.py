#!/usr/bin/env python3
"""Backfill ``pdf_url`` / ``xml_url`` onto skol_dev PMC docs that
were ingested before the PMC ingestor started recording them.

Idempotent — re-runs no-op once URLs are populated.  Supports
``--dry-run`` / ``--limit`` / ``--verbosity``.

The URLs are derived from the doc's existing ``pmcid`` field;
no network calls.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Pure helpers (covered by backfill_pmc_urls_test.py)
# ---------------------------------------------------------------------------


def compute_pmc_url_update(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return the URL fields that would be added to ``doc``.

    Empty dict means "no change needed":
    - doc has no ``pmcid`` (not a PMC doc — out of scope);
    - both URL fields are already populated with non-empty strings.

    Otherwise returns just the missing field(s).
    """
    pmcid = doc.get('pmcid')
    if not pmcid:
        return {}

    from ingestors.pmc import PmcIngestor
    update: Dict[str, Any] = {}
    if not (doc.get('pdf_url') or '').strip():
        update['pdf_url'] = PmcIngestor.pmc_article_url(pmcid)
    if not (doc.get('xml_url') or '').strip():
        update['xml_url'] = PmcIngestor.pmc_oai_xml_url(pmcid)
    return update


# ---------------------------------------------------------------------------
# CouchDB iteration / write (network-touching)
# ---------------------------------------------------------------------------


def _iter_db(db: Any, verbosity: int = 1) -> Iterable[Dict[str, Any]]:
    """Yield every non-design doc from ``db``."""
    for doc_id in db:
        if doc_id.startswith('_design/'):
            continue
        try:
            yield db[doc_id]
        except Exception as exc:
            if verbosity >= 1:
                print(f'  warning: could not load {doc_id}: {exc}',
                      file=sys.stderr)


def backfill(
    db: Any,
    dry_run: bool = False,
    limit: Optional[int] = None,
    verbosity: int = 1,
) -> Dict[str, int]:
    """Walk ``db``; for each PMC doc missing URLs, apply the update.

    Returns ``{scanned, eligible, written}``.
    """
    counts = {'scanned': 0, 'eligible': 0, 'written': 0}
    for doc in _iter_db(db, verbosity=verbosity):
        counts['scanned'] += 1
        if verbosity >= 2 and counts['scanned'] % 1000 == 0:
            print(f'  scanned {counts["scanned"]} docs...')
        update = compute_pmc_url_update(doc)
        if not update:
            continue
        counts['eligible'] += 1
        if limit is not None and counts['written'] >= limit:
            if verbosity >= 1:
                print(f'  stop: hit --limit {limit}')
            break
        if verbosity >= 1:
            tag = '(DRY RUN) ' if dry_run else ''
            fields = ', '.join(sorted(update.keys()))
            print(f'  {tag}{doc["_id"]}: + {fields}')
        if not dry_run:
            doc.update(update)
            db.save(doc)
        counts['written'] += 1
    return counts


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Backfill pdf_url / xml_url on PMC docs in '
                    'skol_dev that were ingested before the URL '
                    'fields were recorded.',
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview updates without writing.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Stop after N successful updates.')
    parser.add_argument('--verbosity', type=int, default=None,
                        help='0=quiet, 1=normal, 2=verbose.')
    args, _unknown = parser.parse_known_args()

    from env_config import get_env_config
    config = get_env_config()
    verbosity = (args.verbosity if args.verbosity is not None
                 else config.get('verbosity', 1))

    import couchdb
    server = couchdb.Server(config['couchdb_url'])
    src_user = config.get('couchdb_username') or ''
    src_pass = config.get('couchdb_password') or ''
    if src_user and src_pass:
        server.resource.credentials = (src_user, src_pass)
    db_name = (config.get('ingest_db_name')
               or config.get('couchdb_database')
               or 'skol_dev')
    if db_name not in server:
        print(f'error: database {db_name!r} not found at '
              f'{config["couchdb_url"]}', file=sys.stderr)
        return 2
    if verbosity >= 1:
        print(f'Target DB: {db_name} '
              f'{"(DRY RUN)" if args.dry_run else ""}')

    counts = backfill(
        db=server[db_name],
        dry_run=args.dry_run,
        limit=args.limit,
        verbosity=verbosity,
    )

    print()
    print(f'Scanned:   {counts["scanned"]}')
    print(f'Eligible:  {counts["eligible"]}')
    print(f'{"Would write" if args.dry_run else "Written"}: '
          f'{counts["written"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
