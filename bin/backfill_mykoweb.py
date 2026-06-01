#!/usr/bin/env python3
"""Backfill curated journal/book metadata onto existing
mykoweb-literature documents in skol_dev.

The LocalMykowebLiteratureIngestor now consumes
``systematics_pdf_metadata.json`` and writes curated fields
(itemtype=article, journal, volume, issue, pages, author, year,
title) for new ingestions.  Documents already in skol_dev — ingested
before the integration landed — still carry the old filename-as-
title / itemtype=book shape and are short-circuited by the ingestor's
"already has PDF, skip" dedup logic.  This script walks skol_dev,
applies the same metadata_to_doc_fields() translation in place, and
saves only the docs that would actually change.  Idempotent — re-runs
are no-ops on docs already at the target state.

Usage:
    bin/backfill_mykoweb.py [--dry-run] [--limit N] [--verbosity N]

Reads metadata from
``/data/skol/www/mykoweb.com/systematics_pdf_metadata.json`` by
default; override with ``--metadata-path``.  CouchDB target comes
from env_config (COUCHDB_URL / INGEST_DATABASE / etc.) — same as
backfill_journal.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingestors.mykoweb_metadata import (  # noqa: E402
    load_metadata_index,
    metadata_to_doc_fields,
)


_MYKOWEB_HOST_PREFIX = 'https://mykoweb.com/'


# ---------------------------------------------------------------------------
# Pure helpers (covered by backfill_mykoweb_test.py)
# ---------------------------------------------------------------------------


def pdf_url_to_lookup_key(pdf_url: Optional[str]) -> Optional[str]:
    """Convert a stored ``pdf_url`` (full mykoweb URL) into the
    site-relative key used in systematics_pdf_metadata.json.

    Returns None for URLs that aren't under ``mykoweb.com/``.
    """
    if not pdf_url:
        return None
    if not pdf_url.startswith(_MYKOWEB_HOST_PREFIX):
        return None
    return pdf_url[len(_MYKOWEB_HOST_PREFIX):]


def is_mykoweb_literature(doc: Dict[str, Any]) -> bool:
    """Identify mykoweb-literature docs eligible for this backfill.

    The LocalMykowebLiteratureIngestor stamps each ingested doc with
    ``meta = {'source': 'mykoweb', 'type': 'literature'}``.  Older
    docs without a ``meta`` block (or with a non-dict meta) are
    ineligible.
    """
    meta = doc.get('meta')
    if not isinstance(meta, dict):
        return False
    return (meta.get('source') == 'mykoweb'
            and meta.get('type') == 'literature')


def compute_field_update(
    existing_doc: Dict[str, Any],
    metadata_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the dict of fields that would change on this doc.

    Looks up the doc's ``pdf_url`` in ``metadata_index``, runs
    ``metadata_to_doc_fields`` to get the canonical field set, and
    drops any fields that already match the existing doc.  Empty
    return value means "no update needed" — the caller skips the
    db.save() in that case.
    """
    key = pdf_url_to_lookup_key(existing_doc.get('pdf_url'))
    if key is None:
        return {}
    record = metadata_index.get(key)
    if record is None:
        return {}
    new_fields = metadata_to_doc_fields(record)
    return {
        k: v for k, v in new_fields.items()
        if existing_doc.get(k) != v
    }


# ---------------------------------------------------------------------------
# CouchDB iteration / write (network-touching)
# ---------------------------------------------------------------------------


def _iter_eligible_docs(
    db: Any, verbosity: int = 1,
) -> Iterator[Dict[str, Any]]:
    """Yield mykoweb-literature docs from ``db`` in deterministic
    (``_id``-sorted) order."""
    doc_count = 0
    eligible = 0
    for doc_id in db:
        if doc_id.startswith('_design/'):
            continue
        doc_count += 1
        try:
            doc = db[doc_id]
        except Exception as exc:
            if verbosity >= 1:
                print(f'  warning: could not load {doc_id}: {exc}')
            continue
        if is_mykoweb_literature(doc):
            eligible += 1
            yield doc
    if verbosity >= 1:
        print(f'  Scanned {doc_count} docs, {eligible} eligible '
              f'(meta.source=mykoweb, meta.type=literature)')


def backfill(
    db: Any,
    metadata_index: Dict[str, Dict[str, Any]],
    dry_run: bool = False,
    limit: Optional[int] = None,
    verbosity: int = 1,
) -> Dict[str, int]:
    """Walk eligible docs, compute updates, apply them.

    Returns a counts dict: ``{eligible, updated, no_change,
    not_in_index}``.
    """
    counts = {'eligible': 0, 'updated': 0,
              'no_change': 0, 'not_in_index': 0}
    for doc in _iter_eligible_docs(db, verbosity=verbosity):
        counts['eligible'] += 1
        if limit is not None and counts['updated'] >= limit:
            if verbosity >= 1:
                print(f'  stop: hit --limit {limit}')
            break
        key = pdf_url_to_lookup_key(doc.get('pdf_url'))
        if key is None or key not in metadata_index:
            counts['not_in_index'] += 1
            if verbosity >= 2:
                print(f'  - {doc.get("_id")}: no metadata for '
                      f'{doc.get("pdf_url")}')
            continue
        update = compute_field_update(doc, metadata_index)
        if not update:
            counts['no_change'] += 1
            if verbosity >= 2:
                print(f'  ✓ {doc.get("_id")}: already up to date')
            continue
        if verbosity >= 1:
            preview = ', '.join(f'{k}={v!r}'
                                for k, v in list(update.items())[:3])
            tag = '(DRY RUN) ' if dry_run else ''
            print(f'  {tag}update {doc.get("_id")}: {preview}'
                  f'{" ..." if len(update) > 3 else ""}')
        if not dry_run:
            doc.update(update)
            db.save(doc)
        counts['updated'] += 1
    return counts


def main() -> int:
    """CLI entry point — parse args, dispatch to ``backfill()``."""
    parser = argparse.ArgumentParser(
        description='Backfill curated mykoweb metadata onto existing '
                    'mykoweb-literature docs in skol_dev.',
    )
    parser.add_argument(
        '--metadata-path', type=Path,
        default=Path('/data/skol/www/mykoweb.com/'
                     'systematics_pdf_metadata.json'),
        help='Path to systematics_pdf_metadata.json',
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview updates without writing.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Stop after N successful updates.')
    parser.add_argument('--verbosity', type=int, default=None,
                        help='0=quiet, 1=normal, 2=verbose.')
    # parse_known_args so env_config CLI flags (--couchdb-url,
    # --couchdb-username, --couchdb-password, --ingest-db-name, ...)
    # pass through.
    args, _unknown = parser.parse_known_args()

    # Lazy imports so the unit tests don't need couchdb / env_config.
    from env_config import get_env_config
    config = get_env_config()
    verbosity = (args.verbosity if args.verbosity is not None
                 else config.get('verbosity', 1))

    metadata_index = load_metadata_index(args.metadata_path)
    if not metadata_index:
        print(f'error: no metadata loaded from {args.metadata_path}',
              file=sys.stderr)
        return 2
    if verbosity >= 1:
        print(f'Loaded {len(metadata_index)} metadata records '
              f'from {args.metadata_path}')

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
        metadata_index=metadata_index,
        dry_run=args.dry_run,
        limit=args.limit,
        verbosity=verbosity,
    )

    print()
    print(f'Eligible:      {counts["eligible"]}')
    print(f'Updated:       {counts["updated"]}'
          f'{" (would have, dry-run)" if args.dry_run else ""}')
    print(f'No change:     {counts["no_change"]}')
    print(f'Not in index:  {counts["not_in_index"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
