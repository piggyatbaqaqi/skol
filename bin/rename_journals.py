#!/usr/bin/env python3
"""Rewrite ``journal`` fields on skol_dev docs to match the post-
refactor canonical names.

Phase 2 of the JOURNALS+SOURCES consolidation
(docs/publications_metadata_consolidation.md).  Walks the ingest
database, identifies docs whose ``journal`` field carries an old
compound suffix (``" (PMC)"``, ``" (Taylor & Francis)"``,
``" (Internet Archive)"``), and rewrites them to the canonical
journal name.

Idempotent — re-running is a no-op once docs are at the target
state.  Supports ``--dry-run``, ``--limit``, ``--verbosity``.

Phase 3 will reuse this script with a different mapping function
(post-JOURNAL_NAME_ALIASES-migration ``normalize_journal_name``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Pure helpers (covered by rename_journals_test.py)
# ---------------------------------------------------------------------------


def compute_renames(
    docs: Iterable[Dict[str, Any]],
    mapping_fn: Callable[[str], str],
) -> Dict[str, Tuple[str, str]]:
    """Walk ``docs``; return ``{doc_id: (old_journal, new_journal)}``
    for entries the script would actually update.

    Skips:
    - ``_design/...`` documents (CouchDB internals);
    - docs missing the ``journal`` field;
    - docs with empty / whitespace-only / non-string ``journal``;
    - docs where ``mapping_fn(journal) == journal`` (no-op).
    """
    renames: Dict[str, Tuple[str, str]] = {}
    for doc in docs:
        doc_id = doc.get('_id')
        if not isinstance(doc_id, str) or doc_id.startswith('_design/'):
            continue
        journal = doc.get('journal')
        if not isinstance(journal, str) or not journal.strip():
            continue
        new = mapping_fn(journal)
        if not isinstance(new, str) or new == journal:
            continue
        renames[doc_id] = (journal, new)
    return renames


# ---------------------------------------------------------------------------
# CouchDB iteration / write (network-touching)
# ---------------------------------------------------------------------------


def _iter_db(db: Any, verbosity: int = 1) -> Iterable[Dict[str, Any]]:
    """Yield every non-design doc from ``db``.  Wraps the
    document-load in try/except so a single corrupted doc doesn't
    abort the whole walk."""
    for doc_id in db:
        if doc_id.startswith('_design/'):
            continue
        try:
            yield db[doc_id]
        except Exception as exc:
            if verbosity >= 1:
                print(f'  warning: could not load {doc_id}: {exc}',
                      file=sys.stderr)


def apply_renames(
    db: Any,
    mapping_fn: Callable[[str], str],
    dry_run: bool = False,
    limit: Optional[int] = None,
    verbosity: int = 1,
) -> Dict[str, int]:
    """Walk ``db`` and apply ``mapping_fn`` to each doc's ``journal``
    field.  Writes to CouchDB only when ``dry_run`` is False.

    Returns a counts dict: ``{scanned, eligible, written}``.
    """
    counts = {'scanned': 0, 'eligible': 0, 'written': 0}
    for doc in _iter_db(db, verbosity=verbosity):
        counts['scanned'] += 1
        if (verbosity >= 2
                and counts['scanned'] % 1000 == 0):
            print(f'  scanned {counts["scanned"]} docs...')
        renames = compute_renames([doc], mapping_fn)
        if not renames:
            continue
        counts['eligible'] += 1
        if limit is not None and counts['written'] >= limit:
            if verbosity >= 1:
                print(f'  stop: hit --limit {limit}')
            break
        _doc_id, (old, new) = next(iter(renames.items()))
        if verbosity >= 1:
            tag = '(DRY RUN) ' if dry_run else ''
            print(f'  {tag}{doc["_id"]}: {old!r} → {new!r}')
        if not dry_run:
            doc['journal'] = new
            db.save(doc)
        counts['written'] += 1
    return counts


def main() -> int:
    """CLI entry point — phase-2 default mapping is
    ``strip_publisher_suffix`` (drop ``" (PMC)"`` /
    ``" (Taylor & Francis)"`` / ``" (Internet Archive)"``)."""
    parser = argparse.ArgumentParser(
        description='Rewrite compound journal names in skol_dev '
                    'to their canonical form.',
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview updates without writing.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Stop after N successful renames.')
    parser.add_argument('--verbosity', type=int, default=None,
                        help='0=quiet, 1=normal, 2=verbose.')
    args, _unknown = parser.parse_known_args()

    # Lazy imports so the unit tests don't need couchdb / env_config.
    from env_config import get_env_config
    from ingestors.publications import strip_publisher_suffix
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
        print(f'Mapping: strip_publisher_suffix '
              f'(drop "(PMC)" / "(Taylor & Francis)" / '
              f'"(Internet Archive)")')

    counts = apply_renames(
        db=server[db_name],
        mapping_fn=strip_publisher_suffix,
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
