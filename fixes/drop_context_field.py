#!/usr/bin/env python3
"""Drop the retired ``context`` field from annotation documents.

``context`` was added on 2026-09-01 to hold a growth condition, and
named badly: in mycology *context* is the flesh — *pileal context* —
and ``schemas/pileus.json`` already uses ``context_color`` in that
sense, so the field put a second meaning of the word in an adjacent
file.  It also conflated two dimensions, storing ``Asci in culture
MEA`` as the uninterpretable string ``'culture MEA'``.

Both jobs moved to the canonical layer on 2026-09-03, where they are
named — ``medium`` (a list, since ``Colony on OA and PCA`` is one
observation on two media) and ``condition`` — and the write paths
stopped setting ``context``.  This removes what they already wrote.

**Nothing is lost.**  The field was derived from ``feature_label``,
which is untouched, so the same information is re-derivable and in
fact already re-derived in ``features_canonical``.

Idempotent: a document without the field is skipped.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

import couchdb  # noqa: E402

from env_config import get_env_config  # noqa: E402

_FIELD = 'context'


def documents_to_strip(
        docs: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Copies of the documents that carry the field, without it.

    Returns new dicts rather than mutating: the caller writes these
    back, and a document that never had the field is not touched at
    all, which is what makes a re-run free.
    """
    out: List[Dict[str, Any]] = []
    for doc in docs:
        if _FIELD not in doc:
            continue
        stripped = {k: v for k, v in doc.items() if k != _FIELD}
        out.append(stripped)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--db', action='append', dest='dbs', metavar='NAME',
        help='database to strip; repeatable.  Defaults to the '
             'production_v4 candidate and hand DBs.',
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='report what would change and write nothing')
    args = parser.parse_args()

    dbs = args.dbs or [
        'skol_exp_production_v4_02_50_features_candidate',
        'skol_exp_production_v4_02_55_features_hand',
    ]
    config = get_env_config()
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (config['couchdb_username'],
                                   config['couchdb_password'])

    status = 0
    for name in dbs:
        if name not in server:
            print(f'{name}: not found', file=sys.stderr)
            status = 1
            continue
        database = server[name]
        docs = [
            dict(row.doc)
            for row in database.view('_all_docs', include_docs=True)
            if row.doc and not row.id.startswith('_design/')
        ]
        updates = documents_to_strip(docs)
        print(f'{name}: {len(docs):,} docs, {len(updates):,} carry '
              f'{_FIELD!r}')
        if args.dry_run:
            print('    --dry-run: nothing written.')
            continue
        if updates:
            database.update(updates)
            print(f'    stripped {len(updates):,} docs.')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
