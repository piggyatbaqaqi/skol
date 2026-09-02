#!/usr/bin/env python3
"""Backfill the `context` field onto existing feature annotations.

`Colony on MEA` carries two facts in one string: the feature and the
medium it was observed on.  `docs/feature_label_non_synonyms.md`
refuses to collapse that family — *"the medium is the entire point of
the observation"* — and names the fix: *"a separate `context` field,
not a longer label"*.  The three write paths set it as of the
context-field commit; this is the one-shot for what already exists.

**Additive, and that is deliberate.**  `feature_label` is identity —
it keys both the candidate doc and the hand doc as
``<treatment_id>:<feature_label>:<start>`` — so this script never
touches it.  Every backfilled doc keeps its `_id`, nothing re-keys,
and no existing brat export starts diffing as delete+add.  Making the
label bare is a separate migration that has to move the round files
and the exports on disk with it.

Because the field is derived from the label, the backfill is
reversible by deletion and re-derivable at any time: it adds no
information that is not already in `feature_label`.  What it buys is
that the medium becomes *queryable*, which the label string never was.

Idempotent: a doc already carrying the context it would be given is
skipped, so re-running is free.
"""

import argparse
import collections
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

import couchdb  # noqa: E402

from env_config import get_env_config  # noqa: E402
from treatments_to_structured.feature_label_rules import (  # noqa: E402
    split_medium_context,
)


def context_for(doc: Dict[str, Any]) -> Optional[str]:
    """The context this doc should carry, or ``None`` for none.

    ``None`` means "leave the doc alone" — a label with no growth
    condition omits the key rather than storing a null, following the
    round-provenance convention in ``brat_ingest``.
    """
    label = doc.get('feature_label')
    if not isinstance(label, str) or not label.strip():
        return None
    _, context = split_medium_context(label)
    return context


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--db', action='append', dest='dbs', metavar='NAME',
        help='database to backfill; repeatable.  Defaults to the '
             'production_v4 candidate and hand DBs.',
    )
    ap.add_argument('--dry-run', action='store_true',
                    help='report what would change and write nothing')
    args = ap.parse_args()

    dbs = args.dbs or [
        'skol_exp_production_v4_02_50_features_candidate',
        'skol_exp_production_v4_02_55_features_hand',
    ]

    cfg = get_env_config()
    server = couchdb.Server(cfg['couchdb_url'])
    server.resource.credentials = (cfg['couchdb_username'],
                                   cfg['couchdb_password'])

    status = 0
    for name in dbs:
        if name not in server:
            print(f'{name}: not found', file=sys.stderr)
            status = 1
            continue
        db = server[name]
        tally: collections.Counter = collections.Counter()
        contexts: collections.Counter = collections.Counter()
        updates: List[Dict[str, Any]] = []
        for row in db.view('_all_docs', include_docs=True):
            doc = row.doc
            if not doc or doc['_id'].startswith('_design/'):
                continue
            tally['examined'] += 1
            context = context_for(doc)
            if context is None:
                tally['no condition'] += 1
                continue
            if doc.get('context') == context:
                tally['already set'] += 1
                continue
            doc['context'] = context
            updates.append(doc)
            contexts[context] += 1
            tally['to set'] += 1

        print(f"{name}: {tally['examined']:,} docs examined")
        for key in ('no condition', 'already set', 'to set'):
            print(f'   {key:<14}{tally[key]:>7,}')
        if contexts:
            print(f'   contexts: {dict(sorted(contexts.items()))}')

        if args.dry_run:
            print('   --dry-run: nothing written.')
            continue
        if updates:
            db.update(updates)
            print(f'   wrote {len(updates):,} docs.')

    return status


if __name__ == '__main__':
    raise SystemExit(main())
