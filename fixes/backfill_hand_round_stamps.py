#!/usr/bin/env python3
"""Backfill `round` provenance onto `features_hand` annotations.

T0e stamped `round`, `round_file` and `round_provenance` onto candidate
and status docs, but not onto the hand (reviewed) side.  Measured
2026-09-01: **all 2 244 hand docs carried no round field**, so the T5
statistics had to join through the round *file*, and the database-side
round query T0e exists to enable did not work.

`bin/brat_ingest` stamps new hand docs as of the round-stamping commit;
this is the one-shot for what already exists.

**Resolution order**, matching `make_reviewed_doc`:

1. the candidate doc with the same ``_id`` (86.5 % of docs — a `kept`
   annotation is the same annotation, so its round is authoritative);
2. otherwise ``round_fields_for_treatment`` over the treatment's other
   candidates (13.5 % — `added` annotations have no candidate of their
   own but their treatment was drawn in a round).

Measured coverage is 100 %; a doc that resolves by neither route is
reported and left alone rather than guessed at.

Idempotent: a doc already carrying the round it would be given is
skipped, so re-running is free.
"""

import argparse
import collections
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

import couchdb  # noqa: E402

from env_config import get_env_config  # noqa: E402
from treatments_to_structured.brat_ingest import (  # noqa: E402
    round_fields_for_treatment,
)

_ROUND_KEYS = ('round', 'round_file', 'round_provenance')


def resolve(
    doc: Dict[str, Any],
    candidates_by_id: Dict[str, Dict[str, Any]],
    candidates_by_treatment: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Round fields for one hand doc; ``{}`` when unresolvable."""
    exact = candidates_by_id.get(doc['_id'])
    if exact is not None and isinstance(exact.get('round'), int):
        return {k: exact[k] for k in _ROUND_KEYS if exact.get(k) is not None}
    return round_fields_for_treatment(
        candidates_by_treatment.get(doc.get('treatment_id'), [])
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', default='production_v4')
    ap.add_argument('--hand-db',
                    default='skol_exp_production_v4_02_55_features_hand')
    ap.add_argument('--candidate-db',
                    default='skol_exp_production_v4_02_50_features_candidate')
    ap.add_argument('--dry-run', action='store_true',
                    help='report what would change and write nothing')
    args = ap.parse_args()

    cfg = get_env_config()
    server = couchdb.Server(cfg['couchdb_url'])
    server.resource.credentials = (cfg['couchdb_username'],
                                   cfg['couchdb_password'])
    hand_db = server[args.hand_db]
    cand_db = server[args.candidate_db]

    by_id: Dict[str, Dict[str, Any]] = {}
    by_treatment: Dict[str, List[Dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in cand_db.view('_all_docs', include_docs=True):
        doc = row.doc
        if not doc:
            continue
        by_id[doc['_id']] = doc
        tid = doc.get('treatment_id')
        if tid:
            by_treatment[tid].append(doc)

    tally: collections.Counter = collections.Counter()
    rounds: collections.Counter = collections.Counter()
    updates: List[Dict[str, Any]] = []
    for row in hand_db.view('_all_docs', include_docs=True):
        doc = row.doc
        if not doc:
            continue
        tally['examined'] += 1
        fields = resolve(doc, by_id, by_treatment)
        if not fields:
            tally['unresolvable'] += 1
            print(f"  unresolvable: {doc['_id']}", file=sys.stderr)
            continue
        if all(doc.get(k) == v for k, v in fields.items()):
            tally['already stamped'] += 1
            continue
        doc.update(fields)
        updates.append(doc)
        tally['to stamp'] += 1
        rounds[fields.get('round')] += 1

    print(f"{args.hand_db}: {tally['examined']:,} docs examined")
    for key in ('already stamped', 'to stamp', 'unresolvable'):
        print(f"   {key:<18}{tally[key]:>7,}")
    print(f"   rounds: {dict(sorted(rounds.items()))}")

    if args.dry_run:
        print('\n--dry-run: nothing written.')
        return 0
    if not updates:
        print('\nNothing to do.')
        return 0
    for i in range(0, len(updates), 500):
        hand_db.update(updates[i:i + 500])
    print(f"\nStamped {len(updates):,} documents.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
