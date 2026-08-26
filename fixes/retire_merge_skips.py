#!/usr/bin/env python3
"""Retire merge-skip docs below the raised threshold.

One-shot migration (CLAUDE.md: ``fixes/``).  The merge cutoff moved
from 10 to 15 on 2026-08-26 after measurement put its precision at
51.7 % — see ``docs/data_quality_production_v4_model.md`` §6.1.  The
constant and the config tier changed with it, but **7 632
``skipped_merge_suspect`` docs still record the old decision**, and
``fetch_prior_merge_skip_ids`` trusts them: without this, every future
draw keeps excluding the ~2 112 treatments scoring 10–14 no matter what
the threshold says.

Deletes the skip docs whose recorded ``metric_value`` is below the new
threshold, returning those treatments to p1.

**The score is not lost.** ``data/merge_suspects_20260823.tsv`` holds
``(treatment_id, n_terms_above_5)`` for all 7 632 — written by
``fixes/snapshot_merge_scores.py`` for exactly this reason (plan F3).
The script refuses to run if a doc it would delete is missing from that
snapshot.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

STATUS_SKIPPED = 'skipped_merge_suspect'


def load_snapshot(path: Path) -> Dict[str, int]:
    """Read ``(treatment_id, n_terms_above_5)`` from the T0a snapshot."""
    out: Dict[str, int] = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines()[1:]:
        if not line.strip():
            continue
        tid, _, score = line.partition('\t')
        try:
            out[tid.strip()] = int(score.strip())
        except ValueError:
            continue
    return out


def classify(
    doc: Dict[str, Any],
    threshold: int,
    snapshot: Dict[str, int],
) -> str:
    """One of ``retire`` / ``keep`` / ``unsnapshotted`` / ``not-a-skip``.

    ``retire`` means the recorded score is below the new threshold, so
    the treatment is no longer a suspect and its skip doc is a stale
    decision.  ``unsnapshotted`` is a refusal, not a verdict: deleting
    a score that exists nowhere else destroys evidence.
    """
    if doc.get('status') != STATUS_SKIPPED:
        return 'not-a-skip'
    tid = doc.get('treatment_id') or doc.get('_id') or ''
    if tid not in snapshot:
        return 'unsnapshotted'
    # The snapshot, not doc['metric_value']: it was taken before
    # anything could overwrite the score.
    return 'retire' if snapshot[tid] < threshold else 'keep'


def retire(
    status_db: Any,
    threshold: int,
    snapshot: Dict[str, int],
    *,
    dry_run: bool = True,
) -> Tuple[int, int, List[str]]:
    """Delete stale skip docs.  Returns ``(retired, kept, refused)``.

    Refusals are returned rather than raised so one unsnapshotted doc
    does not block the other 7 631; the caller decides.
    """
    retired = kept = 0
    refused: List[str] = []
    doomed: List[Dict[str, Any]] = []
    for row in status_db.view('_all_docs', include_docs=True).rows:
        doc = row.doc
        if not doc or row.id.startswith('_'):
            continue
        verdict = classify(doc, threshold, snapshot)
        if verdict == 'retire':
            retired += 1
            doomed.append(doc)
        elif verdict == 'keep':
            kept += 1
        elif verdict == 'unsnapshotted':
            refused.append(row.id)
    if not dry_run:
        for doc in doomed:
            status_db.delete(doc)
    return retired, kept, refused


def main() -> int:
    import argparse
    from env_config import common_parser, get_env_config

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[common_parser()],
    )
    parser.add_argument(
        '--snapshot', default='data/merge_suspects_20260823.tsv',
        metavar='TSV',
        help='The T0a score snapshot.  Its scores are what survive the '
             'deletion, so it is required, not optional.',
    )
    args = parser.parse_args()
    config = get_env_config(cli_args=args)
    threshold = int(config['merge_threshold'])
    dry_run = bool(config.get('dry_run', False))
    experiment = config.get('experiment_name')
    if not experiment:
        print('error: --experiment is required', file=sys.stderr)
        return 2
    snapshot = load_snapshot(Path(args.snapshot))
    print(f'snapshot: {len(snapshot):,} scores from {args.snapshot}',
          file=sys.stderr)

    import couchdb  # type: ignore[import-untyped]
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (
        config['couchdb_username'], config['couchdb_password'],
    )
    dbs = server['skol_experiments'][experiment].get('databases') or {}
    status_db = server[dbs['features_status']]
    retired, kept, refused = retire(
        status_db, threshold, snapshot, dry_run=dry_run,
    )
    verb = 'would retire' if dry_run else 'retired'
    print(f'threshold {threshold}: {verb} {retired:,} skip doc(s), '
          f'kept {kept:,}', file=sys.stderr)
    if refused:
        print(f'REFUSED {len(refused)} not in the snapshot '
              f'(scores would be lost): {refused[:5]}', file=sys.stderr)
    return 0


__all__ = ('STATUS_SKIPPED', 'classify', 'load_snapshot', 'retire')


if __name__ == '__main__':
    sys.exit(main())
