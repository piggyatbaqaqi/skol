#!/usr/bin/env python3
"""Backfill round provenance onto rounds 1-5's docs and sidecars.

One-shot migration (CLAUDE.md: ``fixes/``, not ``bin/``).  T0e made
``bin/llm_annotate_features`` stamp ``round`` / ``round_file`` /
``round_provenance`` onto every candidate and status doc it writes,
but the 1 588 annotations and 117 status docs from rounds 1-5 predate
that and carry nothing.  Until they do, a query against CouchDB alone
still cannot tell round 3 — the only random sample — from the biased
rounds, which is the whole problem T0e exists to close.

**Measured 2026-08-25, and the numbers make this clean:**

* All **110** distinct treatments in ``features_candidate`` appear in a
  round file.  Zero orphans.
* **113** of the **117** attempted status docs are covered; the four
  that are not were ad-hoc runs and correctly stay unstamped.
* The other **7 636** status docs are ``skipped_merge_suspect``,
  written by the *selector* rather than the annotator.  They are
  population-level, not round-level, and are left alone.

Skeleton only — implementation follows test confirmation.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treatments_to_structured.round_provenance import (  # noqa: E402
    PROVENANCE_RECONSTRUCTED,
    RoundIdentity,
    round_identity,
    stamp_round,
)

# Stamped into every bands_derivation note.  A literal, not a call to
# datetime.now(): re-running the backfill must not rewrite the note
# with a new date when nothing about the derivation changed.
_DERIVED_ON = '2026-08-25'

# Status values written by the annotator.  Only these are stamped:
# `skipped_merge_suspect` docs come from bin/select_for_annotation and
# describe a population decision, not an annotation round.
_ANNOTATOR_STATUSES = frozenset({'success', 'partial', 'error'})


def assign_rounds(
    round_files: Dict[str, List[str]],
    rounds_dir: Optional[Path] = None,
) -> Dict[str, RoundIdentity]:
    """Map every treatment id to the round that actually annotated it.

    **The lowest round number wins**, which is the opposite of the live
    path's last-write-wins (round_provenance.stamp_round) — and both
    are right, for reasons that differ:

    * On the live path a re-run *writes*, so the later round is the one
      whose prompt produced the surviving annotations.
    * Here nothing was ever re-written.  ``filter_already_annotated``
      skips ``status='success'`` by default, so the second round file
      to list a treatment did not re-annotate it.

    That is verified rather than assumed: ``taxon_2b793602`` is the
    only id in two round files (1 and 2), and **all 117 attempted
    status docs read ``attempt_count: 1``** — nothing in this corpus
    was annotated twice.

    Args:
        round_files: ``{round_file_stem: [treatment_id, ...]}``.

    Returns:
        ``{treatment_id: RoundIdentity}``, one entry per distinct id.
    """
    base = Path(rounds_dir) if rounds_dir is not None else Path('.')

    def _path(stem: str) -> Path:
        return base / f'{stem}.txt'

    out: Dict[str, RoundIdentity] = {}
    # Sort by (round number, stem) so the result never depends on dict
    # iteration order.  The stem tiebreak keeps `round5` ahead of
    # `round5_manual`, which share a number.
    ordered = sorted(
        round_files.items(),
        key=lambda kv: (round_identity(_path(kv[0])).round, kv[0]),
    )
    for stem, ids in ordered:
        identity = round_identity(_path(stem))
        for tid in ids:
            # First writer wins: the lowest round is the one that
            # annotated, because later rounds skipped it.
            out.setdefault(tid, identity)
    return out


def reconstructed_sidecar(
    round_file: Path,
    treatment_ids: List[str],
    bands: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a backfilled sidecar for a round drawn before T0e.

    Args:
        round_file: The round's ``.txt`` path.
        treatment_ids: Its ids.
        bands: From ``recover_bands``, or None when the band structure
            could not be recovered.  When given, the sidecar records
            the quotas and slices **plus a derivation note** -- a
            derived field that does not say it was derived is
            indistinguishable from a logged one.

    Records **only what is recoverable**.  The selector invocation and
    the seed for rounds 1-4 are gone, and emitting them as nulls would
    dress an irretrievable gap as a recorded fact — so they are absent,
    and ``provenance`` says why.
    """
    identity = round_identity(round_file)
    meta: Dict[str, Any] = {
        'round': identity.round,
        'round_file': identity.round_file,
        'experiment': identity.experiment,
        'provenance': identity.provenance or PROVENANCE_RECONSTRUCTED,
        'n_selected': len(treatment_ids),
    }
    if bands:
        meta['band_quotas'] = list(bands['band_quotas'])
        meta['band_slices'] = list(bands['band_slices'])
        # Band-by-band emission is what made the recovery possible, so
        # recording it is not redundant -- it is the evidence.
        meta['output_order'] = 'band-by-band'
        meta['bands_derivation'] = (
            f"recovered {_DERIVED_ON} from file order plus the "
            f"equal-slice rule: assigning members to bands by the "
            f"population cut points leaves file order band-monotonic "
            f"at k={bands['k']} and at no other k.  Band NAMES are "
            f"not recoverable -- the count sets the cut points and "
            f"the names are decoration -- so only quotas are given."
        )
    return meta


def recover_bands(
    treatment_ids: List[str],
    scores: List[float],
    cut_fn: Any,
) -> Optional[Dict[str, Any]]:
    """Recover a round's band structure from its file order, or None.

    ``select_treatments`` emits **band-by-band**, so a banded round's
    file is band-monotonic in score: every band-0 member precedes every
    band-1 member.  Assign each member to a band by the population's
    equal-slice cut points, then ask whether file order respects that
    assignment.  Exactly one ``k`` passing is a recovery; zero or
    several is not.

    **The obvious weaker test is vacuous.**  "Do the cut points fall in
    gaps between the sorted sample scores" passes for round 3 -- a
    known-random round -- at k=2, 3 AND 4, because every value between
    two adjacent samples lies in some gap by construction.  Order is
    the only real signal, which is why rounds 1-3 are unrecoverable:
    their files are sorted by treatment id.

    Args:
        treatment_ids: In file order.
        scores: Complexity scores, in the same order.
        cut_fn: ``k -> [cut_1, ..., cut_{k-1}]`` over the population.

    Returns:
        ``{k, band_quotas, band_slices}`` or None.
    """
    if len(treatment_ids) != len(scores) or len(scores) < 2:
        return None
    hits = []
    for k in range(2, 6):
        cut_points = list(cut_fn(k))
        if len(cut_points) != k - 1:
            continue
        bands = [
            sum(1 for cv in cut_points if score >= cv)
            for score in scores
        ]
        if len(set(bands)) != k:
            continue
        if any(bands[i] > bands[i + 1] for i in range(len(bands) - 1)):
            continue
        slices = []
        for j in range(k):
            member = [s for s, b in zip(scores, bands) if b == j]
            row: Dict[str, Any] = {
                'quota': len(member),
                'observed_min': min(member),
                'observed_max': max(member),
            }
            if j:
                row['cut_min'] = cut_points[j - 1]
            if j < k - 1:
                row['cut_max'] = cut_points[j]
            slices.append(row)
        hits.append({
            'k': k,
            'band_quotas': [bands.count(j) for j in range(k)],
            'band_slices': slices,
        })
    # Exactly one k is a recovery.  Several means the evidence does not
    # single out a draw design, and picking the smallest would put a
    # guess into the provenance record.
    return hits[0] if len(hits) == 1 else None


def stamp_docs(
    db: Any,
    assignments: Dict[str, RoundIdentity],
    *,
    id_to_treatment: Any,
    dry_run: bool = True,
    statuses: Optional[frozenset] = None,
) -> Tuple[int, int]:
    """Stamp round fields onto a candidate or status database.

    Args:
        db: An open CouchDB database.
        assignments: From ``assign_rounds``.
        id_to_treatment: Callable mapping a doc ``_id`` to its
            treatment id — differs between the two databases
            (``<tid>:<label>:<start>`` vs the bare tid).
        dry_run: When True, count without writing.
        statuses: When given, only stamp docs whose ``status`` is in
            this set.  Used to leave the 7 636
            ``skipped_merge_suspect`` docs alone.

    Returns:
        ``(stamped, skipped)``.
    """
    stamped = skipped = 0
    for row in db.view('_all_docs', include_docs=True).rows:
        doc_id = row.id
        if doc_id.startswith('_'):
            continue
        doc = dict(row.doc or {})
        identity = assignments.get(id_to_treatment(doc_id))
        if identity is None:
            skipped += 1
            continue
        if statuses is not None and doc.get('status') not in statuses:
            skipped += 1
            continue
        # Compare every field stamp_round writes.  Comparing only
        # round and round_file would call a doc already-correct while
        # its provenance was still missing.
        if (doc.get('round') == identity.round
                and doc.get('round_file') == identity.round_file
                and doc.get('round_provenance') == identity.provenance):
            # Already correct.  Re-saving would burn a revision to
            # change nothing, and would hide a real disagreement
            # behind an overwrite.
            skipped += 1
            continue
        stamped += 1
        if dry_run:
            continue
        doc['_id'] = doc_id
        stamp_round(doc, identity)
        db.save(doc)
    return stamped, skipped


__all__ = (
    'assign_rounds',
    'recover_bands',
    'reconstructed_sidecar',
    'stamp_docs',
)


def main() -> int:
    """Backfill sidecars and round stamps for rounds 1-5.

    Reads ``data/annotation_rounds/*.txt``, recovers round 4's bands,
    writes a ``.meta.json`` beside each round file, and stamps the
    candidate and status databases.  ``--dry-run`` (the default from
    ``common_parser``) rehearses without writing.
    """
    import argparse
    import json as _json

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))
    from env_config import common_parser, get_env_config  # noqa: E402
    from treatments_to_structured.complexity import (  # noqa: E402
        complexity_score,
    )
    from treatments_to_structured.merge_metric import (  # noqa: E402
        treatment_merge_metric,
    )

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[common_parser()],
    )
    parser.add_argument(
        '--rounds-dir', default='data/annotation_rounds', metavar='DIR',
    )
    parser.add_argument(
        '--merge-threshold', type=int, default=10, metavar='N',
        help='Merge-suspect cutoff defining the eligible population.',
    )
    args = parser.parse_args()
    config = get_env_config(cli_args=args)
    dry_run = bool(config.get('dry_run', False))
    experiment = config.get('experiment_name')
    if not experiment:
        print('error: --experiment is required', file=sys.stderr)
        return 2

    rounds_dir = Path(args.rounds_dir)
    round_files = {
        p.stem: [ln.strip() for ln in
                 p.read_text(encoding='utf-8').splitlines() if ln.strip()]
        for p in sorted(rounds_dir.glob(f'{experiment}_round*.txt'))
    }
    if not round_files:
        print(f'error: no round files in {rounds_dir}', file=sys.stderr)
        return 2

    import couchdb  # type: ignore[import-untyped]
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (
        config['couchdb_username'], config['couchdb_password'],
    )
    exp_doc = server['skol_experiments'][experiment]
    dbs = exp_doc.get('databases') or {}
    treatments = server[dbs['treatments_prose']]

    # The eligible population, for the equal-slice cut points.
    print('Scoring the eligible population...', file=sys.stderr)
    pop = []
    for row in treatments.view('_all_docs', include_docs=True):
        if row.id.startswith('_'):
            continue
        score = complexity_score(row.doc)
        if score > 0 and treatment_merge_metric(row.doc) < args.merge_threshold:
            pop.append(score)
    pop.sort()
    total = len(pop)
    print(f'  {total} eligible treatments', file=sys.stderr)

    def cut_fn(k: int) -> List[float]:
        return [pop[(i * total) // k] for i in range(1, k)]

    # Sidecars
    for stem, ids in round_files.items():
        path = rounds_dir / f'{stem}.txt'
        try:
            scores = [complexity_score(treatments[t]) for t in ids]
        except Exception:
            scores = []
        bands = (recover_bands(ids, scores, cut_fn)
                 if len(scores) == len(ids) else None)
        meta = reconstructed_sidecar(path, ids, bands=bands)
        note = (f"bands k={bands['k']} quotas {bands['band_quotas']}"
                if bands else 'bands not recoverable')
        print(f"  {stem}: n={len(ids)} {note}", file=sys.stderr)
        if not dry_run:
            path.with_suffix('.meta.json').write_text(
                _json.dumps(meta, indent=2) + '\n', encoding='utf-8',
            )

    # Document stamps
    assignments = assign_rounds(round_files, rounds_dir=rounds_dir)
    print(f'\n{len(assignments)} treatments assigned to a round',
          file=sys.stderr)
    for name, key, statuses in (
        ('candidate', dbs['features_candidate'],
         None),
        ('status', dbs['features_status'], _ANNOTATOR_STATUSES),
    ):
        stamped, skipped = stamp_docs(
            server[key], assignments,
            id_to_treatment=(lambda i: i.split(':', 1)[0]),
            dry_run=dry_run, statuses=statuses,
        )
        verb = 'would stamp' if dry_run else 'stamped'
        print(f'  {name:<10} {verb} {stamped}, left {skipped} alone',
              file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
