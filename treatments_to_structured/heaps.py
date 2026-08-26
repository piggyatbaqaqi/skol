#!/usr/bin/env python3
"""Heaps' Law vocabulary curves, computed defensibly.

`jupyter/heaps_law_analysis.ipynb` computed its curve three ways that
do not survive scrutiny at n=1 000, which is why this logic moved into
a module with tests rather than staying in cells.

**The ordering was an artefact.**  `created_at` is stamped *after*
`client.messages.create` returns (`bin/llm_annotate_features`), and the
run uses `ThreadPoolExecutor(5)` with `as_completed`, so completion
order tracks latency, which tracks output length, which tracks
vocabulary richness.  Label-poor treatments cluster at the head and the
curve reads concave-**up**, over-estimating β — the analysis would then
demand far more sampling than it needs.  Noise at n=109; systematic at
n=1 000.

**The curve keyed on timestamp strings.**  Two treatments sharing a
`created_at` had their labels attributed to whichever came first *and*
counted again for the second.  Measured 2026-08-26: 986 treatments,
986 distinct timestamps, zero collisions — so the bug is latent rather
than active, and this module removes it structurally by never keying on
time at all.

**One order is one draw.**  The honest estimator averages over
permutations and shows the spread.

See `docs/plans/annotation-activity-split.md` F1, F2 and T4.
"""

import collections
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def labels_by_treatment(
    annotations: Iterable[Dict[str, object]],
    canonicalizer: Optional[Dict[str, str]] = None,
) -> Dict[str, Set[str]]:
    """Group each treatment's distinct labels.

    ``canonicalizer`` maps drift forms to canonical ones, applied here
    so the raw and canonical curves differ only by this argument.
    """
    out: Dict[str, Set[str]] = collections.defaultdict(set)
    for ann in annotations:
        tid = ann.get('treatment_id')
        label = ann.get('feature_label')
        if not tid or not label:
            continue
        text = str(label)
        if canonicalizer is not None:
            text = canonicalizer.get(text, text)
        out[str(tid)].add(text)
    return dict(out)


def cumulative_curve(
    order: Sequence[str],
    by_treatment: Dict[str, Set[str]],
) -> Tuple[List[int], List[int]]:
    """Cumulative distinct labels against treatments processed.

    ``order`` is an explicit sequence of treatment ids — the draw
    order, or one permutation of it.  **Never timestamps**: keying on
    time is what allowed a shared ``created_at`` to double-count.

    A treatment in ``order`` with no labels still advances x by one; it
    was sampled, and pretending otherwise would compress the curve.
    """
    xs: List[int] = []
    ys: List[int] = []
    seen: Set[str] = set()
    for i, tid in enumerate(order, start=1):
        seen |= by_treatment.get(tid, set())
        xs.append(i)
        ys.append(len(seen))
    return xs, ys


def permutation_band(
    by_treatment: Dict[str, Set[str]],
    n_permutations: int = 200,
    seed: int = 20260826,
    ids: Optional[Sequence[str]] = None,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Average the curve over random orders.

    Returns ``(xs, mean, lo, hi)`` with lo/hi the 2.5th and 97.5th
    percentiles at each x.

    **This is the estimator, and any single ordering is one draw from
    it.**  A curve plotted in one order — temporal, or the draw's own —
    carries an ordering artefact that averaging removes by
    construction.

    ``ids`` is the population to permute.  It defaults to the
    treatments that produced labels, but **pass the whole drawn set**
    when you have it: a treatment that produced nothing was still
    sampled, and omitting it shortens the x-axis and steepens the
    curve.  Round 5 drew 1 000 and 877 produced labels.
    """
    ids = list(ids) if ids is not None else sorted(by_treatment)
    if not ids:
        return [], [], [], []
    rng = random.Random(seed)
    runs: List[List[int]] = []
    for _ in range(max(1, n_permutations)):
        order = list(ids)
        rng.shuffle(order)
        runs.append(cumulative_curve(order, by_treatment)[1])
    xs = list(range(1, len(ids) + 1))
    mean: List[float] = []
    lo: List[float] = []
    hi: List[float] = []
    for i in range(len(ids)):
        col = sorted(run[i] for run in runs)
        mean.append(sum(col) / len(col))
        lo.append(float(col[int(0.025 * (len(col) - 1))]))
        hi.append(float(col[int(0.975 * (len(col) - 1))]))
    return xs, mean, lo, hi


def timestamp_collisions(
    annotations: Iterable[Dict[str, object]],
) -> Dict[str, Set[str]]:
    """Timestamps claimed by more than one treatment.

    Empty is the healthy answer.  Kept as a guard because the old
    curve's correctness silently depended on it, and nothing checked.
    """
    by_ts: Dict[str, Set[str]] = collections.defaultdict(set)
    for ann in annotations:
        tid = ann.get('treatment_id')
        ts = ann.get('created_at')
        if not tid or not ts:
            continue
        by_ts[str(ts)].add(str(tid))
    return {ts: tids for ts, tids in by_ts.items() if len(tids) > 1}


__all__ = (
    'cumulative_curve',
    'labels_by_treatment',
    'permutation_band',
    'timestamp_collisions',
)
