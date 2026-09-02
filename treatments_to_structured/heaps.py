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
import math
import random
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


def labels_by_treatment(
    annotations: Iterable[Dict[str, object]],
    canonicalizer: Optional[
        Union[Mapping[str, str], Callable[[str], str]]
    ] = None,
) -> Dict[str, Set[str]]:
    """Group each treatment's distinct labels.

    ``canonicalizer`` maps drift forms to canonical ones, applied here
    so the raw and canonical curves differ only by this argument.  A
    **Mapping** covers the hand map; a **callable** covers the rules
    in ``feature_label_rules``, which settle whole families that no
    finite map enumerates.
    """
    out: Dict[str, Set[str]] = collections.defaultdict(set)
    for ann in annotations:
        tid = ann.get('treatment_id')
        label = ann.get('feature_label')
        if not tid or not label:
            continue
        text = str(label)
        if callable(canonicalizer):
            text = canonicalizer(text)
        elif canonicalizer is not None:
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


def fit_beta(
    xs: Sequence[int],
    ys: Sequence[float],
    *,
    min_n: int = 200,
) -> Tuple[float, float]:
    """Least-squares ``(K, beta)`` for ``V = K n**beta``, fitted over
    ``n >= min_n``.

    **The window is the point.**  The head of a vocabulary curve is
    not Heaps: at n=1 every label is new, so V tracks n and the slope
    approaches 1.  Fitting from the origin therefore reports a
    steeper exponent than the corpus has -- 0.645 against 0.601 on
    round 5 -- in the direction that demands more sampling than is
    needed.  ``min_n`` says where the power law is believed to hold.

    Points with ``n < min_n`` or a non-positive ``V`` are dropped;
    fewer than two survivors is a ``ValueError`` rather than a
    silently meaningless fit.
    """
    points = [
        (math.log(x), math.log(y))
        for x, y in zip(xs, ys)
        if x >= max(1, min_n) and x > 0 and y > 0
    ]
    if len(points) < 2:
        raise ValueError(
            f'need at least 2 points with n >= {min_n} and V > 0; '
            f'got {len(points)}'
        )
    n = len(points)
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    sum_xx = sum(p[0] * p[0] for p in points)
    sum_xy = sum(p[0] * p[1] for p in points)
    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0:
        raise ValueError('degenerate fit: every point shares one n')
    beta = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - beta * sum_x) / n
    return math.exp(intercept), beta


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
