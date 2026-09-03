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
from dataclasses import dataclass
from typing import (
    Callable,
    FrozenSet,
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


@dataclass(frozen=True)
class Coverage:
    """Result of :func:`out_of_sample_coverage`.

    Three numbers, not one, because they answer different questions
    and reporting a single figure invites the reader to assume it is
    whichever one they had in mind:

    * ``type_coverage`` — mean over treatments of the share of a
      treatment's *distinct* labels already known.
    * ``instance_coverage`` — the same mean, weighted within each
      treatment by how often each label occurs.
    * ``pooled_instance_coverage`` — every annotation instance in one
      pool, so a treatment with forty annotations counts forty times.

    On round 6 these read 90.9 %, 91.3 % and 90.6 %.
    """

    treatments: int
    instances: int
    type_coverage: Optional[float]
    instance_coverage: Optional[float]
    pooled_instance_coverage: Optional[float]
    novel_labels: FrozenSet[str]


def instances_by_treatment(
    annotations: Iterable[Dict[str, object]],
    canonicalizer: Optional[
        Union[Mapping[str, str], Callable[[str], str]]
    ] = None,
) -> Dict[str, 'collections.Counter[str]']:
    """Count each treatment's label occurrences.

    The counting sibling of :func:`labels_by_treatment`, which
    collapses repeats.  Both are needed: distinct labels answer "does
    this treatment use vocabulary we have seen", occurrences answer
    "how much of what it says do we already understand".
    """
    out: Dict[str, 'collections.Counter[str]'] = collections.defaultdict(
        collections.Counter,
    )
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
        out[str(tid)][text] += 1
    return dict(out)


def out_of_sample_coverage(
    known: Set[str],
    by_treatment: Dict[str, Set[str]],
    instances: Optional[Dict[str, 'collections.Counter[str]']] = None,
) -> Coverage:
    """How much of a held-out round the ``known`` vocabulary covers.

    **This is the honest form of the coverage question.**  A
    permutation band over a single round is in-sample: it asks what
    that round's treatments look like to a vocabulary built from the
    rest of the same round, drawn from the same population in the same
    draw.  Here ``known`` comes from one round and ``by_treatment``
    from another, so nothing about the answer is circular — which is
    what makes it the right measurement when two rounds were drawn
    from populations that are not identical.

    A treatment that was sampled but produced no labels is **excluded
    from the means**, not scored.  Zero would understate coverage and
    one would overstate it; the treatment simply carries no evidence.
    ``treatments`` reports how many were actually measured.

    ``instances`` is optional.  Without it the instance figures are
    ``None`` rather than a copy of the type figure under another name.
    """
    measured = [t for t, labels in by_treatment.items() if labels]
    if not measured:
        return Coverage(
            treatments=0, instances=0, type_coverage=None,
            instance_coverage=None, pooled_instance_coverage=None,
            novel_labels=frozenset(),
        )

    type_shares: List[float] = []
    instance_shares: List[float] = []
    pooled_known = pooled_total = 0
    for tid in measured:
        labels = by_treatment[tid]
        type_shares.append(
            sum(1 for label in labels if label in known) / len(labels)
        )
        counts = (instances or {}).get(tid)
        if not counts:
            continue
        total = sum(counts.values())
        seen = sum(n for label, n in counts.items() if label in known)
        instance_shares.append(seen / total)
        pooled_known += seen
        pooled_total += total

    novel = frozenset(
        label
        for labels in by_treatment.values()
        for label in labels
        if label not in known
    )
    return Coverage(
        treatments=len(measured),
        instances=pooled_total,
        type_coverage=sum(type_shares) / len(type_shares),
        instance_coverage=(
            sum(instance_shares) / len(instance_shares)
            if instance_shares else None
        ),
        pooled_instance_coverage=(
            pooled_known / pooled_total if pooled_total else None
        ),
        novel_labels=novel,
    )


def prequential_curve(
    raw_by_treatment: Mapping[str, Sequence[str]],
    order: Sequence[str],
    transform: Callable[
        [str, Mapping[str, str], Mapping[str, str]], Sequence[str]],
    *,
    min_df: int = 5,
) -> Tuple[List[int], List[float]]:
    """Cumulative canonical vocabulary, canonicalized **causally**.

    ``transform(label, known, established)`` returns the canonical
    label or labels for one raw label.  Both indices map a lower-cased
    label to its canonical form and are rebuilt as the walk proceeds
    from the treatments **already seen** — so the point at position
    *n* is what an operator would have had after *n* treatments, not
    what hindsight would give them.

    Contrast :func:`permutation_band` over pre-canonicalized labels,
    where the canonicalizer saw the whole corpus: that curve is fine
    to describe and unsafe to extrapolate, because its early points
    know their own future.

    At ``n=1`` both indices are empty, so no rule can fire.  That is
    the honest answer, not a defect: a vocabulary of one treatment
    supports no consolidation.
    """
    seen_vocabulary: Dict[str, str] = {}
    treatment_frequency: 'collections.Counter[str]' = collections.Counter()
    accumulated: Set[str] = set()
    xs: List[int] = []
    ys: List[float] = []

    for position, treatment_id in enumerate(order, start=1):
        established = {
            key: label
            for key, label in seen_vocabulary.items()
            if treatment_frequency[label] >= min_df
        }
        produced: Set[str] = set()
        for raw in raw_by_treatment.get(treatment_id, ()):
            for canonical in transform(raw, seen_vocabulary, established):
                produced.add(canonical)

        accumulated |= produced
        # The indices grow only after this treatment has been scored.
        for label in produced:
            seen_vocabulary.setdefault(label.lower(), label)
            treatment_frequency[label] += 1

        xs.append(position)
        ys.append(len(accumulated))
    return xs, ys


def prequential_band(
    raw_by_treatment: Mapping[str, Sequence[str]],
    ids: Sequence[str],
    transform: Callable[
        [str, Mapping[str, str], Mapping[str, str]], Sequence[str]],
    *,
    n_permutations: int = 200,
    seed: int = 20260903,
    min_df: int = 5,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Average :func:`prequential_curve` over random orders.

    Returns ``(xs, mean, lo, hi)`` with lo/hi the 2.5th and 97.5th
    percentiles, matching :func:`permutation_band`'s shape.  One
    ordering is one draw from the estimator; averaging is what removes
    the ordering artefact, and here it does double duty, since the
    causal index makes a single order's answer genuinely
    order-dependent.
    """
    population = list(ids)
    if not population:
        return [], [], [], []
    rng = random.Random(seed)
    runs: List[List[float]] = []
    for _ in range(max(1, n_permutations)):
        shuffled = list(population)
        rng.shuffle(shuffled)
        runs.append(prequential_curve(
            raw_by_treatment, shuffled, transform, min_df=min_df)[1])

    xs = list(range(1, len(population) + 1))
    mean: List[float] = []
    lo: List[float] = []
    hi: List[float] = []
    for index in range(len(population)):
        column = sorted(run[index] for run in runs)
        mean.append(sum(column) / len(column))
        lo.append(float(column[int(0.025 * (len(column) - 1))]))
        hi.append(float(column[int(0.975 * (len(column) - 1))]))
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
