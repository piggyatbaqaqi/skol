#!/usr/bin/env python3
"""Precision and recall for an annotation round, reported honestly.

The two metrics need different treatment, and the difference is not
stylistic.  Measured over the 85 treatments reviewed by 2026-08-23:

    metric      pooled   naive +/-   clustered +/-   design effect
    precision   98.48 %     0.90 pp        1.14 pp            1.6x
    recall      83.13 %     2.42 pp       15.11 pp           38.8x

**Precision** is well behaved: report it pooled with a
treatment-level bootstrap interval.

**Recall is not estimable at these sample sizes.**  A +/-15 pp interval
spans 68-98 % and supports no conclusion.  The cause is the shape of the
data rather than the design: most treatments add nothing, one adds many
-- round 1's 36.3 % recall was a single document contributing 136 of 263
additions.  So this module **does not offer a recall interval at all**;
it reports the distribution (median additions, the fraction of
treatments needing any, top-k concentration) and the raw counts.

See ``docs/plans/annotation-activity-split.md`` T5.
"""

import random
import statistics
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

_VALID_ACTIONS = ('kept', 'added')


class TreatmentCounts(NamedTuple):
    """One reviewed treatment's tallies.

    ``candidates`` is what the model proposed; ``kept`` is how many of
    those the reviewer retained; ``added`` is annotations the reviewer
    supplied that the model missed.
    """

    treatment_id: str
    kept: int
    added: int
    candidates: int


def per_treatment_counts(
    hand: Iterable[Dict[str, object]],
    candidate: Iterable[Dict[str, object]],
    treatment_ids: Sequence[str],
) -> List[TreatmentCounts]:
    """Tally each reviewed treatment.

    ``treatment_ids`` is the **reviewed set**, and every one of them
    appears in the result even if it produced nothing: a treatment that
    was read and needed no change is evidence, and dropping it would
    remove the zeros that dominate the recall distribution.
    """
    wanted = list(treatment_ids)
    keep: Dict[str, int] = {t: 0 for t in wanted}
    add: Dict[str, int] = {t: 0 for t in wanted}
    cand: Dict[str, int] = {t: 0 for t in wanted}
    for ann in hand:
        tid = ann.get('treatment_id')
        action = ann.get('reviewer_action')
        if action not in _VALID_ACTIONS:
            raise ValueError(
                f'unrecognised reviewer_action {action!r}; expected one '
                f'of {_VALID_ACTIONS}'
            )
        if tid in keep:
            (keep if action == 'kept' else add)[str(tid)] += 1
    for ann in candidate:
        tid = ann.get('treatment_id')
        if tid in cand:
            cand[str(tid)] += 1
    return [
        TreatmentCounts(treatment_id=t, kept=keep[t], added=add[t],
                        candidates=cand[t])
        for t in wanted
    ]


def precision_bootstrap(
    counts: Sequence[TreatmentCounts],
    n_resamples: int = 2000,
    seed: int = 20260901,
) -> Optional[Tuple[float, float, float]]:
    """Pooled precision with a **treatment-level** bootstrap interval.

    Returns ``(point, lo, hi)`` at the 2.5th and 97.5th percentiles, or
    ``None`` when no treatment carried a candidate.

    **The resampling unit is the treatment, not the annotation.**  An
    annotation-level interval treats a treatment's annotations as
    independent, which they are not, and comes out roughly six times too
    narrow.
    """
    # Sorted so the statistic does not depend on the order the caller
    # happened to supply.  The resampler indexes into this sequence, so
    # an unsorted pool gives different intervals for identical data —
    # see `test_is_independent_of_input_order`.
    pool = sorted((c for c in counts if c.candidates > 0),
                  key=lambda c: c.treatment_id)
    if not pool:
        return None
    total_c = sum(c.candidates for c in pool)
    point = sum(c.kept for c in pool) / total_c
    rng = random.Random(seed)
    draws: List[float] = []
    n = len(pool)
    for _ in range(max(1, n_resamples)):
        sample = [pool[rng.randrange(n)] for _ in range(n)]
        denom = sum(c.candidates for c in sample)
        if denom:
            draws.append(sum(c.kept for c in sample) / denom)
    if not draws:
        return point, point, point
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return point, lo, hi


def recall_distribution(
    counts: Sequence[TreatmentCounts],
) -> Dict[str, object]:
    """Recall as a **distribution**, deliberately without an interval.

    The pooled ratio is returned as ``pooled_recall`` alongside its raw
    numerator and denominator so it can be quoted with its own scale,
    but **no confidence interval is offered for it** -- at these sample
    sizes one would span 68-98 % and mislead.  What is robust is the
    shape: how many treatments needed nothing, and how concentrated the
    additions are.
    """
    adds = sorted((c.added for c in counts), reverse=True)
    total_add = sum(adds)
    kept = sum(c.kept for c in counts)
    denom = kept + total_add
    return {
        'treatments': len(counts),
        'kept': kept,
        'added': total_add,
        'pooled_recall': (kept / denom) if denom else None,
        'median_additions': statistics.median(adds) if adds else 0,
        'fraction_needing_additions': (
            sum(1 for a in adds if a) / len(adds) if adds else 0.0
        ),
        'top_1_share': (adds[0] / total_add) if total_add else 0.0,
        'top_5_share': (sum(adds[:5]) / total_add) if total_add else 0.0,
        'max_additions': adds[0] if adds else 0,
    }


__all__ = (
    'TreatmentCounts',
    'per_treatment_counts',
    'precision_bootstrap',
    'recall_distribution',
)
