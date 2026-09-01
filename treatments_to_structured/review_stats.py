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
    raise NotImplementedError


def precision_bootstrap(
    counts: Sequence[TreatmentCounts],
    n_resamples: int = 2000,
    seed: int = 20260901,
) -> Optional[Tuple[float, float, float]]:
    raise NotImplementedError


def recall_distribution(
    counts: Sequence[TreatmentCounts],
) -> Dict[str, object]:
    raise NotImplementedError


__all__ = (
    'TreatmentCounts',
    'per_treatment_counts',
    'precision_bootstrap',
    'recall_distribution',
)
