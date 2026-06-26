"""Score-banded selection of Treatments for the Phase 1 bootstrap pass.

Pure logic — no CouchDB, no I/O.  The companion CLI
``bin/select_for_annotation.py`` wires this up to a real
treatments_prose database.

See docs/schema_constrained_pipeline.md §10.4 deliverable 2.
"""

import random
from typing import List, Tuple


def parse_band_spec(spec: str) -> List[Tuple[str, int]]:
    """Parse a comma-separated band spec like ``"low:25,mid:50,high:25"``.

    Returns ``[(band_name, quota), ...]`` in declaration order.
    Whitespace around tokens is tolerated.

    Raises:
        ValueError: on empty input, missing ``:<count>``, non-integer
            count, non-positive count, or empty band name.
    """
    if not spec.strip():
        raise ValueError("band spec is empty")
    result: List[Tuple[str, int]] = []
    for raw_part in spec.split(','):
        part = raw_part.strip()
        if ':' not in part:
            raise ValueError(
                f"missing ':<count>' in band spec part: {raw_part!r}"
            )
        name, _, count_str = part.partition(':')
        name = name.strip()
        count_str = count_str.strip()
        if not name:
            raise ValueError(f"empty band name in: {raw_part!r}")
        try:
            count = int(count_str)
        except ValueError as exc:
            raise ValueError(
                f"non-integer count in: {raw_part!r}"
            ) from exc
        if count <= 0:
            raise ValueError(
                f"band count must be positive, got {count} in: "
                f"{raw_part!r}"
            )
        result.append((name, count))
    return result


def select_treatments(
    scored: List[Tuple[str, float]],
    band_specs: List[Tuple[str, int]],
    rng: random.Random,
) -> List[str]:
    """Sample treatment IDs from score-banded populations.

    Algorithm:
    1. Sort ``scored`` ascending by score (ties keep insertion order).
    2. Partition the sorted list into ``len(band_specs)`` equal-size
       slices (last slice absorbs any leftovers if population doesn't
       divide evenly).
    3. For each ``(band_name, quota)`` in ``band_specs`` order, take
       a random ``rng.sample`` of size ``quota`` from that band's slice.

    Output order is band-by-band in ``band_specs`` declaration order;
    within a band, the order is whatever ``rng.sample`` produces.

    Args:
        scored: ``(treatment_id, complexity_score)`` tuples.
        band_specs: ``(band_name, quota)`` tuples; total quota must
            be ≤ population size.
        rng: random source — pass a seeded ``random.Random`` for
            reproducibility.

    Returns:
        Selected treatment IDs, length == sum(quotas).

    Raises:
        ValueError: on empty ``scored``, empty ``band_specs``, total
            quota > population, or a single band's quota > its slice
            population.
    """
    if not scored:
        raise ValueError("scored population is empty")
    if not band_specs:
        raise ValueError("band_specs is empty")

    total_quota = sum(count for _, count in band_specs)
    if total_quota > len(scored):
        raise ValueError(
            f"total quota {total_quota} exceeds scored population "
            f"{len(scored)}"
        )

    n_bands = len(band_specs)
    sorted_scored = sorted(scored, key=lambda pair: pair[1])
    population = len(sorted_scored)
    slices: List[List[Tuple[str, float]]] = []
    for i in range(n_bands):
        start = (i * population) // n_bands
        end = ((i + 1) * population) // n_bands
        slices.append(sorted_scored[start:end])

    selected: List[str] = []
    for (band_name, quota), slice_pop in zip(band_specs, slices):
        if quota > len(slice_pop):
            raise ValueError(
                f"band {band_name!r} quota {quota} exceeds slice "
                f"population {len(slice_pop)}"
            )
        ids = [doc_id for doc_id, _score in slice_pop]
        sampled = rng.sample(ids, quota)
        selected.extend(sampled)
    return selected
