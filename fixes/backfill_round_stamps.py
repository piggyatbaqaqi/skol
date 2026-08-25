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
    RoundIdentity,
)

# Status values written by the annotator.  Only these are stamped:
# `skipped_merge_suspect` docs come from bin/select_for_annotation and
# describe a population decision, not an annotation round.
_ANNOTATOR_STATUSES = frozenset({'success', 'partial', 'error'})


def assign_rounds(
    round_files: Dict[str, List[str]],
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


__all__ = (
    'assign_rounds',
    'recover_bands',
    'reconstructed_sidecar',
    'stamp_docs',
)
