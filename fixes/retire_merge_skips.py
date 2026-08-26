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

Skeleton only — implementation follows test confirmation.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

STATUS_SKIPPED = 'skipped_merge_suspect'


def load_snapshot(path: Path) -> Dict[str, int]:
    """Read ``(treatment_id, n_terms_above_5)`` from the T0a snapshot."""
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


__all__ = ('STATUS_SKIPPED', 'classify', 'load_snapshot', 'retire')
