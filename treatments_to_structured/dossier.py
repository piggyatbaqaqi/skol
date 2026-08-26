#!/usr/bin/env python3
"""Assemble the diagnostic context a reviewer cannot see in brat.

Every pathology diagnosed during round-4 review came from data absent
from the brat surface: the layout label each span carried, the
paragraph numbers, **the blocks that sat between consecutive spans**,
`merge_metric`, the triage flags, the source document's identity.  The
brat `.txt` shows `=== description ===` and prose; the signal is all
somewhere else, and the human is asked to infer it.  That asymmetry is
why pathology findings cost ~20 minutes each.

This module is the data half of `bin/treatment_dossier` (plan T3e).
It is deliberately **read-only and side-effect free** — it renders
nothing and writes nothing, so the same assembly backs the HTML page,
T3a's merge-suspect table and T3d's cross-tab rather than each getting
its own throwaway script.

**Offsets are into `article.txt.ann`, not `article.txt`** — see memo
§16.  Every function here works in that space.

Skeleton only — the implementation lands after the tests in
``dossier_test.py`` are confirmed (CLAUDE.md TDD).
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestors.yedda_tags import (  # noqa: E402
    YeddaBlock,
    parse_yedda_blocks,
)

# The block index comes from ingestors.yedda_tags, which owns the
# YEDDA format in both directions.  Re-exported here so callers of the
# dossier need only one import; NOT re-implemented -- the regex used
# to be copied into every caller, which is how a reader drifts from
# its writer.
Block = YeddaBlock


@dataclass(frozen=True)
class SpanRef:
    """One entry of a treatment's ``*_spans``, tagged with its field."""

    field: str
    start: int
    end: int
    paragraph: Optional[int] = None
    head: Optional[str] = None


# Structural furniture: present between almost any two spans and
# never the finding.  Measured 2026-08-25 over 300 treatments -- 94.2 %
# have at least one gap block, median 9 and max 8 356, dominated by
# Page-header and Bibliography.  Excluding these takes the median run
# between consecutive spans from 2 blocks to 1.
FURNITURE = frozenset({
    'Page-header', 'Page-footer', 'Bibliography', 'Index', 'ToC-entry',
})

# Runs longer than this are summarised rather than listed.  60 % of
# non-furniture runs are a single block and 92 % are <= 5, so the cap
# hides almost nothing while bounding the pathological tail.
MAX_GAP_BLOCKS = 5


@dataclass(frozen=True)
class Gap:
    """Unclaimed blocks between two *consecutive* spans.

    Per-pair rather than per-treatment, because the whole-bracket form
    is unusable: a treatment's first and last span can bracket most of
    a book, and the furniture in between drowns the signal.  Every
    pathology this was built to catch -- the `Table` hiding a species
    heading in `taxon_fdbd1b53`, the `Figure-caption` holding a severed
    Notes in `taxon_ecb0124d`, the genus header in `taxon_fd50457a` --
    is a *single block between two adjacent spans*.
    """

    after: SpanRef
    before: SpanRef
    blocks: List[Block] = field(default_factory=list)
    n_furniture: int = 0
    n_omitted: int = 0


@dataclass
class Dossier:
    """Everything known about one treatment, assembled for reading."""

    treatment_id: str
    spans: List[SpanRef] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    gaps: List[Gap] = field(default_factory=list)
    labels: Dict[str, List[str]] = field(default_factory=dict)


def parse_blocks(ann_text: str) -> List[Block]:
    """Index every ``[@…#Label*]`` block in an ``article.txt.ann``.

    Thin alias for ``ingestors.yedda_tags.parse_yedda_blocks``, kept so
    the dossier reads as one module; the behaviour and the offsets are
    that function's.
    """
    return parse_yedda_blocks(ann_text)


def treatment_spans(treatment: Dict[str, Any]) -> List[SpanRef]:
    """Collect every ``*_spans`` entry, tagged with its field name.

    Sorted by start offset, so field boundaries disappear — which is
    the point.  `taxon_ecb0124d`'s severed Notes was only visible
    because a `Figure-caption` block sat between two spans belonging
    to *different* fields.

    Offsets are coerced: some stored spans carry them as strings
    (`taxon_09b97d5f`'s `diagnosis_spans`), and a span missing either
    offset is skipped rather than raising — a dossier is a reading
    aid, and refusing to render a whole treatment over one bad span
    would defeat it.
    """
    raise NotImplementedError


def labels_for_span(blocks: List[Block], span: SpanRef) -> List[str]:
    """The layout labels of every block the span overlaps.

    A list, not a single label: a span that crosses a block boundary
    is itself a finding — it means the extractor joined material the
    layout pass had separated.
    """
    raise NotImplementedError


def gaps(
    blocks: List[Block],
    spans: List[SpanRef],
    *,
    exclude: FrozenSet[str] = FURNITURE,
    max_blocks: int = MAX_GAP_BLOCKS,
) -> List[Gap]:
    """Unclaimed blocks between each consecutive pair of spans.

    **The centrepiece.**  "What sat between these two spans, and what
    label did it carry" is the question that exposed the nine fused
    genera in `taxon_8d815304`, the *Rhodoveronaea* header in
    `taxon_fd50457a`, and the species heading hidden in a `Table` in
    `taxon_fdbd1b53`.

    Bounded by the first and last span -- material outside belongs to
    neighbouring treatments -- and reported **per adjacent pair**, not
    pooled.  Overlapping spans are merged before pairing, so a span
    contained in another cannot manufacture a phantom gap.

    ``exclude`` suppresses structural furniture from the listing while
    still counting it in ``n_furniture``; pass an empty set to see
    everything.  Runs longer than ``max_blocks`` are truncated with the
    remainder in ``n_omitted``, never silently dropped.

    Pairs with nothing between them yield no ``Gap`` at all.
    """
    raise NotImplementedError


def build_dossier(
    treatment: Dict[str, Any],
    ann_text: str,
) -> Dossier:
    """Assemble the whole picture for one treatment."""
    raise NotImplementedError


__all__ = (
    'Block',
    'FURNITURE',
    'Gap',
    'MAX_GAP_BLOCKS',
    'parse_yedda_blocks',
    'Dossier',
    'SpanRef',
    'build_dossier',
    'gaps',
    'labels_for_span',
    'parse_blocks',
    'treatment_spans',
)
