#!/usr/bin/env python3
"""Round identity for annotation runs — who drew this, and when.

The bootstrap annotation rounds produced a pooled precision/recall
statistic that described *the selection* rather than the corpus, and
the error was invisible from the database side because **no candidate
or status doc records which round it came from**.  Reconstructing it
meant reading round files out of the repository.

This module is the fix: it turns a round file's path into a
``RoundIdentity`` that ``bin/llm_annotate_features`` stamps onto every
``features_candidate`` and ``features_status`` doc it writes, so a
query against CouchDB alone can stratify by round.

See ``docs/plans/annotation-activity-split.md`` T0e.

Skeleton only — the implementation lands after the tests in
``round_provenance_test.py`` are confirmed (CLAUDE.md TDD).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# The round-file naming regex.  DELIBERATELY different from
# ``select_for_annotation.default_output_path``'s: that one must NOT
# match ``_manual`` files, so the selector keeps numbering normally
# (documented in data/annotation_rounds/README.md).  This one MUST
# match them, because a ``_manual`` file belongs to the round named in
# it — it is hand-picked material added to round N, not a round of its
# own.
_ROUND_FILE_RE = r'^(?P<experiment>.+)_round(?P<n>\d+)(?P<suffix>_manual)?$'

PROVENANCE_SELECTOR = 'selector'
PROVENANCE_MANUAL = 'manual'
PROVENANCE_RECONSTRUCTED = 'reconstructed'


@dataclass(frozen=True)
class RoundIdentity:
    """Where a set of treatment ids came from.

    Attributes:
        round: The round number, from the file name.
        round_file: The file's stem, e.g. ``production_v4_round6``.
            Always populated; unambiguous even for ``_manual`` files,
            which share a round number with their parent round.
        experiment: The experiment name parsed from the file name.
        provenance: ``selector`` (drawn by select_for_annotation),
            ``manual`` (hand-picked ``_manual`` file), or
            ``reconstructed`` (sidecar backfilled after the fact).
            None when no sidecar exists and the name gives no clue.
        selection: ``uniform`` / ``stratified``, from the sidecar.
            None when there is no sidecar.
    """

    round: int
    round_file: str
    experiment: str
    provenance: Optional[str] = None
    selection: Optional[str] = None


class RoundProvenanceError(ValueError):
    """A round file's identity cannot be established, or disagrees."""


def round_identity(path: Path) -> RoundIdentity:
    """Derive a ``RoundIdentity`` from a round file's path.

    The **file name is authoritative for the round number**.
    ``default_output_path`` guarantees ``<experiment>_round<N>.txt`` is
    unique and never reused, which makes the name the one identifier
    that cannot drift.  A sibling ``.meta.json`` sidecar enriches the
    identity with provenance and selection; it never overrides the
    number.

    Raises:
        RoundProvenanceError: if the name carries no round number, or
            a sidecar's ``round`` disagrees with the name's.
    """
    raise NotImplementedError


def read_round_file(path: Path) -> Tuple[List[str], RoundIdentity]:
    """Read treatment ids and the round identity from one round file.

    Returns:
        ``(treatment_ids, identity)``.  Blank lines are skipped and
        ids are stripped, matching ``read_treatment_ids``.

    Raises:
        RoundProvenanceError: as ``round_identity``.
        FileNotFoundError: if the round file does not exist.
    """
    raise NotImplementedError


def stamp_round(
    doc: dict,
    identity: Optional[RoundIdentity],
) -> dict:
    """Stamp a candidate-annotation doc with its round, in place.

    Used for ``features_candidate`` docs; ``make_status_doc`` takes the
    identity as a keyword argument instead, since it builds its doc
    from scratch.

    A None identity leaves the doc untouched — the cron path passes no
    round file, and ``round: null`` would misrepresent that as an
    unidentifiable round rather than no round at all.

    Returns the same dict, for chaining.
    """
    raise NotImplementedError


__all__ = (
    'PROVENANCE_MANUAL',
    'PROVENANCE_RECONSTRUCTED',
    'PROVENANCE_SELECTOR',
    'RoundIdentity',
    'RoundProvenanceError',
    'read_round_file',
    'stamp_round',
    'round_identity',
)
