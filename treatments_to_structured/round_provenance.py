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
"""

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    stem = Path(path).stem
    match = re.match(_ROUND_FILE_RE, stem)
    if match is None:
        raise RoundProvenanceError(
            f"cannot derive a round number from {stem!r}; expected "
            f"<experiment>_round<N>[_manual].  Refusing to stamp "
            f"documents with a guessed round."
        )
    identity = RoundIdentity(
        round=int(match.group('n')),
        round_file=stem,
        experiment=match.group('experiment'),
        # The `_manual` suffix is itself the evidence: those files are
        # written by hand and will never carry a selector sidecar.
        provenance=(
            PROVENANCE_MANUAL if match.group('suffix') else None
        ),
    )
    return _apply_sidecar(identity, Path(path))


def _apply_sidecar(
    identity: RoundIdentity, path: Path,
) -> RoundIdentity:
    """Fold a sibling ``.meta.json`` into an identity, if one exists.

    Absent is fine — rounds 1-4 predate the sidecar entirely and must
    stay stampable.  Unreadable is *not* fine: a truncated sidecar
    means something went wrong upstream, and continuing would stamp
    documents with an identity nobody checked.
    """
    meta_path = path.with_suffix('.meta.json')
    if not meta_path.exists():
        return identity
    try:
        meta: Dict[str, Any] = json.loads(
            meta_path.read_text(encoding='utf-8')
        )
    except (OSError, ValueError) as exc:
        raise RoundProvenanceError(
            f"sidecar {meta_path.name} is unreadable: {exc}"
        ) from exc
    if not isinstance(meta, dict):
        raise RoundProvenanceError(
            f"sidecar {meta_path.name} is not a JSON object"
        )
    declared = meta.get('round')
    # Absence is not disagreement: sidecars written before T0e carry
    # no `round` at all, and those must still enrich.
    if declared is not None and int(declared) != identity.round:
        raise RoundProvenanceError(
            f"sidecar {meta_path.name} declares round {declared} but "
            f"the file is named for round {identity.round}.  One of "
            f"them was copied; refusing to choose."
        )
    return replace(
        identity,
        provenance=meta.get('provenance') or identity.provenance,
        selection=meta.get('selection') or identity.selection,
    )


def read_round_file(path: Path) -> Tuple[List[str], RoundIdentity]:
    """Read treatment ids and the round identity from one round file.

    Returns:
        ``(treatment_ids, identity)``.  Blank lines are skipped and
        ids are stripped, matching ``read_treatment_ids``.

    Raises:
        RoundProvenanceError: as ``round_identity``, and if the file
            holds no ids.
        FileNotFoundError: if the round file does not exist.
    """
    path = Path(path)
    identity = round_identity(path)
    ids = [
        line.strip()
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if not ids:
        # Returning [] would let the annotator print "No treatments to
        # process" and exit 0, which reads as success when in fact the
        # draw produced nothing.
        raise RoundProvenanceError(
            f"round file {path.name} holds no treatment ids"
        )
    return ids, identity


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
    if identity is None:
        return doc
    # Deliberately NOT the `_id`.  annotation_doc_id is
    # <tid>:<label>:<start>, and a round in the key would make a
    # re-run create a second doc at the same offset instead of
    # replacing the first.
    doc['round'] = identity.round
    doc['round_file'] = identity.round_file
    if identity.provenance is not None:
        doc['round_provenance'] = identity.provenance
    return doc


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
