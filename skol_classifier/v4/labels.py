"""v4 Pass-1 label projection + YEDDA-block → line-index alignment.

Three helpers used by ``bin/train_crf_layout.py`` (and later Step 5's
predictor when it needs to map predictions back to YEDDA tags):

* :func:`map_yedda_to_layout` — project the 19 ACTIVE_TAGS_19 down to
  the 8 Pass-1 labels (7 layout tags pass through; everything else
  collapses to ``Other``).
* :func:`yedda_blocks_to_line_indices` — turn
  ``parse_yedda_sections`` output (char offsets) into
  ``(label, [line_idx, ...])`` tuples.
* :func:`build_label_sequence` — end-to-end per-doc label-index
  sequence ready to feed to the CRF.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ingestors.yedda_tags import ACTIVE_TAGS_19
from skol_classifier.v4.crf_layout import (
    LABEL_TO_INDEX,
    OTHER_INDEX,
)


# The 7 YEDDA tags that map to themselves in the Pass-1 label space.
LAYOUT_YEDDA_TAGS: Tuple[str, ...] = (
    'Page-header',
    'Figure-caption',
    'Table',
    'Key',
    'Bibliography',
    'Index',
    'ToC-entry',
)


# The 12 YEDDA tags that map to themselves in the Pass-2 label space.
# Anything not in this set (including the 7 layout tags) folds to
# 'Misc-exposition' — the catch-all the v4 plan §Label-space
# partition specifies for the treatment vocab.
TREATMENT_YEDDA_TAGS: Tuple[str, ...] = (
    'Nomenclature',
    'Description',
    'Diagnosis',
    'Etymology',
    'Materials-examined',
    'Materials-and-methods',
    'Type-designation',
    'Biology',
    'Phylogeny',
    'New-combinations',
    'Notes',
    'Misc-exposition',
)


# The 19 YEDDA tags spanning Pass 1 ∪ Pass 2 — the label space the
# Step 6.F single-CRF baseline operates on directly.  Order matches
# ``ingestors/yedda_tags.ACTIVE_TAGS_19`` (the Tag-enum declaration
# order), so the index identities stay stable across builds.
ACTIVE_YEDDA_TAGS: Tuple[str, ...] = tuple(t.value for t in ACTIVE_TAGS_19)


# Lowercase-keyed lookups so case drift in YEDDA files doesn't drop
# blocks to the catch-all.
_LAYOUT_BY_LOWER = {tag.lower(): tag for tag in LAYOUT_YEDDA_TAGS}
_TREATMENT_BY_LOWER = {tag.lower(): tag for tag in TREATMENT_YEDDA_TAGS}
_ACTIVE_BY_LOWER = {tag.lower(): tag for tag in ACTIVE_YEDDA_TAGS}


def map_yedda_to_layout(yedda_tag: str) -> str:
    """Project a YEDDA tag to a Pass-1 label.

    Tags in :data:`LAYOUT_YEDDA_TAGS` pass through (with canonical
    capitalization); everything else returns ``'Other'``.
    Case-insensitive on the input.
    """
    return _LAYOUT_BY_LOWER.get(yedda_tag.lower(), 'Other')


def map_yedda_to_treatment(yedda_tag: str) -> str:
    """Project a YEDDA tag to a Pass-2 treatment label.

    Tags in :data:`TREATMENT_YEDDA_TAGS` pass through (with canonical
    capitalization); the 7 layout tags + any unknown tag collapse to
    ``'Misc-exposition'``.  Case-insensitive on the input.
    """
    return _TREATMENT_BY_LOWER.get(yedda_tag.lower(), 'Misc-exposition')


def map_yedda_to_active(yedda_tag: str) -> str:
    """Project a YEDDA tag to the 19-label ACTIVE_TAGS_19 vocab.

    Members of :data:`ACTIVE_YEDDA_TAGS` pass through (case-
    insensitive); deprecated tags (Holotype / Distribution / FIX) and
    any unknown tag collapse to ``'Misc-exposition'`` — same catch-all
    Pass-2 uses, which keeps round-tripping intuitive.
    """
    return _ACTIVE_BY_LOWER.get(yedda_tag.lower(), 'Misc-exposition')


# YEDDA block regex: ``[@text#Label*]``.  Same shape as
# ingestors/extract_plaintext.py and bin/annotate_spans.py.
_YEDDA_RE = re.compile(r'\[@(.*?)#([^*]+)\*\]', re.DOTALL)


def _parse_yedda_blocks(
    ann_text: str, plaintext: str,
) -> List[Tuple[str, int, int]]:
    """Locate each ``[@text#Label*]`` block of ``ann_text`` inside
    ``plaintext`` and return ``(label, char_start, char_end)``.

    Adapted from :func:`bin.annotate_spans.parse_yedda_sections` so
    this module doesn't add a bin-package import to its dependency
    chain.  Blocks whose text can't be found anywhere in the
    plaintext are silently skipped.
    """
    sections: List[Tuple[str, int, int]] = []
    search_from = 0
    for m in _YEDDA_RE.finditer(ann_text):
        block_text = m.group(1)
        label = m.group(2).strip()
        idx = plaintext.find(block_text, search_from)
        if idx == -1:
            idx = plaintext.find(block_text)
        if idx == -1:
            continue
        sections.append((label, idx, idx + len(block_text)))
        search_from = idx + len(block_text)
    return sections


def yedda_blocks_to_line_indices(
    plaintext: str,
    blocks: Sequence[Tuple[str, int, int]],
) -> List[Tuple[str, List[int]]]:
    """Convert ``[(label, char_start, char_end), ...]`` to
    ``[(label, [line_idx, ...]), ...]``.

    Char-offset → line-index uses the
    ``plaintext.count('\\n', 0, offset)`` convention shared with the
    rest of v4 (e.g. ``tests/test_page_header_golden.py``).
    """
    if not blocks:
        return []
    line_count = plaintext.count('\n') + 1
    out: List[Tuple[str, List[int]]] = []
    for label, start, end in blocks:
        end = min(end, len(plaintext))
        if start >= end:
            continue
        first_line = plaintext.count('\n', 0, start)
        last_line = plaintext.count('\n', 0, max(start, end - 1))
        line_indices = [
            li for li in range(first_line, last_line + 1)
            if li < line_count
        ]
        if line_indices:
            out.append((label, line_indices))
    return out


def yedda_tag_per_line(
    plaintext: str,
    ann_text: str,
) -> List[str]:
    """Return the raw YEDDA tag for each line of ``plaintext``.

    Lines outside any YEDDA block default to ``'Misc-exposition'``
    — same catch-all the Pass-2 vocab uses.  Reuses
    :func:`_parse_yedda_blocks` + :func:`yedda_blocks_to_line_indices`.
    """
    if not plaintext and not ann_text:
        return []
    n_lines = len(plaintext.split('\n')) if plaintext else 0
    tags: List[str] = ['Misc-exposition'] * n_lines
    blocks = _parse_yedda_blocks(ann_text, plaintext)
    for label, line_indices in yedda_blocks_to_line_indices(
        plaintext, blocks,
    ):
        for li in line_indices:
            if 0 <= li < n_lines:
                tags[li] = label
    return tags


def build_treatment_label_sequence(
    plaintext: str,
    ann_text: str,
) -> List[int]:
    """Per-line Pass-2 treatment label indices.

    For each line: look up its raw YEDDA tag (or ``'Misc-exposition'``
    when outside any block), project via :func:`map_yedda_to_treatment`,
    and resolve to the index in
    ``skol_classifier.v4.crf_treatment.LABEL_TO_INDEX``.  Length
    always equals ``len(plaintext.split('\\n'))``.

    Used by the trainer to construct the per-doc tag tensor that
    Pass-2's CRF consumes — after the trainer has filtered to the
    Pass-1 non-layout subsequence.  Lazy import keeps this module
    free of a `crf_treatment` runtime dependency at module load
    (the inverse dependency would create a cycle).
    """
    from skol_classifier.v4.crf_treatment import (
        LABEL_TO_INDEX as TREATMENT_LABEL_TO_INDEX,
    )
    return [
        TREATMENT_LABEL_TO_INDEX[map_yedda_to_treatment(tag)]
        for tag in yedda_tag_per_line(plaintext, ann_text)
    ]


def build_active_label_sequence(
    plaintext: str,
    ann_text: str,
) -> List[int]:
    """Per-line 19-label index sequence for Step 6.F's single-CRF
    baseline.

    For each line: look up its raw YEDDA tag (defaulting to
    ``'Misc-exposition'`` outside any block), project via
    :func:`map_yedda_to_active`, and resolve to the index in
    ``skol_classifier.v4.crf_single.LABEL_TO_INDEX``.  Length always
    equals ``len(plaintext.split('\\n'))``.  Lazy import avoids a
    `crf_single` → `labels` → `crf_single` cycle at module load.
    """
    from skol_classifier.v4.crf_single import (
        LABEL_TO_INDEX as ACTIVE_LABEL_TO_INDEX,
    )
    return [
        ACTIVE_LABEL_TO_INDEX[map_yedda_to_active(tag)]
        for tag in yedda_tag_per_line(plaintext, ann_text)
    ]


def build_label_sequence(
    plaintext: str,
    ann_text: str,
) -> List[int]:
    """Per-line Pass-1 label indices for a single doc.

    Lines not covered by any YEDDA block — or covered by a block
    whose tag doesn't map to one of the 7 layout tags — get
    ``LABEL_TO_INDEX['Other']``.  Returned list length always equals
    ``len(plaintext.split('\\n'))``.

    Used by the trainer to construct the per-doc tag tensor that
    pytorch-crf consumes.
    """
    if not plaintext and not ann_text:
        return []
    n_lines = len(plaintext.split('\n')) if plaintext else 0
    sequence: List[int] = [OTHER_INDEX] * n_lines

    blocks = _parse_yedda_blocks(ann_text, plaintext)
    for label, line_indices in yedda_blocks_to_line_indices(
        plaintext, blocks,
    ):
        layout_label = map_yedda_to_layout(label)
        layout_idx = LABEL_TO_INDEX[layout_label]
        for li in line_indices:
            if 0 <= li < n_lines:
                sequence[li] = layout_idx
    return sequence
