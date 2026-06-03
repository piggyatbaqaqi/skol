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


# Lowercase-keyed lookup so case drift in YEDDA files doesn't drop
# blocks to Other.
_LAYOUT_BY_LOWER = {tag.lower(): tag for tag in LAYOUT_YEDDA_TAGS}


def map_yedda_to_layout(yedda_tag: str) -> str:
    """Project a YEDDA tag to a Pass-1 label.

    Tags in :data:`LAYOUT_YEDDA_TAGS` pass through (with canonical
    capitalization); everything else returns ``'Other'``.
    Case-insensitive on the input.
    """
    return _LAYOUT_BY_LOWER.get(yedda_tag.lower(), 'Other')


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
