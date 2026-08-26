"""Shared YEDDA tag types and utilities.

Provides the canonical Tag enum, TaggedBlock dataclass, and shared
functions used by JATS annotation converters.

This module owns the YEDDA on-disk format in both directions:
``tagged_blocks_to_yedda`` writes it and ``parse_yedda_blocks`` reads
it back.  Keeping the pair together is deliberate -- the block regex
had been copied into three separate callers, which is how offsets
drift apart between a writer and its readers.
"""

import dataclasses
import re
from enum import Enum
from typing import FrozenSet, List, Tuple

# The one YEDDA block pattern.  Was duplicated verbatim in
# bin/migrate_labels.py, fixes/merge_yedda.py and (briefly)
# treatments_to_structured/dossier.py before being pulled here.
YEDDA_BLOCK_RE = re.compile(
    r"\[@\s*(.*?)\s*#([A-Za-z][A-Za-z0-9_-]{0,49})\*\]", re.DOTALL
)


@dataclasses.dataclass(frozen=True)
class YeddaBlock:
    """One ``[@text#Label*]`` block, with its position in the source.

    Distinct from ``TaggedBlock``, which carries a validated ``Tag``
    for material this project *writes*.  A block read back off disk
    may carry any label string -- layout labels such as
    ``Misc-exposition``, or a tag from a schema version that predates
    the current enum -- so ``label`` is a plain ``str`` and validation
    is the caller's business.

    ``start``/``end`` bound the **inner text**, not the delimiters, so
    they are directly comparable with the ``*_spans`` offsets stored on
    treatment documents (memo section 16).
    """

    index: int
    label: str
    text: str
    start: int
    end: int

    @property
    def head(self) -> str:
        """First non-blank line, for one-line display."""
        for line in self.text.split("\n"):
            if line.strip():
                return line.strip()
        return ""


def parse_yedda_blocks(text: str) -> List["YeddaBlock"]:
    """Read a YEDDA string back into blocks, with offsets.

    The inverse of ``tagged_blocks_to_yedda``.  Text is stripped the
    same way the regex has always stripped it, so existing callers see
    no change; ``start``/``end`` locate the stripped text within
    ``text``.

    """
    blocks: List[YeddaBlock] = []
    for i, m in enumerate(YEDDA_BLOCK_RE.finditer(text)):
        # group(1) already excludes the surrounding whitespace the
        # pattern consumes, so its span IS the stripped text's span --
        # which is what makes the offsets comparable with *_spans.
        blocks.append(YeddaBlock(
            index=i,
            label=m.group(2),
            text=m.group(1),
            start=m.start(1),
            end=m.end(1),
        ))
    return blocks


class Tag(str, Enum):
    """YEDDA annotation tags for taxonomic text classification."""

    NOMENCLATURE = "Nomenclature"
    DESCRIPTION = "Description"
    DIAGNOSIS = "Diagnosis"
    ETYMOLOGY = "Etymology"
    MATERIALS_EXAMINED = "Materials-examined"
    MATERIALS_AND_METHODS = "Materials-and-methods"
    TYPE_DESIGNATION = "Type-designation"
    BIOLOGY = "Biology"
    PHYLOGENY = "Phylogeny"
    NEW_COMBINATIONS = "New-combinations"
    NOTES = "Notes"
    KEY = "Key"
    FIGURE_CAPTION = "Figure-caption"
    BIBLIOGRAPHY = "Bibliography"
    TABLE = "Table"
    INDEX = "Index"
    TOC = "ToC-entry"
    MISC_EXPOSITION = "Misc-exposition"
    FIX = "FIX"
    # Structural / pagination tags — not taxonomic content, but used in
    # PDF-sourced annotations to mark running heads and section dividers.
    PAGE_HEADER = "Page-header"
    # Deprecated: retained so existing .ann files with Holotype remain parseable.
    HOLOTYPE = "Holotype"
    # Deprecated: Replace with BIOLOGY.
    DISTRIBUTION = "Distribution"


DEPRECATED_TAGS: FrozenSet[Tag] = frozenset({
    Tag.HOLOTYPE,      # folded into TYPE_DESIGNATION
    Tag.DISTRIBUTION,  # folded into BIOLOGY
    Tag.FIX,           # workflow marker, not a semantic class
})
"""Tags excluded from ACTIVE_TAGS_19. HOLOTYPE / DISTRIBUTION are
semantically deprecated (their content folds into the listed
replacements); FIX is a workflow marker, not a label."""


ACTIVE_TAGS_19: Tuple[Tag, ...] = tuple(
    t for t in Tag if t not in DEPRECATED_TAGS
)
"""The 19 canonical active tags. Source of truth for any consumer
that needs to enumerate the label space — classifier MODEL_CONFIG
class_weights, schema validation, the JATS converter's emit set.
Order is the Tag enum declaration order (stable for serialisation)."""


@dataclasses.dataclass
class TaggedBlock:
    """A passage of text with its assigned YEDDA tag."""

    text: str
    tag: Tag


def clean_passage_text(text: str) -> str:
    """Strip BOM characters and normalize whitespace.

    Args:
        text: Raw passage text.

    Returns:
        Cleaned text with BOM characters removed and leading/trailing
        whitespace stripped.
    """
    return text.replace("\ufeff", "").strip()


def tagged_blocks_to_yedda(blocks: List[TaggedBlock]) -> str:
    """Render tagged blocks as YEDDA-annotated text.

    Each block becomes ``[@text#Tag*]``, separated by blank lines.

    Args:
        blocks: List of TaggedBlock objects.

    Returns:
        YEDDA-formatted string.
    """
    parts: List[str] = []
    for block in blocks:
        parts.append(f"[@{block.text}#{block.tag.value}*]")
    return "\n\n".join(parts) + "\n"
