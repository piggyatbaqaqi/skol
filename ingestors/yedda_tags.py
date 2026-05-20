"""Shared YEDDA tag types and utilities.

Provides the canonical Tag enum, TaggedBlock dataclass, and shared
functions used by JATS annotation converters.
"""

import dataclasses
from enum import Enum
from typing import FrozenSet, List, Tuple


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
