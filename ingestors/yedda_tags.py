"""Shared YEDDA tag types and utilities.

Provides the canonical Tag enum, TaggedBlock dataclass, and shared
functions used by JATS annotation converters.
"""

import dataclasses
from enum import Enum
from typing import List


class Tag(str, Enum):
    """YEDDA annotation tags for taxonomic text classification."""

    NOMENCLATURE = "Nomenclature"
    DESCRIPTION = "Description"
    ETYMOLOGY = "Etymology"
    HOLOTYPE = "Holotype"
    NOTES = "Notes"
    KEY = "Key"
    FIGURE_CAPTION = "Figure-caption"
    MISC_EXPOSITION = "Misc-exposition"


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
