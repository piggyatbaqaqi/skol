"""
Convert BioC-JSON documents to YEDDA-annotated text for classifier training.

Reads BioC-JSON (as stored in CouchDB by PmcBiocIngestor) and produces
passage-level YEDDA annotations with taxonomy-specific tags:
Nomenclature, Description, Etymology, Holotype, Notes, Key,
Figure-caption, and Misc-exposition.
"""

import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional


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
        text: Raw passage text from BioC-JSON.

    Returns:
        Cleaned text with BOM characters removed and leading/trailing
        whitespace stripped.
    """
    return text.replace("\ufeff", "").strip()


class BiocTagAssigner:
    """State machine that assigns YEDDA tags to BioC-JSON passages.

    Processes passages sequentially, tracking the current subsection
    within RESULTS to correctly tag paragraphs that follow title_4
    headings like "Description.", "Etymology.", "Holotype.", "Notes."

    FIG passages are tagged independently as Figure-caption without
    changing the subsection state, since they interleave RESULTS
    passages.
    """

    def __init__(self) -> None:
        self.current_subsection: Optional[str] = None
        self.in_key_section: bool = False

    def assign_tags(self, passages: List[Dict[str, Any]]) -> List[TaggedBlock]:
        """Process all passages and return tagged blocks.

        Args:
            passages: List of BioC passage dicts, each with 'infons'
                      and 'text' fields.

        Returns:
            List of TaggedBlock objects with cleaned text and assigned tags.
            Empty passages are omitted.
        """
        blocks: List[TaggedBlock] = []
        for passage in passages:
            text = clean_passage_text(passage.get("text", ""))
            if not text:
                continue
            infons = passage.get("infons", {})
            section_type = infons.get("section_type", "")
            ptype = infons.get("type", "")
            tag = self._assign_tag(section_type, ptype, text)
            blocks.append(TaggedBlock(text=text, tag=tag))
        return blocks

    def _assign_tag(
        self, section_type: str, ptype: str, text: str
    ) -> Tag:
        # FIG passages always get Figure-caption, never change state.
        if section_type == "FIG":
            return Tag.FIGURE_CAPTION

        # Non-RESULTS sections reset state and return Misc-exposition.
        if section_type != "RESULTS":
            self._reset_results_state()
            return Tag.MISC_EXPOSITION

        # RESULTS section uses the state machine.
        return self._tag_results_passage(ptype, text)

    def _tag_results_passage(self, ptype: str, text: str) -> Tag:
        if ptype == "title_1":
            self.current_subsection = None
            self.in_key_section = False
            return Tag.MISC_EXPOSITION

        if ptype == "title_2":
            self.current_subsection = None
            text_lower = text.lower()
            if "key to" in text_lower:
                self.in_key_section = True
                return Tag.KEY
            self.in_key_section = False
            return Tag.MISC_EXPOSITION

        if ptype == "title_3":
            text_lower = text.lower()
            if "key to" in text_lower:
                self.in_key_section = True
                self.current_subsection = None
                return Tag.KEY
            self.in_key_section = False
            self.current_subsection = "nomenclature_preamble"
            return Tag.NOMENCLATURE

        if ptype == "title_4":
            return self._tag_title_4(text)

        if ptype == "paragraph":
            return self._tag_results_paragraph()

        # Other RESULTS passage types (table_caption, table, etc.)
        return Tag.MISC_EXPOSITION

    def _tag_title_4(self, text: str) -> Tag:
        heading = text.lower().strip().rstrip(".")
        if heading.startswith("etymology"):
            self.current_subsection = "etymology"
            return Tag.ETYMOLOGY
        if heading.startswith("holotype") or heading == "type":
            self.current_subsection = "holotype"
            return Tag.HOLOTYPE
        if heading.startswith("description") or heading.startswith("diagnosis"):
            self.current_subsection = "description"
            return Tag.DESCRIPTION
        if heading.startswith("culture"):
            self.current_subsection = "description"
            return Tag.DESCRIPTION
        if heading.startswith("note"):
            self.current_subsection = "notes"
            return Tag.NOTES
        # Material examined, Known distribution, Known hosts, etc.
        self.current_subsection = "misc"
        return Tag.MISC_EXPOSITION

    def _tag_results_paragraph(self) -> Tag:
        if self.in_key_section:
            return Tag.KEY
        _SUBSECTION_MAP = {
            "nomenclature_preamble": Tag.NOMENCLATURE,
            "description": Tag.DESCRIPTION,
            "etymology": Tag.ETYMOLOGY,
            "holotype": Tag.HOLOTYPE,
            "notes": Tag.NOTES,
            "misc": Tag.MISC_EXPOSITION,
        }
        return _SUBSECTION_MAP.get(
            self.current_subsection, Tag.MISC_EXPOSITION  # type: ignore[arg-type]
        )

    def _reset_results_state(self) -> None:
        self.current_subsection = None
        self.in_key_section = False


def bioc_json_to_tagged_blocks(
    bioc_json: List[Dict[str, Any]],
) -> List[TaggedBlock]:
    """Convert a BioC-JSON structure to a list of tagged blocks.

    Args:
        bioc_json: The BioC-JSON list as stored in CouchDB
                   (bioc_json[0]["documents"][0]["passages"]).

    Returns:
        List of TaggedBlock objects.

    Raises:
        ValueError: If the BioC-JSON structure is missing expected fields.
    """
    if not bioc_json:
        raise ValueError("Empty BioC-JSON list")
    collection = bioc_json[0]
    documents = collection.get("documents", [])
    if not documents:
        raise ValueError("No documents in BioC-JSON collection")
    passages = documents[0].get("passages", [])
    if not passages:
        raise ValueError("No passages in BioC-JSON document")

    assigner = BiocTagAssigner()
    return assigner.assign_tags(passages)


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


def bioc_json_to_yedda(bioc_json: List[Dict[str, Any]]) -> str:
    """Convert BioC-JSON to a YEDDA-annotated string.

    Convenience function combining passage extraction, tag assignment,
    and YEDDA rendering.

    Args:
        bioc_json: The BioC-JSON list as stored in CouchDB.

    Returns:
        YEDDA-formatted string with all passages tagged.
    """
    blocks = bioc_json_to_tagged_blocks(bioc_json)
    return tagged_blocks_to_yedda(blocks)
