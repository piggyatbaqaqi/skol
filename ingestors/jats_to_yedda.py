"""
Convert JATS/TaxPub XML documents to YEDDA-annotated text for classifier training.

Reads JATS XML with TaxPub extension (as stored in CouchDB by PensoftIngestor)
and produces passage-level YEDDA annotations with taxonomy-specific tags:
Nomenclature, Description, Etymology, Holotype, Notes, Key,
Figure-caption, and Misc-exposition.

TaxPub provides explicit structural markup via tp:taxon-treatment,
tp:nomenclature, and tp:treatment-sec elements, making tag assignment
straightforward without state tracking.
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Optional, Set

from ingestors.bioc_to_yedda import (
    Tag,
    TaggedBlock,
    clean_passage_text,
    tagged_blocks_to_yedda,
)

# TaxPub namespace
TP_NS = "http://www.plazi.org/taxpub"
TP_PREFIX = f"{{{TP_NS}}}"

# Tags to skip during text extraction (contain UUIDs, MycoBank IDs, DOIs)
_DEFAULT_SKIP_TAGS = frozenset({"object-id"})
# Also skip fig elements when extracting parent section text
_SKIP_TAGS_WITH_FIG = frozenset({"object-id", "fig"})


def _local_tag(elem: ET.Element) -> str:
    """Strip namespace prefix from element tag."""
    tag = elem.tag
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def extract_text(
    elem: ET.Element,
    skip_tags: Optional[Set[str]] = None,
) -> str:
    """Recursively extract text from an XML element, skipping specified tags.

    Args:
        elem: XML element to extract text from.
        skip_tags: Set of local tag names to skip (e.g., {"object-id", "fig"}).

    Returns:
        Concatenated text content with inline markup stripped.
    """
    if skip_tags is None:
        skip_tags = _DEFAULT_SKIP_TAGS
    local = _local_tag(elem)
    if local in skip_tags:
        return ""
    parts: List[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(extract_text(child, skip_tags))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def extract_fig_blocks(elem: ET.Element) -> List[TaggedBlock]:
    """Extract Figure-caption blocks from all <fig> elements within an element.

    Args:
        elem: XML element that may contain nested <fig> elements.

    Returns:
        List of TaggedBlock with Figure-caption tag for each fig found.
    """
    blocks: List[TaggedBlock] = []
    for fig in elem.iter("fig"):
        parts: List[str] = []
        label = fig.find("label")
        if label is not None:
            label_text = extract_text(label).strip()
            if label_text:
                parts.append(label_text)
        caption = fig.find("caption")
        if caption is not None:
            caption_text = extract_text(caption).strip()
            if caption_text:
                parts.append(caption_text)
        text = clean_passage_text(" ".join(parts))
        if text:
            blocks.append(TaggedBlock(text=text, tag=Tag.FIGURE_CAPTION))
    return blocks


def sec_type_to_tag(sec_type: str) -> Tag:
    """Map a TaxPub sec-type attribute value to a YEDDA Tag.

    Case-insensitive matching. Handles the variety of sec-type values
    found in Pensoft journals.

    Args:
        sec_type: The sec-type attribute value from tp:treatment-sec.

    Returns:
        The corresponding Tag enum value.
    """
    st = sec_type.lower().strip().strip("\ufeff")

    if st in ("description", "diagnosis"):
        return Tag.DESCRIPTION

    if st == "etymology":
        return Tag.ETYMOLOGY

    if st in ("holotype", "material", "type material"):
        return Tag.HOLOTYPE

    if st in ("type species", "type genus"):
        return Tag.HOLOTYPE

    if st in ("notes", "comments", "note"):
        return Tag.NOTES

    if st == "key" or "key to" in st:
        return Tag.KEY

    # Morphological subsections are descriptions
    _DESCRIPTION_SUBTYPES = {
        "fruiting body",
        "hyphal system",
        "hymenial layer",
        "basidiospores",
        "culture characteristics",
        "basidiomata",
        "pileipellis",
        "stipitipellis",
        "basidia",
        "cystidia",
        "clamp connections",
        "spores",
        "ascospores",
        "asci",
        "conidiophores",
        "conidia",
        "colonies",
        "colony",
    }
    if st in _DESCRIPTION_SUBTYPES:
        return Tag.DESCRIPTION

    # Anything starting with "description" or "diagnosis" (handles variants)
    if st.startswith("description") or st.startswith("diagnosis"):
        return Tag.DESCRIPTION

    return Tag.MISC_EXPOSITION


def process_treatment(treatment_elem: ET.Element) -> List[TaggedBlock]:
    """Process a single tp:taxon-treatment element into tagged blocks.

    Extracts nomenclature, treatment sections, and figures in document order.

    Args:
        treatment_elem: A <tp:taxon-treatment> XML element.

    Returns:
        List of TaggedBlock in document order.
    """
    blocks: List[TaggedBlock] = []

    for child in treatment_elem:
        local = _local_tag(child)

        if local == "treatment-meta":
            continue

        if local == "nomenclature":
            text = clean_passage_text(
                re.sub(r"\s+", " ", extract_text(child, _DEFAULT_SKIP_TAGS))
            )
            if text:
                blocks.append(TaggedBlock(text=text, tag=Tag.NOMENCLATURE))
            continue

        if local == "treatment-sec":
            sec_type = child.get("sec-type", "")
            tag = sec_type_to_tag(sec_type)

            # Extract section text without fig content
            text = clean_passage_text(
                re.sub(r"\s+", " ", extract_text(child, _SKIP_TAGS_WITH_FIG))
            )
            if text:
                blocks.append(TaggedBlock(text=text, tag=tag))

            # Extract fig blocks separately
            blocks.extend(extract_fig_blocks(child))
            continue

    return blocks


def process_key_section(sec_elem: ET.Element) -> List[TaggedBlock]:
    """Process a key section (containing dichotomous key) into tagged blocks.

    Handles both <table-wrap content-type="key"> and plain paragraph keys.

    Args:
        sec_elem: A <sec> element whose sec-type indicates a key.

    Returns:
        List of TaggedBlock with Key tag.
    """
    text = clean_passage_text(
        re.sub(r"\s+", " ", extract_text(sec_elem, _SKIP_TAGS_WITH_FIG))
    )
    blocks: List[TaggedBlock] = []
    if text:
        blocks.append(TaggedBlock(text=text, tag=Tag.KEY))
    blocks.extend(extract_fig_blocks(sec_elem))
    return blocks


def _is_key_section(sec_elem: ET.Element) -> bool:
    """Check if a <sec> element is a dichotomous key section."""
    sec_type = sec_elem.get("sec-type", "").lower().strip().strip("\ufeff")
    if "key to" in sec_type or sec_type == "key":
        return True
    # Also check for table-wrap with content-type="key"
    for tw in sec_elem.iter("table-wrap"):
        if tw.get("content-type", "") == "key":
            return True
    return False


def _has_treatments(sec_elem: ET.Element) -> bool:
    """Check if a <sec> element contains tp:taxon-treatment elements."""
    for _ in sec_elem.iter(f"{TP_PREFIX}taxon-treatment"):
        return True
    return False


def _process_body_section(sec_elem: ET.Element) -> List[TaggedBlock]:
    """Process a top-level body <sec> element.

    Routes to treatment processing, key processing, or misc-exposition
    based on the section content.
    """
    # Key sections
    if _is_key_section(sec_elem):
        return process_key_section(sec_elem)

    # Sections containing taxon treatments
    if _has_treatments(sec_elem):
        blocks: List[TaggedBlock] = []
        # Process nested sections that may contain treatments
        for child in sec_elem:
            local = _local_tag(child)
            if local == "taxon-treatment":
                blocks.extend(process_treatment(child))
            elif local == "sec":
                # Recurse into nested sections
                blocks.extend(_process_body_section(child))
        return blocks

    # Check nested <sec> children for keys or treatments
    nested_secs = sec_elem.findall("sec")
    if nested_secs:
        blocks = []
        has_special = False
        for nested in nested_secs:
            if _is_key_section(nested) or _has_treatments(nested):
                has_special = True
                blocks.extend(_process_body_section(nested))
        if has_special:
            # Also add the non-special text as Misc-exposition
            # (e.g., section title/intro paragraphs)
            return blocks

    # Plain section (intro, methods, discussion, etc.) → Misc-exposition
    text = clean_passage_text(
        re.sub(r"\s+", " ", extract_text(sec_elem, _SKIP_TAGS_WITH_FIG))
    )
    blocks = []
    if text:
        blocks.append(TaggedBlock(text=text, tag=Tag.MISC_EXPOSITION))
    blocks.extend(extract_fig_blocks(sec_elem))
    return blocks


def jats_xml_to_tagged_blocks(xml_string: str) -> List[TaggedBlock]:
    """Convert a JATS/TaxPub XML string to a list of tagged blocks.

    Args:
        xml_string: The full JATS XML document as a string.

    Returns:
        List of TaggedBlock objects in document order.

    Raises:
        ValueError: If the XML cannot be parsed or has no body.
    """
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse JATS XML: {exc}") from exc

    blocks: List[TaggedBlock] = []

    # Abstract
    for abstract in root.iter("abstract"):
        text = clean_passage_text(
            re.sub(r"\s+", " ", extract_text(abstract, _DEFAULT_SKIP_TAGS))
        )
        if text:
            blocks.append(TaggedBlock(text=text, tag=Tag.MISC_EXPOSITION))

    # Body sections
    body = root.find(".//body")
    if body is None:
        raise ValueError("No <body> element found in JATS XML")

    for sec in body:
        local = _local_tag(sec)
        if local == "sec":
            blocks.extend(_process_body_section(sec))
        elif local == "taxon-treatment":
            # Treatments directly under body (unusual but possible)
            blocks.extend(process_treatment(sec))

    # Back matter (references, acknowledgments, etc.)
    back = root.find(".//back")
    if back is not None:
        text = clean_passage_text(
            re.sub(r"\s+", " ", extract_text(back, _DEFAULT_SKIP_TAGS))
        )
        if text:
            blocks.append(TaggedBlock(text=text, tag=Tag.MISC_EXPOSITION))

    return blocks


def jats_xml_to_yedda(xml_string: str) -> str:
    """Convert JATS/TaxPub XML to a YEDDA-annotated string.

    Convenience function combining XML parsing, tag assignment,
    and YEDDA rendering.

    Args:
        xml_string: The full JATS XML document as a string.

    Returns:
        YEDDA-formatted string with all passages tagged.
    """
    blocks = jats_xml_to_tagged_blocks(xml_string)
    return tagged_blocks_to_yedda(blocks)
