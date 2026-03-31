"""Tests for yedda_tags.py — Tag enum and utilities."""

from typing import List

import pytest

from ingestors.yedda_tags import Tag, TaggedBlock, tagged_blocks_to_yedda


class TestTagEnum:
    """The 14-tag enum (12 active + Holotype deprecated + Bibliography + Table)."""

    def test_all_active_tags_present(self) -> None:
        tag_values = {t.value for t in Tag}
        expected = {
            "Nomenclature", "Description", "Diagnosis", "Etymology",
            "Distribution", "Materials-examined", "Type-designation", "Biology",
            "Notes", "Key", "Figure-caption", "Bibliography", "Table",
            "Misc-exposition",
        }
        assert expected.issubset(tag_values)

    def test_new_tags_have_correct_values(self) -> None:
        assert Tag.DIAGNOSIS.value == "Diagnosis"
        assert Tag.DISTRIBUTION.value == "Distribution"
        assert Tag.MATERIALS_EXAMINED.value == "Materials-examined"
        assert Tag.TYPE_DESIGNATION.value == "Type-designation"
        assert Tag.BIOLOGY.value == "Biology"
        assert Tag.BIBLIOGRAPHY.value == "Bibliography"
        assert Tag.TABLE.value == "Table"

    def test_existing_tags_unchanged(self) -> None:
        assert Tag.NOMENCLATURE.value == "Nomenclature"
        assert Tag.DESCRIPTION.value == "Description"
        assert Tag.ETYMOLOGY.value == "Etymology"
        assert Tag.NOTES.value == "Notes"
        assert Tag.KEY.value == "Key"
        assert Tag.FIGURE_CAPTION.value == "Figure-caption"
        assert Tag.MISC_EXPOSITION.value == "Misc-exposition"

    def test_holotype_deprecated_alias_readable(self) -> None:
        """Old .ann files containing Holotype must remain parseable."""
        tag = Tag("Holotype")
        assert tag == Tag.HOLOTYPE

    def test_tag_is_str(self) -> None:
        """Tag extends str — values compare equal to plain strings."""
        assert Tag.DIAGNOSIS == "Diagnosis"
        assert Tag.DISTRIBUTION == "Distribution"
        assert Tag.MATERIALS_EXAMINED == "Materials-examined"
        assert Tag.TYPE_DESIGNATION == "Type-designation"
        assert Tag.BIOLOGY == "Biology"

    def test_lookup_by_value_round_trips(self) -> None:
        """Tag(value) round-trips for all tags including deprecated Holotype."""
        for tag in Tag:
            assert Tag(tag.value) is tag


class TestTaggedBlock:
    """TaggedBlock dataclass."""

    def test_attributes(self) -> None:
        block = TaggedBlock(text="foo", tag=Tag.DIAGNOSIS)
        assert block.text == "foo"
        assert block.tag == Tag.DIAGNOSIS


class TestTaggedBlocksToYedda:
    """tagged_blocks_to_yedda() renders new tags correctly."""

    def test_diagnosis_renders(self) -> None:
        blocks: List[TaggedBlock] = [TaggedBlock(text="differs from X by...", tag=Tag.DIAGNOSIS)]
        result = tagged_blocks_to_yedda(blocks)
        assert "[@differs from X by...#Diagnosis*]" in result

    def test_distribution_renders(self) -> None:
        blocks: List[TaggedBlock] = [TaggedBlock(text="Europe, Asia", tag=Tag.DISTRIBUTION)]
        result = tagged_blocks_to_yedda(blocks)
        assert "[@Europe, Asia#Distribution*]" in result

    def test_materials_examined_renders(self) -> None:
        blocks: List[TaggedBlock] = [TaggedBlock(text="Holotype: NY 12345", tag=Tag.MATERIALS_EXAMINED)]
        result = tagged_blocks_to_yedda(blocks)
        assert "[@Holotype: NY 12345#Materials-examined*]" in result

    def test_type_designation_renders(self) -> None:
        blocks: List[TaggedBlock] = [TaggedBlock(text="Type: Amanita muscaria", tag=Tag.TYPE_DESIGNATION)]
        result = tagged_blocks_to_yedda(blocks)
        assert "[@Type: Amanita muscaria#Type-designation*]" in result

    def test_biology_renders(self) -> None:
        blocks: List[TaggedBlock] = [TaggedBlock(text="Saprotrophic on oak.", tag=Tag.BIOLOGY)]
        result = tagged_blocks_to_yedda(blocks)
        assert "[@Saprotrophic on oak.#Biology*]" in result

    def test_bibliography_renders(self) -> None:
        blocks: List[TaggedBlock] = [
            TaggedBlock(text="Smith J. 2001. Fungi. J. Bot. 1: 1–10.", tag=Tag.BIBLIOGRAPHY)
        ]
        result = tagged_blocks_to_yedda(blocks)
        assert "[@Smith J. 2001. Fungi. J. Bot. 1: 1–10.#Bibliography*]" in result

    def test_table_renders(self) -> None:
        blocks: List[TaggedBlock] = [
            TaggedBlock(text="Table 1. Conidia dimensions.", tag=Tag.TABLE)
        ]
        result = tagged_blocks_to_yedda(blocks)
        assert "[@Table 1. Conidia dimensions.#Table*]" in result
