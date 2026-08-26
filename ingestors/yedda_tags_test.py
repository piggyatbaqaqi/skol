"""Tests for yedda_tags.py — Tag enum and utilities."""

from typing import List

import pytest

from ingestors.yedda_tags import (
    ACTIVE_TAGS_19,
    DEPRECATED_TAGS,
    Tag,
    TaggedBlock,
    YEDDA_BLOCK_RE,
    parse_yedda_blocks,
    tagged_blocks_to_yedda,
)


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
        blocks: List[TaggedBlock] = [
            TaggedBlock(text="Holotype: NY 12345",
                        tag=Tag.MATERIALS_EXAMINED)]
        result = tagged_blocks_to_yedda(blocks)
        assert "[@Holotype: NY 12345#Materials-examined*]" in result

    def test_type_designation_renders(self) -> None:
        blocks: List[TaggedBlock] = [
            TaggedBlock(text="Type: Amanita muscaria",
                        tag=Tag.TYPE_DESIGNATION)]
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


class TestActiveTags19:
    """``ACTIVE_TAGS_19`` is the canonical 19-tag label set used by all
    v3 consumers (classifier MODEL_CONFIGs, schema validation, the
    JATS converter's emit set). Per Step 1 of
    docs/production_v3_plan.md: 22 declared tags minus 2 deprecated
    (``HOLOTYPE``, ``DISTRIBUTION``) minus 1 workflow marker (``FIX``)
    = 19 active. These tests pin that partition so a future enum
    addition can't slip past unclassified."""

    def test_count_is_19(self) -> None:
        """Anchor count: any tag added to the enum must be explicitly
        categorised into ACTIVE_TAGS_19 or DEPRECATED_TAGS, never
        forgotten."""
        assert len(ACTIVE_TAGS_19) == 19

    def test_holotype_excluded(self) -> None:
        """Deprecated — folds into TYPE_DESIGNATION."""
        assert Tag.HOLOTYPE not in ACTIVE_TAGS_19

    def test_distribution_excluded(self) -> None:
        """Deprecated — folds into BIOLOGY."""
        assert Tag.DISTRIBUTION not in ACTIVE_TAGS_19

    def test_fix_excluded(self) -> None:
        """Workflow marker, not a semantic class."""
        assert Tag.FIX not in ACTIVE_TAGS_19

    def test_partition_covers_enum(self) -> None:
        """ACTIVE_TAGS_19 ∪ DEPRECATED_TAGS = every Tag — catches a
        future tag added but never categorised."""
        assert set(ACTIVE_TAGS_19) | set(DEPRECATED_TAGS) == set(Tag)

    def test_active_disjoint_from_deprecated(self) -> None:
        """No tag is both active and deprecated — guards against the
        natural copy-paste mistake."""
        assert set(ACTIVE_TAGS_19).isdisjoint(set(DEPRECATED_TAGS))

    def test_exact_string_anchor(self) -> None:
        """Pin the literal 19 tag values. Renames force a deliberate
        test update so a typo'd value can't slip through. Order
        independence: compare as sets."""
        expected = {
            "Nomenclature", "Description", "Diagnosis", "Etymology",
            "Materials-examined", "Materials-and-methods",
            "Type-designation", "Biology", "Phylogeny",
            "New-combinations", "Notes", "Key", "Figure-caption",
            "Bibliography", "Table", "Index", "ToC-entry",
            "Misc-exposition", "Page-header",
        }
        assert {t.value for t in ACTIVE_TAGS_19} == expected

    def test_is_ordered_tuple(self) -> None:
        """ACTIVE_TAGS_19 is a tuple, not a set — order is the Tag
        enum declaration order. Documents the contract for any
        consumer that needs a stable iteration order (ordered
        class_weight dicts, serialised feature columns)."""
        assert isinstance(ACTIVE_TAGS_19, tuple)


@pytest.mark.xfail(raises=NotImplementedError, strict=True,
                   reason="extraction follows test confirmation")
class TestParseYeddaBlocks:
    """The reader half of the format, pulled out of three callers.

    ``bin/migrate_labels.py`` and ``fixes/merge_yedda.py`` each carried
    a verbatim copy of the block regex, and
    ``treatments_to_structured/dossier`` was about to make a third.
    These tests pin the behaviour those callers already depend on, so
    the extraction cannot change it silently.
    """

    def test_reads_text_and_label(self) -> None:
        got = parse_yedda_blocks("[@Pileus brown.#Description*]")
        assert (got[0].text, got[0].label) == ("Pileus brown.",
                                               "Description")

    def test_round_trips_with_the_writer(self) -> None:
        """`tagged_blocks_to_yedda` is the inverse and lives beside it;
        a reader that cannot read its own writer's output is broken.
        """
        blocks = [TaggedBlock(text="Pileus brown.", tag=Tag.DESCRIPTION),
                  TaggedBlock(text="Notes here.", tag=Tag.NOTES)]
        back = parse_yedda_blocks(tagged_blocks_to_yedda(blocks))
        assert [(b.text, b.label) for b in back] == [
            ("Pileus brown.", Tag.DESCRIPTION.value),
            ("Notes here.", Tag.NOTES.value)]

    def test_strips_surrounding_whitespace_as_the_regex_always_has(
        self,
    ) -> None:
        """Both existing callers do `m.group(1).strip()`; the shared
        parser must strip too or their output changes.
        """
        got = parse_yedda_blocks("[@   Pileus brown.   #Description*]")
        assert got[0].text == "Pileus brown."

    def test_offsets_locate_the_stripped_text(self) -> None:
        """New capability, and the reason for the extraction: stored
        `*_spans` offsets point at block text, so a parser whose
        offsets included the delimiters would never line up.
        """
        raw = "[@Pileus brown.#Description*]"
        b = parse_yedda_blocks(raw)[0]
        assert raw[b.start:b.end] == "Pileus brown."

    def test_offsets_survive_leading_whitespace(self) -> None:
        raw = "[@   Pileus brown.#Description*]"
        b = parse_yedda_blocks(raw)[0]
        assert raw[b.start:b.end] == "Pileus brown."

    def test_multiline_blocks_are_one_block(self) -> None:
        """The regex is DOTALL in both existing copies."""
        got = parse_yedda_blocks("[@line one\nline two#Description*]")
        assert len(got) == 1 and "line two" in got[0].text

    def test_blocks_are_indexed_in_document_order(self) -> None:
        got = parse_yedda_blocks(
            "[@A#Description*]\n\n[@B#Notes*]\n\n[@C#Table*]")
        assert [b.index for b in got] == [0, 1, 2]
        assert [b.label for b in got] == ["Description", "Notes", "Table"]

    def test_head_is_the_first_non_blank_line(self) -> None:
        got = parse_yedda_blocks(
            "[@\n\nAsci clavate.\nSpores.#Description*]")
        assert got[0].head == "Asci clavate."

    def test_arbitrary_labels_are_accepted(self) -> None:
        """A block read off disk may carry a layout label or a tag from
        an older schema.  Validation is the caller's business, not the
        parser's -- refusing here would make the dossier unable to show
        exactly the blocks worth looking at.
        """
        got = parse_yedda_blocks(
            "[@x#Misc-exposition*]\n\n[@y#Not-A-Tag*]")
        assert [b.label for b in got] == ["Misc-exposition", "Not-A-Tag"]

    def test_empty_input_yields_no_blocks(self) -> None:
        assert parse_yedda_blocks("") == []

    def test_untagged_text_is_ignored(self) -> None:
        assert parse_yedda_blocks("plain text, no blocks") == []

    def test_matches_the_regex_the_callers_used(self) -> None:
        """Guard on the extraction itself: same groups, same order."""
        raw = "[@A#Description*]\n\n[@B#Table*]"
        assert [(m.group(1), m.group(2))
                for m in YEDDA_BLOCK_RE.finditer(raw)] == [
            (b.text, b.label) for b in parse_yedda_blocks(raw)]
