"""Tests for BioC-JSON to YEDDA translator."""

import unittest
from typing import Any, Dict, List

from .bioc_to_yedda import (
    BiocTagAssigner,
    Tag,
    TaggedBlock,
    bioc_json_to_tagged_blocks,
    bioc_json_to_yedda,
    clean_passage_text,
    tagged_blocks_to_yedda,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_passage(
    section_type: str,
    ptype: str,
    text: str,
    **extra_infons: Any,
) -> Dict[str, Any]:
    """Build a single BioC passage dict."""
    infons: Dict[str, Any] = {
        "section_type": section_type,
        "type": ptype,
    }
    infons.update(extra_infons)
    return {
        "bioctype": "BioCPassage",
        "offset": 0,
        "infons": infons,
        "text": text,
        "sentences": [],
        "annotations": [],
        "relations": [],
    }


def _make_bioc_json(
    passages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Wrap passages in the BioCCollection / BioCDocument structure."""
    return [
        {
            "bioctype": "BioCCollection",
            "source": "PMC",
            "date": "20260101",
            "key": "pmc.key",
            "version": "1.0",
            "infons": {},
            "documents": [
                {
                    "bioctype": "BioCDocument",
                    "id": "PMC0000000",
                    "infons": {"license": "CC BY"},
                    "passages": passages,
                    "annotations": [],
                    "relations": [],
                }
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCleanPassageText(unittest.TestCase):
    """BOM stripping and whitespace normalization."""

    def test_strips_single_bom(self) -> None:
        self.assertEqual(clean_passage_text("\ufeffHello"), "Hello")

    def test_strips_multiple_boms(self) -> None:
        self.assertEqual(
            clean_passage_text("\ufeff\ufeff\ufeffTitle"), "Title"
        )

    def test_strips_whitespace(self) -> None:
        self.assertEqual(clean_passage_text("  text  "), "text")

    def test_empty_after_strip(self) -> None:
        self.assertEqual(clean_passage_text("\ufeff  "), "")

    def test_preserves_internal_bom(self) -> None:
        # BOM in the middle is unusual but should still be stripped
        self.assertEqual(clean_passage_text("a\ufeffb"), "ab")


class TestSimpleSectionTagging(unittest.TestCase):
    """Non-RESULTS sections all map to Misc-exposition."""

    def _assert_section_tag(
        self, section_type: str, ptype: str, expected: Tag
    ) -> None:
        passages = [_make_passage(section_type, ptype, "Some text.")]
        assigner = BiocTagAssigner()
        blocks = assigner.assign_tags(passages)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tag, expected)

    def test_title_front(self) -> None:
        self._assert_section_tag("TITLE", "front", Tag.MISC_EXPOSITION)

    def test_abstract(self) -> None:
        self._assert_section_tag("ABSTRACT", "abstract", Tag.MISC_EXPOSITION)

    def test_abstract_title(self) -> None:
        self._assert_section_tag(
            "ABSTRACT", "abstract_title_1", Tag.MISC_EXPOSITION
        )

    def test_intro_paragraph(self) -> None:
        self._assert_section_tag("INTRO", "paragraph", Tag.MISC_EXPOSITION)

    def test_intro_title(self) -> None:
        self._assert_section_tag("INTRO", "title_1", Tag.MISC_EXPOSITION)

    def test_methods_paragraph(self) -> None:
        self._assert_section_tag("METHODS", "paragraph", Tag.MISC_EXPOSITION)

    def test_methods_title(self) -> None:
        self._assert_section_tag("METHODS", "title_2", Tag.MISC_EXPOSITION)

    def test_table_caption(self) -> None:
        self._assert_section_tag("TABLE", "table_caption", Tag.MISC_EXPOSITION)

    def test_table_data(self) -> None:
        self._assert_section_tag("TABLE", "table", Tag.MISC_EXPOSITION)

    def test_table_footnote(self) -> None:
        self._assert_section_tag(
            "TABLE", "table_footnote", Tag.MISC_EXPOSITION
        )

    def test_discuss_paragraph(self) -> None:
        self._assert_section_tag("DISCUSS", "paragraph", Tag.MISC_EXPOSITION)

    def test_suppl_paragraph(self) -> None:
        self._assert_section_tag("SUPPL", "paragraph", Tag.MISC_EXPOSITION)

    def test_ref_entry(self) -> None:
        self._assert_section_tag("REF", "ref", Tag.MISC_EXPOSITION)

    def test_unknown_section(self) -> None:
        self._assert_section_tag("UNKNOWN", "paragraph", Tag.MISC_EXPOSITION)


class TestFigCaptionTagging(unittest.TestCase):
    """FIG section_type always produces Figure-caption."""

    def test_fig_caption(self) -> None:
        passages = [
            _make_passage("FIG", "fig_caption", "Figure 1. Colonies on host.")
        ]
        assigner = BiocTagAssigner()
        blocks = assigner.assign_tags(passages)
        self.assertEqual(blocks[0].tag, Tag.FIGURE_CAPTION)


class TestSpeciesHeadingTagging(unittest.TestCase):
    """RESULTS title_3 passages are Nomenclature; preamble inherits."""

    def test_title_3_is_nomenclature(self) -> None:
        passages = [
            _make_passage(
                "RESULTS",
                "title_3",
                "Kirschsteiniothelia inthanonensis sp. nov.",
            )
        ]
        assigner = BiocTagAssigner()
        blocks = assigner.assign_tags(passages)
        self.assertEqual(blocks[0].tag, Tag.NOMENCLATURE)

    def test_preamble_paragraphs_are_nomenclature(self) -> None:
        passages = [
            _make_passage(
                "RESULTS",
                "title_3",
                "Kirschsteiniothelia inthanonensis sp. nov.",
            ),
            _make_passage(
                "RESULTS",
                "paragraph",
                "MycoBank No: MB 851234",
            ),
            _make_passage(
                "RESULTS",
                "paragraph",
                "Fig. 3",
            ),
        ]
        assigner = BiocTagAssigner()
        blocks = assigner.assign_tags(passages)
        self.assertEqual(blocks[0].tag, Tag.NOMENCLATURE)
        self.assertEqual(blocks[1].tag, Tag.NOMENCLATURE)
        self.assertEqual(blocks[2].tag, Tag.NOMENCLATURE)


class TestSubsectionTagging(unittest.TestCase):
    """title_4 headings set subsection state for subsequent paragraphs."""

    def _make_species_block(
        self, title_4_text: str, para_text: str = "Body text."
    ) -> List[TaggedBlock]:
        passages = [
            _make_passage(
                "RESULTS", "title_3", "Species name sp. nov."
            ),
            _make_passage("RESULTS", "title_4", title_4_text),
            _make_passage("RESULTS", "paragraph", para_text),
        ]
        assigner = BiocTagAssigner()
        return assigner.assign_tags(passages)

    def test_etymology_heading_and_paragraph(self) -> None:
        blocks = self._make_species_block("Etymology.")
        self.assertEqual(blocks[1].tag, Tag.ETYMOLOGY)
        self.assertEqual(blocks[2].tag, Tag.ETYMOLOGY)

    def test_holotype_heading_and_paragraph(self) -> None:
        blocks = self._make_species_block("Holotype.")
        self.assertEqual(blocks[1].tag, Tag.HOLOTYPE)
        self.assertEqual(blocks[2].tag, Tag.HOLOTYPE)

    def test_type_heading_maps_to_holotype(self) -> None:
        blocks = self._make_species_block("Type.")
        self.assertEqual(blocks[1].tag, Tag.HOLOTYPE)
        self.assertEqual(blocks[2].tag, Tag.HOLOTYPE)

    def test_description_heading_and_paragraph(self) -> None:
        blocks = self._make_species_block("Description.")
        self.assertEqual(blocks[1].tag, Tag.DESCRIPTION)
        self.assertEqual(blocks[2].tag, Tag.DESCRIPTION)

    def test_diagnosis_heading_maps_to_description(self) -> None:
        blocks = self._make_species_block("Diagnosis.")
        self.assertEqual(blocks[1].tag, Tag.DESCRIPTION)
        self.assertEqual(blocks[2].tag, Tag.DESCRIPTION)

    def test_culture_characteristics_maps_to_description(self) -> None:
        blocks = self._make_species_block("Culture characteristics.")
        self.assertEqual(blocks[1].tag, Tag.DESCRIPTION)
        self.assertEqual(blocks[2].tag, Tag.DESCRIPTION)

    def test_notes_heading_and_paragraph(self) -> None:
        blocks = self._make_species_block("Notes.")
        self.assertEqual(blocks[1].tag, Tag.NOTES)
        self.assertEqual(blocks[2].tag, Tag.NOTES)

    def test_note_singular_maps_to_notes(self) -> None:
        blocks = self._make_species_block("Note.")
        self.assertEqual(blocks[1].tag, Tag.NOTES)
        self.assertEqual(blocks[2].tag, Tag.NOTES)

    def test_material_examined_is_misc(self) -> None:
        blocks = self._make_species_block("Material examined.")
        self.assertEqual(blocks[1].tag, Tag.MISC_EXPOSITION)
        self.assertEqual(blocks[2].tag, Tag.MISC_EXPOSITION)

    def test_known_distribution_is_misc(self) -> None:
        blocks = self._make_species_block("Known distribution.")
        self.assertEqual(blocks[1].tag, Tag.MISC_EXPOSITION)
        self.assertEqual(blocks[2].tag, Tag.MISC_EXPOSITION)

    def test_specimens_examined_is_misc(self) -> None:
        blocks = self._make_species_block("Specimens examined.")
        self.assertEqual(blocks[1].tag, Tag.MISC_EXPOSITION)
        self.assertEqual(blocks[2].tag, Tag.MISC_EXPOSITION)


class TestStateReset(unittest.TestCase):
    """State resets correctly at species and section boundaries."""

    def test_new_title_3_resets_to_nomenclature_preamble(self) -> None:
        passages = [
            _make_passage(
                "RESULTS", "title_3", "Species A sp. nov."
            ),
            _make_passage("RESULTS", "title_4", "Description."),
            _make_passage("RESULTS", "paragraph", "Descr text."),
            # New species heading resets state
            _make_passage(
                "RESULTS", "title_3", "Species B sp. nov."
            ),
            _make_passage("RESULTS", "paragraph", "Preamble para."),
        ]
        assigner = BiocTagAssigner()
        blocks = assigner.assign_tags(passages)
        self.assertEqual(blocks[0].tag, Tag.NOMENCLATURE)  # Species A
        self.assertEqual(blocks[1].tag, Tag.DESCRIPTION)  # Description.
        self.assertEqual(blocks[2].tag, Tag.DESCRIPTION)  # Descr text
        self.assertEqual(blocks[3].tag, Tag.NOMENCLATURE)  # Species B
        self.assertEqual(blocks[4].tag, Tag.NOMENCLATURE)  # Preamble

    def test_leaving_results_resets_state(self) -> None:
        passages = [
            _make_passage(
                "RESULTS", "title_3", "Species name sp. nov."
            ),
            _make_passage("RESULTS", "title_4", "Description."),
            _make_passage("RESULTS", "paragraph", "Desc text."),
            # Discussion section resets state
            _make_passage("DISCUSS", "title_1", "Discussion"),
            _make_passage("DISCUSS", "paragraph", "Discussion text."),
        ]
        assigner = BiocTagAssigner()
        blocks = assigner.assign_tags(passages)
        self.assertEqual(blocks[3].tag, Tag.MISC_EXPOSITION)
        self.assertEqual(blocks[4].tag, Tag.MISC_EXPOSITION)

    def test_subsection_changes_on_new_title_4(self) -> None:
        passages = [
            _make_passage(
                "RESULTS", "title_3", "Species name sp. nov."
            ),
            _make_passage("RESULTS", "title_4", "Description."),
            _make_passage("RESULTS", "paragraph", "Desc text."),
            _make_passage("RESULTS", "title_4", "Notes."),
            _make_passage("RESULTS", "paragraph", "Notes text."),
        ]
        assigner = BiocTagAssigner()
        blocks = assigner.assign_tags(passages)
        self.assertEqual(blocks[2].tag, Tag.DESCRIPTION)
        self.assertEqual(blocks[3].tag, Tag.NOTES)
        self.assertEqual(blocks[4].tag, Tag.NOTES)


class TestFigInterleaving(unittest.TestCase):
    """FIG passages do not reset subsection state."""

    def test_fig_between_description_paragraphs(self) -> None:
        passages = [
            _make_passage(
                "RESULTS", "title_3", "Species name sp. nov."
            ),
            _make_passage("RESULTS", "title_4", "Description."),
            _make_passage("RESULTS", "paragraph", "Part 1 of desc."),
            _make_passage("FIG", "fig_caption", "Figure 1. Conidiophores."),
            _make_passage("RESULTS", "paragraph", "Part 2 of desc."),
        ]
        assigner = BiocTagAssigner()
        blocks = assigner.assign_tags(passages)
        self.assertEqual(blocks[2].tag, Tag.DESCRIPTION)  # Part 1
        self.assertEqual(blocks[3].tag, Tag.FIGURE_CAPTION)  # Fig
        self.assertEqual(blocks[4].tag, Tag.DESCRIPTION)  # Part 2

    def test_fig_between_nomenclature_preamble(self) -> None:
        passages = [
            _make_passage(
                "RESULTS", "title_3", "Species name sp. nov."
            ),
            _make_passage("RESULTS", "paragraph", "MycoBank No: 123"),
            _make_passage("FIG", "fig_caption", "Figure 2. Habitat."),
            _make_passage("RESULTS", "paragraph", "Fig. 2"),
        ]
        assigner = BiocTagAssigner()
        blocks = assigner.assign_tags(passages)
        self.assertEqual(blocks[1].tag, Tag.NOMENCLATURE)
        self.assertEqual(blocks[2].tag, Tag.FIGURE_CAPTION)
        self.assertEqual(blocks[3].tag, Tag.NOMENCLATURE)


class TestKeyDetection(unittest.TestCase):
    """Dichotomous key detection from title_2 and title_3 headings."""

    def test_key_in_title_2(self) -> None:
        passages = [
            _make_passage(
                "RESULTS",
                "title_2",
                "Key to species of Kirschsteiniothelia",
            ),
            _make_passage("RESULTS", "paragraph", "1a. Conidia large..."),
            _make_passage("RESULTS", "paragraph", "1b. Conidia small..."),
        ]
        assigner = BiocTagAssigner()
        blocks = assigner.assign_tags(passages)
        self.assertEqual(blocks[0].tag, Tag.KEY)
        self.assertEqual(blocks[1].tag, Tag.KEY)
        self.assertEqual(blocks[2].tag, Tag.KEY)

    def test_key_in_title_3(self) -> None:
        passages = [
            _make_passage(
                "RESULTS",
                "title_3",
                "Key to the known species of Endocalyx",
            ),
            _make_passage("RESULTS", "paragraph", "1. Spores smooth..."),
        ]
        assigner = BiocTagAssigner()
        blocks = assigner.assign_tags(passages)
        self.assertEqual(blocks[0].tag, Tag.KEY)
        self.assertEqual(blocks[1].tag, Tag.KEY)

    def test_key_ends_at_next_title_3(self) -> None:
        passages = [
            _make_passage(
                "RESULTS",
                "title_2",
                "Key to species of Genus",
            ),
            _make_passage("RESULTS", "paragraph", "Key couplet 1..."),
            _make_passage(
                "RESULTS",
                "title_3",
                "Genus newspecies sp. nov.",
            ),
            _make_passage("RESULTS", "paragraph", "Preamble."),
        ]
        assigner = BiocTagAssigner()
        blocks = assigner.assign_tags(passages)
        self.assertEqual(blocks[0].tag, Tag.KEY)
        self.assertEqual(blocks[1].tag, Tag.KEY)
        self.assertEqual(blocks[2].tag, Tag.NOMENCLATURE)
        self.assertEqual(blocks[3].tag, Tag.NOMENCLATURE)

    def test_non_key_title_2_is_misc(self) -> None:
        passages = [
            _make_passage("RESULTS", "title_2", "Phylogenetic analyses"),
            _make_passage("RESULTS", "paragraph", "Analysis text."),
        ]
        assigner = BiocTagAssigner()
        blocks = assigner.assign_tags(passages)
        self.assertEqual(blocks[0].tag, Tag.MISC_EXPOSITION)
        self.assertEqual(blocks[1].tag, Tag.MISC_EXPOSITION)


class TestYeddaOutput(unittest.TestCase):
    """Correct YEDDA format rendering."""

    def test_single_block(self) -> None:
        blocks = [TaggedBlock(text="Hello world", tag=Tag.MISC_EXPOSITION)]
        result = tagged_blocks_to_yedda(blocks)
        self.assertEqual(result, "[@Hello world#Misc-exposition*]\n")

    def test_multiple_blocks_separated_by_blank_lines(self) -> None:
        blocks = [
            TaggedBlock(text="Title", tag=Tag.MISC_EXPOSITION),
            TaggedBlock(text="Species name sp. nov.", tag=Tag.NOMENCLATURE),
        ]
        result = tagged_blocks_to_yedda(blocks)
        expected = (
            "[@Title#Misc-exposition*]\n"
            "\n"
            "[@Species name sp. nov.#Nomenclature*]\n"
        )
        self.assertEqual(result, expected)

    def test_multiline_passage(self) -> None:
        blocks = [
            TaggedBlock(
                text="Line 1\nLine 2\nLine 3",
                tag=Tag.DESCRIPTION,
            )
        ]
        result = tagged_blocks_to_yedda(blocks)
        expected = "[@Line 1\nLine 2\nLine 3#Description*]\n"
        self.assertEqual(result, expected)

    def test_empty_blocks_list(self) -> None:
        result = tagged_blocks_to_yedda([])
        self.assertEqual(result, "\n")


class TestEmptyPassages(unittest.TestCase):
    """Empty or whitespace-only passages are omitted."""

    def test_empty_text_skipped(self) -> None:
        passages = [
            _make_passage("INTRO", "paragraph", ""),
            _make_passage("INTRO", "paragraph", "Real text."),
        ]
        assigner = BiocTagAssigner()
        blocks = assigner.assign_tags(passages)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "Real text.")

    def test_bom_only_text_skipped(self) -> None:
        passages = [
            _make_passage("INTRO", "paragraph", "\ufeff  "),
            _make_passage("INTRO", "paragraph", "Real text."),
        ]
        assigner = BiocTagAssigner()
        blocks = assigner.assign_tags(passages)
        self.assertEqual(len(blocks), 1)


class TestBiocJsonToTaggedBlocks(unittest.TestCase):
    """Integration: bioc_json_to_tagged_blocks with nested structure."""

    def test_extracts_passages_from_nested_structure(self) -> None:
        bioc_json = _make_bioc_json([
            _make_passage("TITLE", "front", "\ufeffArticle Title"),
            _make_passage("ABSTRACT", "abstract", "Abstract text."),
        ])
        blocks = bioc_json_to_tagged_blocks(bioc_json)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].text, "Article Title")  # BOM stripped
        self.assertEqual(blocks[0].tag, Tag.MISC_EXPOSITION)

    def test_empty_bioc_json_raises(self) -> None:
        with self.assertRaises(ValueError):
            bioc_json_to_tagged_blocks([])

    def test_no_documents_raises(self) -> None:
        bioc_json = [{"bioctype": "BioCCollection", "documents": []}]
        with self.assertRaises(ValueError):
            bioc_json_to_tagged_blocks(bioc_json)

    def test_no_passages_raises(self) -> None:
        bioc_json = [
            {
                "bioctype": "BioCCollection",
                "documents": [{"passages": []}],
            }
        ]
        with self.assertRaises(ValueError):
            bioc_json_to_tagged_blocks(bioc_json)


class TestEndToEnd(unittest.TestCase):
    """Realistic multi-passage fixture mimicking a taxonomy paper."""

    def test_full_article_structure(self) -> None:
        passages = [
            _make_passage("TITLE", "front", "\ufeffNovel species of Genus"),
            _make_passage("ABSTRACT", "abstract_title_1", "\ufeffAbstract"),
            _make_passage(
                "ABSTRACT", "abstract", "Three new species are described."
            ),
            _make_passage("INTRO", "title_1", "\ufeffIntroduction"),
            _make_passage(
                "INTRO", "paragraph", "The genus Genus was established..."
            ),
            _make_passage("METHODS", "title_1", "Material and methods"),
            _make_passage("METHODS", "paragraph", "Specimens were collected."),
            _make_passage("RESULTS", "title_1", "Results"),
            _make_passage("RESULTS", "title_2", "Taxonomy"),
            _make_passage(
                "RESULTS",
                "title_3",
                "Genus newspecies A.Author & B.Author, sp. nov.",
            ),
            _make_passage("RESULTS", "paragraph", "MycoBank No: MB 900001"),
            _make_passage("RESULTS", "paragraph", "Fig. 1"),
            _make_passage("RESULTS", "title_4", "Etymology."),
            _make_passage(
                "RESULTS",
                "paragraph",
                "Named for the type locality.",
            ),
            _make_passage("RESULTS", "title_4", "Holotype."),
            _make_passage(
                "RESULTS",
                "paragraph",
                "THAILAND, Chiang Mai Province...",
            ),
            _make_passage("RESULTS", "title_4", "Description."),
            _make_passage(
                "RESULTS",
                "paragraph",
                "Saprobic on decaying wood. Colonies dark brown.",
            ),
            _make_passage(
                "FIG",
                "fig_caption",
                "Figure 1. Genus newspecies. a Colonies b Conidiophores.",
            ),
            _make_passage(
                "RESULTS",
                "paragraph",
                "Conidia 10-15 x 5-8 \u00b5m, oblong to ellipsoidal.",
            ),
            _make_passage("RESULTS", "title_4", "Culture characteristics."),
            _make_passage(
                "RESULTS",
                "paragraph",
                "Colonies on PDA reaching 30 mm after 4 weeks.",
            ),
            _make_passage("RESULTS", "title_4", "Material examined."),
            _make_passage(
                "RESULTS",
                "paragraph",
                "THAILAND, specimen MFLU 22-0001.",
            ),
            _make_passage("RESULTS", "title_4", "Notes."),
            _make_passage(
                "RESULTS",
                "paragraph",
                "Genus newspecies differs from G. oldspecies...",
            ),
            _make_passage("DISCUSS", "title_1", "Discussion"),
            _make_passage(
                "DISCUSS", "paragraph", "Our results show that..."
            ),
            _make_passage("REF", "title", "References"),
            _make_passage(
                "REF",
                "ref",
                "Author A (2020) Some paper. Mycologia 112: 1-10.",
            ),
        ]
        bioc_json = _make_bioc_json(passages)
        blocks = bioc_json_to_tagged_blocks(bioc_json)

        expected_tags = [
            Tag.MISC_EXPOSITION,   # Title
            Tag.MISC_EXPOSITION,   # Abstract heading
            Tag.MISC_EXPOSITION,   # Abstract text
            Tag.MISC_EXPOSITION,   # Introduction
            Tag.MISC_EXPOSITION,   # Intro paragraph
            Tag.MISC_EXPOSITION,   # Methods heading
            Tag.MISC_EXPOSITION,   # Methods paragraph
            Tag.MISC_EXPOSITION,   # Results heading
            Tag.MISC_EXPOSITION,   # Taxonomy heading
            Tag.NOMENCLATURE,      # Species heading
            Tag.NOMENCLATURE,      # MycoBank
            Tag.NOMENCLATURE,      # Fig ref
            Tag.ETYMOLOGY,         # Etymology heading
            Tag.ETYMOLOGY,         # Etymology text
            Tag.HOLOTYPE,          # Holotype heading
            Tag.HOLOTYPE,          # Holotype text
            Tag.DESCRIPTION,       # Description heading
            Tag.DESCRIPTION,       # Description text
            Tag.FIGURE_CAPTION,    # Figure caption
            Tag.DESCRIPTION,       # More description (state preserved)
            Tag.DESCRIPTION,       # Culture characteristics heading
            Tag.DESCRIPTION,       # Culture characteristics text
            Tag.MISC_EXPOSITION,   # Material examined heading
            Tag.MISC_EXPOSITION,   # Material examined text
            Tag.NOTES,             # Notes heading
            Tag.NOTES,             # Notes text
            Tag.MISC_EXPOSITION,   # Discussion heading
            Tag.MISC_EXPOSITION,   # Discussion text
            Tag.MISC_EXPOSITION,   # References heading
            Tag.MISC_EXPOSITION,   # Reference entry
        ]

        self.assertEqual(len(blocks), len(expected_tags))
        for i, (block, expected) in enumerate(
            zip(blocks, expected_tags)
        ):
            self.assertEqual(
                block.tag,
                expected,
                f"Block {i} ({block.text[:40]}...): "
                f"got {block.tag}, expected {expected}",
            )

    def test_full_yedda_output_not_empty(self) -> None:
        passages = [
            _make_passage("TITLE", "front", "Title"),
            _make_passage("RESULTS", "title_3", "Species sp. nov."),
        ]
        bioc_json = _make_bioc_json(passages)
        yedda = bioc_json_to_yedda(bioc_json)
        self.assertIn("[@Title#Misc-exposition*]", yedda)
        self.assertIn("[@Species sp. nov.#Nomenclature*]", yedda)


class TestRoundTrip(unittest.TestCase):
    """Output parses correctly with yedda_parser."""

    def test_labels_survive_round_trip(self) -> None:
        # Import the existing YEDDA parser
        try:
            from yedda_parser import parse_yedda_string
        except ImportError:
            import sys
            from pathlib import Path

            root = str(Path(__file__).resolve().parent.parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            from yedda_parser import parse_yedda_string

        passages = [
            _make_passage("TITLE", "front", "Article Title"),
            _make_passage(
                "RESULTS",
                "title_3",
                "Species name Author, sp. nov.",
            ),
            _make_passage("RESULTS", "title_4", "Description."),
            _make_passage("RESULTS", "paragraph", "Conidia brown."),
            _make_passage("RESULTS", "title_4", "Etymology."),
            _make_passage(
                "RESULTS", "paragraph", "Named after the locality."
            ),
            _make_passage("RESULTS", "title_4", "Holotype."),
            _make_passage("RESULTS", "paragraph", "MFLU 22-0001."),
            _make_passage("RESULTS", "title_4", "Notes."),
            _make_passage("RESULTS", "paragraph", "Differs from..."),
            _make_passage(
                "FIG", "fig_caption", "Figure 1. Morphology."
            ),
        ]
        bioc_json = _make_bioc_json(passages)
        yedda = bioc_json_to_yedda(bioc_json)

        parsed = parse_yedda_string(yedda)
        # Extract unique labels
        labels = {label for label, _, _ in parsed}
        expected_labels = {
            "Misc-exposition",
            "Nomenclature",
            "Description",
            "Etymology",
            "Holotype",
            "Notes",
            "Figure-caption",
        }
        self.assertEqual(labels, expected_labels)


if __name__ == "__main__":
    unittest.main()
