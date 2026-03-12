"""Tests for JATS/TaxPub XML to YEDDA translator."""

import unittest
import xml.etree.ElementTree as ET

from .bioc_to_yedda import Tag
from .jats_to_yedda import (
    extract_fig_blocks,
    extract_text,
    jats_xml_to_tagged_blocks,
    jats_xml_to_yedda,
    process_key_section,
    process_treatment,
    sec_type_to_tag,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

TP_NS = "http://www.plazi.org/taxpub"


def _wrap_article(body_xml: str, abstract_xml: str = "", back_xml: str = "") -> str:
    """Build a minimal JATS article XML string."""
    abstract_part = ""
    if abstract_xml:
        abstract_part = f"<abstract>{abstract_xml}</abstract>"
    back_part = ""
    if back_xml:
        back_part = f"<back>{back_xml}</back>"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<article xmlns:tp="{TP_NS}">'
        f"<front><article-meta>{abstract_part}</article-meta></front>"
        f"<body>{body_xml}</body>"
        f"{back_part}"
        "</article>"
    )


def _make_treatment(
    nomenclature_xml: str = "",
    sections_xml: str = "",
) -> str:
    """Build a tp:taxon-treatment XML fragment."""
    return (
        "<tp:taxon-treatment>"
        f"<tp:nomenclature>{nomenclature_xml}</tp:nomenclature>"
        f"{sections_xml}"
        "</tp:taxon-treatment>"
    )


def _make_treatment_sec(sec_type: str, content: str) -> str:
    """Build a tp:treatment-sec XML fragment."""
    return (
        f'<tp:treatment-sec sec-type="{sec_type}">'
        f"<title>{sec_type.title()}.</title>"
        f"<p>{content}</p>"
        "</tp:treatment-sec>"
    )


def _make_fig(label: str, caption: str) -> str:
    """Build a <fig> XML fragment."""
    return (
        '<fig id="F1" position="float">'
        f"<label>{label}</label>"
        f"<caption><p>{caption}</p></caption>"
        "</fig>"
    )


# ---------------------------------------------------------------------------
# Tests: extract_text
# ---------------------------------------------------------------------------

class TestExtractText(unittest.TestCase):
    """Test XML text extraction with tag skipping."""

    def test_plain_text(self):
        elem = ET.fromstring("<p>Hello world</p>")
        self.assertEqual(extract_text(elem), "Hello world")

    def test_inline_markup_preserved(self):
        elem = ET.fromstring("<p>A <italic>bold</italic> word</p>")
        self.assertEqual(extract_text(elem), "A bold word")

    def test_skip_object_id(self):
        elem = ET.fromstring(
            "<nom>"
            "<object-id>UUID-123</object-id>"
            "<name>Fungus</name>"
            " author"
            "</nom>"
        )
        self.assertEqual(extract_text(elem, {"object-id"}), "Fungus author")

    def test_skip_fig(self):
        elem = ET.fromstring(
            "<sec>"
            "<title>Description.</title>"
            "<p>Some text.</p>"
            '<fig id="F1"><label>Figure 1.</label>'
            "<caption><p>A photo.</p></caption></fig>"
            "</sec>"
        )
        text = extract_text(elem, {"fig"})
        self.assertIn("Description.", text)
        self.assertIn("Some text.", text)
        self.assertNotIn("Figure", text)
        self.assertNotIn("photo", text)

    def test_nested_taxon_name(self):
        xml = (
            f'<p xmlns:tp="{TP_NS}">'
            "<tp:taxon-name>"
            '<tp:taxon-name-part taxon-name-part-type="genus">Sidera</tp:taxon-name-part> '
            '<tp:taxon-name-part taxon-name-part-type="species">parallela</tp:taxon-name-part>'
            "</tp:taxon-name>"
            " is a species."
            "</p>"
        )
        elem = ET.fromstring(xml)
        text = extract_text(elem)
        self.assertIn("Sidera", text)
        self.assertIn("parallela", text)
        self.assertIn("is a species", text)

    def test_empty_element(self):
        elem = ET.fromstring("<p></p>")
        self.assertEqual(extract_text(elem), "")

    def test_tail_text_preserved(self):
        elem = ET.fromstring("<p>Before <b>bold</b> after</p>")
        self.assertEqual(extract_text(elem), "Before bold after")


# ---------------------------------------------------------------------------
# Tests: extract_fig_blocks
# ---------------------------------------------------------------------------

class TestExtractFigBlocks(unittest.TestCase):
    """Test figure caption extraction."""

    def test_single_fig(self):
        xml = (
            "<sec>"
            '<fig id="F1"><label>Figure 1.</label>'
            "<caption><p>A basidioma photo.</p></caption></fig>"
            "</sec>"
        )
        elem = ET.fromstring(xml)
        blocks = extract_fig_blocks(elem)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tag, Tag.FIGURE_CAPTION)
        self.assertIn("Figure 1.", blocks[0].text)
        self.assertIn("basidioma photo", blocks[0].text)

    def test_multiple_figs(self):
        xml = (
            "<sec>"
            '<fig id="F1"><label>Figure 1.</label>'
            "<caption><p>First.</p></caption></fig>"
            '<fig id="F2"><label>Figure 2.</label>'
            "<caption><p>Second.</p></caption></fig>"
            "</sec>"
        )
        elem = ET.fromstring(xml)
        blocks = extract_fig_blocks(elem)
        self.assertEqual(len(blocks), 2)
        self.assertIn("First", blocks[0].text)
        self.assertIn("Second", blocks[1].text)

    def test_fig_without_caption(self):
        xml = (
            "<sec>"
            '<fig id="F1"><label>Figure 1.</label></fig>'
            "</sec>"
        )
        elem = ET.fromstring(xml)
        blocks = extract_fig_blocks(elem)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "Figure 1.")

    def test_no_figs(self):
        elem = ET.fromstring("<sec><p>No figures here.</p></sec>")
        blocks = extract_fig_blocks(elem)
        self.assertEqual(len(blocks), 0)

    def test_fig_with_object_id_filtered(self):
        xml = (
            "<sec>"
            '<fig id="F1">'
            '<object-id content-type="doi">10.1234/fig1</object-id>'
            "<label>Figure 1.</label>"
            "<caption><p>Caption text.</p></caption>"
            "</fig>"
            "</sec>"
        )
        elem = ET.fromstring(xml)
        blocks = extract_fig_blocks(elem)
        self.assertEqual(len(blocks), 1)
        self.assertNotIn("10.1234", blocks[0].text)
        self.assertIn("Figure 1.", blocks[0].text)


# ---------------------------------------------------------------------------
# Tests: sec_type_to_tag
# ---------------------------------------------------------------------------

class TestSecTypeToTag(unittest.TestCase):
    """Test treatment-sec sec-type to Tag mapping."""

    def test_description(self):
        self.assertEqual(sec_type_to_tag("description"), Tag.DESCRIPTION)

    def test_diagnosis(self):
        self.assertEqual(sec_type_to_tag("diagnosis"), Tag.DESCRIPTION)

    def test_etymology(self):
        self.assertEqual(sec_type_to_tag("etymology"), Tag.ETYMOLOGY)

    def test_holotype(self):
        self.assertEqual(sec_type_to_tag("Holotype"), Tag.HOLOTYPE)

    def test_material(self):
        self.assertEqual(sec_type_to_tag("material"), Tag.HOLOTYPE)

    def test_type_material(self):
        self.assertEqual(sec_type_to_tag("type material"), Tag.HOLOTYPE)

    def test_type_species(self):
        self.assertEqual(sec_type_to_tag("type species"), Tag.HOLOTYPE)

    def test_type_genus(self):
        self.assertEqual(sec_type_to_tag("type genus"), Tag.HOLOTYPE)

    def test_notes(self):
        self.assertEqual(sec_type_to_tag("notes"), Tag.NOTES)

    def test_comments(self):
        self.assertEqual(sec_type_to_tag("comments"), Tag.NOTES)

    def test_key(self):
        self.assertEqual(sec_type_to_tag("key"), Tag.KEY)

    def test_key_to(self):
        self.assertEqual(sec_type_to_tag("Key to Sidera"), Tag.KEY)

    def test_morphological_fruiting_body(self):
        self.assertEqual(sec_type_to_tag("Fruiting body"), Tag.DESCRIPTION)

    def test_morphological_basidiospores(self):
        self.assertEqual(sec_type_to_tag("Basidiospores"), Tag.DESCRIPTION)

    def test_morphological_hyphal_system(self):
        self.assertEqual(sec_type_to_tag("Hyphal system"), Tag.DESCRIPTION)

    def test_morphological_hymenial_layer(self):
        self.assertEqual(sec_type_to_tag("Hymenial layer"), Tag.DESCRIPTION)

    def test_culture_characteristics(self):
        self.assertEqual(sec_type_to_tag("culture characteristics"), Tag.DESCRIPTION)

    def test_additional_specimen_examined(self):
        self.assertEqual(
            sec_type_to_tag("Additional specimen examined"), Tag.MISC_EXPOSITION
        )

    def test_unknown_type(self):
        self.assertEqual(sec_type_to_tag("something else"), Tag.MISC_EXPOSITION)

    def test_case_insensitive(self):
        self.assertEqual(sec_type_to_tag("DESCRIPTION"), Tag.DESCRIPTION)
        self.assertEqual(sec_type_to_tag("Etymology"), Tag.ETYMOLOGY)

    def test_bom_stripped(self):
        self.assertEqual(sec_type_to_tag("\ufeffdescription"), Tag.DESCRIPTION)


# ---------------------------------------------------------------------------
# Tests: process_treatment
# ---------------------------------------------------------------------------

class TestNomenclatureExtraction(unittest.TestCase):
    """Test nomenclature text extraction from treatments."""

    def test_basic_nomenclature(self):
        xml = (
            f'<tp:taxon-treatment xmlns:tp="{TP_NS}">'
            "<tp:nomenclature>"
            "<tp:taxon-name>"
            '<tp:taxon-name-part taxon-name-part-type="genus">Sidera</tp:taxon-name-part> '
            '<tp:taxon-name-part taxon-name-part-type="species">parallela</tp:taxon-name-part>'
            "</tp:taxon-name>"
            "<tp:taxon-authority>Dai &amp; Wu</tp:taxon-authority>"
            "<tp:taxon-status>sp. nov.</tp:taxon-status>"
            "</tp:nomenclature>"
            "</tp:taxon-treatment>"
        )
        elem = ET.fromstring(xml)
        blocks = process_treatment(elem)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tag, Tag.NOMENCLATURE)
        self.assertIn("Sidera", blocks[0].text)
        self.assertIn("parallela", blocks[0].text)
        self.assertIn("Dai & Wu", blocks[0].text)
        self.assertIn("sp. nov.", blocks[0].text)

    def test_nomenclature_skips_object_id(self):
        xml = (
            f'<tp:taxon-treatment xmlns:tp="{TP_NS}">'
            "<tp:nomenclature>"
            "<tp:taxon-name>"
            '<object-id content-type="arpha">UUID-HERE</object-id>'
            '<tp:taxon-name-part taxon-name-part-type="genus">Fungus</tp:taxon-name-part>'
            '<object-id content-type="mycobank">12345</object-id>'
            "</tp:taxon-name>"
            "</tp:nomenclature>"
            "</tp:taxon-treatment>"
        )
        elem = ET.fromstring(xml)
        blocks = process_treatment(elem)
        self.assertEqual(len(blocks), 1)
        self.assertNotIn("UUID", blocks[0].text)
        self.assertNotIn("12345", blocks[0].text)
        self.assertIn("Fungus", blocks[0].text)


class TestTreatmentProcessing(unittest.TestCase):
    """Test full treatment processing with multiple sections."""

    def _build_treatment(self, sections: str) -> ET.Element:
        xml = (
            f'<tp:taxon-treatment xmlns:tp="{TP_NS}">'
            "<tp:nomenclature>"
            "<tp:taxon-name>"
            '<tp:taxon-name-part taxon-name-part-type="genus">Genus</tp:taxon-name-part> '
            '<tp:taxon-name-part taxon-name-part-type="species">species</tp:taxon-name-part>'
            "</tp:taxon-name>"
            "<tp:taxon-authority>Author</tp:taxon-authority>"
            "<tp:taxon-status>sp. nov.</tp:taxon-status>"
            "</tp:nomenclature>"
            f"{sections}"
            "</tp:taxon-treatment>"
        )
        return ET.fromstring(xml)

    def test_full_treatment(self):
        sections = (
            '<tp:treatment-sec sec-type="diagnosis">'
            "<title>Diagnosis.</title>"
            "<p>Differs from G. other by smaller spores.</p>"
            "</tp:treatment-sec>"
            '<tp:treatment-sec sec-type="Holotype">'
            "<title>Holotype.</title>"
            "<p>China, Yunnan, on dead wood.</p>"
            "</tp:treatment-sec>"
            '<tp:treatment-sec sec-type="etymology">'
            "<title>Etymology.</title>"
            "<p>Named after the type locality.</p>"
            "</tp:treatment-sec>"
            '<tp:treatment-sec sec-type="description">'
            "<title>Description.</title>"
            "<p>Basidiomata annual, resupinate.</p>"
            "</tp:treatment-sec>"
            '<tp:treatment-sec sec-type="notes">'
            "<title>Notes.</title>"
            "<p>This species is similar to G. other.</p>"
            "</tp:treatment-sec>"
        )
        elem = self._build_treatment(sections)
        blocks = process_treatment(elem)

        tags = [b.tag for b in blocks]
        self.assertEqual(tags[0], Tag.NOMENCLATURE)
        self.assertEqual(tags[1], Tag.DESCRIPTION)  # diagnosis
        self.assertEqual(tags[2], Tag.HOLOTYPE)
        self.assertEqual(tags[3], Tag.ETYMOLOGY)
        self.assertEqual(tags[4], Tag.DESCRIPTION)
        self.assertEqual(tags[5], Tag.NOTES)

    def test_treatment_with_fig(self):
        sections = (
            '<tp:treatment-sec sec-type="description">'
            "<title>Description.</title>"
            "<p>Basidiomata annual.</p>"
            '<fig id="F1"><label>Figure 1.</label>'
            "<caption><p>Basidioma photo.</p></caption></fig>"
            "</tp:treatment-sec>"
        )
        elem = self._build_treatment(sections)
        blocks = process_treatment(elem)

        # Nomenclature + Description + Figure-caption
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0].tag, Tag.NOMENCLATURE)
        self.assertEqual(blocks[1].tag, Tag.DESCRIPTION)
        self.assertNotIn("Figure", blocks[1].text)
        self.assertNotIn("photo", blocks[1].text)
        self.assertEqual(blocks[2].tag, Tag.FIGURE_CAPTION)
        self.assertIn("Figure 1.", blocks[2].text)
        self.assertIn("Basidioma photo", blocks[2].text)

    def test_treatment_skips_treatment_meta(self):
        xml = (
            f'<tp:taxon-treatment xmlns:tp="{TP_NS}">'
            "<tp:treatment-meta>"
            "<kwd-group><label>Taxon</label></kwd-group>"
            "</tp:treatment-meta>"
            "<tp:nomenclature>"
            "<tp:taxon-name>"
            '<tp:taxon-name-part taxon-name-part-type="genus">Genus</tp:taxon-name-part>'
            "</tp:taxon-name>"
            "</tp:nomenclature>"
            "</tp:taxon-treatment>"
        )
        elem = ET.fromstring(xml)
        blocks = process_treatment(elem)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tag, Tag.NOMENCLATURE)
        self.assertNotIn("Taxon", blocks[0].text)

    def test_morphological_subsections(self):
        sections = ""
        for sec_type in ["Fruiting body", "Hyphal system", "Basidiospores"]:
            sections += (
                f'<tp:treatment-sec sec-type="{sec_type}">'
                f"<title>{sec_type}.</title>"
                f"<p>Details about {sec_type.lower()}.</p>"
                "</tp:treatment-sec>"
            )
        elem = self._build_treatment(sections)
        blocks = process_treatment(elem)
        # Nomenclature + 3 Description blocks
        self.assertEqual(len(blocks), 4)
        for block in blocks[1:]:
            self.assertEqual(block.tag, Tag.DESCRIPTION)


# ---------------------------------------------------------------------------
# Tests: process_key_section
# ---------------------------------------------------------------------------

class TestKeySection(unittest.TestCase):
    """Test key section processing."""

    def test_table_key(self):
        xml = (
            f'<sec xmlns:tp="{TP_NS}" sec-type="Key to species of Sidera">'
            "<title>Key to species of Sidera</title>"
            '<table-wrap content-type="key">'
            "<table><tbody>"
            "<tr><td>1</td><td>Hymenium poroid</td><td>2</td></tr>"
            "<tr><td>-</td><td>Hymenium smooth</td><td>S. lunata</td></tr>"
            "</tbody></table>"
            "</table-wrap>"
            "</sec>"
        )
        elem = ET.fromstring(xml)
        blocks = process_key_section(elem)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tag, Tag.KEY)
        self.assertIn("Hymenium poroid", blocks[0].text)
        self.assertIn("Key to species", blocks[0].text)

    def test_paragraph_key(self):
        xml = (
            f'<sec xmlns:tp="{TP_NS}" sec-type="key">'
            "<title>Key to species</title>"
            "<p>1a. Spores smooth ... 2</p>"
            "<p>1b. Spores ornamented ... S. rugosa</p>"
            "</sec>"
        )
        elem = ET.fromstring(xml)
        blocks = process_key_section(elem)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tag, Tag.KEY)
        self.assertIn("Spores smooth", blocks[0].text)


# ---------------------------------------------------------------------------
# Tests: jats_xml_to_tagged_blocks (full article)
# ---------------------------------------------------------------------------

class TestNonTreatmentSections(unittest.TestCase):
    """Test that non-treatment sections map to Misc-exposition."""

    def test_introduction(self):
        body = '<sec sec-type="Introduction"><title>Introduction</title><p>Background text.</p></sec>'
        xml = _wrap_article(body)
        blocks = jats_xml_to_tagged_blocks(xml)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tag, Tag.MISC_EXPOSITION)
        self.assertIn("Background text", blocks[0].text)

    def test_abstract(self):
        xml = _wrap_article(
            '<sec sec-type="Discussion"><title>Discussion</title><p>Disc.</p></sec>',
            abstract_xml="<p>Abstract text here.</p>",
        )
        blocks = jats_xml_to_tagged_blocks(xml)
        # Abstract + Discussion
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].tag, Tag.MISC_EXPOSITION)
        self.assertIn("Abstract text", blocks[0].text)

    def test_back_matter(self):
        xml = _wrap_article(
            '<sec sec-type="Introduction"><title>Intro</title><p>Text.</p></sec>',
            back_xml='<ref-list><ref id="R1"><mixed-citation>Smith 2020.</mixed-citation></ref></ref-list>',
        )
        blocks = jats_xml_to_tagged_blocks(xml)
        # Intro + References
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[-1].tag, Tag.MISC_EXPOSITION)
        self.assertIn("Smith 2020", blocks[-1].text)


class TestFigSeparation(unittest.TestCase):
    """Test that figs in treatment sections are extracted separately."""

    def test_fig_not_in_description_text(self):
        treatment = _make_treatment(
            nomenclature_xml=(
                "<tp:taxon-name>"
                '<tp:taxon-name-part taxon-name-part-type="genus">Genus</tp:taxon-name-part>'
                "</tp:taxon-name>"
            ),
            sections_xml=(
                '<tp:treatment-sec sec-type="description">'
                "<title>Description.</title>"
                "<p>Basidiomata annual, resupinate.</p>"
                '<fig id="F1">'
                "<label>Figure 1.</label>"
                "<caption><p>Macro photo.</p></caption>"
                "</fig>"
                "</tp:treatment-sec>"
            ),
        )
        body = (
            f'<sec xmlns:tp="{TP_NS}" sec-type="Taxonomy">'
            "<title>Taxonomy</title>"
            f"{treatment}"
            "</sec>"
        )
        xml = _wrap_article(body)
        blocks = jats_xml_to_tagged_blocks(xml)
        desc_blocks = [b for b in blocks if b.tag == Tag.DESCRIPTION]
        fig_blocks = [b for b in blocks if b.tag == Tag.FIGURE_CAPTION]
        self.assertEqual(len(desc_blocks), 1)
        self.assertEqual(len(fig_blocks), 1)
        self.assertNotIn("Macro photo", desc_blocks[0].text)
        self.assertIn("Macro photo", fig_blocks[0].text)


class TestKeyDetection(unittest.TestCase):
    """Test key section detection at various levels."""

    def test_top_level_key_section(self):
        body = (
            f'<sec xmlns:tp="{TP_NS}" sec-type="Introduction">'
            "<title>Introduction</title><p>Intro text.</p></sec>"
            f'<sec xmlns:tp="{TP_NS}" sec-type="Key to species of Fungus">'
            "<title>Key to species of Fungus</title>"
            '<table-wrap content-type="key">'
            "<table><tbody>"
            "<tr><td>1</td><td>Large spores</td><td>F. major</td></tr>"
            "</tbody></table>"
            "</table-wrap>"
            "</sec>"
        )
        xml = _wrap_article(body)
        blocks = jats_xml_to_tagged_blocks(xml)
        key_blocks = [b for b in blocks if b.tag == Tag.KEY]
        self.assertEqual(len(key_blocks), 1)
        self.assertIn("Large spores", key_blocks[0].text)


class TestEndToEnd(unittest.TestCase):
    """Test a realistic multi-treatment article."""

    def _build_article(self) -> str:
        treatment1 = _make_treatment(
            nomenclature_xml=(
                "<tp:taxon-name>"
                '<object-id content-type="arpha">UUID-1</object-id>'
                '<tp:taxon-name-part taxon-name-part-type="genus">Sidera</tp:taxon-name-part> '
                '<tp:taxon-name-part taxon-name-part-type="species">parallela</tp:taxon-name-part>'
                '<object-id content-type="mycobank">829166</object-id>'
                "</tp:taxon-name>"
                "<tp:taxon-authority>Dai &amp; Wu</tp:taxon-authority>"
                "<tp:taxon-status>sp. nov.</tp:taxon-status>"
            ),
            sections_xml=(
                '<tp:treatment-sec sec-type="diagnosis">'
                "<title>Diagnosis.</title>"
                "<p>Differs from S. other.</p>"
                "</tp:treatment-sec>"
                '<tp:treatment-sec sec-type="Holotype">'
                "<title>Holotype.</title>"
                "<p>China, Yunnan.</p>"
                "</tp:treatment-sec>"
                '<tp:treatment-sec sec-type="etymology">'
                "<title>Etymology.</title>"
                "<p>From Latin parallela.</p>"
                '<fig id="F1"><label>Figure 1.</label>'
                "<caption><p>Basidioma.</p></caption></fig>"
                "</tp:treatment-sec>"
                '<tp:treatment-sec sec-type="Fruiting body">'
                "<title>Fruiting body.</title>"
                "<p>Annual, resupinate.</p>"
                "</tp:treatment-sec>"
                '<tp:treatment-sec sec-type="notes">'
                "<title>Notes.</title>"
                "<p>Similar to S. tenuis.</p>"
                "</tp:treatment-sec>"
            ),
        )
        treatment2 = _make_treatment(
            nomenclature_xml=(
                "<tp:taxon-name>"
                '<tp:taxon-name-part taxon-name-part-type="genus">Sidera</tp:taxon-name-part> '
                '<tp:taxon-name-part taxon-name-part-type="species">tenuis</tp:taxon-name-part>'
                "</tp:taxon-name>"
                "<tp:taxon-authority>Wu</tp:taxon-authority>"
                "<tp:taxon-status>sp. nov.</tp:taxon-status>"
            ),
            sections_xml=(
                '<tp:treatment-sec sec-type="description">'
                "<title>Description.</title>"
                "<p>Thin basidiomata.</p>"
                "</tp:treatment-sec>"
            ),
        )
        key_section = (
            '<sec sec-type="Key to species of Sidera">'
            "<title>Key to species of Sidera</title>"
            '<table-wrap content-type="key">'
            "<table><tbody>"
            "<tr><td>1</td><td>Pores large</td><td>S. parallela</td></tr>"
            "<tr><td>-</td><td>Pores small</td><td>S. tenuis</td></tr>"
            "</tbody></table>"
            "</table-wrap>"
            "</sec>"
        )
        body = (
            f'<sec xmlns:tp="{TP_NS}" sec-type="Introduction">'
            "<title>Introduction</title><p>We describe new species.</p></sec>"
            f'<sec xmlns:tp="{TP_NS}" sec-type="Taxonomy">'
            "<title>Taxonomy</title>"
            f"{treatment1}"
            f"{treatment2}"
            "</sec>"
            f'{key_section}'
            f'<sec xmlns:tp="{TP_NS}" sec-type="Discussion">'
            "<title>Discussion</title><p>These findings are important.</p></sec>"
        )
        return _wrap_article(
            body,
            abstract_xml="<p>Two new species of Sidera are described.</p>",
            back_xml='<ref-list><ref><mixed-citation>Smith 2020</mixed-citation></ref></ref-list>',
        )

    def test_tag_distribution(self):
        xml = self._build_article()
        blocks = jats_xml_to_tagged_blocks(xml)

        tag_counts = {}
        for b in blocks:
            tag_counts[b.tag] = tag_counts.get(b.tag, 0) + 1

        self.assertEqual(tag_counts[Tag.NOMENCLATURE], 2)
        self.assertEqual(tag_counts[Tag.DESCRIPTION], 3)  # diagnosis + Fruiting body + description
        self.assertEqual(tag_counts[Tag.ETYMOLOGY], 1)
        self.assertEqual(tag_counts[Tag.HOLOTYPE], 1)
        self.assertEqual(tag_counts[Tag.NOTES], 1)
        self.assertEqual(tag_counts[Tag.KEY], 1)
        self.assertEqual(tag_counts[Tag.FIGURE_CAPTION], 1)
        # Abstract + Introduction + Discussion + References
        self.assertEqual(tag_counts[Tag.MISC_EXPOSITION], 4)

    def test_nomenclature_clean(self):
        xml = self._build_article()
        blocks = jats_xml_to_tagged_blocks(xml)
        nom_blocks = [b for b in blocks if b.tag == Tag.NOMENCLATURE]
        # First nomenclature should not have UUIDs or MycoBank IDs
        self.assertNotIn("UUID", nom_blocks[0].text)
        self.assertNotIn("829166", nom_blocks[0].text)
        self.assertIn("Sidera", nom_blocks[0].text)
        self.assertIn("parallela", nom_blocks[0].text)
        self.assertIn("sp. nov.", nom_blocks[0].text)

    def test_document_order(self):
        xml = self._build_article()
        blocks = jats_xml_to_tagged_blocks(xml)
        tags = [b.tag for b in blocks]
        # Abstract comes first
        self.assertEqual(tags[0], Tag.MISC_EXPOSITION)
        # Introduction
        self.assertEqual(tags[1], Tag.MISC_EXPOSITION)
        # Treatment 1: Nomenclature, then various sections
        self.assertEqual(tags[2], Tag.NOMENCLATURE)
        # Key should come after treatments
        key_idx = tags.index(Tag.KEY)
        last_nom_idx = len(tags) - 1 - tags[::-1].index(Tag.NOMENCLATURE)
        self.assertGreater(key_idx, last_nom_idx)
        # References should be last
        self.assertEqual(tags[-1], Tag.MISC_EXPOSITION)


class TestRoundTrip(unittest.TestCase):
    """Test that output parses correctly through yedda_parser."""

    def test_round_trip(self):
        from yedda_parser import parse_yedda_string

        treatment = _make_treatment(
            nomenclature_xml=(
                "<tp:taxon-name>"
                '<tp:taxon-name-part taxon-name-part-type="genus">Fungus</tp:taxon-name-part> '
                '<tp:taxon-name-part taxon-name-part-type="species">novus</tp:taxon-name-part>'
                "</tp:taxon-name>"
                "<tp:taxon-authority>Author</tp:taxon-authority>"
                "<tp:taxon-status>sp. nov.</tp:taxon-status>"
            ),
            sections_xml=(
                '<tp:treatment-sec sec-type="description">'
                "<title>Description.</title>"
                "<p>Large basidiomata.</p>"
                "</tp:treatment-sec>"
            ),
        )
        body = (
            f'<sec xmlns:tp="{TP_NS}" sec-type="Taxonomy">'
            "<title>Taxonomy</title>"
            f"{treatment}"
            "</sec>"
        )
        xml = _wrap_article(body)
        yedda = jats_xml_to_yedda(xml)

        parsed = parse_yedda_string(yedda)
        labels = {label for label, _, _ in parsed}
        self.assertIn("Nomenclature", labels)
        self.assertIn("Description", labels)


class TestEmptyAndEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_no_body_raises(self):
        xml = (
            '<?xml version="1.0"?>'
            f'<article xmlns:tp="{TP_NS}">'
            "<front><article-meta></article-meta></front>"
            "</article>"
        )
        with self.assertRaises(ValueError):
            jats_xml_to_tagged_blocks(xml)

    def test_invalid_xml_raises(self):
        with self.assertRaises(ValueError):
            jats_xml_to_tagged_blocks("not xml at all")

    def test_empty_nomenclature(self):
        xml = (
            f'<tp:taxon-treatment xmlns:tp="{TP_NS}">'
            "<tp:nomenclature></tp:nomenclature>"
            "</tp:taxon-treatment>"
        )
        elem = ET.fromstring(xml)
        blocks = process_treatment(elem)
        self.assertEqual(len(blocks), 0)

    def test_treatment_with_empty_sections(self):
        xml = (
            f'<tp:taxon-treatment xmlns:tp="{TP_NS}">'
            "<tp:nomenclature>"
            "<tp:taxon-name>"
            '<tp:taxon-name-part taxon-name-part-type="genus">G</tp:taxon-name-part>'
            "</tp:taxon-name>"
            "</tp:nomenclature>"
            '<tp:treatment-sec sec-type="description">'
            "<title></title>"
            "<p></p>"
            "</tp:treatment-sec>"
            "</tp:taxon-treatment>"
        )
        elem = ET.fromstring(xml)
        blocks = process_treatment(elem)
        # Only nomenclature, empty description section skipped
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tag, Tag.NOMENCLATURE)


if __name__ == "__main__":
    unittest.main()
