"""Tests for JATS/TaxPub XML to YEDDA translator."""

import unittest
import xml.etree.ElementTree as ET

from .yedda_tags import ACTIVE_TAGS_19, DEPRECATED_TAGS, Tag
from .jats_to_yedda import (
    JATS_EMIT_TAGS,
    _has_treatments,
    extract_fig_blocks,
    extract_text,
    jats_xml_to_tagged_blocks,
    jats_xml_to_yedda,
    process_jats_treatment,
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
        self.assertEqual(sec_type_to_tag("diagnosis"), Tag.DIAGNOSIS)

    def test_etymology(self):
        self.assertEqual(sec_type_to_tag("etymology"), Tag.ETYMOLOGY)

    def test_holotype(self):
        self.assertEqual(sec_type_to_tag("Holotype"), Tag.TYPE_DESIGNATION)

    def test_material(self):
        self.assertEqual(sec_type_to_tag("material"), Tag.MATERIALS_EXAMINED)

    def test_type_material(self):
        self.assertEqual(sec_type_to_tag("type material"), Tag.TYPE_DESIGNATION)

    def test_type_species(self):
        self.assertEqual(sec_type_to_tag("type species"), Tag.TYPE_DESIGNATION)

    def test_type_genus(self):
        self.assertEqual(sec_type_to_tag("type genus"), Tag.TYPE_DESIGNATION)

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
            sec_type_to_tag("Additional specimen examined"), Tag.MATERIALS_EXAMINED
        )

    def test_distribution(self):
        # Tag.DISTRIBUTION was deprecated and folded into Tag.BIOLOGY (see
        # yedda_tags.py); the converter now maps all locality-style
        # sec-types to BIOLOGY.
        self.assertEqual(sec_type_to_tag("distribution"), Tag.BIOLOGY)

    def test_habitat(self):
        self.assertEqual(sec_type_to_tag("habitat"), Tag.BIOLOGY)

    def test_habitat_distribution(self):
        self.assertEqual(sec_type_to_tag("habitat-distribution"), Tag.BIOLOGY)

    def test_materials_examined(self):
        self.assertEqual(sec_type_to_tag("materials examined"), Tag.MATERIALS_EXAMINED)

    def test_specimens_examined(self):
        self.assertEqual(sec_type_to_tag("specimens examined"), Tag.MATERIALS_EXAMINED)

    def test_biology(self):
        self.assertEqual(sec_type_to_tag("biology"), Tag.BIOLOGY)

    def test_ecology(self):
        self.assertEqual(sec_type_to_tag("ecology"), Tag.BIOLOGY)

    def test_host(self):
        self.assertEqual(sec_type_to_tag("host"), Tag.BIOLOGY)

    def test_diagnosis_prefix_variant(self):
        self.assertEqual(sec_type_to_tag("diagnosis of the genus"), Tag.DIAGNOSIS)

    def test_bibliography(self):
        self.assertEqual(sec_type_to_tag("references"), Tag.BIBLIOGRAPHY)

    def test_bibliography_alias(self):
        self.assertEqual(sec_type_to_tag("bibliography"), Tag.BIBLIOGRAPHY)

    def test_literature_cited(self):
        self.assertEqual(sec_type_to_tag("literature cited"), Tag.BIBLIOGRAPHY)

    def test_table(self):
        self.assertEqual(sec_type_to_tag("table"), Tag.TABLE)

    def test_table_prefix_variant(self):
        self.assertEqual(sec_type_to_tag("table 1"), Tag.TABLE)

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
        self.assertEqual(tags[1], Tag.DIAGNOSIS)
        self.assertEqual(tags[2], Tag.TYPE_DESIGNATION)
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
        self.assertEqual(tag_counts[Tag.DESCRIPTION], 2)  # Fruiting body + description
        self.assertEqual(tag_counts[Tag.DIAGNOSIS], 1)
        self.assertEqual(tag_counts[Tag.ETYMOLOGY], 1)
        self.assertEqual(tag_counts[Tag.TYPE_DESIGNATION], 1)
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


# ---------------------------------------------------------------------------
# Tests: _has_treatments with JATS sec-type style
# ---------------------------------------------------------------------------

class TestHasTreatmentsJatsStyle(unittest.TestCase):
    """Test _has_treatments recognises <sec sec-type="taxon-treatment">."""

    def test_detects_jats_sec_type_treatment(self):
        xml = (
            '<sec sec-type="Taxonomy">'
            '<sec sec-type="taxon-treatment">'
            '<title>Fungus sp.</title>'
            '</sec>'
            '</sec>'
        )
        outer = ET.fromstring(xml)
        self.assertTrue(_has_treatments(outer))

    def test_negative_no_treatment(self):
        xml = '<sec sec-type="Introduction"><p>Text</p></sec>'
        sec = ET.fromstring(xml)
        self.assertFalse(_has_treatments(sec))

    def test_self_taxon_treatment_not_counted(self):
        """_has_treatments on the treatment element itself should be False."""
        xml = '<sec sec-type="taxon-treatment"><title>X</title></sec>'
        sec = ET.fromstring(xml)
        self.assertFalse(_has_treatments(sec))


# ---------------------------------------------------------------------------
# Tests: process_jats_treatment
# ---------------------------------------------------------------------------

_JATS_TREATMENT_XML = (
    '<sec sec-type="taxon-treatment">'
    '<title>Tomentella wumenshanensis C.L.Zhao</title>'
    '<sec sec-type="treatment-Holotype">'
    '<title>Holotype.</title>'
    '<p>China, Yunnan Province.</p>'
    '</sec>'
    '<sec sec-type="treatment-etymology">'
    '<title>Etymology.</title>'
    '<p>Named after the mountain.</p>'
    '</sec>'
    '<sec sec-type="treatment-description">'
    '<title>Description.</title>'
    '<p>Basidiomata annual, resupinate.</p>'
    '</sec>'
    '</sec>'
)


class TestProcessJatsTreatment(unittest.TestCase):
    """Tests for process_jats_treatment (JATS sec-type variant)."""

    def _blocks(self):
        elem = ET.fromstring(_JATS_TREATMENT_XML)
        return process_jats_treatment(elem)

    def test_first_block_is_nomenclature(self):
        blocks = self._blocks()
        self.assertEqual(blocks[0].tag, Tag.NOMENCLATURE)
        self.assertIn("Tomentella", blocks[0].text)

    def test_description_block(self):
        blocks = self._blocks()
        desc = [b for b in blocks if b.tag == Tag.DESCRIPTION]
        self.assertEqual(len(desc), 1)
        self.assertIn("Basidiomata", desc[0].text)

    def test_holotype_block(self):
        blocks = self._blocks()
        holo = [b for b in blocks if b.tag == Tag.TYPE_DESIGNATION]
        self.assertEqual(len(holo), 1)

    def test_etymology_block(self):
        blocks = self._blocks()
        etym = [b for b in blocks if b.tag == Tag.ETYMOLOGY]
        self.assertEqual(len(etym), 1)

    def test_document_order(self):
        blocks = self._blocks()
        self.assertEqual(blocks[0].tag, Tag.NOMENCLATURE)
        # Type-designation, Etymology, Description follow in XML order
        non_nom = [b.tag for b in blocks[1:]]
        self.assertEqual(
            non_nom, [Tag.TYPE_DESIGNATION, Tag.ETYMOLOGY, Tag.DESCRIPTION]
        )

    def test_no_title_no_nomenclature(self):
        xml = (
            '<sec sec-type="taxon-treatment">'
            '<sec sec-type="treatment-description">'
            '<p>Resupinate.</p>'
            '</sec>'
            '</sec>'
        )
        elem = ET.fromstring(xml)
        blocks = process_jats_treatment(elem)
        self.assertFalse(any(b.tag == Tag.NOMENCLATURE for b in blocks))
        self.assertTrue(any(b.tag == Tag.DESCRIPTION for b in blocks))


class TestFullPipelineJatsSecType(unittest.TestCase):
    """Test jats_xml_to_tagged_blocks with JATS sec-type="taxon-treatment"."""

    def test_produces_expected_tags(self):
        body = (
            '<sec sec-type="Taxonomy"><title>Taxonomy</title>'
            + _JATS_TREATMENT_XML
            + '</sec>'
        )
        xml = _wrap_article(body)
        blocks = jats_xml_to_tagged_blocks(xml)
        tags = {b.tag for b in blocks}
        self.assertIn(Tag.NOMENCLATURE, tags)
        self.assertIn(Tag.DESCRIPTION, tags)
        self.assertIn(Tag.TYPE_DESIGNATION, tags)
        self.assertIn(Tag.ETYMOLOGY, tags)

    def test_multiple_jats_treatments(self):
        treatment2 = (
            '<sec sec-type="taxon-treatment">'
            '<title>Tomentella yunnanensis C.L.Zhao</title>'
            '<sec sec-type="treatment-description">'
            '<p>Arachnoid basidiomata.</p>'
            '</sec>'
            '</sec>'
        )
        body = (
            '<sec sec-type="Taxonomy"><title>Taxonomy</title>'
            + _JATS_TREATMENT_XML
            + treatment2
            + '</sec>'
        )
        xml = _wrap_article(body)
        blocks = jats_xml_to_tagged_blocks(xml)
        noms = [b for b in blocks if b.tag == Tag.NOMENCLATURE]
        self.assertEqual(len(noms), 2)
        self.assertIn("wumenshanensis", noms[0].text)
        self.assertIn("yunnanensis", noms[1].text)


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


class TestFigureCitationFolding(unittest.TestCase):
    """Figure-citation secs are folded into the preceding Nomenclature block."""

    def _treatment(self, fig_sec_type: str) -> str:
        return (
            '<tp:taxon-treatment xmlns:tp="http://www.plazi.org/taxpub">'
            "<tp:nomenclature>"
            "<tp:taxon-name>Coprinus urticicola</tp:taxon-name>"
            "</tp:nomenclature>"
            f'<tp:treatment-sec sec-type="{fig_sec_type}">'
            "<p>Figs. 10-14</p>"
            "</tp:treatment-sec>"
            '<tp:treatment-sec sec-type="description">'
            "<p>Pileus 2-4 cm.</p>"
            "</tp:treatment-sec>"
            "</tp:taxon-treatment>"
        )

    def test_figure_citations_folded_into_nomenclature(self) -> None:
        elem = ET.fromstring(self._treatment("figure-citations"))
        blocks = process_treatment(elem)
        nomenclature_blocks = [b for b in blocks if b.tag == Tag.NOMENCLATURE]
        self.assertEqual(len(nomenclature_blocks), 1)
        self.assertIn("Coprinus urticicola", nomenclature_blocks[0].text)
        self.assertIn("Figs. 10-14", nomenclature_blocks[0].text)

    def test_figure_citations_variants(self) -> None:
        for sec_type in ("figure-citations", "figure_citations", "plates",
                         "plate", "figures cited", "figure citation"):
            with self.subTest(sec_type=sec_type):
                elem = ET.fromstring(self._treatment(sec_type))
                blocks = process_treatment(elem)
                nom = [b for b in blocks if b.tag == Tag.NOMENCLATURE]
                self.assertEqual(len(nom), 1)
                self.assertIn("Figs. 10-14", nom[0].text)

    def test_description_block_still_separate(self) -> None:
        elem = ET.fromstring(self._treatment("figure-citations"))
        blocks = process_treatment(elem)
        desc = [b for b in blocks if b.tag == Tag.DESCRIPTION]
        self.assertEqual(len(desc), 1)
        self.assertIn("Pileus", desc[0].text)

    def test_figure_citation_without_preceding_nomenclature(self) -> None:
        """If there is no preceding nomenclature block, start a new one."""
        xml = (
            '<tp:taxon-treatment xmlns:tp="http://www.plazi.org/taxpub">'
            '<tp:treatment-sec sec-type="figure-citations">'
            "<p>Figs. 1-3</p>"
            "</tp:treatment-sec>"
            "</tp:taxon-treatment>"
        )
        elem = ET.fromstring(xml)
        blocks = process_treatment(elem)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tag, Tag.NOMENCLATURE)
        self.assertIn("Figs. 1-3", blocks[0].text)

    def test_jats_treatment_figure_citations_folded(self) -> None:
        """process_jats_treatment also folds figure-citation secs."""
        xml = (
            '<sec sec-type="taxon-treatment">'
            "<title>Coprinus urticicola (Berk.) Buller</title>"
            '<sec sec-type="figure-citations"><p>Figs. 1-5</p></sec>'
            '<sec sec-type="description"><p>Pileus 2-4 cm.</p></sec>'
            "</sec>"
        )
        elem = ET.fromstring(xml)
        blocks = process_jats_treatment(elem)
        nom = [b for b in blocks if b.tag == Tag.NOMENCLATURE]
        self.assertEqual(len(nom), 1)
        self.assertIn("Coprinus urticicola", nom[0].text)
        self.assertIn("Figs. 1-5", nom[0].text)


class TestSecTypeToTagFigureCitations(unittest.TestCase):
    """sec_type_to_tag maps figure-citation variants to NOMENCLATURE."""

    def test_figure_citations(self) -> None:
        self.assertEqual(sec_type_to_tag("figure-citations"), Tag.NOMENCLATURE)

    def test_figure_citations_underscore(self) -> None:
        self.assertEqual(sec_type_to_tag("figure_citations"), Tag.NOMENCLATURE)

    def test_figures_cited(self) -> None:
        self.assertEqual(sec_type_to_tag("figures cited"), Tag.NOMENCLATURE)

    def test_figure_citation_singular(self) -> None:
        self.assertEqual(sec_type_to_tag("figure citation"), Tag.NOMENCLATURE)

    def test_plates(self) -> None:
        self.assertEqual(sec_type_to_tag("plates"), Tag.NOMENCLATURE)

    def test_plate(self) -> None:
        self.assertEqual(sec_type_to_tag("plate"), Tag.NOMENCLATURE)


if __name__ == "__main__":
    unittest.main()


# ============================================================================
# Step 2 — JATS converter extensions (docs/golden_v2_plan.md)
# ============================================================================


class TestSecTypeToTagMaterialsAndMethods(unittest.TestCase):
    """Step 2.A — sec_type_to_tag() must recognise the article-level
    Materials-and-Methods family as Tag.MATERIALS_AND_METHODS, not as the
    MISC_EXPOSITION catch-all."""

    def test_materials_and_methods_hyphenated(self):
        self.assertEqual(
            sec_type_to_tag("materials-and-methods"), Tag.MATERIALS_AND_METHODS
        )

    def test_methods_and_materials_reversed_order(self):
        self.assertEqual(
            sec_type_to_tag("methods-and-materials"), Tag.MATERIALS_AND_METHODS
        )

    def test_methods_alone(self):
        self.assertEqual(sec_type_to_tag("methods"), Tag.MATERIALS_AND_METHODS)

    def test_methodology(self):
        self.assertEqual(
            sec_type_to_tag("methodology"), Tag.MATERIALS_AND_METHODS
        )

    def test_case_insensitive(self):
        self.assertEqual(
            sec_type_to_tag("Materials and Methods"),
            Tag.MATERIALS_AND_METHODS,
        )

    def test_with_underscore_variant(self):
        """Some publishers normalise hyphens to underscores."""
        self.assertEqual(
            sec_type_to_tag("materials_and_methods"),
            Tag.MATERIALS_AND_METHODS,
        )

    def test_not_confused_with_materials_examined(self):
        """The existing 'material' / 'materials examined' mappings must
        still resolve to MATERIALS_EXAMINED (specimen lists), not the
        new MATERIALS_AND_METHODS (technique narrative)."""
        self.assertEqual(
            sec_type_to_tag("materials examined"), Tag.MATERIALS_EXAMINED
        )
        self.assertEqual(sec_type_to_tag("material"), Tag.MATERIALS_EXAMINED)


class TestArticleLevelMaterialsAndMethods(unittest.TestCase):
    """Step 2.B — article-level <sec sec-type="materials-and-methods">
    must emit a MATERIALS_AND_METHODS block rather than falling through
    to MISC_EXPOSITION (as every non-treatment article-level section did
    pre-Step-2)."""

    def test_article_level_materials_and_methods_block(self):
        body_xml = """
        <sec sec-type="materials-and-methods">
          <title>Materials and methods</title>
          <p>We collected specimens at Site A using protocol B.</p>
        </sec>
        """
        xml = _wrap_article(body_xml)
        blocks = jats_xml_to_tagged_blocks(xml)
        mm_blocks = [b for b in blocks if b.tag == Tag.MATERIALS_AND_METHODS]
        self.assertEqual(
            len(mm_blocks), 1,
            f"expected one MATERIALS_AND_METHODS block; got {blocks}",
        )
        self.assertIn("specimens", mm_blocks[0].text)

    def test_article_level_methods_short_sec_type(self):
        body_xml = """
        <sec sec-type="methods">
          <title>Methods</title>
          <p>DNA was extracted via the CTAB method.</p>
        </sec>
        """
        blocks = jats_xml_to_tagged_blocks(_wrap_article(body_xml))
        tags = [b.tag for b in blocks]
        self.assertIn(Tag.MATERIALS_AND_METHODS, tags)
        self.assertNotIn(Tag.MISC_EXPOSITION, tags)

    def test_unrelated_article_section_stays_misc(self):
        """A non-treatment, non-methods section (e.g. introduction) still
        falls to MISC_EXPOSITION — the new behaviour only kicks in for the
        materials-and-methods family."""
        body_xml = """
        <sec sec-type="introduction">
          <title>Introduction</title>
          <p>Background on the genus.</p>
        </sec>
        """
        blocks = jats_xml_to_tagged_blocks(_wrap_article(body_xml))
        tags = [b.tag for b in blocks]
        self.assertIn(Tag.MISC_EXPOSITION, tags)
        self.assertNotIn(Tag.MATERIALS_AND_METHODS, tags)

    def test_article_level_section_without_sec_type_stays_misc(self):
        """A <sec> without any sec-type attribute (common for plain
        narrative passages) keeps falling to MISC_EXPOSITION."""
        body_xml = """
        <sec>
          <title>Discussion</title>
          <p>We discuss the implications.</p>
        </sec>
        """
        blocks = jats_xml_to_tagged_blocks(_wrap_article(body_xml))
        tags = [b.tag for b in blocks]
        self.assertIn(Tag.MISC_EXPOSITION, tags)


class TestNotesAsTreatmentCatchAll(unittest.TestCase):
    """Step 2.C — inside <tp:taxon-treatment>, any treatment-sec whose
    sec-type isn't recognised should produce a NOTES block (not a
    MISC_EXPOSITION one).  The semantic: prose inside a treatment that
    isn't otherwise tagged is part of the treatment's Notes."""

    def test_unknown_sec_type_inside_treatment_is_notes(self):
        # The 'remarks' sec-type is not in sec_type_to_tag() — used to
        # fall through to MISC_EXPOSITION.  Now should become NOTES.
        treatment_xml = """
        <taxon-treatment>
          <nomenclature>
            <taxon-name>Fungus novus</taxon-name>
          </nomenclature>
          <treatment-sec sec-type="remarks">
            <p>This species was first noted in 2019.</p>
          </treatment-sec>
        </taxon-treatment>
        """
        blocks = jats_xml_to_tagged_blocks(_wrap_article(treatment_xml))
        # NOMENCLATURE for the taxon-name + NOTES for the remarks.
        tags = [b.tag for b in blocks]
        self.assertIn(Tag.NOTES, tags)
        # No MISC_EXPOSITION inside a treatment.
        self.assertNotIn(Tag.MISC_EXPOSITION, tags)

    def test_empty_sec_type_inside_treatment_is_notes(self):
        """A treatment-sec with no sec-type attribute (rare but legal)
        also becomes NOTES."""
        treatment_xml = """
        <taxon-treatment>
          <nomenclature>
            <taxon-name>Fungus novus</taxon-name>
          </nomenclature>
          <treatment-sec>
            <p>Untagged commentary.</p>
          </treatment-sec>
        </taxon-treatment>
        """
        blocks = jats_xml_to_tagged_blocks(_wrap_article(treatment_xml))
        tags = [b.tag for b in blocks]
        self.assertIn(Tag.NOTES, tags)
        self.assertNotIn(Tag.MISC_EXPOSITION, tags)

    def test_explicit_notes_sec_type_still_notes(self):
        """The pre-existing 'notes' sec-type mapping is untouched."""
        treatment_xml = """
        <taxon-treatment>
          <nomenclature>
            <taxon-name>Fungus novus</taxon-name>
          </nomenclature>
          <treatment-sec sec-type="notes">
            <p>Explicitly tagged notes.</p>
          </treatment-sec>
        </taxon-treatment>
        """
        blocks = jats_xml_to_tagged_blocks(_wrap_article(treatment_xml))
        notes = [b for b in blocks if b.tag == Tag.NOTES]
        self.assertEqual(len(notes), 1)

    def test_unknown_article_level_sec_still_misc(self):
        """OUTSIDE a treatment, an unrecognised sec-type still falls to
        MISC_EXPOSITION — the NOTES default applies only inside treatments."""
        body_xml = """
        <sec sec-type="acknowledgements">
          <p>We thank our colleagues.</p>
        </sec>
        """
        blocks = jats_xml_to_tagged_blocks(_wrap_article(body_xml))
        tags = [b.tag for b in blocks]
        self.assertIn(Tag.MISC_EXPOSITION, tags)
        self.assertNotIn(Tag.NOTES, tags)

    def test_known_sec_type_inside_treatment_unaffected(self):
        """A recognised treatment-sec sec-type still maps via the existing
        table — Step 2.C only changes the fallback."""
        treatment_xml = """
        <taxon-treatment>
          <nomenclature>
            <taxon-name>Fungus novus</taxon-name>
          </nomenclature>
          <treatment-sec sec-type="description">
            <p>Cap red, stipe white.</p>
          </treatment-sec>
          <treatment-sec sec-type="etymology">
            <p>From the Latin novus.</p>
          </treatment-sec>
        </taxon-treatment>
        """
        blocks = jats_xml_to_tagged_blocks(_wrap_article(treatment_xml))
        tags = [b.tag for b in blocks]
        self.assertIn(Tag.DESCRIPTION, tags)
        self.assertIn(Tag.ETYMOLOGY, tags)


class TestJatsEmitTags(unittest.TestCase):
    """``JATS_EMIT_TAGS`` documents and pins the exact set of Tag values
    that ``sec_type_to_tag()`` and the rest of the converter can return.
    Step 1.C of docs/production_v3_plan.md: the converter's emit set
    must stay a subset of ``ACTIVE_TAGS_19``, never overlap
    ``DEPRECATED_TAGS``. These tests catch a future ``return
    Tag.HOLOTYPE`` slipping in by accident."""

    def test_emit_set_subset_of_active_19(self) -> None:
        """Every Tag the JATS converter can emit is in the 19-tag
        active label set. If you add a new ``return Tag.X`` branch in
        jats_to_yedda, add it to JATS_EMIT_TAGS too; this test will
        fail until you do."""
        self.assertTrue(
            JATS_EMIT_TAGS.issubset(set(ACTIVE_TAGS_19)),
            f"JATS_EMIT_TAGS includes non-active tags: "
            f"{JATS_EMIT_TAGS - set(ACTIVE_TAGS_19)}",
        )

    def test_emit_set_disjoint_from_deprecated(self) -> None:
        """The converter must never emit a deprecated tag. Catches
        accidental regressions to ``return Tag.HOLOTYPE`` /
        ``Tag.DISTRIBUTION`` / ``Tag.FIX``."""
        self.assertTrue(
            JATS_EMIT_TAGS.isdisjoint(DEPRECATED_TAGS),
            f"JATS_EMIT_TAGS overlaps deprecated tags: "
            f"{JATS_EMIT_TAGS & DEPRECATED_TAGS}",
        )

    def test_misc_exposition_in_emit_set(self) -> None:
        """``MISC_EXPOSITION`` is the catch-all default for any
        unmatched sec-type — it must be in the emit set."""
        self.assertIn(Tag.MISC_EXPOSITION, JATS_EMIT_TAGS)

    def test_known_sec_types_emit_documented_tags(self) -> None:
        """A representative spot-check that every sec_type_to_tag
        return path produces a tag that's in JATS_EMIT_TAGS. If a new
        return-path is added without updating JATS_EMIT_TAGS, this
        test fails."""
        observed: set = set()
        for sec_type in (
            "description", "etymology", "diagnosis",
            "type-designation", "materials-examined", "biology",
            "phylogeny", "new-combinations", "nomenclature",
            "notes", "key", "bibliography", "table",
            "materials-and-methods",
        ):
            tag = sec_type_to_tag(sec_type)
            observed.add(tag)
        # Plus the catch-all path for an unrecognised sec-type.
        observed.add(sec_type_to_tag("entirely-unrecognised-foo"))
        self.assertTrue(
            observed.issubset(JATS_EMIT_TAGS),
            f"sec_type_to_tag returned tags not in JATS_EMIT_TAGS: "
            f"{observed - JATS_EMIT_TAGS}",
        )
