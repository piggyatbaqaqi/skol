"""Tests for treatment.py."""

import textwrap
from typing import List
import unittest

import pytest

from label import Label
from line import Line
from paragraph import Paragraph
from treatment import Treatment, _slim_ingest, group_paragraphs
from finder import parse_annotated


def lineify(lines: List[str]) -> List[Line]:
    return [Line(ln) for ln in lines]


class MockFileObject:
    """Mock file object for testing with doc_id support."""

    def __init__(
        self,
        doc_id: str = None,
        filename: str = "test.txt",
        ingest: dict = None,
    ):
        self.doc_id = doc_id
        self.filename = filename
        self.line_number = 1
        self.page_number = 1
        self.pdf_page = 0
        self.pdf_label = None
        self.empirical_page_number = None
        self._empirical_page_number = None
        self.ingest = ingest

    def _set_empirical_page(self, line: str) -> None:
        """Mock implementation of empirical page extraction."""
        import regex as re

        match = re.search(
            r"(^\s*(?P<leading>[mdclxvi\d]+\b))|((?P<trailing>\b[mdclxvi\d]+)\s*$)",
            line,
        )
        if not match:
            self._empirical_page_number = None
        else:
            self._empirical_page_number = match.group(
                "leading"
            ) or match.group("trailing")
        self.empirical_page_number = self._empirical_page_number


def lineify_with_doc_id(lines: List[tuple]) -> List[Line]:
    """Create Lines with specific doc_id values.

    Args:
        lines: List of tuples (line_text, doc_id)

    Returns:
        List of Line objects with doc_id metadata
    """
    result = []
    for line_text, doc_id in lines:
        fileobj = MockFileObject(doc_id=doc_id)
        result.append(Line(line_text, fileobj))
    return result


class TestTreatment(unittest.TestCase):

    def setUp(self):
        Treatment.MISC_GAP_LIMIT = 3  # Give up faster than in real conditions.

    def test_sunny(self):
        test_data = lineify(
            textwrap.dedent(
                """\
        [@paragraph1#Nomenclature*]
        [@paragraph2#Misc-exposition*]
        [@paragraph3#Nomenclature*]
        [@paragraph4#Description*]
        [@paragraph5#Misc-exposition*]
        [@paragraph6#Description*]
        [@paragraph7#Misc-exposition*]
        [@paragraph8#Misc-exposition*]
        [@paragraph9#Nomenclature*]
        [@paragraph10#Misc-exposition*]
        [@paragraph12#Description*]
        [@paragraph13#Misc-exposition*]
        [@paragraph14#Description*]
        [@paragraph15#Description*]
        """
            ).split("\n")
        )

        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 2)
        dictionaries1 = list(taxa[0].dictionaries())
        dictionaries2 = list(taxa[1].dictionaries())
        sn1 = dictionaries1[0]["serial_number"]
        sn2 = dictionaries2[0]["serial_number"]
        self.assertNotEqual(sn1, sn2)

        self.assertEqual(len(dictionaries1), 4)
        self.assertTrue(
            all([d["serial_number"] == sn1 for d in dictionaries1])
        )
        self.assertListEqual(
            [d["paragraph_number"] for d in dictionaries1], [1, 3, 4, 6]
        )

        self.assertEqual(len(dictionaries2), 4)
        self.assertTrue(
            all([d["serial_number"] == sn2 for d in dictionaries2])
        )
        self.assertListEqual(
            [d["paragraph_number"] for d in dictionaries2], [9, 11, 13, 14]
        )

        dict0 = dictionaries1[0]
        dict2 = dictionaries1[2]
        self.assertEqual(dict0["body"], "paragraph1\n")
        self.assertEqual(dict0["label"], "Nomenclature")
        self.assertEqual(dict2["body"], "paragraph4\n")
        self.assertEqual(dict2["label"], "Description")

        dict4 = dictionaries2[0]
        dict7 = dictionaries2[3]
        self.assertEqual(dict4["body"], "paragraph9\n")
        self.assertEqual(dict4["label"], "Nomenclature")
        self.assertEqual(dict7["body"], "paragraph15\n")
        self.assertEqual(dict7["label"], "Description")

    def test_too_long(self):
        test_data = lineify(
            textwrap.dedent(
                """\
        [@ignored1#Nomenclature*]
        [@filler1#Misc-exposition*]
        [@filler2#Misc-exposition*]
        [@filler3#Misc-exposition*]
        [@filler4#Misc-exposition*]
        [@paragraph1#Nomenclature*]
        [@paragraph3#Nomenclature*]
        [@paragraph4#Description*]
        [@paragraph6#Description*]
        [@filler5#Misc-exposition*]
        [@filler6#Misc-exposition*]
        [@filler7#Misc-exposition*]
        [@filler8#Misc-exposition*]
        [@ignored2#Description*]
        [@filler7#Misc-exposition*]
        [@filler8#Misc-exposition*]
        [@paragraph9#Nomenclature*]
        [@paragraph10#Misc-exposition*]
        [@paragraph12#Description*]
        [@paragraph13#Misc-exposition*]
        [@paragraph14#Description*]
        [@paragraph15#Description*]
        """
            ).split("\n")
        )

        taxa = list(group_paragraphs(parse_annotated(test_data)))
        # Now expecting 3 taxa due to stub nomenclature creation for bare Description
        # Treatment 1: paragraph1+3 (nomenclatures) + paragraph4+6 (descriptions)
        # Treatment 2: stub + ignored2 (description that was too far from nomenclature)
        # Treatment 3: paragraph9 (nomenclature) + paragraph12+14+15 (descriptions)
        self.assertEqual(len(taxa), 3)

        dictionaries1 = list(taxa[0].dictionaries())
        dictionaries2 = list(taxa[1].dictionaries())
        dictionaries3 = list(taxa[2].dictionaries())
        sn1 = dictionaries1[0]["serial_number"]
        sn2 = dictionaries2[0]["serial_number"]
        sn3 = dictionaries3[0]["serial_number"]
        self.assertNotEqual(sn1, sn2)
        self.assertNotEqual(sn2, sn3)
        self.assertNotEqual(sn1, sn3)

        # First taxon: paragraph1+3 (nomenclatures) + paragraph4+6
        # (descriptions)
        self.assertEqual(len(dictionaries1), 4)
        self.assertTrue(
            all([d["serial_number"] == sn1 for d in dictionaries1])
        )
        self.assertListEqual(
            [d["paragraph_number"] for d in dictionaries1], [6, 7, 8, 9]
        )

        dict0 = dictionaries1[0]
        dict2 = dictionaries1[2]
        self.assertEqual(dict0["body"], "paragraph1\n")
        self.assertEqual(dict0["label"], "Nomenclature")
        self.assertEqual(dict2["body"], "paragraph4\n")
        self.assertEqual(dict2["label"], "Description")

        # Second taxon: stub + ignored2 (bare description with no preceding
        # nomenclature)
        self.assertEqual(len(dictionaries2), 2)
        self.assertTrue(
            all([d["serial_number"] == sn2 for d in dictionaries2])
        )
        self.assertEqual(dictionaries2[0]["body"], "Nomen ignotum\n")
        self.assertEqual(dictionaries2[0]["label"], "Nomenclature")
        self.assertEqual(dictionaries2[1]["body"], "ignored2\n")
        self.assertEqual(dictionaries2[1]["label"], "Description")

        # Third taxon: paragraph9 (nomenclature) + paragraph12+14+15
        # (descriptions)
        self.assertEqual(len(dictionaries3), 4)
        self.assertTrue(
            all([d["serial_number"] == sn3 for d in dictionaries3])
        )
        self.assertListEqual(
            [d["paragraph_number"] for d in dictionaries3], [17, 19, 21, 22]
        )

        dict4 = dictionaries3[0]
        dict7 = dictionaries3[3]
        self.assertEqual(dict4["body"], "paragraph9\n")
        self.assertEqual(dict4["label"], "Nomenclature")
        self.assertEqual(dict7["body"], "paragraph15\n")
        self.assertEqual(dict7["label"], "Description")

    def test_fall_through_first_description(self):
        """Test fall-through case: first Description after Nomenclature is immediately added.

        When in 'Look for Nomenclatures' state and we encounter a Description paragraph
        after having collected at least one Nomenclature, the state switches to
        'Look for Descriptions' and falls through to immediately add that Description.
        """
        test_data = lineify(
            textwrap.dedent(
                """\
        [@nom1#Nomenclature*]
        [@desc1#Description*]
        [@desc2#Description*]
        """
            ).split("\n")
        )

        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 1, "Should generate exactly one taxon")

        dictionaries = list(taxa[0].dictionaries())
        self.assertEqual(
            len(dictionaries), 3, "Should have 1 nomenclature + 2 descriptions"
        )

        # Verify the first description was captured (fall-through worked)
        self.assertListEqual(
            [d["paragraph_number"] for d in dictionaries], [1, 2, 3]
        )
        self.assertEqual(dictionaries[0]["label"], "Nomenclature")
        self.assertEqual(dictionaries[1]["label"], "Description")
        self.assertEqual(dictionaries[1]["body"], "desc1\n")

    def test_fall_through_gap_reset(self):
        """Test fall-through case: gap causes reset in 'Look for Nomenclatures'.

        When in 'Look for Nomenclatures' state and the Misc-exposition gap exceeds
        MISC_GAP_LIMIT, the taxon is reset and we continue looking for nomenclatures.
        """
        test_data = lineify(
            textwrap.dedent(
                """\
        [@nom1#Nomenclature*]
        [@filler1#Misc-exposition*]
        [@filler2#Misc-exposition*]
        [@filler3#Misc-exposition*]
        [@filler4#Misc-exposition*]
        [@nom2#Nomenclature*]
        [@desc1#Description*]
        """
            ).split("\n")
        )

        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 1, "Should generate exactly one taxon")

        dictionaries = list(taxa[0].dictionaries())
        # Should only have nom2 and desc1, not nom1 (it was reset due to gap)
        self.assertEqual(
            len(dictionaries), 2, "Should have 1 nomenclature + 1 description"
        )
        self.assertListEqual(
            [d["paragraph_number"] for d in dictionaries], [6, 7]
        )
        self.assertEqual(dictionaries[0]["body"], "nom2\n")
        self.assertEqual(dictionaries[1]["body"], "desc1\n")

    def test_document_boundary(self):
        """Test that Nomenclature-Description associations do not cross document boundaries.

        When processing multiple documents (different doc_id values), a Nomenclature
        from one document should not be associated with Descriptions from another document.
        The doc_id boundary should cause the current taxon to be yielded and a new one started.
        """
        # Create test data with two different documents
        # Document A (doc_id='doc_a'): Nomenclature at paragraph 1, Description at paragraph 2
        # Document B (doc_id='doc_b'): Description at paragraph 3, Nomenclature
        # at paragraph 4
        test_data = lineify_with_doc_id(
            [
                ("[@nom_from_doc_a#Nomenclature*]", "doc_a"),
                ("[@desc_from_doc_a#Description*]", "doc_a"),
                ("[@desc_from_doc_b#Description*]", "doc_b"),  # Different doc
                ("[@nom_from_doc_b#Nomenclature*]", "doc_b"),
                ("[@desc2_from_doc_b#Description*]", "doc_b"),
            ]
        )

        taxa = list(group_paragraphs(parse_annotated(test_data)))

        # Should generate 2 taxa, one for each document
        self.assertEqual(
            len(taxa), 2, "Should generate 2 taxa (one per document)"
        )

        # First taxon: from document A
        dictionaries1 = list(taxa[0].dictionaries())
        self.assertEqual(
            len(dictionaries1),
            2,
            "First taxon should have nom + desc from doc_a",
        )
        self.assertEqual(dictionaries1[0]["body"], "nom_from_doc_a\n")
        self.assertEqual(dictionaries1[0]["label"], "Nomenclature")
        self.assertEqual(dictionaries1[1]["body"], "desc_from_doc_a\n")
        self.assertEqual(dictionaries1[1]["label"], "Description")

        self.assertEqual(taxa[0].doc_id(), "doc_a")

        # Second taxon: from document B
        dictionaries2 = list(taxa[1].dictionaries())
        self.assertEqual(
            len(dictionaries2),
            2,
            "Second taxon should have nom + desc from doc_b",
        )
        self.assertEqual(dictionaries2[0]["body"], "nom_from_doc_b\n")
        self.assertEqual(dictionaries2[0]["label"], "Nomenclature")
        self.assertEqual(dictionaries2[1]["body"], "desc2_from_doc_b\n")
        self.assertEqual(dictionaries2[1]["label"], "Description")

        self.assertEqual(taxa[1].doc_id(), "doc_b")

        self.assertNotIn("desc_from_doc_b", [d["body"] for d in dictionaries1])

    def test_document_boundary_while_looking_for_descriptions(self):
        """Test document boundary check in 'Look for Descriptions' state.

        When already collecting descriptions for a nomenclature and we encounter
        a description from a different document, the current taxon should be yielded
        and we should start fresh with the new document.
        """
        test_data = lineify_with_doc_id(
            [
                ("[@nom1#Nomenclature*]", "doc_a"),
                ("[@desc1_a#Description*]", "doc_a"),
                ("[@desc2_a#Description*]", "doc_a"),
                ("[@desc_from_doc_b#Description*]", "doc_b"),
                ("[@nom_from_doc_b#Nomenclature*]", "doc_b"),
                ("[@desc2_b#Description*]", "doc_b"),
            ]
        )

        taxa = list(group_paragraphs(parse_annotated(test_data)))

        self.assertEqual(len(taxa), 2, "Should generate 2 taxa")

        dictionaries1 = list(taxa[0].dictionaries())
        self.assertEqual(len(dictionaries1), 3)
        self.assertEqual(dictionaries1[0]["body"], "nom1\n")
        self.assertEqual(dictionaries1[1]["body"], "desc1_a\n")
        self.assertEqual(dictionaries1[2]["body"], "desc2_a\n")

        self.assertNotIn("desc_from_doc_b", [d["body"] for d in dictionaries1])

        dictionaries2 = list(taxa[1].dictionaries())
        self.assertEqual(len(dictionaries2), 2)
        self.assertEqual(dictionaries2[0]["body"], "nom_from_doc_b\n")
        self.assertEqual(dictionaries2[1]["body"], "desc2_b\n")

    def test_bare_description_creates_stub_nomenclature(self):
        """Test that a Description without preceding Nomenclature creates a stub.

        When we encounter a Description paragraph without any preceding Nomenclature,
        a stub Nomenclature paragraph with 'Nomen ignotum' should be automatically
        created since Descriptions are more reliably detected than Nomenclatures.
        """
        test_data = lineify(
            textwrap.dedent(
                """\
        [@desc1#Description*]
        [@desc2#Description*]
        """
            ).split("\n")
        )

        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 1, "Should generate exactly one taxon")

        dictionaries = list(taxa[0].dictionaries())
        self.assertEqual(
            len(dictionaries),
            3,
            "Should have 1 stub nomenclature + 2 descriptions",
        )

        self.assertEqual(dictionaries[0]["label"], "Nomenclature")
        self.assertEqual(dictionaries[0]["body"], "Nomen ignotum\n")
        self.assertEqual(dictionaries[1]["label"], "Description")
        self.assertEqual(dictionaries[1]["body"], "desc1\n")
        self.assertEqual(dictionaries[2]["label"], "Description")
        self.assertEqual(dictionaries[2]["body"], "desc2\n")

    def test_bare_description_with_nomenclature_later(self):
        """Test stub creation when bare Description comes before actual Nomenclature.

        First taxon should have stub + descriptions, second taxon should have
        actual nomenclature + its descriptions.
        """
        test_data = lineify(
            textwrap.dedent(
                """\
        [@desc1#Description*]
        [@desc2#Description*]
        [@nom1#Nomenclature*]
        [@desc3#Description*]
        """
            ).split("\n")
        )

        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 2, "Should generate 2 taxa")

        dictionaries1 = list(taxa[0].dictionaries())
        self.assertEqual(len(dictionaries1), 3)
        self.assertEqual(dictionaries1[0]["body"], "Nomen ignotum\n")
        self.assertEqual(dictionaries1[0]["label"], "Nomenclature")
        self.assertEqual(dictionaries1[1]["body"], "desc1\n")
        self.assertEqual(dictionaries1[2]["body"], "desc2\n")

        dictionaries2 = list(taxa[1].dictionaries())
        self.assertEqual(len(dictionaries2), 2)
        self.assertEqual(dictionaries2[0]["body"], "nom1\n")
        self.assertEqual(dictionaries2[0]["label"], "Nomenclature")
        self.assertEqual(dictionaries2[1]["body"], "desc3\n")

    def test_as_row_stub_ingest_fallback(self):
        """as_row() uses description ingest when nomenclature stub has none.

        'Nomen ignotum' stubs are synthetic Lines with no fileobj, so
        first_line.ingest is None.  as_row() must fall back to the first
        description paragraph's ingest so the Source Context Viewer can
        locate the document.
        """
        fake_ingest = {
            "_id": "doc123",
            "url": "http://example.com",
            "pdf_url": None,
        }
        fileobj = MockFileObject(doc_id="doc123", ingest=fake_ingest)

        desc_line = Line("[@some description text#Description*]", fileobj)
        desc_para = Paragraph(
            labels=[Label("Description")],
            lines=[desc_line],
            paragraph_number=1,
        )

        nom_stub = Line("Nomen ignotum")  # no fileobj — ingest is None
        nom_para = Paragraph(
            labels=[Label("Nomenclature")],
            lines=[nom_stub],
            paragraph_number=1,
        )

        treatment = Treatment()
        treatment.add_nomenclature(nom_para)
        treatment.add_section("Description", desc_para)

        row = treatment.as_row()
        self.assertIsNotNone(
            row["ingest"],
            "as_row() should fall back to description ingest for stub",
        )
        self.assertEqual(row["ingest"]["_id"], "doc123")


class TestGroupParagraphsNewLabels(unittest.TestCase):
    """group_paragraphs() handles the full 12-tag label set."""

    def setUp(self):
        Treatment.MISC_GAP_LIMIT = 3

    def test_treatment_section_labels_keep_treatment_open(self):
        """Diagnosis/Distribution/Biology do not increment the gap counter."""
        test_data = lineify(
            textwrap.dedent(
                """\
        [@nom1#Nomenclature*]
        [@Differs from A. muscaria.#Diagnosis*]
        [@Found across Europe.#Distribution*]
        [@Saprotrophic on oak.#Biology*]
        [@Pileus 5 cm.#Description*]
        """
            ).split("\n")
        )
        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 1)
        row = taxa[0].as_row()
        self.assertIsNotNone(row["diagnosis"])
        self.assertIsNotNone(row["distribution"])
        self.assertIsNotNone(row["biology"])

    def test_misc_gap_limit_terminates_treatment(self):
        """More than MISC_GAP_LIMIT consecutive Misc-exposition ends the treatment."""
        test_data = lineify(
            textwrap.dedent(
                """\
        [@nom1#Nomenclature*]
        [@desc1#Description*]
        [@filler1#Misc-exposition*]
        [@filler2#Misc-exposition*]
        [@filler3#Misc-exposition*]
        [@filler4#Misc-exposition*]
        [@nom2#Nomenclature*]
        [@desc2#Description*]
        """
            ).split("\n")
        )
        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 2)

    def test_treatment_label_resets_misc_gap(self):
        """A treatment-section label resets the Misc-exposition gap counter."""
        # 2 misc + Diagnosis + 2 misc: counter resets at Diagnosis → 2 at end ≤
        # 3
        test_data = lineify(
            textwrap.dedent(
                """\
        [@nom1#Nomenclature*]
        [@desc1#Description*]
        [@filler1#Misc-exposition*]
        [@filler2#Misc-exposition*]
        [@extra diag.#Diagnosis*]
        [@filler3#Misc-exposition*]
        [@filler4#Misc-exposition*]
        [@desc2#Description*]
        """
            ).split("\n")
        )
        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 1)

    def test_nomenclature_yields_treatment_with_non_description_sections(self):
        """A new Nomenclature yields the current treatment even with only Diagnosis."""
        test_data = lineify(
            textwrap.dedent(
                """\
        [@nom1#Nomenclature*]
        [@diag1#Diagnosis*]
        [@nom2#Nomenclature*]
        [@desc2#Description*]
        """
            ).split("\n")
        )
        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 2)
        self.assertIsNotNone(taxa[0].as_row()["diagnosis"])

    def test_all_treatment_section_labels_accepted(self):
        """All 10 treatment-section labels are accepted without terminating."""
        test_data = lineify(
            textwrap.dedent(
                """\
        [@nom1#Nomenclature*]
        [@d1#Diagnosis*]
        [@e1#Etymology*]
        [@dist1#Distribution*]
        [@mat1#Materials-examined*]
        [@type1#Type-designation*]
        [@bio1#Biology*]
        [@n1#Notes*]
        [@k1#Key*]
        [@fig1#Figure-caption*]
        [@desc1#Description*]
        """
            ).split("\n")
        )
        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 1)
        row = taxa[0].as_row()
        self.assertIsNotNone(row["etymology"])
        self.assertIsNotNone(row["materials_examined"])
        self.assertIsNotNone(row["type_designation"])
        self.assertIsNotNone(row["notes"])
        self.assertIsNotNone(row["key"])
        self.assertIsNotNone(row["figure_captions"])


class TestTreatmentNewFields(unittest.TestCase):
    """Treatment.as_row() exposes flat fields for all 12-tag section types."""

    def _make_para(
        self, text: str, label_str: str, para_num: int = 1
    ) -> Paragraph:
        line = Line(f"[@{text}#{label_str}*]")
        return Paragraph(
            labels=[Label(label_str)], lines=[line], paragraph_number=para_num
        )

    def _treatment_with(self, sections: list) -> Treatment:
        nom = self._make_para("Amanita muscaria", "Nomenclature", para_num=1)
        t = Treatment()
        t.add_nomenclature(nom)
        for i, (text, label_str) in enumerate(sections, start=2):
            t.add_section(
                label_str, self._make_para(text, label_str, para_num=i)
            )
        return t

    def test_new_section_fields_populated(self):
        t = self._treatment_with(
            [
                ("Differs from X.", "Diagnosis"),
                ("Found in Europe.", "Distribution"),
                ("Saprotrophic.", "Biology"),
            ]
        )
        row = t.as_row()
        self.assertIn("Differs from X", row["diagnosis"])
        self.assertIn("Found in Europe", row["distribution"])
        self.assertIn("Saprotrophic", row["biology"])

    def test_absent_sections_are_none(self):
        t = self._treatment_with([("Pileus convex.", "Description")])
        row = t.as_row()
        for field in (
            "diagnosis",
            "etymology",
            "distribution",
            "materials_examined",
            "type_designation",
            "biology",
            "notes",
            "key",
            "figure_captions",
        ):
            self.assertIsNone(row[field], msg=f"{field} should be None")

    def test_multiple_blocks_same_section_concatenated(self):
        t = self._treatment_with(
            [
                ("First diagnosis.", "Diagnosis"),
                ("Second diagnosis.", "Diagnosis"),
            ]
        )
        row = t.as_row()
        self.assertIn("First diagnosis", row["diagnosis"])
        self.assertIn("Second diagnosis", row["diagnosis"])

    def test_span_fields_present_for_all_sections(self):
        t = self._treatment_with([("Differs from X.", "Diagnosis")])
        row = t.as_row()
        for field in (
            "diagnosis_spans",
            "etymology_spans",
            "distribution_spans",
            "materials_examined_spans",
            "type_designation_spans",
            "biology_spans",
            "notes_spans",
        ):
            self.assertIn(field, row, msg=f"{field} missing from as_row()")

    def test_description_field_still_works(self):
        """Existing 'description' field and 'description_spans' still present."""
        t = self._treatment_with([("Pileus convex.", "Description")])
        row = t.as_row()
        self.assertIn("Pileus convex", row["description"])
        self.assertIn("description_spans", row)


class TestIngestSlimming(unittest.TestCase):
    """``Treatment.as_row()`` stores only the essential ingest keys
    (``_id``, ``url``, ``pdf_url``, ``xml_url``, ``db_name``, ``doi``)
    — not the full source CouchDB doc.  Phase B of the embedding-bloat
    fix: skol_treatments_v3_dev was 1.5 GB and embed_treatments could
    not pickle the full DataFrame because per-treatment ingest
    payloads averaged 261 KB (the entire ingest doc was duplicated
    1-to-many).
    """

    def _treatment_with_fat_ingest(self) -> "Treatment":
        from treatment import Treatment
        fat_ingest = {
            "_id": "doc1",
            "url": "https://example.com/doc1",
            "pdf_url": "https://example.com/doc1.pdf",
            "db_name": "skol_dev",
            "_attachments": {"article.pdf": {"content_type": "application/pdf"}},
            "_rev": "1-abc",
            "publication_metadata": {"title": "X", "authors": ["A", "B"]},
            "ingest_timestamp": "2026-01-01",
            "should_be_dropped_too": "x" * 1000,
        }
        fileobj = MockFileObject(doc_id="doc1", ingest=fat_ingest)
        nom_line = Line("[@Foo bar#Nomenclature*]", fileobj)
        nom_para = Paragraph(
            labels=[Label("Nomenclature")],
            lines=[nom_line],
            paragraph_number=1,
        )
        t = Treatment()
        t.add_nomenclature(nom_para)
        return t

    def test_only_essential_keys_kept(self):
        t = self._treatment_with_fat_ingest()
        row = t.as_row()
        self.assertIsNotNone(row["ingest"])
        self.assertEqual(
            set(row["ingest"].keys()),
            {"_id", "url", "pdf_url", "xml_url", "db_name", "doi"},
            msg=(
                "Treatment.as_row() must drop non-essential ingest fields. "
                "Storing the full ingest doc inflates the Treatment by "
                "~250 KB each and blows the Redis embed-pickle past 4 GB."
            ),
        )

    def test_essential_values_passed_through(self):
        t = self._treatment_with_fat_ingest()
        ingest = t.as_row()["ingest"]
        self.assertEqual(ingest["_id"], "doc1")
        self.assertEqual(ingest["url"], "https://example.com/doc1")
        self.assertEqual(ingest["pdf_url"], "https://example.com/doc1.pdf")
        self.assertEqual(ingest["db_name"], "skol_dev")

    def test_missing_essential_key_yields_none_value(self):
        """If an essential key is absent from the source ingest, the
        slim ingest still has that key but with None — preserves the
        4-key shape downstream consumers expect."""
        from treatment import Treatment
        partial_ingest = {"_id": "doc1", "url": "http://example.com"}
        fileobj = MockFileObject(doc_id="doc1", ingest=partial_ingest)
        nom_line = Line("[@Foo bar#Nomenclature*]", fileobj)
        nom_para = Paragraph(
            labels=[Label("Nomenclature")],
            lines=[nom_line],
            paragraph_number=1,
        )
        t = Treatment()
        t.add_nomenclature(nom_para)
        row = t.as_row()
        self.assertEqual(
            set(row["ingest"].keys()),
            {"_id", "url", "pdf_url", "xml_url", "db_name", "doi"},
        )
        self.assertIsNone(row["ingest"]["pdf_url"])
        self.assertIsNone(row["ingest"]["db_name"])
        self.assertIsNone(row["ingest"]["doi"])
        self.assertIsNone(row["ingest"]["xml_url"])

    def test_none_ingest_stays_none(self):
        """When the nomenclature line has no fileobj (synthetic stub
        with no description either), the ingest slot stays None."""
        from treatment import Treatment, SYNTHETIC_NOMENCLATURE_TEXT
        nom_stub = Line(SYNTHETIC_NOMENCLATURE_TEXT)
        nom_para = Paragraph(
            labels=[Label("Nomenclature")],
            lines=[nom_stub],
            paragraph_number=1,
        )
        t = Treatment()
        t.add_nomenclature(nom_para)
        row = t.as_row()
        self.assertIsNone(row["ingest"])


class TestSyntheticNomenclatureFlag(unittest.TestCase):
    """Phase G.2: ``synthetic_nomenclature`` field distinguishes stub
    Nomenclatures (synthesised by ``group_paragraphs`` for orphan
    Description / Diagnosis blocks) from real species-name headings.
    """

    def _make_para(
        self, text: str, label_str: str, para_num: int = 1
    ) -> Paragraph:
        line = Line(f"[@{text}#{label_str}*]")
        return Paragraph(
            labels=[Label(label_str)],
            lines=[line],
            paragraph_number=para_num,
        )

    def test_real_nomenclature_not_synthetic(self):
        from treatment import Treatment
        t = Treatment()
        t.add_nomenclature(
            self._make_para("Amanita muscaria", "Nomenclature", 1)
        )
        self.assertFalse(t.is_synthetic_nomenclature())
        self.assertFalse(t.as_row()["synthetic_nomenclature"])

    def test_stub_nomenclature_is_synthetic(self):
        from treatment import Treatment, SYNTHETIC_NOMENCLATURE_TEXT
        # Mimic what group_paragraphs does when it sees an orphan
        # section: synthesise a stub Nomenclature with the canonical
        # marker string.
        stub_line = Line(SYNTHETIC_NOMENCLATURE_TEXT)
        stub_para = Paragraph(
            labels=[Label("Nomenclature")],
            lines=[stub_line],
            paragraph_number=1,
        )
        t = Treatment()
        t.add_nomenclature(stub_para)
        self.assertTrue(t.is_synthetic_nomenclature())
        self.assertTrue(t.as_row()["synthetic_nomenclature"])

    def test_orphan_description_via_group_paragraphs_is_synthetic(self):
        """End-to-end: a YEDDA stream with a bare Description paragraph
        produces a Treatment that ``is_synthetic_nomenclature`` flags."""
        import textwrap
        from finder import parse_annotated
        from treatment import group_paragraphs
        test_data = lineify(
            textwrap.dedent(
                """\
            [@cap red#Description*]
            [@from latin#Etymology*]
            """
            ).split("\n")
        )
        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 1)
        self.assertTrue(taxa[0].is_synthetic_nomenclature())
        self.assertTrue(taxa[0].as_row()["synthetic_nomenclature"])


class TestSlimIngestPropagatesDoi(unittest.TestCase):
    """``_slim_ingest`` projects the parent ingest doc down to a
    small fixed-shape dict.  ``doi`` is included so each treatment
    row carries the source article's DOI — useful for citation
    rendering and for joining treatments back to the JOURNALS
    registry without going through skol_dev."""

    def test_doi_propagates_when_present(self):
        slim = _slim_ingest({
            '_id':     'doc1',
            'url':     'https://example.com/x',
            'pdf_url': 'https://example.com/x.pdf',
            'xml_url': 'https://example.com/x.xml',
            'db_name': 'skol_dev',
            'doi':     '10.1234/test.001',
        })
        assert slim is not None
        self.assertEqual(slim['doi'], '10.1234/test.001')

    def test_xml_url_propagates_when_present(self):
        """Mirrors ``doi`` — PMC and Pensoft docs carry an
        ``xml_url`` (the OAI-PMH / journal XML fetch endpoint)
        that consumers may need to re-fetch the canonical source."""
        slim = _slim_ingest({
            '_id':     'doc1',
            'xml_url': 'https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/'
                       '?verb=GetRecord'
                       '&identifier=oai:pubmedcentral.nih.gov:1234567'
                       '&metadataPrefix=pmc',
        })
        assert slim is not None
        self.assertIn('verb=GetRecord', slim['xml_url'])

    def test_doi_is_none_when_absent_from_ingest(self):
        """Stable 5-key shape — missing ``doi`` materialises as
        ``None`` so downstream code can index without ``KeyError``."""
        slim = _slim_ingest({'_id': 'doc1'})
        assert slim is not None
        self.assertIn('doi', slim)
        self.assertIsNone(slim['doi'])

    def test_none_input_returns_none(self):
        self.assertIsNone(_slim_ingest(None))

    def test_other_keys_still_dropped(self):
        """Unknown keys on the ingest doc still get filtered out —
        the slim projection is a strict allow-list."""
        slim = _slim_ingest({
            '_id':       'doc1',
            'doi':       '10.x',
            'title':     'Should Not Propagate',
            'journal':   'Should Not Propagate',
        })
        assert slim is not None
        self.assertNotIn('title', slim)
        self.assertNotIn('journal', slim)
        self.assertEqual(slim['doi'], '10.x')


class TestAsRowExposesDoi(unittest.TestCase):
    """Integration check: ``Treatment.as_row()['ingest']`` carries
    the parent doc's DOI when present.  Walks the same materialisation
    path the Spark extractor uses to write the treatments DB."""

    def test_doi_present_in_as_row(self):
        fake_ingest = {
            '_id':     'doc123',
            'pdf_url': 'http://example.com/x.pdf',
            'doi':     '10.5678/test.042',
        }
        fileobj = MockFileObject(doc_id='doc123', ingest=fake_ingest)
        nom_line = Line('[@Foo bar Smith#Nomenclature*]', fileobj)
        nom_para = Paragraph(
            labels=[Label('Nomenclature')],
            lines=[nom_line],
            paragraph_number=1,
        )

        treatment = Treatment()
        treatment.add_nomenclature(nom_para)

        row = treatment.as_row()
        self.assertEqual(row['ingest']['doi'], '10.5678/test.042')


class TestSourceAnchorsEmission(unittest.TestCase):
    """Trello #401 Phase 1 (Commit A): ``Treatment.as_row()`` emits a
    polymorphic ``source_anchors`` list.  This commit covers PDF and
    Plazi kinds only; JATS (arpha, jats_section, mycobank) lands in
    Commit B.  Values are stringified for the ArrayType(MapType(String,
    String)) Spark schema — same convention as span fields.
    """

    def _build_treatment(
        self,
        pdf_page: int = 0,
        pdf_label=None,
        plazi_uuids=None,
    ):
        """Minimal Treatment with a single Nomenclature paragraph on a
        Line whose fileobj carries the requested pdf_page / pdf_label
        and ingest.plazi.uuids values."""
        ingest = {'_id': 'doc123'}
        if plazi_uuids is not None:
            ingest['plazi'] = {'uuids': plazi_uuids}
        fileobj = MockFileObject(doc_id='doc123', ingest=ingest)
        fileobj.pdf_page = pdf_page
        fileobj.pdf_label = pdf_label
        line = Line('[@Foo bar sp. nov.#Nomenclature*]', fileobj)
        para = Paragraph(
            labels=[Label('Nomenclature')],
            lines=[line],
            paragraph_number=1,
        )
        treatment = Treatment()
        treatment.add_nomenclature(para)
        return treatment

    def test_no_anchors_empty_list(self):
        """No PDF page marker, no Plazi UUIDs → empty list."""
        treatment = self._build_treatment()
        row = treatment.as_row()
        self.assertEqual(row['source_anchors'], [])

    def test_pdf_anchor_with_page_and_label(self):
        """pdf_page > 0 with explicit label → one pdf anchor."""
        treatment = self._build_treatment(pdf_page=3, pdf_label='iii')
        row = treatment.as_row()
        self.assertEqual(row['source_anchors'], [
            {'kind': 'pdf', 'page': '3', 'label': 'iii'},
        ])

    def test_pdf_anchor_without_label_falls_back_to_page_string(self):
        """pdf_page > 0 with no pdf_label → label defaults to the
        stringified page number.  Matches how pdf_page_pattern's group
        2 was optional in the raw JATS-wrapped marker case (#399)."""
        treatment = self._build_treatment(pdf_page=77)
        row = treatment.as_row()
        self.assertEqual(row['source_anchors'], [
            {'kind': 'pdf', 'page': '77', 'label': '77'},
        ])

    def test_no_pdf_anchor_when_page_zero(self):
        """pdf_page == 0 is the "not tracked" sentinel — the Pensoft
        pre-#401 shape.  Emit no PDF anchor; the absence is caught
        downstream by n_source_anchors."""
        treatment = self._build_treatment(pdf_page=0, pdf_label=None)
        row = treatment.as_row()
        self.assertFalse(
            any(a['kind'] == 'pdf' for a in row['source_anchors']),
            f"expected no pdf anchor, got {row['source_anchors']}",
        )

    def test_plazi_anchors_from_ingest_uuids(self):
        """doc.plazi.uuids[] → one plazi anchor per UUID, in list
        order.  Article-level anchors — Plazi's resolver handles
        per-treatment picking (see docs/source-anchors.md)."""
        uuids = [
            '0A4F6E6CD877BD32697F3B6BB9EF2AB5',
            '0E8816DBE87F389A52ABD068AF80F54C',
        ]
        treatment = self._build_treatment(plazi_uuids=uuids)
        row = treatment.as_row()
        plazi_anchors = [
            a for a in row['source_anchors'] if a['kind'] == 'plazi'
        ]
        self.assertEqual(plazi_anchors, [
            {'kind': 'plazi', 'uuid': uuids[0]},
            {'kind': 'plazi', 'uuid': uuids[1]},
        ])

    def test_pdf_and_plazi_together_both_emitted(self):
        """Both signals present → both anchors emitted.  Storage
        order is NOT part of the contract — Django's serializer
        imposes its own priority policy (see
        docs/source-anchors.md).  This test asserts membership,
        not order."""
        uuid = '54496DA047999E86D24CD1B3B71B163D'
        treatment = self._build_treatment(
            pdf_page=1, pdf_label='1', plazi_uuids=[uuid],
        )
        anchors = treatment.as_row()['source_anchors']
        self.assertEqual(len(anchors), 2)
        self.assertIn(
            {'kind': 'pdf', 'page': '1', 'label': '1'}, anchors,
        )
        self.assertIn({'kind': 'plazi', 'uuid': uuid}, anchors)

    def test_plazi_empty_uuids_no_anchors(self):
        """doc.plazi present but uuids array empty → no plazi
        anchors.  Matches the 000ab462 shape in skol_dev where
        Plazi lookup ran but found no coverage for the article."""
        treatment = self._build_treatment(plazi_uuids=[])
        row = treatment.as_row()
        plazi_anchors = [
            a for a in row['source_anchors'] if a['kind'] == 'plazi'
        ]
        self.assertEqual(plazi_anchors, [])


_XFAIL_TAXPUB_ANCHORS = pytest.mark.xfail(
    reason=(
        "Trello #401 Phase 1 Commit B: taxpub anchor emission "
        "implementation lands in follow-up commit."
    ),
    strict=True,
)


class TestTaxpubAnchorEmission(unittest.TestCase):
    """Trello #401 Phase 1 (Commit B): ``Treatment.as_row()`` emits
    arpha / jats_section / mycobank anchors when a per-treatment
    taxpub bundle has been attached via ``set_taxpub_anchors``.

    The assembler is responsible for calling ``set_taxpub_anchors``
    with the bundle extracted from the JATS XML by
    ``extract_taxpub_anchor_bundles``.  The assembler-level
    integration is tested separately; this class asserts the
    Treatment-level emission contract in isolation.
    """

    def _build_treatment(self, bundle=None):
        """Minimal Treatment with a Nomenclature line, optionally
        pre-attached to a taxpub anchor bundle."""
        fileobj = MockFileObject(
            doc_id='doc123', ingest={'_id': 'doc123'},
        )
        line = Line('[@Foo bar sp. nov.#Nomenclature*]', fileobj)
        para = Paragraph(
            labels=[Label('Nomenclature')],
            lines=[line],
            paragraph_number=1,
        )
        treatment = Treatment()
        treatment.add_nomenclature(para)
        if bundle is not None:
            treatment.set_taxpub_anchors(bundle)
        return treatment

    def test_full_bundle_emits_three_kinds(self):
        """All three fields populated → three anchors emitted."""
        bundle = {
            'arpha_uuid': 'BA154B2C-A975-5BFF-BEEA-42184807F9D3',
            'mycobank_id': '853632',
            'first_section_id': 'SECID0EGZGK',
            'doi': '10.3897/mycokeys.108.130565',
        }
        anchors = self._build_treatment(bundle).as_row()['source_anchors']
        self.assertIn(
            {'kind': 'arpha',
             'uuid': 'BA154B2C-A975-5BFF-BEEA-42184807F9D3'},
            anchors,
        )
        self.assertIn(
            {'kind': 'jats_section',
             'doi': '10.3897/mycokeys.108.130565',
             'section_id': 'SECID0EGZGK'},
            anchors,
        )
        self.assertIn(
            {'kind': 'mycobank', 'id': '853632'},
            anchors,
        )

    def test_arpha_only_bundle(self):
        """Only arpha_uuid populated → only arpha anchor emitted."""
        bundle = {
            'arpha_uuid': 'AAAAAAAA-1234-5678-9ABC-DEF012345678',
            'mycobank_id': None,
            'first_section_id': None,
            'doi': None,
        }
        anchors = self._build_treatment(bundle).as_row()['source_anchors']
        arpha = [a for a in anchors if a['kind'] == 'arpha']
        self.assertEqual(len(arpha), 1)
        # No other JATS kinds emitted.
        for kind in ('jats_section', 'mycobank'):
            self.assertFalse(
                any(a['kind'] == kind for a in anchors),
                f"{kind} anchor emitted with null source field",
            )

    def test_jats_section_needs_both_doi_and_section_id(self):
        """jats_section is skipped when either doi or section_id is
        None — the URL construction contract requires both."""
        for bundle in (
            {'arpha_uuid': None, 'mycobank_id': None,
             'first_section_id': 'SECID0AAA', 'doi': None},
            {'arpha_uuid': None, 'mycobank_id': None,
             'first_section_id': None,
             'doi': '10.0/x'},
        ):
            anchors = self._build_treatment(bundle).as_row()[
                'source_anchors'
            ]
            self.assertFalse(
                any(a['kind'] == 'jats_section' for a in anchors),
                f"jats_section emitted with incomplete bundle {bundle}",
            )

    def test_bundle_with_all_none_emits_nothing(self):
        """Bundle where every field is None → no taxpub anchors.
        Downstream still gets an empty list (or PDF/Plazi from
        other paths)."""
        bundle = {
            'arpha_uuid': None,
            'mycobank_id': None,
            'first_section_id': None,
            'doi': None,
        }
        anchors = self._build_treatment(bundle).as_row()['source_anchors']
        for kind in ('arpha', 'jats_section', 'mycobank'):
            self.assertFalse(
                any(a['kind'] == kind for a in anchors),
            )

    def test_taxpub_combines_with_pdf_and_plazi(self):
        """Taxpub bundle + PDF page + Plazi UUIDs → all anchors
        emitted together.  Storage order is not part of the
        contract (Django serializer imposes policy)."""
        # Build with all three signal sources.
        ingest = {
            '_id': 'doc123',
            'plazi': {'uuids': ['54496DA047999E86D24CD1B3B71B163D']},
        }
        fileobj = MockFileObject(doc_id='doc123', ingest=ingest)
        fileobj.pdf_page = 3
        fileobj.pdf_label = '3'
        line = Line('[@Foo bar sp. nov.#Nomenclature*]', fileobj)
        para = Paragraph(
            labels=[Label('Nomenclature')],
            lines=[line],
            paragraph_number=1,
        )
        treatment = Treatment()
        treatment.add_nomenclature(para)
        treatment.set_taxpub_anchors({
            'arpha_uuid': 'BA154B2C-A975-5BFF-BEEA-42184807F9D3',
            'mycobank_id': '853632',
            'first_section_id': 'SECID0EGZGK',
            'doi': '10.3897/mycokeys.108.130565',
        })

        anchors = treatment.as_row()['source_anchors']
        kinds = {a['kind'] for a in anchors}
        self.assertEqual(
            kinds,
            {'pdf', 'plazi', 'arpha', 'jats_section', 'mycobank'},
        )
        self.assertEqual(len(anchors), 5)


if __name__ == "__main__":
    unittest.main()
