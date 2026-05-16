"""Tests for treatment.py."""

import textwrap
from typing import List
import unittest

from label import Label
from line import Line
from paragraph import Paragraph
from treatment import Treatment, group_paragraphs
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
        self.assertEqual(dictionaries2[0]["body"], "Nomen undetected\n")
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
        a stub Nomenclature paragraph with 'Nomen undetected' should be automatically
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
        self.assertEqual(dictionaries[0]["body"], "Nomen undetected\n")
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
        self.assertEqual(dictionaries1[0]["body"], "Nomen undetected\n")
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

        'Nomen undetected' stubs are synthetic Lines with no fileobj, so
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

        nom_stub = Line("Nomen undetected")  # no fileobj — ingest is None
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


if __name__ == "__main__":
    unittest.main()
