"""Tests for file.py."""
import textwrap
from typing import List
import unittest

from line import Line
from taxon import Taxon, group_paragraphs
from finder import parse_annotated


def lineify(lines: List[str]) -> List[Line]:
    return [Line(l) for l in lines]


class MockFileObject:
    """Mock file object for testing with doc_id support."""
    def __init__(self, doc_id: str = None, filename: str = "test.txt"):
        self.doc_id = doc_id
        self.filename = filename
        self.line_number = 1
        self.page_number = 1
        self.pdf_page = 0
        self.pdf_label = None
        self.empirical_page_number = None
        self._empirical_page_number = None

    def _set_empirical_page(self, line: str) -> None:
        """Mock implementation of empirical page extraction."""
        import regex as re
        match = re.search(
            r'(^\s*(?P<leading>[mdclxvi\d]+\b))|((?P<trailing>\b[mdclxvi\d]+)\s*$)',
            line
        )
        if not match:
            self._empirical_page_number = None
        else:
            self._empirical_page_number = (
                match.group('leading') or match.group('trailing')
            )
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


class TestTaxon(unittest.TestCase):

    def setUp(self):
        Taxon.LONG_GAP = 3  # Give up faster than in real conditions.

    def test_sunny(self):
        test_data = lineify(textwrap.dedent("""\
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
        """).split('\n'))

        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 2)
        dictionaries1 = list(taxa[0].dictionaries())
        dictionaries2 = list(taxa[1].dictionaries())
        sn1 = dictionaries1[0]['serial_number']
        sn2 = dictionaries2[0]['serial_number']
        self.assertNotEqual(sn1, sn2)

        self.assertEqual(len(dictionaries1), 4)
        self.assertTrue(all([d['serial_number'] == sn1 for d in dictionaries1]))
        self.assertListEqual([d['paragraph_number'] for d in dictionaries1],
                             [1, 3, 4, 6])

        self.assertEqual(len(dictionaries2), 4)
        self.assertTrue(all([d['serial_number'] == sn2 for d in dictionaries2]))
        self.assertListEqual([d['paragraph_number'] for d in dictionaries2],
                             [9, 11, 13, 14])

        dict0 = dictionaries1[0]
        dict2 = dictionaries1[2]
        self.assertEqual(dict0['body'], 'paragraph1\n')
        self.assertEqual(dict0['label'], 'Nomenclature')
        self.assertEqual(dict2['body'], 'paragraph4\n')
        self.assertEqual(dict2['label'], 'Description')

        dict4 = dictionaries2[0]
        dict7 = dictionaries2[3]
        self.assertEqual(dict4['body'], 'paragraph9\n')
        self.assertEqual(dict4['label'], 'Nomenclature')
        self.assertEqual(dict7['body'], 'paragraph15\n')
        self.assertEqual(dict7['label'], 'Description')


    def test_too_long(self):
        test_data = lineify(textwrap.dedent("""\
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
        """).split('\n'))

        taxa = list(group_paragraphs(parse_annotated(test_data)))
        # Now expecting 3 taxa due to stub nomenclature creation for bare Description
        # Taxon 1: paragraph1+3 (nomenclatures) + paragraph4+6 (descriptions)
        # Taxon 2: stub + ignored2 (description that was too far from nomenclature)
        # Taxon 3: paragraph9 (nomenclature) + paragraph12+14+15 (descriptions)
        self.assertEqual(len(taxa), 3)

        dictionaries1 = list(taxa[0].dictionaries())
        dictionaries2 = list(taxa[1].dictionaries())
        dictionaries3 = list(taxa[2].dictionaries())
        sn1 = dictionaries1[0]['serial_number']
        sn2 = dictionaries2[0]['serial_number']
        sn3 = dictionaries3[0]['serial_number']
        self.assertNotEqual(sn1, sn2)
        self.assertNotEqual(sn2, sn3)
        self.assertNotEqual(sn1, sn3)

        # First taxon: paragraph1+3 (nomenclatures) + paragraph4+6 (descriptions)
        self.assertEqual(len(dictionaries1), 4)
        self.assertTrue(all([d['serial_number'] == sn1 for d in dictionaries1]))
        self.assertListEqual([d['paragraph_number'] for d in dictionaries1],
                             [6, 7, 8, 9])

        dict0 = dictionaries1[0]
        dict2 = dictionaries1[2]
        self.assertEqual(dict0['body'], 'paragraph1\n')
        self.assertEqual(dict0['label'], 'Nomenclature')
        self.assertEqual(dict2['body'], 'paragraph4\n')
        self.assertEqual(dict2['label'], 'Description')

        # Second taxon: stub + ignored2 (bare description with no preceding nomenclature)
        self.assertEqual(len(dictionaries2), 2)
        self.assertTrue(all([d['serial_number'] == sn2 for d in dictionaries2]))
        self.assertEqual(dictionaries2[0]['body'], 'Nomen undetected\n')
        self.assertEqual(dictionaries2[0]['label'], 'Nomenclature')
        self.assertEqual(dictionaries2[1]['body'], 'ignored2\n')
        self.assertEqual(dictionaries2[1]['label'], 'Description')

        # Third taxon: paragraph9 (nomenclature) + paragraph12+14+15 (descriptions)
        self.assertEqual(len(dictionaries3), 4)
        self.assertTrue(all([d['serial_number'] == sn3 for d in dictionaries3]))
        self.assertListEqual([d['paragraph_number'] for d in dictionaries3],
                             [17, 19, 21, 22])

        dict4 = dictionaries3[0]
        dict7 = dictionaries3[3]
        self.assertEqual(dict4['body'], 'paragraph9\n')
        self.assertEqual(dict4['label'], 'Nomenclature')
        self.assertEqual(dict7['body'], 'paragraph15\n')
        self.assertEqual(dict7['label'], 'Description')

    def test_fall_through_first_description(self):
        """Test fall-through case: first Description after Nomenclature is immediately added.

        When in 'Look for Nomenclatures' state and we encounter a Description paragraph
        after having collected at least one Nomenclature, the state switches to
        'Look for Descriptions' and falls through to immediately add that Description.
        """
        test_data = lineify(textwrap.dedent("""\
        [@nom1#Nomenclature*]
        [@desc1#Description*]
        [@desc2#Description*]
        """).split('\n'))

        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 1, "Should generate exactly one taxon")

        dictionaries = list(taxa[0].dictionaries())
        self.assertEqual(len(dictionaries), 3, "Should have 1 nomenclature + 2 descriptions")

        # Verify the first description was captured (fall-through worked)
        self.assertListEqual([d['paragraph_number'] for d in dictionaries],
                             [1, 2, 3])
        self.assertEqual(dictionaries[0]['label'], 'Nomenclature')
        self.assertEqual(dictionaries[1]['label'], 'Description')
        self.assertEqual(dictionaries[1]['body'], 'desc1\n')

    def test_fall_through_gap_reset(self):
        """Test fall-through case: been_too_long() causes reset in 'Look for Nomenclatures'.

        When in 'Look for Nomenclatures' state and the gap becomes too long,
        the taxon is reset and we continue looking for nomenclatures.
        This prevents incomplete taxa from being yielded.
        """
        test_data = lineify(textwrap.dedent("""\
        [@nom1#Nomenclature*]
        [@filler1#Misc-exposition*]
        [@filler2#Misc-exposition*]
        [@filler3#Misc-exposition*]
        [@filler4#Misc-exposition*]
        [@nom2#Nomenclature*]
        [@desc1#Description*]
        """).split('\n'))

        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 1, "Should generate exactly one taxon")

        dictionaries = list(taxa[0].dictionaries())
        # Should only have nom2 and desc1, not nom1 (it was reset due to gap)
        self.assertEqual(len(dictionaries), 2, "Should have 1 nomenclature + 1 description")
        self.assertListEqual([d['paragraph_number'] for d in dictionaries],
                             [6, 7])
        self.assertEqual(dictionaries[0]['body'], 'nom2\n')
        self.assertEqual(dictionaries[1]['body'], 'desc1\n')

    def test_document_boundary(self):
        """Test that Nomenclature-Description associations do not cross document boundaries.

        When processing multiple documents (different doc_id values), a Nomenclature
        from one document should not be associated with Descriptions from another document.
        The doc_id boundary should cause the current taxon to be yielded and a new one started.
        """
        # Create test data with two different documents
        # Document A (doc_id='doc_a'): Nomenclature at paragraph 1, Description at paragraph 2
        # Document B (doc_id='doc_b'): Description at paragraph 3, Nomenclature at paragraph 4
        test_data = lineify_with_doc_id([
            ('[@nom_from_doc_a#Nomenclature*]', 'doc_a'),
            ('[@desc_from_doc_a#Description*]', 'doc_a'),
            ('[@desc_from_doc_b#Description*]', 'doc_b'),  # Different doc - should NOT associate with nom_from_doc_a
            ('[@nom_from_doc_b#Nomenclature*]', 'doc_b'),
            ('[@desc2_from_doc_b#Description*]', 'doc_b'),
        ])

        taxa = list(group_paragraphs(parse_annotated(test_data)))

        # Should generate 2 taxa, one for each document
        self.assertEqual(len(taxa), 2, "Should generate 2 taxa (one per document)")

        # First taxon: from document A
        dictionaries1 = list(taxa[0].dictionaries())
        self.assertEqual(len(dictionaries1), 2, "First taxon should have nom + desc from doc_a")
        self.assertEqual(dictionaries1[0]['body'], 'nom_from_doc_a\n')
        self.assertEqual(dictionaries1[0]['label'], 'Nomenclature')
        self.assertEqual(dictionaries1[1]['body'], 'desc_from_doc_a\n')
        self.assertEqual(dictionaries1[1]['label'], 'Description')

        # Verify first taxon is from doc_a
        taxon1_row = taxa[0].as_row()
        self.assertEqual(taxon1_row['source']['doc_id'], 'doc_a')

        # Second taxon: from document B
        dictionaries2 = list(taxa[1].dictionaries())
        self.assertEqual(len(dictionaries2), 2, "Second taxon should have nom + desc from doc_b")
        self.assertEqual(dictionaries2[0]['body'], 'nom_from_doc_b\n')
        self.assertEqual(dictionaries2[0]['label'], 'Nomenclature')
        self.assertEqual(dictionaries2[1]['body'], 'desc2_from_doc_b\n')
        self.assertEqual(dictionaries2[1]['label'], 'Description')

        # Verify second taxon is from doc_b
        taxon2_row = taxa[1].as_row()
        self.assertEqual(taxon2_row['source']['doc_id'], 'doc_b')

        # Ensure desc_from_doc_b was NOT associated with nom_from_doc_a
        # (it should have been skipped due to document boundary)
        self.assertNotIn('desc_from_doc_b', [d['body'] for d in dictionaries1])

    def test_document_boundary_while_looking_for_descriptions(self):
        """Test document boundary check in 'Look for Descriptions' state.

        When already collecting descriptions for a nomenclature and we encounter
        a description from a different document, the current taxon should be yielded
        and we should start fresh with the new document.
        """
        test_data = lineify_with_doc_id([
            ('[@nom1#Nomenclature*]', 'doc_a'),
            ('[@desc1_a#Description*]', 'doc_a'),
            ('[@desc2_a#Description*]', 'doc_a'),
            ('[@desc_from_doc_b#Description*]', 'doc_b'),  # Boundary while collecting descriptions
            ('[@nom_from_doc_b#Nomenclature*]', 'doc_b'),
            ('[@desc2_b#Description*]', 'doc_b'),
        ])

        taxa = list(group_paragraphs(parse_annotated(test_data)))

        # Should generate 2 taxa
        self.assertEqual(len(taxa), 2, "Should generate 2 taxa")

        # First taxon: nom1 + desc1_a + desc2_a from doc_a
        dictionaries1 = list(taxa[0].dictionaries())
        self.assertEqual(len(dictionaries1), 3)
        self.assertEqual(dictionaries1[0]['body'], 'nom1\n')
        self.assertEqual(dictionaries1[1]['body'], 'desc1_a\n')
        self.assertEqual(dictionaries1[2]['body'], 'desc2_a\n')

        # desc_from_doc_b should NOT be in first taxon
        self.assertNotIn('desc_from_doc_b', [d['body'] for d in dictionaries1])

        # Second taxon: nom_from_doc_b + desc2_b from doc_b
        dictionaries2 = list(taxa[1].dictionaries())
        self.assertEqual(len(dictionaries2), 2)
        self.assertEqual(dictionaries2[0]['body'], 'nom_from_doc_b\n')
        self.assertEqual(dictionaries2[1]['body'], 'desc2_b\n')

    def test_bare_description_creates_stub_nomenclature(self):
        """Test that a Description without preceding Nomenclature creates a stub.

        When we encounter a Description paragraph without any preceding Nomenclature,
        a stub Nomenclature paragraph with 'Nomen undetected' should be automatically
        created since Descriptions are more reliably detected than Nomenclatures.
        """
        test_data = lineify(textwrap.dedent("""\
        [@desc1#Description*]
        [@desc2#Description*]
        """).split('\n'))

        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 1, "Should generate exactly one taxon")

        dictionaries = list(taxa[0].dictionaries())
        self.assertEqual(len(dictionaries), 3, "Should have 1 stub nomenclature + 2 descriptions")

        # Verify the stub nomenclature was created
        self.assertEqual(dictionaries[0]['label'], 'Nomenclature')
        self.assertEqual(dictionaries[0]['body'], 'Nomen undetected\n')
        self.assertEqual(dictionaries[1]['label'], 'Description')
        self.assertEqual(dictionaries[1]['body'], 'desc1\n')
        self.assertEqual(dictionaries[2]['label'], 'Description')
        self.assertEqual(dictionaries[2]['body'], 'desc2\n')

    def test_bare_description_with_nomenclature_later(self):
        """Test stub creation when bare Description comes before actual Nomenclature.

        First taxon should have stub + descriptions, second taxon should have
        actual nomenclature + its descriptions.
        """
        test_data = lineify(textwrap.dedent("""\
        [@desc1#Description*]
        [@desc2#Description*]
        [@nom1#Nomenclature*]
        [@desc3#Description*]
        """).split('\n'))

        taxa = list(group_paragraphs(parse_annotated(test_data)))
        self.assertEqual(len(taxa), 2, "Should generate 2 taxa")

        # First taxon: stub + desc1 + desc2
        dictionaries1 = list(taxa[0].dictionaries())
        self.assertEqual(len(dictionaries1), 3)
        self.assertEqual(dictionaries1[0]['body'], 'Nomen undetected\n')
        self.assertEqual(dictionaries1[0]['label'], 'Nomenclature')
        self.assertEqual(dictionaries1[1]['body'], 'desc1\n')
        self.assertEqual(dictionaries1[2]['body'], 'desc2\n')

        # Second taxon: nom1 + desc3
        dictionaries2 = list(taxa[1].dictionaries())
        self.assertEqual(len(dictionaries2), 2)
        self.assertEqual(dictionaries2[0]['body'], 'nom1\n')
        self.assertEqual(dictionaries2[0]['label'], 'Nomenclature')
        self.assertEqual(dictionaries2[1]['body'], 'desc3\n')

if __name__ == '__main__':
    unittest.main()
