"""Tests for file.py."""
import textwrap
from typing import List
import unittest

from line import Line
from taxon import Taxon, group_paragraphs
from finder import parse_annotated


def lineify(lines: List[str]) -> List[Line]:
    return [Line(l) for l in lines]


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
                             ['1', '3', '4', '6'])

        self.assertEqual(len(dictionaries2), 4)
        self.assertTrue(all([d['serial_number'] == sn2 for d in dictionaries2]))
        self.assertListEqual([d['paragraph_number'] for d in dictionaries2],
                             ['9', '11', '13', '14'])

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
        self.assertEqual(len(taxa), 2)

        dictionaries1 = list(taxa[0].dictionaries())
        dictionaries2 = list(taxa[1].dictionaries())
        sn1 = dictionaries1[0]['serial_number']
        sn2 = dictionaries2[0]['serial_number']
        self.assertNotEqual(sn1, sn2)

        self.assertEqual(len(dictionaries1), 4)
        self.assertTrue(all([d['serial_number'] == sn1 for d in dictionaries1]))
        self.assertListEqual([d['paragraph_number'] for d in dictionaries1],
                             ['6', '7', '8', '9'])

        self.assertEqual(len(dictionaries2), 4)
        self.assertTrue(all([d['serial_number'] == sn2 for d in dictionaries2]))
        self.assertListEqual([d['paragraph_number'] for d in dictionaries2],
                             ['17', '19', '21', '22'])

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
                             ['1', '2', '3'])
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
                             ['6', '7'])
        self.assertEqual(dictionaries[0]['body'], 'nom2\n')
        self.assertEqual(dictionaries[1]['body'], 'desc1\n')

if __name__ == '__main__':
    unittest.main()
