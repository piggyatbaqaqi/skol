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
        self.taxon = Taxon()
        self.taxon.LONG_GAP = 5  # Give up faster than in real conditions.

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


if __name__ == '__main__':
    unittest.main()
