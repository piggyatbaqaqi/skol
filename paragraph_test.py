"""Tests for paragraph.py."""

import unittest

from line import Line
from paragraph import Paragraph

class TestParagraph(unittest.TestCase):
    def setUp(self):
        self.pp = Paragraph()
        self.pp2 = Paragraph()

    def test_append(self):
        self.pp.append(Line('hamster'))
        self.pp.append(Line('gerbil'))
        got = str(self.pp)
        expected = 'hamster\ngerbil\n'
        self.assertEqual(got, expected)

    def test_append_ahead(self):
        self.pp.append_ahead(Line('hamster'))
        self.pp.append_ahead(Line('gerbil'))
        self.pp.append_ahead(Line('rabbit'))
        got = str(self.pp)
        expected = 'hamster\ngerbil\n'
        self.assertEqual(got, expected)
        self.assertEqual(self.pp.next_line.line, 'rabbit')

    def test_is_figure(self):
        self.pp.append(Line('  Fig. 2.7  '))
        self.pp.append(Line('hamster'))
        self.assertTrue(self.pp.is_figure())
        self.assertFalse(self.pp2.is_figure())

        self.pp2.append(Line('rabbit'))
        self.assertFalse(self.pp2.is_figure())

    def test_is_table(self):
        self.pp.append(Line('  Table 1 '))
        self.pp.append(Line('hamster'))
        self.assertTrue(self.pp.is_table())
        self.assertFalse(self.pp2.is_table())

        self.pp2.append(Line('rabbit'))
        self.assertFalse(self.pp2.is_table())

    def test_is_key(self):
        self.pp.append(Line('Key to Ijuhya species with fasciculate hairs'))
        self.assertTrue(self.pp.is_key())

    def test_last_line(self):
        self.pp.append_ahead(Line('hamster'))
        self.assertEqual(str(self.pp), '\n')
        self.pp.append_ahead(Line('gerbil'))
        self.pp.append_ahead(Line('rabbit'))

        self.assertEqual(self.pp.last_line.line, 'gerbil')
        self.pp.close()
        self.assertEqual(self.pp.last_line.line, 'rabbit')

    def test_next_paragraph(self):
        self.pp.append_ahead(Line('hamster'))
        self.pp.append_ahead(Line('gerbil'))
        self.pp.append_ahead(Line('rabbit'))
        pp, pp2 = self.pp.next_paragraph()
        pp2.close()
        self.assertEqual(str(pp), 'hamster\ngerbil\n')
        self.assertEqual(pp.paragraph_number, 0)
        self.assertEqual(str(pp2), 'rabbit\n')
        self.assertEqual(pp2.paragraph_number, 1)


if __name__ == '__main__':
    unittest.main()