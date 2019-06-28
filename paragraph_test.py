"""Tests for paragraph.py."""

import textwrap
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

    def test_contains_nomenclature(self):
        self.pp.append_ahead(Line('hamster'))
        self.pp.append_ahead(
            Line('≡ Polyporus mori (Pollini) Fr., Systema Mycologicum 1:'))
        self.pp.append_ahead(Line('344 (1821)'))
        self.pp.close()
        self.assertTrue(self.pp.contains_nomenclature())

    def test_split_at_nomenclature(self):
        self.pp.append_ahead(Line('hamster'))
        self.pp.append_ahead(
            Line('≡ Polyporus mori (Pollini) Fr., Systema Mycologicum 1:'))
        self.pp.append_ahead(Line('344 (1821)'))
        self.pp.append_ahead(Line('gerbil'))
        result = self.pp.split_at_nomenclature()
        self.assertEqual(str(self.pp), 'hamster\n')
        self.assertEqual(
            str(result),
            '≡ Polyporus mori (Pollini) Fr., Systema Mycologicum 1:\n'
            '344 (1821)\n')
        self.assertEqual(result.next_line.line, 'gerbil')

    def test_split_at_nomenclature_rainy_day(self):
        result = self.pp.split_at_nomenclature()
        self.pp.append_ahead(Line('hamster'))
        self.pp.append_ahead(Line('gerbil'))
        self.pp.append_ahead(Line('rabbit'))
        self.assertFalse(self.pp.split_at_nomenclature())

class TestReinterpret(unittest.TestCase):

    def setUp(self):
        self.pp = Paragraph()
        self.pp.append_ahead(Line('Julella sublactea (Nylander) R.C. Harris in Egan, Bryologist 90: 163. 1987;\n'))
        self.pp.append_ahead(Line('Verrucaria sublactea Nylander, Flora 69: 464. 1886. syn. nov.\n'))
        self.pp.close()

    def test_latinate(self):
        self.pp.set_reinterpretations(['latinate'])
        got = self.pp.reinterpret()
        expected = (
            ' PLATINATE   PLATINATE  (Nylander) R.C.  PLATINATE  in Egan, Bryologist 90: 163. 1987;\n'
            ' PLATINATE   PLATINATE  Nylander, Flora 69: 464. 1886. syn. nov.\n'
        )
        self.assertEqual(got, expected)

    def test_suffix(self):
        self.pp.set_reinterpretations(['suffix'])
        got = self.pp.reinterpret()
        expected = (
            ' ella   ea  (Nylander) R.C.  is  in Egan, Bryologist 90: 163. 1987;\n'
            ' ia   ea  Nylander, Flora 69: 464. 1886. syn. nov.\n'
        )
        self.assertEqual(got, expected)

    def test_latinate_suffix(self):
        self.pp.set_reinterpretations(['latinate', 'suffix'])
        got = self.pp.reinterpret()
        expected = (
            ' PLATINATE ella   PLATINATE ea  (Nylander) R.C.  PLATINATE is  in Egan, Bryologist 90: 163. 1987;\n'
            ' PLATINATE ia   PLATINATE ea  Nylander, Flora 69: 464. 1886. syn. nov.\n'
        )
        self.assertEqual(got, expected)

    def test_punctuation(self):
        self.pp.set_reinterpretations(['punctuation'])
        got = self.pp.reinterpret()
        expected = textwrap.dedent("""\
        Julella sublactea  PLPAREN Nylander PRPAREN  R PDOT C PDOT  Harris in Egan PCOMMA  Bryologist 90 PCOLON  163 PDOT  1987 PSEMI\x20
        Verrucaria sublactea Nylander PCOMMA  Flora 69 PCOLON  464 PDOT  1886 PDOT  syn PDOT  nov PDOT\x20
        """)

        self.assertEqual(got, expected)

    def test_year(self):
        self.pp.set_reinterpretations(['year'])
        got = self.pp.reinterpret()
        expected = textwrap.dedent("""\
        Julella sublactea (Nylander) R.C. Harris in Egan, Bryologist 90: 163.  PYEAR ;
        Verrucaria sublactea Nylander, Flora 69: 464.  PYEAR . syn. nov.
        """)

        self.assertEqual(got, expected)

    def test_abbrev(self):
        self.pp.set_reinterpretations(['abbrev'])
        got = self.pp.reinterpret()
        expected = textwrap.dedent("""\
        Julella sublactea (Nylander)  PABBREV  PABBREV  Harris in Egan, Bryologist 90: 163. 1987;
        Verrucaria sublactea Nylander, Flora 69: 464. 1886.  PABBREV   PABBREV\x20
        """)

        self.assertEqual(got, expected)



if __name__ == '__main__':
    unittest.main()
