"""Tests for paragraph.py."""

import textwrap
from typing import List
import unittest

from line import Line
from paragraph import Paragraph

def lineify(lines: List[str]) -> List[Line]:
    return [Line(l) for l in lines]

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

        pp2 = Paragraph()
        pp2.append(Line('Araneosa columellata Long, Mycologia 33 (1941) 353.'))
        self.assertTrue(pp2.contains_nomenclature())

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

    def test_contains_nomenclature_bulk(self):
        test_data = [
            '    Bovista hyalothrix Cooke & Massee, Grevillea 16 (1888) 73.\n',

            '    Catastoma hyalothrix (Cooke & Massee) Lloyd, The Lycoperdaceae of Australia, New Zealand\n'
            ' and Neighbouring Islands (1905) 27.\n',

            '    Disciseda pedicellata (Morgan) Hollós, Természetrajzi Füz. 25 (1902) 103.\n',

            '4.  Entoloma indigoticoumbrinum G. Gates & Noordel., spec. nov. — Fig. 4, Plate 4\n',

            '8. Entoloma contrastans G. Gates & Noordel., spec. nov. — Fig. 8, Plate 8\n',

            '11. Entoloma obscureotenax G. Gates & Noordel., spec. nov. — Fig. 11, Plate 11\n',

            '15.	Entoloma fuligineopallescens G. Gates & Noordel., spec. nov. — Fig. 15,\n',

            '19. Entoloma austroprunicolor G. Gates & Noordel., spec. nov. — Fig. 19, Plate 19\n',

            '21. Entoloma carminicolor G. Gates & Noordel., spec. nov. — Fig. 21, Plate 21\n',

            '22. Entoloma obscureovirens G. Gates & Noordel., spec. nov. — Fig. 22, Plate 22\n',

            '23. Entoloma albidosimulans G. Gates & Noordeloos, spec. nov. — Fig. 23, Plate 23\n',

            '.	Entoloma stramineopallescens G. Gates & Noordel., spec. nov. — Fig. 27,\n',

            '29.         Entoloma tomentosolilacinum G. Gates & Noordel., spec. nov. — Fig. 29,\n',

            '31. Entoloma austrorhodocalyx G. Gates & Noordel., spec. nov. — Fig. 31, Plate 31\n',

            '≡ Xenasma macrosporum Liberta, Mycologia 52 (6) (1962 ‘1960’) 899.\n',

            '≡ Epithele macrospora (Liberta) Boquiren, Mycologia 63 (5) (1971) 949.\n',

            'Chamaeota pusilla (Pat. & Gaillard) Beardslee, Mycologia 26 (1934) 254\n',

            'Hygrocybe comosa Bas & Arnolds, spec. nov. — Plate 1, Figs. 1–3\n',

            'Diplodina lycopersici Hollós, Annls hist.-nat. Mus. natn. hung. 5 (1907) 461.\n',

            'Phyllosticta lycii Ellis & Kellerm., Am. Nat. 17 (1883) 1166.\n'

            'Pleurotus djamor (Rumph. apud Fr.) Boedijn, in: H.C.D. de Wit (ed.), Rumphius\n'
            'Memorial Vol. (1959) 292.\n',

            'Comatricha nigricapillitium (Nann.-Bremek. & Bozonnet) A. Castillo, G. Moreno & Illana,\n'
            'Mycol. Res. 101 (1997) 1331.\n',

            '= Collaria chionophila Lado, Anales Jard. Bot. Madrid 50 (1992) 9 & 11.\n',

            '= Lepidoderma chailletit Rostaf., Sluzowce Monogr. (1874) 189\n',

            'Physarum alpestre Mitchel, S.W, Chapm. & M.L. Farr, Mycologia 78 (1986) 68.\n',

            'Trichia sordida var. sordida Johannesen, Mycotaxon 20 (1984) 81-82.\n',

            '- Trichia bicolor S.L. Stephenson & M.L. Farr, Mycologia 82 (1990) 513.\n',

            'z Trichia contorta var. engadinensis Meyl., Bull. Soc. Vaud. Sci. Nat. 53 (1921) 460.\n',

            'Pseudobaeospora dichroa forma cystidiata Bas, forma nov.\n',

            'Pseudobaeospora oligophylla! (Singer) Singer, Lilloa 22 (*1949") (1951) 438 ; Baeo-\n'
            'spora oligophylla Singer, Rev. Mycol. 3 (1938) 194.\n',

            '|. Clasterosporium cyperacearum Hosag., spec. nov. — Fig.1\n',

            'Balladyna uncinata Syd., Ann. Mycol. 12 (1914) 546.\n',

            'Kusanobotrys bambusae Hino & Katum., Bull. Yamaguti Univ. 5 (1954) 218.\n',

            'Erysiphe hellebori Rankovic, spec. nov.\n',

            '13. Pseudobaeospora subglobispora, nom. prov. — Fig. 13\n',

            'Catastoma pedicellatum Morgan, J. Cincinn. Soc. Nat. Hist. 14 (1892) 143-144.\n',

            'Disciseda arida Nelen., Novit. Mycol. (1939) 169.\n',

            'Basionym: Galera rickenii Schaeff., Z. Pilzk. 6 (1930) 171.\n',

            'Comatricha alpina Kowalski, Madrono 22 (1973) 152.\n',

            '7 Comatricha suksdorfii Elis & Everh. var. aggregata Meyl., Bull. Soc. Vaud. Sci. Nat. 53 (1921)\n'
            '455.\n',

            'Comatricha anastomosans Kowalski, Mycologia 64 (1972) 362.\n',

            'Comatricha filamentosa Meyl., Bull. Soc. Vaud. Sci. Nat. 53 (1921) 456.\n',

            'Hebeloma hiemale Bres., Fung. trident. 2 (1892) 52.\n',

            'Hebeloma crustuliniforme var. tiliae Bresinsky, Z. Mykol. 53 (1987) 294.\n',

            'Galerina acuta Barkman nom. prov., Coolia 14 (1969) 62 — Fig. 1\n',
        ]

        # Build paragraphs.
        test_pp = []
        for text in test_data:
            pp = Paragraph()
            for l in lineify(text.split('\n')):
                pp.append(l)
            test_pp.append(pp)

        for pp in test_pp:
            if not pp.contains_nomenclature():
                print('Failed paragraph:', str(pp))
            self.assertTrue(pp.contains_nomenclature())

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
