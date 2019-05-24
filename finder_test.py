import finder
from finder import Label, Line, Paragraph
import textwrap
from typing import List
import unittest

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
        self.assertEqual(self.pp.next_line().line(), 'rabbit')

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

    def test_last_line(self):
        self.pp.append_ahead(Line('hamster'))
        self.assertEqual(str(self.pp), '\n')
        self.pp.append_ahead(Line('gerbil'))
        self.pp.append_ahead(Line('rabbit'))

        self.assertEqual(self.pp.last_line.line(), 'gerbil')
        self.pp.close()
        self.assertEqual(self.pp.last_line.line(), 'rabbit')


class TestLine(unittest.TestCase):
    def test_line(self):
        data = '[@New records of smut fungi. 4. Microbotryum coronariae comb. nov.#Title*]'
        line = Line(data)

        self.assertEqual(line.line(), 'New records of smut fungi. 4. Microbotryum coronariae comb. nov.')
        self.assertTrue(line.contains_start())
        self.assertEqual(line.end_label(), 'Title')
        self.assertFalse(line.is_short(50))
        self.assertFalse(line.is_blank())


    def test_middle_start(self):
        test_data = textwrap.dedent("""\
        multiformibus ornata. [@Habitat in herbidis locis.#Habitat-distribution*]
        """).split('\n')
        with self.assertRaisesRegex(ValueError, r'Label open not at start of line: [^:]+:[0-9]+:'):
            lineify(test_data)

    def test_middle_end(self):
        test_data = textwrap.dedent("""\
        multiformibus ornata.#Description*] Habitat in herbidis locis
        """).split('\n')

        with self.assertRaisesRegex(ValueError, r'Label close not at end of line: [^:]+:[0-9]+:'):
            lineify(test_data)


class TestParser(unittest.TestCase):

    def test_regression1(self):
        test_data = lineify(textwrap.dedent("""\
        ISSN (print) 0093-4666

        © 2011. Mycotaxon, Ltd.

        ISSN (online) 2154-8889

        MYCOTAXON
        Volume 118, pp. 273–282

        http://dx.doi.org/10.5248/118.273""").split('\n'))

        paragraphs = list(finder.parse_paragraphs(test_data))
        self.assertEqual(len(paragraphs), 10)

    def test_initial_no_break(self):
        """A single initial (letter followed by .) prevents a paragraph break."""
        test_data = lineify(textwrap.dedent("""\
        Stenellopsis nepalensis is the fourth species of the genus to be
        discovered. Stenellopsis was established to accommodate a single species, S.
        fagraeae Huguenin, occurring on leaves of Fagraea schlechteri Gilg & Benedict
        in New Caledonia (Huguenin, 1966). A second species was added to the genus
        when Singh (1979) described S. shoreae S.M. Singh[as S. shorae], found on
        """).split('\n'))
        expected0 = textwrap.dedent("""\
        Stenellopsis nepalensis is the fourth species of the genus to be
        discovered. Stenellopsis was established to accommodate a single species, S.
        fagraeae Huguenin, occurring on leaves of Fagraea schlechteri Gilg & Benedict
        in New Caledonia (Huguenin, 1966). A second species was added to the genus
        when Singh (1979) described S. shoreae S.M. Singh[as S. shorae], found on
        """)
        paragraphs = list(finder.parse_paragraphs(test_data))
        self.assertEqual(str(paragraphs[0]), expected0)
        
    def test_page_break(self):
        test_data = lineify(textwrap.dedent("""\
        [@GERMANY: Freiburg, on Betula, 21 IV 1916, Lettau s. n. (B); Westfalen,
        Wolbeck, on Betula, Bellebaum s. n. (B, 2 specimens); Frankfurt, on Betula,
        6#Misc-exposition*]

        [@Metzler s. n. (B); Bonn, on Betula, Dreesen s. n. (B); Heidelberg, on Betula,
        Zwackh s. n. (B); 'Rubbia', on Quercus, Stricker s. n., distributed in Koerber,
        Lichenes selecti Germaniae 410 (B).#Misc-exposition*]
        """).split('\n'))

        expected0 = textwrap.dedent("""\
        GERMANY: Freiburg, on Betula, 21 IV 1916, Lettau s. n. (B); Westfalen,
        Wolbeck, on Betula, Bellebaum s. n. (B, 2 specimens); Frankfurt, on Betula,
        """)
        expected1 = textwrap.dedent("""\
        6
        """)
        expected3 = textwrap.dedent("""\
        Metzler s. n. (B); Bonn, on Betula, Dreesen s. n. (B); Heidelberg, on Betula,
        Zwackh s. n. (B); 'Rubbia', on Quercus, Stricker s. n., distributed in Koerber,
        Lichenes selecti Germaniae 410 (B).
        """)
        paragraphs = list(finder.parse_paragraphs(test_data))
        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(str(paragraphs[1]), expected1)
        self.assertEqual(str(paragraphs[2]), '\n')
        self.assertEqual(str(paragraphs[3]), expected3)

    def test_year_in_parens_break(self):
        test_data = lineify(textwrap.dedent("""\
        [@Pertusaria persulphurata Müll.Arg., Nuovo Giorn. Bot. Ital. 23: 391 (1891)#Taxonomy*]
        [@Type: AUSTRALIA, Queensland, Brisbane, F.M. Bailey s.n.; holo: G.#Misc-exposition*]
        """).split('\n'))

        expected0 = textwrap.dedent("""\
        Pertusaria persulphurata Müll.Arg., Nuovo Giorn. Bot. Ital. 23: 391 (1891)
        """)
        expected1 = textwrap.dedent("""\
        Type: AUSTRALIA, Queensland, Brisbane, F.M. Bailey s.n.; holo: G.
        """)

        paragraphs = list(finder.parse_paragraphs(test_data))
        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(str(paragraphs[1]), expected1)

    def test_syn_break(self):
        test_data = lineify(textwrap.dedent("""\
        [@Arthonia apatetica (A. Massal.) Th. Fr. (Syn. A. exilis auct.)#Taxonomy*]
        [@GRAHAM ISLAND: 2 mi. W of Tow Hill (Yakan Point) on north shore,
        54°04’N 131°50’W, 15 June 1967, Brodo 9896R.#Misc-exposition*]
        """).split('\n'))

        expected0 = textwrap.dedent("""\
        Arthonia apatetica (A. Massal.) Th. Fr. (Syn. A. exilis auct.)
        """)

        expected1 = textwrap.dedent("""\
        GRAHAM ISLAND: 2 mi. W of Tow Hill (Yakan Point) on north shore,
        54°04’N 131°50’W, 15 June 1967, Brodo 9896R.
        """)

        paragraphs = list(finder.parse_paragraphs(test_data))
        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(str(paragraphs[1]), expected1)

    def test_hyphen_break(self):
        test_data = lineify(textwrap.dedent("""\
        [@*Marasmius ferrugineus (Berkeley*)Berkeley & Curtis
        -Leprieur 1025 ; cited in Montagne 1854#Misc-exposition*]
        """).split('\n'))

        expected0 = "*Marasmius ferrugineus (Berkeley*)Berkeley & Curtis\n"
        expected1 = "-Leprieur 1025 ; cited in Montagne 1854\n"

        paragraphs = list(finder.parse_paragraphs(test_data))
        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(str(paragraphs[1]), expected1)

    def test_abbrev_non_break(self):
        test_data = lineify(textwrap.dedent("""\
        [@As most of the species considered have previously been described in detail in
        other revisional treatments, including those of SHEARD (1967), MAYRHOFER &
        POELT (1979), MAYRHOFER (1984), Fox & Purvis (1992), MAYRHOFER et al.#Misc-exposition*]
        [@(1993) and MARZER et al. (1994), the descriptions given here simply
        emphasize characters of value for species identification and do not repeat
        features common to all species.#Misc-exposition*]
        [@48#Misc-exposition*]
        """).split('\n'))

        expected0 = textwrap.dedent("""\
        As most of the species considered have previously been described in detail in
        other revisional treatments, including those of SHEARD (1967), MAYRHOFER &
        POELT (1979), MAYRHOFER (1984), Fox & Purvis (1992), MAYRHOFER et al.
        (1993) and MARZER et al. (1994), the descriptions given here simply
        emphasize characters of value for species identification and do not repeat
        features common to all species.
        """)
        expected1 = textwrap.dedent("""\
        48
        """)

        paragraphs = list(finder.parse_paragraphs(test_data))
        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(str(paragraphs[1]), expected1)

    def test_single_character_non_break(self):
        test_data = lineify(textwrap.dedent("""\
        [@MEXICO: Baja California, Cerro Kenton, on Euphorbia, 4 | 1989, A. & M.#Misc-exposition*]
        [@Aptroot 24428 (Herb. Aptroot); Same locality, on shrub, 4 I 1989, A. & M. Aptroot
        24455 (Herb. Aptroot).#Misc-exposition*]
        """).split('\n'))

        expected0 = textwrap.dedent("""\
        MEXICO: Baja California, Cerro Kenton, on Euphorbia, 4 | 1989, A. & M.
        Aptroot 24428 (Herb. Aptroot); Same locality, on shrub, 4 I 1989, A. & M. Aptroot
        24455 (Herb. Aptroot).
        """)
        paragraphs = list(finder.parse_paragraphs(test_data))
        self.assertEqual(str(paragraphs[0]), expected0)


    def test_table(self):
        test_data = lineify(textwrap.dedent("""\
        Table 1.

        short
        shorter
        long
        longer

        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore""").split('\n'))

        expected0 = textwrap.dedent("""\
        Table 1.

        short
        shorter
        long
        longer
        """)

        expected1 = textwrap.dedent("""\
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore""")

        paragraphs = list(finder.parse_paragraphs(test_data))

        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(str(paragraphs[1]), expected1)

    def test_figure(self):
        test_data = lineify(textwrap.dedent("""\
          Fig 1. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
        tempor incididunt ut labore et dolore

        magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
        nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit
        in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

        Figure 2. Excepteur sint occaecat cupidatat non proident,

        Photo 1. culpa qui officia deserunt mollit anim id est laborum.
        """).split('\n'))

        expected0 = textwrap.dedent("""\
        Fig 1. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
        tempor incididunt ut labore et dolore
        """)

        expected2 = textwrap.dedent("""\
        magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
        nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit
        in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
        """)

        expected4 = textwrap.dedent("""\
        Figure 2. Excepteur sint occaecat cupidatat non proident,
        """)

        expected6 = textwrap.dedent("""\
        Photo 1. culpa qui officia deserunt mollit anim id est laborum.
        """)

        paragraphs = list(finder.parse_paragraphs(test_data))

        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertTrue(paragraphs[0].is_figure())
        self.assertEqual(str(paragraphs[1]), '\n')
        self.assertEqual(str(paragraphs[2]), expected2)
        self.assertFalse(paragraphs[2].is_figure())
        self.assertEqual(str(paragraphs[3]), '\n')
        self.assertEqual(str(paragraphs[4]), expected4)
        self.assertTrue(paragraphs[4].is_figure())
        self.assertEqual(str(paragraphs[5]), '\n')
        self.assertEqual(str(paragraphs[6]), expected6)
        self.assertTrue(paragraphs[6].is_figure())
        self.assertEqual(str(paragraphs[7]), '\n')

    def test_table(self):
        test_data = lineify(textwrap.dedent("""\
          Table 1. Lorem ipsum dolor sit amet, consectetur adipiscing
        elit, sed do eiusmod tempor
        incididunt ut labore et dolore
        magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
        nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit
        in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
        Tbl. 2. Excepteur sint occaecat cupidatat non proident,
        """).split('\n'))

        expected0 = textwrap.dedent("""\
        Table 1. Lorem ipsum dolor sit amet, consectetur adipiscing
        elit, sed do eiusmod tempor
        incididunt ut labore et dolore
        """)
        expected1 = textwrap.dedent("""\
        magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
        nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit
        in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
        """)
        # Final table ends with two newlines.
        expected2 = textwrap.dedent("""\
        Tbl. 2. Excepteur sint occaecat cupidatat non proident,

        """)

        paragraphs = list(finder.parse_paragraphs(test_data))

        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertTrue(paragraphs[0].is_table())
        self.assertEqual(str(paragraphs[1]), expected1)
        self.assertFalse(paragraphs[1].is_table())
        self.assertEqual(str(paragraphs[2]), expected2)
        self.assertTrue(paragraphs[2].is_table())


class TestLabeling(unittest.TestCase):

    def test_simple_label(self):
        test_data = lineify(textwrap.dedent("""\
        [@Abstract — In this ﬁrst of a series of three papers, new combinations in the genus
        Lactiﬂuus are proposed. This paper treats the subgenera Edules, Lactariopsis, and Russulopsis
        (all proposed here as new combinations in Lactiﬂuus). In Lactiﬂuus subg. Edules, eight
        combinations at species level are proposed. In Lactiﬂuus subg. Lactariopsis, the following
        three new combinations are proposed at sectional level: Lactiﬂuus sect. Lactariopsis with
        seven newly combined species, L. sect. Chamaeleontini with eight newly combined species,
        and L. sect. Albati with four newly combined species plus two species previously combined
        in Lactiﬂuus. Finally, in L. subg. Russulopsis, eight new combinations at species level are
        proposed.#Abstract*]
        [@Key words — milkcaps, nomenclature#Key-words*]
        """).split('\n'))

        expected0 = textwrap.dedent("""\
        Abstract — In this ﬁrst of a series of three papers, new combinations in the genus
        Lactiﬂuus are proposed. This paper treats the subgenera Edules, Lactariopsis, and Russulopsis
        (all proposed here as new combinations in Lactiﬂuus). In Lactiﬂuus subg. Edules, eight
        combinations at species level are proposed. In Lactiﬂuus subg. Lactariopsis, the following
        three new combinations are proposed at sectional level: Lactiﬂuus sect. Lactariopsis with
        seven newly combined species, L. sect. Chamaeleontini with eight newly combined species,
        and L. sect. Albati with four newly combined species plus two species previously combined
        in Lactiﬂuus. Finally, in L. subg. Russulopsis, eight new combinations at species level are
        proposed.
        """)

        expected1 = textwrap.dedent("""\
        Key words — milkcaps, nomenclature
        """)

        paragraphs = list(finder.parse_paragraphs(test_data))

        self.maxDiff = None
        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(paragraphs[0].labels, [Label('Abstract')])
        self.assertEqual(str(paragraphs[1]), expected1)
        self.assertEqual(paragraphs[1].labels, [Label('Key-words')])

    def test_doubled_abstract(self):

        test_data = lineify(textwrap.dedent("""\
        [@New records of smut fungi. 4. Microbotryum coronariae comb. nov.#Title*]
        [@Cvetomir M. Denchev & Teodor T. Denchev#Author*]
        [@Institute of Biodiversity and Ecosystem Research, Bulgarian Academy of Sciences,
        2 Gagarin St., 1113 Soﬁa, Bulgaria#Institution*]
        * Correspondence to: cmdenchev@yahoo.co.uk
        [@Abstract — For Ustilago coronariae on Lychnis ﬂos-cuculi, a new combination in
        Microbotryum, M. coronariae, is proposed. It is reported as new to Bulgaria.#Abstract*]
        [@Key words — Microbotryaceae, taxonomy#Key-words*]
        """).split('\n'))

        paragraphs = list(finder.parse_paragraphs(test_data))

        self.assertEqual(len(paragraphs), 7)
        self.assertEqual(paragraphs[0].labels, [Label('Title')])
        self.assertEqual(paragraphs[1].labels, [Label('Author')])
        self.assertEqual(paragraphs[2].labels, [Label('Institution')])
        self.assertEqual(paragraphs[3].labels, [])  # correspondence
        self.assertEqual(paragraphs[4].labels, [Label('Abstract')])
        self.assertEqual(paragraphs[5].labels, [Label('Key-words')])
        self.assertEqual(paragraphs[6].labels, []) # Paragraph('\n')


class TestTargetClasses(unittest.TestCase):

    def test_target_classes(self):
        test_data = lineify(textwrap.dedent("""\
        [@New records of smut fungi. 4. Microbotryum coronariae comb. nov.#Title*]
        [@Abstract — In this ﬁrst of a series of three papers, new combinations in the genus
        Lactiﬂuus are proposed. This paper treats the subgenera Edules, Lactariopsis, and Russulopsis
        (all proposed here as new combinations in Lactiﬂuus). In Lactiﬂuus subg. Edules, eight
        combinations at species level are proposed. In Lactiﬂuus subg. Lactariopsis, the following
        three new combinations are proposed at sectional level: Lactiﬂuus sect. Lactariopsis with
        seven newly combined species, L. sect. Chamaeleontini with eight newly combined species,
        and L. sect. Albati with four newly combined species plus two species previously combined
        in Lactiﬂuus. Finally, in L. subg. Russulopsis, eight new combinations at species level are
        proposed.#Abstract*]
        [@Key words — milkcaps, nomenclature#Key-words*]

        [@Tulostoma exasperatum Mont., Ann. Sci. Nat., Bot., Sér. 2, 8: 362. 1837.#Taxonomy*]
        [@Basidiomata 1.1–7.0 cm high. Spore sac globose to depressed-globose,
        0.6–0.8 cm high × 1.8–2.2 cm broad. Exoperidium spiny, light brown (5E7),
        peeling oﬀ at maturity. Endoperidium reticulate, papery, yellowish white (2A2)
        to pale yellow (4A3); peristome conical, slightly lighter than endoperidium,
        ﬁbrillose, delimited. Gleba dull yellow (3B3). Stipe 0.9–6.1 cm high × 0.2–0.25
        cm diam., light brown (5E7), with longitudinally arranged scales.

        [@Gasteroids from the Amazon (Brazil) ... 279#Page-header*]

        [@Fig. 3. Gasteroid species from the Brazilian Amazon rainforest.
        A. Geastrum lageniforme. B. Tulostoma exasperatum. C. Mutinus caninus.#Figure*]

        Basidiospores globose to subglobose, 6–7.5 μm diam., yellowish in KOH,
        with a columnar-reticulate ornamentation. Capillitial hyphae straight to
        tortuous, thick-walled, swollen at the septa, branched, light yellow in KOH,
        4–7 μm diam.#Description*]

        We thank the people of Paiter; ‘Associação Metareilá do Povo Indígena Suruí’; ‘Equipe
        de Conservação da Amazônia – ACT Brasil’; ‘Associação de Defesa Etnoambiental
        Kanindé’ and ‘Fundação Nacional do Índio (FUNAI)’. The United States Agency for
        International Development (USAID) is acknowledged for ﬁnancial support. We
        also would like to thank the curators of SP, ICN and MBM for specimen loans
        """).split('\n'))

        labels_before = [
            Label('Title'), Label('Abstract'), Label('Key-words'),
            None, Label('Taxonomy'), Label('Description'), Label('Description'),
            Label('Page-header'), Label('Description'),
            Label('Figure'), Label('Figure'),
            Label('Description'), Label('Description'), None, None, None,
        ]
        labels_after = [
            Label('Misc-exposition'), Label('Misc-exposition'),
            Label('Misc-exposition'), Label('Taxonomy'), Label('Description'),
            Label('Misc-exposition'), Label('Misc-exposition'),
            Label('Description'), Label('Misc-exposition'),
        ]

        phase1 = list(finder.parse_paragraphs(test_data))
        self.assertListEqual([pp.top_label() for pp in phase1], labels_before)

        phase2 = list(finder.remove_interstitials(phase1))
        self.assertEqual(len(phase2), 9)

        phase3 = list(finder.target_classes(
            phase2,
            default=Label('Misc-exposition'),
            keep=[Label('Taxonomy'), Label('Description')]
        ))
        self.assertListEqual([pp.top_label() for pp in phase3], labels_after)


if __name__ == '__main__':
    unittest.main()
