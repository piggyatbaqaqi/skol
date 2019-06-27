import finder
from finder import Label
from line import Line
from paragraph import Paragraph
import textwrap
from typing import Iterable, List
import unittest


def lineify(lines: List[str]) -> List[Line]:
    return [Line(l) for l in lines]


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

    def test_first_line(self):
        """The first line of many paragraphs is shorter than other lines.

        Confirm that these are included in the paragraph.
        """

        test_data = lineify(textwrap.dedent("""\
        [@Conidia of A. japonica, like those of other members
        of the A. cheiranthi species-group, are essentially beak-
        less. Field specimens give evidence of production of short
        chains of probably 2, at most 3 spores in addition to the
        mostly solitary conidia.#Misc-exposition*]

        [@Field conidia are broadly ellipsoid or ovoid to
        obclavate, with a bluntly rounded apical cell that may
        develop through an abrupt transition into a broad, short l-
        2-celled secondary conidiophore. Largest field conidia
        reach a range of ca. 80-100 x 20-30um, initially are smooth
        but may become evenly pitted or punctate-rough at maturity,
        are a medium translucent tawny brown in color, and have 7-
        10 transverse septa and 1-3 longisepta in several of the
        transverse segments.#Description*]
        """).split('\n'))

        expected0 = textwrap.dedent("""\
        Conidia of A. japonica, like those of other members
        of the A. cheiranthi species-group, are essentially beak-
        less. Field specimens give evidence of production of short
        chains of probably 2, at most 3 spores in addition to the
        mostly solitary conidia.
        """)
        # This second paragraph was having its first line split off.
        expected2 = textwrap.dedent("""\
        Field conidia are broadly ellipsoid or ovoid to
        obclavate, with a bluntly rounded apical cell that may
        develop through an abrupt transition into a broad, short l-
        2-celled secondary conidiophore. Largest field conidia
        reach a range of ca. 80-100 x 20-30um, initially are smooth
        but may become evenly pitted or punctate-rough at maturity,
        are a medium translucent tawny brown in color, and have 7-
        10 transverse septa and 1-3 longisepta in several of the
        transverse segments.
        """)

        paragraphs = list(finder.parse_paragraphs(test_data))
        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(str(paragraphs[2]), expected2)


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

    def test_page_number_no_break(self):
        """An apparent page number prevents a paragraph break."""
        test_data = lineify(textwrap.dedent("""\
        [@= Pseudovalsa nigrofacta (Cooke & Ellis) Cooke, Grevillea 14: 55.
        1885; as "nigrifacta".#Nomenclature*]
        [@= Valsaria clethraecola (Cooke & Ellis) Sacc., Syll. Fung. |, p. 748.
        1882.#Nomenclature*]
        """).split('\n'))

        expected0 = textwrap.dedent("""\
        = Pseudovalsa nigrofacta (Cooke & Ellis) Cooke, Grevillea 14: 55.
        1885; as "nigrifacta".
        """)
        expected1 = textwrap.dedent("""\
        = Valsaria clethraecola (Cooke & Ellis) Sacc., Syll. Fung. |, p. 748.
        1882.
        """)
        paragraphs = list(finder.parse_paragraphs(test_data))
        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(str(paragraphs[1]), expected1)


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
        [@Pertusaria persulphurata Müll.Arg., Nuovo Giorn. Bot. Ital. 23: 391 (1891)#Nomenclature*]
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

    def test_nomenclature_plate(self):
        test_data = lineify(textwrap.dedent("""\
        Hygrocybe comosa Bas & Arnolds, spec. nov. — Plate 1, Figs. 1–3
        \t Pileus 9–19 mm latus, conico-convexus, dein plano-convexus vel depressus, papilla centralis munitus, subhygrophanus, obscure purpureo-griseo-brunneus vel brunneus, dein violaceo-griseo-brunneus,
        substriatus, in sicco pallide brunneo-griseus, superﬁcie sicca, subﬁbrillosa, centro squamulis conicis
        """).split('\n'))
        expected0 = textwrap.dedent("""\
        Hygrocybe comosa Bas & Arnolds, spec. nov. — Plate 1, Figs. 1–3
        """)
        expected1 = textwrap.dedent("""\
        \t Pileus 9–19 mm latus, conico-convexus, dein plano-convexus vel depressus, papilla centralis munitus, subhygrophanus, obscure purpureo-griseo-brunneus vel brunneus, dein violaceo-griseo-brunneus,
        substriatus, in sicco pallide brunneo-griseus, superﬁcie sicca, subﬁbrillosa, centro squamulis conicis
        """)

        paragraphs = list(finder.parse_paragraphs(test_data))
        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(str(paragraphs[1]), expected1)

    def test_tab_break(self):
        test_data = lineify(textwrap.dedent("""\
        Collybia-like habit, often depressed, distinctly squamulose pileus; and Omphalinoid,
        with a depressed pileus and decurrent lamellae.
        \t The current infrageneric taxonomy of Entoloma (Romagnesi & Gilles, 1979; Noorde­
        loos, 1992, 2005; Largent, 1994) is primarily based on European, North American, and
        Berkeley (1859), Cleland (1934, 1935), Stevenson (1962), Horak (1973, 1976, 1977,
        1980, 1982) and Grgurinovic (1997), no attempt has been made so far to place them
        into an infrageneric context.
        """).split('\n'))

        expected0 = textwrap.dedent("""\
        Collybia-like habit, often depressed, distinctly squamulose pileus; and Omphalinoid,
        with a depressed pileus and decurrent lamellae.
        """)
        expected1 = textwrap.dedent("""\
        \t The current infrageneric taxonomy of Entoloma (Romagnesi & Gilles, 1979; Noorde­
        loos, 1992, 2005; Largent, 1994) is primarily based on European, North American, and
        Berkeley (1859), Cleland (1934, 1935), Stevenson (1962), Horak (1973, 1976, 1977,
        1980, 1982) and Grgurinovic (1997), no attempt has been made so far to place them
        into an infrageneric context.
        """)
        paragraphs = list(finder.parse_paragraphs(test_data))
        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(str(paragraphs[1]), expected1)

    def test_syn_break(self):
        test_data = lineify(textwrap.dedent("""\
        [@Arthonia apatetica (A. Massal.) Th. Fr. (Syn. A. exilis auct.)#Nomenclature*]
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

    def test_table_long(self):
        test_data = lineify(textwrap.dedent("""\
        Table 2. Comparison of diﬀering sequences and number of ﬁxed alleles in Ceratocystis
        spp. from mango and the closely related species C. ﬁmbriatomima. Shaded cells
        indicate variations within each species.
        C. mangicola

        C. mangivora

        C. manginecans

        C. ﬁmbriatomima

        C. mangicola
        C. mangivora
        C. manginecans
        C. ﬁmbriatomima

        4
        16
        6
        10

        16
        2
        20
        14

        6
        20
        0
        14

        10
        14
        14
        1

        βt
        C. mangicola
        C. mangivora
        C. manginecans
        C. ﬁmbriatomima

        C. mangicola
        1
        0
        5
        8

        C. mangivora
        0
        3
        4
        7

        C. manginecans
        5
        4
        0
        3

        C. ﬁmbriatomima
        8
        7
        3
        1

        EF-1α
        C. mangicola
        C. mangivora
        C. manginecans
        C. ﬁmbriatomima

        C. mangicola
        1
        0
        1
        0

        C. mangivora
        0
        9
        1
        0

        C. manginecans
        1
        1
        0
        1

        C. ﬁmbriatomima
        0
        0
        1
        0

        ITS

        Ceratocystis spp. nov. (Brazil) ... 391
        """).split('\n'))

        expected0 = '\n'

        expected2 = textwrap.dedent("""\
        Ceratocystis spp. nov. (Brazil) ... 391
        """)
        paragraphs = list(finder.parse_paragraphs(test_data))

        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(str(paragraphs[2]), expected2)

    def test_table_short(self):
        test_data = lineify(textwrap.dedent("""\
        Table 1.

        short
        shorter
        long
        longer

        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore""").split('\n'))

        expected1 = textwrap.dedent("""\
        Table 1.

        short
        shorter
        long
        longer

        """)

        expected2 = textwrap.dedent("""\
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore
        """)

        paragraphs = list(finder.parse_paragraphs(test_data))

        self.assertEqual(str(paragraphs[1]), expected1)
        self.assertEqual(str(paragraphs[2]), expected2)

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

    def test_table_long_line_break(self):
        test_data = lineify(textwrap.dedent("""\
          Table 1. Lorem ipsum dolor sit amet, consectetur adipiscing
        elit, sed do eiusmod tempor
        incididunt ut labore et dolore
        magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
        nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit
        in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
        Tbl. 2. Excepteur sint occaecat cupidatat non proident,
        """).split('\n'))

        expected1 = textwrap.dedent("""\
        Table 1. Lorem ipsum dolor sit amet, consectetur adipiscing
        elit, sed do eiusmod tempor
        incididunt ut labore et dolore
        """)
        expected2 = textwrap.dedent("""\
        magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
        nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit
        in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
        """)
        # Final table ends with two newlines.
        expected3 = textwrap.dedent("""\
        Tbl. 2. Excepteur sint occaecat cupidatat non proident,

        """)

        paragraphs = list(finder.parse_paragraphs(test_data))

        self.assertEqual(str(paragraphs[1]), expected1)
        self.assertTrue(paragraphs[1].is_table())
        self.assertEqual(str(paragraphs[2]), expected2)
        self.assertFalse(paragraphs[2].is_table())
        self.assertEqual(str(paragraphs[3]), expected3)
        self.assertTrue(paragraphs[3].is_table())

    def test_nomenclature(self):
        test_data = lineify(textwrap.dedent("""\
        Key words — milkcaps, nomenclature Lorem ipsum dolor sit amet,
        Tulostoma exasperatum Mont., Ann. Sci. Nat., Bot., Sér. 2, 8: 362.
        1837.
        Basidiomata 1.1–7.0 cm high. Spore sac globose to depressed-globose,
        0.6–0.8 cm high × 1.8–2.2 cm broad. Exoperidium spiny, light brown (5E7),
        peeling oﬀ at maturity. Endoperidium reticulate, papery, yellowish white (2A2)
        """).split('\n'))

        paragraphs = list(finder.parse_paragraphs(test_data))

        expected1 = textwrap.dedent("""\
        Tulostoma exasperatum Mont., Ann. Sci. Nat., Bot., Sér. 2, 8: 362.
        1837.
        """)

        self.assertEqual(str(paragraphs[1]), expected1)


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

        [@Tulostoma exasperatum Mont., Ann. Sci. Nat., Bot., Sér. 2, 8: 362. 1837.#Nomenclature*]
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

        # As hand-labeled:
        labels_before = [
            Label('Title'), Label('Abstract'), Label('Key-words'),
            None, Label('Nomenclature'), Label('Description'), Label('Description'),
            Label('Page-header'), Label('Description'),
            Label('Figure'), Label('Figure'),
            Label('Description'), Label('Description'), None, None, None,
        ]
        # With interstitials removed:
        labels_mid = [
            Label('Title'), Label('Abstract'), Label('Key-words'),
            Label('Nomenclature'), Label('Description'), Label('Figure'),
            Label('Description'), None,
        ]
        # With labels reduced to basic set:
        labels_after = [
            Label('Misc-exposition'), Label('Misc-exposition'), Label('Misc-exposition'),
            Label('Nomenclature'), Label('Description'), Label('Misc-exposition'),
            Label('Description'), Label('Misc-exposition'),
        ]

        phase1 = list(finder.parse_paragraphs(test_data))
        self.assertListEqual([pp.top_label() for pp in phase1], labels_before)

        phase2 = list(finder.remove_interstitials(phase1))
        self.assertListEqual([pp.top_label() for pp in phase2], labels_mid)

        phase3 = list(finder.target_classes(
            phase2,
            default=Label('Misc-exposition'),
            keep=[Label('Nomenclature'), Label('Description')]
        ))
        self.assertListEqual([pp.top_label() for pp in phase3], labels_after)


if __name__ == '__main__':
    unittest.main()
