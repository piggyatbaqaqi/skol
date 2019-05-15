import finder
import textwrap
import unittest

class TestParagraph(unittest.TestCase):

    def setUp(self):
        self.pp = finder.Paragraph()
        self.pp2 = finder.Paragraph()

    def test_append(self):
        self.pp.append(finder.Line('hamster'))
        self.pp.append(finder.Line('gerbil'))
        got = str(self.pp)
        expected = 'hamster\ngerbil\n'
        self.assertEqual(got, expected)

    def test_append_ahead(self):
        self.pp.append_ahead(finder.Line('hamster'))
        self.pp.append_ahead(finder.Line('gerbil'))
        self.pp.append_ahead(finder.Line('rabbit'))
        got = str(self.pp)
        expected = 'hamster\ngerbil\n'
        self.assertEqual(got, expected)
        self.assertEqual(self.pp.next_line().line(), 'rabbit')

    def test_is_figure(self):
        self.pp.append(finder.Line('  Fig. 2.7  '))
        self.pp.append(finder.Line('hamster'))
        self.assertTrue(self.pp.is_figure())
        self.assertFalse(self.pp2.is_figure())

        self.pp2.append(finder.Line('rabbit'))
        self.assertFalse(self.pp2.is_figure())

    def test_is_table(self):
        self.pp.append(finder.Line('  Table 1 '))
        self.pp.append(finder.Line('hamster'))
        self.assertTrue(self.pp.is_table())
        self.assertFalse(self.pp2.is_table())

        self.pp2.append(finder.Line('rabbit'))
        self.assertFalse(self.pp2.is_table())

    def test_last_line(self):
        self.pp.append_ahead(finder.Line('hamster'))
        self.assertEqual(str(self.pp), '\n')
        self.pp.append_ahead(finder.Line('gerbil'))
        self.pp.append_ahead(finder.Line('rabbit'))

        self.assertEqual(self.pp.last_line.line(), 'gerbil')
        self.pp.close()
        self.assertEqual(self.pp.last_line.line(), 'rabbit')


class TestLine(unittest.TestCase):
    def test_line(self):
        data = '[@New records of smut fungi. 4. Microbotryum coronariae comb. nov.#Title*]'
        line = finder.Line(data)

        self.assertEqual(line.line(), 'New records of smut fungi. 4. Microbotryum coronariae comb. nov.')
        self.assertTrue(line.contains_start())
        self.assertEqual(line.end_label(), 'Title')
        self.assertFalse(line.is_short(50))
        self.assertFalse(line.is_blank())


class TestParser(unittest.TestCase):

    def test_regression1(self):
        test_data = textwrap.dedent("""\
        ISSN (print) 0093-4666

        © 2011. Mycotaxon, Ltd.

        ISSN (online) 2154-8889

        MYCOTAXON
        Volume 118, pp. 273–282

        http://dx.doi.org/10.5248/118.273""").split('\n')

        paragraphs = list(finder.parse_paragraphs(finder.Line(l) for l in test_data))
        self.assertEqual(len(paragraphs), 10)

    def test_table(self):
        test_data = textwrap.dedent("""\
        Table 1.

        short
        shorter
        long
        longer

        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore""").split('\n')

        expected0 = textwrap.dedent("""\
        Table 1.

        short
        shorter
        long
        longer
        """)

        expected1 = textwrap.dedent("""\
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore""")

        paragraphs = list(finder.parse_paragraphs(finder.Line(l) for l in test_data))

        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(str(paragraphs[1]), expected1)

    def test_middle_start(self):
        test_data = textwrap.dedent("""\
        multiformibus ornata. [@Habitat in herbidis locis.#Habitat-distribution*]
        """).split('\n')
        with self.assertRaisesRegex(ValueError, r'Label open not at start of line: [^:]+:[0-9]+:'):
            for p in finder.parse_paragraphs(finder.Line(l) for l in test_data):
                pass

    def test_middle_end(self):
        test_data = textwrap.dedent("""\
        multiformibus ornata.#Description*] Habitat in herbidis locis
        """).split('\n')

        with self.assertRaisesRegex(ValueError, r'Label close not at end of line: [^:]+:[0-9]+:'):
            for p in finder.parse_paragraphs(finder.Line(l) for l in test_data):
                pass

    def test_figure(self):
        test_data = textwrap.dedent("""\
          Fig 1. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
        tempor incididunt ut labore et dolore

        magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
        nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit
        in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

        Figure 2. Excepteur sint occaecat cupidatat non proident,

        Photo 1. culpa qui officia deserunt mollit anim id est laborum.
        """).split('\n')

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

        paragraphs = list(finder.parse_paragraphs(finder.Line(l) for l in test_data))

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
        test_data = textwrap.dedent("""\
          Table 1. Lorem ipsum dolor sit amet, consectetur adipiscing
        elit, sed do eiusmod tempor
        incididunt ut labore et dolore
        magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
        nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit
        in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
        Tbl. 2. Excepteur sint occaecat cupidatat non proident,
        """).split('\n')

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

        paragraphs = list(finder.parse_paragraphs(finder.Line(l) for l in test_data))

        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertTrue(paragraphs[0].is_table())
        self.assertEqual(str(paragraphs[1]), expected1)
        self.assertFalse(paragraphs[1].is_table())
        self.assertEqual(str(paragraphs[2]), expected2)
        self.assertTrue(paragraphs[2].is_table())


class TestLabeling(unittest.TestCase):

    def test_simple_label(self):
        test_data = textwrap.dedent("""\
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
        """).split('\n')

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

        paragraphs = list(finder.parse_paragraphs(finder.Line(l) for l in test_data))

        self.maxDiff = None
        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(paragraphs[0].labels, [finder.Label('Abstract')])
        self.assertEqual(str(paragraphs[1]), expected1)
        self.assertEqual(paragraphs[1].labels, [finder.Label('Key-words')])

    def test_doubled_abstract(self):

        test_data = textwrap.dedent("""\
        [@New records of smut fungi. 4. Microbotryum coronariae comb. nov.#Title*]
        [@Cvetomir M. Denchev & Teodor T. Denchev#Author*]
        [@Institute of Biodiversity and Ecosystem Research, Bulgarian Academy of Sciences,
        2 Gagarin St., 1113 Soﬁa, Bulgaria#Institution*]
        * Correspondence to: cmdenchev@yahoo.co.uk
        [@Abstract — For Ustilago coronariae on Lychnis ﬂos-cuculi, a new combination in
        Microbotryum, M. coronariae, is proposed. It is reported as new to Bulgaria.#Abstract*]
        [@Key words — Microbotryaceae, taxonomy#Key-words*]
        """).split('\n')

        paragraphs = list(finder.parse_paragraphs(finder.Line(l) for l in test_data))

        self.assertEqual(len(paragraphs), 7)
        self.assertEqual(paragraphs[0].labels, [finder.Label('Title')])
        self.assertEqual(paragraphs[1].labels, [finder.Label('Author')])
        self.assertEqual(paragraphs[2].labels, [finder.Label('Institution')])
        self.assertEqual(paragraphs[3].labels, [])  # correspondence
        self.assertEqual(paragraphs[4].labels, [finder.Label('Abstract')])
        self.assertEqual(paragraphs[5].labels, [finder.Label('Key-words')])
        self.assertEqual(paragraphs[6].labels, []) # Paragraph('\n')


if __name__ == '__main__':
    unittest.main()
