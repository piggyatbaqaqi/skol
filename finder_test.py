import finder
import textwrap
import unittest

class TestParagraph(unittest.TestCase):

    def setUp(self):
        self.pp = finder.Paragraph()
        self.pp2 = finder.Paragraph()

    def test_append(self):
        self.pp.append('hamster')
        self.pp.append('gerbil')
        got = str(self.pp)
        expected = 'hamster\ngerbil\n'
        self.assertEqual(got, expected)

    def test_append_ahead(self):
        self.pp.append_ahead('hamster')
        self.pp.append_ahead('gerbil')
        self.pp.append_ahead('rabbit')
        got = str(self.pp)
        expected = 'hamster\ngerbil\n'
        self.assertEqual(got, expected)
        self.assertEqual(self.pp.successor, 'rabbit')

    def test_is_header(self):
        self.pp.append('headline')
        self.pp.append('hamster')

        self.assertTrue(self.pp.is_header())
        self.assertFalse(self.pp2.is_header())

        self.pp2.append('rabbit')
        self.assertFalse(self.pp2.is_header())

    def test_is_figure(self):
        self.pp.append('  Fig. 2.7  ')
        self.pp.append('hamster')
        self.assertTrue(self.pp.is_figure())
        self.assertFalse(self.pp2.is_figure())

        self.pp2.append('rabbit')
        self.assertFalse(self.pp2.is_figure())

    def test_is_table(self):
        self.pp.append('  Table 1 ')
        self.pp.append('hamster')
        self.assertTrue(self.pp.is_table())
        self.assertFalse(self.pp2.is_table())

        self.pp2.append('rabbit')
        self.assertFalse(self.pp2.is_table())
        
    def test_last_line(self):
        self.pp.append_ahead('hamster')
        self.assertEqual(str(self.pp), '\n')
        self.pp.append_ahead('gerbil')
        self.pp.append_ahead('rabbit')

        self.assertEqual(self.pp.last_line(), 'gerbil')
        self.pp.close()
        self.assertEqual(self.pp.last_line(), 'rabbit')
        

class TestParser(unittest.TestCase):

    def test_regression1(self):
        test_data = textwrap.dedent("""\
        ISSN (print) 0093-4666

        © 2011. Mycotaxon, Ltd.

        ISSN (online) 2154-8889

        MYCOTAXON
        Volume 118, pp. 273–282

        http://dx.doi.org/10.5248/118.273""").split('\n')

        paragraphs = list(finder.parse_paragraphs(test_data))        
        self.assertEqual(len(paragraphs), 9)

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

        paragraphs = list(finder.parse_paragraphs(test_data))

        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(str(paragraphs[1]), expected1)
        
    def test_figure(self):
        test_data = textwrap.dedent("""\
          Fig 1. Lorem ipsum dolor sit amet, consectetur adipiscing
        elit, sed do eiusmod tempor incididunt ut labore et dolore

        magna aliqua. Ut enim ad minim veniam, quis nostrud
        exercitation ullamco laboris nisi ut aliquip ex ea commodo
        consequat. Duis aute irure dolor in reprehenderit in voluptate
        velit esse cillum dolore eu fugiat nulla pariatur.

        Figure 2. Excepteur sint occaecat cupidatat non proident,
        """).split('\n')

        expected0 = textwrap.dedent("""\
          Fig 1. Lorem ipsum dolor sit amet, consectetur adipiscing
        elit, sed do eiusmod tempor incididunt ut labore et dolore
        """)

        expected2 = textwrap.dedent("""\
        magna aliqua. Ut enim ad minim veniam, quis nostrud
        exercitation ullamco laboris nisi ut aliquip ex ea commodo
        consequat. Duis aute irure dolor in reprehenderit in voluptate
        velit esse cillum dolore eu fugiat nulla pariatur.
        """)

        expected4 = textwrap.dedent("""\
        Figure 2. Excepteur sint occaecat cupidatat non proident,
        """)

        paragraphs = list(finder.parse_paragraphs(test_data))

        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertTrue(paragraphs[0].is_figure())
        self.assertEqual(str(paragraphs[1]), '\n')
        self.assertEqual(str(paragraphs[2]), expected2)
        self.assertFalse(paragraphs[2].is_figure())
        self.assertEqual(str(paragraphs[1]), '\n')
        self.assertEqual(str(paragraphs[4]), expected4)
        self.assertTrue(paragraphs[4].is_figure())
        self.assertEqual(str(paragraphs[1]), '\n')
        self.assertFalse(paragraphs[5].is_figure())
        
    def test_table(self):
        test_data = textwrap.dedent("""\
          Table 1. Lorem ipsum dolor sit amet, consectetur adipiscing
        elit, sed do eiusmod tempor
        incididunt ut labore et dolore
        magna aliqua. Ut enim ad minim veniam, quis nostrud
        exercitation ullamco laboris nisi ut aliquip ex ea commodo
        consequat. Duis aute irure dolor in reprehenderit in voluptate
        velit esse cillum dolore eu fugiat nulla pariatur.
        Tbl. 2. Excepteur sint occaecat cupidatat non proident,
        """).split('\n')

        expected0 = textwrap.dedent("""\
          Table 1. Lorem ipsum dolor sit amet, consectetur adipiscing
        elit, sed do eiusmod tempor
        incididunt ut labore et dolore
        """)
        expected1 = textwrap.dedent("""\
        magna aliqua. Ut enim ad minim veniam, quis nostrud
        exercitation ullamco laboris nisi ut aliquip ex ea commodo
        consequat. Duis aute irure dolor in reprehenderit in voluptate
        velit esse cillum dolore eu fugiat nulla pariatur.
        """)
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

    def test_simple_lable(self):
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

        paragraphs = list(finder.parse_paragraphs(test_data))

        self.maxDiff = None
        self.assertEqual(str(paragraphs[0]), expected0)
        self.assertEqual(paragraphs[0].lable, 'Abstract')
        self.assertEqual(str(paragraphs[1]), expected1)
        self.assertEqual(paragraphs[0].lable, 'Key-words')



if __name__ == '__main__':
    unittest.main()
