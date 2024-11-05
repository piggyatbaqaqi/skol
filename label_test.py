"""Tests for label.py."""

import textwrap
from typing import List
import unittest

import finder
from line import Line
from label import Label


def lineify(lines: List[str]) -> List[Line]:
    return [Line(l) for l in lines]


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
        [@Key words — Microbotryaceae, nomenclature#Key-words*]
        """).split('\n'))

        paragraphs = list(finder.parse_paragraphs(test_data))

        self.assertEqual(len(paragraphs), 6)
        self.assertEqual(paragraphs[0].labels, [Label('Title')])
        self.assertEqual(paragraphs[1].labels, [Label('Author')])
        self.assertEqual(paragraphs[2].labels, [Label('Institution')])
        self.assertEqual(paragraphs[3].labels, [])  # correspondence
        self.assertEqual(paragraphs[4].labels, [Label('Abstract'), Label('Key-words')])
        self.assertEqual(paragraphs[5].labels, []) # Paragraph('\n')


if __name__ == '__main__':
    unittest.main()
