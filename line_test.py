"""Tests for line.py"""
import textwrap
from typing import List
import unittest


from line import Line

def lineify(lines: List[str]) -> List[Line]:
    return [Line(l) for l in lines]

class TestLine(unittest.TestCase):
    def test_line(self):
        data = '[@New records of smut fungi. 4. Microbotryum coronariae comb. nov.#Title*]'
        line = Line(data)

        self.assertEqual(line.line, 'New records of smut fungi. 4. Microbotryum coronariae comb. nov.')
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


if __name__ == '__main__':
    unittest.main()
