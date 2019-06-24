"""Tests for file.py."""
import textwrap
import unittest

from file import File

class TestFile(unittest.TestCase):
    def setUp(self):
        pass

    def test_line_number(self):
        test_data = textwrap.dedent("""\
        one
        two
        three
        """).split('\n')
        f = File(contents=test_data)
        got = []
        for l in f.read_line():
            got.append(l)

        self.assertEqual(got[0].line, 'one')
        self.assertEqual(got[1].line, 'two')
        self.assertEqual(got[2].line, 'three')
        self.assertEqual(got[0].line_number, 1)
        self.assertEqual(got[1].line_number, 2)
        self.assertEqual(got[2].line_number, 3)


    def test_page_number(self):
        test_data = textwrap.dedent("""\
        xi lorem ipsum

        page 1, line 3
        page 1, line 4
        dolor sit  xii

        page 2, line 3
        page 2, line 4

        1 amet, consectetur

        page 3, line 3
        adipiscing elit  kn

        page 4, line 3

        3 sed do eiusmod

        page 5, line 3
        """).split('\n')
        f = File(contents=test_data)
        got = [l for l in f.read_line()]
        self.assertEqual(got[0].empirical_page_number, 'xi')
        self.assertEqual(got[0].line, 'xi lorem ipsum')
        self.assertEqual(got[0].line_number, 1)
        self.assertEqual(got[0].page_number, 1)

        self.assertEqual(got[3].line, 'page 1, line 4')
        self.assertEqual(got[3].line_number, 4)
        self.assertEqual(got[3].page_number, 1)

        self.assertEqual(got[6].empirical_page_number, 'xii')
        self.assertEqual(got[6].line, 'page 2, line 3')
        self.assertEqual(got[6].line_number, 3)
        self.assertEqual(got[6].page_number, 2)

        self.assertIsNone(got[14].empirical_page_number)
        self.assertEqual(got[14].line, 'page 4, line 3')
        self.assertEqual(got[14].line_number, 3)
        self.assertEqual(got[14].page_number, 4)

        self.assertEqual(got[18].empirical_page_number, '3')


if __name__ == '__main__':
    unittest.main()
