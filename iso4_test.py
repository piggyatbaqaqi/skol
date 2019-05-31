import iso4
import unittest

class TestLTWA(unittest.TestCase):

    def test_eat_tokens(self):
        test_data = 'mykol. N. Am. j. Friesland Barl. XXXX.'
        expected = ['mykol.', 'N. Am.', 'j.', 'Friesland', 'Barl.']
        
        ltwa = iso4.LTWA()
        for (m, l) in ltwa.eat_tokens(test_data):
            self.assertEqual(m.string[m.start():m.end()], expected.pop(0))
        self.assertEqual(l, 'XXXX.')


if __name__ == '__main__':
    unittest.main()
            
