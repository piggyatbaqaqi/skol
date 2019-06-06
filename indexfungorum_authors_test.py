import indexfungorum_authors
import unittest

class TestLTWA(unittest.TestCase):

    def test_tokenize(self):
        test_data = (
            'Goossens-Fontana, J.A.A.    \n'
            '(Stizenberger ex Arnold) R.C. Harris in Egan\n'
            'Nees & T. Nees ex Link, Willdenow\n'
            'Guzmán, Ram.-Guill.,\n'
            '(Pers. : Fr.) W. Phillips'
        )
        expected = [
            'Goossens', 'Fontana', 'J.A.A.',
            'Stizenberger', 'ex', 'Arnold', 'R.C.', 'Harris', 'in', 'Egan',
            'Nees', '&', 'T.', 'Nees', 'ex', 'Link', 'Willdenow',
            'Guzmán', 'Ram.', 'Guill.',
            'Pers.', ':', 'Fr.', 'W.', 'Phillips',
        ]
        ifa = indexfungorum_authors.IndexFungorumAuthors()
        
        for (m, l) in ifa.tokenize(test_data):
            self.assertEqual(m, expected.pop(0))

if __name__ == '__main__':
    unittest.main()
