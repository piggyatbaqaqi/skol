import mycobank_species
import unittest

class TestMycoBank(unittest.TestCase):

    def test_tokenize(self):
        test_data = (
            'Polyblastiopsis rappii '
            'Boletus edulis var. hamster '
        )
        expected = [
            'Polyblastiopsis', 'rappii',
            'Boletus', 'edulis', 'var.'
        ]
        
        tokenizer = mycobank_species.MycoBankSpecies()
        for (m, l) in tokenizer.tokenize(test_data):
            if m is not None:
                self.assertEqual(m, expected.pop(0))
        self.assertEqual(l, 'hamster')


if __name__ == '__main__':
    unittest.main()
            
