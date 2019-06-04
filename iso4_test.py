import iso4
import unittest

class TestLTWA(unittest.TestCase):

    def test_eat_tokens(self):
        test_data = (
            'mykol. N. Am. j.'
            'Friesland Barl.'
            'N. Y. St. Mus. Sci. Surv.'
            'Bot. Jahrb. Syst.'
            'Annals of Natural History '
            'Sydowia Persoonia '
            'Mycotaxon '
            'Nova Hedwigia Beihefte '
            'Ann. Cryptog. Exot. '
            'Agric. Handbook U.S. Dep. Agric. '
            'Ann. IV Congr. Soc. Bot. Brazil '
            'Mem. NY Bot. Gard. '
            'NXXXX.')
        expected = [
            'mykol.', 'n. am.', 'j.', 'friesland', 'barl.',
            'n. y.', 'st.', 'mus.', 'sci.', 'surv.',
            'bot.', 'jahrb.', 'syst.',
            'annals', 'of', 'natural', 'history',
            'sydowia', 'persoon', 'ia',
            'mycotaxon',
            'nova', 'hedwigia', 'beihefte',
            'ann.', 'cryptog.', 'exot.',
            'agric.', 'handbook', 'u.s.', 'dep.', 'agric.',
            'ann.', 'iv', 'congr.', 'soc.', 'bot.', 'brazil',
            'mem.', 'ny', 'bot.', 'gard.',
        ]
        
        ltwa = iso4.LTWA()
        for (m, l) in ltwa.eat_tokens(test_data):
            self.assertEqual(m.string[m.start():m.end()], expected.pop(0))
        self.assertEqual(l, 'NXXXX.')


if __name__ == '__main__':
    unittest.main()
            
