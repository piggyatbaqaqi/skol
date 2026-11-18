import iso4
import unittest
from pathlib import Path

import pytest

class TestLTWA(unittest.TestCase):

    def test_tokenize(self):
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
            'Arkiv. fór Botanik '
            'Ann. Mycol. '
            'Grevillea '
            'NXXXX.')
        expected = [
            'mykol.', 'n. am.', 'j.', 'friesland', 'barl.',
            'n. y.', 'st.', 'mus.', 'sci.', 'surv.',
            'bot.', 'jahrb.', 'syst.',
            'annals', 'of', 'natural', 'history',
            'sydowia', 'persoonia',
            'mycotaxon',
            'nova', 'hedwigia', 'beihefte',
            'ann.', 'cryptog.', 'exot.',
            'agric.', 'handbook', 'u.s.', 'dep.', 'agric.',
            'ann.', 'iv', 'congr.', 'soc.', 'bot.', 'brazil',
            'mem.', 'ny', 'bot.', 'gard.',
            'arkiv', '.', 'fór', 'botanik',
            'ann.', 'mycol.',
            'grevillea',
        ]
        
        ltwa = iso4.LTWA()
        for (m, l) in ltwa.tokenize(test_data):
            self.assertEqual(m.string[m.start():m.end()], expected.pop(0))
        self.assertEqual(l, 'NXXXX.')


@pytest.mark.xfail(
    strict=True,
    reason='the default file moves to skol_share (Trello EMbuJc0r)')
def test_default_file_comes_from_skol_share(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no filename, LTWA opens LTWA_20160915.txt from the
    share directory, whatever the current directory.  It used to open
    it relative to the current directory, which an installed package
    does not have."""
    share = tmp_path / 'share'
    target = share / 'LTWA_20160915.txt'
    target.parent.mkdir(parents=True)
    target.write_text(''.join([
        'WORD\tABBREVIATIONS\tLANGUAGE CODES\n'
        'mycologia\tmycol.\teng\n'
    ]), encoding='utf-8')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.setenv('SKOL_SHARE', str(share))
    monkeypatch.chdir(elsewhere)
    tokenizer = iso4.LTWA()
    assert getattr(tokenizer._file, 'name', None) == str(target)


if __name__ == '__main__':
    unittest.main()
            
