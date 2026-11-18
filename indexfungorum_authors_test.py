import indexfungorum_authors
import unittest
from pathlib import Path

import pytest

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


def test_default_file_comes_from_skol_share(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no filename, IndexFungorumAuthors opens authors/FungalNameAuthors.txt from the
    share directory, whatever the current directory.  It used to open
    it relative to the current directory, which an installed package
    does not have."""
    share = tmp_path / 'share'
    target = share / 'authors/FungalNameAuthors.txt'
    target.parent.mkdir(parents=True)
    target.write_text(''.join([
        'header\n'
        '-----------------\n'
        'Fries\n'
    ]), encoding='utf-8')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.setenv('SKOL_SHARE', str(share))
    monkeypatch.chdir(elsewhere)
    tokenizer = indexfungorum_authors.IndexFungorumAuthors()
    assert getattr(tokenizer._file, 'name', None) == str(target)


if __name__ == '__main__':
    unittest.main()
