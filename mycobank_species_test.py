import mycobank_species
import unittest
from pathlib import Path

import pytest

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


def test_default_file_comes_from_skol_share(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no filename, MycoBankSpecies opens species/Export.csv from the
    share directory, whatever the current directory.  It used to open
    it relative to the current directory, which an installed package
    does not have."""
    share = tmp_path / 'share'
    target = share / 'species/Export.csv'
    target.parent.mkdir(parents=True)
    target.write_text(''.join([
        'ID,Taxon_name,Rank.ID,Rank.Rank_name,Authors,'
        'Year_of_publication,MycoBank__,Current name.Taxon_name,'
        'Classification\n'
        '1,Abortiporus biennis,20,sp.,(Bulliard) Singer,1944,283905,'
        'Abortiporus biennis,Fungi\n'
    ]), encoding='utf-8')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.setenv('SKOL_SHARE', str(share))
    monkeypatch.chdir(elsewhere)
    tokenizer = mycobank_species.MycoBankSpecies()
    assert getattr(tokenizer._file, 'name', None) == str(target)


if __name__ == '__main__':
    unittest.main()
            
