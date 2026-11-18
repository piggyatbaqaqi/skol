"""Tests for mycobank_authors.py.

Covers where the MycoBank export is found (Trello EMbuJc0r).

Run with: python -m pytest mycobank_authors_test.py -v
"""
from pathlib import Path

import pytest

import mycobank_authors


def test_default_file_comes_from_skol_share(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no filename, MycoBankAuthors opens species/Export.csv from the
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
    tokenizer = mycobank_authors.MycoBankAuthors()
    assert getattr(tokenizer._file, 'name', None) == str(target)


def test_reads_authors_column(tmp_path: Path) -> None:
    """Each author name, lower-cased and stripped of punctuation."""
    export = tmp_path / 'Export.csv'
    export.write_text('ID,Taxon_name,Authors\n1,X y,(Bulliard) Singer\n')
    authors = mycobank_authors.MycoBankAuthors(str(export))
    assert {'bulliard', 'singer'} <= authors._pattern
