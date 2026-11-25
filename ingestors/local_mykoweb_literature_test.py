"""Tests for ingestors/local_mykoweb_literature.py."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestors import local_mykoweb_literature as lml  # noqa: E402
from ingestors.local_mykoweb_literature import (  # noqa: E402
    LocalMykowebLiteratureIngestor,
)


def _ingestor(tmp_path: Path, verbosity: int) -> LocalMykowebLiteratureIngestor:
    ing = LocalMykowebLiteratureIngestor(
        root=tmp_path,
        local_path_prefix=str(tmp_path),
        url_prefix='https://mykoweb.com/systematics/literature',
        site_root=str(tmp_path),
        metadata_path=tmp_path / 'no-such-metadata.json',
        db=MagicMock(), user_agent='test-agent', robot_parser=MagicMock(),
        verbosity=verbosity)
    ing._ingest_documents = MagicMock()  # type: ignore[method-assign]
    return ing


def _pdf(tmp_path: Path, name: str = 'Smith 1923 Agarics.pdf') -> Path:
    sub = tmp_path / 'books'
    sub.mkdir(exist_ok=True)
    path = sub / name
    path.write_bytes(b'%PDF-1.4')
    return path


def test_ingests_each_pdf_with_its_url_and_filename_title(
        tmp_path: Path) -> None:
    _pdf(tmp_path)
    (tmp_path / 'books' / 'notes.txt').write_text('not a pdf')
    ing = _ingestor(tmp_path, verbosity=0)
    ing.ingest()
    ing._ingest_documents.assert_called_once()  # type: ignore[attr-defined]
    kwargs: Dict[str, Any] = (
        ing._ingest_documents.call_args.kwargs)  # type: ignore[attr-defined]
    url = ('https://mykoweb.com/systematics/literature/books/'
           'Smith 1923 Agarics.pdf')
    assert kwargs['documents'] == [
        {'url': url, 'title': 'Smith 1923 Agarics', 'itemtype': 'book'}]
    assert kwargs['bibtex_link'] == url
    assert kwargs['meta'] == {'source': 'mykoweb', 'type': 'literature'}


@pytest.mark.xfail(strict=True, reason='NameError: title at verbosity 3')
def test_verbose_run_prints_the_filename_title(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _pdf(tmp_path)
    _ingestor(tmp_path, verbosity=3).ingest()
    assert '    Title: Smith 1923 Agarics\n' in capsys.readouterr().out


@pytest.mark.xfail(strict=True, reason='NameError: title at verbosity 3')
def test_verbose_run_prints_the_curated_title(
        tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    # The printed title is the one ingested, not the filename.
    _pdf(tmp_path)

    def lookup(*_args: Any) -> Optional[Dict[str, Any]]:
        return {'title': 'Agarics of North America'}

    monkeypatch.setattr(lml, 'lookup_pdf_metadata', lookup)
    monkeypatch.setattr(lml, 'metadata_to_doc_fields',
                        lambda record: {'title': record['title']})
    _ingestor(tmp_path, verbosity=3).ingest()
    assert '    Title: Agarics of North America\n' in capsys.readouterr().out
