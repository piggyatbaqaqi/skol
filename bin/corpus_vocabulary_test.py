"""Tests for bin/corpus_vocabulary.

The CouchDB scan is exercised through a small fake; the counting
and threshold logic is tested directly.
"""

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_vocabulary import (  # noqa: E402
    document_frequencies,
    field_tokens,
    select_vocabulary,
    write_vocabulary,
)


_XFAIL = pytest.mark.xfail(
    reason="2026-08-21: corpus_vocabulary not implemented yet",
    strict=True,
)

_ENGLISH = {'the', 'and', 'colonies', 'white', 'smooth'}


class _FakeRow:
    def __init__(self, doc_id, doc):
        self.id = doc_id
        self.doc = doc


class _FakeDb:
    """Stands in for a treatments_prose database."""

    def __init__(self, docs):
        self._docs = docs

    def view(self, _name, include_docs=True):
        return [_FakeRow(i, d) for i, d in self._docs]


@_XFAIL
class TestFieldTokens:
    """Only description + diagnosis are read, lowercased, and
    filtered to out-of-vocabulary alphabetic forms of 4+ chars."""

    def test_reads_description_and_diagnosis(self) -> None:
        doc = {'description': 'Pileus glabra', 'diagnosis': 'Sporae'}
        assert field_tokens(doc, _ENGLISH) == {'pileus', 'glabra', 'sporae'}

    def test_drops_english_words(self) -> None:
        doc = {'description': 'the colonies smooth glabra'}
        assert field_tokens(doc, _ENGLISH) == {'glabra'}

    def test_drops_short_and_non_alphabetic(self) -> None:
        doc = {'description': 'ab abc 12um x3y glabra'}
        assert field_tokens(doc, _ENGLISH) == {'glabra'}

    def test_counts_each_form_once_per_document(self) -> None:
        doc = {'description': 'glabra glabra glabra'}
        assert field_tokens(doc, _ENGLISH) == {'glabra'}

    def test_missing_fields_are_empty(self) -> None:
        assert field_tokens({}, _ENGLISH) == set()


@_XFAIL
class TestDocumentFrequencies:
    def test_counts_documents_not_occurrences(self) -> None:
        db = _FakeDb([
            ('taxon_a', {'description': 'glabra glabra pileus'}),
            ('taxon_b', {'description': 'glabra'}),
        ])
        freqs, scanned = document_frequencies(db, _ENGLISH)
        assert freqs['glabra'] == 2
        assert freqs['pileus'] == 1
        assert scanned == 2

    def test_skips_non_taxon_and_empty_docs(self) -> None:
        db = _FakeDb([
            ('_design/x', {'description': 'glabra'}),
            ('taxon_a', None),
            ('taxon_b', {'description': ''}),
            ('taxon_c', {'description': 'glabra'}),
        ])
        freqs, scanned = document_frequencies(db, _ENGLISH)
        assert freqs['glabra'] == 1 and scanned == 1


@_XFAIL
class TestSelectVocabulary:
    """The threshold is the guard against OCR corruption entering
    the vocabulary: a corrupt form must recur across that many
    distinct documents to survive."""

    def test_applies_threshold(self) -> None:
        freqs = {'glabra': 50, 'pileus': 49, 'artbroconiuia': 2}
        assert select_vocabulary(freqs, 50) == ['glabra']

    def test_threshold_is_inclusive(self) -> None:
        assert select_vocabulary({'a': 3}, 3) == ['a']

    def test_result_is_sorted(self) -> None:
        freqs = {'zeta': 9, 'alpha': 9, 'mu': 9}
        assert select_vocabulary(freqs, 1) == ['alpha', 'mu', 'zeta']


@_XFAIL
class TestWriteVocabulary:
    def test_one_form_per_line(self) -> None:
        out = io.StringIO()
        write_vocabulary(['alpha', 'beta'], out)
        assert out.getvalue() == 'alpha\nbeta\n'
