"""Tests for bin/enrich_ann_merged.py.

Covers DOI extraction, Crossref response shaping, and the per-step
skip-if-present logic.  Network calls and CouchDB writes are mocked.
"""

import io
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from enrich_ann_merged import (  # type: ignore[import]  # noqa: E402
    _extract_doi,
    _first_page_text,
    _shape_crossref_message,
    needs_attachment,
    needs_crossref,
    needs_golden_flag,
)


# ---------------------------------------------------------------------------
# First-page slicing
# ---------------------------------------------------------------------------


class TestFirstPageText:
    def test_slice_between_markers(self) -> None:
        text = (
            "--- PDF Page 1 Label 1 ---\n"
            "First page content with DOI: 10.1234/abc.def\n"
            "--- PDF Page 2 Label 2 ---\n"
            "Second page content\n"
        )
        first = _first_page_text(text)
        assert "First page content" in first
        assert "Second page content" not in first

    def test_no_markers_returns_head(self) -> None:
        text = "no markers here\n" + ("filler " * 1000)
        first = _first_page_text(text)
        # Falls back to head of document, not the whole thing.
        assert first.startswith("no markers here")
        assert len(first) < len(text)

    def test_only_first_marker_returns_remainder(self) -> None:
        text = "--- PDF Page 1 Label 1 ---\nonly one page here\n"
        first = _first_page_text(text)
        assert "only one page here" in first


# ---------------------------------------------------------------------------
# DOI extraction
# ---------------------------------------------------------------------------


class TestExtractDoi:
    def test_finds_simple_doi(self) -> None:
        assert _extract_doi("see DOI: 10.1234/abc.def for details") \
            == "10.1234/abc.def"

    def test_finds_doi_with_url_prefix(self) -> None:
        text = "Available at https://doi.org/10.5678/foo.bar.42 today."
        assert _extract_doi(text) == "10.5678/foo.bar.42"

    def test_finds_doi_anywhere_in_text(self) -> None:
        text = "preamble preamble 10.1000/xyzABC_123 trailing"
        assert _extract_doi(text) == "10.1000/xyzABC_123"

    def test_returns_none_when_absent(self) -> None:
        assert _extract_doi("nothing here, no DOI") is None

    def test_returns_first_match(self) -> None:
        text = "10.1234/aaa and also 10.5678/bbb"
        assert _extract_doi(text) == "10.1234/aaa"

    def test_strips_trailing_punctuation(self) -> None:
        """A DOI followed by a period/comma should not include it."""
        assert _extract_doi("see 10.1234/abc.def.") == "10.1234/abc.def"
        assert _extract_doi("see 10.1234/abc,") == "10.1234/abc"


# ---------------------------------------------------------------------------
# Crossref response shaping
# ---------------------------------------------------------------------------


_SAMPLE_CROSSREF_MESSAGE: Dict[str, Any] = {
    "DOI": "10.1234/abc.def",
    "title": ["A Title"],
    "author": [
        {"given": "Alice", "family": "One"},
        {"given": "Bob", "family": "Two"},
    ],
    "container-title": ["Journal of Things"],
    "published-print": {"date-parts": [[2024, 3, 15]]},
    "volume": "42",
    "issue": "3",
    "page": "100-110",
    "publisher": "Some Publisher",
    "type": "journal-article",
    "ISSN": ["1234-5678"],
    "URL": "https://doi.org/10.1234/abc.def",
    "irrelevant_field": "drop me",
}


class TestShapeCrossrefMessage:
    def test_keeps_known_fields(self) -> None:
        out = _shape_crossref_message(_SAMPLE_CROSSREF_MESSAGE)
        assert out["doi"] == "10.1234/abc.def"
        assert out["title"] == "A Title"
        assert out["container_title"] == "Journal of Things"
        assert out["volume"] == "42"
        assert out["issue"] == "3"
        assert out["page"] == "100-110"
        assert out["publisher"] == "Some Publisher"
        assert out["type"] == "journal-article"
        assert out["url"] == "https://doi.org/10.1234/abc.def"
        assert out["issn"] == ["1234-5678"]

    def test_authors_flattened(self) -> None:
        out = _shape_crossref_message(_SAMPLE_CROSSREF_MESSAGE)
        assert out["authors"] == [
            {"given": "Alice", "family": "One"},
            {"given": "Bob", "family": "Two"},
        ]

    def test_year_extracted_from_date_parts(self) -> None:
        out = _shape_crossref_message(_SAMPLE_CROSSREF_MESSAGE)
        assert out["year"] == 2024

    def test_year_falls_back_to_published_online(self) -> None:
        msg = {
            "DOI": "10.1/x",
            "published-online": {"date-parts": [[2023]]},
        }
        out = _shape_crossref_message(msg)
        assert out["year"] == 2023

    def test_drops_irrelevant_fields(self) -> None:
        out = _shape_crossref_message(_SAMPLE_CROSSREF_MESSAGE)
        assert "irrelevant_field" not in out

    def test_handles_missing_title(self) -> None:
        out = _shape_crossref_message({"DOI": "10.1/x"})
        assert out["doi"] == "10.1/x"
        assert "title" not in out


# ---------------------------------------------------------------------------
# Skip-if-present helpers
# ---------------------------------------------------------------------------


class TestNeedsHelpers:
    def test_needs_attachment_when_missing(self) -> None:
        doc: Dict[str, Any] = {"_attachments": {}}
        assert needs_attachment(doc, "article.txt") is True

    def test_needs_attachment_false_when_present(self) -> None:
        doc = {"_attachments": {"article.txt": {}}}
        assert needs_attachment(doc, "article.txt") is False

    def test_needs_golden_flag_when_missing(self) -> None:
        assert needs_golden_flag({}) is True

    def test_needs_golden_flag_when_present(self) -> None:
        assert needs_golden_flag({"is_golden": False}) is False
        assert needs_golden_flag({"is_golden": True}) is False

    def test_needs_crossref_default_skip(self) -> None:
        assert needs_crossref({"publication_metadata": {"doi": "x"}},
                              force=False) is False

    def test_needs_crossref_force(self) -> None:
        assert needs_crossref({"publication_metadata": {"doi": "x"}},
                              force=True) is True

    def test_needs_crossref_when_missing(self) -> None:
        assert needs_crossref({}, force=False) is True


# ---------------------------------------------------------------------------
# process_doc integration
# ---------------------------------------------------------------------------


class _FakeAttachment:
    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    def read(self) -> bytes:
        return self._buf.read()


class _FakeDb:
    def __init__(self, docs: Optional[Dict[str, Dict[str, Any]]] = None,
                 attachments: Optional[Dict[str, Dict[str, bytes]]] = None,
                 ) -> None:
        self.docs: Dict[str, Dict[str, Any]] = docs or {}
        self.attachments: Dict[str, Dict[str, bytes]] = attachments or {}
        self.saved: list = []

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        if doc_id not in self.docs:
            raise KeyError(doc_id)
        doc = dict(self.docs[doc_id])
        atts = self.attachments.get(doc_id, {})
        if atts:
            doc["_attachments"] = {n: {} for n in atts}
        elif "_attachments" not in doc:
            doc["_attachments"] = {}
        return doc

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.docs

    def get_attachment(self, doc_id: str,
                       name: str) -> Optional[_FakeAttachment]:
        data = self.attachments.get(doc_id, {}).get(name)
        return _FakeAttachment(data) if data is not None else None

    def save(self, doc: Dict[str, Any]) -> None:
        # Mirror couchdb-python's behavior: save updates the doc.
        self.docs[doc["_id"]] = {
            k: v for k, v in doc.items() if k != "_attachments"
        }
        self.saved.append(dict(doc))

    def put_attachment(self, doc: Dict[str, Any], content: bytes,
                       filename: str, content_type: str) -> None:
        self.attachments.setdefault(doc["_id"], {})[filename] = content


class _StubHttp:
    def __init__(self, payload: Dict[str, Any], status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.calls: list = []

    def get(self, url: str, headers: Dict[str, str],
            timeout: int) -> Any:  # noqa: ARG002
        self.calls.append(url)
        response = MagicMock()
        response.status_code = self.status
        response.json.return_value = self.payload
        response.raise_for_status = MagicMock()
        if self.status >= 400:
            from requests import HTTPError
            response.raise_for_status.side_effect = HTTPError("oops")
        return response


class TestProcessDoc:
    def test_copies_missing_attachments(self) -> None:
        from enrich_ann_merged import process_doc  # type: ignore[import]
        merged = _FakeDb(docs={"d1": {"_id": "d1"}})
        training = _FakeDb(
            docs={"d1": {"_id": "d1"}},
            attachments={"d1": {"article.txt": b"text content",
                                "article.pdf": b"%PDF"}},
        )
        golden = _FakeDb()  # empty → not golden
        http = _StubHttp({"message": {"DOI": "10.1/x"}})
        process_doc("d1", merged, training, golden, http, force=False)
        assert merged.attachments["d1"]["article.txt"] == b"text content"
        assert merged.attachments["d1"]["article.pdf"] == b"%PDF"

    def test_sets_golden_flag_from_membership(self) -> None:
        from enrich_ann_merged import process_doc  # type: ignore[import]
        merged = _FakeDb(docs={"a": {"_id": "a"}, "b": {"_id": "b"}})
        training = _FakeDb(docs={"a": {"_id": "a"}, "b": {"_id": "b"}})
        golden = _FakeDb(docs={"a": {"_id": "a"}})  # only "a" is golden
        http = _StubHttp({"message": {"DOI": "10.1/x"}})
        process_doc("a", merged, training, golden, http)
        process_doc("b", merged, training, golden, http)
        assert merged.docs["a"]["is_golden"] is True
        assert merged.docs["b"]["is_golden"] is False

    def test_doi_lookup_populates_metadata(self) -> None:
        from enrich_ann_merged import process_doc  # type: ignore[import]
        merged = _FakeDb(docs={"d1": {"_id": "d1"}})
        text = (
            "--- PDF Page 1 Label 1 ---\n"
            "Title here. DOI: 10.1234/abc.def\n"
            "--- PDF Page 2 Label 2 ---\n"
        )
        training = _FakeDb(
            docs={"d1": {"_id": "d1"}},
            attachments={"d1": {"article.txt": text.encode("utf-8")}},
        )
        golden = _FakeDb()
        http = _StubHttp({"message": _SAMPLE_CROSSREF_MESSAGE})
        process_doc("d1", merged, training, golden, http)
        meta = merged.docs["d1"]["publication_metadata"]
        assert meta["doi"] == "10.1234/abc.def"
        assert meta["title"] == "A Title"
        assert http.calls[0].endswith("/works/10.1234/abc.def")

    def test_skips_crossref_when_already_present(self) -> None:
        from enrich_ann_merged import process_doc  # type: ignore[import]
        merged = _FakeDb(docs={"d1": {
            "_id": "d1",
            "publication_metadata": {"doi": "10.old/x"},
        }})
        training = _FakeDb(
            docs={"d1": {"_id": "d1"}},
            attachments={"d1": {"article.txt": (
                "--- PDF Page 1 Label 1 ---\n"
                "DOI 10.new/y\n"
                "--- PDF Page 2 Label 2 ---"
            ).encode("utf-8")}},
        )
        golden = _FakeDb()
        http = _StubHttp({"message": _SAMPLE_CROSSREF_MESSAGE})
        process_doc("d1", merged, training, golden, http, force=False)
        assert http.calls == []
        # Old metadata untouched.
        assert merged.docs["d1"]["publication_metadata"]["doi"] == "10.old/x"

    def test_force_refetches_crossref(self) -> None:
        from enrich_ann_merged import process_doc  # type: ignore[import]
        merged = _FakeDb(docs={"d1": {
            "_id": "d1",
            "publication_metadata": {"doi": "10.old/x"},
        }})
        training = _FakeDb(
            docs={"d1": {"_id": "d1"}},
            attachments={"d1": {"article.txt": (
                "--- PDF Page 1 Label 1 ---\n"
                "DOI 10.1234/abc.def\n"
                "--- PDF Page 2 Label 2 ---"
            ).encode("utf-8")}},
        )
        golden = _FakeDb()
        http = _StubHttp({"message": _SAMPLE_CROSSREF_MESSAGE})
        process_doc("d1", merged, training, golden, http, force=True)
        assert len(http.calls) == 1
        assert merged.docs["d1"]["publication_metadata"]["doi"] \
            == "10.1234/abc.def"

    def test_no_doi_leaves_metadata_unset(self) -> None:
        from enrich_ann_merged import process_doc  # type: ignore[import]
        merged = _FakeDb(docs={"d1": {"_id": "d1"}})
        training = _FakeDb(
            docs={"d1": {"_id": "d1"}},
            attachments={"d1": {"article.txt": b"no doi here at all"}},
        )
        golden = _FakeDb()
        http = _StubHttp({"message": _SAMPLE_CROSSREF_MESSAGE})
        process_doc("d1", merged, training, golden, http)
        assert "publication_metadata" not in merged.docs["d1"]
        assert http.calls == []
