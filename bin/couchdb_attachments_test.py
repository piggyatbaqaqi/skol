"""Tests for bin/couchdb_attachments.py (extract/insert helpers)."""

import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from couchdb_attachments import (  # type: ignore[import]  # noqa: E402
    _content_type_for,
    extract_doc,
    insert_doc,
    select_doc_ids,
)


# ---------------------------------------------------------------------------
# In-memory CouchDB stand-in
# ---------------------------------------------------------------------------


class FakeDb:
    """Minimal CouchDB-like object used to drive extract_doc / insert_doc."""

    def __init__(self, docs: Optional[Dict[str, Dict[str, Any]]] = None,
                 attachments: Optional[Dict[str, Dict[str, bytes]]] = None,
                 ) -> None:
        self.docs: Dict[str, Dict[str, Any]] = docs or {}
        # attachments[doc_id][filename] = bytes
        self.attachments: Dict[str, Dict[str, bytes]] = attachments or {}

    # --- iteration helpers (extract path) -----------------------------------

    def view(self, name: str, include_docs: bool = False):  # noqa: ARG002
        rows = []
        for doc_id in sorted(self.docs):
            row = MagicMock()
            row.id = doc_id
            rows.append(row)
        return rows

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        if doc_id not in self.docs:
            raise KeyError(doc_id)
        # Mirror CouchDB's _attachments stub-listing behaviour.
        doc = dict(self.docs[doc_id])
        atts = self.attachments.get(doc_id, {})
        if atts:
            doc["_attachments"] = {n: {} for n in atts}
        return doc

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.docs

    def get_attachment(self, doc_id: str, name: str):
        data = self.attachments.get(doc_id, {}).get(name)
        return io.BytesIO(data) if data is not None else None

    # --- mutation helpers (insert path) -------------------------------------

    def save(self, doc: Dict[str, Any]) -> None:
        doc_id = doc["_id"]
        self.docs[doc_id] = dict(doc)

    def put_attachment(self, doc: Dict[str, Any], content: bytes,
                       filename: str, content_type: str) -> None:
        doc_id = doc["_id"]
        self.attachments.setdefault(doc_id, {})[filename] = content
        # Simulate revision increment so callers fetching afresh still work.
        self.docs.setdefault(doc_id, {"_id": doc_id})


# ---------------------------------------------------------------------------
# select_doc_ids
# ---------------------------------------------------------------------------


class TestSelectDocIds:
    def test_returns_explicit_list(self) -> None:
        db = FakeDb(docs={"a": {}, "b": {}, "c": {}})
        assert select_doc_ids(db, ["b", "a"]) == ["b", "a"]

    def test_lists_all_when_doc_ids_none(self) -> None:
        db = FakeDb(docs={"a": {}, "b": {}, "_design/x": {}})
        assert select_doc_ids(db, None) == ["a", "b"]


# ---------------------------------------------------------------------------
# extract_doc
# ---------------------------------------------------------------------------


class TestExtractDoc:
    def test_writes_each_attachment(self, tmp_path: Path) -> None:
        db = FakeDb(
            docs={"d1": {"_id": "d1"}},
            attachments={"d1": {"article.txt": b"hello",
                                "article.pdf": b"%PDF"}},
        )
        n = extract_doc(db, "d1", tmp_path)
        assert n == 2
        assert (tmp_path / "d1" / "article.txt").read_bytes() == b"hello"
        assert (tmp_path / "d1" / "article.pdf").read_bytes() == b"%PDF"

    def test_no_attachments_creates_no_dir(self, tmp_path: Path) -> None:
        db = FakeDb(docs={"d1": {"_id": "d1"}})
        n = extract_doc(db, "d1", tmp_path)
        assert n == 0
        assert not (tmp_path / "d1").exists()

    def test_missing_document_logs_and_returns_zero(
        self, tmp_path: Path,
    ) -> None:
        db = FakeDb()
        assert extract_doc(db, "ghost", tmp_path) == 0
        assert not (tmp_path / "ghost").exists()


# ---------------------------------------------------------------------------
# insert_doc
# ---------------------------------------------------------------------------


class TestInsertDoc:
    def test_uploads_each_file(self, tmp_path: Path) -> None:
        src = tmp_path / "d1"
        src.mkdir()
        (src / "article.txt").write_bytes(b"hello")
        (src / "article.pdf").write_bytes(b"%PDF")
        db = FakeDb()
        n = insert_doc(db, "d1", src)
        assert n == 2
        assert db.attachments["d1"]["article.txt"] == b"hello"
        assert db.attachments["d1"]["article.pdf"] == b"%PDF"
        # Doc was created (cp semantics: missing target → create).
        assert "d1" in db.docs

    def test_overwrites_existing_attachment(self, tmp_path: Path) -> None:
        src = tmp_path / "d1"
        src.mkdir()
        (src / "article.txt").write_bytes(b"NEW")
        db = FakeDb(
            docs={"d1": {"_id": "d1"}},
            attachments={"d1": {"article.txt": b"OLD"}},
        )
        insert_doc(db, "d1", src)
        assert db.attachments["d1"]["article.txt"] == b"NEW"

    def test_skips_hidden_files_and_subdirs(self, tmp_path: Path) -> None:
        src = tmp_path / "d1"
        (src / "nested").mkdir(parents=True)
        (src / "article.txt").write_bytes(b"hello")
        (src / ".DS_Store").write_bytes(b"junk")
        db = FakeDb()
        n = insert_doc(db, "d1", src)
        assert n == 1
        assert list(db.attachments["d1"]) == ["article.txt"]


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_extract_then_insert_preserves_content(
        self, tmp_path: Path,
    ) -> None:
        original = FakeDb(
            docs={"d1": {"_id": "d1"}, "d2": {"_id": "d2"}},
            attachments={
                "d1": {"a.txt": b"alpha", "b.txt": b"beta"},
                "d2": {"c.bin": b"\x00\x01\x02"},
            },
        )
        from couchdb_attachments import (  # type: ignore[import]
            cmd_extract,
            cmd_insert,
        )
        cmd_extract(original, tmp_path, doc_ids=None, verbosity=0)

        target = FakeDb()
        cmd_insert(target, tmp_path, doc_ids=None, verbosity=0)

        assert target.attachments == original.attachments


# ---------------------------------------------------------------------------
# content type guess
# ---------------------------------------------------------------------------


class TestContentType:
    def test_known_extensions(self) -> None:
        assert _content_type_for("a.txt").startswith("text/")
        assert _content_type_for("a.pdf") == "application/pdf"

    def test_unknown_extension_falls_back_to_octet_stream(self) -> None:
        assert _content_type_for("a.unknownext") == "application/octet-stream"
        assert _content_type_for("noext") == "application/octet-stream"


__all__: List[str] = []  # silence "unused import" warnings on tests
