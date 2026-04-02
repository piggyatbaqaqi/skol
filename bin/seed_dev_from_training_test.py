#!/usr/bin/env python3
"""Tests for seed_dev_from_training.py."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from seed_dev_from_training import (
    _COPY_FIELDS,
    _build_dev_doc,
    _find_docs_needing_seeding,
    seed_dev_from_training,
)


# ---------------------------------------------------------------------------
# _build_dev_doc
# ---------------------------------------------------------------------------

class TestBuildDevDoc:
    def _make_training_doc(self, **kwargs) -> Dict[str, Any]:
        base = {
            "_id": "abc123",
            "_rev": "1-xyz",
            "journal": "Mycotaxon",
            "volume": "119",
            "itemtype": "article",
            "skol_dev_id": None,
        }
        base.update(kwargs)
        return base

    def test_copies_journal_and_volume(self):
        doc = _build_dev_doc(self._make_training_doc())
        assert doc["journal"] == "Mycotaxon"
        assert doc["volume"] == "119"

    def test_omits_null_fields(self):
        training = self._make_training_doc(author=None, title=None)
        doc = _build_dev_doc(training)
        assert "author" not in doc
        assert "title" not in doc

    def test_includes_non_null_optional_fields(self):
        training = self._make_training_doc(
            author="Smith, J.",
            title="A new species",
            year="2012",
            pages="1-10",
        )
        doc = _build_dev_doc(training)
        assert doc["author"] == "Smith, J."
        assert doc["title"] == "A new species"
        assert doc["year"] == "2012"
        assert doc["pages"] == "1-10"

    def test_seeded_from_training_field(self):
        doc = _build_dev_doc(self._make_training_doc())
        assert doc["seeded_from_training"] == "abc123"

    def test_has_create_and_modification_time(self):
        doc = _build_dev_doc(self._make_training_doc())
        assert "create_time" in doc
        assert "modification_time" in doc
        # ISO-8601 format check
        assert "T" in doc["create_time"]

    def test_meta_is_empty_dict(self):
        doc = _build_dev_doc(self._make_training_doc())
        assert doc["meta"] == {}

    def test_does_not_copy_internal_fields(self):
        """_id, _rev, skol_dev_id are not copied into the new doc."""
        doc = _build_dev_doc(self._make_training_doc())
        assert "_id" not in doc
        assert "_rev" not in doc
        assert "skol_dev_id" not in doc

    def test_all_copy_fields_attempted(self):
        """All _COPY_FIELDS present in training doc appear in output."""
        training = self._make_training_doc(
            **{f: f"val_{f}" for f in _COPY_FIELDS}
        )
        doc = _build_dev_doc(training)
        for field in _COPY_FIELDS:
            assert doc[field] == f"val_{field}"


# ---------------------------------------------------------------------------
# _find_docs_needing_seeding (uses mock CouchDB)
# ---------------------------------------------------------------------------

class TestFindDocsNeedingSeeding:
    def _make_db(self, docs: List[Dict[str, Any]]) -> Any:
        """Build a mock CouchDB database returning the given docs."""
        db = MagicMock()
        rows = []
        for doc in docs:
            row = MagicMock()
            row.id = doc["_id"]
            rows.append(row)
        db.view.return_value = rows
        db.__getitem__ = lambda self_, key: next(
            d for d in docs if d["_id"] == key
        )
        return db

    def test_returns_ids_without_skol_dev_id(self):
        docs = [
            {"_id": "aaa", "skol_dev_id": "dev_1"},
            {"_id": "bbb"},                          # missing field
            {"_id": "ccc", "skol_dev_id": None},     # explicit None
            {"_id": "ddd", "skol_dev_id": "dev_2"},
        ]
        db = self._make_db(docs)
        missing = _find_docs_needing_seeding(db)
        assert set(missing) == {"bbb", "ccc"}

    def test_skips_design_docs(self):
        docs = [
            {"_id": "_design/views"},
            {"_id": "aaa"},
        ]
        db = self._make_db(docs)
        missing = _find_docs_needing_seeding(db)
        assert "_design/views" not in missing
        assert "aaa" in missing

    def test_empty_db(self):
        db = self._make_db([])
        assert _find_docs_needing_seeding(db) == []

    def test_all_have_skol_dev_id(self):
        docs = [
            {"_id": "aaa", "skol_dev_id": "dev_1"},
            {"_id": "bbb", "skol_dev_id": "dev_2"},
        ]
        db = self._make_db(docs)
        assert _find_docs_needing_seeding(db) == []


# ---------------------------------------------------------------------------
# seed_dev_from_training (end-to-end with mocks)
# ---------------------------------------------------------------------------

class TestSeedDevFromTraining:
    def _make_training_db(self, doc: Dict[str, Any], pdf_bytes: bytes) -> Any:
        """Build a mock skol_training database with one document."""
        db = MagicMock()
        row = MagicMock()
        row.id = doc["_id"]
        db.view.return_value = [row]
        db.__getitem__ = MagicMock(return_value=dict(doc))
        att = MagicMock()
        att.read.return_value = pdf_bytes
        db.get_attachment.return_value = att
        db.save = MagicMock()
        return db

    def _make_dev_db(self) -> Any:
        db = MagicMock()
        db.__setitem__ = MagicMock()
        db.__getitem__ = MagicMock(return_value={})
        db.put_attachment = MagicMock()
        return db

    def _make_server(self, uuid: str = "new_uuid_001") -> Any:
        server = MagicMock()
        server.uuids.return_value = [uuid]
        return server

    def test_dry_run_does_not_write(self):
        training_doc = {"_id": "train1", "journal": "Mycotaxon", "volume": "1",
                        "itemtype": "article"}
        training_db = self._make_training_db(training_doc, b"%PDF-fake")
        dev_db = self._make_dev_db()
        server = self._make_server()

        with patch("seed_dev_from_training._extract_plaintext",
                   return_value="Plain text content."):
            summary = seed_dev_from_training(
                training_db, dev_db, server, dry_run=True, verbosity=0
            )

        assert summary["docs_found"] == 1
        assert summary["docs_seeded"] == 1
        # No writes in dry-run.
        dev_db.__setitem__.assert_not_called()
        dev_db.put_attachment.assert_not_called()
        training_db.save.assert_not_called()

    def test_creates_dev_doc_and_updates_training(self):
        training_doc = {"_id": "train1", "journal": "Mycotaxon", "volume": "2",
                        "itemtype": "article"}
        training_db = self._make_training_db(training_doc, b"%PDF-fake")
        dev_db = self._make_dev_db()
        server = self._make_server("generated_uuid")

        with patch("seed_dev_from_training._extract_plaintext",
                   return_value="--- PDF Page 1 Label 1 ---\nHeader\nText."):
            summary = seed_dev_from_training(
                training_db, dev_db, server, dry_run=False, verbosity=0
            )

        assert summary["docs_seeded"] == 1
        assert summary["docs_skipped_no_pdf"] == 0

        # dev_db should have been written to.
        dev_db.__setitem__.assert_called_once()
        assert dev_db.put_attachment.call_count == 2  # PDF + TXT

        # skol_training should have been updated.
        training_db.save.assert_called_once()
        saved_doc = training_db.save.call_args[0][0]
        assert saved_doc["skol_dev_id"] == "generated_uuid"

    def test_skips_doc_without_pdf(self):
        training_doc = {"_id": "train1", "journal": "Mycotaxon", "volume": "3",
                        "itemtype": "article"}
        training_db = self._make_training_db(training_doc, b"")
        training_db.get_attachment.return_value = None  # no PDF

        dev_db = self._make_dev_db()
        server = self._make_server()

        summary = seed_dev_from_training(
            training_db, dev_db, server, dry_run=False, verbosity=0
        )

        assert summary["docs_skipped_no_pdf"] == 1
        assert summary["docs_seeded"] == 0
        dev_db.__setitem__.assert_not_called()

    def test_skips_doc_when_text_extraction_fails(self):
        training_doc = {"_id": "train1", "journal": "Mycotaxon", "volume": "4",
                        "itemtype": "article"}
        training_db = self._make_training_db(training_doc, b"%PDF-fake")
        dev_db = self._make_dev_db()
        server = self._make_server()

        with patch("seed_dev_from_training._extract_plaintext", return_value=None):
            summary = seed_dev_from_training(
                training_db, dev_db, server, dry_run=False, verbosity=0
            )

        assert summary["docs_skipped_no_text"] == 1
        assert summary["docs_seeded"] == 0
        dev_db.__setitem__.assert_not_called()

    def test_no_docs_to_seed(self):
        """All skol_training docs already have skol_dev_id → nothing done."""
        training_db = MagicMock()
        row = MagicMock()
        row.id = "train1"
        training_db.view.return_value = [row]
        training_db.__getitem__ = MagicMock(
            return_value={"_id": "train1", "skol_dev_id": "existing_dev_id"}
        )

        dev_db = self._make_dev_db()
        server = self._make_server()

        summary = seed_dev_from_training(
            training_db, dev_db, server, dry_run=False, verbosity=0
        )

        assert summary["docs_found"] == 0
        assert summary["docs_seeded"] == 0
        dev_db.__setitem__.assert_not_called()
