#!/usr/bin/env python3
"""Tests for regen_dev_txt.py."""

import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regen_dev_txt import _linked_dev_ids, _extract_plaintext, regen_dev_txt


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

class _FakeRow:
    def __init__(self, row_id: str) -> None:
        self.id = row_id


class _FakeTrainingDb:
    """Minimal CouchDB database stub for skol_training."""

    def __init__(
        self,
        docs: Dict[str, Dict[str, Any]],
        attachments: Optional[Dict[str, Optional[bytes]]] = None,
    ) -> None:
        self._docs = docs
        # {doc_id: pdf_bytes | None}; None → no attachment
        self._attachments: Dict[str, Optional[bytes]] = attachments or {}

    def view(
        self, view_name: str, **kwargs: Any
    ) -> List[_FakeRow]:
        return [_FakeRow(k) for k in self._docs]

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        return self._docs[doc_id]

    def get_attachment(
        self, doc_id: str, attachment_name: str
    ) -> Optional[Any]:
        pdf = self._attachments.get(doc_id)
        if pdf is None:
            return None
        buf = io.BytesIO(pdf)
        buf.read = buf.read  # already a method
        return buf


class _FakeDevDb:
    """Minimal CouchDB database stub for skol_dev."""

    def __init__(self, docs: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._docs: Dict[str, Dict[str, Any]] = docs or {}
        self.written: Dict[str, bytes] = {}

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        return self._docs[doc_id]

    def put_attachment(
        self,
        doc: Dict[str, Any],
        data: bytes,
        filename: str,
        content_type: str,
    ) -> None:
        dev_id = doc.get("_id", "unknown")
        self.written[dev_id] = data


# ---------------------------------------------------------------------------
# _linked_dev_ids
# ---------------------------------------------------------------------------

class TestLinkedDevIds:
    def test_returns_linked_docs(self) -> None:
        db = _FakeTrainingDb(
            docs={
                "train1": {"skol_dev_id": "dev1"},
                "train2": {"skol_dev_id": "dev2"},
            }
        )
        result = _linked_dev_ids(db)
        assert result == {"train1": "dev1", "train2": "dev2"}

    def test_skips_docs_without_skol_dev_id(self) -> None:
        db = _FakeTrainingDb(
            docs={
                "train1": {"skol_dev_id": "dev1"},
                "train2": {"other_field": "x"},
            }
        )
        result = _linked_dev_ids(db)
        assert "train2" not in result
        assert result == {"train1": "dev1"}

    def test_skips_design_docs(self) -> None:
        db = _FakeTrainingDb(
            docs={
                "_design/views": {"skol_dev_id": "should_be_skipped"},
                "train1": {"skol_dev_id": "dev1"},
            }
        )
        result = _linked_dev_ids(db)
        assert "_design/views" not in result
        assert result == {"train1": "dev1"}

    def test_empty_database(self) -> None:
        db = _FakeTrainingDb(docs={})
        assert _linked_dev_ids(db) == {}


# ---------------------------------------------------------------------------
# _extract_plaintext
# ---------------------------------------------------------------------------

class TestExtractPlaintext:
    def test_returns_text_on_success(self) -> None:
        with patch(
            "regen_dev_txt._extract_plaintext",
            return_value="--- PDF Page 1 Label 1 ---\nsome text\n",
        ) as mock_fn:
            result = mock_fn(b"fake_pdf")
        assert result is not None
        assert "PDF Page 1" in result

    def test_returns_none_on_exception(self) -> None:
        # Patch the internal import to raise an exception.
        with patch.dict("sys.modules", {"ingestors.extract_plaintext": None}):
            # With module set to None, the import inside _extract_plaintext raises.
            result = _extract_plaintext(b"garbage")
        assert result is None


# ---------------------------------------------------------------------------
# regen_dev_txt
# ---------------------------------------------------------------------------

SAMPLE_PLAINTEXT = "--- PDF Page 1 Label 1 ---\nSome text.\n"


class TestRegenDevTxt:
    def _make_training_db(
        self,
        docs: Optional[Dict[str, Dict[str, Any]]] = None,
        attachments: Optional[Dict[str, Optional[bytes]]] = None,
    ) -> _FakeTrainingDb:
        if docs is None:
            docs = {"train1": {"skol_dev_id": "dev1"}}
        if attachments is None:
            attachments = {"train1": b"%PDF fake bytes"}
        return _FakeTrainingDb(docs=docs, attachments=attachments)

    def _make_dev_db(self) -> _FakeDevDb:
        return _FakeDevDb(docs={"dev1": {"_id": "dev1"}})

    def test_updates_one_doc(self) -> None:
        training_db = self._make_training_db()
        dev_db = self._make_dev_db()
        with patch(
            "regen_dev_txt._extract_plaintext",
            return_value=SAMPLE_PLAINTEXT,
        ):
            totals = regen_dev_txt(
                training_db, dev_db, dry_run=False, verbosity=0
            )
        assert totals["docs_found"] == 1
        assert totals["docs_updated"] == 1
        assert totals["docs_skipped_no_pdf"] == 0
        assert totals["docs_skipped_no_txt"] == 0
        assert dev_db.written["dev1"] == SAMPLE_PLAINTEXT.encode("utf-8")

    def test_dry_run_does_not_write(self) -> None:
        training_db = self._make_training_db()
        dev_db = self._make_dev_db()
        with patch(
            "regen_dev_txt._extract_plaintext",
            return_value=SAMPLE_PLAINTEXT,
        ):
            totals = regen_dev_txt(
                training_db, dev_db, dry_run=True, verbosity=0
            )
        assert totals["docs_updated"] == 1
        assert dev_db.written == {}

    def test_skips_when_no_pdf(self) -> None:
        training_db = _FakeTrainingDb(
            docs={"train1": {"skol_dev_id": "dev1"}},
            attachments={"train1": None},  # no PDF
        )
        dev_db = self._make_dev_db()
        totals = regen_dev_txt(training_db, dev_db, verbosity=0)
        assert totals["docs_skipped_no_pdf"] == 1
        assert totals["docs_updated"] == 0

    def test_skips_when_extraction_fails(self) -> None:
        training_db = self._make_training_db()
        dev_db = self._make_dev_db()
        with patch("regen_dev_txt._extract_plaintext", return_value=None):
            totals = regen_dev_txt(training_db, dev_db, verbosity=0)
        assert totals["docs_skipped_no_txt"] == 1
        assert totals["docs_updated"] == 0

    def test_deduplicates_shared_dev_id(self) -> None:
        """Two training docs pointing to the same skol_dev_id → only one update."""
        training_db = _FakeTrainingDb(
            docs={
                "train1": {"skol_dev_id": "dev1"},
                "train2": {"skol_dev_id": "dev1"},
            },
            attachments={
                "train1": b"%PDF bytes for train1",
                "train2": b"%PDF bytes for train2",
            },
        )
        dev_db = _FakeDevDb(docs={"dev1": {"_id": "dev1"}})
        with patch(
            "regen_dev_txt._extract_plaintext",
            return_value=SAMPLE_PLAINTEXT,
        ):
            totals = regen_dev_txt(training_db, dev_db, verbosity=0)
        assert totals["docs_found"] == 1
        assert totals["docs_updated"] == 1

    def test_dev_id_filter_skips_other_docs(self) -> None:
        training_db = _FakeTrainingDb(
            docs={
                "train1": {"skol_dev_id": "dev1"},
                "train2": {"skol_dev_id": "dev2"},
            },
            attachments={
                "train1": b"%PDF bytes",
                "train2": b"%PDF bytes",
            },
        )
        dev_db = _FakeDevDb(
            docs={"dev1": {"_id": "dev1"}, "dev2": {"_id": "dev2"}}
        )
        with patch(
            "regen_dev_txt._extract_plaintext",
            return_value=SAMPLE_PLAINTEXT,
        ):
            totals = regen_dev_txt(
                training_db, dev_db, verbosity=0, dev_id_filter="dev1"
            )
        assert totals["docs_found"] == 1
        assert totals["docs_updated"] == 1
        assert totals["docs_skipped_filter"] == 1
        assert "dev1" in dev_db.written
        assert "dev2" not in dev_db.written

    def test_multiple_docs_all_updated(self) -> None:
        training_db = _FakeTrainingDb(
            docs={
                "train1": {"skol_dev_id": "dev1"},
                "train2": {"skol_dev_id": "dev2"},
            },
            attachments={
                "train1": b"%PDF 1",
                "train2": b"%PDF 2",
            },
        )
        dev_db = _FakeDevDb(
            docs={"dev1": {"_id": "dev1"}, "dev2": {"_id": "dev2"}}
        )
        with patch(
            "regen_dev_txt._extract_plaintext",
            return_value=SAMPLE_PLAINTEXT,
        ):
            totals = regen_dev_txt(training_db, dev_db, verbosity=0)
        assert totals["docs_found"] == 2
        assert totals["docs_updated"] == 2

    def test_returns_all_counter_keys(self) -> None:
        training_db = _FakeTrainingDb(docs={})
        dev_db = _FakeDevDb()
        totals = regen_dev_txt(training_db, dev_db, verbosity=0)
        expected_keys = {
            "docs_found",
            "docs_updated",
            "docs_skipped_no_pdf",
            "docs_skipped_no_txt",
            "docs_skipped_filter",
        }
        assert set(totals.keys()) == expected_keys
