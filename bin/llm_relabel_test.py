"""Tests for bin/llm_relabel.py.

Covers prompt construction, YEDDA diff, block-count validation,
and CouchDB staging helpers — all without making real API calls.

Run with: python -m pytest bin/llm_relabel_test.py -v
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_relabel import (
    _build_user_prompt,
    _write_to_staging,
    diff_yedda,
    relabel_ann,
)
from ingestors.yedda_tags import Tag

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ANN_8TAG = (
    "[@Amanita muscaria (L.) Lam.#Nomenclature*]\n\n"
    "[@Pileus 5–15 cm diam., convex then flat.#Description*]\n\n"
    "[@Holotype: NY 12345.#Holotype*]\n\n"
    "[@This species is common in boreal forests.#Misc-exposition*]\n"
)

_ANN_12TAG = (
    "[@Amanita muscaria (L.) Lam.#Nomenclature*]\n\n"
    "[@Pileus 5–15 cm diam., convex then flat.#Description*]\n\n"
    "[@Holotype: NY 12345.#Type-designation*]\n\n"
    "[@This species is common in boreal forests.#Biology*]\n"
)


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt(unittest.TestCase):

    def test_contains_all_tags(self) -> None:
        prompt = _build_user_prompt(_ANN_8TAG)
        for tag in Tag:
            if tag == Tag.HOLOTYPE:
                continue  # deprecated, not in definitions
            self.assertIn(tag.value, prompt)

    def test_contains_input_text(self) -> None:
        prompt = _build_user_prompt(_ANN_8TAG)
        self.assertIn("Amanita muscaria", prompt)

    def test_contains_rules(self) -> None:
        prompt = _build_user_prompt(_ANN_8TAG)
        self.assertIn("Do NOT change any text content", prompt)
        self.assertIn("Type-designation", prompt)

    def test_contains_tag_definitions(self) -> None:
        prompt = _build_user_prompt(_ANN_8TAG)
        self.assertIn("Differential diagnosis", prompt)
        self.assertIn("holotype/lectotype", prompt.lower())


# ---------------------------------------------------------------------------
# diff_yedda
# ---------------------------------------------------------------------------


class TestDiffYedda(unittest.TestCase):

    def test_no_changes_returns_empty(self) -> None:
        self.assertEqual(diff_yedda(_ANN_8TAG, _ANN_8TAG), [])

    def test_detects_changed_blocks(self) -> None:
        changes = diff_yedda(_ANN_8TAG, _ANN_12TAG)
        old_tags = {c["old_tag"] for c in changes}
        new_tags = {c["new_tag"] for c in changes}
        self.assertIn("Holotype", old_tags)
        self.assertIn("Type-designation", new_tags)
        self.assertIn("Misc-exposition", old_tags)
        self.assertIn("Biology", new_tags)

    def test_change_has_required_keys(self) -> None:
        changes = diff_yedda(_ANN_8TAG, _ANN_12TAG)
        for c in changes:
            self.assertIn("block_index", c)
            self.assertIn("old_tag", c)
            self.assertIn("new_tag", c)
            self.assertIn("snippet", c)

    def test_snippet_truncated_to_80_chars(self) -> None:
        long_text = "x" * 200
        old = f"[@{long_text}#Description*]\n"
        new = f"[@{long_text}#Diagnosis*]\n"
        changes = diff_yedda(old, new)
        self.assertLessEqual(len(changes[0]["snippet"]), 80)

    def test_block_index_is_correct(self) -> None:
        changes = diff_yedda(_ANN_8TAG, _ANN_12TAG)
        indices = {c["block_index"] for c in changes}
        # Block 2 (Holotype) and block 3 (Misc-exposition) changed
        self.assertIn(2, indices)
        self.assertIn(3, indices)


# ---------------------------------------------------------------------------
# relabel_ann
# ---------------------------------------------------------------------------


class TestRelabelAnn(unittest.TestCase):

    def _make_client(self, response_text: str) -> MagicMock:
        client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text=response_text)]
        client.messages.create.return_value = msg
        return client

    def test_returns_new_text_and_changes(self) -> None:
        client = self._make_client(_ANN_12TAG)
        new_text, changes = relabel_ann(client, _ANN_8TAG, "test-model", "doc1")
        self.assertIn("Type-designation", new_text)
        self.assertTrue(len(changes) > 0)

    def test_unchanged_returns_empty_diff(self) -> None:
        client = self._make_client(_ANN_8TAG)
        _, changes = relabel_ann(client, _ANN_8TAG, "test-model", "doc1")
        self.assertEqual(changes, [])

    def test_block_count_mismatch_raises(self) -> None:
        # Response has fewer blocks than input
        bad_response = "[@Amanita muscaria (L.) Lam.#Nomenclature*]\n"
        client = self._make_client(bad_response)
        with self.assertRaises(RuntimeError):
            relabel_ann(client, _ANN_8TAG, "test-model", "doc1")

    def test_empty_response_raises(self) -> None:
        client = self._make_client("   ")
        with self.assertRaises(RuntimeError):
            relabel_ann(client, _ANN_8TAG, "test-model", "doc1")

    def test_retries_on_transient_error(self) -> None:
        client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text=_ANN_12TAG)]
        # Fail twice, succeed on third attempt
        client.messages.create.side_effect = [
            Exception("timeout"),
            Exception("timeout"),
            msg,
        ]
        with patch("llm_relabel.time.sleep"):
            new_text, _ = relabel_ann(client, _ANN_8TAG, "test-model", "doc1")
        self.assertEqual(client.messages.create.call_count, 3)
        self.assertIn("Type-designation", new_text)

    def test_all_retries_exhausted_raises(self) -> None:
        client = MagicMock()
        client.messages.create.side_effect = Exception("persistent error")
        with patch("llm_relabel.time.sleep"):
            with self.assertRaises(RuntimeError):
                relabel_ann(client, _ANN_8TAG, "test-model", "doc1")


# ---------------------------------------------------------------------------
# _write_to_staging
# ---------------------------------------------------------------------------


class TestWriteToStaging(unittest.TestCase):

    def _make_staging_db(self, doc_exists: bool = False) -> MagicMock:
        db = MagicMock()
        if doc_exists:
            db.__contains__ = MagicMock(return_value=True)
            db.__getitem__ = MagicMock(return_value={"_id": "doc1", "_rev": "1-abc"})
        else:
            db.__contains__ = MagicMock(return_value=False)
        return db

    def test_creates_new_doc_when_absent(self) -> None:
        db = self._make_staging_db(doc_exists=False)
        db.__getitem__ = MagicMock(return_value={"_id": "doc1", "_rev": "1-new"})
        _write_to_staging(db, "skol_training", "doc1", _ANN_12TAG, [])
        db.save.assert_called_once()
        saved_doc = db.save.call_args[0][0]
        self.assertEqual(saved_doc["_id"], "doc1")
        self.assertEqual(saved_doc["source_db"], "skol_training")
        self.assertTrue(saved_doc["llm_relabeled"])

    def test_skips_save_when_doc_exists(self) -> None:
        db = self._make_staging_db(doc_exists=True)
        _write_to_staging(db, "skol_training", "doc1", _ANN_12TAG, [])
        db.save.assert_not_called()

    def test_always_puts_attachment(self) -> None:
        db = self._make_staging_db(doc_exists=True)
        _write_to_staging(db, "skol_training", "doc1", _ANN_12TAG, [{"x": 1}])
        db.put_attachment.assert_called_once()
        _, ann_bytes, *_ = db.put_attachment.call_args[0]
        self.assertIn(b"Type-designation", ann_bytes)

    def test_change_count_stored_on_new_doc(self) -> None:
        db = self._make_staging_db(doc_exists=False)
        db.__getitem__ = MagicMock(return_value={"_id": "doc1", "_rev": "1-new"})
        changes = [{"block_index": 0, "old_tag": "Holotype",
                    "new_tag": "Type-designation", "snippet": "x"}]
        _write_to_staging(db, "src", "doc1", _ANN_12TAG, changes)
        saved_doc = db.save.call_args[0][0]
        self.assertEqual(saved_doc["change_count"], 1)


if __name__ == "__main__":
    unittest.main()
