"""Tests for bin/llm_relabel.py.

Covers prompt construction, YEDDA diff, block-count validation,
oversized-document skipping, and CouchDB staging helpers — all
without making real API calls.

Run with: python -m pytest bin/llm_relabel_test.py -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_relabel import (
    _DEFAULT_CHUNK_SIZE,
    _MAX_BLOCKS,
    _YEDDA_BLOCK_RE,
    _build_user_prompt,
    _write_to_staging,
    chunk_ann,
    diff_yedda,
    process_documents,
    relabel_ann,
    relabel_ann_chunked,
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


def _make_oversized_ann(n_blocks: int) -> str:
    """Build a synthetic .ann with n_blocks Description blocks."""
    return "\n\n".join(
        f"[@Block {i}.#Description*]" for i in range(n_blocks)
    ) + "\n"


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
        new_text, changes = relabel_ann(
            client, _ANN_8TAG, "test-model", "doc1"
        )
        self.assertIn("Type-designation", new_text)
        self.assertTrue(len(changes) > 0)

    def test_unchanged_returns_empty_diff(self) -> None:
        client = self._make_client(_ANN_8TAG)
        _, changes = relabel_ann(client, _ANN_8TAG, "test-model", "doc1")
        self.assertEqual(changes, [])

    def test_block_count_mismatch_raises(self) -> None:
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
        client.messages.create.side_effect = [
            Exception("timeout"),
            Exception("timeout"),
            msg,
        ]
        with patch("llm_relabel.time.sleep"):
            new_text, _ = relabel_ann(
                client, _ANN_8TAG, "test-model", "doc1"
            )
        self.assertEqual(client.messages.create.call_count, 3)
        self.assertIn("Type-designation", new_text)

    def test_all_retries_exhausted_raises(self) -> None:
        client = MagicMock()
        client.messages.create.side_effect = Exception("persistent error")
        with patch("llm_relabel.time.sleep"):
            with self.assertRaises(RuntimeError):
                relabel_ann(client, _ANN_8TAG, "test-model", "doc1")

    def test_uses_max_tokens_8192(self) -> None:
        client = self._make_client(_ANN_8TAG)
        relabel_ann(client, _ANN_8TAG, "test-model", "doc1")
        _, kwargs = client.messages.create.call_args
        self.assertEqual(kwargs.get("max_tokens") or
                         client.messages.create.call_args[1].get("max_tokens"),
                         8192)


# ---------------------------------------------------------------------------
# process_documents — oversized document skipping
# ---------------------------------------------------------------------------


class TestProcessDocumentsSkipsOversized(unittest.TestCase):
    """Oversized documents are filtered before any API call is made."""

    def _make_client(self, response_text: str) -> MagicMock:
        client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text=response_text)]
        client.messages.create.return_value = msg
        return client

    def _make_staging_db(self) -> MagicMock:
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)
        db.__getitem__ = MagicMock(return_value={"_id": "x", "_rev": "1"})
        return db

    def test_oversized_doc_not_sent_to_api(self) -> None:
        big = _make_oversized_ann(_MAX_BLOCKS + 1)
        client = self._make_client(_ANN_12TAG)
        staging = self._make_staging_db()
        process_documents(
            client=client,
            source_db=MagicMock(),
            source_db_name="src",
            staging_db=staging,
            ann_texts={"big_doc": big},
            model="test-model",
            workers=1,
            dry_run=True,
            log_file=None,
            verbosity=0,
            max_blocks=_MAX_BLOCKS,
        )
        client.messages.create.assert_not_called()

    def test_oversized_counted_in_skipped(self) -> None:
        big = _make_oversized_ann(_MAX_BLOCKS + 1)
        client = self._make_client(_ANN_12TAG)
        staging = self._make_staging_db()
        summary = process_documents(
            client=client,
            source_db=MagicMock(),
            source_db_name="src",
            staging_db=staging,
            ann_texts={"big_doc": big},
            model="test-model",
            workers=1,
            dry_run=True,
            log_file=None,
            verbosity=0,
            max_blocks=_MAX_BLOCKS,
        )
        self.assertEqual(summary["docs_skipped"], 1)
        self.assertEqual(summary["docs_processed"], 0)

    def test_normal_doc_processed_alongside_oversized(self) -> None:
        big = _make_oversized_ann(_MAX_BLOCKS + 1)
        client = self._make_client(_ANN_8TAG)
        staging = self._make_staging_db()
        summary = process_documents(
            client=client,
            source_db=MagicMock(),
            source_db_name="src",
            staging_db=staging,
            ann_texts={"big_doc": big, "small_doc": _ANN_8TAG},
            model="test-model",
            workers=1,
            dry_run=True,
            log_file=None,
            verbosity=0,
            max_blocks=_MAX_BLOCKS,
        )
        self.assertEqual(summary["docs_skipped"], 1)
        self.assertEqual(summary["docs_processed"], 1)
        client.messages.create.assert_called_once()

    def test_custom_max_blocks_respected(self) -> None:
        five_blocks = _make_oversized_ann(5)
        client = self._make_client(five_blocks)
        staging = self._make_staging_db()
        summary = process_documents(
            client=client,
            source_db=MagicMock(),
            source_db_name="src",
            staging_db=staging,
            ann_texts={"doc": five_blocks},
            model="test-model",
            workers=1,
            dry_run=True,
            log_file=None,
            verbosity=0,
            max_blocks=4,  # lower than 5
        )
        self.assertEqual(summary["docs_skipped"], 1)
        client.messages.create.assert_not_called()

    def test_default_max_blocks_constant_is_sane(self) -> None:
        # Sanity-check constant is sane; large because chunking handles big docs
        self.assertGreater(_MAX_BLOCKS, 50)
        self.assertLess(_MAX_BLOCKS, 100_000)

    def test_yedda_block_re_counts_correctly(self) -> None:
        ann = _make_oversized_ann(10)
        self.assertEqual(len(_YEDDA_BLOCK_RE.findall(ann)), 10)


# ---------------------------------------------------------------------------
# _write_to_staging
# ---------------------------------------------------------------------------


class TestWriteToStaging(unittest.TestCase):

    def _make_staging_db(self, doc_exists: bool = False) -> MagicMock:
        db = MagicMock()
        if doc_exists:
            db.__contains__ = MagicMock(return_value=True)
            db.__getitem__ = MagicMock(
                return_value={"_id": "doc1", "_rev": "1-abc"}
            )
        else:
            db.__contains__ = MagicMock(return_value=False)
        return db

    def test_creates_new_doc_when_absent(self) -> None:
        db = self._make_staging_db(doc_exists=False)
        db.__getitem__ = MagicMock(
            return_value={"_id": "doc1", "_rev": "1-new"}
        )
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
        _write_to_staging(
            db, "skol_training", "doc1", _ANN_12TAG, [{"x": 1}]
        )
        db.put_attachment.assert_called_once()
        _, ann_bytes, *_ = db.put_attachment.call_args[0]
        self.assertIn(b"Type-designation", ann_bytes)

    def test_change_count_stored_on_new_doc(self) -> None:
        db = self._make_staging_db(doc_exists=False)
        db.__getitem__ = MagicMock(
            return_value={"_id": "doc1", "_rev": "1-new"}
        )
        changes = [{
            "block_index": 0, "old_tag": "Holotype",
            "new_tag": "Type-designation", "snippet": "x",
        }]
        _write_to_staging(db, "src", "doc1", _ANN_12TAG, changes)
        saved_doc = db.save.call_args[0][0]
        self.assertEqual(saved_doc["change_count"], 1)


# ---------------------------------------------------------------------------
# chunk_ann
# ---------------------------------------------------------------------------


class TestChunkAnn(unittest.TestCase):

    def test_small_doc_returns_single_chunk(self) -> None:
        chunks = chunk_ann(_ANN_8TAG, chunk_size=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], _ANN_8TAG)

    def test_chunk_size_one_returns_one_chunk_per_block(self) -> None:
        chunks = chunk_ann(_ANN_8TAG, chunk_size=1)
        n_blocks = len(_YEDDA_BLOCK_RE.findall(_ANN_8TAG))
        self.assertEqual(len(chunks), n_blocks)

    def test_chunks_reassemble_to_original(self) -> None:
        ann = _make_oversized_ann(20)
        chunks = chunk_ann(ann, chunk_size=7)
        reassembled = "\n\n".join(c.strip() for c in chunks) + "\n"
        # Block count must match
        orig_blocks = _YEDDA_BLOCK_RE.findall(ann)
        reassembled_blocks = _YEDDA_BLOCK_RE.findall(reassembled)
        self.assertEqual(len(orig_blocks), len(reassembled_blocks))

    def test_each_chunk_respects_size_limit(self) -> None:
        ann = _make_oversized_ann(30)
        chunk_size = 8
        chunks = chunk_ann(ann, chunk_size=chunk_size)
        for chunk in chunks:
            self.assertLessEqual(
                len(_YEDDA_BLOCK_RE.findall(chunk)), chunk_size
            )

    def test_default_chunk_size_constant_is_sane(self) -> None:
        self.assertGreater(_DEFAULT_CHUNK_SIZE, 10)
        self.assertLess(_DEFAULT_CHUNK_SIZE, 1000)

    def test_empty_ann_returns_single_empty_chunk(self) -> None:
        chunks = chunk_ann("", chunk_size=10)
        self.assertEqual(len(chunks), 1)

    def test_exact_chunk_size_boundary(self) -> None:
        # 6 blocks split at chunk_size=3 should give exactly 2 chunks
        ann = _make_oversized_ann(6)
        chunks = chunk_ann(ann, chunk_size=3)
        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(len(_YEDDA_BLOCK_RE.findall(chunk)), 3)


# ---------------------------------------------------------------------------
# relabel_ann_chunked
# ---------------------------------------------------------------------------


class TestRelabelAnnChunked(unittest.TestCase):

    def _make_client(self, response_text: str) -> MagicMock:
        client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text=response_text)]
        client.messages.create.return_value = msg
        return client

    def test_small_doc_returns_same_as_relabel_ann(self) -> None:
        client = self._make_client(_ANN_12TAG)
        new_text, changes = relabel_ann_chunked(
            client, _ANN_8TAG, "test-model", "doc1", chunk_size=100
        )
        self.assertIn("Type-designation", new_text)
        self.assertTrue(len(changes) > 0)

    def test_unchanged_returns_empty_diff(self) -> None:
        client = self._make_client(_ANN_8TAG)
        _, changes = relabel_ann_chunked(
            client, _ANN_8TAG, "test-model", "doc1", chunk_size=100
        )
        self.assertEqual(changes, [])

    def test_chunked_block_indices_are_global(self) -> None:
        """block_index in changes must be doc-global, not chunk-local."""
        # 6-block doc chunked at 3 → 2 API calls.
        # API swaps Description→Diagnosis in both chunks.
        chunk1_response = (
            "[@Block 0.#Diagnosis*]\n\n"
            "[@Block 1.#Diagnosis*]\n\n"
            "[@Block 2.#Diagnosis*]\n"
        )
        chunk2_response = (
            "[@Block 3.#Diagnosis*]\n\n"
            "[@Block 4.#Diagnosis*]\n\n"
            "[@Block 5.#Diagnosis*]\n"
        )

        def _make_msg(text: str) -> MagicMock:
            m = MagicMock()
            m.content = [MagicMock(text=text)]
            return m

        client = MagicMock()
        client.messages.create.side_effect = [
            _make_msg(chunk1_response),
            _make_msg(chunk2_response),
        ]
        ann = _make_oversized_ann(6)
        _, changes = relabel_ann_chunked(
            client, ann, "test-model", "doc1", chunk_size=3
        )
        indices = [c["block_index"] for c in changes]
        # Should span 0–5 (all 6 blocks changed), not 0–2 twice
        self.assertEqual(sorted(indices), list(range(6)))

    def test_large_doc_calls_api_multiple_times(self) -> None:
        ann = _make_oversized_ann(9)
        client = self._make_client(
            _make_oversized_ann(3)  # each chunk has 3 blocks
        )
        relabel_ann_chunked(
            client, ann, "test-model", "doc1", chunk_size=3
        )
        # 9 blocks / 3 per chunk = 3 API calls
        self.assertEqual(client.messages.create.call_count, 3)


if __name__ == "__main__":
    unittest.main()
