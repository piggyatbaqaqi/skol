"""Tests for llm_merge_yedda — LLM-assisted YEDDA merge for hard documents.

These tests exercise the pure logic (conflict counting, prompt construction,
response parsing, block alignment) without making real API calls.  The Claude
client is replaced by a lightweight stub.
"""
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_merge_yedda import (  # type: ignore[import]
    build_merge_prompt,
    count_conflicts,
    merge_via_llm,
    parse_llm_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client_stub(response_text: str) -> Any:
    """Return a minimal Anthropic client stub that yields response_text."""
    content_block = MagicMock()
    content_block.text = response_text
    message = MagicMock()
    message.content = [content_block]
    client = MagicMock()
    client.messages.create.return_value = message
    return client


# ---------------------------------------------------------------------------
# count_conflicts
# ---------------------------------------------------------------------------


class TestCountConflicts:
    def test_no_conflicts(self) -> None:
        text = "[@some text#Description*]\n\n[@other#Nomenclature*]\n"
        assert count_conflicts(text) == 0

    def test_one_conflict(self) -> None:
        text = (
            "[@placed#Description*]\n\n"
            "<<<<<<< annotation\n"
            "[@old block#Nomenclature*]\n"
            "=======\n"
            "new block\n"
            ">>>>>>> new_ocr\n"
        )
        assert count_conflicts(text) == 1

    def test_three_conflicts(self) -> None:
        conflict = (
            "<<<<<<< annotation\n"
            "[@x#Notes*]\n"
            "=======\n"
            "y\n"
            ">>>>>>> new_ocr\n\n"
        )
        text = conflict * 3
        assert count_conflicts(text) == 3

    def test_empty_text(self) -> None:
        assert count_conflicts("") == 0


# ---------------------------------------------------------------------------
# build_merge_prompt
# ---------------------------------------------------------------------------


class TestBuildMergePrompt:
    def test_contains_reviewed_ann(self) -> None:
        reviewed = "[@taxon name#Nomenclature*]\n"
        new_text = "taxon name\n"
        prompt = build_merge_prompt(reviewed, new_text)
        assert reviewed.strip() in prompt

    def test_contains_new_text(self) -> None:
        reviewed = "[@taxon name#Nomenclature*]\n"
        new_text = "taxon name extra\n"
        prompt = build_merge_prompt(reviewed, new_text)
        assert new_text.strip() in prompt

    def test_contains_tag_definitions(self) -> None:
        prompt = build_merge_prompt("[@x#Description*]\n", "x\n")
        assert "Nomenclature" in prompt
        assert "Description" in prompt
        assert "Page-header" in prompt
        assert "Misc-exposition" in prompt

    def test_instructs_yedda_output(self) -> None:
        prompt = build_merge_prompt("[@x#Description*]\n", "x\n")
        assert "[@" in prompt or "YEDDA" in prompt

    def test_holotype_passthrough_mentioned(self) -> None:
        """Prompt should tell the model to preserve Holotype labels as-is."""
        prompt = build_merge_prompt("[@x#Holotype*]\n", "x\n")
        assert "Holotype" in prompt


# ---------------------------------------------------------------------------
# parse_llm_response
# ---------------------------------------------------------------------------


class TestParseLlmResponse:
    def test_clean_yedda_returned_as_is(self) -> None:
        response = "[@taxon#Nomenclature*]\n\n[@desc text#Description*]\n"
        result = parse_llm_response(response, reviewed_ann=response)
        assert "[@taxon#Nomenclature*]" in result
        assert "[@desc text#Description*]" in result

    def test_markdown_fence_stripped(self) -> None:
        response = "```\n[@taxon#Nomenclature*]\n```"
        result = parse_llm_response(
            response, reviewed_ann="[@taxon#Nomenclature*]\n"
        )
        assert "```" not in result
        assert "[@taxon#Nomenclature*]" in result

    def test_no_yedda_blocks_raises(self) -> None:
        with pytest.raises(ValueError, match="no YEDDA blocks"):
            parse_llm_response(
                "This is not YEDDA output.", reviewed_ann="[@x#Notes*]\n"
            )

    def test_block_count_mismatch_falls_back_to_lcs(self) -> None:
        """When model drops a block, LCS alignment preserves original tag."""
        reviewed = (
            "[@aaa#Nomenclature*]\n\n"
            "[@bbb#Description*]\n\n"
            "[@ccc#Notes*]\n"
        )
        # Model returns only 2 of 3 blocks.
        response = "[@aaa#Nomenclature*]\n\n[@ccc#Notes*]\n"
        result = parse_llm_response(response, reviewed_ann=reviewed)
        # All three original block texts must appear in output.
        assert "aaa" in result
        assert "bbb" in result
        assert "ccc" in result

    def test_extra_preamble_stripped(self) -> None:
        response = (
            "Here is the annotated text:\n\n"
            "[@taxon#Nomenclature*]\n"
        )
        result = parse_llm_response(
            response, reviewed_ann="[@taxon#Nomenclature*]\n"
        )
        assert "Here is the annotated text" not in result
        assert "[@taxon#Nomenclature*]" in result


# ---------------------------------------------------------------------------
# merge_via_llm — integration (stubbed client)
# ---------------------------------------------------------------------------


class TestMergeViaLlm:
    def test_clean_response_returned(self) -> None:
        reviewed = (
            "[@first block#Nomenclature*]\n\n"
            "[@second block#Description*]\n"
        )
        new_text = "first block\n\nsecond block\n"
        expected_response = (
            "[@first block#Nomenclature*]\n\n"
            "[@second block#Description*]\n"
        )
        client = _make_client_stub(expected_response)
        result = merge_via_llm(client, reviewed, new_text, doc_id="test_doc")
        assert "[@first block#Nomenclature*]" in result
        assert "[@second block#Description*]" in result

    def test_retries_on_empty_response(self) -> None:
        """merge_via_llm retries when model returns no YEDDA blocks."""
        reviewed = "[@text#Nomenclature*]\n"
        new_text = "text\n"
        good_response = "[@text#Nomenclature*]\n"

        content_bad = MagicMock()
        content_bad.text = "I cannot process this."
        content_good = MagicMock()
        content_good.text = good_response
        msg_bad = MagicMock()
        msg_bad.content = [content_bad]
        msg_good = MagicMock()
        msg_good.content = [content_good]

        client = MagicMock()
        client.messages.create.side_effect = [msg_bad, msg_good]
        result = merge_via_llm(
            client, reviewed, new_text, doc_id="retry_doc"
        )
        assert "[@text#Nomenclature*]" in result
        assert client.messages.create.call_count == 2

    def test_all_retries_exhausted_raises(self) -> None:
        reviewed = "[@text#Nomenclature*]\n"
        new_text = "text\n"
        client = _make_client_stub("not yedda")
        with pytest.raises(RuntimeError, match="attempts failed"):
            merge_via_llm(
                client, reviewed, new_text, doc_id="fail_doc"
            )

    def test_page_header_blocks_preserved(self) -> None:
        """Page-header blocks in reviewed_ann must appear in output."""
        reviewed = (
            "[@first#Nomenclature*]\n\n"
            "[@--- PDF Page 2 Label 2 ---#Page-header*]\n\n"
            "[@second#Description*]\n"
        )
        new_text = "first\n\n--- PDF Page 2 Label 2 ---\n\nsecond\n"
        expected = (
            "[@first#Nomenclature*]\n\n"
            "[@--- PDF Page 2 Label 2 ---#Page-header*]\n\n"
            "[@second#Description*]\n"
        )
        client = _make_client_stub(expected)
        result = merge_via_llm(
            client, reviewed, new_text, doc_id="page_doc"
        )
        assert "[@--- PDF Page 2 Label 2 ---#Page-header*]" in result


# ---------------------------------------------------------------------------
# _split_new_text and merge_via_llm_chunked
# ---------------------------------------------------------------------------


class TestSplitNewText:
    def test_single_chunk_returned_unchanged(self) -> None:
        from llm_merge_yedda import _split_new_text  # type: ignore[import]
        text = "para one\n\npara two\n\npara three\n"
        parts = _split_new_text(text, 1)
        assert len(parts) == 1
        assert parts[0].strip() == text.strip()

    def test_two_chunks_cover_all_text(self) -> None:
        from llm_merge_yedda import _split_new_text  # type: ignore[import]
        text = "aaa\n\nbbb\n\nccc\n\nddd\n"
        parts = _split_new_text(text, 2)
        assert len(parts) == 2
        combined = "\n\n".join(p.strip() for p in parts if p.strip())
        # All paragraphs must appear in the combined output.
        for para in ["aaa", "bbb", "ccc", "ddd"]:
            assert para in combined

    def test_no_paragraph_lost(self) -> None:
        from llm_merge_yedda import _split_new_text  # type: ignore[import]
        paras = [f"paragraph {i}" for i in range(20)]
        text = "\n\n".join(paras) + "\n"
        parts = _split_new_text(text, 4)
        combined = " ".join(parts)
        for para in paras:
            assert para in combined

    def test_more_chunks_than_paragraphs(self) -> None:
        """Asking for more chunks than paragraphs gives at most one per para."""
        from llm_merge_yedda import _split_new_text  # type: ignore[import]
        text = "only\n\ntwo\n"
        parts = _split_new_text(text, 10)
        assert all(p.strip() for p in parts)
        combined = " ".join(parts)
        assert "only" in combined and "two" in combined


class TestMergeViaLlmChunked:
    def test_single_chunk_delegates_to_merge_via_llm(self) -> None:
        from llm_merge_yedda import merge_via_llm_chunked  # type: ignore[import]
        reviewed = "[@aaa#Nomenclature*]\n\n[@bbb#Description*]\n"
        new_text = "aaa\n\nbbb\n"
        expected = "[@aaa#Nomenclature*]\n\n[@bbb#Description*]\n"
        client = _make_client_stub(expected)
        result = merge_via_llm_chunked(
            client, reviewed, new_text, doc_id="small_doc",
            chunk_size=100,
        )
        assert "[@aaa#Nomenclature*]" in result
        assert "[@bbb#Description*]" in result

    def test_multi_chunk_concatenates_results(self) -> None:
        from llm_merge_yedda import merge_via_llm_chunked  # type: ignore[import]
        # 6 blocks, chunk_size=2 → 3 chunks
        blocks = [f"[@block{i}#Description*]" for i in range(6)]
        reviewed = "\n\n".join(blocks) + "\n"
        new_text = "\n\n".join(f"block{i}" for i in range(6)) + "\n"

        # Stub returns corresponding 2-block chunk each call.
        call_count = 0
        responses = [
            "[@block0#Description*]\n\n[@block1#Description*]\n",
            "[@block2#Description*]\n\n[@block3#Description*]\n",
            "[@block4#Description*]\n\n[@block5#Description*]\n",
        ]

        content_mocks = []
        for r in responses:
            cm = MagicMock()
            cm.text = r
            msg = MagicMock()
            msg.content = [cm]
            content_mocks.append(msg)

        client = MagicMock()
        client.messages.create.side_effect = content_mocks

        result = merge_via_llm_chunked(
            client, reviewed, new_text, doc_id="large_doc",
            chunk_size=2,
        )
        for i in range(6):
            assert f"block{i}" in result
        assert client.messages.create.call_count == 3
