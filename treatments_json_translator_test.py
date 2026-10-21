"""Tests for treatments_json_translator helper functions.

Covers ``build_llm_input_text`` — the helper that builds the
text fed to the JSON-translation LLM from a Treatment CouchDB doc.
Per docs/v3_buildout.md and the user directive, the LLM input must
include both Description AND Diagnosis so the annotator sees the
``feature->subfeature->value`` statements that Diagnosis typically
carries.

Also covers the description token threshold: ``make_token_counter``,
``partition_by_token_budget`` and
``TreatmentsJSONTranslator._drop_oversized_descriptions``.

Run with: python -m pytest treatments_json_translator_test.py -v
"""

import contextlib
import io
import math
import unittest
from typing import Any, Dict, List
from unittest import mock


from treatments_json_translator import (
    TreatmentsJSONTranslator,
    _chunk_budget_chars,
    build_llm_input_text,
    make_token_counter,
    partition_by_token_budget,
)


class TestBuildLlmInputText(unittest.TestCase):
    def test_description_only(self):
        doc = {"description": "Cap red.", "diagnosis": None}
        self.assertEqual(build_llm_input_text(doc), "Cap red.")

    def test_diagnosis_only(self):
        """Orphan-Diagnosis Treatments (synthetic Nomen ignotum stubs
        built from a bare Diagnosis paragraph) still get their content
        to the LLM."""
        doc = {"description": None, "diagnosis": "Differs from X by Y."}
        self.assertEqual(
            build_llm_input_text(doc),
            "Differs from X by Y.",
        )

    def test_both_present_concatenated_with_double_newline(self):
        doc = {
            "description": "Cap red.",
            "diagnosis": "Differs from X by Y.",
        }
        self.assertEqual(
            build_llm_input_text(doc),
            "Cap red.\n\nDiffers from X by Y.",
        )

    def test_both_empty_returns_empty_string(self):
        """Treatments with neither section yield an empty string —
        the LLM annotator is expected to short-circuit on empty input
        rather than the helper itself filtering them out."""
        self.assertEqual(build_llm_input_text({}), "")
        self.assertEqual(
            build_llm_input_text({"description": "", "diagnosis": ""}),
            "",
        )
        self.assertEqual(
            build_llm_input_text({"description": None, "diagnosis": None}),
            "",
        )

    def test_missing_keys_treated_as_absent(self):
        """A doc dict missing the keys entirely behaves the same as
        having them set to None."""
        self.assertEqual(
            build_llm_input_text({"description": "X"}),
            "X",
        )
        self.assertEqual(
            build_llm_input_text({"diagnosis": "Y"}),
            "Y",
        )

    def test_diagnosis_only_no_leading_separator(self):
        """When only Diagnosis is present, the LLM input must not
        start with a stray ``\\n\\n`` separator."""
        out = build_llm_input_text(
            {"description": None, "diagnosis": "D."},
        )
        self.assertFalse(out.startswith("\n"))


class TestChunkBudget(unittest.TestCase):
    """Contract for ``_chunk_budget_chars``.

    A chunk's prompt is ``scaffold + content``. The budget reserves
    ``SCAFFOLD_RESERVE_TOKENS`` for the scaffold and spends the rest on
    content, converting tokens->chars by an implicit chars/token factor.
    For the tokenizer not to exceed its configured ``model_max_length``
    (which triggers the HF "Token indices sequence length is longer than
    the specified maximum" warning), the *worst-case* token count of a
    full-budget chunk plus the scaffold reserve must stay within
    ``max_length``. That holds iff the implicit factor does not exceed
    the corpus's worst-case character density.
    """

    # Token-dense taxonomic text (unicode measurements like "10-15 x 3-4 um",
    # μm, ×, digits, scientific names) packs ~3.0 chars/token in the Mistral
    # tokenizer. The budget's implicit factor must not exceed this floor.
    WORST_CASE_CHARS_PER_TOKEN = 3.0
    # Mirrors the reserve documented in _chunk_budget_chars.
    SCAFFOLD_RESERVE_TOKENS = 768

    def test_budget_stays_within_token_limit(self):
        for max_length in (2048, 4096, 8192):
            with self.subTest(max_length=max_length):
                budget = _chunk_budget_chars(max_length)
                worst_case_content_tokens = math.ceil(
                    budget / self.WORST_CASE_CHARS_PER_TOKEN
                )
                total = worst_case_content_tokens + self.SCAFFOLD_RESERVE_TOKENS
                self.assertLessEqual(
                    total,
                    max_length,
                    f"max_length={max_length}: a full-budget chunk "
                    f"({budget} chars) can reach {worst_case_content_tokens} "
                    f"content tokens; with the {self.SCAFFOLD_RESERVE_TOKENS}-"
                    f"token scaffold that is {total} tokens, exceeding "
                    f"{max_length} and tripping the tokenizer overflow warning.",
                )

    def test_budget_honors_minimum_floor(self):
        """When max_length is small enough that the computed budget would
        fall below 1000 chars, the floor keeps it at 1000."""
        self.assertEqual(_chunk_budget_chars(900), 1000)


class FakeTokenizer:
    """Stands in for a Hugging Face tokenizer: two tokens per word.

    Not one, so an implementation that counts words instead of
    tokens gets different numbers and fails.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, texts: List[str], add_special_tokens: bool = True
                 ) -> Dict[str, List[List[int]]]:
        self.calls.append(
            {"texts": list(texts), "add_special_tokens": add_special_tokens})
        return {"input_ids": [[0] * (2 * len(t.split())) for t in texts]}


def word_counter(texts: List[str]) -> List[int]:
    return [len(t.split()) for t in texts]


def refuse_to_count(texts: List[str]) -> List[int]:
    raise AssertionError("counted tokens although no threshold is set")


def item(doc_id: str, description: Any) -> Dict[str, Any]:
    return {"_id": doc_id, "treatment": f"T {doc_id}",
            "description": description}


class TestMakeTokenCounter(unittest.TestCase):

    def test_counts_each_text_in_order(self):
        count = make_token_counter(FakeTokenizer())
        self.assertEqual(count(["a b c", "d", "e f"]), [6, 2, 4])

    def test_excludes_special_tokens(self):
        """The threshold is about the description's own length; BOS/EOS
        belong to the prompt scaffold."""
        tokenizer = FakeTokenizer()
        make_token_counter(tokenizer)(["a b"])
        self.assertFalse(tokenizer.calls[0]["add_special_tokens"])

    def test_tokenizes_the_whole_list_in_one_call(self):
        """Batching is what keeps 80k descriptions at seconds, not minutes."""
        tokenizer = FakeTokenizer()
        make_token_counter(tokenizer)(["a", "b c", "d e f"])
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertEqual(tokenizer.calls[0]["texts"], ["a", "b c", "d e f"])

    def test_empty_list(self):
        self.assertEqual(make_token_counter(FakeTokenizer())([]), [])


class TestPartitionByTokenBudget(unittest.TestCase):

    def test_none_threshold_keeps_everything_without_counting(self):
        items = [item("a", "one two"), item("b", "x " * 500)]
        kept, dropped = partition_by_token_budget(items, refuse_to_count, None)
        self.assertEqual(kept, items)
        self.assertEqual(dropped, [])

    def test_threshold_is_inclusive(self):
        """A description of exactly max_tokens is kept; one more is dropped."""
        items = [item("at", "a b c"), item("over", "a b c d")]
        kept, dropped = partition_by_token_budget(items, word_counter, 3)
        self.assertEqual([i["_id"] for i in kept], ["at"])
        self.assertEqual(dropped, [("over", 4)])

    def test_kept_items_keep_their_order_and_fields(self):
        items = [item("a", "x"), item("big", "x x x x x"), item("b", "x x"),
                 item("c", "x")]
        items[0]["page_number"] = 7
        kept, _ = partition_by_token_budget(items, word_counter, 2)
        self.assertEqual([i["_id"] for i in kept], ["a", "b", "c"])
        self.assertIs(kept[0], items[0])

    def test_dropped_reports_every_id_with_its_count(self):
        items = [item("a", "x " * 10), item("b", "x"), item("c", "x " * 7)]
        _, dropped = partition_by_token_budget(items, word_counter, 5)
        self.assertEqual(dropped, [("a", 10), ("c", 7)])

    def test_missing_description_counts_as_empty_and_is_kept(self):
        seen: List[List[str]] = []

        def counter(texts: List[str]) -> List[int]:
            seen.append(list(texts))
            return word_counter(texts)

        items = [item("none", None), item("empty", "")]
        kept, dropped = partition_by_token_budget(items, counter, 1)
        self.assertEqual(seen, [["", ""]])
        self.assertEqual(kept, items)
        self.assertEqual(dropped, [])

    def test_counts_all_descriptions_in_one_call(self):
        calls: List[List[str]] = []

        def counter(texts: List[str]) -> List[int]:
            calls.append(list(texts))
            return word_counter(texts)

        partition_by_token_budget(
            [item("a", "x"), item("b", "y z")], counter, 5)
        self.assertEqual(calls, [["x", "y z"]])

    def test_empty_input(self):
        self.assertEqual(
            partition_by_token_budget([], word_counter, 5), ([], []))

    def test_non_positive_threshold_is_rejected(self):
        for bad in (0, -1):
            with self.subTest(max_tokens=bad):
                with self.assertRaises(ValueError):
                    partition_by_token_budget(
                        [item("a", "x")], word_counter, bad)


def make_translator(**kwargs: Any) -> TreatmentsJSONTranslator:
    with contextlib.redirect_stdout(io.StringIO()):
        return TreatmentsJSONTranslator(
            spark=mock.Mock(), couchdb_url="http://localhost:5984",
            base_model_id="example/model", **kwargs)


class TestDropOversizedDescriptions(unittest.TestCase):

    def run_drop(self, translator: TreatmentsJSONTranslator,
                 items: List[Dict[str, Any]]):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            kept = translator._drop_oversized_descriptions(items)
        return kept, out.getvalue()

    def test_no_threshold_returns_input_and_loads_no_tokenizer(self):
        translator = make_translator(token_counter=refuse_to_count)
        items = [item("a", "x " * 100000)]
        with mock.patch("treatments_json_translator.AutoTokenizer") as tok:
            kept, _ = self.run_drop(translator, items)
        self.assertEqual(kept, items)
        tok.from_pretrained.assert_not_called()

    def test_drops_only_oversized(self):
        translator = make_translator(
            max_description_tokens=3, token_counter=word_counter)
        items = [item("small", "a b"), item("big", "a b c d e"),
                 item("edge", "a b c")]
        kept, _ = self.run_drop(translator, items)
        self.assertEqual([i["_id"] for i in kept], ["small", "edge"])

    def test_reports_each_dropped_record_and_a_summary(self):
        translator = make_translator(
            max_description_tokens=3, token_counter=word_counter)
        items = [item("small", "a"), item("taxon_big", "a b c d e f g")]
        _, log = self.run_drop(translator, items)
        self.assertIn("taxon_big", log)
        self.assertIn("7", log)
        self.assertRegex(log, r"[Dd]ropp?ed 1 of 2\b")
        self.assertIn("3", log)

    def test_silent_when_nothing_is_dropped(self):
        translator = make_translator(
            max_description_tokens=10, token_counter=word_counter)
        _, log = self.run_drop(translator, [item("a", "x")])
        self.assertNotRegex(log, r"[Dd]ropp?ed")

    def test_builds_counter_from_base_model_tokenizer_once(self):
        """Counts must come from the configured model's tokenizer:
        switching --base-model changes what a token is.  "x y" is two
        words but four FakeTokenizer tokens, so counting words would keep
        it under the 3-token threshold."""
        translator = make_translator(max_description_tokens=3)
        with mock.patch("treatments_json_translator.AutoTokenizer") as tok:
            tok.from_pretrained.return_value = FakeTokenizer()
            first, _ = self.run_drop(translator, [item("a", "x y")])
            second, _ = self.run_drop(translator, [item("b", "x")])
        self.assertEqual(first, [])
        self.assertEqual([i["_id"] for i in second], ["b"])
        tok.from_pretrained.assert_called_once()
        self.assertEqual(
            tok.from_pretrained.call_args.args[0], "example/model")


if __name__ == "__main__":
    unittest.main()
