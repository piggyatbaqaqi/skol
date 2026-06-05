"""Tests for treatments_json_translator helper functions.

Currently covers ``build_llm_input_text`` — the helper that builds the
text fed to the JSON-translation LLM from a Treatment CouchDB doc.
Per docs/v3_buildout.md and the user directive, the LLM input must
include both Description AND Diagnosis so the annotator sees the
``feature->subfeature->value`` statements that Diagnosis typically
carries.

Run with: python -m pytest treatments_json_translator_test.py -v
"""

import math
import unittest

from treatments_json_translator import (
    _chunk_budget_chars,
    build_llm_input_text,
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


if __name__ == "__main__":
    unittest.main()
