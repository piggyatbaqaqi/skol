"""Tests for treatments_json_translator helper functions.

Currently covers ``build_llm_input_text`` — the helper that builds the
text fed to the JSON-translation LLM from a Treatment CouchDB doc.
Per docs/v3_buildout.md and the user directive, the LLM input must
include both Description AND Diagnosis so the annotator sees the
``feature->subfeature->value`` statements that Diagnosis typically
carries.

Run with: python -m pytest treatments_json_translator_test.py -v
"""

import unittest

from treatments_json_translator import build_llm_input_text


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


if __name__ == "__main__":
    unittest.main()
