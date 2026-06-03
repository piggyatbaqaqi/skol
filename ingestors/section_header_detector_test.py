"""Tests for ingestors/section_header_detector.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestors.section_header_detector import (  # noqa: E402
    detect_section_headers,
)
from ingestors.spans import Span  # noqa: E402


def _labels(spans: List[Span]) -> List[str]:
    return [s.label for s in spans]


class TestDetectSectionHeaders(unittest.TestCase):
    """The detector finds whole-line section headers and emits one
    ``Span(label='section-header')`` per match.  Metadata carries the
    canonical (lower-cased) section name and a YEDDA-tag hint that
    downstream consumers (Step 2 feature assembler, Step 1.D
    annotate_v4.py) can use without re-parsing the text."""

    def test_canonical_introduction_matches(self):
        spans = detect_section_headers('Introduction\n')
        self.assertEqual(_labels(spans), ['section-header'])

    def test_all_caps_matches(self):
        """Older mycology journals frequently use ALL CAPS section
        headers."""
        spans = detect_section_headers('MATERIALS AND METHODS\n')
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].label, 'section-header')

    def test_trailing_colon_matches(self):
        spans = detect_section_headers('Introduction:\n')
        self.assertEqual(len(spans), 1)

    def test_trailing_em_dash_matches(self):
        """Journal-style section markers often use an em-dash."""
        spans = detect_section_headers('Introduction—\n')
        self.assertEqual(len(spans), 1)

    def test_numbered_prefix_matches(self):
        """Leading numbering (Arabic or Roman) shouldn't block the
        match."""
        digit = detect_section_headers('1. Introduction\n')
        roman = detect_section_headers('IV. Materials and methods\n')
        self.assertEqual(len(digit), 1)
        self.assertEqual(len(roman), 1)

    def test_inline_text_rejected(self):
        """A line that *mentions* a section name as part of body
        text must not match.  Line-anchoring + tight trailing-punct
        rules guard against this."""
        spans = detect_section_headers(
            'In the introduction to this paper, we discuss the topic.\n'
        )
        self.assertEqual(spans, [])

    def test_mixed_case_in_body_rejected(self):
        """``Introduction is the first chapter`` — there's text after
        the matcher, so it's body content, not a header."""
        spans = detect_section_headers(
            'Introduction is the first chapter\n'
        )
        self.assertEqual(spans, [])

    def test_metadata_carries_yedda_hint(self):
        spans = detect_section_headers('References\n')
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].metadata.get('yedda_hint'), 'Bibliography'
        )

    def test_metadata_canonical_lowercased(self):
        """Downstream consumers should be able to group case variants
        by the canonical lower-cased form."""
        spans = detect_section_headers('MATERIALS AND METHODS\n')
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].metadata.get('canonical'), 'materials and methods'
        )

    def test_multiple_sections_in_doc(self):
        """End-to-end behaviour over a small synthetic doc."""
        text = (
            'Abstract\n'
            'Some abstract text goes here.\n'
            '\n'
            'Introduction\n'
            'The body of the introduction.\n'
            '\n'
            'Materials and methods\n'
            'How we did things.\n'
            '\n'
            'References\n'
            'Some bibliography text.\n'
        )
        spans = detect_section_headers(text)
        canonicals = [s.metadata.get('canonical') for s in spans]
        self.assertEqual(
            canonicals,
            ['abstract', 'introduction',
             'materials and methods', 'references'],
        )

    def test_specimens_examined_synonym(self):
        """``Specimens examined`` is a synonym for ``Materials
        examined`` — the detector should map both to the
        ``Materials-examined`` YEDDA hint."""
        spans = detect_section_headers('Specimens examined:\n')
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].metadata.get('yedda_hint'), 'Materials-examined'
        )

    def test_empty_input_returns_empty_list(self):
        """Defensive: zero-length / whitespace input doesn't crash."""
        self.assertEqual(detect_section_headers(''), [])
        self.assertEqual(detect_section_headers('   \n'), [])

    def test_span_offsets_match_text(self):
        """Span start/end indices should slice back to the matched
        text (a contract every other Span producer satisfies)."""
        text = 'body line 1\nIntroduction\nbody line 2\n'
        spans = detect_section_headers(text)
        self.assertEqual(len(spans), 1)
        s = spans[0]
        self.assertEqual(text[s.start:s.end], s.text)


if __name__ == '__main__':
    unittest.main()
