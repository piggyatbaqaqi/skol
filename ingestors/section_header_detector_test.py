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


class TestTaxonomicSectionStarts(unittest.TestCase):
    """User-supplied real-world taxonomic-section start phrases.  All
    hint Nomenclature: treatments downstream of these headers
    typically begin with a nomenclatural act."""

    def _canonicals(self, text: str):
        return [
            s.metadata.get('canonical')
            for s in detect_section_headers(text)
        ]

    def test_taxonomic_description(self):
        self.assertEqual(
            self._canonicals('Taxonomic description\n'),
            ['taxonomic description'],
        )

    def test_taxonomic_revision(self):
        self.assertEqual(
            self._canonicals('Taxonomic revision\n'),
            ['taxonomic revision'],
        )

    def test_taxonomic_part(self):
        self.assertEqual(
            self._canonicals('Taxonomic part\n'),
            ['taxonomic part'],
        )

    def test_the_species(self):
        self.assertEqual(
            self._canonicals('The species\n'),
            ['the species'],
        )

    def test_descriptive_part_all_caps(self):
        self.assertEqual(
            self._canonicals('DESCRIPTIVE PART\n'),
            ['descriptive part'],
        )

    def test_descriptions_of_the_species_all_caps(self):
        self.assertEqual(
            self._canonicals('DESCRIPTIONS OF THE SPECIES\n'),
            ['descriptions of the species'],
        )

    def test_taxa_studied(self):
        self.assertEqual(
            self._canonicals('Taxa studied\n'),
            ['taxa studied'],
        )

    def test_species_recorded(self):
        self.assertEqual(
            self._canonicals('Species recorded\n'),
            ['species recorded'],
        )

    def test_nomenclator_and_taxonomic_description(self):
        """Real-world all-caps header from older systematic
        treatments."""
        self.assertEqual(
            self._canonicals('NOMENCLATOR AND TAXONOMIC DESCRIPTION\n'),
            ['nomenclator and taxonomic description'],
        )

    def test_all_hint_nomenclature(self):
        """All taxonomic-section starts emit yedda_hint=Nomenclature
        so the layout CRF learns a uniform downstream signal."""
        text = (
            'Taxonomic description\n'
            'The species\n'
            'DESCRIPTIVE PART\n'
            'Taxa studied\n'
        )
        hints = {
            s.metadata.get('yedda_hint')
            for s in detect_section_headers(text)
        }
        self.assertEqual(hints, {'Nomenclature'})


class TestNewSpeciesWildcard(unittest.TestCase):
    """``A new species of <binomial>`` — variable trailing text after
    the matcher, exercised via the full-regex override path."""

    def test_a_new_species_of_boletus_uppercase(self):
        """Real-world all-caps header style."""
        spans = detect_section_headers(
            'A NEW SPECIES OF BOLETUS\n'
        )
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].metadata.get('yedda_hint'),
                         'Nomenclature')

    def test_a_new_species_of_boletus_titlecase(self):
        spans = detect_section_headers(
            'A new species of Boletus\n'
        )
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].metadata.get('yedda_hint'),
                         'Nomenclature')

    def test_a_new_species_alone_still_matches(self):
        """No trailing ``of X`` — header is just ``A new species``."""
        spans = detect_section_headers('A new species\n')
        self.assertEqual(len(spans), 1)

    def test_a_new_species_of_multi_word_genus(self):
        spans = detect_section_headers(
            'A new species of Cintractiella from Micronesia\n'
        )
        self.assertEqual(len(spans), 1)

    def test_inline_a_new_species_rejected(self):
        """Body-text mention shouldn't match (no leading anchor on
        the wildcard's full regex either)."""
        spans = detect_section_headers(
            'In this paper we describe a new species of Boletus '
            'that was found in Patagonia.\n'
        )
        self.assertEqual(spans, [])


class TestNotesBlockHint(unittest.TestCase):
    """User rule: ``Discussion`` / ``Remarks`` / ``Comments`` /
    ``Notes`` all default to Notes.  The Diagnosis disambiguation
    is left to the downstream layout / treatment CRFs that see the
    section body."""

    def _hint_of(self, text: str):
        spans = detect_section_headers(text)
        return spans[0].metadata.get('yedda_hint') if spans else None

    def test_discussion_now_hints_notes(self):
        """Previously Misc-exposition; flipped per user choice."""
        self.assertEqual(self._hint_of('Discussion\n'), 'Notes')

    def test_remarks_hints_notes(self):
        self.assertEqual(self._hint_of('Remarks\n'), 'Notes')

    def test_comments_hints_notes(self):
        self.assertEqual(self._hint_of('Comments\n'), 'Notes')

    def test_notes_hints_notes(self):
        self.assertEqual(self._hint_of('Notes\n'), 'Notes')


if __name__ == '__main__':
    unittest.main()
