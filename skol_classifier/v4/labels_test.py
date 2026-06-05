"""Tests for skol_classifier/v4/labels.py — Pass-1 label projection
and YEDDA-block → line-index alignment."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from skol_classifier.v4.crf_layout import (  # noqa: E402
    LABEL_TO_INDEX,
    OTHER_INDEX,
)
from skol_classifier.v4.labels import (  # noqa: E402
    ACTIVE_YEDDA_TAGS,
    LAYOUT_YEDDA_TAGS,
    TREATMENT_YEDDA_TAGS,
    build_active_label_sequence,
    build_label_sequence,
    build_treatment_label_sequence,
    map_yedda_to_active,
    map_yedda_to_layout,
    map_yedda_to_treatment,
    yedda_blocks_to_line_indices,
    yedda_tag_per_line,
)


# ---------------------------------------------------------------------------
# 1. map_yedda_to_layout
# ---------------------------------------------------------------------------


class TestMapYeddaToLayout(unittest.TestCase):
    """Project the 19 ACTIVE_TAGS_19 down to the 8 Pass-1 labels."""

    def test_layout_tag_passthrough(self):
        for tag in LAYOUT_YEDDA_TAGS:
            self.assertEqual(map_yedda_to_layout(tag), tag)

    def test_unknown_tag_maps_to_other(self):
        for tag in (
            'Nomenclature', 'Description', 'Diagnosis',
            'Etymology', 'Materials-examined', 'Notes',
            'Phylogeny', 'Misc-exposition',
        ):
            self.assertEqual(map_yedda_to_layout(tag), 'Other')

    def test_empty_string_maps_to_other(self):
        self.assertEqual(map_yedda_to_layout(''), 'Other')

    def test_case_insensitive_passthrough(self):
        """``page-header`` should match ``Page-header`` — robust
        against case drift in YEDDA files."""
        self.assertEqual(
            map_yedda_to_layout('page-header'), 'Page-header',
        )
        self.assertEqual(
            map_yedda_to_layout('BIBLIOGRAPHY'), 'Bibliography',
        )


# ---------------------------------------------------------------------------
# 2. yedda_blocks_to_line_indices
# ---------------------------------------------------------------------------


class TestYeddaBlocksToLineIndices(unittest.TestCase):
    """char-offset → line-index conversion using the same
    plaintext.count('\\n', 0, offset) convention used elsewhere
    in v4."""

    def test_single_block_single_line(self):
        plaintext = 'Page header text\nbody line 0\nbody line 1\n'
        blocks = [('Page-header', 0, 16)]   # covers "Page header text"
        result = yedda_blocks_to_line_indices(plaintext, blocks)
        self.assertEqual(result, [('Page-header', [0])])

    def test_block_spans_two_lines(self):
        plaintext = 'line zero\nline one\nline two\n'
        # Cover from middle of line 0 to middle of line 1.
        blocks = [('Page-header', 5, 13)]
        result = yedda_blocks_to_line_indices(plaintext, blocks)
        self.assertEqual(result, [('Page-header', [0, 1])])

    def test_multiple_blocks_doc_order(self):
        plaintext = (
            'Top page header\n'        # line 0
            'body\n'                   # line 1
            'Footer page-num 5\n'      # line 2
            'body\n'                   # line 3
            'Top page header 2\n'      # line 4
        )
        blocks = [
            ('Page-header', 0, 15),
            ('Page-header', 21, 38),
            ('Page-header', 43, 60),
        ]
        result = yedda_blocks_to_line_indices(plaintext, blocks)
        labels = [r[0] for r in result]
        self.assertEqual(labels, ['Page-header'] * 3)
        # All three line indices appear, in order.
        all_lines = [li for _, li_list in result for li in li_list]
        self.assertEqual(sorted(all_lines), all_lines)

    def test_block_past_eof_clamps(self):
        """Defensive: end_offset > len(text) shouldn't crash."""
        plaintext = 'short\n'
        blocks = [('Page-header', 0, 100)]
        result = yedda_blocks_to_line_indices(plaintext, blocks)
        # The block should still be returned, mapped to all available
        # line indices.
        self.assertEqual(result, [('Page-header', [0])])

    def test_empty_blocks_returns_empty_list(self):
        self.assertEqual(
            yedda_blocks_to_line_indices('x\n', []), [],
        )


# ---------------------------------------------------------------------------
# 3. build_label_sequence
# ---------------------------------------------------------------------------


class TestBuildLabelSequence(unittest.TestCase):
    """End-to-end per-doc label sequence: ann_text + plaintext → per-line
    Pass-1 label indices."""

    def _ann(self, *blocks: str) -> str:
        return ''.join(blocks)

    def test_layout_block_yields_correct_indices(self):
        """A Page-header YEDDA block over the first two lines
        produces Page-header indices at those positions; other
        lines default to Other."""
        plaintext = 'Page top\nVol 5 (2)\nBoletus edulis\nIntroduction text\n'
        ann_text = self._ann(
            '[@Page top\nVol 5 (2)#Page-header*]',
            '[@Boletus edulis#Nomenclature*]',
            '[@Introduction text#Misc-exposition*]',
        )
        seq = build_label_sequence(plaintext, ann_text)
        # First two lines should be Page-header; rest Other.
        self.assertEqual(seq[0], LABEL_TO_INDEX['Page-header'])
        self.assertEqual(seq[1], LABEL_TO_INDEX['Page-header'])
        self.assertEqual(seq[2], LABEL_TO_INDEX['Other'])
        self.assertEqual(seq[3], LABEL_TO_INDEX['Other'])

    def test_non_layout_tags_default_to_other(self):
        plaintext = 'Boletus edulis Bull.\nA fine mushroom.\n'
        ann_text = self._ann(
            '[@Boletus edulis Bull.#Nomenclature*]',
            '[@A fine mushroom.#Description*]',
        )
        seq = build_label_sequence(plaintext, ann_text)
        for label_idx in seq[:2]:
            self.assertEqual(label_idx, OTHER_INDEX)

    def test_lines_outside_any_block_default_to_other(self):
        """Whitespace-only gap lines between YEDDA blocks default to
        Other (the projection from parse_yedda_sections leaves
        between-block gap lines uncovered)."""
        plaintext = 'Page-header\n\nNomenclature line\n'
        ann_text = self._ann(
            '[@Page-header#Page-header*]',
            '[@Nomenclature line#Nomenclature*]',
        )
        seq = build_label_sequence(plaintext, ann_text)
        # 4 elements: the trailing \n yields an empty 4th line.
        self.assertEqual(len(seq), 4)
        self.assertEqual(seq[0], LABEL_TO_INDEX['Page-header'])
        # The gap line between blocks is Other.
        self.assertEqual(seq[1], OTHER_INDEX)
        # Nomenclature folds to Other.
        self.assertEqual(seq[2], OTHER_INDEX)

    def test_length_matches_plaintext_lines(self):
        plaintext = 'a\nb\nc\nd\ne\n'
        ann_text = '[@a\nb\nc#Page-header*][@d\ne#Bibliography*]'
        seq = build_label_sequence(plaintext, ann_text)
        # Plaintext has 6 lines (the trailing \n produces an empty
        # 6th line after split).  Sequence length must match.
        self.assertEqual(len(seq), len(plaintext.split('\n')))

    def test_empty_inputs(self):
        self.assertEqual(build_label_sequence('', ''), [])


# ---------------------------------------------------------------------------
# 4. Parity
# ---------------------------------------------------------------------------


class TestParity(unittest.TestCase):
    """Same input -> identical output (no nondeterminism)."""

    def test_idempotent_same_input(self):
        plaintext = 'header line\nbody line\nfooter line\n'
        ann_text = (
            '[@header line#Page-header*]'
            '[@body line#Description*]'
            '[@footer line#Page-header*]'
        )
        a = build_label_sequence(plaintext, ann_text)
        b = build_label_sequence(plaintext, ann_text)
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# 5. map_yedda_to_treatment
# ---------------------------------------------------------------------------


class TestMapYeddaToTreatment(unittest.TestCase):
    """Project YEDDA tags down to the 12 Pass-2 treatment labels.
    Pass-1 layout tags collapse to ``Misc-exposition`` (the
    catch-all) since Pass 2 trains only on non-layout lines, but
    defensive handling for any layout tags that slip through is
    still useful."""

    def test_treatment_tag_passthrough(self):
        for tag in TREATMENT_YEDDA_TAGS:
            self.assertEqual(map_yedda_to_treatment(tag), tag)

    def test_layout_tag_maps_to_misc_exposition(self):
        for tag in LAYOUT_YEDDA_TAGS:
            self.assertEqual(
                map_yedda_to_treatment(tag), 'Misc-exposition',
            )

    def test_unknown_tag_maps_to_misc_exposition(self):
        self.assertEqual(map_yedda_to_treatment(''), 'Misc-exposition')
        self.assertEqual(
            map_yedda_to_treatment('SomeNewTag'), 'Misc-exposition',
        )

    def test_case_insensitive_passthrough(self):
        self.assertEqual(
            map_yedda_to_treatment('description'), 'Description',
        )
        self.assertEqual(
            map_yedda_to_treatment('NOMENCLATURE'), 'Nomenclature',
        )


# ---------------------------------------------------------------------------
# 6. yedda_tag_per_line
# ---------------------------------------------------------------------------


class TestYeddaTagPerLine(unittest.TestCase):
    """Return the raw YEDDA tag for each line — lines outside any
    block default to ``Misc-exposition``."""

    def test_block_lines_carry_block_tag(self):
        plaintext = 'Boletus edulis Bull.\nA fine mushroom.\n'
        ann_text = (
            '[@Boletus edulis Bull.#Nomenclature*]'
            '[@A fine mushroom.#Description*]'
        )
        tags = yedda_tag_per_line(plaintext, ann_text)
        self.assertEqual(tags[0], 'Nomenclature')
        self.assertEqual(tags[1], 'Description')

    def test_gap_lines_default_misc_exposition(self):
        """A blank line between two YEDDA blocks gets
        ``Misc-exposition`` — same default the catch-all gives."""
        plaintext = 'Description line\n\nNotes line\n'
        ann_text = (
            '[@Description line#Description*]'
            '[@Notes line#Notes*]'
        )
        tags = yedda_tag_per_line(plaintext, ann_text)
        self.assertEqual(tags[0], 'Description')
        self.assertEqual(tags[1], 'Misc-exposition')  # gap
        self.assertEqual(tags[2], 'Notes')


# ---------------------------------------------------------------------------
# 7. build_treatment_label_sequence
# ---------------------------------------------------------------------------


class TestBuildTreatmentLabelSequence(unittest.TestCase):
    """Per-line Pass-2 treatment label indices."""

    def test_length_matches_plaintext_lines(self):
        plaintext = 'a\nb\nc\n'
        ann_text = '[@a#Nomenclature*][@b#Description*][@c#Notes*]'
        seq = build_treatment_label_sequence(plaintext, ann_text)
        self.assertEqual(len(seq), len(plaintext.split('\n')))

    def test_treatment_tags_yield_correct_indices(self):
        from skol_classifier.v4.crf_treatment import LABEL_TO_INDEX
        plaintext = 'a\nb\n'
        ann_text = '[@a#Nomenclature*][@b#Diagnosis*]'
        seq = build_treatment_label_sequence(plaintext, ann_text)
        self.assertEqual(seq[0], LABEL_TO_INDEX['Nomenclature'])
        self.assertEqual(seq[1], LABEL_TO_INDEX['Diagnosis'])

    def test_layout_tags_collapse_to_misc_exposition(self):
        """A Page-header YEDDA block in the source ann_text should
        be folded to Misc-exposition (we only get here for lines
        Pass-1 says are non-layout, but the helper is defensive)."""
        from skol_classifier.v4.crf_treatment import LABEL_TO_INDEX
        plaintext = 'a\nb\n'
        ann_text = '[@a#Page-header*][@b#Description*]'
        seq = build_treatment_label_sequence(plaintext, ann_text)
        self.assertEqual(
            seq[0], LABEL_TO_INDEX['Misc-exposition'],
        )
        self.assertEqual(
            seq[1], LABEL_TO_INDEX['Description'],
        )

    def test_empty_inputs(self):
        self.assertEqual(build_treatment_label_sequence('', ''), [])


# ---------------------------------------------------------------------------
# 8. map_yedda_to_active (v4 Step 6.F — single-CRF baseline)
# ---------------------------------------------------------------------------


class TestMapYeddaToActive(unittest.TestCase):
    """Project a YEDDA tag to the 19 ACTIVE_TAGS_19 label space.
    Unknown tags collapse to 'Misc-exposition' — same catch-all
    convention Pass-2 uses."""

    def test_all_19_tags_pass_through(self):
        """Every member of ACTIVE_YEDDA_TAGS round-trips through
        the projector unchanged."""
        for tag in ACTIVE_YEDDA_TAGS:
            self.assertEqual(map_yedda_to_active(tag), tag)

    def test_unknown_tag_maps_to_misc_exposition(self):
        self.assertEqual(
            map_yedda_to_active('NotARealTag'), 'Misc-exposition',
        )
        self.assertEqual(map_yedda_to_active(''), 'Misc-exposition')

    def test_case_insensitive_passthrough(self):
        self.assertEqual(
            map_yedda_to_active('page-header'), 'Page-header',
        )
        self.assertEqual(
            map_yedda_to_active('NOMENCLATURE'), 'Nomenclature',
        )

    def test_deprecated_tags_fold_to_misc(self):
        """Holotype / Distribution / FIX (the DEPRECATED_TAGS set)
        are NOT in ACTIVE_TAGS_19; they fold to Misc-exposition."""
        self.assertEqual(map_yedda_to_active('Holotype'), 'Misc-exposition')
        self.assertEqual(map_yedda_to_active('FIX'), 'Misc-exposition')


# ---------------------------------------------------------------------------
# 9. build_active_label_sequence
# ---------------------------------------------------------------------------


class TestBuildActiveLabelSequence(unittest.TestCase):
    """Per-line 19-label index sequence used by Step 6.F's
    single-CRF baseline trainer."""

    def test_length_matches_plaintext_lines(self):
        plaintext = 'a\nb\nc'
        seq = build_active_label_sequence(plaintext, '')
        self.assertEqual(len(seq), 3)

    def test_block_lines_carry_block_tag_index(self):
        """A tagged line gets the index of its YEDDA tag in the
        ACTIVE_TAGS_19 ordering."""
        from skol_classifier.v4.crf_single import LABEL_TO_INDEX
        plaintext = 'header\nbody\nfooter'
        ann_text = (
            '[@header#Page-header*]'
            '[@body#Description*]'
            '[@footer#Bibliography*]'
        )
        seq = build_active_label_sequence(plaintext, ann_text)
        self.assertEqual(seq[0], LABEL_TO_INDEX['Page-header'])
        self.assertEqual(seq[1], LABEL_TO_INDEX['Description'])
        self.assertEqual(seq[2], LABEL_TO_INDEX['Bibliography'])

    def test_gap_lines_default_misc_exposition(self):
        from skol_classifier.v4.crf_single import LABEL_TO_INDEX
        plaintext = 'tagged\nuntagged\ntagged-too'
        ann_text = (
            '[@tagged#Description*]'
            '[@tagged-too#Notes*]'
        )
        seq = build_active_label_sequence(plaintext, ann_text)
        self.assertEqual(seq[0], LABEL_TO_INDEX['Description'])
        self.assertEqual(seq[1], LABEL_TO_INDEX['Misc-exposition'])
        self.assertEqual(seq[2], LABEL_TO_INDEX['Notes'])

    def test_empty_inputs(self):
        self.assertEqual(build_active_label_sequence('', ''), [])


# ---------------------------------------------------------------------------
# 10. ACTIVE_YEDDA_TAGS shape
# ---------------------------------------------------------------------------


class TestActiveYeddaTags(unittest.TestCase):

    def test_19_members(self):
        self.assertEqual(len(ACTIVE_YEDDA_TAGS), 19)

    def test_superset_of_layout_and_treatment(self):
        s = set(ACTIVE_YEDDA_TAGS)
        for tag in LAYOUT_YEDDA_TAGS:
            self.assertIn(tag, s)
        for tag in TREATMENT_YEDDA_TAGS:
            self.assertIn(tag, s)


if __name__ == '__main__':
    unittest.main()
