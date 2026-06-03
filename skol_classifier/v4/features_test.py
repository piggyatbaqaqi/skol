"""Tests for skol_classifier/v4/features.py — line-feature assembler."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ingestors.spans import Span  # noqa: E402

from skol_classifier.v4.features import (  # noqa: E402
    PARTICLE_VOCAB,
    LineFeatures,
    build_line_features,
    layout_features,
    page_header_score,
    particle_counts,
    section_header_flag,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _doc_with_lines(*lines: str) -> str:
    """Build a plaintext doc from individual line strings (no trailing
    newline on the last line — caller adds one if needed)."""
    return '\n'.join(lines)


def _stub_sbert_zero(_line: str) -> Optional[np.ndarray]:
    """sbert_lookup that always returns a zero vector (cache hit)."""
    return np.zeros(768, dtype=np.float32)


def _stub_sbert_miss(_line: str) -> Optional[np.ndarray]:
    """sbert_lookup that always returns None (cache miss)."""
    return None


def _stub_sbert_const(value: float):
    """sbert_lookup that returns a constant-valued vector."""
    def _lookup(_line: str) -> Optional[np.ndarray]:
        return np.full(768, value, dtype=np.float32)
    return _lookup


# ---------------------------------------------------------------------------
# 1. Layout features
# ---------------------------------------------------------------------------


class TestLayoutFeatures(unittest.TestCase):
    """Layout block (8 floats): length, indent_pct, allcaps_pct,
    digit_pct, trailing_digit_flag, is_short, blank_before,
    blank_after."""

    def test_blank_line_layout(self):
        lines = ['some body', '', 'more body']
        v = layout_features('', 1, lines)
        self.assertEqual(v.shape, (8,))
        self.assertEqual(v.dtype, np.float32)
        self.assertEqual(v[0], 0.0)        # length
        self.assertEqual(v[1], 0.0)        # indent_pct
        self.assertEqual(v[2], 0.0)        # allcaps_pct
        self.assertEqual(v[3], 0.0)        # digit_pct
        self.assertEqual(v[4], 0.0)        # trailing_digit_flag
        self.assertEqual(v[5], 1.0)        # is_short (empty is short)
        # blank_before/after depend on neighbours.

    def test_indent_pct(self):
        lines = ['  hi']
        v = layout_features('  hi', 0, lines)
        self.assertAlmostEqual(float(v[1]), 0.5, places=5)

    def test_allcaps_pct(self):
        lines = ['AaBb']
        v = layout_features('AaBb', 0, lines)
        self.assertAlmostEqual(float(v[2]), 0.5, places=5)

    def test_digit_pct_and_trailing(self):
        lines = ['abc123']
        v = layout_features('abc123', 0, lines)
        self.assertAlmostEqual(float(v[3]), 0.5, places=5)
        self.assertEqual(float(v[4]), 1.0)  # trailing_digit_flag

    def test_is_short_threshold(self):
        line_29 = 'A' * 29
        line_30 = 'A' * 30
        lines = [line_29, line_30]
        self.assertEqual(layout_features(line_29, 0, lines)[5], 1.0)
        self.assertEqual(layout_features(line_30, 1, lines)[5], 0.0)

    def test_blank_before_after(self):
        lines = ['', 'middle line', '']
        v = layout_features('middle line', 1, lines)
        self.assertEqual(float(v[6]), 1.0)  # blank_before
        self.assertEqual(float(v[7]), 1.0)  # blank_after

    def test_neither_blank_neighbour(self):
        lines = ['top body', 'middle body', 'bottom body']
        v = layout_features('middle body', 1, lines)
        self.assertEqual(float(v[6]), 0.0)
        self.assertEqual(float(v[7]), 0.0)

    def test_first_line_blank_before_true(self):
        lines = ['only line']
        v = layout_features('only line', 0, lines)
        self.assertEqual(float(v[6]), 1.0)  # no previous line
        self.assertEqual(float(v[7]), 1.0)  # no next line


# ---------------------------------------------------------------------------
# 2. Particle counts (12 = 11 labels + SP_NOV flag)
# ---------------------------------------------------------------------------


class TestParticleCounts(unittest.TestCase):
    """particle_counts maps each of the 11 PARTICLE_VOCAB labels to a
    count slot, plus slot 11 = SP_NOV flag derived from TaxonName
    spans on the line."""

    def test_vector_shape_is_12(self):
        v = particle_counts([], 0, 0)
        self.assertEqual(v.shape, (12,))
        self.assertEqual(v.dtype, np.float32)

    def test_count_per_label(self):
        """Two DOIs + one MB-number → DOI=2, MB-number=1, rest=0."""
        spans = [
            Span(start=0, end=10, label='DOI', text='x',
                 source='regex'),
            Span(start=15, end=25, label='DOI', text='y',
                 source='regex'),
            Span(start=30, end=40, label='MB-number', text='z',
                 source='regex'),
        ]
        v = particle_counts(spans, 0, 100)
        doi_idx = PARTICLE_VOCAB.index('DOI')
        mb_idx = PARTICLE_VOCAB.index('MB-number')
        self.assertEqual(float(v[doi_idx]), 2.0)
        self.assertEqual(float(v[mb_idx]), 1.0)
        # All other slots zero.
        zero_slots = [
            i for i in range(12) if i not in (doi_idx, mb_idx)
        ]
        for i in zero_slots:
            self.assertEqual(float(v[i]), 0.0)

    def test_taxonname_and_sp_nov_flag(self):
        """A TaxonName with annot_nomen_type=SP_NOV bumps the
        TaxonName slot AND the SP_NOV-flag slot (last position)."""
        spans = [
            Span(
                start=0, end=14, label='TaxonName',
                text='Boletus edulis', source='gnfinder',
                metadata={'annot_nomen_type': 'SP_NOV'},
            ),
        ]
        v = particle_counts(spans, 0, 100)
        tn_idx = PARTICLE_VOCAB.index('TaxonName')
        self.assertEqual(float(v[tn_idx]), 1.0)
        self.assertEqual(float(v[11]), 1.0)  # SP_NOV slot

    def test_taxonname_without_sp_nov(self):
        """TaxonName without SP_NOV annotation → TaxonName slot=1,
        SP_NOV slot=0."""
        spans = [
            Span(
                start=0, end=14, label='TaxonName',
                text='Boletus edulis', source='gnfinder',
                metadata={},
            ),
        ]
        v = particle_counts(spans, 0, 100)
        tn_idx = PARTICLE_VOCAB.index('TaxonName')
        self.assertEqual(float(v[tn_idx]), 1.0)
        self.assertEqual(float(v[11]), 0.0)

    def test_no_overlap_means_no_count(self):
        """Span entirely outside the line's byte range → not counted."""
        spans = [
            Span(start=0, end=10, label='DOI', text='x',
                 source='regex'),
        ]
        # Line range [100, 200) — span is way before.
        v = particle_counts(spans, 100, 200)
        for slot in range(12):
            self.assertEqual(float(v[slot]), 0.0)

    def test_section_header_excluded_from_particles(self):
        """``section-header`` spans don't contaminate the particle
        vector — they have their own dedicated feature."""
        spans = [
            Span(start=0, end=12, label='section-header',
                 text='Introduction', source='regex'),
        ]
        v = particle_counts(spans, 0, 100)
        for slot in range(12):
            self.assertEqual(float(v[slot]), 0.0)


# ---------------------------------------------------------------------------
# 3. Page-header score (2 = confidence + binary flag)
# ---------------------------------------------------------------------------


class TestPageHeaderScore(unittest.TestCase):

    def test_per_line_confidence_in_slot_0(self):
        ph = {
            'schema_version': '1',
            'n_lines': 3,
            'per_line_confidence': [0.0, 0.85, 0.0],
            'regions': [], 'sequence_fit': None,
            'alternation_score': 0.0,
        }
        v = page_header_score(1, ph)
        self.assertEqual(v.shape, (2,))
        self.assertAlmostEqual(float(v[0]), 0.85, places=5)

    def test_binary_flag_in_slot_1(self):
        ph_yes = {'per_line_confidence': [0.7]}
        ph_no = {'per_line_confidence': [0.0]}
        self.assertEqual(float(page_header_score(0, ph_yes)[1]), 1.0)
        self.assertEqual(float(page_header_score(0, ph_no)[1]), 0.0)

    def test_short_array_defaults_zero(self):
        """Defensive: line_index past end of confidence array → 0.0
        with no IndexError."""
        ph = {'per_line_confidence': [0.5]}
        v = page_header_score(99, ph)
        self.assertEqual(float(v[0]), 0.0)
        self.assertEqual(float(v[1]), 0.0)

    def test_missing_confidence_key_defaults_zero(self):
        v = page_header_score(0, {})
        self.assertEqual(float(v[0]), 0.0)

    def test_clamping_to_unit_interval(self):
        """Even if upstream produced > 1.0 by accident, we clamp."""
        ph = {'per_line_confidence': [1.5]}
        self.assertEqual(float(page_header_score(0, ph)[0]), 1.0)


# ---------------------------------------------------------------------------
# 4. Section-header flag
# ---------------------------------------------------------------------------


class TestSectionHeaderFlag(unittest.TestCase):

    def test_section_header_overlap_sets_flag(self):
        spans = [
            Span(start=0, end=12, label='section-header',
                 text='Introduction', source='regex'),
        ]
        v = section_header_flag(spans, 0, 12)
        self.assertEqual(v.shape, (1,))
        self.assertEqual(float(v[0]), 1.0)

    def test_no_overlap_zero(self):
        spans = [
            Span(start=100, end=112, label='section-header',
                 text='Discussion', source='regex'),
        ]
        v = section_header_flag(spans, 0, 50)
        self.assertEqual(float(v[0]), 0.0)

    def test_non_section_span_does_not_count(self):
        spans = [
            Span(start=0, end=10, label='DOI', text='x',
                 source='regex'),
        ]
        v = section_header_flag(spans, 0, 50)
        self.assertEqual(float(v[0]), 0.0)


# ---------------------------------------------------------------------------
# 5. build_line_features orchestrator
# ---------------------------------------------------------------------------


class TestBuildLineFeatures(unittest.TestCase):

    def _basic_inputs(self):
        lines = ['first line', 'second line', 'third line']
        spans = [
            Span(start=0, end=10, label='DOI', text='10.x/y',
                 source='regex'),
        ]
        page_headers = {
            'per_line_confidence': [0.0, 0.0, 0.0],
        }
        return lines, spans, page_headers

    def test_returns_correct_shape_concat_791(self):
        lines, spans, page_headers = self._basic_inputs()
        feats = build_line_features(
            line_text='first line', line_index=0,
            doc_lines=lines, spans=spans,
            page_headers=page_headers,
            sbert_lookup=_stub_sbert_zero,
        )
        self.assertIsInstance(feats, LineFeatures)
        vec = feats.concat()
        self.assertEqual(vec.shape, (791,))
        self.assertEqual(vec.dtype, np.float32)

    def test_sbert_cache_hit_uses_cached_vector(self):
        lines, spans, page_headers = self._basic_inputs()
        feats = build_line_features(
            line_text='first line', line_index=0,
            doc_lines=lines, spans=spans,
            page_headers=page_headers,
            sbert_lookup=_stub_sbert_const(0.5),
        )
        vec = feats.concat()
        # Slots 0..767 should be the cached value 0.5.
        self.assertTrue(np.all(vec[:768] == np.float32(0.5)))

    def test_sbert_cache_miss_zero_vector(self):
        """On cache miss the assembler falls back to a zero vector
        (per user-locked choice).  Slots 0..767 must be zero."""
        lines, spans, page_headers = self._basic_inputs()
        feats = build_line_features(
            line_text='first line', line_index=0,
            doc_lines=lines, spans=spans,
            page_headers=page_headers,
            sbert_lookup=_stub_sbert_miss,
        )
        vec = feats.concat()
        self.assertTrue(np.all(vec[:768] == np.float32(0.0)))

    def test_block_ordering_in_concat(self):
        """Verify the order: sbert[768] + particles[12] + layout[8] +
        page_header_score[2] + section_header_flag[1]."""
        lines, spans, page_headers = self._basic_inputs()
        feats = build_line_features(
            line_text='first line', line_index=0,
            doc_lines=lines, spans=spans,
            page_headers=page_headers,
            sbert_lookup=_stub_sbert_zero,
        )
        vec = feats.concat()
        np.testing.assert_array_equal(vec[:768], feats.sbert)
        np.testing.assert_array_equal(vec[768:780], feats.particles)
        np.testing.assert_array_equal(vec[780:788], feats.layout)
        np.testing.assert_array_equal(vec[788:790],
                                      feats.page_header_score)
        np.testing.assert_array_equal(vec[790:791],
                                      feats.section_header_flag)


# ---------------------------------------------------------------------------
# 6. Parity (Step 2.C)
# ---------------------------------------------------------------------------


class TestParityAcrossTwoCalls(unittest.TestCase):
    """Step 2.C: same inputs → identical output vectors.  Catches a
    future regression where someone introduces nondeterminism (e.g.
    random ordering of spans, time-based seed, dict-iteration order)."""

    def test_same_inputs_yield_identical_vectors(self):
        lines = ['Introduction', 'body text', '']
        spans = [
            Span(start=0, end=12, label='section-header',
                 text='Introduction', source='regex'),
            Span(start=13, end=22, label='DOI',
                 text='10.x/abc', source='regex'),
        ]
        page_headers = {'per_line_confidence': [0.0, 0.0, 0.0]}

        a = build_line_features(
            'Introduction', 0, lines, spans, page_headers,
            _stub_sbert_const(0.3),
        )
        b = build_line_features(
            'Introduction', 0, lines, spans, page_headers,
            _stub_sbert_const(0.3),
        )
        np.testing.assert_array_equal(a.concat(), b.concat())


# ---------------------------------------------------------------------------
# 7. PARTICLE_VOCAB integrity
# ---------------------------------------------------------------------------


class TestParticleVocab(unittest.TestCase):

    def test_vocab_length_and_uniqueness(self):
        self.assertEqual(len(PARTICLE_VOCAB), 11)
        self.assertEqual(len(set(PARTICLE_VOCAB)),
                         len(PARTICLE_VOCAB))

    def test_section_header_not_in_vocab(self):
        """Per user choice — section-header has its own dedicated
        feature, not a particle slot."""
        self.assertNotIn('section-header', PARTICLE_VOCAB)

    def test_pdf_page_marker_not_in_vocab(self):
        """Per user choice — implied by page_header_score."""
        self.assertNotIn('PDF-page-marker', PARTICLE_VOCAB)


if __name__ == '__main__':
    unittest.main()
