"""Tests for skol_classifier/v4/predictor.py — v4 end-to-end predictor.

Two layers:

* ``TestCoalesceBlocks`` — pure (lines, tags) → ``List[TaggedBlock]``
  helper.  No models, no CouchDB.
* ``TestPredictDoc`` — assembles a tiny ``LayoutCRF`` + ``TreatmentCRF``
  with hand-crafted emission weights so decoding is deterministic
  without real training, then verifies the splice + round-trip.

The strategy mirrors ``crf_layout_test.py``: use a low-dim model
(feature_dim == n_labels) with an identity emission, so the
"feature i is one-hot at the desired label" trick gives us exact
control over per-line decoded labels.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ingestors.yedda_tags import Tag, TaggedBlock  # noqa: E402

from skol_classifier.v4.crf_layout import (  # noqa: E402
    LAYOUT_LABELS,
    LayoutCRF,
    OTHER_INDEX as LAYOUT_OTHER_INDEX,
)
from skol_classifier.v4.crf_treatment import (  # noqa: E402
    TREATMENT_LABELS,
    TreatmentCRF,
)
from skol_classifier.v4.labels import yedda_tag_per_line  # noqa: E402


# ---------------------------------------------------------------------------
# Test helpers — minimal CRFs with identity-style emissions
# ---------------------------------------------------------------------------


def _identity_layout_crf() -> LayoutCRF:
    """Build a ``LayoutCRF`` with ``feature_dim == n_labels == 8`` and
    identity emission so ``decode([one_hot_at_k]) == [k]``."""
    model = LayoutCRF(feature_dim=8, n_labels=8)
    with torch.no_grad():
        model.emission.weight.copy_(torch.eye(8))
        model.emission.bias.zero_()
        # Push the CRF transition matrix toward "no transition penalty"
        # so the per-line emission dominates Viterbi.  Default init is
        # random-uniform; zeroing it makes the identity test robust.
        model.crf.transitions.zero_()
        model.crf.start_transitions.zero_()
        model.crf.end_transitions.zero_()
    return model


def _identity_treatment_crf() -> TreatmentCRF:
    """Same trick for ``TreatmentCRF``: ``feature_dim == 12``."""
    model = TreatmentCRF(feature_dim=12, n_labels=12)
    with torch.no_grad():
        model.emission.weight.copy_(torch.eye(12))
        model.emission.bias.zero_()
        model.crf.transitions.zero_()
        model.crf.start_transitions.zero_()
        model.crf.end_transitions.zero_()
    return model


def _layout_one_hot(label_idx: int) -> np.ndarray:
    v = np.zeros(8, dtype=np.float32)
    v[label_idx] = 1.0
    return v


def _treatment_one_hot(label_idx: int) -> np.ndarray:
    v = np.zeros(12, dtype=np.float32)
    v[label_idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# 1. _coalesce_blocks
# ---------------------------------------------------------------------------


class TestCoalesceBlocks(unittest.TestCase):
    """Pure (lines, tags) → blocks helper.  Break on label change OR
    blank line.  Blank lines are dropped (not their own block)."""

    def _coalesce(self, lines, tags) -> List[TaggedBlock]:
        from skol_classifier.v4.predictor import _coalesce_blocks
        return _coalesce_blocks(lines, tags)

    def test_breaks_on_label_change(self):
        blocks = self._coalesce(
            ['line a', 'line b'],
            ['Description', 'Bibliography'],
        )
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].text, 'line a')
        self.assertEqual(blocks[0].tag, Tag.DESCRIPTION)
        self.assertEqual(blocks[1].text, 'line b')
        self.assertEqual(blocks[1].tag, Tag.BIBLIOGRAPHY)

    def test_breaks_on_blank_line_same_label(self):
        """Two Description runs separated by a blank line stay as
        TWO blocks, matching the hand-annotated v2 corpus shape."""
        blocks = self._coalesce(
            ['Body 1', '', 'Body 2'],
            ['Description', 'Misc-exposition', 'Description'],
        )
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].text, 'Body 1')
        self.assertEqual(blocks[1].text, 'Body 2')

    def test_drops_blank_lines(self):
        """Blank lines (after strip) never become their own block."""
        blocks = self._coalesce(
            ['', 'real line', ''],
            ['Misc-exposition', 'Description', 'Misc-exposition'],
        )
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, 'real line')

    def test_multiline_run_joined_with_newlines(self):
        """A run of N same-label consecutive non-blank lines becomes
        ONE block whose text is the lines joined by '\\n'."""
        blocks = self._coalesce(
            ['line 0', 'line 1', 'line 2'],
            ['Description', 'Description', 'Description'],
        )
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, 'line 0\nline 1\nline 2')
        self.assertEqual(blocks[0].tag, Tag.DESCRIPTION)

    def test_single_line_block(self):
        blocks = self._coalesce(['solo'], ['Page-header'])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, 'solo')
        self.assertEqual(blocks[0].tag, Tag.PAGE_HEADER)

    def test_whitespace_only_treated_as_blank(self):
        """A line of just spaces is also a block separator."""
        blocks = self._coalesce(
            ['a', '   ', 'b'],
            ['Description', 'Description', 'Description'],
        )
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].text, 'a')
        self.assertEqual(blocks[1].text, 'b')

    def test_empty_input(self):
        self.assertEqual(self._coalesce([], []), [])

    def test_all_blank_input(self):
        self.assertEqual(
            self._coalesce(['', '   '], ['Misc-exposition'] * 2),
            [],
        )


# ---------------------------------------------------------------------------
# 2. predict_doc — end-to-end
# ---------------------------------------------------------------------------


def _stub_sbert_lookup(_text: str) -> Optional[np.ndarray]:
    """Default lookup: cache miss → predictor uses zero SBERT vector.
    Tests that need per-line control patch the feature path instead.
    """
    return None


class TestPredictDoc(unittest.TestCase):
    """Predictor end-to-end with hand-rigged CRFs.

    We patch ``_features.build_line_features`` (via dependency
    injection in the predictor) or, easier, pass models with
    ``feature_dim`` matching the per-line one-hot vectors and patch
    the per-line feature assembler.  Two strategies are used:

    * For tests that care about decode → splice → emit, we replace
      the per-line feature vector with a one-hot whose width matches
      the CRF.  That requires a custom ``predict_doc`` entry point
      that takes pre-built features.  This module exposes
      ``predict_from_features`` for exactly that purpose.
    """

    def _predict_from_features(
        self,
        lines,
        layout_features,    # (T, 8) ndarray
        treatment_features,  # (T, 12) ndarray
        *,
        layout_crf=None,
        treatment_crf=None,
    ):
        """Skip feature assembly; feed the two pre-built feature
        tensors directly to ``predict_from_features`` and return
        ``(per_line_tags, ann_text)``."""
        from skol_classifier.v4.predictor import predict_from_features
        layout = layout_crf or _identity_layout_crf()
        treatment = treatment_crf or _identity_treatment_crf()
        return predict_from_features(
            lines,
            np.asarray(layout_features, dtype=np.float32),
            np.asarray(treatment_features, dtype=np.float32),
            layout,
            treatment,
            device='cpu',
        )

    def test_layout_only_doc_emits_layout_blocks(self):
        """Every line is Page-header (layout idx 0).  Pass-2 mask
        is empty; the predictor must not crash and must emit
        page-header blocks for each non-blank line."""
        lines = ['Header A', 'Header B']
        layout_feats = np.stack([
            _layout_one_hot(0),  # Page-header
            _layout_one_hot(0),
        ])
        # Treatment features irrelevant — content_mask is empty —
        # but the array still has to be the right (T, 12) shape.
        treat_feats = np.zeros((2, 12), dtype=np.float32)

        tags, ann = self._predict_from_features(
            lines, layout_feats, treat_feats,
        )
        self.assertEqual(tags, ['Page-header', 'Page-header'])
        self.assertIn('Page-header', ann)
        # Two same-label non-blank consecutive lines → one block.
        self.assertEqual(ann.count('[@'), 1)

    def test_treatment_only_doc_emits_treatment_blocks(self):
        """Every line is treatment (layout idx 7 = Other).  Pass-2
        sees a contiguous tensor; treatments splice into every
        position."""
        lines = ['Nomen line', 'Descr line']
        layout_feats = np.stack([
            _layout_one_hot(LAYOUT_OTHER_INDEX),
            _layout_one_hot(LAYOUT_OTHER_INDEX),
        ])
        treat_feats = np.stack([
            _treatment_one_hot(0),   # Nomenclature
            _treatment_one_hot(1),   # Description
        ])

        tags, ann = self._predict_from_features(
            lines, layout_feats, treat_feats,
        )
        self.assertEqual(tags, ['Nomenclature', 'Description'])
        self.assertIn('Nomenclature', ann)
        self.assertIn('Description', ann)

    def test_interior_layout_run_splice_order(self):
        """Treatment / layout / treatment alternation — verify the
        Pass-2 predictions splice back to the correct line positions
        and don't drift through the layout gap."""
        lines = [
            'Nomen line',  # 0 — treatment
            'Header line',  # 1 — layout (Bibliography idx 4)
            'Descr line',  # 2 — treatment
        ]
        layout_feats = np.stack([
            _layout_one_hot(LAYOUT_OTHER_INDEX),
            _layout_one_hot(4),   # Bibliography
            _layout_one_hot(LAYOUT_OTHER_INDEX),
        ])
        # Pass-2 sees only positions 0 and 2 (the non-layout filter).
        # We need the filtered features [Nomen, Descr] at indices 0, 1.
        treat_feats = np.stack([
            _treatment_one_hot(0),   # Nomenclature   (line 0)
            np.zeros(12, dtype=np.float32),  # dummy (line 1, filtered out)
            _treatment_one_hot(1),   # Description    (line 2)
        ])

        tags, ann = self._predict_from_features(
            lines, layout_feats, treat_feats,
        )
        self.assertEqual(
            tags, ['Nomenclature', 'Bibliography', 'Description'],
        )
        # 3 blocks — no two consecutive lines share a label.
        self.assertEqual(ann.count('[@'), 3)

    def test_round_trip_through_yedda_tag_per_line(self):
        """Predict, then parse the emitted .ann back via
        ``yedda_tag_per_line``: per-line tags must match what we
        decoded (modulo blank lines, which default to
        Misc-exposition on the parse side)."""
        lines = [
            'Top header',           # 0 — layout
            '',                     # 1 — blank
            'Nomen line',           # 2 — treatment Nomenclature
            'Descr line 1',         # 3 — treatment Description
            'Descr line 2',         # 4 — treatment Description
        ]
        layout_feats = np.stack([
            _layout_one_hot(0),                  # Page-header
            _layout_one_hot(LAYOUT_OTHER_INDEX),  # default
            _layout_one_hot(LAYOUT_OTHER_INDEX),
            _layout_one_hot(LAYOUT_OTHER_INDEX),
            _layout_one_hot(LAYOUT_OTHER_INDEX),
        ])
        treat_feats = np.stack([
            np.zeros(12, dtype=np.float32),       # line 0 filtered
            _treatment_one_hot(11),               # Misc-exposition (blank)
            _treatment_one_hot(0),                # Nomenclature
            _treatment_one_hot(1),                # Description
            _treatment_one_hot(1),                # Description
        ])

        tags, ann = self._predict_from_features(
            lines, layout_feats, treat_feats,
        )
        # Predictor's per-line tag for the blank line is whatever
        # Pass-2 chose for that slot — it's still a "valid" tag
        # because content_mask is True there.  But the emitter drops
        # blank-line blocks entirely, so the round-trip parse sees
        # only the three non-blank blocks.
        plaintext = '\n'.join(lines)
        parsed_tags = yedda_tag_per_line(plaintext, ann)
        # For non-blank lines, parsed must equal predicted.
        for li, line in enumerate(lines):
            if line.strip():
                self.assertEqual(
                    parsed_tags[li], tags[li],
                    f'round-trip mismatch on line {li}: '
                    f'predicted {tags[li]!r}, parsed {parsed_tags[li]!r}',
                )

    def test_returns_empty_for_empty_plaintext(self):
        from skol_classifier.v4.predictor import predict_from_features
        tags, ann = predict_from_features(
            [], np.zeros((0, 8), dtype=np.float32),
            np.zeros((0, 12), dtype=np.float32),
            _identity_layout_crf(), _identity_treatment_crf(),
            device='cpu',
        )
        self.assertEqual(tags, [])
        self.assertEqual(ann, '')


# ---------------------------------------------------------------------------
# 3. Full predict_doc pipeline (feature assembly + decode + emit)
# ---------------------------------------------------------------------------


class TestPredictDocFullPipeline(unittest.TestCase):
    """Exercise the public ``predict_doc`` entry point including
    feature assembly.  Uses the default 791-d CRFs whose untrained
    decode is non-deterministic — so these tests assert SHAPE and
    contracts (right number of lines, valid tags, parseable .ann),
    not exact label values."""

    def test_returns_one_tag_per_plaintext_line(self):
        from skol_classifier.v4.predictor import predict_doc
        plaintext = 'Line one\nLine two\nLine three'
        spans = {'version': '1', 'spans': []}
        page_headers = {'per_line_confidence': [0.0, 0.0, 0.0]}
        layout = LayoutCRF()    # default 791-d, untrained
        treatment = TreatmentCRF()
        tags, ann = predict_doc(
            plaintext, spans, page_headers,
            layout, treatment, _stub_sbert_lookup,
            device='cpu',
        )
        self.assertEqual(len(tags), 3)
        valid = set(LAYOUT_LABELS) | set(TREATMENT_LABELS)
        valid.discard('Other')   # sentinel never reaches output
        for t in tags:
            self.assertIn(t, valid)
        self.assertTrue(ann.endswith('\n'))

    def test_empty_plaintext_returns_empty(self):
        from skol_classifier.v4.predictor import predict_doc
        tags, ann = predict_doc(
            '', {'version': '1', 'spans': []},
            {'per_line_confidence': []},
            LayoutCRF(), TreatmentCRF(),
            _stub_sbert_lookup, device='cpu',
        )
        # '' splits to [''] → 1 blank line.  Tag list length matches.
        self.assertEqual(len(tags), 1)
        # Blank-only doc emits no blocks.
        self.assertEqual(ann, '')


# ---------------------------------------------------------------------------
# 4. FEATURE_DIM mismatch guard
# ---------------------------------------------------------------------------


class TestFeatureDimMismatch(unittest.TestCase):

    def test_raises_when_layout_feature_dim_disagrees(self):
        """If the Pass-1 CRF was trained with a different feature
        width than the active ``features.FEATURE_DIM`` (e.g. SBERT
        dim drift), ``predict_doc`` must error out clearly rather
        than silently emit nonsense."""
        from skol_classifier.v4.predictor import predict_doc

        bad_layout = LayoutCRF(feature_dim=100, n_labels=8)
        good_treatment = TreatmentCRF()

        with self.assertRaises(ValueError) as cm:
            predict_doc(
                'one line', {'version': '1', 'spans': []},
                {'per_line_confidence': [0.0]},
                bad_layout, good_treatment,
                _stub_sbert_lookup, device='cpu',
            )
        self.assertIn('feature_dim', str(cm.exception).lower())

    def test_raises_when_treatment_feature_dim_disagrees(self):
        from skol_classifier.v4.predictor import predict_doc

        good_layout = LayoutCRF()
        bad_treatment = TreatmentCRF(feature_dim=100, n_labels=12)

        with self.assertRaises(ValueError) as cm:
            predict_doc(
                'one line', {'version': '1', 'spans': []},
                {'per_line_confidence': [0.0]},
                good_layout, bad_treatment,
                _stub_sbert_lookup, device='cpu',
            )
        self.assertIn('feature_dim', str(cm.exception).lower())


# ---------------------------------------------------------------------------
# 5. predict_doc_single (Step 6.F single-CRF baseline)
# ---------------------------------------------------------------------------


def _identity_single_crf():
    """SingleCRF with feature_dim == n_labels == 19 and identity
    emission, so ``decode([one_hot_at_k]) == [k]`` deterministically
    without training."""
    from skol_classifier.v4.crf_single import SingleCRF
    model = SingleCRF(feature_dim=19, n_labels=19)
    with torch.no_grad():
        model.emission.weight.copy_(torch.eye(19))
        model.emission.bias.zero_()
        model.crf.transitions.zero_()
        model.crf.start_transitions.zero_()
        model.crf.end_transitions.zero_()
    return model


def _single_one_hot(label_idx: int) -> np.ndarray:
    v = np.zeros(19, dtype=np.float32)
    v[label_idx] = 1.0
    return v


class TestPredictDocSingle(unittest.TestCase):
    """Single-CRF inference: features → decode → coalesce → emit.
    Mirrors :class:`TestPredictDoc` but for the 19-label CRF, which
    decides every line directly (no Pass-1 / Pass-2 split)."""

    def test_returns_one_tag_per_line(self):
        from skol_classifier.v4.crf_single import LABEL_TO_INDEX
        from skol_classifier.v4.predictor import predict_from_features_single

        lines = [
            'Header A',                # 0 — Page-header
            'Body text',               # 1 — Description
            'Bibliography entry',      # 2 — Bibliography
        ]
        feats = np.stack([
            _single_one_hot(LABEL_TO_INDEX['Page-header']),
            _single_one_hot(LABEL_TO_INDEX['Description']),
            _single_one_hot(LABEL_TO_INDEX['Bibliography']),
        ])
        tags, ann = predict_from_features_single(
            lines, feats, _identity_single_crf(), device='cpu',
        )
        self.assertEqual(
            tags, ['Page-header', 'Description', 'Bibliography'],
        )
        # 3 different labels in a row → 3 blocks.
        self.assertEqual(ann.count('[@'), 3)

    def test_round_trip_through_yedda_tag_per_line(self):
        """Predict → emit → re-parse: every non-blank line's tag
        round-trips."""
        from skol_classifier.v4.crf_single import LABEL_TO_INDEX
        from skol_classifier.v4.predictor import predict_from_features_single

        lines = [
            'Header',                  # 0 — Page-header
            '',                        # 1 — blank
            'Nomen line',              # 2 — Nomenclature
            'Desc line 1',             # 3 — Description
            'Desc line 2',             # 4 — Description
        ]
        feats = np.stack([
            _single_one_hot(LABEL_TO_INDEX['Page-header']),
            _single_one_hot(LABEL_TO_INDEX['Misc-exposition']),
            _single_one_hot(LABEL_TO_INDEX['Nomenclature']),
            _single_one_hot(LABEL_TO_INDEX['Description']),
            _single_one_hot(LABEL_TO_INDEX['Description']),
        ])
        tags, ann = predict_from_features_single(
            lines, feats, _identity_single_crf(), device='cpu',
        )
        plaintext = '\n'.join(lines)
        parsed = yedda_tag_per_line(plaintext, ann)
        for li, line in enumerate(lines):
            if line.strip():
                self.assertEqual(
                    parsed[li], tags[li],
                    f'round-trip mismatch on line {li}: '
                    f'predicted {tags[li]!r}, parsed {parsed[li]!r}',
                )

    def test_returns_empty_for_empty_input(self):
        from skol_classifier.v4.predictor import predict_from_features_single

        tags, ann = predict_from_features_single(
            [], np.zeros((0, 19), dtype=np.float32),
            _identity_single_crf(), device='cpu',
        )
        self.assertEqual(tags, [])
        self.assertEqual(ann, '')


class TestPredictDocSingleFullPipeline(unittest.TestCase):
    """``predict_doc_single`` end-to-end with feature assembly."""

    def test_returns_one_tag_per_plaintext_line(self):
        from skol_classifier.v4.crf_single import (
            ACTIVE_LABELS, SingleCRF,
        )
        from skol_classifier.v4.predictor import predict_doc_single

        plaintext = 'Line one\nLine two\nLine three'
        spans = {'version': '1', 'spans': []}
        page_headers = {'per_line_confidence': [0.0, 0.0, 0.0]}
        single = SingleCRF()
        tags, ann = predict_doc_single(
            plaintext, spans, page_headers, single,
            _stub_sbert_lookup, device='cpu',
        )
        self.assertEqual(len(tags), 3)
        valid = set(ACTIVE_LABELS)
        for t in tags:
            self.assertIn(t, valid)
        self.assertTrue(ann.endswith('\n') or ann == '')


class TestPredictDocSingleFeatureDim(unittest.TestCase):

    def test_raises_when_single_crf_feature_dim_disagrees(self):
        from skol_classifier.v4.crf_single import SingleCRF
        from skol_classifier.v4.predictor import predict_doc_single

        bad_single = SingleCRF(feature_dim=100, n_labels=19)
        with self.assertRaises(ValueError) as cm:
            predict_doc_single(
                'one line', {'version': '1', 'spans': []},
                {'per_line_confidence': [0.0]},
                bad_single,
                _stub_sbert_lookup, device='cpu',
            )
        self.assertIn('feature_dim', str(cm.exception).lower())


if __name__ == '__main__':
    unittest.main()
