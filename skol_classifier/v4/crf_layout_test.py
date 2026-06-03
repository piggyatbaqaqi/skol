"""Tests for skol_classifier/v4/crf_layout.py — v4 Pass-1 CRF model."""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from skol_classifier.v4.crf_layout import (  # noqa: E402
    INDEX_TO_LABEL,
    LABEL_TO_INDEX,
    LAYOUT_LABELS,
    N_LABELS,
    OTHER_INDEX,
    LayoutCRF,
    deserialize,
    serialize,
)


# ---------------------------------------------------------------------------
# 1. Label space
# ---------------------------------------------------------------------------


class TestLabelSpace(unittest.TestCase):
    """The Pass-1 vocab is locked at 7 layout labels + 1 sentinel
    'Other' = 8 total, per the v4 plan §Label-space partition."""

    def test_layout_labels_count(self):
        self.assertEqual(len(LAYOUT_LABELS), 8)
        self.assertEqual(N_LABELS, 8)

    def test_layout_labels_contents(self):
        """All 7 plan-listed layout tags plus the synthetic Other
        sentinel must be present."""
        expected = {
            'Page-header', 'Figure-caption', 'Table', 'Key',
            'Bibliography', 'Index', 'ToC-entry', 'Other',
        }
        self.assertEqual(set(LAYOUT_LABELS), expected)

    def test_label_map_round_trips(self):
        for i, label in enumerate(LAYOUT_LABELS):
            self.assertEqual(LABEL_TO_INDEX[label], i)
            self.assertEqual(INDEX_TO_LABEL[i], label)

    def test_other_index_consistency(self):
        self.assertEqual(LAYOUT_LABELS[OTHER_INDEX], 'Other')


# ---------------------------------------------------------------------------
# 2. Construction
# ---------------------------------------------------------------------------


class TestConstruction(unittest.TestCase):

    def test_default_shapes(self):
        """Default LayoutCRF matches the v4 feature dim (791) and the
        8-label space."""
        model = LayoutCRF()
        self.assertEqual(model.emission.in_features, 791)
        self.assertEqual(model.emission.out_features, 8)
        self.assertEqual(model.crf.num_tags, 8)

    def test_custom_dims(self):
        """Custom dims (used in tests + future smaller-model
        experiments) propagate cleanly."""
        model = LayoutCRF(feature_dim=128, n_labels=3)
        self.assertEqual(model.emission.in_features, 128)
        self.assertEqual(model.emission.out_features, 3)
        self.assertEqual(model.crf.num_tags, 3)


# ---------------------------------------------------------------------------
# 3. Forward and decode
# ---------------------------------------------------------------------------


class TestForwardAndDecode(unittest.TestCase):

    def _batch(self, B=2, T=5, D=791, n_labels=8):
        features = torch.randn(B, T, D)
        tags = torch.randint(0, n_labels, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)
        return features, tags, mask

    def test_forward_returns_scalar(self):
        model = LayoutCRF()
        features, tags, mask = self._batch()
        loss = model(features, tags, mask)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_decode_returns_list_per_batch(self):
        model = LayoutCRF()
        features, _, mask = self._batch(B=2, T=5)
        labels = model.decode(features, mask)
        self.assertIsInstance(labels, list)
        self.assertEqual(len(labels), 2)
        for row in labels:
            self.assertEqual(len(row), 5)
            for li in row:
                self.assertIn(li, range(8))

    def test_decode_respects_mask(self):
        """A row with the last 2 positions masked out → returned label
        list has length 3 for that row."""
        model = LayoutCRF()
        features = torch.randn(1, 5, 791)
        mask = torch.tensor([[True, True, True, False, False]])
        labels = model.decode(features, mask)
        self.assertEqual(len(labels[0]), 3)


# ---------------------------------------------------------------------------
# 4. Synthetic convergence
# ---------------------------------------------------------------------------


class TestSyntheticConvergence(unittest.TestCase):
    """Train on a tiny but unambiguous fixture and verify the CRF
    converges to the correct labels.  This is the Step 3.D
    "synthetic-data test confirming Viterbi decodes correctly" — it
    proves training is wired up end-to-end."""

    def test_can_learn_id_function(self):
        """10-line sequence where each line's feature vector is a
        one-hot at its label index.  After a few SGD steps the CRF
        should decode the input back to the original tags exactly."""
        torch.manual_seed(0)
        n_labels = 4  # smaller label set keeps the test fast
        seq_len = 10
        # tags cycle through the labels.
        tags = torch.tensor([[i % n_labels for i in range(seq_len)]])
        features = torch.zeros(1, seq_len, n_labels)
        for t in range(seq_len):
            features[0, t, tags[0, t]] = 1.0
        mask = torch.ones(1, seq_len, dtype=torch.bool)

        model = LayoutCRF(feature_dim=n_labels, n_labels=n_labels)
        # Initialise the emission to identity so the synthetic signal
        # is "loud" and convergence is robust to seed.
        with torch.no_grad():
            model.emission.weight.copy_(torch.eye(n_labels))
            model.emission.bias.zero_()
        opt = torch.optim.Adam(model.parameters(), lr=0.5)
        for _ in range(50):
            opt.zero_grad()
            loss = model(features, tags, mask)
            loss.backward()
            opt.step()

        decoded = model.decode(features, mask)
        self.assertEqual(decoded[0], tags[0].tolist())


# ---------------------------------------------------------------------------
# 5. Serialize / deserialize round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip(unittest.TestCase):

    def test_serialize_deserialize_preserves_weights(self):
        """Train for one step, serialize, deserialize, decode on a
        fixed input.  Labels must match the pre-serialise model."""
        torch.manual_seed(7)
        model = LayoutCRF(feature_dim=16, n_labels=4)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        features = torch.randn(1, 8, 16)
        tags = torch.randint(0, 4, (1, 8))
        mask = torch.ones(1, 8, dtype=torch.bool)
        opt.zero_grad()
        model(features, tags, mask).backward()
        opt.step()

        before = model.decode(features, mask)

        state_bytes, meta_bytes = serialize(model, metadata={
            'schema_version': 1,
            'feature_dim': 16,
            'n_labels': 4,
            'label_map': {f'L{i}': i for i in range(4)},
        })
        restored, meta = deserialize(state_bytes, meta_bytes)

        self.assertEqual(meta['n_labels'], 4)
        self.assertEqual(meta['feature_dim'], 16)
        # Decode after restore must match decode before serialize.
        after = restored.decode(features, mask)
        self.assertEqual(before, after)

    def test_deserialize_rejects_mismatched_dims(self):
        """A metadata blob claiming feature_dim=16 but a state_dict
        for a 791-dim model should fail loudly rather than silently
        load the wrong shape."""
        big_model = LayoutCRF(feature_dim=791, n_labels=8)
        state_bytes, _ = serialize(big_model, metadata={})
        bogus_meta = ('{"schema_version":1,"feature_dim":16,'
                      '"n_labels":8,"label_map":{}}').encode('utf-8')
        with self.assertRaises(Exception):
            deserialize(state_bytes, bogus_meta)


# ---------------------------------------------------------------------------
# 6. Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency(unittest.TestCase):
    """Step 3.D idempotency contract: training with a fixed seed
    produces byte-identical state dicts.  Same input + same code =
    same model, every time."""

    def _train_one_step(self, seed):
        torch.manual_seed(seed)
        model = LayoutCRF(feature_dim=16, n_labels=4)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        features = torch.randn(1, 8, 16)
        tags = torch.randint(0, 4, (1, 8))
        mask = torch.ones(1, 8, dtype=torch.bool)
        opt.zero_grad()
        model(features, tags, mask).backward()
        opt.step()
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.getvalue()

    def test_same_seed_yields_identical_weights(self):
        a = self._train_one_step(seed=42)
        b = self._train_one_step(seed=42)
        self.assertEqual(a, b)

    def test_different_seeds_yield_different_weights(self):
        """Sanity: the seed actually does something."""
        a = self._train_one_step(seed=1)
        b = self._train_one_step(seed=2)
        self.assertNotEqual(a, b)


if __name__ == '__main__':
    unittest.main()
