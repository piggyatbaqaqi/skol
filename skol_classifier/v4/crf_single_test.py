"""Tests for skol_classifier/v4/crf_single.py — v4 Step 6.F
single-CRF baseline.

1-for-1 mirror of crf_layout_test.py / crf_treatment_test.py; only
the label vocab differs (19 ACTIVE_TAGS_19 instead of 8 / 12).
"""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ingestors.yedda_tags import ACTIVE_TAGS_19  # noqa: E402

from skol_classifier.v4.crf_single import (  # noqa: E402
    ACTIVE_LABELS,
    INDEX_TO_LABEL,
    LABEL_TO_INDEX,
    MISC_EXPOSITION_INDEX,
    N_LABELS,
    SingleCRF,
    deserialize,
    serialize,
)


# ---------------------------------------------------------------------------
# 1. Label space
# ---------------------------------------------------------------------------


class TestLabelSpace(unittest.TestCase):

    def test_active_labels_count(self):
        self.assertEqual(len(ACTIVE_LABELS), 19)
        self.assertEqual(N_LABELS, 19)

    def test_active_labels_matches_active_tags_19(self):
        """ACTIVE_LABELS preserves the Tag-enum declaration order
        from ingestors/yedda_tags.ACTIVE_TAGS_19 so the integer
        index is stable across re-imports."""
        self.assertEqual(
            ACTIVE_LABELS,
            tuple(t.value for t in ACTIVE_TAGS_19),
        )

    def test_label_map_round_trips(self):
        for i, label in enumerate(ACTIVE_LABELS):
            self.assertEqual(LABEL_TO_INDEX[label], i)
            self.assertEqual(INDEX_TO_LABEL[i], label)

    def test_misc_exposition_index_consistency(self):
        self.assertEqual(
            ACTIVE_LABELS[MISC_EXPOSITION_INDEX], 'Misc-exposition',
        )


# ---------------------------------------------------------------------------
# 2. Construction
# ---------------------------------------------------------------------------


class TestConstruction(unittest.TestCase):

    def test_default_shapes(self):
        model = SingleCRF()
        self.assertEqual(model.emission.in_features, 791)
        self.assertEqual(model.emission.out_features, 19)
        self.assertEqual(model.crf.num_tags, 19)

    def test_custom_dims(self):
        model = SingleCRF(feature_dim=64, n_labels=7)
        self.assertEqual(model.emission.in_features, 64)
        self.assertEqual(model.emission.out_features, 7)
        self.assertEqual(model.crf.num_tags, 7)


# ---------------------------------------------------------------------------
# 3. Forward + decode
# ---------------------------------------------------------------------------


class TestForwardAndDecode(unittest.TestCase):

    def _batch(self, B=2, T=5, D=791, n_labels=19):
        features = torch.randn(B, T, D)
        tags = torch.randint(0, n_labels, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)
        return features, tags, mask

    def test_forward_returns_scalar(self):
        model = SingleCRF()
        features, tags, mask = self._batch()
        loss = model(features, tags, mask)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_decode_returns_list_per_batch(self):
        model = SingleCRF()
        features, _, mask = self._batch(B=2, T=5)
        labels = model.decode(features, mask)
        self.assertEqual(len(labels), 2)
        for row in labels:
            self.assertEqual(len(row), 5)
            for li in row:
                self.assertIn(li, range(19))


# ---------------------------------------------------------------------------
# 4. Synthetic convergence
# ---------------------------------------------------------------------------


class TestSyntheticConvergence(unittest.TestCase):

    def test_can_learn_id_function(self):
        torch.manual_seed(0)
        n_labels = 5
        seq_len = 15
        tags = torch.tensor([[i % n_labels for i in range(seq_len)]])
        features = torch.zeros(1, seq_len, n_labels)
        for t in range(seq_len):
            features[0, t, tags[0, t]] = 1.0
        mask = torch.ones(1, seq_len, dtype=torch.bool)

        model = SingleCRF(feature_dim=n_labels, n_labels=n_labels)
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
        torch.manual_seed(7)
        model = SingleCRF(feature_dim=16, n_labels=5)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        features = torch.randn(1, 8, 16)
        tags = torch.randint(0, 5, (1, 8))
        mask = torch.ones(1, 8, dtype=torch.bool)
        opt.zero_grad()
        model(features, tags, mask).backward()
        opt.step()

        before = model.decode(features, mask)

        state_bytes, meta_bytes = serialize(model, metadata={
            'schema_version': 1,
            'feature_dim': 16,
            'n_labels': 5,
            'label_map': {f'L{i}': i for i in range(5)},
        })
        restored, meta = deserialize(state_bytes, meta_bytes)

        self.assertEqual(meta['n_labels'], 5)
        self.assertEqual(meta['feature_dim'], 16)
        after = restored.decode(features, mask)
        self.assertEqual(before, after)


# ---------------------------------------------------------------------------
# 6. Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency(unittest.TestCase):

    def _train_one_step(self, seed):
        torch.manual_seed(seed)
        model = SingleCRF(feature_dim=16, n_labels=5)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        features = torch.randn(1, 8, 16)
        tags = torch.randint(0, 5, (1, 8))
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
        a = self._train_one_step(seed=1)
        b = self._train_one_step(seed=2)
        self.assertNotEqual(a, b)


if __name__ == '__main__':
    unittest.main()
