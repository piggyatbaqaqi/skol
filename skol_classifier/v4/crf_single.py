"""v4 single-CRF baseline (Step 6.F ablation).

Linear-chain CRF over the full 19-label `ACTIVE_TAGS_19` vocab,
trained directly without a Pass-1 / Pass-2 split.  Same emission +
CRF shape as :class:`skol_classifier.v4.crf_layout.LayoutCRF`; only
the label vocab + Redis key differ.

Purpose: provide the comparison row Step 7's report uses to decide
whether v4's two-pass design is worth its complexity.  See
docs/v4_classifier_plan.md §Step 6.F.

The trainer (``bin/train_crf_single.py``) walks the hand corpus
(no JATS dilution, per the user-locked scope), builds 791-d
features for every line (no layout filtering — that's the point),
projects YEDDA tags via :func:`skol_classifier.v4.labels.map_yedda_to_active`,
fits the CRF, and persists via the helpers below.

Default Redis bundle:

    skol:classifier:model:v4_single_hand           state_dict bytes
    skol:classifier:model:v4_single_hand:meta      JSON metadata
"""
from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchcrf import CRF  # type: ignore[import-untyped]

from ingestors.yedda_tags import ACTIVE_TAGS_19
from skol_classifier.v4 import features as _features


# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------

ACTIVE_LABELS: Tuple[str, ...] = tuple(t.value for t in ACTIVE_TAGS_19)
"""19-tag vocab, ordered to match the ``Tag`` enum declaration order
(stable across rebuilds; used as the integer-index basis)."""

N_LABELS = len(ACTIVE_LABELS)
LABEL_TO_INDEX: Dict[str, int] = {
    label: idx for idx, label in enumerate(ACTIVE_LABELS)
}
INDEX_TO_LABEL: Tuple[str, ...] = ACTIVE_LABELS
MISC_EXPOSITION_INDEX = LABEL_TO_INDEX['Misc-exposition']


# Default feature dim — same 791 as the two-pass CRFs.
FEATURE_DIM: int = _features.FEATURE_DIM


_DEFAULT_REDIS_KEY = 'skol:classifier:model:v4_single_hand'
_DEFAULT_META_KEY = 'skol:classifier:model:v4_single_hand:meta'
_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SingleCRF(nn.Module):
    """Linear-chain CRF over the 19 ACTIVE_TAGS_19 labels.

    Architecturally identical to ``LayoutCRF`` and ``TreatmentCRF``;
    only the label count differs.  Trained on all lines of every
    doc (no layout/treatment split) so the comparison vs the
    two-pass production model is a clean architectural ablation.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        n_labels: int = N_LABELS,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.n_labels = n_labels
        self.emission = nn.Linear(feature_dim, n_labels)
        self.crf = CRF(n_labels, batch_first=True)

    def forward(
        self,
        features: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean negative log-likelihood for one batch."""
        emissions = self.emission(features)
        return -self.crf(
            emissions, tags, mask=mask, reduction='mean',
        )

    def decode(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[int]]:
        """Viterbi decode.  Returns one list of label indices per
        batch row, length equal to that row's mask sum."""
        emissions = self.emission(features)
        return self.crf.decode(emissions, mask=mask)


# ---------------------------------------------------------------------------
# Serialize / deserialize
# ---------------------------------------------------------------------------


def _default_metadata(model: SingleCRF) -> Dict[str, Any]:
    return {
        'schema_version': _SCHEMA_VERSION,
        'feature_dim': model.feature_dim,
        'n_labels': model.n_labels,
        'label_map': dict(LABEL_TO_INDEX),
        'torch_version': torch.__version__,
    }


def serialize(
    model: SingleCRF,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bytes, bytes]:
    """Return ``(state_dict_bytes, metadata_json_bytes)``."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    state_bytes = buf.getvalue()

    merged: Dict[str, Any] = _default_metadata(model)
    if metadata:
        merged.update(metadata)
    meta_bytes = json.dumps(merged).encode('utf-8')
    return state_bytes, meta_bytes


def deserialize(
    state_dict_bytes: bytes,
    metadata_json_bytes: bytes,
    *,
    map_location: Optional[str] = None,
) -> Tuple[SingleCRF, Dict[str, Any]]:
    """Inverse of :func:`serialize`.  Mismatched dims raise from
    ``load_state_dict``."""
    metadata: Dict[str, Any] = json.loads(
        metadata_json_bytes.decode('utf-8'),
    )
    feature_dim = int(metadata.get('feature_dim', FEATURE_DIM))
    n_labels = int(metadata.get('n_labels', N_LABELS))

    model = SingleCRF(feature_dim=feature_dim, n_labels=n_labels)
    state_dict = torch.load(
        io.BytesIO(state_dict_bytes),
        map_location=map_location,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    return model, metadata


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


def save_to_redis(
    model: SingleCRF,
    redis_client: Any,
    *,
    key: str = _DEFAULT_REDIS_KEY,
    meta_key: str = _DEFAULT_META_KEY,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write the state_dict + metadata bundle to Redis."""
    state_bytes, meta_bytes = serialize(model, metadata=metadata)
    redis_client.set(key, state_bytes)
    redis_client.set(meta_key, meta_bytes)


def load_from_redis(
    redis_client: Any,
    *,
    key: str = _DEFAULT_REDIS_KEY,
    meta_key: str = _DEFAULT_META_KEY,
    map_location: Optional[str] = None,
) -> Tuple[SingleCRF, Dict[str, Any]]:
    """Inverse of :func:`save_to_redis`."""
    state_bytes = redis_client.get(key)
    meta_bytes = redis_client.get(meta_key)
    if state_bytes is None or meta_bytes is None:
        raise RuntimeError(
            f'Missing v4_single model at {key!r} / {meta_key!r}'
        )
    return deserialize(
        state_bytes, meta_bytes, map_location=map_location,
    )
