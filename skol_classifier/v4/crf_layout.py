"""v4 Pass-1 layout CRF.

Linear-chain CRF over the 8 layout labels (7 from ACTIVE_TAGS_19 +
synthetic ``Other``) listed in
docs/v4_classifier_plan.md §Label-space partition.

Architecture:
    feature_vec  ->  Linear(feature_dim, n_labels)  ->  emissions
    emissions    ->  torchcrf.CRF                    ->  NLL loss (train)
                                                       Viterbi (decode)

The module is pure compute — no I/O.  Persistence is via
:func:`serialize` / :func:`deserialize` (in-memory bytes) plus the
Redis helpers :func:`save_to_redis` / :func:`load_from_redis`.

This commit lands the model + persistence layer.  The trainer
(``bin/train_crf_layout.py``) is a separate plan; it walks
``skol_training_v2_no_golden``, builds line-indexed YEDDA-tag labels,
calls Step 2's ``build_line_features`` to get per-line vectors,
fits the CRF, and persists via the helpers below.
"""
from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchcrf import CRF  # type: ignore[import-untyped]

from skol_classifier.v4.features import _PARTICLE_DIM  # noqa: F401  (anchor v4 features import)
from skol_classifier.v4 import features as _features


# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------

LAYOUT_LABELS: Tuple[str, ...] = (
    'Page-header',
    'Figure-caption',
    'Table',
    'Key',
    'Bibliography',
    'Index',
    'ToC-entry',
    'Other',
)
N_LABELS = len(LAYOUT_LABELS)
LABEL_TO_INDEX: Dict[str, int] = {
    label: idx for idx, label in enumerate(LAYOUT_LABELS)
}
INDEX_TO_LABEL: Tuple[str, ...] = LAYOUT_LABELS
OTHER_INDEX = LABEL_TO_INDEX['Other']


# Default feature dim from Step 2: sbert[768] + particles[12] +
# layout[8] + page_header_score[2] + section_header_flag[1] = 791.
FEATURE_DIM: int = 768 + _features._PARTICLE_DIM + 8 + 2 + 1


_DEFAULT_REDIS_KEY = 'skol:classifier:model:v4_layout'
_DEFAULT_META_KEY = 'skol:classifier:model:v4_layout:meta'
_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LayoutCRF(nn.Module):
    """Linear-chain CRF over the 8 Pass-1 layout labels.

    Variable-length sequences are supported via the caller-supplied
    ``mask`` (True at valid positions).  No special start/end tokens
    — torchcrf handles boundary transitions internally.
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
        """Negative log-likelihood loss for one batch.

        Args:
            features: ``(batch, seq, feature_dim)`` float32.
            tags:     ``(batch, seq)`` int64.
            mask:     ``(batch, seq)`` bool (True = valid position).

        Returns:
            Scalar tensor — mean NLL across the batch.
        """
        emissions = self.emission(features)
        return -self.crf(
            emissions, tags, mask=mask, reduction='mean',
        )

    def decode(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[int]]:
        """Viterbi decode.

        Returns one list of label indices per batch row, length equal
        to that row's mask sum.
        """
        emissions = self.emission(features)
        return self.crf.decode(emissions, mask=mask)


# ---------------------------------------------------------------------------
# Serialize / deserialize
# ---------------------------------------------------------------------------


def _default_metadata(model: LayoutCRF) -> Dict[str, Any]:
    return {
        'schema_version': _SCHEMA_VERSION,
        'feature_dim': model.feature_dim,
        'n_labels': model.n_labels,
        'label_map': dict(LABEL_TO_INDEX),
        'torch_version': torch.__version__,
    }


def serialize(
    model: LayoutCRF,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bytes, bytes]:
    """Return ``(state_dict_bytes, metadata_json_bytes)`` ready to
    write to Redis under the two-key bundle.

    Caller's ``metadata`` is merged on top of a default set so the
    schema fields (``feature_dim``, ``n_labels``, ``label_map``) are
    always populated.
    """
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
) -> Tuple[LayoutCRF, Dict[str, Any]]:
    """Inverse of :func:`serialize`.

    Reads metadata first, instantiates a LayoutCRF with the right
    dims, then loads weights.  Raises :class:`RuntimeError` if the
    metadata claims dims that don't match the state_dict shapes.
    """
    metadata: Dict[str, Any] = json.loads(
        metadata_json_bytes.decode('utf-8'),
    )
    feature_dim = int(metadata.get('feature_dim', FEATURE_DIM))
    n_labels = int(metadata.get('n_labels', N_LABELS))

    model = LayoutCRF(feature_dim=feature_dim, n_labels=n_labels)
    state_dict = torch.load(
        io.BytesIO(state_dict_bytes),
        map_location=map_location,
        weights_only=True,
    )
    # ``load_state_dict`` raises on shape mismatch — that's the
    # "fail loudly" behaviour test_deserialize_rejects_mismatched_dims
    # is locking in.
    model.load_state_dict(state_dict)
    return model, metadata


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


def save_to_redis(
    model: LayoutCRF,
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
) -> Tuple[LayoutCRF, Dict[str, Any]]:
    """Inverse of :func:`save_to_redis`."""
    state_bytes = redis_client.get(key)
    meta_bytes = redis_client.get(meta_key)
    if state_bytes is None or meta_bytes is None:
        raise RuntimeError(
            f'Missing v4_layout model at {key!r} / {meta_key!r}'
        )
    return deserialize(
        state_bytes, meta_bytes, map_location=map_location,
    )
