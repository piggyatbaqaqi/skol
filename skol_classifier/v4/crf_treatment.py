"""v4 Pass-2 treatment CRF.

Linear-chain CRF over the 12 treatment labels listed in
docs/v4_classifier_plan.md §Label-space partition.  Same architecture
as Pass 1's :class:`skol_classifier.v4.crf_layout.LayoutCRF`; only
the label vocab + Redis key differ.

Trained against the **non-layout subsequence** of each doc.  Pass 1
identifies which lines are layout artefacts; this CRF labels the
remaining lines with one of the 12 treatment categories.  See
:mod:`bin.train_crf_treatment` for the per-doc filtering logic.

The default Redis bundle:

    skol:classifier:model:v4_treatment           state_dict bytes
    skol:classifier:model:v4_treatment:meta      JSON metadata
"""
from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchcrf import CRF  # type: ignore[import-untyped]

from skol_classifier.v4 import features as _features


# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------

TREATMENT_LABELS: Tuple[str, ...] = (
    'Nomenclature',
    'Description',
    'Diagnosis',
    'Etymology',
    'Materials-examined',
    'Materials-and-methods',
    'Type-designation',
    'Biology',
    'Phylogeny',
    'New-combinations',
    'Notes',
    'Misc-exposition',
)
N_LABELS = len(TREATMENT_LABELS)
LABEL_TO_INDEX: Dict[str, int] = {
    label: idx for idx, label in enumerate(TREATMENT_LABELS)
}
INDEX_TO_LABEL: Tuple[str, ...] = TREATMENT_LABELS
MISC_EXPOSITION_INDEX = LABEL_TO_INDEX['Misc-exposition']


# Default feature dim — same 791 as Pass 1.
FEATURE_DIM: int = 768 + _features._PARTICLE_DIM + 8 + 2 + 1


_DEFAULT_REDIS_KEY = 'skol:classifier:model:v4_treatment'
_DEFAULT_META_KEY = 'skol:classifier:model:v4_treatment:meta'
_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TreatmentCRF(nn.Module):
    """Linear-chain CRF over the 12 Pass-2 treatment labels.

    Variable-length sequences are supported via the caller-supplied
    ``mask`` (True at valid positions).  pytorch-crf requires the
    mask to be contiguous (valid positions before invalid), which
    matches how the trainer prepares Pass-2 inputs: lines are
    filtered to the non-layout subsequence before reaching the CRF.
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


def _default_metadata(model: TreatmentCRF) -> Dict[str, Any]:
    return {
        'schema_version': _SCHEMA_VERSION,
        'feature_dim': model.feature_dim,
        'n_labels': model.n_labels,
        'label_map': dict(LABEL_TO_INDEX),
        'torch_version': torch.__version__,
    }


def serialize(
    model: TreatmentCRF,
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
) -> Tuple[TreatmentCRF, Dict[str, Any]]:
    """Inverse of :func:`serialize`.  Mismatched dims raise from
    ``load_state_dict``."""
    metadata: Dict[str, Any] = json.loads(
        metadata_json_bytes.decode('utf-8'),
    )
    feature_dim = int(metadata.get('feature_dim', FEATURE_DIM))
    n_labels = int(metadata.get('n_labels', N_LABELS))

    model = TreatmentCRF(feature_dim=feature_dim, n_labels=n_labels)
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
    model: TreatmentCRF,
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
) -> Tuple[TreatmentCRF, Dict[str, Any]]:
    """Inverse of :func:`save_to_redis`."""
    state_bytes = redis_client.get(key)
    meta_bytes = redis_client.get(meta_key)
    if state_bytes is None or meta_bytes is None:
        raise RuntimeError(
            f'Missing v4_treatment model at {key!r} / {meta_key!r}'
        )
    return deserialize(
        state_bytes, meta_bytes, map_location=map_location,
    )
