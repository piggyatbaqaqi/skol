"""Every v4 CRF checkpoint declares its label set, and loading refuses
a mismatch (E6-0, Experiment 6 plan §8.5).

Covers the three CRFs: crf_layout (projection ``layout``),
crf_treatment (``treatment``) and crf_single (``active``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from skol_classifier.label_set import (  # noqa: E402
    LabelSetMismatch,
    load_label_set,
)
from skol_classifier.v4 import (  # noqa: E402
    crf_layout,
    crf_single,
    crf_treatment,
)

CRFS: Dict[str, Tuple[ModuleType, str]] = {
    'layout': (crf_layout, 'LayoutCRF'),
    'treatment': (crf_treatment, 'TreatmentCRF'),
    'active': (crf_single, 'SingleCRF'),
}


def _bundle(projection: str) -> Tuple[ModuleType, bytes, Dict[str, Any]]:
    module, cls_name = CRFS[projection]
    model = getattr(module, cls_name)(feature_dim=4)
    state, meta = module.serialize(model)
    return module, state, json.loads(meta)


@pytest.mark.parametrize('projection', sorted(CRFS))
def test_serialize_declares_v4(projection: str) -> None:
    _, _, meta = _bundle(projection)
    v4 = load_label_set('v4')
    for key, value in v4.declaration(projection).items():
        assert meta[key] == value


@pytest.mark.parametrize('projection', sorted(CRFS))
def test_deserialize_refuses_another_label_set(projection: str) -> None:
    module, state, meta = _bundle(projection)
    meta['label_set'] = 'v6'
    with pytest.raises(LabelSetMismatch):
        module.deserialize(state, json.dumps(meta).encode('utf-8'))


@pytest.mark.parametrize('projection', sorted(CRFS))
def test_deserialize_accepts_a_pre_e6_checkpoint(projection: str) -> None:
    # Checkpoints already in Redis carry label_map but no declaration.
    module, state, meta = _bundle(projection)
    for key in ('label_set', 'label_set_digest', 'projection'):
        meta.pop(key, None)
    _, loaded = module.deserialize(state, json.dumps(meta).encode('utf-8'))
    assert loaded['label_map'] == meta['label_map']
