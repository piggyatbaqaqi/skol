"""v4 end-to-end predictor (Step 5).

Per-doc inference pipeline:

    plaintext
       │
       ▼
    feature assembler  (skol_classifier.v4.features.build_line_features)
       │      (per-line 791-d vectors: sbert + particles + layout + ph + sh)
       ▼
    Pass 1 — LayoutCRF.decode  →  8 layout labels per line
       │
       ▼  (mask = layout_label == 'Other')
    Pass 2 — TreatmentCRF.decode over the non-layout subsequence
       │      (pytorch-crf needs contiguous masks, so we filter →
       │       decode → splice back to the original line positions)
       ▼
    merge layout + treatment into per-line YEDDA tags
       │
       ▼  (break runs on label change OR blank line; drop blanks)
    coalesce  →  List[TaggedBlock]
       │
       ▼
    ingestors.yedda_tags.tagged_blocks_to_yedda
       │
       ▼
    article.txt.ann text

This module exposes two entry points:

* :func:`predict_doc` — the full pipeline; takes raw plaintext +
  parsed spans + page-headers + both CRFs + a callable
  sbert_lookup.  Used by ``bin/predict_v4.py``.
* :func:`predict_from_features` — the decode + emit half; takes
  pre-built feature tensors.  Used by the unit tests so they don't
  need an SBERT cache.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ingestors.spans import Span  # noqa: E402
from ingestors.yedda_tags import Tag, TaggedBlock, tagged_blocks_to_yedda  # noqa: E402

from skol_classifier.v4 import features as _features  # noqa: E402
from skol_classifier.v4.crf_layout import (  # noqa: E402
    INDEX_TO_LABEL as LAYOUT_LABELS,
    LayoutCRF,
    OTHER_INDEX as LAYOUT_OTHER_INDEX,
)
from skol_classifier.v4.crf_treatment import (  # noqa: E402
    INDEX_TO_LABEL as TREATMENT_LABELS,
    TreatmentCRF,
)


SbertLookup = Callable[[str], Optional[np.ndarray]]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def predict_doc(
    plaintext: str,
    spans_dict: Dict[str, Any],
    page_headers_dict: Dict[str, Any],
    layout_crf: LayoutCRF,
    treatment_crf: TreatmentCRF,
    sbert_lookup: SbertLookup,
    *,
    device: str = 'cpu',
) -> Tuple[List[str], str]:
    """Run feature assembly + Pass 1 + Pass 2 + coalesce + emit.

    Returns ``(per_line_yedda_tags, ann_text)``.  Layout-line
    positions carry the Pass-1 YEDDA tag (Page-header /
    Figure-caption / ...); non-layout positions carry the Pass-2
    YEDDA tag (Nomenclature / Description / ...).

    Raises ``ValueError`` if either CRF's ``feature_dim`` disagrees
    with the active ``features.FEATURE_DIM`` — protects against the
    "trained with a different SBERT model" failure mode.
    """
    _check_feature_dims(layout_crf, treatment_crf)

    lines = plaintext.split('\n')
    if not lines or (len(lines) == 1 and lines[0] == ''):
        return ([''] if plaintext == '' else []), ''

    spans = _parse_spans(spans_dict)
    line_starts = _features.compute_line_starts(lines)

    layout_feats = np.zeros(
        (len(lines), layout_crf.feature_dim), dtype=np.float32,
    )
    for i, line in enumerate(lines):
        lf = _features.build_line_features(
            line_text=line, line_index=i,
            doc_lines=lines, spans=spans,
            page_headers=page_headers_dict,
            sbert_lookup=sbert_lookup,
            line_starts=line_starts,
        )
        layout_feats[i] = lf.concat()
    # Pass 2 uses the same feature width — single per-line vector.
    treatment_feats = layout_feats
    return predict_from_features(
        lines,
        layout_feats, treatment_feats,
        layout_crf, treatment_crf,
        device=device,
    )


def predict_from_features(
    lines: Sequence[str],
    layout_features: np.ndarray,
    treatment_features: np.ndarray,
    layout_crf: LayoutCRF,
    treatment_crf: TreatmentCRF,
    *,
    device: str = 'cpu',
) -> Tuple[List[str], str]:
    """Decode + splice + coalesce + emit, given pre-built features.

    Both feature tensors must be shape ``(len(lines), feature_dim)``.
    ``layout_features`` and ``treatment_features`` are usually the
    same array (Pass 1 and Pass 2 share the 791-d vector); the test
    suite passes width-matched one-hots that differ per pass.
    """
    if len(lines) == 0:
        return [], ''

    layout_indices = _crf_decode_one(
        layout_crf, layout_features, device,
    )
    layout_arr = np.asarray(layout_indices, dtype=np.int64)
    content_mask = layout_arr == LAYOUT_OTHER_INDEX

    treatment_full = np.full(len(lines), -1, dtype=np.int64)
    if bool(content_mask.any()):
        treatment_indices = _crf_decode_one(
            treatment_crf,
            treatment_features[content_mask],
            device,
        )
        treatment_full[content_mask] = treatment_indices

    per_line_tags: List[str] = []
    for li in range(len(lines)):
        if content_mask[li]:
            per_line_tags.append(TREATMENT_LABELS[treatment_full[li]])
        else:
            per_line_tags.append(LAYOUT_LABELS[layout_arr[li]])

    blocks = _coalesce_blocks(list(lines), per_line_tags)
    ann_text = tagged_blocks_to_yedda(blocks) if blocks else ''
    return per_line_tags, ann_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_feature_dims(
    layout_crf: LayoutCRF, treatment_crf: TreatmentCRF,
) -> None:
    expected = _features.FEATURE_DIM
    if layout_crf.feature_dim != expected:
        raise ValueError(
            f'Layout CRF feature_dim {layout_crf.feature_dim} does '
            f'not match features.FEATURE_DIM={expected}.  '
            'Retrain Pass 1 against the current feature assembler.'
        )
    if treatment_crf.feature_dim != expected:
        raise ValueError(
            f'Treatment CRF feature_dim {treatment_crf.feature_dim} '
            f'does not match features.FEATURE_DIM={expected}.  '
            'Retrain Pass 2 against the current feature assembler.'
        )


def _parse_spans(spans_dict: Dict[str, Any]) -> List[Span]:
    """Reconstruct ``Span`` objects from the v1 envelope dict.

    Mirrors ``bin/train_crf_layout._parse_spans`` but takes the
    already-parsed dict (the predictor's caller has the JSON
    blob in hand and parses once)."""
    spans: List[Span] = []
    for s in spans_dict.get('spans', []) or []:
        spans.append(Span(
            start=s['start'], end=s['end'],
            label=s['label'], text=s['text'],
            source=s.get('source', 'unknown'),
            confidence=s.get('confidence', 1.0),
            metadata=s.get('metadata') or {},
        ))
    return spans


def _crf_decode_one(
    crf: Any, features_np: np.ndarray, device: str,
) -> List[int]:
    """Single-row Viterbi decode helper.  Returns the index list."""
    t = torch.from_numpy(features_np).unsqueeze(0).to(device)
    mask = torch.ones(t.shape[:2], dtype=torch.bool, device=device)
    decoded: List[List[int]] = crf.decode(t, mask)
    return decoded[0]


def _coalesce_blocks(
    lines: List[str], tags: List[str],
) -> List[TaggedBlock]:
    """Walk ``(line, tag)`` pairs; break runs on label change OR
    blank line (after strip); drop blank lines entirely; emit one
    ``TaggedBlock`` per non-empty run with its lines joined by
    ``\\n``.  Tag strings are mapped to ``Tag`` enum instances; an
    unknown tag becomes ``Tag.MISC_EXPOSITION`` (the catch-all)
    rather than raising — keeps the emitter forgiving against future
    label vocab additions."""
    out: List[TaggedBlock] = []
    buf_lines: List[str] = []
    buf_tag: Optional[str] = None
    for line, tag in zip(lines, tags):
        if not line.strip():
            if buf_lines and buf_tag is not None:
                out.append(TaggedBlock(
                    text='\n'.join(buf_lines),
                    tag=_to_tag_enum(buf_tag),
                ))
            buf_lines, buf_tag = [], None
            continue
        if buf_tag is None or tag == buf_tag:
            buf_lines.append(line)
            buf_tag = tag
        else:
            out.append(TaggedBlock(
                text='\n'.join(buf_lines),
                tag=_to_tag_enum(buf_tag),
            ))
            buf_lines, buf_tag = [line], tag
    if buf_lines and buf_tag is not None:
        out.append(TaggedBlock(
            text='\n'.join(buf_lines),
            tag=_to_tag_enum(buf_tag),
        ))
    return out


def _to_tag_enum(label: str) -> Tag:
    """Project a YEDDA-label string to its ``Tag`` enum instance.

    Falls back to ``Tag.MISC_EXPOSITION`` on any unrecognised input
    so the emitter is robust against label-vocab drift."""
    try:
        return Tag(label)
    except ValueError:
        return Tag.MISC_EXPOSITION
