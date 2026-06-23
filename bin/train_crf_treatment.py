#!/usr/bin/env python3
"""Train the v4 Pass-2 treatment CRF.

Pass 2 of v4's two-pass design.  Same shape as
``bin/train_crf_layout.py`` but over the 12 treatment YEDDA labels
(per docs/v4_classifier_plan.md §Label-space partition) and crucially
trains on the **non-layout subsequence** of each doc: lines whose
Pass-1 oracle label is one of the 7 layout tags are removed before
the CRF sees them.

pytorch-crf's mask only supports contiguous valid positions, so we
filter to the content subsequence at data-prep time and feed
pytorch-crf an all-True mask of that length.

Pre-requisite: the operator runs
``bin/annotate_v4 --database <SAME-DB>`` once to populate
``article.spans.v4.json`` + ``article.page-headers.json``.  Docs
missing either v4 attachment are counted + skipped.

Default source DB is ``skol_training_v3_combined_no_golden`` per
plan §4.B.  The §6.C training-corpus ablation
(``v4_pass2_hand`` vs ``v4_pass2_combined``) is an operational
invocation of this trainer with ``--source-db`` + ``--redis-key``
overrides, not a separate script.

Helpers (``split_docs``, ``inverse_frequency_weights``,
``make_sbert_lookup``, the data-loading bits) are duplicated from
``bin/train_crf_layout.py`` per user choice — once both trainers
exist they get factored into ``bin/_crf_trainer_lib.py`` as a
follow-up.

Usage::

    bin/train_crf_treatment --source-db skol_training_v3_combined_no_golden \\
        --epochs 20 --verbosity 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple,
)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import couchdb  # type: ignore[import]  # noqa: E402

from env_config import (  # type: ignore[import]  # noqa: E402
    common_parser, create_redis_client, get_env_config,
)
from ingestors.extract_plaintext import (  # noqa: E402
    plaintext_from_pdf, plaintext_from_yedda,
)
from ingestors.spans import Span  # noqa: E402

from skol_classifier.v4.crf_layout import (  # noqa: E402
    LABEL_TO_INDEX as LAYOUT_LABEL_TO_INDEX,
    OTHER_INDEX as LAYOUT_OTHER_INDEX,
    load_from_redis as load_layout_from_redis,
)
from skol_classifier.v4.crf_treatment import (  # noqa: E402
    FEATURE_DIM,
    N_LABELS,
    TreatmentCRF,
    save_to_redis,
)
from skol_classifier.v4.features import (  # noqa: E402
    build_line_features,
    compute_line_starts,
)
from skol_classifier.v4.labels import (  # noqa: E402
    build_label_sequence,
    build_treatment_label_sequence,
)


_DEFAULT_REDIS_KEY = 'skol:classifier:model:v4_treatment'
_DEFAULT_META_KEY = 'skol:classifier:model:v4_treatment:meta'
_SPANS_ATTACHMENT = 'article.spans.v4.json'
_PAGE_HEADERS_ATTACHMENT = 'article.page-headers.json'
_ANN_ATTACHMENT = 'article.txt.ann'
_PLAINTEXT_ATTACHMENT = 'article.txt'


SbertLookup = Callable[[str], Optional[np.ndarray]]


# ---------------------------------------------------------------------------
# Pure helpers (duplicated from train_crf_layout.py)
# ---------------------------------------------------------------------------


def split_docs(
    doc_lengths: Sequence[Tuple[str, int]],
    *,
    dev_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Quartile-stratified train/dev split by doc length.  Identical
    to the v4 Pass-1 implementation."""
    rng = random.Random(seed)
    sorted_by_len = sorted(doc_lengths, key=lambda t: t[1])
    n = len(sorted_by_len)
    train: List[str] = []
    dev: List[str] = []
    for q in range(4):
        lo = (n * q) // 4
        hi = (n * (q + 1)) // 4
        quartile = list(sorted_by_len[lo:hi])
        rng.shuffle(quartile)
        n_dev = round(len(quartile) * dev_fraction)
        for i, (doc_id, _) in enumerate(quartile):
            (dev if i < n_dev else train).append(doc_id)
    train.sort()
    dev.sort()
    return train, dev


def inverse_frequency_weights(
    label_counts: np.ndarray,
    n_labels: int = N_LABELS,
) -> np.ndarray:
    """Inverse-frequency class weights over the Pass-2 label space."""
    weights = np.ones(n_labels, dtype=np.float64)
    total = float(label_counts.sum())
    if total <= 0.0:
        return weights
    for i in range(n_labels):
        count = float(label_counts[i])
        if count > 0.0:
            weights[i] = total / (n_labels * count)
    return weights


def make_sbert_lookup(
    redis_client: Any,
    model_tag: str = 'mpnet',
    dim: int = 768,
) -> SbertLookup:
    """SBERT cache lookup callable — same as Pass-1's trainer."""
    prefix = f'skol:sbert:{model_tag}:'

    def _lookup(line: str) -> Optional[np.ndarray]:
        key = prefix + hashlib.sha256(
            line.encode('utf-8'),
        ).hexdigest()
        raw = redis_client.get(key)
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float32).copy()
    return _lookup


# ---------------------------------------------------------------------------
# Per-doc data extraction (Pass-2-specific masking)
# ---------------------------------------------------------------------------


def _read_attachment_bytes(
    db: Any, doc_id: str, name: str,
) -> Optional[bytes]:
    try:
        raw = db.get_attachment(doc_id, name)
        if raw is None:
            return None
        if hasattr(raw, 'read'):
            raw = raw.read()
        if isinstance(raw, str):
            return raw.encode('utf-8')
        return raw  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001
        return None


def _read_attachment_text(
    db: Any, doc_id: str, name: str,
) -> Optional[str]:
    raw = _read_attachment_bytes(db, doc_id, name)
    if raw is None:
        return None
    return raw.decode('utf-8', errors='ignore')


def _load_plaintext(
    db: Any, doc_id: str,
) -> Optional[str]:
    text = _read_attachment_text(db, doc_id, _PLAINTEXT_ATTACHMENT)
    if text is not None:
        return text
    pdf_bytes = _read_attachment_bytes(db, doc_id, 'article.pdf')
    if pdf_bytes is not None:
        return plaintext_from_pdf(pdf_bytes)
    ann = _read_attachment_text(db, doc_id, _ANN_ATTACHMENT)
    if ann is not None:
        return plaintext_from_yedda(ann)
    return None


def _parse_spans(spans_bytes: bytes) -> List[Span]:
    blob = json.loads(spans_bytes.decode('utf-8'))
    spans: List[Span] = []
    for s in blob.get('spans', []):
        spans.append(Span(
            start=s['start'], end=s['end'],
            label=s['label'], text=s['text'],
            source=s.get('source', 'unknown'),
            confidence=s.get('confidence', 1.0),
            metadata=s.get('metadata') or {},
        ))
    return spans


def _prepare_doc_pass2(
    db: Any,
    doc_id: str,
    sbert_lookup: SbertLookup,
    *,
    use_predicted_layout: bool = False,
    layout_crf: Any = None,
    device: str = 'cpu',
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build (features_array, labels_array) for one Pass-2 doc.

    The default (oracle) path computes Pass-1 labels from the doc's
    YEDDA blocks via ``build_label_sequence``.  Step 7.δ's
    ``use_predicted_layout=True`` mode replaces that with the
    output of ``layout_crf.decode()`` — so Pass-2 is trained on the
    same noisy subsequence it will see at inference, closing the
    train/test distribution gap that exposure-bias measures.

    In either case, lines whose layout label is one of the 7 layout
    tags are filtered out; what reaches the CRF is the contiguous
    content subsequence of the doc.

    Returns None when:
    - either v4 attachment is missing
    - plaintext / ann_text can't be loaded
    - no content lines survive Pass-1 filtering

    Raises ``ValueError`` if ``use_predicted_layout=True`` but
    ``layout_crf`` is missing — the caller must load Pass-1 from
    Redis once and pass it in.
    """
    if use_predicted_layout and layout_crf is None:
        raise ValueError(
            'use_predicted_layout=True requires layout_crf '
            '(load it via crf_layout.load_from_redis).'
        )

    spans_bytes = _read_attachment_bytes(
        db, doc_id, _SPANS_ATTACHMENT,
    )
    ph_bytes = _read_attachment_bytes(
        db, doc_id, _PAGE_HEADERS_ATTACHMENT,
    )
    if spans_bytes is None or ph_bytes is None:
        return None
    plaintext = _load_plaintext(db, doc_id)
    ann_text = _read_attachment_text(db, doc_id, _ANN_ATTACHMENT)
    if plaintext is None or ann_text is None:
        return None

    spans = _parse_spans(spans_bytes)
    try:
        page_headers = json.loads(ph_bytes.decode('utf-8'))
    except Exception:  # noqa: BLE001
        page_headers = {'per_line_confidence': []}
    lines = plaintext.split('\n')
    line_starts = compute_line_starts(lines)

    treatment_seq = np.asarray(
        build_treatment_label_sequence(plaintext, ann_text),
        dtype=np.int64,
    )
    if treatment_seq.shape[0] != len(lines):
        return None

    # Build features for ALL lines first; filtering after avoids
    # the awkward double pass through the line_starts table.
    # Pass-1 decode (exposure-bias mode) also needs the full
    # feature tensor, so we always build it before deciding the mask.
    features_full = np.zeros(
        (len(lines), FEATURE_DIM), dtype=np.float32,
    )
    for i, line in enumerate(lines):
        feats = build_line_features(
            line_text=line, line_index=i,
            doc_lines=lines, spans=spans,
            page_headers=page_headers,
            sbert_lookup=sbert_lookup,
            line_starts=line_starts,
        )
        features_full[i] = feats.concat()

    if use_predicted_layout:
        x = torch.from_numpy(features_full).unsqueeze(0).to(device)
        mask = torch.ones(
            1, len(lines), dtype=torch.bool, device=device,
        )
        layout_seq = np.asarray(
            layout_crf.decode(x, mask)[0], dtype=np.int64,
        )
    else:
        layout_seq = np.asarray(
            build_label_sequence(plaintext, ann_text), dtype=np.int64,
        )
    if layout_seq.shape[0] != len(lines):
        return None

    pass2_mask = layout_seq == LAYOUT_OTHER_INDEX
    n_content = int(pass2_mask.sum())
    if n_content == 0:
        return None

    features_filtered = features_full[pass2_mask]
    labels_filtered = treatment_seq[pass2_mask]
    return features_filtered, labels_filtered


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _per_label_counts(label_arrays: Iterable[np.ndarray]) -> np.ndarray:
    counts = np.zeros(N_LABELS, dtype=np.float64)
    for labels in label_arrays:
        for c in range(N_LABELS):
            counts[c] += float((labels == c).sum())
    return counts


def _evaluate(
    model: TreatmentCRF,
    prepared: List[Tuple[str, np.ndarray, np.ndarray]],
    *,
    device: str,
) -> Tuple[float, List[float]]:
    """Macro F1 + per-label F1 over Pass-2 dev docs."""
    model.eval()
    tp = np.zeros(N_LABELS, dtype=np.int64)
    fp = np.zeros(N_LABELS, dtype=np.int64)
    fn = np.zeros(N_LABELS, dtype=np.int64)
    with torch.no_grad():
        for _doc_id, features, labels in prepared:
            x = torch.from_numpy(features).unsqueeze(0).to(device)
            mask = torch.ones(
                1, features.shape[0], dtype=torch.bool, device=device,
            )
            pred = model.decode(x, mask)[0]
            for true_lbl, pred_lbl in zip(labels.tolist(), pred):
                if true_lbl == pred_lbl:
                    tp[true_lbl] += 1
                else:
                    fp[pred_lbl] += 1
                    fn[true_lbl] += 1
    model.train()

    per_label_f1: List[float] = []
    for c in range(N_LABELS):
        precision = (
            tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        )
        recall = (
            tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )
        per_label_f1.append(float(f1))
    macro_f1 = float(np.mean(per_label_f1))
    return macro_f1, per_label_f1


def train_one_run(
    db: Any,
    redis_client: Any,
    *,
    doc_ids: Sequence[str],
    sbert_lookup: SbertLookup,
    epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
    dev_fraction: float = 0.2,
    device: str = 'cpu',
    redis_key: str = _DEFAULT_REDIS_KEY,
    meta_key: str = _DEFAULT_META_KEY,
    sbert_model: str = 'mpnet',
    source_db: str = '',
    dry_run: bool = False,
    skip_existing: bool = False,
    verbosity: int = 1,
    use_predicted_layout: bool = False,
    layout_crf: Any = None,
) -> Dict[str, Any]:
    """End-to-end Pass-2 training pass."""
    counts: Dict[str, Any] = {
        'scanned_docs': len(doc_ids),
        'skipped_no_spans': 0,
        'trained_docs': 0,
        'dev_docs': 0,
        'short_circuited': 0,
        'dev_macro_f1_per_epoch': [],
    }

    if skip_existing:
        if (redis_client.get(redis_key) is not None
                and redis_client.get(meta_key) is not None):
            counts['short_circuited'] = 1
            if verbosity >= 1:
                print(
                    f'  ✓ Redis already has {redis_key!r} '
                    '+ meta — skip training (use --force to retrain).'
                )
            return counts

    prepared: List[Tuple[str, np.ndarray, np.ndarray]] = []
    doc_lengths: List[Tuple[str, int]] = []
    for doc_id in doc_ids:
        item = _prepare_doc_pass2(
            db, doc_id, sbert_lookup,
            use_predicted_layout=use_predicted_layout,
            layout_crf=layout_crf,
            device=device,
        )
        if item is None:
            counts['skipped_no_spans'] += 1
            if verbosity >= 2:
                print(f'  skip {doc_id}: missing v4 attachments '
                      'or no content lines')
            continue
        features, labels = item
        prepared.append((doc_id, features, labels))
        doc_lengths.append((doc_id, features.shape[0]))

    if not prepared:
        if verbosity >= 1:
            print('  ✗ No docs prepared; aborting.')
        return counts

    train_ids, dev_ids = split_docs(
        doc_lengths, dev_fraction=dev_fraction, seed=seed,
    )
    train_ids_set = set(train_ids)
    dev_ids_set = set(dev_ids)
    dev_set = [t for t in prepared if t[0] in dev_ids_set]
    train_list = [t for t in prepared if t[0] in train_ids_set]
    counts['trained_docs'] = len(train_list)
    counts['dev_docs'] = len(dev_set)

    class_weights = inverse_frequency_weights(
        _per_label_counts(l for _, _, l in train_list),
    )

    if verbosity >= 1:
        print(
            f'  prepared={len(prepared)} '
            f'train={len(train_list)} dev={len(dev_set)} '
            f'epochs={epochs} lr={lr} seed={seed} device={device}'
        )

    torch.manual_seed(seed)
    model = TreatmentCRF().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    started_at = datetime.now(timezone.utc).isoformat(timespec='seconds')

    macro_f1 = 0.0
    per_label_f1 = [0.0] * N_LABELS
    for epoch in range(epochs):
        rng.shuffle(train_list)
        loss_sum = 0.0
        for _doc_id, features, labels in train_list:
            x = torch.from_numpy(features).unsqueeze(0).to(device)
            y = torch.from_numpy(labels).unsqueeze(0).to(device)
            mask = torch.ones(
                1, features.shape[0], dtype=torch.bool, device=device,
            )
            opt.zero_grad()
            base_loss = model(x, y, mask)
            doc_weight = float(np.mean(class_weights[labels]))
            loss = base_loss * doc_weight
            loss.backward()
            opt.step()
            loss_sum += float(base_loss.detach().cpu())
        if dev_set:
            macro_f1, per_label_f1 = _evaluate(
                model, dev_set, device=device,
            )
        counts['dev_macro_f1_per_epoch'].append(macro_f1)
        if verbosity >= 1:
            print(
                f'  epoch {epoch+1:>2d}/{epochs}: '
                f'train_loss={loss_sum / max(1, len(train_list)):.3f}  '
                f'dev_macro_f1={macro_f1:.3f}'
            )

    finished_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    counts['final_dev_macro_f1'] = macro_f1
    counts['final_dev_per_label_f1'] = per_label_f1

    metadata: Dict[str, Any] = {
        'training': {
            'epochs': epochs, 'lr': lr, 'seed': seed,
            'source_db': source_db,
            'sbert_model': sbert_model,
            'dev_fraction': dev_fraction,
            'device': device,
            'train_doc_count': len(train_list),
            'dev_doc_count': len(dev_set),
            'dev_macro_f1_per_epoch':
                list(counts['dev_macro_f1_per_epoch']),
            'final_dev_macro_f1': macro_f1,
            'final_dev_per_label_f1': per_label_f1,
            'class_weights': class_weights.tolist(),
            'started_at': started_at,
            'finished_at': finished_at,
        },
    }

    if dry_run:
        if verbosity >= 1:
            print(
                f'  *** DRY RUN: skipping Redis write to {redis_key!r}'
            )
    else:
        save_to_redis(
            model, redis_client,
            key=redis_key, meta_key=meta_key, metadata=metadata,
        )
        if verbosity >= 1:
            print(
                f'  ✓ wrote {redis_key!r} + {meta_key!r}'
            )

    counts['metadata'] = metadata
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _open_db(config: Dict[str, Any], db_name: str) -> Any:
    server = couchdb.Server(config['couchdb_url'])
    username = config.get('couchdb_username')
    password = config.get('couchdb_password')
    if username and password:
        server.resource.credentials = (username, password)
    return server[db_name]


def _iter_doc_ids(
    db: Any, limit: Optional[int] = None,
) -> List[str]:
    out: List[str] = []
    for row in db.view('_all_docs'):
        rid = str(row.id)
        if rid.startswith('_'):
            continue
        out.append(rid)
        if limit is not None and len(out) >= limit:
            break
    return out


def _resolve_device(choice: str) -> str:
    if choice == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return choice


def main() -> int:
    parser = argparse.ArgumentParser(
        parents=[common_parser()],
        description='Train the v4 Pass-2 treatment CRF.',
    )
    parser.add_argument(
        '--source-db', default=None,
        help=(
            'CouchDB DB (default: skol_training_v3_combined_no_golden '
            'per plan §4.B).'
        ),
    )
    parser.add_argument(
        '--sbert-model', default='mpnet',
        help='SBERT cache namespace tag (default: mpnet).',
    )
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dev-fraction', type=float, default=0.2)
    parser.add_argument(
        '--device',
        choices=('auto', 'cpu', 'cuda'),
        default='auto',
    )
    parser.add_argument('--redis-key', default=_DEFAULT_REDIS_KEY)
    parser.add_argument(
        '--redis-meta-key', default=_DEFAULT_META_KEY,
    )
    parser.add_argument(
        '--use-predicted-layout',
        dest='use_predicted_layout',
        action='store_true',
        help=(
            'Step 7.δ exposure-bias mode: build the per-line '
            'Pass-1 sequence by running --pass1-key\'s CRF over '
            'the doc, instead of from the YEDDA oracle.  Trains '
            'Pass-2 on sequences that match what it will see at '
            'inference time.'
        ),
    )
    parser.add_argument(
        '--pass1-key', dest='pass1_key',
        default='skol:classifier:model:v4_layout',
        help=(
            'Pass-1 (LayoutCRF) Redis state-key.  Only loaded '
            'when --use-predicted-layout is on.'
        ),
    )
    args = parser.parse_args()

    config = get_env_config(cli_args=args)
    verbosity = int(config.get('verbosity', 1) or 0)
    dry_run = bool(config.get('dry_run', False))
    force = bool(config.get('force', False))
    skip_existing = not force
    limit_raw = config.get('limit')
    limit = (
        int(limit_raw) if limit_raw not in (None, '') else None
    )

    db_name = (
        args.source_db
        or 'skol_training_v3_combined_no_golden'
    )
    try:
        db = _open_db(config, db_name)
    except Exception as exc:  # noqa: BLE001
        print(f'✗ cannot open {db_name!r}: {exc}', file=sys.stderr)
        return 1
    redis_client = create_redis_client(decode_responses=False)
    sbert_lookup = make_sbert_lookup(
        redis_client, model_tag=args.sbert_model,
    )

    device = _resolve_device(args.device)
    if verbosity >= 1:
        print(f'train_crf_treatment — db={db_name} device={device}')

    layout_crf: Any = None
    if args.use_predicted_layout:
        try:
            layout_crf, _layout_meta = load_layout_from_redis(
                redis_client,
                key=args.pass1_key,
                meta_key=f'{args.pass1_key}:meta',
                map_location=device,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f'✗ cannot load Pass-1 CRF from {args.pass1_key!r}: '
                f'{exc}',
                file=sys.stderr,
            )
            return 1
        layout_crf.to(device)
        layout_crf.eval()
        if verbosity >= 1:
            print(
                f'  --use-predicted-layout ON — Pass-1 loaded '
                f'from {args.pass1_key!r}',
            )

    doc_ids = _iter_doc_ids(db, limit=limit)
    if verbosity >= 1:
        print(f'  scanning {len(doc_ids)} docs')

    counts = train_one_run(
        db, redis_client,
        doc_ids=doc_ids,
        sbert_lookup=sbert_lookup,
        epochs=args.epochs, lr=args.lr,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        device=device,
        redis_key=args.redis_key,
        meta_key=args.redis_meta_key,
        sbert_model=args.sbert_model,
        source_db=db_name,
        dry_run=dry_run,
        skip_existing=skip_existing,
        verbosity=verbosity,
        use_predicted_layout=args.use_predicted_layout,
        layout_crf=layout_crf,
    )

    if verbosity >= 1:
        print()
        for k in (
            'scanned_docs', 'skipped_no_spans',
            'trained_docs', 'dev_docs',
            'short_circuited', 'final_dev_macro_f1',
        ):
            if k in counts:
                print(f'  {k:<25} {counts[k]}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
