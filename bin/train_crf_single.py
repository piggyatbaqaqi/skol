#!/usr/bin/env python3
"""Train the v4 single-CRF baseline (Step 6.F ablation).

Walks ``skol_training_v2_no_golden`` (the hand corpus, per the
user-locked scope), reads each doc's pre-computed v4 attachments
(``article.spans.v4.json`` + ``article.page-headers.json``), builds
per-line features via Step 2's ``build_line_features`` + per-line
labels via :func:`skol_classifier.v4.labels.build_active_label_sequence`
in the 19-label ACTIVE_TAGS_19 vocab, fits the CRF across
``--epochs`` Adam steps with inverse-frequency class weighting, and
writes the trained bundle to Redis under
``skol:classifier:model:v4_single_hand`` + ``:meta``.

Unlike the two-pass trainers, this script does NOT filter Pass-1
layout lines out of the sequence — the whole point of the single-
CRF baseline is to train one CRF over every line in the 19-label
space.  The comparison row drives Step 7's "is the two-pass design
worth its complexity?" decision.

Pre-requisite: the operator runs
``bin/annotate_v4 --database <SAME-DB>`` once before training to
populate the v4 attachments.  Docs missing either v4 attachment are
counted + skipped.

Usage::

    bin/train_crf_single --source-db skol_training_v2_no_golden \\
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

from skol_classifier.v4.crf_single import (  # noqa: E402
    FEATURE_DIM,
    N_LABELS,
    SingleCRF,
    save_to_redis,
)
from skol_classifier.v4.features import (  # noqa: E402
    build_line_features,
    compute_line_starts,
)
from skol_classifier.v4.labels import build_active_label_sequence  # noqa: E402


_DEFAULT_REDIS_KEY = 'skol:classifier:model:v4_single_hand'
_DEFAULT_META_KEY = 'skol:classifier:model:v4_single_hand:meta'
_SPANS_ATTACHMENT = 'article.spans.v4.json'
_PAGE_HEADERS_ATTACHMENT = 'article.page-headers.json'
_ANN_ATTACHMENT = 'article.txt.ann'
_PLAINTEXT_ATTACHMENT = 'article.txt'


SbertLookup = Callable[[str], Optional[np.ndarray]]


# ---------------------------------------------------------------------------
# Pure helpers (covered by train_crf_layout_test.py)
# ---------------------------------------------------------------------------


def split_docs(
    doc_lengths: Sequence[Tuple[str, int]],
    *,
    dev_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Quartile-stratified train/dev split by doc length.

    Given ``[(doc_id, n_lines), ...]`` and a target dev fraction,
    returns ``(train_ids, dev_ids)``.  Stratifies by ``n_lines``
    quartile so each quartile contributes ``dev_fraction`` of its
    docs to the dev set — keeps the dev distribution close to the
    overall distribution.  Idempotent under ``seed``.
    """
    rng = random.Random(seed)
    sorted_by_len = sorted(doc_lengths, key=lambda t: t[1])
    n = len(sorted_by_len)
    train: List[str] = []
    dev: List[str] = []
    # Walk the sorted list in 4 equal-ish slices.
    for q in range(4):
        lo = (n * q) // 4
        hi = (n * (q + 1)) // 4
        quartile = list(sorted_by_len[lo:hi])
        rng.shuffle(quartile)
        n_dev = round(len(quartile) * dev_fraction)
        for i, (doc_id, _) in enumerate(quartile):
            (dev if i < n_dev else train).append(doc_id)
    # Sort for deterministic test assertions; re-shuffling happens
    # epoch-by-epoch inside the training loop anyway.
    train.sort()
    dev.sort()
    return train, dev


def inverse_frequency_weights(
    label_counts: np.ndarray,
    n_labels: int = N_LABELS,
) -> np.ndarray:
    """Inverse-frequency class weights.

    Returns a 1-D float64 array of length ``n_labels``.  Labels with
    nonzero support get ``weight = total / (n_labels × count)``;
    labels with zero support default to ``1.0`` (no penalty, no
    boost — defensive against zero-occurrence labels in small
    training corpora).
    """
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
    """Build the SBERT cache lookup callable for ``build_line_features``.

    Each cache key is ``skol:sbert:<model_tag>:<sha256(line_utf8)>``
    (matching ``bin/embed_lines.py``).  Missing keys return None;
    the caller (``build_line_features``) falls back to a zero
    vector.
    """
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
# Per-doc data extraction
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
    """3-path fallback (article.txt → article.pdf → YEDDA-strip)."""
    text = _read_attachment_text(
        db, doc_id, _PLAINTEXT_ATTACHMENT,
    )
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
    """Reconstruct Span objects from the JSON envelope produced by
    ``ingestors.spans.spans_to_json``."""
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


def _doc_n_lines(
    db: Any, doc_id: str,
) -> int:
    text = _load_plaintext(db, doc_id)
    if text is None:
        return 0
    return len(text.split('\n'))


def _prepare_doc(
    db: Any,
    doc_id: str,
    sbert_lookup: SbertLookup,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build (features_array, labels_array) for one doc.

    Returns None if any required attachment is missing or empty.
    """
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
    page_headers = json.loads(ph_bytes.decode('utf-8'))
    lines = plaintext.split('\n')
    line_starts = compute_line_starts(lines)

    features = np.zeros(
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
        features[i] = feats.concat()

    labels = np.asarray(
        build_active_label_sequence(plaintext, ann_text),
        dtype=np.int64,
    )
    if labels.shape[0] != features.shape[0]:
        return None
    if labels.shape[0] == 0:
        return None
    return features, labels


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _per_label_counts(label_arrays: Iterable[np.ndarray]) -> np.ndarray:
    """Sum per-label line counts across the training corpus."""
    counts = np.zeros(N_LABELS, dtype=np.float64)
    for labels in label_arrays:
        for c in range(N_LABELS):
            counts[c] += float((labels == c).sum())
    return counts


def _evaluate(
    model: SingleCRF,
    prepared: List[Tuple[str, np.ndarray, np.ndarray]],
    *,
    device: str,
) -> Tuple[float, List[float]]:
    """Macro F1 + per-label F1 over ``prepared`` dev docs."""
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
) -> Dict[str, Any]:
    """End-to-end training pass.  Returns a counts/stats dict."""
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

    # Phase 1: prepare data.
    prepared: List[Tuple[str, np.ndarray, np.ndarray]] = []
    doc_lengths: List[Tuple[str, int]] = []
    for doc_id in doc_ids:
        item = _prepare_doc(db, doc_id, sbert_lookup)
        if item is None:
            counts['skipped_no_spans'] += 1
            if verbosity >= 2:
                print(f'  skip {doc_id}: missing v4 attachments')
            continue
        features, labels = item
        prepared.append((doc_id, features, labels))
        doc_lengths.append((doc_id, features.shape[0]))

    if not prepared:
        if verbosity >= 1:
            print('  ✗ No docs prepared; aborting.')
        return counts

    # Phase 2: split.
    train_ids, dev_ids = split_docs(
        doc_lengths, dev_fraction=dev_fraction, seed=seed,
    )
    train_ids_set = set(train_ids)
    dev_ids_set = set(dev_ids)
    dev_set = [t for t in prepared if t[0] in dev_ids_set]
    train_list = [t for t in prepared if t[0] in train_ids_set]
    counts['trained_docs'] = len(train_list)
    counts['dev_docs'] = len(dev_set)

    # Phase 3: class weights.
    class_weights = inverse_frequency_weights(
        _per_label_counts(l for _, _, l in train_list),
    )

    if verbosity >= 1:
        print(
            f'  prepared={len(prepared)} '
            f'train={len(train_list)} dev={len(dev_set)} '
            f'epochs={epochs} lr={lr} seed={seed} device={device}'
        )

    # Phase 4: train.
    torch.manual_seed(seed)
    model = SingleCRF().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    started_at = datetime.now(timezone.utc).isoformat(timespec='seconds')

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
            # Class-weighted per-doc scaling.
            doc_weight = float(
                np.mean(class_weights[labels])
            )
            loss = base_loss * doc_weight
            loss.backward()
            opt.step()
            loss_sum += float(base_loss.detach().cpu())
        if dev_set:
            macro_f1, per_label_f1 = _evaluate(
                model, dev_set, device=device,
            )
        else:
            macro_f1, per_label_f1 = 0.0, [0.0] * N_LABELS
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

    # Phase 5: save.
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
        description='Train the v4 single-CRF baseline (Step 6.F).',
    )
    parser.add_argument(
        '--source-db', default=None,
        help='CouchDB DB (default: env_config couchdb_database).',
    )
    parser.add_argument(
        '--sbert-model', default='mpnet',
        help='SBERT cache namespace tag (default: mpnet).',
    )
    parser.add_argument(
        '--epochs', type=int, default=20,
    )
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dev-fraction', type=float, default=0.2)
    parser.add_argument(
        '--device',
        choices=('auto', 'cpu', 'cuda'),
        default='auto',
    )
    parser.add_argument(
        '--redis-key', default=_DEFAULT_REDIS_KEY,
    )
    parser.add_argument(
        '--redis-meta-key', default=_DEFAULT_META_KEY,
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
        or config.get('couchdb_database', 'skol_training_v2_no_golden')
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
        print(f'train_crf_single — db={db_name} device={device}')

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
