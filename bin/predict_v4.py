#!/usr/bin/env python3
"""v4 end-to-end predictor CLI.

Reads each doc from ``--golden-db`` (or the experiment's ingest
DB), runs :func:`skol_classifier.v4.predictor.predict_doc`, and
writes the resulting ``article.txt.ann`` attachment to
``--output-database``.

Required flags:

* ``--experiment <name>`` — resolves Pass-1 / Pass-2 Redis keys
  via the experiment doc's ``redis_keys.classifier_model_pass1``
  / ``…_pass2`` fields (see ``bin/env_config.py:_apply_experiment``).
* ``--golden-db <db>`` — the source DB carrying ``article.txt``,
  ``article.spans.v4.json``, and ``article.page-headers.json``.
  ``annotate_v4`` must have run over this DB first to populate
  the v4 attachments.
* ``--output-database <db>`` — where ``article.txt.ann`` lands.
  Required if the experiment doc doesn't specify
  ``databases.annotations``.

Env-var convenience: ``SKIP_EXISTING``, ``FORCE``, ``DRY_RUN``,
``LIMIT`` and ``VERBOSITY`` all flow through ``env_config`` and
match the corresponding CLI flags.

Usage::

    bin/predict_v4 --experiment production_v4 \\
        --golden-db skol_golden_v2 \\
        --output-database skol_exp_production_v4_ann \\
        --skip-existing -v 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterator, Optional, Tuple,
)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import couchdb  # type: ignore[import]  # noqa: E402

from env_config import (  # type: ignore[import]  # noqa: E402
    create_redis_client, get_env_config,
)
from ingestors.extract_plaintext import (  # noqa: E402
    plaintext_from_pdf, plaintext_from_yedda,
)

from skol_classifier.v4.crf_layout import (  # noqa: E402
    load_from_redis as load_layout_from_redis,
)
from skol_classifier.v4.crf_treatment import (  # noqa: E402
    load_from_redis as load_treatment_from_redis,
)
from skol_classifier.v4.predictor import predict_doc  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_DEFAULT_PASS1_STATE_KEY = 'skol:classifier:model:v4_layout'
_DEFAULT_PASS2_STATE_KEY = 'skol:classifier:model:v4_treatment'
_SPANS_ATTACHMENT = 'article.spans.v4.json'
_PAGE_HEADERS_ATTACHMENT = 'article.page-headers.json'
_PLAINTEXT_ATTACHMENT = 'article.txt'
_ANN_ATTACHMENT = 'article.txt.ann'


SbertLookup = Callable[[str], Optional[np.ndarray]]


# ---------------------------------------------------------------------------
# Pure helpers (unit-test surface)
# ---------------------------------------------------------------------------


def resolve_redis_keys(config: Dict[str, Any]) -> Dict[str, str]:
    """Pull the Pass-1 + Pass-2 Redis-key bundle out of env_config,
    falling back to the trainer defaults when either is unset or
    empty.  ``:meta`` is always derived from the state key."""
    pass1 = (
        config.get('classifier_model_key_pass1')
        or _DEFAULT_PASS1_STATE_KEY
    )
    pass2 = (
        config.get('classifier_model_key_pass2')
        or _DEFAULT_PASS2_STATE_KEY
    )
    return {
        'pass1_state': pass1,
        'pass1_meta': f'{pass1}:meta',
        'pass2_state': pass2,
        'pass2_meta': f'{pass2}:meta',
    }


def _iter_doc_ids(
    db: Any, limit: Optional[int] = None,
) -> Iterator[str]:
    """Yield doc IDs from ``db``, skipping design docs."""
    count = 0
    for row in db.view('_all_docs'):
        rid = str(row.id)
        if rid.startswith('_'):
            continue
        yield rid
        count += 1
        if limit is not None and count >= limit:
            break


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


def _load_plaintext(db: Any, doc_id: str) -> Optional[str]:
    """3-path fallback: article.txt → article.pdf → article.txt.ann."""
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


def _has_existing_ann(db: Any, doc_id: str) -> bool:
    try:
        doc = db[doc_id]
    except Exception:  # noqa: BLE001
        return False
    atts = doc.get('_attachments') or {}
    return _ANN_ATTACHMENT in atts


def make_sbert_lookup(
    redis_client: Any, model_tag: str = 'mpnet',
) -> SbertLookup:
    """Build the Redis-backed SBERT lookup the assembler expects.

    Same shape as ``bin/train_crf_layout.make_sbert_lookup`` —
    returns None on cache miss so the assembler falls back to a
    zero vector.
    """
    import hashlib

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


def _open_db(config: Dict[str, Any], db_name: str) -> Any:
    server = couchdb.Server(config['couchdb_url'])
    username = config.get('couchdb_username')
    password = config.get('couchdb_password')
    if username and password:
        server.resource.credentials = (username, password)
    if db_name not in server:
        raise RuntimeError(f"database {db_name!r} not found")
    return server[db_name]


def _open_or_create_db(config: Dict[str, Any], db_name: str) -> Any:
    server = couchdb.Server(config['couchdb_url'])
    username = config.get('couchdb_username')
    password = config.get('couchdb_password')
    if username and password:
        server.resource.credentials = (username, password)
    if db_name not in server:
        return server.create(db_name)
    return server[db_name]


def _resolve_device(choice: str) -> str:
    if choice == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return choice


# ---------------------------------------------------------------------------
# Per-doc + per-corpus loops
# ---------------------------------------------------------------------------


def _prepare_doc_inputs(
    input_db: Any, doc_id: str,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return ``(plaintext, spans_dict, page_headers_dict)``.

    Any value can be ``None``; the caller decides which absence is
    fatal-skip versus fall-through."""
    plaintext = _load_plaintext(input_db, doc_id)

    spans_bytes = _read_attachment_bytes(
        input_db, doc_id, _SPANS_ATTACHMENT,
    )
    spans_dict: Optional[Dict[str, Any]] = None
    if spans_bytes is not None:
        try:
            spans_dict = json.loads(spans_bytes.decode('utf-8'))
        except Exception:  # noqa: BLE001
            spans_dict = None

    ph_bytes = _read_attachment_bytes(
        input_db, doc_id, _PAGE_HEADERS_ATTACHMENT,
    )
    ph_dict: Optional[Dict[str, Any]] = None
    if ph_bytes is not None:
        try:
            ph_dict = json.loads(ph_bytes.decode('utf-8'))
        except Exception:  # noqa: BLE001
            ph_dict = None

    return plaintext, spans_dict, ph_dict


def predict_all(
    input_db: Any,
    output_db: Any,
    *,
    layout_crf: Any,
    treatment_crf: Any,
    sbert_lookup: SbertLookup,
    device: str = 'cpu',
    skip_existing: bool = False,
    force: bool = False,
    dry_run: bool = False,
    limit: Optional[int] = None,
    verbosity: int = 1,
) -> Dict[str, int]:
    """Walk ``input_db``, predict, write to ``output_db``.

    Returns a counts dict for the smoke-test report."""
    counts = {
        'scanned': 0,
        'skipped_existing': 0,
        'skipped_no_attachments': 0,
        'skipped_no_plaintext': 0,
        'predicted': 0,
    }
    for doc_id in _iter_doc_ids(input_db, limit=limit):
        counts['scanned'] += 1

        if skip_existing and not force:
            if _has_existing_ann(output_db, doc_id):
                counts['skipped_existing'] += 1
                if verbosity >= 2:
                    print(f'  skip {doc_id}: .ann already present')
                continue

        plaintext, spans_dict, ph_dict = _prepare_doc_inputs(
            input_db, doc_id,
        )
        if spans_dict is None or ph_dict is None:
            counts['skipped_no_attachments'] += 1
            if verbosity >= 1:
                print(
                    f'  skip {doc_id}: missing spans / page-headers '
                    '— re-run annotate_v4',
                )
            continue
        if plaintext is None:
            counts['skipped_no_plaintext'] += 1
            if verbosity >= 1:
                print(f'  skip {doc_id}: no plaintext source')
            continue

        per_line_tags, ann_text = predict_doc(
            plaintext, spans_dict, ph_dict,
            layout_crf, treatment_crf, sbert_lookup,
            device=device,
        )
        counts['predicted'] += 1

        if dry_run:
            if verbosity >= 1:
                n_blocks = ann_text.count('[@')
                print(
                    f'  dry-run {doc_id}: {len(per_line_tags)} lines '
                    f'→ {n_blocks} blocks ({len(ann_text)} bytes)',
                )
            continue

        # Ensure the doc exists in the output DB before put_attachment;
        # match what bin/annotate_v4.py does — re-fetch + write.
        if doc_id not in output_db:
            output_db.save({'_id': doc_id})
        target_doc = output_db[doc_id]
        output_db.put_attachment(
            target_doc, ann_text.encode('utf-8'),
            filename=_ANN_ATTACHMENT,
            content_type='text/plain',
        )
        if verbosity >= 2:
            print(f'  wrote {doc_id} ({len(ann_text)} bytes)')

    return counts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description='v4 end-to-end predictor.',
    )
    parser.add_argument(
        '--golden-db', dest='golden_db', default=None,
        help='Source DB (default: env_config golden_db_name).',
    )
    parser.add_argument(
        '--output-database', dest='output_database', default=None,
        help=(
            'Where article.txt.ann is written.  '
            'Defaults to the experiment doc\'s '
            'databases.annotations.'
        ),
    )
    parser.add_argument(
        '--sbert-model', default='mpnet',
        help='SBERT cache namespace tag (default: mpnet).',
    )
    parser.add_argument(
        '--device',
        choices=('auto', 'cpu', 'cuda'),
        default='auto',
    )
    parser.add_argument(
        '--pass1-key', dest='pass1_key', default=None,
        help=(
            'Override the Pass-1 (layout CRF) Redis state-key.  '
            'Takes precedence over the experiment doc and the '
            'classifier_model_key_pass1 env_config value.'
        ),
    )
    parser.add_argument(
        '--pass2-key', dest='pass2_key', default=None,
        help='Same as --pass1-key for the Pass-2 (treatment) CRF.',
    )
    args, _ = parser.parse_known_args()

    config = get_env_config()
    verbosity = int(config.get('verbosity', 1) or 0)
    dry_run = bool(config.get('dry_run', False))
    force = bool(config.get('force', False))
    skip_existing = bool(config.get('skip_existing', False)) and not force
    limit_raw = config.get('limit')
    limit = (
        int(limit_raw) if limit_raw not in (None, '') else None
    )

    input_db_name = (
        args.golden_db
        or config.get('golden_db_name')
        or config.get('ingest_db_name')
    )
    if not input_db_name:
        print(
            '✗ --golden-db (or env_config golden_db_name) is required.',
            file=sys.stderr,
        )
        return 1

    output_db_name = (
        args.output_database
        or config.get('annotations_db_name')
    )
    if not output_db_name:
        print(
            '✗ --output-database (or experiment.databases.annotations) '
            'is required.',
            file=sys.stderr,
        )
        return 1

    try:
        input_db = _open_db(config, input_db_name)
    except Exception as exc:  # noqa: BLE001
        print(f'✗ cannot open {input_db_name!r}: {exc}', file=sys.stderr)
        return 1
    try:
        output_db = _open_or_create_db(config, output_db_name)
    except Exception as exc:  # noqa: BLE001
        print(
            f'✗ cannot open/create {output_db_name!r}: {exc}',
            file=sys.stderr,
        )
        return 1

    # CLI --pass1-key / --pass2-key override the env_config /
    # experiment-doc resolution.  Useful for ad-hoc smoke runs against
    # alternate model bundles without editing the experiment doc.
    if args.pass1_key:
        config['classifier_model_key_pass1'] = args.pass1_key
    if args.pass2_key:
        config['classifier_model_key_pass2'] = args.pass2_key
    redis_keys = resolve_redis_keys(config)
    redis_client = create_redis_client(decode_responses=False)
    device = _resolve_device(args.device)

    try:
        layout_crf, layout_meta = load_layout_from_redis(
            redis_client,
            key=redis_keys['pass1_state'],
            meta_key=redis_keys['pass1_meta'],
            map_location=device,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f'✗ cannot load Pass-1 CRF from '
            f'{redis_keys["pass1_state"]!r}: {exc}',
            file=sys.stderr,
        )
        return 1
    try:
        treatment_crf, treatment_meta = load_treatment_from_redis(
            redis_client,
            key=redis_keys['pass2_state'],
            meta_key=redis_keys['pass2_meta'],
            map_location=device,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f'✗ cannot load Pass-2 CRF from '
            f'{redis_keys["pass2_state"]!r}: {exc}',
            file=sys.stderr,
        )
        return 1
    layout_crf.to(device)
    treatment_crf.to(device)
    layout_crf.eval()
    treatment_crf.eval()

    sbert_lookup = make_sbert_lookup(
        redis_client, model_tag=args.sbert_model,
    )

    if verbosity >= 1:
        print(
            f'predict_v4 — in={input_db_name} out={output_db_name} '
            f'device={device}',
        )
        print(
            f'  pass1={redis_keys["pass1_state"]} '
            f'pass2={redis_keys["pass2_state"]}',
        )

    counts = predict_all(
        input_db, output_db,
        layout_crf=layout_crf,
        treatment_crf=treatment_crf,
        sbert_lookup=sbert_lookup,
        device=device,
        skip_existing=skip_existing, force=force,
        dry_run=dry_run, limit=limit,
        verbosity=verbosity,
    )

    if verbosity >= 1:
        print()
        for k in (
            'scanned', 'skipped_existing',
            'skipped_no_attachments', 'skipped_no_plaintext',
            'predicted',
        ):
            print(f'  {k:<28} {counts[k]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
