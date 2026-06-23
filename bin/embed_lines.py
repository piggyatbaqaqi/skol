#!/usr/bin/env python3
"""Per-line SBERT cache writer for the v4 classifier.

Walks a CouchDB database, reconstructs plaintext for each document via
a 3-path fallback (``article.txt`` -> ``article.pdf`` ->
``article.txt.ann``), splits the text into unique non-empty lines, and
caches the SBERT embedding for each line in Redis under
``skol:sbert:<model>:<sha256(line)>``.

Idempotent: lines already present in Redis are skipped unless
``--force`` is set.  The cache is consumed by the Step 2 feature
assembler and both CRF passes (Steps 3-4) of the v4 plan
(docs/v4_classifier_plan.md).

Usage:
    bin/embed_lines.py --model mpnet \\
        --source-db skol_training_v3_combined_no_golden [--limit N] \\
        [--dry-run] [--force] [--batch-size N] [--verbosity 0..3]

Selects between ``all-mpnet-base-v2`` (768-dim) and ``all-MiniLM-L6-v2``
(384-dim) at run time; the two models live under separate key prefixes.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import socket
import sys
from collections import defaultdict
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Tuple,
)

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import couchdb  # type: ignore[import]  # noqa: E402
import redis  # noqa: E402

from annotate_v4 import (  # type: ignore[import]  # noqa: E402
    _xml_attachments_present,
)

from env_config import (  # type: ignore[import]  # noqa: E402
    common_parser, create_redis_client, get_env_config,
)
from ingestors.extract_plaintext import (  # noqa: E402
    plaintext_from_pdf, plaintext_from_yedda,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAMES: Dict[str, str] = {
    'mpnet': 'all-mpnet-base-v2',
    'minilm': 'all-MiniLM-L6-v2',
}

_MODEL_DIMS: Dict[str, int] = {
    'mpnet': 768,
    'minilm': 384,
}

_DEFAULT_KEY_PREFIX = 'skol:sbert:'
_LOCK_TTL = 24 * 60 * 60  # 24 h — long enough for a full-corpus pass


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def redis_key(
    line_text: str,
    model_tag: str,
    prefix: str = _DEFAULT_KEY_PREFIX,
) -> str:
    """Cache key for ``line_text`` under ``model_tag``.

    Key format: ``{prefix}{model_tag}:{sha256_hex(line_text_utf8)}``.
    The sha256 is over the line text's UTF-8 byte encoding so non-ASCII
    text (Cintractiella, π, …) hashes consistently across platforms.
    """
    h = hashlib.sha256(line_text.encode('utf-8')).hexdigest()
    return f'{prefix}{model_tag}:{h}'


def iter_unique_lines(plaintext: str) -> Iterator[str]:
    """Yield each unique non-empty / non-whitespace-only line of
    ``plaintext`` in first-seen order.

    Splits on ``'\\n'`` rather than ``str.splitlines()`` to match the
    canonical line-split used in
    ``skol_classifier/extraction_modes/line_mode.py``.  Internal
    whitespace is preserved verbatim so Step 2 can look up the same
    line text by exact match.
    """
    seen: set = set()
    for line in plaintext.split('\n'):
        if not line.strip():
            continue
        if line in seen:
            continue
        seen.add(line)
        yield line


def _attachment_to_text(attach: Any) -> Optional[str]:
    """Coerce a couchdb.Database.get_attachment return value to ``str``.

    Returns ``None`` if the attachment doesn't exist.  Handles file-like
    objects, raw bytes (decoded utf-8), and already-decoded strings.
    """
    if attach is None:
        return None
    if hasattr(attach, 'read'):
        attach = attach.read()
    if isinstance(attach, bytes):
        return attach.decode('utf-8', errors='ignore')
    return str(attach)


def _attachment_to_bytes(attach: Any) -> Optional[bytes]:
    """Coerce a get_attachment return to ``bytes``; ``None`` if absent."""
    if attach is None:
        return None
    if hasattr(attach, 'read'):
        attach = attach.read()
    if isinstance(attach, str):
        return attach.encode('utf-8')
    assert isinstance(attach, bytes)
    return attach


def load_plaintext(db: Any, doc_id: str) -> Tuple[Optional[str], str]:
    """Reconstruct plaintext for ``doc_id`` via the 3-path fallback chain.

    Priority order:

    1. ``article.txt`` attachment (canonical plaintext, when present).
    2. ``article.pdf`` attachment -> :func:`plaintext_from_pdf`.  Covers
       the ~160 hand-annotated PDF docs that lack a bare ``article.txt``.
    3. ``article.txt.ann`` attachment -> :func:`plaintext_from_yedda`.
       Covers the ~1724 JATS-derived docs in
       ``skol_training_v3_combined_no_golden`` which only carry the
       YEDDA annotation file.

    Returns ``(text, source_tag)`` where ``source_tag`` is one of
    ``'article.txt'`` / ``'article.pdf'`` / ``'article.txt.ann'`` /
    ``'missing'``.  ``text`` is ``None`` only when ``source_tag`` is
    ``'missing'``.
    """
    raw = db.get_attachment(doc_id, 'article.txt')
    text = _attachment_to_text(raw)
    if text is not None:
        return text, 'article.txt'

    pdf_bytes = _attachment_to_bytes(db.get_attachment(doc_id, 'article.pdf'))
    if pdf_bytes is not None:
        return plaintext_from_pdf(pdf_bytes), 'article.pdf'

    ann_text = _attachment_to_text(
        db.get_attachment(doc_id, 'article.txt.ann'))
    if ann_text is not None:
        return plaintext_from_yedda(ann_text), 'article.txt.ann'

    return None, 'missing'


# ---------------------------------------------------------------------------
# LineEmbedder
# ---------------------------------------------------------------------------


EncoderFn = Callable[[List[str]], np.ndarray]


class LineEmbedder:
    """Wraps a SBERT encoder + a Redis client for one model variant.

    The encoder is constructed lazily on first call to :meth:`embed_batch`
    so that unit tests passing an explicit ``encoder=`` callable never
    load the real model weights (~200 MB).
    """

    def __init__(
        self,
        model_tag: str,
        redis_client: 'redis.Redis',
        *,
        dim: Optional[int] = None,
        encoder: Optional[EncoderFn] = None,
        key_prefix: str = _DEFAULT_KEY_PREFIX,
        batch_size: int = 128,
    ) -> None:
        if model_tag not in _MODEL_NAMES:
            raise ValueError(
                f'Unknown model_tag {model_tag!r}; '
                f'expected one of {sorted(_MODEL_NAMES)}'
            )
        self.model_tag = model_tag
        self.redis_client = redis_client
        self.dim = dim if dim is not None else _MODEL_DIMS[model_tag]
        self.key_prefix = key_prefix
        self.batch_size = batch_size
        self._encoder = encoder

    def _ensure_encoder(self) -> None:
        if self._encoder is not None:
            return
        # Lazy import — keeps unit tests free of the SBERT dependency
        # when they supply their own encoder callable.
        from sentence_transformers import (  # type: ignore[import]
            SentenceTransformer,
        )
        st_model = SentenceTransformer(_MODEL_NAMES[self.model_tag])
        bs = self.batch_size

        def _encode(lines: List[str]) -> np.ndarray:
            return st_model.encode(
                lines, batch_size=bs, convert_to_numpy=True,
            ).astype(np.float32)
        self._encoder = _encode

    def embed_batch(self, lines: List[str]) -> np.ndarray:
        """Encode ``lines`` -> ndarray of shape ``(len(lines), self.dim)``."""
        if not lines:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_encoder()
        assert self._encoder is not None  # for mypy
        vectors = self._encoder(lines)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors.astype(np.float32)

    def key_for(self, line: str) -> str:
        return redis_key(line, self.model_tag, prefix=self.key_prefix)

    def get_existing(self, line: str) -> Optional[bytes]:
        """Return the cached bytes for ``line`` or ``None``.

        Single-line helper.  Hot-path callers should use
        :meth:`get_existing_batch` so the Redis round trips are
        amortized across the doc rather than incurred per line."""
        result = self.redis_client.get(self.key_for(line))
        return result  # type: ignore[return-value]

    def get_existing_batch(
        self, lines: List[str],
    ) -> List[Optional[bytes]]:
        """Return cached bytes for each line, parallel to ``lines``.

        Single ``MGET`` round trip; ``None`` for cache misses.
        Replaces N per-line ``get_existing`` calls inside the doc
        loop — at ~1 k lines per doc on a 30 k-doc corpus, the
        round-trip savings dominate wall-clock.
        """
        if not lines:
            return []
        keys = [self.key_for(line) for line in lines]
        return list(self.redis_client.mget(keys))

    def write_one(self, line: str, vector: np.ndarray) -> None:
        """Cache one line→vector pair.

        Single-line helper.  Hot-path callers should use
        :meth:`write_many` so the writes for one doc go in a single
        pipelined burst."""
        self.redis_client.set(
            self.key_for(line),
            vector.astype(np.float32).tobytes(),
        )

    def write_many(
        self, items: 'List[Tuple[str, np.ndarray]]',
    ) -> None:
        """Cache multiple ``(line, vector)`` pairs in one pipelined
        Redis burst.

        ``transaction=False`` because the writes are independent —
        atomicity isn't required and non-transactional pipelined
        writes are faster (the server can interleave them with
        other clients' commands).
        """
        if not items:
            return
        pipe = self.redis_client.pipeline(transaction=False)
        for line, vec in items:
            pipe.set(
                self.key_for(line),
                vec.astype(np.float32).tobytes(),
            )
        pipe.execute()


# ---------------------------------------------------------------------------
# process_doc — the per-document driver
# ---------------------------------------------------------------------------


def process_doc(
    db: Any,
    doc_id: str,
    embedder: LineEmbedder,
    *,
    skip_existing: bool = True,
    force: bool = False,
    dry_run: bool = False,
    verbosity: int = 0,
) -> Dict[str, Any]:
    """Embed every unique non-empty line of ``doc_id``'s plaintext.

    Returns a counts dict: ``{lines_total, unique, cached_hits,
    embedded, skipped_empty, source}``.  ``source`` is the attachment
    that supplied the plaintext (see :func:`load_plaintext`).
    """
    counts: Dict[str, Any] = {
        'lines_total': 0, 'unique': 0, 'cached_hits': 0,
        'embedded': 0, 'skipped_empty': 0, 'source': 'missing',
    }

    text, source = load_plaintext(db, doc_id)
    counts['source'] = source
    if text is None:
        if verbosity >= 2:
            msg = f'  {doc_id}: no plaintext source available'
            try:
                doc = db[doc_id]
            except Exception:  # noqa: BLE001
                doc = {}
            xml_present = _xml_attachments_present(doc)
            if xml_present:
                # Operator hint: doc isn't orphan — see annotate_v4
                # for the same diagnostic.
                msg += (
                    f' — but XML attachment present: '
                    f'{", ".join(xml_present)}'
                )
            print(msg)
        return counts

    raw_lines = text.split('\n')
    counts['lines_total'] = len(raw_lines)
    counts['skipped_empty'] = sum(1 for ln in raw_lines if not ln.strip())

    unique_lines = list(iter_unique_lines(text))
    counts['unique'] = len(unique_lines)

    if force:
        to_embed = unique_lines
    else:
        # ONE MGET round trip covers the whole doc.  Pre-batching
        # this was the dominant wall-clock cost on mostly-cached
        # corpora because the previous per-line GET loop forced
        # the GPU to sit idle waiting on Redis I/O.
        existing = embedder.get_existing_batch(unique_lines)
        to_embed = []
        for line, val in zip(unique_lines, existing):
            if skip_existing and val is not None:
                counts['cached_hits'] += 1
            else:
                to_embed.append(line)

    if not to_embed or dry_run:
        return counts

    vectors = embedder.embed_batch(to_embed)
    # ONE pipelined burst writes every new embedding for this doc;
    # the previous per-line SET loop had the same round-trip
    # amplification issue as the cache check above.
    embedder.write_many(list(zip(to_embed, vectors)))
    counts['embedded'] = len(to_embed)
    return counts


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _resolve_skip_existing(config: Dict[str, Any]) -> bool:
    """Decide whether to skip lines already cached in Redis.

    For this embedder the only useful semantics are idempotent-by-
    default with ``--force`` as the override.  env_config's own
    ``skip_existing`` field is intentionally ignored: env_config
    hardcodes it to ``False`` as the env-var default
    (``_get_env('SKIP_EXISTING', '').lower() in ('1','true','yes')``
    is ``False`` for an unset env var), which would make every run
    re-embed every cached line.  Per CLAUDE.md rule 11 the hardcoded
    default should reflect intent, and for a cache builder that's
    "skip what's already cached."  ``--force`` flips it to re-embed.
    """
    return not bool(config.get('force'))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _acquire_lock(
    r: 'redis.Redis', lock_key: str, verbosity: int = 1,
) -> bool:
    """Acquire the build lock.

    Stores ``<hostname>:<pid>`` as the lock value so an operator
    looking at a stuck lock can immediately tell which process to
    check (e.g. ``ps -p <pid>`` / ``/proc/<pid>/cmdline``) instead
    of having to scan every host that might run embed_lines.

    The semantics (existence + TTL) are unchanged from the
    pre-2026-06-10 ``b'building'`` placeholder — nothing in the
    codebase reads the value today, so this is purely additive
    diagnostic data.
    """
    holder = f'{socket.gethostname()}:{os.getpid()}'.encode()
    acquired = r.set(lock_key, holder, nx=True, ex=_LOCK_TTL)
    if not acquired:
        existing = r.get(lock_key)
        existing_str = (
            existing.decode(errors='replace')
            if isinstance(existing, bytes) else str(existing)
        )
        if verbosity >= 1:
            print(
                f'Another build holds {lock_key} '
                f'(held by {existing_str}); exiting.  '
                f'Check ``ps -p <pid>`` on that host to see if '
                f'the holder is still alive.'
            )
        return False
    if verbosity >= 1:
        print(
            f'Acquired build lock: {lock_key} '
            f'(holder={holder.decode()}, TTL {_LOCK_TTL}s)'
        )
    return True


def _release_lock(
    r: 'redis.Redis', lock_key: str, verbosity: int = 1,
) -> None:
    try:
        r.delete(lock_key)
        if verbosity >= 1:
            print(f'Released build lock: {lock_key}')
    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
        if verbosity >= 1:
            print(f'Could not release lock (will auto-expire): {exc}')


def main() -> int:
    parser = argparse.ArgumentParser(
        parents=[common_parser()],
        description='Cache per-line SBERT embeddings in Redis '
                    'for the v4 classifier.',
    )
    parser.add_argument(
        '--sbert-model', required=True, choices=sorted(_MODEL_NAMES),
        dest='sbert_model',
        help='SBERT variant (mpnet=768-dim, minilm=384-dim). '
             '(Renamed from --model to avoid argparse prefix collision '
             "with env_config's --model-version / --model-name.)",
    )
    parser.add_argument(
        '--source-db', required=True, metavar='DB_NAME',
        help='CouchDB database to walk.',
    )
    parser.add_argument(
        '--batch-size', type=int, default=128, metavar='N',
        help='SentenceTransformer batch size (default: 128). '
             '192 and 256 have caused hard GPU lockups on '
             'puchpuchobs (RTX 5090, driver 595, CUDA 13.2) — '
             'likely a driver-level Blackwell edge case, not a '
             'soft OOM PyTorch could recover from.  128 is '
             'operationally verified.  For smaller GPUs (3050, '
             '4080 12 GB) pass an even smaller explicit value.',
    )
    parser.add_argument(
        '--skip-lock', action='store_true',
        help='Bypass the distributed lock (debug only).',
    )
    parser.add_argument(
        '--key-prefix', default=_DEFAULT_KEY_PREFIX, metavar='STR',
        help=f'Redis key prefix (default: {_DEFAULT_KEY_PREFIX!r}).',
    )
    # common_parser() (the parents= above) contributes --dry-run, --force,
    # --skip-existing, --limit, --verbosity, --couchdb-url, --redis-host,
    # etc., so parse_args() accepts them while rejecting typos.
    args = parser.parse_args()

    config = get_env_config(cli_args=args)
    verbosity = int(config.get('verbosity', 1) or 0)
    dry_run = bool(config.get('dry_run', False))
    skip_existing = _resolve_skip_existing(config)
    force = bool(config.get('force', False))
    limit_raw = config.get('limit')
    limit = int(limit_raw) if limit_raw not in (None, '') else None

    # CouchDB
    couchdb_url = config['couchdb_url']
    server = couchdb.Server(couchdb_url)
    username = config.get('couchdb_username')
    password = config.get('couchdb_password')
    if username and password:
        server.resource.credentials = (username, password)
    if args.source_db not in server:
        print(f'Source DB {args.source_db!r} not found on {couchdb_url}')
        return 1
    db = server[args.source_db]

    # Redis + lock
    r = create_redis_client(decode_responses=False)
    lock_key = f'skol:build:sbert:{args.sbert_model}:lock'
    if not args.skip_lock and not dry_run:
        if not _acquire_lock(r, lock_key, verbosity=verbosity):
            return 1

    embedder = LineEmbedder(
        model_tag=args.sbert_model,
        redis_client=r,
        key_prefix=args.key_prefix,
        batch_size=args.batch_size,
    )

    totals: Dict[str, int] = defaultdict(int)
    sources: Dict[str, int] = defaultdict(int)
    docs_processed = 0
    try:
        for doc_id in db:
            if doc_id.startswith('_design/'):
                continue
            if limit is not None and docs_processed >= limit:
                break
            try:
                counts = process_doc(
                    db, doc_id, embedder,
                    skip_existing=skip_existing,
                    force=force, dry_run=dry_run,
                    verbosity=verbosity,
                )
            except Exception as exc:  # noqa: BLE001
                if verbosity >= 1:
                    print(f'  {doc_id}: ERROR {exc}')
                continue
            docs_processed += 1
            for k, v in counts.items():
                if isinstance(v, int):
                    totals[k] += v
            sources[str(counts.get('source', 'missing'))] += 1
            if verbosity >= 2:
                print(
                    f'  {doc_id}: unique={counts["unique"]} '
                    f'embedded={counts["embedded"]} '
                    f'cached={counts["cached_hits"]} '
                    f'source={counts["source"]}'
                )
            elif verbosity >= 1 and docs_processed % 100 == 0:
                print(f'  processed {docs_processed} docs ...')
    finally:
        if not args.skip_lock and not dry_run:
            _release_lock(r, lock_key, verbosity=verbosity)

    if verbosity >= 1:
        print(f'\nProcessed {docs_processed} docs from {args.source_db!r}')
        print(f'  Totals: {dict(totals)}')
        print(f'  Source distribution: {dict(sources)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
