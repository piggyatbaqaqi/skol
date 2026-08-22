#!/usr/bin/env python3
"""Derive a technical vocabulary from the treatment corpus itself.

Half of the production vocabulary for the §9 mode-B OCR detector
(``D8`` in ``docs/data_quality_production_v4_model.md``); the other
half is ``data/botanical_latin_wordlist.txt``.

D8 spots OCR space-displacement by testing whether a run of
out-of-vocabulary tokens rejoins into a real word, so it is blind to
any vocabulary it lacks.  Published dictionaries do not carry the
descriptive neo-Latin this corpus lives on — Whitaker's *WORDS* has
neither ``basidiomata`` nor ``campanulatus`` at any age — but the
corpus does, by construction.

Method: collect out-of-vocabulary alphabetic forms of 4+ characters
from ``description`` and ``diagnosis`` across every treatment, count
how many **distinct documents** each appears in, and keep those above
a threshold.

**The threshold is the whole safety argument.**  The corpus contains
OCR-corrupted treatments, so a naive vocabulary would absorb their
corruption — the ``vere``-for-``were`` hazard, where a junk token in
the word list masks the very damage being hunted.  Counting documents
rather than occurrences means a form repeated forty times inside one
mangled treatment still counts once, and the default threshold of 50
requires a form to recur across fifty independent documents before it
is believed.

Measured trade-off (see D8 for the full table): at ``df >= 2`` the
vocabulary is large enough to manufacture spurious rejoins and the
worst poster child reaches 6.59 %; at ``df >= 50`` every poster child
sits at 0.00 % while both corruption cases clear 22 %.  Lower is not
safer here.

Usage::

    bin/corpus_vocabulary --experiment production_v4 \\
        --output data/corpus_vocabulary.txt
"""

import argparse
import collections
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, TextIO, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import common_parser, get_env_config  # noqa: E402

_TOKEN_RE = re.compile(r'[A-Za-z]+')
_FIELDS = ('description', 'diagnosis')
_MIN_LENGTH = 4
_DEFAULT_THRESHOLD = 50
_DEFAULT_WORDLIST = '/usr/share/dict/american-english'


def load_english(path: str) -> Set[str]:
    """Read the base English word list used to filter tokens out."""
    with open(path, encoding='utf-8', errors='replace') as handle:
        return {line.strip().lower() for line in handle if line.strip()}


def field_tokens(doc: Dict[str, Any], english: Set[str]) -> Set[str]:
    """Distinct out-of-vocabulary forms in one treatment.

    A set, not a list: each form counts once per document however
    often it occurs, which is what makes the threshold a recurrence
    guard rather than a frequency one.
    """
    text = ' '.join((doc.get(field) or '') for field in _FIELDS)
    return {
        token for token in (t.lower() for t in _TOKEN_RE.findall(text))
        if len(token) >= _MIN_LENGTH and token not in english
    }


def document_frequencies(
    db: Any, english: Set[str],
) -> Tuple[Dict[str, int], int]:
    """Count treatments each out-of-vocabulary form appears in.

    Returns ``(frequencies, treatments_scanned)``.  Design documents
    and treatments with no text are skipped.
    """
    freqs: Dict[str, int] = collections.Counter()
    scanned = 0
    for row in db.view('_all_docs', include_docs=True):
        doc = getattr(row, 'doc', None)
        doc_id = getattr(row, 'id', '')
        if not doc or not doc_id.startswith('taxon_'):
            continue
        tokens = field_tokens(doc, english)
        if not tokens:
            continue
        scanned += 1
        for token in tokens:
            freqs[token] += 1
    return dict(freqs), scanned


def select_vocabulary(freqs: Dict[str, int], threshold: int) -> List[str]:
    """Sorted forms occurring in at least ``threshold`` documents."""
    return sorted(w for w, n in freqs.items() if n >= threshold)


def write_vocabulary(words: List[str], stream: TextIO) -> None:
    """One form per line."""
    for word in words:
        stream.write(word + '\n')


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[common_parser()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--output', required=True, metavar='PATH',
        help='Vocabulary to write, one form per line.',
    )
    parser.add_argument(
        '--threshold', type=int, default=_DEFAULT_THRESHOLD,
        metavar='N',
        help=f'Minimum distinct treatments a form must appear in '
             f'(default {_DEFAULT_THRESHOLD}).  Lower is NOT safer: '
             f'a large vocabulary manufactures spurious rejoins and '
             f'lets OCR corruption in.',
    )
    parser.add_argument(
        '--english-wordlist', default=_DEFAULT_WORDLIST, metavar='PATH',
        help=f'Base English list to filter against (default '
             f'{_DEFAULT_WORDLIST}, from the wamerican package).',
    )
    args = parser.parse_args()
    config = get_env_config(cli_args=args)

    experiment = config.get('experiment_name')
    if not experiment:
        print("error: --experiment is required", file=sys.stderr)
        return 2
    if args.threshold < 1:
        print("error: --threshold must be at least 1", file=sys.stderr)
        return 2
    try:
        english = load_english(args.english_wordlist)
    except OSError as exc:
        print(f"error: cannot read {args.english_wordlist}: {exc}",
              file=sys.stderr)
        return 2

    import couchdb  # type: ignore[import-untyped]
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (
        config['couchdb_username'], config['couchdb_password'],
    )
    db_name = config['treatments_db_name']
    if db_name not in server:
        print(f"error: {db_name} not found", file=sys.stderr)
        return 2

    freqs, scanned = document_frequencies(server[db_name], english)
    words = select_vocabulary(freqs, args.threshold)
    if not words:
        print(f"error: no forms reached df >= {args.threshold}",
              file=sys.stderr)
        return 2

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as handle:
        write_vocabulary(words, handle)
    print(f"{scanned} treatments scanned, {len(freqs)} distinct forms, "
          f"{len(words)} at df >= {args.threshold} -> {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
