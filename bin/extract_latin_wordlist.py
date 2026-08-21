#!/usr/bin/env python3
"""Build a Latin word list from scraped Latdict word-list pages.

Produces the Latin / descriptive-botanical vocabulary layer used by
the §9 mode-B OCR detector (``D8`` in
``docs/data_quality_production_v4_model.md``).  That detector spots
OCR space-displacement by checking whether a run of out-of-vocabulary
tokens rejoins into a real word, so it is blind to any language it
has no vocabulary for — and Latin diagnoses are 79.5 % out-of-
vocabulary against an English dictionary.

Input is a local mirror of ``latin-dictionary.net/list/letter/``.
Each page carries entries of the shape::

    <li id="word-3" class="word"><a href="...">kalo, kalare,
        kalavi, kalatus</a> v</li>

The anchor text is the dictionary's principal parts, so one entry
yields several *inflected* forms — verb principal parts, adjective
genders, noun nominative + genitive — not just a lemma.  That matters:
the corpus needs ``albae`` and ``adscendentes``, not only ``albus``.

**Licensing**: see ``data/latin_wordlist.CITATION.md``.  The source
site is marked "All rights reserved"; read that file before
redistributing the output.

Usage::

    bin/extract_latin_wordlist \\
        --source-dir ~/www/http/latin-dictionary.net/list/letter/ \\
        --output data/latin_wordlist.txt
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# <li id="word-N" class="word"><a href="...">PARTS</a> POS</li>
_ENTRY_RE = re.compile(
    r'<li id="word-\d+" class="word">'
    r'<a href="[^"]*">(.*?)</a>\s*([a-z]*)</li>'
)

# Trailing "(i)" style optional-letter alternation: kalendari(i).
_PAREN_RE = re.compile(r'^([A-Za-zÀ-ÿ]+)\(([A-Za-zÀ-ÿ]+)\)$')

# "Babylonos/is" — the chunk after the slash replaces an
# equal-length suffix of the chunk before it.
_SLASH_RE = re.compile(r'^([A-Za-zÀ-ÿ]+)/([A-Za-zÀ-ÿ]+)$')

_LIGATURES = {'æ': 'ae', 'Æ': 'ae', 'œ': 'oe', 'Œ': 'oe'}

# One-letter forms are abbreviation residue (K., A., Cn.).  Keeping
# them would let the D8 rejoin metric split ordinary words at any
# point and still find both halves "known".
_MIN_LENGTH = 2


def parse_entries(html: str) -> List[Tuple[str, str]]:
    """Return ``(principal_parts, part_of_speech)`` per word entry.

    Non-word markup (the A–Z pager, navigation) is ignored: only
    ``<li class="word">`` elements match.
    """
    return [(m.group(1), m.group(2)) for m in _ENTRY_RE.finditer(html)]


def expand_variants(part: str) -> List[str]:
    """Expand one principal part into every orthographic form.

    Handles the dictionary's two compression conventions and drops
    markers (``(gen.)``, ``abb.``, ``-``) outright.
    """
    part = part.strip()
    if not part:
        return []
    paren = _PAREN_RE.match(part)
    if paren:
        stem, extra = paren.group(1), paren.group(2)
        return [stem, stem + extra]
    slash = _SLASH_RE.match(part)
    if slash:
        stem, alt = slash.group(1), slash.group(2)
        if len(alt) < len(stem):
            return [stem, stem[:-len(alt)] + alt]
        return [stem]
    if not re.fullmatch(r'[A-Za-zÀ-ÿ]+', part):
        return []
    return [part]


def fold_form(form: str) -> Optional[str]:
    """Normalise one form for set-membership use.

    Lowercases, expands ``æ``/``œ`` ligatures and strips combining
    marks so ``abscīdō`` matches ``abscido``.  Returns ``None`` for
    anything that isn't purely alphabetic after folding, or that is
    shorter than ``_MIN_LENGTH``.

    Folding belongs to the *word list*.  Never fold treatment text —
    rewriting it would invalidate every stored span offset.
    """
    if not form:
        return None
    for lig, repl in _LIGATURES.items():
        form = form.replace(lig, repl)
    decomposed = unicodedata.normalize('NFKD', form)
    stripped = ''.join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    )
    stripped = stripped.lower()
    if not stripped.isalpha() or not stripped.isascii():
        return None
    if len(stripped) < _MIN_LENGTH:
        return None
    return stripped


def wordlist_from_html(html: str) -> List[str]:
    """Sorted, deduplicated, folded word list for one page."""
    out: Set[str] = set()
    for parts, _pos in parse_entries(html):
        for part in parts.split(','):
            for variant in expand_variants(part):
                folded = fold_form(variant)
                if folded:
                    out.add(folded)
    return sorted(out)


def wordlist_from_dir(source_dir: Path) -> List[str]:
    """Merge every ``*.html`` page under ``source_dir``.

    ``index.html`` is skipped — Latdict serves it as a byte-for-byte
    copy of the ``a`` page, so including it would just double the
    parse work.
    """
    out: Set[str] = set()
    pages = sorted(
        p for p in source_dir.glob('*.html') if p.name != 'index.html'
    )
    if not pages:
        raise ValueError(f"no *.html pages found under {source_dir}")
    for page in pages:
        html = page.read_text(encoding='utf-8', errors='replace')
        out.update(wordlist_from_html(html))
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--source-dir', required=True, metavar='DIR',
        help='Local mirror of latin-dictionary.net/list/letter/.',
    )
    parser.add_argument(
        '--output', required=True, metavar='PATH',
        help='Word list to write, one folded form per line.',
    )
    args = parser.parse_args()

    source = Path(args.source_dir).expanduser()
    if not source.is_dir():
        print(f"error: {source} is not a directory", file=sys.stderr)
        return 2
    try:
        words = wordlist_from_dir(source)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as handle:
        for word in words:
            handle.write(word + '\n')
    print(f"{len(words)} forms -> {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
