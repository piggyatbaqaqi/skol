#!/usr/bin/env python3
"""Build a botanical-Latin word list from the V. F. Thomas Co. site.

Source: *Botanical Latin Words*, a project of V. F. Thomas Co.,
<https://www.vfthomas.com/botanicalLatinwords/> — 26 alphabetical
pages.  See ``data/botanical_latin_wordlist.CITATION.md``.

**Licence unresolved**: the site carries no copyright or licence
notice.  Do not redistribute the derived list until that is settled.

Two kinds of line are collected, and the second is the reason this
source matters:

``<b><i>baccatus, -a, -um</i></b> - adjective, group A: berry-like``
    A **headword**.  Parts beginning ``-`` are endings that replace
    the lemma's nominative ending — and the replaced length varies:
    ``baccatus``/``-a`` strips ``us``, ``bacca``/``-ae`` strips
    ``a``, ``acaulis``/``-e`` strips ``is``.

``&nbsp;&nbsp;&nbsp;<i>baccis</i> - ablative plural [p. 173. …]``
    An **attested inflected form**, read out of Linnaeus's *Species
    Plantarum* with its grammatical parse and page citation.  Every
    other Latin source tried here supplies lemmas (FreeDict, DCC) or
    roots (the Wikipedia systematic list) or generated paradigms
    over the wrong vocabulary (WORDS).  These are real descriptive
    botanical inflections, which is what corpus Latin actually
    looks like.

Both kinds of line sit among citation links dense with
``<i>Genus species</i>``, so the patterns are anchored: a headword is
``<b><i>…</i></b>``, an attested form is an indented ``<i>…</i>``
followed by ``-`` and a lowercase grammatical reading.

Usage::

    bin/botanical_latin_wordlist \\
        --source-dir ~/www/https/www.vfthomas.com/botanicalLatinwords/ \\
        --output /tmp/botanical_latin_wordlist.txt
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Optional, Sequence, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_HEADWORD_RE = re.compile(r'<b><i>([^<]+)</i></b>')
_ATTESTED_RE = re.compile(
    r'&nbsp;<i>([^<]+)</i>\s*-\s*[a-z]'
)

# Nominative endings a "-SUF" part may replace, longest first so the
# longest legitimate ending wins.
_NOMINATIVE_ENDINGS = (
    'us', 'os', 'um', 'on', 'is', 'es', 'as', 'ns', 'or', 'ix',
    'ex', 'er', 'a', 'e', 'o', 'i', 's', 'n', 'r', 'x',
)

_MIN_LENGTH = 2


def parse_headwords(page_html: str) -> List[str]:
    """Every ``<b><i>…</i></b>`` headword, in page order."""
    return [m.group(1).strip() for m in _HEADWORD_RE.finditer(page_html)]


def _fold(word: str) -> Optional[str]:
    """Lowercase, strip combining marks, reject non-alphabetic.

    Folding applies to the word list only — never to treatment text,
    where it would invalidate stored span offsets.
    """
    decomposed = unicodedata.normalize('NFD', word)
    stripped = ''.join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    ).lower().strip()
    if not stripped.isalpha() or not stripped.isascii():
        return None
    if len(stripped) < _MIN_LENGTH:
        return None
    return stripped


def _stem_for(lemma: str, suffix: str) -> str:
    """Strip the nominative ending ``suffix`` is meant to replace."""
    for ending in _NOMINATIVE_ENDINGS:
        if lemma.endswith(ending) and len(lemma) > len(ending):
            return lemma[:-len(ending)]
    return lemma


def expand_headword(headword: str) -> Set[str]:
    """Expand one headword into every form it names.

    Parts are comma-separated.  A part beginning ``-`` is an ending
    applied to the first part's stem; anything else is a form in its
    own right.
    """
    parts = [p.strip() for p in headword.split(',') if p.strip()]
    if not parts:
        return set()
    out: Set[str] = set()
    lemma = None
    for part in parts:
        if part.startswith('-'):
            if lemma is None:
                continue
            candidate = _stem_for(lemma, part[1:]) + part[1:]
        else:
            candidate = part
            if lemma is None:
                lemma = _fold(part) or part
        folded = _fold(candidate)
        if folded:
            out.add(folded)
    return out


def attested_forms(page_html: str) -> Set[str]:
    """Inflected forms attested in *Species Plantarum*."""
    out: Set[str] = set()
    for match in _ATTESTED_RE.finditer(page_html):
        folded = _fold(match.group(1))
        if folded:
            out.add(folded)
    return out


def build_wordlist(page_html: str) -> List[str]:
    """Sorted union of expanded headwords and attested forms."""
    words: Set[str] = set()
    for headword in parse_headwords(page_html):
        words |= expand_headword(headword)
    words |= attested_forms(page_html)
    return sorted(words)


def wordlist_from_dir(source_dir: Path) -> List[str]:
    """Merge every ``*words.htm`` page under ``source_dir``."""
    pages = sorted(source_dir.glob('*words.htm'))
    if not pages:
        raise ValueError(f"no *words.htm pages under {source_dir}")
    words: Set[str] = set()
    for page in pages:
        words.update(build_wordlist(
            page.read_text(encoding='utf-8', errors='replace')
        ))
    return sorted(words)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--source-dir', required=True, metavar='DIR',
        help='Local mirror of the Botanical Latin Words pages.',
    )
    parser.add_argument(
        '--output', required=True, metavar='PATH',
        help='Word list to write, one folded form per line.',
    )
    args = parser.parse_args(argv)

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
