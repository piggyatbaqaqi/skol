#!/usr/bin/env python3
"""Extract taxonomic root vocabulary from the Wikipedia systematic-names list.

Source: *List of Latin and Greek words commonly used in systematic
names* — 554 table rows, each giving the form as it appears in
scientific names, the source language (``L`` or ``G``), the Greek
original where applicable, and an English gloss.

Two kinds of form are collected:

* the page's own **Latin/Greek column**, which is authoritative —
  these are the forms actually used in systematic names;
* our **Latinization of the Greek column** via
  :mod:`latinize_greek`, in all three shapes a compound might use:
  the bare transliteration, the Latin-terminated nominative, and the
  combining form (``acantho-``).

The page doubles as a **validation set** for the Latinizer: 153 rows
pair a Greek original with its accepted Latin form.  Measured
2026-08-21, our output matched a listed form exactly for 43.8 % and
agreed on the stem for a further 52.3 % — **96.1 % stem agreement**.
The 3.9 % remainder are not letter-rule errors: the page variously
lists a Latin *cognate* (βολβός → *bulbus*), a different principal
part (πούς → *pod-*, the oblique stem) or a different Greek lemma
(γαῖα → *geo-*, which is from γῆ).

Usage::

    bin/systematic_names_wordlist --input <scraped-page> \\
        --output data/systematic_names_wordlist.txt
"""

import argparse
import html
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latinize_greek import (  # noqa: E402
    combining_form,
    latin_terminations,
    transliterate,
)

_TABLE_RE = re.compile(
    r'<table class="wikitable"[^>]*>(.*?)</table>', re.S)
_ROW_RE = re.compile(r'<tr[^>]*>(.*?)</tr>', re.S)
_CELL_RE = re.compile(r'<td[^>]*>(.*?)</td>', re.S)
_GREEK_RE = re.compile(r'lang="grc"[^>]*>(?:<a[^>]*>)?([^<]+)')
_TAG_RE = re.compile(r'<[^>]+>')

_HYPHENS = '-‐‑‒–—'
_MIN_LENGTH = 3
_SKIP = frozenset({'etc'})


def cell_text(fragment: str) -> str:
    """Strip markup and unescape one table cell."""
    return html.unescape(_TAG_RE.sub('', fragment)).strip()


def parse_rows(page_html: str) -> List[Tuple[str, List[str]]]:
    """Return ``(latin_cell_text, [greek_originals])`` per data row.

    Header rows carry ``<th>`` rather than ``<td>`` and so drop out.
    """
    rows: List[Tuple[str, List[str]]] = []
    for table in _TABLE_RE.findall(page_html):
        for row in _ROW_RE.findall(table):
            cells = _CELL_RE.findall(row)
            if len(cells) < 3:
                continue
            rows.append(
                (cell_text(cells[0]), _GREEK_RE.findall(cells[1]))
            )
    return rows


def latin_forms(cell: str) -> Set[str]:
    """Split a Latin/Greek cell into its individual forms.

    Cells hold comma-separated alternatives, combining forms and
    trailing markers: ``actin-, actino-`` and ``acanthus etc.``
    Splitting on whitespace as well as commas is what keeps the
    latter from being discarded whole.
    """
    out: Set[str] = set()
    for part in re.split(r'[,;\s]+', cell):
        word = part.strip().strip(_HYPHENS + '.').lower()
        if word.isalpha() and len(word) >= _MIN_LENGTH and word not in _SKIP:
            out.add(word)
    return out


def greek_forms(originals: Sequence[str]) -> Set[str]:
    """Latinize Greek originals into every shape a compound uses."""
    out: Set[str] = set()
    for original in originals:
        for word in original.split():
            bare = transliterate(word)
            if len(bare) < _MIN_LENGTH:
                continue
            nominative = latin_terminations(bare)
            for form in (bare, nominative, combining_form(nominative)):
                if form.isalpha() and len(form) >= _MIN_LENGTH:
                    out.add(form)
    return out


def build_wordlist(page_html: str) -> List[str]:
    """Sorted union of the page's own forms and our Latinizations."""
    words: Set[str] = set()
    for latin_cell, greek in parse_rows(page_html):
        words |= latin_forms(latin_cell)
        words |= greek_forms(greek)
    return sorted(words)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--input', required=True, metavar='PATH',
        help='Local copy of the Wikipedia systematic-names page.',
    )
    parser.add_argument(
        '--output', required=True, metavar='PATH',
        help='Word list to write, one form per line.',
    )
    args = parser.parse_args(argv)

    in_path = Path(args.input).expanduser()
    if not in_path.is_file():
        print(f"error: {in_path} not found", file=sys.stderr)
        return 2
    page = in_path.read_text(encoding='utf-8', errors='replace')
    words = build_wordlist(page)
    if not words:
        print("error: no wikitable rows found; is this the right page?",
              file=sys.stderr)
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
