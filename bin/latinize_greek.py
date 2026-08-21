#!/usr/bin/env python3
"""Latinize Greek words by the standard scholarly rules.

This is the convention botanical and mycological nomenclature
follows (ICN Rec. 60A), and it is *not* the same as modern Greek
romanization: Latin has no ``k``, ``w`` or standalone ``y``-as-vowel
outside Greek loans, so κ becomes ``c`` and υ becomes ``y`` only when
it is not part of a diphthong.

Motivation: the §9 mode-B OCR detector (``D8`` in
``docs/data_quality_production_v4_model.md``) needs Latin vocabulary,
and the neo-Latin coinages this corpus lives on — *basidium*,
*ascus*, *pileus*, *acanthocystides* — are Greek-rooted and absent
from Whitaker's WORDS at every age.  Latinizing a Greek core
vocabulary fills that gap.

The rules
=========

**Diacritics.**  Polytonic Greek is precomposed (``ὁ`` is one
character, not ``ο`` + breathing), which defeats any bare-letter
rule table.  Decompose first (NFD), then:

* **dasia** (rough breathing, U+0314) is the only mark that carries
  sound — it becomes ``h`` before the vowel, or ``rh`` after ρ.
  Read it *before* stripping marks.
* psili (smooth breathing), tonos/oxia, varia, perispomeni,
  ypogegrammeni (iota subscript) are all silent — strip them.
* dialytika (diaeresis) marks a vowel pair that is *not* a
  diphthong, so it must block diphthong formation rather than
  simply vanish.

**Clusters, applied before single letters.**

===========  ========  ==================================
γ + γ κ χ ξ  n         ἄγγελος→angelus, ἄγκυρα→ancora,
                       σφίγξ→sphinx
ρρ           rrh       διάρροια→diarrhoea
initial ρ    rh        ῥίζα→rhiza
===========  ========  ==================================

**Diphthongs.**

====  ===  =====================================
αι    ae
οι    oe
ει    i    χείρ→chir- (chiroptera)
ου    u
αυ    au   υ is ``u`` in a diphthong, not ``y``
ευ    eu
ηυ    eu
υι    yi
====  ===  =====================================

**Single letters.**  α a, β b, γ g, δ d, ε e, ζ z, η e, θ th,
ι i, **κ c**, λ l, μ m, ν n, ξ x, ο o, π p, ρ r, σ/ς s, τ t,
**υ y**, φ ph, χ ch, ψ ps, ω o.

**Terminations** (ICN Rec. 60A): ``-ος`` → ``-us``, ``-ον`` →
``-um``.  Applied by :func:`latin_terminations`, which
:func:`transliterate` does *not* call — the bare transliteration is
often what a compound needs.

Usage::

    bin/latinize_greek --input data/dcc/greek-core-list.csv \\
        --output data/dcc/greek-core-latinized.csv
"""

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_DASIA = '̔'          # rough breathing
_DIALYTIKA = '̈'      # diaeresis: blocks a diphthong
_SEPARATOR = '\x00'        # internal marker for "not a diphthong"

# Longest first, so clusters win over their own prefixes.
_DIPHTHONGS = (
    ('αι', 'ae'), ('οι', 'oe'), ('ει', 'i'), ('ου', 'u'),
    ('ηυ', 'eu'), ('αυ', 'au'), ('ευ', 'eu'), ('υι', 'yi'),
)

_LETTERS = {
    'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ε': 'e', 'ζ': 'z',
    'η': 'e', 'θ': 'th', 'ι': 'i', 'κ': 'c', 'λ': 'l', 'μ': 'm',
    'ν': 'n', 'ξ': 'x', 'ο': 'o', 'π': 'p', 'ρ': 'r', 'σ': 's',
    'ς': 's', 'τ': 't', 'υ': 'y', 'φ': 'ph', 'χ': 'ch', 'ψ': 'ps',
    'ω': 'o',
}

_VELARS = 'γκχξ'

_TERMINATIONS = (('os', 'us'), ('on', 'um'))


def strip_diacritics(text: str) -> Tuple[str, bool]:
    """Return ``(bare_greek, has_rough_breathing)``.

    Decomposes, notes whether a dasia is present, drops every other
    combining mark, and lowercases.  A
    dialytika is replaced by an internal separator so the vowel pair
    it marks is not later read as a diphthong.
    """
    decomposed = unicodedata.normalize('NFD', text)
    rough = _DASIA in decomposed
    out = []
    for ch in decomposed:
        if ch == _DIALYTIKA:
            out.append(_SEPARATOR)
            continue
        if unicodedata.combining(ch):
            continue
        out.append(ch)
    # Final sigma is left alone: it is orthography, not a
    # diacritic, and _LETTERS maps both forms to 's'.
    return ''.join(out).lower(), rough


def _apply_clusters(text: str) -> str:
    """γ-nasal, doubled rho, then the diphthongs."""
    text = re.sub(rf'γ(?=[{_VELARS}])', 'ν', text)
    text = text.replace('ρρ', 'ρ\x01')          # \x01 -> 'rh' later
    for greek, latin in _DIPHTHONGS:
        # A separator between the vowels blocks the diphthong.
        text = text.replace(greek, f'\x02{latin}\x02')
    return text


def transliterate(word: str) -> str:
    """Greek to Latin letters, without termination conversion."""
    bare, rough = strip_diacritics(word)
    if not bare:
        return ''
    text = _apply_clusters(bare)
    out = []
    for ch in text:
        if ch == '\x01':
            out.append('rh')
        elif ch in ('\x02', _SEPARATOR):
            continue
        else:
            out.append(_LETTERS.get(ch, ch if ch.isascii() else ''))
    result = ''.join(out)
    if result.startswith('r'):          # initial rho is aspirated
        result = 'rh' + result[1:]
    elif rough:
        result = 'h' + result
    return result


def latin_terminations(word: str) -> str:
    """Apply ICN Rec. 60A endings: ``-os``→``-us``, ``-on``→``-um``."""
    for greek_end, latin_end in _TERMINATIONS:
        if word.endswith(greek_end) and len(word) > len(greek_end):
            return word[:-len(greek_end)] + latin_end
    return word


def latinize(text: str) -> str:
    """Transliterate and Latinize every whitespace-separated word."""
    words = [w for w in text.split() if w.strip()]
    return ' '.join(
        latin_terminations(transliterate(w)) for w in words
    ).strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--input', required=True, metavar='CSV',
        help='CSV with a Headword column of Greek forms.',
    )
    parser.add_argument(
        '--output', required=True, metavar='CSV',
        help='CSV to write, with Transliterated and Latinized '
             'columns appended.',
    )
    parser.add_argument(
        '--wordlist', metavar='PATH',
        help='Also write a plain one-form-per-line word list, for '
             'use as D8 vocabulary.',
    )
    args = parser.parse_args(argv)

    in_path = Path(args.input).expanduser()
    if not in_path.is_file():
        print(f"error: {in_path} not found", file=sys.stderr)
        return 2
    with in_path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"error: {in_path} has no rows", file=sys.stderr)
        return 2
    if 'Headword' not in rows[0]:
        print(f"error: {in_path} has no Headword column; found "
              f"{sorted(rows[0])}", file=sys.stderr)
        return 2

    words = set()
    for row in rows:
        head = row.get('Headword') or ''
        row['Transliterated'] = ' '.join(
            transliterate(w) for w in head.split()
        ).strip()
        row['Latinized'] = latinize(head)
        for field in ('Transliterated', 'Latinized'):
            for form in row[field].split():
                if len(form) >= 2 and form.isalpha():
                    words.add(form)

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with out_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} headwords -> {out_path}")

    if args.wordlist:
        wl_path = Path(args.wordlist).expanduser()
        wl_path.parent.mkdir(parents=True, exist_ok=True)
        with wl_path.open('w', encoding='utf-8') as handle:
            for word in sorted(words):
                handle.write(word + '\n')
        print(f"{len(words)} distinct forms -> {wl_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
