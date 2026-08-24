#!/usr/bin/env python3
"""Classify OCR damage in a block of treatment text.

**OCR damage is not one thing**, and that is the whole reason this
module exists.  Three modes occur in this corpus, they are close to
independent, and a detector tuned on one is blind to the others:

* **Replacement characters** (§9 mode A) — runs of U+FFFD where the
  decoder could not interpret the bytes.  The original §9 detector
  keyed on this alone.
* **Space displacement** (§9 mode B) — spaces landing inside words:
  ``f ung i wi t h fissi on``.  Contains zero U+FFFD, so mode A cannot
  see it.
* **Character substitution** (§9 mode C) — letters swapped inside
  otherwise well-spaced words: ``Oidiode11dron``, ``RaiUo``,
  ``KJocker``.  Contains zero U+FFFD *and* rejoins at baseline, so
  neither of the other two can see it.

`taxon_8d815304` is the case that forced the distinction: severe
mode-C damage, yet its windowed rejoin rate reads 2.2 % against a
corpus p90 of 5.7 %.  A rejoin-only gate calls it clean.

Two measurement traps are handled here rather than left to callers,
because both were hit during the analysis that produced this module:

* **Rejoin must be windowed, not pairwise.**  ``Ana mo rph`` needs a
  three-token window and ``A n am o r ph`` a six-token one; no
  adjacent *pair* of those fragments is a word, so a pairwise
  implementation scores them 0.
* **Substitution must exclude figure references, micrometre
  measurements and accession numbers.**  Without those exclusions the
  metric ranks modern molecular papers above scanned monographs —
  ``Fig. 2C``, ``4-7um`` and ``KY784257`` all look like intra-word
  corruption.

`oov_rate` is reported but deliberately **never fires a mode**: it is
contaminated by proper nouns, and nomenclature-dense text reaches
31.6 % legitimately because genus and author names are absent from any
dictionary.

Thresholds are measured corpus quantiles, not guesses — see
``docs/data_quality_production_v4_model.md`` §9 and D8.

**Two known gaps, both measured rather than suspected:**

* **Garbling beyond recognition is not a mode.**  Text like
  ``i m< ta l< trongly farin ( om lamellae adnate`` is obviously
  ruined to a human, but it neither rejoins (the fragments are not
  words) nor trips the substitution signatures often enough to cross
  the threshold.  `oov_rate` would catch it, and is the reason that
  rate is computed at all — but it cannot be promoted to a mode while
  proper nouns contaminate it.
* **These rates describe what SURVIVED extraction, not the source.**
  On a shredded document the mangled passages are routed to
  ``Table``/``Key``/``Misc-exposition`` and never reach the treatment
  fields, so the remnant reads clean.  Measured on `taxon_a686d7ab`:
  the treatment text scores 0.0 % rejoin over 384 tokens while the
  source region it was drawn from holds 2 222 tokens — 5.8× more —
  at 2.4 %.  On `taxon_a5efbd0b` the ratio is 25×.  **To assess a
  source, pass the resolved attachment region, not the treatment.**

Usage::

    from treatments_to_structured.ocr_damage import OcrDamage, load_vocabulary

    vocab = load_vocabulary()
    damage = OcrDamage(treatment['description'], vocabulary=vocab)
    if damage.modes():
        print(damage.profile())
"""

import dataclasses
import re
from pathlib import Path
from typing import AbstractSet, List, Optional, Sequence, Set, Tuple

MODE_REPLACEMENT = 'replacement-char'
MODE_SPACING = 'space-displacement'
MODE_SUBSTITUTION = 'character-substitution'

# Corpus quantiles measured 2026-08-24 over 48 738 treatments.
# Windowed rejoin: median 0.0 %, p90 5.7 %, p99 20.0 %.
_REJOIN_THRESHOLD = 8.0
# Intra-word corruption (after the exclusions below): median 0.74 %,
# p90 4.17 %, p99 11.96 %.
_SUBSTITUTION_THRESHOLD = 4.0
# Any replacement character at all is damage; the rate only grades it.
_REPLACEMENT_THRESHOLD = 0.0

_MIN_TOKENS = 40

_DEFAULT_VOCAB_FILES = (
    'data/corpus_vocabulary.txt',
    'data/botanical_latin_wordlist.txt',
    '/usr/share/dict/american-english',
    '/usr/share/dict/british-english',
)

_WORD = re.compile(r'[A-Za-z]+')
_TOKEN = re.compile(r'\S+')
_REPLACEMENT = '�'

# Intra-word corruption signatures.
_DIGIT_IN_WORD = re.compile(r'[A-Za-z]\d|\d[A-Za-z]')
_CAPITAL_IN_WORD = re.compile(r'[a-z][A-Z]')
_JUNK_IN_WORD = re.compile(r'[A-Za-z][·•~^|\\<>]|[·•~^|\\<>][A-Za-z]')

# Legitimate constructions that look like intra-word corruption.
# Each was an observed false positive, not a hypothetical one.
_NOT_DAMAGE = (
    re.compile(r'^\(?(?:figs?|pl|plate|tab|table)\.?$', re.I),
    re.compile(r'^\(?\d+[A-Za-z]\)?[.,;:]?$'),          # 2C, 6A, (3E)
    re.compile(r'^[\d.–—-]*\d\s*[uµ]m\)?[.,;:]?$', re.I),  # 4-7um
    re.compile(r'^\d+[x×]\d+[uµ]m\)?[.,;:]?$', re.I),           # 10x3um
    re.compile(r'^[A-Z]{1,3}\d{4,}[.,;:]?$'),           # KY784257
    re.compile(r'^(?:its|tef|rpb|lsu|ssu|hsp|btub)\d?[.,;:]?$', re.I),
)


def load_vocabulary(
    paths: Sequence[str] = _DEFAULT_VOCAB_FILES,
    *,
    root: Optional[Path] = None,
) -> Set[str]:
    """Load the production vocabulary, skipping files that are absent.

    The production decision (operator, 2026-08-22) is **corpus
    df ≥ 50 plus botanical Latin**, with `wamerican` *and* `wbritish`
    — not all authors writing in English use American spelling.

    Missing files are skipped rather than raising: the dictionaries
    are packaged dependencies that may be absent on a dev box, and a
    vocabulary that is merely smaller degrades the rates gracefully.
    """
    if root is None:
        root = Path(__file__).resolve().parent.parent
    words: Set[str] = set()
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        try:
            with path.open(encoding='utf-8') as handle:
                words |= {
                    line.strip().lower() for line in handle
                    if line.strip() and not line.startswith('#')
                }
        except OSError:
            continue
    return words


@dataclasses.dataclass(frozen=True)
class DamageProfile:
    """Every rate plus the modes that fired, for reporting."""

    n_tokens: int
    replacement_rate: float
    rejoin_rate: float
    substitution_rate: float
    oov_rate: float
    modes: Tuple[str, ...]


class OcrDamage:
    """OCR-damage rates for one block of text.

    Args:
        text: The text to assess — typically a treatment's
            ``description``, or its prose fields joined.
        vocabulary: Lowercased known words.  Without one, the
            vocabulary-dependent rates (rejoin, OOV) read 0; the
            substitution and replacement rates are unaffected.
        min_tokens: Below this, no rate is reported.  Short text
            produces rates dominated by a single token.
    """

    def __init__(
        self,
        text: str,
        *,
        vocabulary: Optional[AbstractSet[str]] = None,
        min_tokens: int = _MIN_TOKENS,
    ) -> None:
        self._text = text or ''
        self._vocab = vocabulary if vocabulary is not None else frozenset()
        self._min_tokens = min_tokens
        self._tokens = [
            t for t in _TOKEN.findall(self._text) if _WORD.search(t)
        ]
        self._fragments = _WORD.findall(self._text)

    @property
    def measurable(self) -> bool:
        """Whether there is enough text for the rates to mean anything."""
        return len(self._tokens) >= self._min_tokens

    @property
    def n_tokens(self) -> int:
        """Whitespace-delimited tokens containing at least one letter."""
        return len(self._tokens)

    def replacement_rate(self) -> float:
        """Percentage of characters that are U+FFFD (§9 mode A)."""
        if not self.measurable or not self._text:
            return 0.0
        return 100.0 * self._text.count(_REPLACEMENT) / len(self._text)

    def rejoin_rate(
        self,
        *,
        max_window: int = 6,
        fragment_len: int = 3,
    ) -> float:
        """Percentage of fragment runs that merge into a known word.

        §9 mode B.  Scans windows of up to ``max_window`` consecutive
        short fragments, **not** adjacent pairs: ``Ana mo rph`` needs
        three and ``A n am o r ph`` six, and pairwise scores both 0.
        The default window is 6 for exactly that second case.
        """
        if not self.measurable or not self._vocab:
            return 0.0
        merged, runs = self._rejoin_counts(max_window, fragment_len)
        return 100.0 * merged / runs if runs else 0.0

    def _rejoin_counts(
        self, max_window: int, fragment_len: int,
    ) -> Tuple[int, int]:
        frags = self._fragments
        runs = merged = 0
        index = 0
        while index < len(frags):
            if len(frags[index]) > fragment_len:
                index += 1
                continue
            end = index
            while end < len(frags) and len(frags[end]) <= fragment_len:
                end += 1
            run: List[str] = frags[index:end]
            # A run often ends on the word's final, longer syllable
            # ('Ana mo rph'), so extend one token right.
            if end < len(frags) and len(run) >= 2 and len(frags[end]) <= 6:
                run = run + [frags[end]]
                end += 1
            if len(run) >= 2:
                runs += 1
                if self._run_merges(run, max_window):
                    merged += 1
            index = end
        return merged, runs

    def _run_merges(self, run: Sequence[str], max_window: int) -> bool:
        for width in range(min(max_window, len(run)), 1, -1):
            for start in range(len(run) - width + 1):
                joined = ''.join(run[start:start + width]).lower()
                if len(joined) >= 5 and joined in self._vocab:
                    return True
        return False

    def substitution_rate(self) -> float:
        """Percentage of tokens showing intra-word corruption (§9 mode C).

        Excludes figure references, micrometre measurements and
        accession numbers — all observed false positives, and together
        enough to invert the ranking between scanned monographs and
        modern molecular papers.
        """
        if not self.measurable:
            return 0.0
        hits = sum(1 for token in self._tokens if self._is_corrupt(token))
        return 100.0 * hits / len(self._tokens)

    @staticmethod
    def _is_corrupt(token: str) -> bool:
        core = token.strip('.,;:()[]{}"\'')
        if not core:
            return False
        if any(pattern.match(core) for pattern in _NOT_DAMAGE):
            return False
        if _DIGIT_IN_WORD.search(core) or _JUNK_IN_WORD.search(core):
            return True
        match = _CAPITAL_IN_WORD.search(core)
        if match is None:
            return False
        # An interior capital is damage only if it does not look like a
        # camelCase compound: 'GenBank' and 'MycoBank' split into two
        # plausible words, whereas 'RaiUo' leaves a 2-character tail.
        tail = core[match.start() + 1:]
        return not (len(tail) >= 4 and tail[1:].islower())

    def oov_rate(self) -> float:
        """Percentage of word-tokens absent from the vocabulary.

        **Reported, never used to fire a mode.**  It is contaminated by
        proper nouns — genus names, author names, journal abbreviations
        — so nomenclature-dense text reaches 31.6 % while being
        perfectly clean.
        """
        if not self.measurable or not self._vocab:
            return 0.0
        words = [w for w in self._fragments if len(w) > 1]
        if not words:
            return 0.0
        unknown = sum(1 for w in words if w.lower() not in self._vocab)
        return 100.0 * unknown / len(words)

    def modes(self) -> Tuple[str, ...]:
        """Which damage modes exceed their measured thresholds.

        Order is stable (replacement, spacing, substitution) so the
        result can be compared and stored directly.
        """
        if not self.measurable:
            return ()
        found: List[str] = []
        if self.replacement_rate() > _REPLACEMENT_THRESHOLD:
            found.append(MODE_REPLACEMENT)
        if self.rejoin_rate() >= _REJOIN_THRESHOLD:
            found.append(MODE_SPACING)
        if self.substitution_rate() >= _SUBSTITUTION_THRESHOLD:
            found.append(MODE_SUBSTITUTION)
        return tuple(found)

    def profile(self) -> DamageProfile:
        """All rates and modes in one immutable record."""
        return DamageProfile(
            n_tokens=self.n_tokens,
            replacement_rate=self.replacement_rate(),
            rejoin_rate=self.rejoin_rate(),
            substitution_rate=self.substitution_rate(),
            oov_rate=self.oov_rate(),
            modes=self.modes(),
        )
