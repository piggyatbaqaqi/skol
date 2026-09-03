#!/usr/bin/env python3
"""Recover the punctuation ``brat_safe_type`` had to remove.

``brat_safe_type`` replaces every character outside ``[A-Za-z0-9_-]``
with an underscore, because brat's storage regex demands it.  That is
correct for brat and lossy for the record: ``Schäffer's reaction`` is
stored as ``Sch ffer s reaction`` — a diacritic and an apostrophe gone
from a person's name — and ``conidium length/width ratio`` becomes
``Conidium length width ratio``, losing the relation the label was
describing.  The stored form is the *identity*; this module recovers a
form fit to *display*.

**The span is the witness.**  ``source_text`` is stored verbatim, so
the characters brat removed are usually still in it.  Matching the
label's words against its own span is evidence.  The alternative — a
"capitalised word + whitespace + naked s" heuristic — measured 0
correct out of 7 on the span corpus: six were OCR word-splitting and
one was ``sensu stricto``, and it is blind to owners whose name
already ends in s (``Gams'``, ``Fries'``).

**Case comes from the label, punctuation from the span.**  A label's
capitalisation is the annotator's choice of heading; the span's is an
accident of sentence position.  So recovery splices the label's own
words with the separators found between them.

What this does **not** do is repair the source.  ``Sabouraud ' s`` is
what the page says — OCR destroyed that apostrophe before any label
existed — and inventing it here would be a different job (§9's OCR
detectors) wearing this one's clothes.
"""

import re
from typing import List, Optional

# Bounded on purpose: an unbounded run lets `Spore print` match
# `Spore` and a `print` forty characters downstream and splice the two
# into a label that appears nowhere.
_SEPARATOR = r'[^A-Za-z0-9]{0,4}'


def recover_display_label(label: str, source_text: str) -> Optional[str]:
    """A richer form of ``label`` witnessed by ``source_text``.

    Returns ``None`` when there is nothing to recover — the label is
    absent from its span, or differs only in case or spacing, which is
    ``fold_case``'s business rather than this one's.
    """
    words: List[str] = [word for word in re.split(r'\s+', label.strip())
                        if word]
    if len(words) < 2 or not source_text:
        return None

    pattern = _SEPARATOR.join(re.escape(word) for word in words)
    match = re.search(pattern, source_text, re.IGNORECASE)
    if match is None:
        return None

    # Splice: the label's words, the span's separators.
    spliced: List[str] = []
    position = match.start()
    for index, word in enumerate(words):
        found = re.compile(re.escape(word), re.IGNORECASE).search(
            source_text, position)
        if found is None:              # pragma: no cover - regex already matched
            return None
        if index:
            spliced.append(source_text[position:found.start()])
        spliced.append(word)
        position = found.end()
    recovered = ''.join(spliced)

    # Only punctuation counts.  Collapsing whitespace first means a
    # doubled space or a non-breaking one is not mistaken for a find.
    if re.sub(r'\s+', ' ', recovered) == re.sub(r'\s+', ' ', label.strip()):
        return None
    return recovered
