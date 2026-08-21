# DCC Greek core vocabulary

`greek-core-list.csv` — 524 headwords, the Dickinson College
Commentaries **Greek Core Vocabulary**: the most frequent words of
classical Greek prose, with definitions, part of speech, semantic
group and frequency rank.

> **Dickinson College Commentaries**, Greek Core Vocabulary.
> <http://dcc.dickinson.edu/greek-core-list>
> Christopher Francese *et al.*, Dickinson College.

**Licence not yet confirmed.**  DCC materials are generally released
under Creative Commons (CC BY-SA), but that has not been verified for
this file against the source page.  Confirm before redistributing —
same caution that applied to the abandoned Latdict list.

## Derived files

`greek-core-latinized.csv`
    The source CSV with two columns appended by
    [`bin/latinize_greek.py`](../../bin/latinize_greek.py):
    `Transliterated` (letters only) and `Latinized` (with ICN
    Rec. 60A terminations, `-ος`→`-us`, `-ον`→`-um`).

`greek-core-wordlist.txt`
    873 distinct forms, one per line, from both columns.

## It does not help the D8 detector — and why

This vocabulary was Latinized to fill the Greek-rooted gap in
`data/latin_wordlist.txt`: neo-Latin coinages like *basidium*,
*ascus*, *pileus* and *acanthocystides* are absent from Whitaker's
WORDS at every age.  Measured, it fills none of it:

| vocabulary | Latin cov. | corrupt-Latin | taxon_43a7b19e |
|---|---:|---:|---:|
| English + WORDS | 58.5 % | 14.67 % | 16.22 % |
| **+ this list** | 58.5 % | 14.67 % | 16.22 % |
| English + corpus df ≥ 50 | 62.2 % | 22.67 % | 22.19 % |
| **+ this list** | 62.2 % | 22.67 % | 21.88 % |

873 forms, 666 of them new against English + WORDS, and **zero** of
the 82 Latin word forms in `taxon_d2a4c584` covered.  Not one
detection number moves.

The reason is a corpus mismatch, not a defect in the Latinizer.  A
*core* vocabulary is the ~500 most **frequent** words of classical
prose — ἀγαθός, ἄνθρωπος, λέγω, πόλις.  The Greek this corpus needs
is **technical and infrequent**: βασίδιον, ἀσκός, μύκης, σπόρα,
πίλος.  Technical roots are by definition not in a frequency-ranked
core list.

The Latinizer itself handles those correctly — `βασίδιον` →
`basidium`, `ἀσκός` → `ascus`, `μύκης` → `myces`, `ῥίζα` → `rhiza`
are all in its test suite.  What is missing is a Greek **botanical
root** list to feed it.  Candidates: Stearn's *Botanical Latin*
vocabulary, or harvesting Greek-derived stems from the corpus
directly.

Kept because the Latinizer is worth having and this file is its
regression input, not because it earns a place in the D8 vocabulary.
