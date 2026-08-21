# `latin_wordlist.txt` — source, licence, and provenance

## What it is

83 779 Latin word forms, one per line, lowercased and folded to
ASCII.  Built by
[`bin/extract_latin_wordlist.py`](../bin/extract_latin_wordlist.py)
from a local mirror of the Latdict A–Z word-list pages.

Unlike a lemma list, this carries **inflected forms**: each Latdict
entry gives the dictionary's principal parts, so `albus, alba,
album` contributes three forms and `kalo, kalare, kalavi, kalatus`
contributes four.  39 224 entries expand to 83 779 distinct folded
forms.

## Source

> **Latdict — Latin Dictionary and Grammar Resources**
> <https://latin-dictionary.net/>
> Kevin D. Mahoney (@kabojnk) and the Latdict Group.
> Word-list pages `https://latin-dictionary.net/list/letter/{a…z}`.
> Retrieved 2026-08-21.

Latdict credits the late **USAF Col. William Whitaker**, whose
*WORDS* Latin dictionary is the lexicon most Latdict data derives
from.  Whitaker released *WORDS* into the **public domain**.

## Licence — read before redistributing

**The Latdict site footer reads:**

> Site copyright © 2002-2026 Kevin D. Mahoney (@kabojnk) and the
> Latdict Group. **All rights reserved.**

That notice covers the site.  It is not the same question as whether
this derived word list may be redistributed, and the two should not
be conflated:

* The list contains **no definitions, glosses, examples, or
  arrangement** from Latdict — only the headword forms, folded and
  sorted alphabetically.  What survives is the vocabulary of a
  language, which is fact rather than authorship.
* The underlying lexicon appears to descend from Whitaker's *WORDS*,
  which is public domain.  If so, the forms themselves carry no
  restriction; only Latdict's presentation of them would.

**Neither point is a legal opinion, and neither has been confirmed
with the rights holder.**  Before this repository is made public, do
one of:

1. **Ask.** Kevin D. Mahoney is reachable through the site; a short
   note asking whether an extracted, definition-free form list may
   be redistributed would settle it.
2. **Re-source from Whitaker's *WORDS* directly** (`DICTPAGE.RAW`,
   public domain).  This gives materially the same forms with clean
   provenance and removes the question entirely.  **Preferred.**
3. **Drop the file** and generate the vocabulary from the SKOL
   corpus instead — see "Alternative" below, which measures *better*
   than this list on the task it was built for.

Until one of those is done, treat this file as **provisional**.

## Why it exists

The §9 mode-B OCR detector (`D8` in
[`docs/data_quality_production_v4_model.md`](../docs/data_quality_production_v4_model.md))
finds OCR space-displacement by testing whether a run of
out-of-vocabulary tokens rejoins into a real word.  It is therefore
blind to any language it lacks vocabulary for, and Latin diagnoses
run **79.5 %** out-of-vocabulary against an English dictionary.
Without Latin, a corrupted Latin passage scored 1.33 % — below the
2 % detection floor — while identical corruption in English scored
18.18 %.

## Measured results

Against the 82 Latin word forms in `taxon_d2a4c584`'s Latin half,
and the corrupted-Latin / `taxon_43a7b19e` / poster-child battery:

| vocabulary | size | Latin cov. | corrupt-Latin | taxon_43a7b19e | worst poster child |
|---|---:|---:|---:|---:|---:|
| English only | — | 0.0 % | 1.33 % ✗ | 19.13 % | 0.00 % |
| + `dict-freedict-lat-eng` | 2 296 | 6.1 % | 1.33 % ✗ | 19.13 % | 0.00 % |
| **+ this list** | **83 779** | **35.4 %** | **8.00 %** ✓ | 19.20 % | 0.00 % |
| + corpus-derived (df ≥ 50) | 4 269 | 62.2 % | 22.67 % ✓ | 22.19 % | 0.00 % |
| + both | 87 452 | 64.6 % | 22.67 % ✓ | 21.81 % | 0.00 % |

This list **works where FreeDict does not** — 6× the coverage, and
it lifts corrupted Latin over the detection floor.  Adding 83 779
words introduces **no** false positives: every poster child stays at
0.00 %.

## Alternative, and why it may be better

The corpus-derived vocabulary — out-of-vocabulary forms of 4+
characters appearing in ≥ 50 distinct `treatments_prose` documents —
is **4 269 forms**, 5 % the size, and scores *higher* (62.2 % vs
35.4 % coverage, 22.67 % vs 8.00 %).  It wins because the Latin this
corpus actually uses is descriptive neo-Latin, not classical:
`basidiomata`, `acanthocystides`, `adscendentes`,
`brunneovinescens` are absent from Latdict but common here.

Combining both adds little over the corpus list alone (64.6 % vs
62.2 %, identical detection numbers).

The corpus list has its own hazard — it is derived from a corpus
containing OCR-corrupted treatments, so a systematic corruption
recurring across 50+ documents could enter it.  The two sources have
**independent** failure modes, which is the one real argument for
keeping both: a form in Latdict is verified Latin regardless of what
the corpus contains.

## Reproducing

```sh
bin/extract_latin_wordlist \
    --source-dir ~/www/http/latin-dictionary.net/list/letter/ \
    --output data/latin_wordlist.txt
```

The mirror itself is **not** checked in (15 MB of HTML).  Folding —
lowercase, macron stripping, `æ`/`œ` expansion — is applied to the
**word list only**.  Never fold treatment text: rewriting it would
invalidate every stored `*_spans` offset.
