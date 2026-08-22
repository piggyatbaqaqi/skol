# Botanical Latin word list (V. F. Thomas Co.) — **not checked in**

## Status: licence unresolved

> ***Botanical Latin Words***, a project of **V. F. Thomas Co.**,
> P. O. Box 84, Hulls Cove, Maine 04644 — info@vfthomas.com.
> <https://www.vfthomas.com/botanicalLatinwords/>
> 26 alphabetical pages, mirrored 2026-08-21 (site updated
> 28 October 2025).

**The site carries no copyright or licence notice** that the operator
or I could find.  The operator has written to the author.  Until a
reply arrives, **the derived word list is deliberately not committed**
— only
[`bin/botanical_latin_wordlist.py`](../bin/botanical_latin_wordlist.py),
which is our own code.  Regenerate locally:

```sh
bin/botanical_latin_wordlist \
    --source-dir ~/www/https/www.vfthomas.com/botanicalLatinwords/ \
    --output /tmp/botanical_latin.txt
```

Page numbers on the site cite the Ray Society's 1957 facsimile of the
first edition (1753) of Linnaeus's *Species Plantarum*.  The Linnaean
text is long out of copyright; the compilation, glossing and
grammatical analysis are the author's work.

## What it contains

**5 679 forms** from 1 436 headwords and **3 187 attested inflected
forms** read out of *Species Plantarum*, each with its grammatical
parse and page citation.

The attested forms are why this source matters.  Every other Latin
source tried here gives lemmas (FreeDict, DCC Greek core), roots (the
Wikipedia systematic-names list), or generated paradigms over the
wrong vocabulary (Whitaker's WORDS, which lacks *campanulatus* and
*glabra* entirely).  These are real descriptive-botanical
inflections — `baccis`, `abbreviatis`, `acaulibus` — which is exactly
what corpus Latin looks like.

## Measured against the corpus-derived vocabulary

| vocabulary | size | Latin cov. | corrupt-Latin | taxon_43a7b19e | worst poster child |
|---|---:|---:|---:|---:|---:|
| English only | — | 0.0 % | 1.33 % ✗ | 19.13 % | 0.00 % |
| + systematic-names | 849 | 8.5 % | 2.67 % | 19.28 % | 0.00 % |
| + WORDS | 899 973 | 58.5 % | 14.67 % | 16.22 % | 0.00 % |
| **+ this list** | **5 679** | 26.8 % | **13.33 %** | **19.13 %** | 0.61 % |
| + corpus df ≥ 50 | 4 269 | 62.2 % | 22.67 % | 22.19 % | 0.00 % |
| **+ this list AND corpus** | **9 471** | **65.9 %** | **24.00 %** | **22.19 %** | **0.00 %** |
| + WORDS + this + corpus | 905 799 | 78.0 % | 25.33 % | 18.67 % | 0.00 % |

### Three things worth noting

**It is 158× smaller than WORDS for 91 % of the effect.**  5 679
forms reach 13.33 % on the corrupted-Latin case where WORDS' 899 973
reach 14.67 % — and unlike WORDS it does **not** absorb real
corruption: `taxon_43a7b19e` stays at 19.13 % rather than dropping to
16.22 %.

**It is almost disjoint from the corpus vocabulary.**  Overlap is
just **477 forms, 11.2 %** of the corpus list.  5 202 forms are new
to it, 3 792 of the corpus list are new to this.  They are measuring
different things, which is why combining them is worth more than
either alone.

**It is cleaner than the corpus vocabulary**, which matters for the
circularity caution recorded in D8.  The corpus-only forms include
English technical terms (`abaxial`, `abhymenial`, `abscission`),
French (`abondantes`, `abord`) and truncation debris (`abun`,
`acad`).  This list is Latin throughout, verified against a printed
source.

### One caution

At 5 679 forms **alone** it is the first vocabulary to put a
poster child above zero, at **0.61 %** — still far below any
threshold in the 2–19 % band, and it returns to **0.00 %** once
combined with the corpus list.  Worth re-checking if the list is
ever used on its own.

## Recommendation

**Corpus df ≥ 50 + this list — 9 471 forms, roughly 90 KB — is the
best configuration measured.**  65.9 % Latin coverage, 24.00 % on
corrupted Latin, 22.19 % on `taxon_43a7b19e`, no false positives.

That makes the 11 MB `data/latin_wordlist.txt` (WORDS) **droppable**:
adding it raises Latin coverage to 78.0 % but *lowers* the real
detection case to 18.67 %, and it costs 900 000 forms to do so.

Contingent on the licence question being resolved.  If the author
declines, corpus df ≥ 50 alone remains the fallback at 62.2 % /
22.67 % / 22.19 %.
