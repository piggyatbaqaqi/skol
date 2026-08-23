# Annotation-round selections

One file per bootstrap-annotation round: `production_v4_roundN.txt`,
holding the treatment IDs that round covered, one per line.

Written automatically by
[`bin/select_for_annotation.py`](../../bin/select_for_annotation.py),
which defaults `--output` here and picks the next free round number.
The plain-ID format is what
[`bin/llm_annotate_features.py`](../../bin/llm_annotate_features.py)
reads on stdin, so a file can be piped straight back in:

```sh
bin/llm_annotate_features --experiment production_v4 \
    < data/annotation_rounds/production_v4_round4.txt
```

**No comments or blank-line markers.** `read_treatment_ids` treats
every non-blank line as an ID, so a `#` line would be read as a
treatment. Provenance notes belong in this README, not in the files.

## Why these files exist

A selection is not reproducible from `--seed` for long.
`--exclude-annotated` reads the candidate DB live, so once the
annotator has run, re-running the same command with the same seed
returns a *different* set. The file is the only durable record of
what a round actually covered.

## Provenance

| round | IDs | selection | source | reviewed |
|---|---:|---|---|---|
| 1 | 6 | **biased** | reconstructed 2026-08-20 | 5 on 2026-07-01, 1 on 2026-07-03 |
| 2 | 48 | **biased** | reconstructed 2026-08-20 | 48 on 2026-07-03 |
| 3 | 9 | **random** | reconstructed 2026-08-20 | 9 on 2026-07-07 |
| 4 | 50 | **biased** | captured from the selector, 2026-08-14 | 24 of 50 on 2026-08-23 |

## Only round 3 is a random sample

**This is the single most important fact about these files.**
Rounds 1, 2 and 4 were selected with a bias toward suspected
problems; **round 3 was a random sample** (operator,
2026-08-23).  Any statistic pooled across rounds therefore
describes *the selection*, not the corpus, and pooling the
biased rounds with round 3 produces a number that means
nothing in particular.

Measured 2026-08-23, after `bin/brat_ingest` landed the first
24 treatments of round 4.  *Precision* = candidate
annotations the reviewer kept; *recall* = kept ÷ (kept +
hand-added).  Rejections appear as **absence** from the hand
DB — `brat_ingest` writes only `kept` and `added` — so
rejections are counted by set difference against
`features_candidate`.

| round | selection | n | cand | rej | added | precision | recall |
|---|---|---:|---:|---:|---:|---|---|
| 1 | biased | 5 | 81 | 0 | 142 | 100 % | **36.3 %** |
| 2 | biased | 48 | 900 | 13 | 240 | 98.6 % | 78.7 % |
| **3** | **random** | 9 | 96 | **0** | **1** | **100 %** [96.2, 100] | **99.0 %** [94.4, 99.8] |
| 4 | biased | 24 | 258 | 7 | 16 | 97.3 % | 94.0 % |
| — | pooled | 85 | 1316 | 20 | 263 | 98.5 % | 83.1 % |

Wilson 95 % intervals.  **Use the round-3 row for any claim
about corpus-level label quality; use the pooled row for
nothing.**

The gradient tracks selection bias exactly, and the
mechanism is concentrated rather than diffuse: **`taxon_2b793602`
alone contributes 136 of all 263 additions (52 %)**, and the
top five treatments contribute 69.6 %.  That treatment is the
flora-chapter-slice key recorded as
`§8-flora-chapter-slice-unnumbered-key` — a key body whose
dozens of short clauses each want an annotation.  Round 1's
36.3 % recall is very nearly that one document.

**What this means operationally.**  On random corpus material
the Claude API labelling is *already at ceiling* — 96 of 96
kept, one miss.  Continuing a biased round does not sharpen
that estimate, because it does not sample the same
population.  The two questions have come apart and should be
run as separate activities:

* **Label validation** — needs a fresh **random** sample.  It
  is also cheap: round 3 took 96 annotations and one
  addition, where round 1 took 81 candidates and 142
  additions.  Random treatments are far faster to review than
  selected-pathological ones.
* **Detector / pathology evidence** — what the biased rounds
  are actually good for, and what rounds 2 and 4 have in fact
  produced.  Keep doing it, but bill it as pathology hunting,
  not as label confirmation.

**Rounds 1–3 are reconstructed**, not original. They predate the
selector writing files at all, and were rebuilt from the `.ann`
filenames in
`brat/data/skol_segments/production_v4_round{1,2,3}/`. Two
consequences:

* They record what was **exported to brat** — i.e. what the annotator
  produced annotations for — which is not necessarily what was
  *selected*. A selected treatment that yielded zero annotations
  would be absent. Round 4 shows this is a real gap: 3 of its 50 have
  `annotation_count: 0` and would vanish from a filename-based
  reconstruction.
* Ordering is lost. Round 4 is in selection order (band by band);
  rounds 1–3 are sorted, because filenames carry no order.

Their union is exactly **62**, matching the 62 treatments with a
`reviewer_action` in the features_status DB — so no round is missing
and none is double-counted.

**Review does not reach `features_hand` until `brat_ingest`
runs.** Editing the `.ann` files in
`brat/data/skol_segments/production_v4_roundN/` leaves the
work invisible to every measurement, because precision and
recall are computed from `features_candidate` against
`features_hand`. On 2026-08-23 round 4 showed 9 of 50 in the
hand DB while 24 had in fact been reviewed; the gap closed
only when the ingest was run:

```sh
bin/brat_ingest --experiment production_v4 \
    --ann-dir brat/data/skol_segments/production_v4_round4/
```

Run it at the end of every review session, not at the end of
the round.

**`taxon_2b793602…` appears in both round 1 and round 2.** It was
exported with round 1 but not reviewed until the 07-03 session, so it
was re-exported alongside round 2. Rounds are otherwise disjoint. It
is also the flora-chapter-slice pathology recorded in
`tests/fixtures/pathologies.json` as
`§8-flora-chapter-slice-unnumbered-key`.

## The brat directories are not the same thing

`brat/data/skol_segments/production_v4_roundN/` holds the exported
`.txt` / `.ann` pairs for review. For rounds 1–3 those directories
happen to be per-round, which is what made the reconstruction
possible.

`bin/brat_export` exports every treatment the annotator has annotated
unless restricted with `--doc-id`, so a directory name matching a
round does not imply its contents are that round. The first round-4
export was made without `--doc-id` and came out cumulative — 109
treatments, all 62 from rounds 1–3 plus the 47 of round 4 that
produced annotations. It is preserved as
`erroneous_production_v4_round4/`.

To scope an export to exactly one round:

```sh
bin/brat_export --experiment production_v4 \
    --output-dir .../skol_segments/production_v4_roundN/ \
    --skip-unannotated \
    --doc-id "$(paste -sd, data/annotation_rounds/production_v4_roundN.txt)"
```

`--skip-unannotated` is needed because `--doc-id` otherwise fails if
*any* requested ID has no annotations. That check exists to catch
typos, and it cannot tell a typo from a treatment the annotator
legitimately returned no spans for — round 4 has 3 of the latter, and
they were failing the whole batch. With the flag, an ID that is a real
treatment but unannotated is dropped with a warning; an ID that is not
a treatment at all still errors out, so typos are still caught.

## Manual additions

`production_v4_roundN_manual.txt` holds treatments that must be
**included in round N regardless of what the selector picks** —
usually because review of an earlier round identified a specific
gap worth covering.

The name deliberately does *not* match
`production_v4_round<digits>.txt`, so
`select_for_annotation`'s next-round numbering ignores it and
still produces `production_v4_roundN.txt` normally.

**The selector does not read these files.**  Merge them by hand
when exporting the round:

```sh
sort -u data/annotation_rounds/production_v4_round5.txt \
        data/annotation_rounds/production_v4_round5_manual.txt \
  > /tmp/round5_all.txt

bin/brat_export --experiment production_v4 \
    --output-dir .../skol_segments/production_v4_round5/ \
    --skip-unannotated \
    --doc-id "$(paste -sd, /tmp/round5_all.txt)"
```

A manual addition will usually need annotating first — the
selector's candidates already have LLM annotations, a
hand-picked treatment may not.  Check with
`bin/llm_annotate_features` before exporting, or
`--skip-unannotated` will silently drop it.

### Pending: `production_v4_round5_manual.txt`

* **`taxon_46ff7dde…`** — *Endogonales* Jacz. & P.A.Jacz.,
  emend. Tedersoo.  Added 2026-08-23.  A correctly-attached
  **supra-generic emendation**: order-rank nomenclature with a
  matching `Type family. Endogonaceae Paol.` marker, single
  description span, merge_metric 0, **zero flags**.  Wanted as
  the poster child for a shape the reference set has no example
  of — every current poster child is a genus or species.
  `taxon_7cb84fba` would have served but is mis-attached (a
  family description under a species name, `§2-family-
  description-on-species-nomenclature`), so D14 must fire on it
  and it cannot be a poster child.

  **It has zero annotations**, which is the point of putting it
  through a round rather than adding it to the fixture directly.
  One thing to look at during review: the description ends with
  a run of GenBank/UNITE accession numbers (`EUK1100757,
  LC002628, LC431107, EUK1104693 and UDB025468`), which is
  molecular data sitting in a morphological description and may
  itself be a §12 leak.

## Adding a round

Nothing to do by hand — `select_for_annotation` writes the file. The
number continues past the highest present rather than filling gaps: a
missing file means that round's selection was never captured, not
that the number is free.
