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

Remeasured **2026-08-25**, after round 4 closed at 47 of 50
and round attribution was stamped onto the status docs
(T0e), so these figures come from the database rather than
from filename bookkeeping.  Counts are read from each status
doc's `reviewer_action`, which `bin/brat_ingest` writes at
ingest time; `kept_count + deleted_count == annotation_count`
holds for all 109 reviewed treatments, and `kept_count`
agrees with the hand DB exactly.

**There are two precisions and they answer different
questions.**  A reviewer who nudges a span boundary by one
character has not rejected the label:

* **Label precision** — did the reviewer accept the feature
  name?  This is what the bootstrap rounds exist to
  validate.
* **Span-exact precision** — did they accept the name *and*
  the exact offsets?  Strictly lower, and a different
  measurement.

| round | selection | n | cand | label precision | span-exact precision |
|---|---|---:|---:|---|---|
| 1 | biased | 6 | 82 | 98.78 % [93.9, 100] | 98.78 % [93.9, 100] |
| 2 | biased | 47 | 881 | 98.52 % [97.1, 99.4] | 96.94 % [93.4, 99.0] |
| **3** | **random** | 9 | 96 | **100 %** — see below | 98.96 % [95.8, 100] |
| 4 | biased | 47 | 523 | 99.04 % [98.1, 99.8] | 98.66 % [97.7, 99.5] |
| — | pooled | 109 | 1582 | 98.80 % [98.0, 99.4] | 97.72 % [96.0, 98.9] |

**Treatment-level bootstrap intervals**, 20 000 resamples,
seed 20260825 — *not* the annotation-level Wilson intervals
the previous version of this table carried.  **Use the
round-3 row for any claim about label quality on the
annotatable population; use the pooled row for nothing.**

**Round 3's label precision has no interval, and reporting
one would be false precision.**  It made zero label errors
in 96 candidates, so every bootstrap resample returns 100 %
and the interval collapses to [100, 100] — an artifact of
resampling zero events, not a finding.  The honest form is
the **rule of three**: zero errors in 96 gives an upper
bound of 3/96 on the error rate, i.e. **label precision
≥ 96.9 %** at 95 % confidence.

### What changed from the 2026-08-23 figures, and why

Three of the differences are corrections, not drift.

* **`taxon_2b793602…` was counted twice.**  It sits in both
  round 1's and round 2's files (see below), and the old
  table charged its **136 additions to both rows** — round 1
  at 142 and round 2 at 240.  It is now attributed once, to
  round 1, under the lowest-round-wins rule that
  `fixes/backfill_round_stamps.py` applies.  That alone
  moves round 2's additions from 240 to 104 and its recall
  ratio from 78.7 % to 89.1 %.  The old *pooled* row was
  right; only the per-round rows double-counted.
* **Round 3's "100 %, 96 of 96 kept" was right about
  labels** and silent about spans.  One candidate,
  `Asci` on `taxon_adcb2fcc`, had its end offset moved from
  602 to **603** — a single character.  `brat_ingest`
  records that as one deletion plus one addition, which is
  why span-exact precision reads 98.96 %.  No label was
  rejected.
* **Round 4 went from 24 reviewed to 47**, closing the
  round.  Three of its 50 produced zero annotations and so
  can never be reviewed.

### The 36 "rejections" are almost all refinements

Classifying every candidate that did not survive verbatim:

| what happened | n |
|---|---:|
| relabelled at the same offset | 18 |
| boundary moved, same label and start | 15 |
| same label, span moved | 2 |
| **label genuinely rejected** | **1** |

**Exactly one candidate in 1 582 was a wrong feature**
(`Conidia` on `taxon_ba964a8b`).  Everything else the
reviewer corrected was a *name* or an *offset*, having
already agreed something was there.

The 18 relabels are all naming, not misidentification:

| from | to | n | |
|---|---|---:|---|
| `Colonies` | `Colony` | 6 | in the canonicalization map |
| `Pileus` | `Basidiocarp` | 3 | part-vs-whole |
| `Spores` | `Basidiospores` | 3 | clade-specific spore term |
| `Spores` | `Ascospores` | 1 | same class |
| `Stalk` | `Stipe` | 1 | synonym |
| `Anamorph` | `Asexual morph` | 1 | in the canonicalization map |
| `CultureCharacteristics` | `Culture characteristics` | 1 | **map says `Cultural characteristics`** |
| `Conidial germination` | `Culture characteristics` | 1 | |
| `General veil` | `Universal veil` | 1 | |

**7 of the 18 are already entries in
`docs/feature_label_canonicalization.json`.**  Scoring
against canonical forms rather than literal strings would
take pooled label precision from 98.80 % to about
**99.24 %** — which is the measured case for T6's label-schema
work, and the reason those `Spores` treatments are queued
rather than patched.

**One conflict to settle in T6**: the reviewer chose
`Culture characteristics` where the map says
`Cultural characteristics`.  The map and the reviewer
disagree, and nothing currently notices.

All 18 relabels fall in rounds 2 and 4 (13 and 5).  **Rounds
1 and 3 contain none.**

> ### ⚠️ The recall column does not support the reading it invites
>
> Those Wilson intervals are computed **per annotation**, and
> annotations cluster hard within treatments.  Bootstrapping by
> *treatment* over the 85 reviewed treatments (2026-08-23) gives
> the honest widths:
>
> | metric | pooled | per-annotation ± | **per-treatment ±** | design effect |
> |---|---:|---:|---:|---:|
> | precision | 98.48 % | 0.90 pp | **1.14 pp** | 1.6× |
> | recall | 83.13 % | 2.42 pp | **15.11 pp** | **38.8×** |
>
> So the **precision** column is sound, and the **recall**
> column is not: at a 38.8× design effect the per-round recall
> intervals overlap comfortably and **the apparent
> 36.3 → 78.7 → 99.0 → 94.0 gradient is largely noise**, not the
> clean bias signal it resembles.
>
> The cause is the shape of the data, not the arithmetic:
> **53 of 85 treatments add nothing**, one adds 136, and the top
> five treatments contribute 69.6 % of all additions.  A
> near-zero-inflated heavy-tailed count is not estimable from
> samples this size.
>
> **Report recall as a distribution** — median additions per
> treatment (0), fraction of treatments needing ≥ 1, top-k
> concentration — never as a pooled ratio with an interval.
>
> The one claim that survives unaided: round 1's low recall is a
> single document, visible directly in the counts.
>
> Round 3 is also only **9 treatments**, so its 100 % precision
> is suggestive rather than conclusive.  A 50-treatment random
> review gives ±1.1 pp and is what actually settles the question
> — see
> [docs/plans/annotation-activity-split.md](../../docs/plans/annotation-activity-split.md).

#### Recall, reported the way it should be (2026-08-25)

Same 20 000-resample treatment-level bootstrap, over the 109
reviewed treatments with `taxon_2b793602…` attributed once:

| round | median adds | need ≥ 1 | top-1 | top-5 | total adds | ratio, with its real interval |
|---|---:|---:|---:|---:|---:|---|
| 1 | 0 | 2/6 (33 %) | 96 % | 100 % | 142 | 36.3 % **[17.3, 100.0]** — 83 pp wide |
| 2 | 0 | 20/47 (43 %) | 18 % | 53 % | 104 | 89.1 % [82.7, 93.6] |
| **3** | **0** | **1/9 (11 %)** | 100 % | 100 % | **1** | 99.0 % [95.8, 100] |
| 4 | 0 | 12/47 (26 %) | 22 % | 70 % | 23 | 95.7 % [92.8, 98.2] |
| — | pooled | 35/109 (32 %) | 50 % | 68 % | 270 | 85.1 % [71.7, 94.2] |

**The distribution columns are the report; the ratio column
is shown only to demonstrate that it should not be.**  Round
1's interval spans 17 % to 100 % — it is not a measurement of
anything, and its point estimate of 36.3 % has been quoted
as though it were.

The shape is unchanged by the corrections: **the median
treatment needs zero additions in every round**, roughly a
third need any at all, and one treatment still carries half
the corpus total.  Round 2's ratio moved from 78.7 % to
89.1 % purely by removing the double-count, which is itself a
demonstration that the ratio is fragile in a way the
distribution is not.

### "Random" means random over 47 % of the corpus

`select_for_annotation` applies two exclusions **before** any
sampling, so even an unbanded run is not random over the
corpus (measured 2026-08-23):

| | treatments | share |
|---|---:|---:|
| corpus | 81 527 | — |
| complexity score 0 — no description/diagnosis prose | −35 482 | 43.5 % |
| `skipped_merge_suspect` in features_status | −7 632 | 9.4 % |
| **sampling population** | **38 413** | **47.1 %** |

`--exclude-suspected-merges` is **on by default**
(`--no-exclude-suspected-merges` to bypass).  So round 3's
100 % / 99 % describes *annotatable, non-merge-suspect*
treatments.

That is the right population for the operational question —
it is exactly what production would annotate — but it is not
the corpus, and two consequences follow:

* **The label figures do not cover merge suspects.**  Nothing
  says what labelling looks like on the 7 632, and the
  §6/§12 cases reviewed this session suggest it is worse.
* **Annotation rounds cannot discover merge pathologies by
  default**, because merge suspects are excluded before
  sampling.  Pathology work has to bypass the filter
  deliberately, which is part of why the biased rounds exist.

Treat these as two populations with different purposes:
**P1**, the 38 413 annotatable treatments, for label
validation and vocabulary estimation; **P2**, the remaining
43 114, for pathology work.

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

Their union is **66** — 6 + 51 + 10 with `taxon_2b793602…` shared
between rounds 1 and 2 — of which **62** carry a `reviewer_action` in
the features_status DB. The four-way difference is the zero-annotation
set recovered below; those were selected but produced nothing to
review.

*Before 2026-08-25 this read "exactly 62, matching the 62 with a
`reviewer_action`". The match was the bug, not the check: the files
were built from `.ann` filenames, so they could only ever contain
treatments that produced annotations, and agreeing with the reviewed
count was guaranteed rather than reassuring.*

### The zero-annotation gap is not hypothetical — four treatments fell through

Identified 2026-08-25 from an operator triage spreadsheet
(`triage_production_v4`, tab "Round 2") that listed **three treatments
absent from every round file**. Chasing them found a fourth, and all
four are annotator runs whose round file lost them by exactly the
mechanism warned about above: **zero annotations produced, therefore no
`.ann` file, therefore invisible to a filename-based reconstruction.**

Every attempted annotation in the corpus now reconciles:

| run date | attempted | accounted for as |
|---|---:|---|
| 2026-06-29 | 56 | round 1 (6) + round 2 (48) − 1 shared + **3 lost** |
| 2026-07-05 | 10 | round 3 (9) + **1 lost** |
| 2026-08-14 | 50 | round 4 |
| 2026-08-24 | 1 | the `taxon_46ff7dde` canary |

The four, placed by `last_attempt_at` against each round's run window:

| treatment | run at | belongs to | status | why it yielded nothing |
|---|---|---|---|---|
| `bb43b1ae` | 06-29 22:32:11 | **round 2** | success | `description` empty, 234-char diagnosis |
| `fb7bd18d` | 06-29 22:32:20 | **round 2** | success | `Nomen ignotum`, 1 225-char description |
| `cda95f9f` | 06-29 22:32:40 | **round 2** | **error** | 2 921-char diagnosis, no description |
| `bc52ee90` | 07-05 21:01:04 | **round 3** | success | 595-char description |

Round 2's window is 19:37:47–22:35:47 and round 3's is 21:01:06–21:01:33
on their respective days, so the assignment is unambiguous — and the
per-day totals leave no other run that could explain them.

**Consequences, none of which move precision or recall.** All four are
unreviewed, so they contribute no kept and no added annotations and the
ratios are unchanged. What changes is **n**:

* **Round 2 selected 51, not 48.**
* **Round 3 selected 10, not 9** — and round 3 is the only random
  sample, so this is the one that matters. Its tenth member produced
  zero candidates, so the "100 % precision, 99 % recall" figures stand
  exactly as measured; but any treatment-level bootstrap should resample
  **10** units, one of which contributes nothing, rather than 9.

**The round files were corrected on 2026-08-25** and
`fixes/backfill_round_stamps.py` re-run. Every attempted annotation in
the corpus now carries a round, with none left over:

| round | status docs stamped | provenance |
|---|---:|---|
| 1 | 6 | reconstructed |
| 2 | 50 | reconstructed |
| 3 | 10 | reconstructed |
| 4 | 50 | reconstructed |
| 5 | 1 | manual |

Round 2 stamps **50** against a 51-line file because
`taxon_2b793602…` is attributed to round 1 — the lowest round wins in
the backfill, since `filter_already_annotated` skips
`status='success'` and every attempted status doc reads
`attempt_count: 1`, so round 2's run never re-annotated it.

**Note what the correction does not do.** The added ids are still
absent from `brat/data/skol_segments/production_v4_round{2,3}/`,
because there was never an `.ann` file to export — that is the whole
reason they were lost. A future reconstruction from filenames would
lose them again; the round file is now the only place they exist.

**The `n` column in the round-comparison table above is a *reviewed*
count, not a selected one**, so it is unaffected by this correction —
but it is stale for a different reason: it records round 4 at 24, and
round 4 closed at **47 of 50** on 2026-08-25. Recomputing that table
with treatment-level bootstrap intervals is T1a in
`docs/plans/annotation-activity-split.md`, not part of this fix.

**Why the spreadsheet could not answer the question it was fetched
for.** It was retrieved to test whether it preserved round 2's
*generation* order, which would have made round 2's `--bands` structure
recoverable the way round 4's was. It does not: it is a triage worklist
in two blocks, each sorted by `merge_metric` descending, and the
band-monotonic test finds nothing in it. Round 2's bands stay
unrecoverable. Its `merge_metric` column is also unchanged from current
values in all 56 rows, so it carries no historical snapshot either.

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
