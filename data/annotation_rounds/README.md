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

| round | IDs | source | reviewed |
|---|---:|---|---|
| 1 | 6 | reconstructed 2026-08-20 | 5 on 2026-07-01, 1 on 2026-07-03 |
| 2 | 48 | reconstructed 2026-08-20 | 48 on 2026-07-03 |
| 3 | 9 | reconstructed 2026-08-20 | 9 on 2026-07-07 |
| 4 | 50 | captured from the selector, 2026-08-14 | not yet reviewed |

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

**`production_v4_round4/` is cumulative** — 109 treatments, being all
62 from rounds 1–3 plus the 47 of round 4 that produced annotations.
`bin/brat_export` exports every treatment the annotator has annotated
unless restricted with `--doc-id`, so a directory name matching a
round does not imply its contents are that round. Pass
`--doc-id "$(paste -sd, production_v4_roundN.txt)"` to scope an export
to one round.

## Adding a round

Nothing to do by hand — `select_for_annotation` writes the file. The
number continues past the highest present rather than filling gaps: a
missing file means that round's selection was never captured, not
that the number is free.
