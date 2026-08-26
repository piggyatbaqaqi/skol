# Action plan: three activities, one draw, four populations

## Context

Bootstrap annotation rounds were set up to validate Claude API feature
labels. Measured 2026-08-23, that purpose is essentially complete:
**round 3 — the only random sample — reads precision 100 %, recall
99.0 %** (96 of 96 candidates kept, one addition). The pooled
98.5 %/83.1 % describes the *selection*, not the corpus, and its low
recall is one document — `taxon_2b793602` contributes 136 of all 263
additions.

> **Remeasured 2026-08-25 (T1a).** The claim survives, sharpened.
> Round 3's **label** precision is exactly 100 % — no interval, because
> zero errors in 96 candidates collapses every bootstrap resample to
> 100 %; the honest bound is the rule of three, **≥ 96.9 %**. Its
> *span-exact* precision is 98.96 %, the difference being a **one-
> character** boundary nudge on `Asci` in `taxon_adcb2fcc`, not a
> rejected label. Corpus-wide the two split 98.80 % / 97.72 %.
>
> The broader finding: of 36 candidates that did not survive verbatim,
> **exactly one was a wrong feature**. Eighteen were relabels and 17
> were offset adjustments — the reviewer had already agreed something
> was there. Seven of the 18 relabels are entries in
> `docs/feature_label_canonicalization.json` already, so scoring
> against canonical forms lifts label precision to about **99.24 %**.
> That is the measured case for T6, and it is why the `Spores` cases
> stay queued rather than patched.
>
> The one-document caveat is unchanged but its arithmetic moved: with
> round 4 closed there are **270** additions over 109 treatments, and
> `taxon_2b793602` still carries **136 of them — exactly half**. It is
> also now attributed to round 1 alone; the old per-round table charged
> its additions to rounds 1 *and* 2, which is the whole of round 2's
> apparent recall jump from 78.7 % to 89.1 %. See
> `data/annotation_rounds/README.md`.

Meanwhile the rounds have been yielding evidence about two *other*
components — the treatment grouper (§6) and the layout/block classifier
(§12) — which is valuable but is being harvested at ~20 minutes per
treatment by accident.

This plan separates what had been one activity into three, each with the
right instrument and the right population:

| activity | population | instrument | cost |
|---|---|---|---|
| Label validation | random, P1 | brat review of 50 | operator time |
| Heaps' law vocabulary | random, P1 | `features_candidate` only, **no review** | API only |
| Pathology detection | P2a, P2b, layout labels, round-4 tail | queries + targeted review | ~free |

**The populations are an exact partition** (verified: 38 413 + 7 632 +
35 482 = 81 527):

| tag | definition | n |
|---|---|---:|
| `p1` | complexity > 0, not merge-suspect — the annotatable pool | 38 413 |
| `p2a` | merge suspects (`n_terms_above_5 ≥ 10`) | 7 632 |
| `p2b` | complexity 0 — no description/diagnosis prose | 35 482 |

`select_for_annotation` applies the complexity filter *first*, so **the
merge metric was never computed for p2b**. That is a free, testable
hypothesis (§T3d).

*Created 2026-08-23. Companion to
[production-v5-execution.md](production-v5-execution.md), which it
amends — see the final section.*

### It amends `production-v5-execution.md` — Track A's premise is wrong

That plan's **Track A** targets *"~200-250 reviewer-verified treatments
to satisfy the Heaps' Law vocabulary-coverage threshold"*, and its
"Heaps' Law dependency" section makes M5 gate on reaching that number.

**Hand review is not what the vocabulary curve needs.**
`jupyter/heaps_law_analysis.ipynb` computes its curve from
`features_candidate` **alone** (cell 4's `load_candidate_annotations`);
`features_hand` appears in exactly one separate comparison cell. So
vocabulary coverage is bought with **API volume, not operator hours** —
which is why this plan can put 1 000 treatments on the curve in an
afternoon while reviewing only 50.

Two consequences to fold back into the v5 plan when this lands:

* **Track A's 200-250 target is not a Heaps prerequisite.** It may still
  be the right target for *M5* — retraining the segment classifier on
  verified data genuinely needs verified data — but that is a training-set
  argument, not a vocabulary-coverage one, and the two were conflated.
* **Track B's ordering constraint is softer than stated.** The plan's
  "NOT doing first" list has *"More vocabulary sampling before terminology
  dict pass — Track B first; new samples without it just replay the
  drift."* But the notebook canonicalizes **post-hoc** via the JSON map,
  and plots raw and canonical curves side by side. So sampling first is
  safe: the map can grow later and the canonical curve be recomputed
  without re-annotating. That is what makes decision 3 — baseline now,
  schema fix after — coherent rather than a violation of the v5 plan.

### Operator decisions taken

1. Pathology detection works **all four** populations.
2. Heaps sample **~1 000**, priced with `--estimate` first.
3. **Baseline on the current prompt**, fix the label schema after.
4. Credentials: out of scope.

---

## Findings that shape the plan (all verified, not inferred)

**F1 — the Heaps curve is currently ordered by API latency.**
`created_at` is stamped at `bin/llm_annotate_features.py:459`, *after*
`client.messages.create` returns; the notebook sorts by it
(`heaps_law_analysis.ipynb` cell 6). With `ThreadPoolExecutor(5)` +
`as_completed`, completion order ≈ latency order ≈ output length ≈
vocabulary richness. Label-poor treatments cluster at the head, so the
curve reads concave-**up** and **β is overestimated** — the analysis
would say we need far more samples than we do. Noise at n=109; a
systematic artefact at n=1 000. Fix is notebook-only and must land
before the *analysis*, not before the run.

**F2 — the notebook loads the whole candidate DB.** After the run,
`features_candidate` holds the new random draw *plus* the 1 582
annotations from biased rounds 1/2/4. Without a round filter the curve's
first ~109 steps are non-random — the same disease as the pooled
statistic.

**F3 — annotating a merge suspect destroys the evidence.**
`skipped_merge_suspect` status docs carry
`metrics.n_terms_above_5` (a *graded* severity score, values 11/12/47
observed). `filter_already_annotated` (`:255-297`) skips only
`STATUS_SUCCESS`, so merge suspects fall through and **would be
annotated**, and the main loop replaces the whole status doc — wiping the
score. It then cascades: `fetch_prior_merge_skip_ids` no longer sees
them, so a later `--exclude-suspected-merges` run silently re-admits them
into p1. **Snapshot before anything touches p2a.**

**F4 — the annotator is structurally blind to p2b.**
`bin/llm_annotate_features.py:391-403`: an empty synth doc returns
`STATUS_SUCCESS` with **no API call**. Annotating p2b would cost $0 and
return 35 482 vacuous successes. p2b needs a field-presence cross-tab,
not an annotation round.

**Corrections to earlier assumptions.** The Anthropic SDK *does* retry —
`anthropic` 0.87.0 `DEFAULT_MAX_RETRIES = 2` with backoff honouring
`retry-after`; the bare constructor at `:643` inherits it. And the
v3_hand `--min-pass-rate 90` floor is justified by *missing attachments*
(`debian/skol.cron:148-153`), which the span-head backfill did not
address — so the floor is **not** raisable on the backfill's evidence.

---

## T0 — Preconditions (do first, all cheap)

**a. Snapshot the merge-suspect scores.** `fixes/snapshot_merge_scores.py`
→ dump `(treatment_id, n_terms_above_5)` for all 7 632 to
`data/merge_suspects_20260823.tsv`. Non-negotiable per F3, and it doubles
as the severity-ranked queue for T3a.

**b. U2 — reorder the D-item headings** in
`docs/data_quality_production_v4_model.md`. Actual order is
`D1..D8 D10 D12 D13 D14 D15 D11 D9`. Mechanical, not aesthetic: T3b and
T3d both add D-items, and an earlier scripted edit against these
out-of-order headings silently duplicated a whole section, caught only by
`grep -c`. Also fix the U2 entry itself, which omits D14/D15. Verify with
`grep -c '^### D<n> '` = 1 for all n.

**c. Record the v3_hand backfill** in `CHANGELOG.md` — 73 139 examined,
527 610 heads across 58 099 treatments, 7 648 skipped (down from 65 747
pre-fix). Evidence is perishable: it lives only in
`/var/log/skol/backfill-span-heads-manual-production_v3_hand2.log` under
rotation. **Do not** change `--min-pass-rate 90` on this basis.

**d. Stdin modes — *streaming* for `brat_ingest`, batch for
`brat_export`.** Both already honour `--doc-id`
(`brat_ingest.py:249`, `brat_export.py:355`); neither reads stdin.

The two tools want **different** semantics, and this is deliberate:

* **`brat_ingest` — streaming.** Leave one window running and paste ids
  as each treatment is finished; each is ingested **the moment its line
  arrives**, with no EOF. This is the operator's actual review loop, and
  it also fixes the "review is invisible until `brat_ingest` runs"
  problem structurally rather than by remembering to run it.
* **`brat_export` — batch.** It produces a directory to review, so it
  genuinely wants the whole set up front. `read_treatment_ids` reused
  unchanged.

```sh
# leave this running in its own window for the whole review session
bin/brat_ingest --experiment production_v4 --doc-id - \
    --ann-dir brat/data/skol_segments/production_v4_round5/
```

### The buffering trap

`read_treatment_ids` (`llm_annotate_features.py:169`) is a **batch**
function — `[line.strip() for line in stdin_stream if line.strip()]`
consumes to EOF, which is precisely the behaviour to avoid here.

Worse, the obvious streaming rewrite is also wrong: **`for line in
sys.stdin` block-buffers when stdin is a pipe**, so ids would sit
unprocessed until the buffer filled — defeating the entire point while
looking correct when tested against a TTY. The iteration must be
`iter(stream.readline, '')`.

### Shape

Add `iter_treatment_ids(stream)` beside `read_treatment_ids` in
`llm_annotate_features.py`, yielding stripped non-empty lines via
`readline`, and refactor `read_treatment_ids` to
`list(iter_treatment_ids(...))` plus its existing guards — behaviour and
its tests preserved, one source of truth. Cross-`bin/` import is
established precedent (`select_for_annotation.py:490`).

In `brat_ingest`, `--doc-id -` switches from
"discover-then-loop" to "loop over stdin, resolve `<ann-dir>/<tid>.ann`
per line". Requirements for a window that stays up all session:

* **Never exit on a bad line.** Missing `.ann`, unknown treatment id,
  transient CouchDB error → print and continue. A long session must not
  die on one typo.
* **One-line summary per id** — `taxon_8ebf437c… 18 kept, 3 added` — so
  the window doubles as a live review log.
* **Warn when the `.ann` mtime predates this session**, which catches the
  commonest mistake: pasting the id before saving in brat.
* `^D` or `^C` ends the session cleanly; `< file` still works and simply
  streams.

The `-` sentinel matters for a second reason: "no filter" is the
legitimate default in both tools, and **cron supplies a non-TTY
`/dev/null` stdin** — so making stdin-reading unconditional would break
every existing cron line.

TDD per CLAUDE.md, modelled on `brat_ingest_test.py:122`:

* *ingest, streaming*: ids fed one at a time through a pipe are each
  processed **before** the writer closes the stream (the test that pins
  the buffering fix — assert the effect after writing one line, without
  closing); a bad id mid-stream is reported and the **next** id still
  processes; `^D` after N ids exits 0.
* *export, batch*: `--doc-id -` filters to the piped ids; empty stdin
  exits 2.
* *both*: **no `--doc-id` with a non-TTY stdin still processes the whole
  `--ann-dir`** — the cron-regression test, and the important one;
  literal `--doc-id a,b` unchanged.

**e. Make the selector record its own bias.** Round 5 is the baseline
everything later compares against, so it should be the first round whose
provenance is machine-written rather than remembered.

**Bias is not a boolean, and should not be a hand-typed label.**
`select_for_annotation` narrows the population through a funnel, and
every stage is a bias source:

| stage | always on? | effect |
|---|---|---|
| complexity > 0 | yes | drops p2b (35 482) |
| `--exclude-suspected-merges` | **default on** | drops p2a (7 632) |
| `--exclude-annotated` | opt-in | biases away from prior rounds |
| `--bands` | opt-in | stratifies — the classic "biased" case |
| `_manual.txt` additions | out-of-band | hand-picked, not sampled |

So "random" always means *uniform over the survivors*, and the honest
record is **the funnel itself**, not a label. `select_for_annotation`
already knows every number — it computes them — so have it emit
`production_v4_round5.meta.json` beside the round file:

```json
{"round": 6, "experiment": "production_v4",
 "selector_argv": ["--n","1000","--exclude-annotated","--seed","20260823"],
 "seed": 20260823, "n_requested": 1000, "n_selected": 1000,
 "bands": null,
 "population_funnel": [
   {"stage": "all_treatments",        "n": 81527},
   {"stage": "complexity_gt_0",       "n": 46045},
   {"stage": "not_merge_suspect",     "n": 38413},
   {"stage": "not_already_annotated", "n": 38304}],
 "population": "p1", "selection": "uniform",
 "purpose": "heaps-baseline+validation",
 "model": "claude-opus-4-7",
 "drawn_at": "2026-08-23T…"}
```

`selection` is **derived**, not typed: `uniform` when `bands` is null,
`stratified` otherwise. `population` follows the T0-partition tags. A
reader who distrusts either can recompute both from the funnel.

### The `--bands` record must be the *realized* bands, not the flag

Reading `select_treatments` (`treatments_to_structured/select.py:98-116`)
shows why the raw flag string is not enough:

* **Band names carry no meaning.** `low`/`mid`/`high` are labels. The
  partition is by *position* in the score-sorted population, into
  `len(band_specs)` equal-size slices — `start = (i * population) //
  n_bands`. The **count** of bands sets the cut points; the names are
  decoration.
* **The cut points are recomputed per run.** They are percentiles of
  whatever survived the funnel, and `--exclude-annotated` shrinks that
  population every round. So the identical string
  `--bands low:25,mid:50,high:25` denotes **different score ranges** on
  two different runs. A record of the string alone is not reproducible
  and not comparable.

So record the realized slices, with their score ranges:

```json
"bands_raw": "low:25,mid:50,high:25",
"band_specs": [["low",25],["mid",50],["high",25]],
"band_slices": [
  {"name":"low",  "quota":25, "slice_n":12768, "score_min":0.10, "score_max":2.40},
  {"name":"mid",  "quota":50, "slice_n":12768, "score_min":2.40, "score_max":5.80},
  {"name":"high", "quota":25, "slice_n":12768, "score_min":5.80, "score_max":41.00}],
"merge_threshold": 10,
"output_order": "band-by-band"
```

For round 5, `bands_raw` is `null`, there is a single implicit `('all',
n)` band, and `output_order` is `"uniform"`.

### `output_order` is load-bearing, not decoration

`select_treatments` emits **band-by-band in declaration order**, and only
*within* a band is the order `rng.sample`'s. Two consequences:

* **T5's "first 50 lines" rule is valid only because round 5 is
  unbanded.** On a banded round the first 50 lines would come entirely
  from the first band — a maximally biased subset that looks like an
  innocent `head -50`. The sidecar's `output_order` is what lets the
  review-subset rule check itself instead of relying on someone
  remembering.
* The Heaps notebook's round-file ordering (T4) is only a valid draw
  order for `"uniform"` rounds. For banded rounds it must fall back to
  permutation-averaging alone.

Also record `merge_threshold` (default 10 — it decides p1/p2a membership)
and whether `--force` was passed (recompute the merge metric versus trust
cached skip decisions). Both change what was excluded, so both are bias
parameters.

### The seed: record it, and never let it be implicit

`--seed` defaults to `None`, and `select_for_annotation.py:558-562` then
uses a bare `random.Random()` seeded from OS entropy:

```python
rng = (random.Random(args.seed)
       if args.seed is not None
       else random.Random())
```

When the seed is omitted it is **unrecoverable** — the draw is
unreproducible even in principle, and a sidecar recording `"seed": null`
documents an irretrievable gap rather than preventing one.

**Fix: the selector generates a seed when none is given**, uses it, and
records it — so every round has a concrete integer, and `--seed` becomes
"pin this value" rather than "switch on reproducibility". Log it to
stdout as well, since that is where the operator will look first.

Two caveats that keep this honest:

* **A recorded seed does not by itself reproduce the draw.**
  `--exclude-annotated` reads the candidate DB live, so the surviving
  population changes the moment the annotator runs — the same seed over
  a different population yields a different sample. Reproducibility needs
  the seed **and** the funnel, which is why the sidecar records both.
  The round file stays the only durable record of actual membership.
* **Ties break by insertion order.**
  `sorted(scored, key=lambda pair: pair[1])` is stable, and insertion
  order is `_all_docs` order — i.e. taxon-hash order. Deterministic, but
  it means equal-scoring treatments land in a band by hash. Harmless;
  worth knowing before someone reads meaning into a band edge.

**Stamp the round onto the data too**, so DB-only queries can stratify
without needing the repo. This is the fix that would actually have
prevented today's pooled statistic — I was querying CouchDB, not reading
round files. Give `llm_annotate_features` a `--round-file PATH` that
reads the ids *and* learns the round identity from the sidecar, then
writes `round` onto every `features_candidate` and `features_status` doc
it creates. `brat_ingest` inherits `round` from the candidate doc it
diffs against.

**Historical rounds cannot be recovered this way.** Rounds 1–3 were
reconstructed from `.ann` filenames and their selector invocations are
gone; round 4's is not captured either. Those stay hand-recorded in the
README table from operator knowledge — which is exactly why round 3's
randomness had to be volunteered rather than looked up. Backfill the four
existing rounds' sidecars with `"provenance": "reconstructed"` so the gap
is explicit rather than implied.

*Zero-code fallback if this slips:* the selector already prints its
selection to stdout — `tee` it to
`data/annotation_rounds/production_v4_round5.log` at T2. Strictly worse
(unstructured, not joinable) but better than nothing.

**f. Fix `--dry-run` in `select_for_annotation` — it is a bug, not a
gap.** The flag is accepted (from `common_parser()`) and **silently
ignored**: `dry_run` appears nowhere in the file, while the run creates
the status DB (`:499-507`), writes `skipped_merge_suspect` docs
(`:528-545`) and writes the round file (`:576`). That is precisely the
silently-swallowed-flag class `docs/bin-argparse-strict.md` exists to
prevent, arriving through the shared parser instead of
`parse_known_args()`.

Required behaviour: **emit the selection to stdout and write nothing.**
No status DB creation, no skip docs, no round file, no sidecar.

Two semantics to settle while implementing:

* **The merge filter still has to work.** It may *read* prior
  `skipped_merge_suspect` ids, but must not write new ones. If the status
  DB does not exist, warn and proceed — the merge metrics are computed in
  the same scan anyway (`score_treatments_in_db` returns
  `(scored, merge_metrics)`), so filtering stays correct; only
  persistence is skipped.
* **Print the funnel and the realized bands**, not just the ids. That is
  what makes the flag useful: it is how you confirm the seed and the band
  cut points behave as intended, which is exactly the verification T0e's
  sidecar needs and cannot self-check.

This is a prerequisite for T0e's verification, and it makes the draw at
T2 rehearsable — today there is no rehearsal at all.

### Do we need explicit band cutoffs? Yes — but after `--dry-run`, not before

The percentile banding has a real defect for cross-round work: cut points
are recomputed from the surviving population every run, so the same
`--bands` string denotes different score ranges each time (T0e). Bands are
therefore **not comparable between rounds**, which is a problem the moment
anyone asks "is the high band getting better?"

Explicit cutoffs — e.g. `--band-cutoffs 2.4,5.8` alongside
`--bands low:25,mid:50,high:25` — fix that, and additionally allow
deliberately targeting a range ("30 treatments scoring above 20").

But do **not** build it speculatively, for two reasons:

* **You cannot pick good cutoffs without seeing the distribution**, and
  the fixed `--dry-run` is what shows it. The natural workflow is:
  dry-run with percentile bands → read off the realized cut points →
  pin those as explicit cutoffs for subsequent rounds. Building the flag
  first means guessing the numbers it takes.
* **Fixed quotas stop being satisfiable.** A pinned cutoff can leave a
  band holding fewer treatments than its quota, which currently raises
  (`select.py:109-113`). That needs a deliberate policy — fail loudly, or
  take what is available and record the shortfall — and that policy is
  easier to choose once you have seen real band populations.

Neither scheme is universally right: percentile bands adapt to a shifting
corpus, absolute bands stay comparable over time. The sidecar records the
realized cut points either way, so **nothing is lost by deferring** —
round 5 is unbanded and does not need this at all.

---

## T1 — Warm-up and canary

**a. Finish round 4's remaining 23.** Already exported; pure operator
time; closes the round-4 row in `data/annotation_rounds/README.md` and
reacquaints you with brat before the round-5 sitting. Its numbers stay
**out** of the pooled statistic (it is a biased round).

**b. Annotate `taxon_46ff7dde`** — queued in
`production_v4_round5_manual.txt` with zero annotations, needed before
export regardless. It earns this slot as the **canary**: one treatment,
~$0.05, exercising API-key resolution, model string, both DB writes, the
status doc and the `metrics` sub-doc. Never let call #1 of 1 000 be the
first call.

```sh
bin/llm_annotate_features --experiment production_v4 \
    --doc-id taxon_46ff7dde… --llm-model claude-opus-4-7
```

---

## T2 — Draw the sample, then price it

```sh
bin/select_for_annotation --experiment production_v4 --n 1000 \
    --exclude-annotated --seed 20260823 \
    --output data/annotation_rounds/production_v4_round5.txt
```

`--bands` omitted → single unbanded uniform draw over p1. Banding would
make the draw non-uniform and the curve uninterpretable.

Rehearse first, now that T0f has made `--dry-run` real — same command
with `--dry-run`, confirming the funnel, the recorded seed and a count of
1 000 before anything is written:

```sh
bin/select_for_annotation --experiment production_v4 --n 1000 \
    --exclude-annotated --seed 20260823 --dry-run | wc -l
```

⚠️ Without T0f this command has **no rehearsal**: `--dry-run` is accepted
and silently ignored, and the real run creates the status DB, writes skip
docs and writes the round file. Run T0a before either form.

```sh
bin/llm_annotate_features --experiment production_v4 --estimate \
    --llm-model claude-opus-4-7 \
    < data/annotation_rounds/production_v4_round5.txt
```

**GO/NO-GO on the printed cost.** Rough expectation $22–56 at
`_OPUS = Price(input=5.00, output=25.00)`. Sanity check: if the treatment
count is well below 1 000, the draw picked up p2 members and the
exclusion is broken.

> ### T2 executed 2026-08-25 — drawn and priced, awaiting GO/NO-GO
>
> **Draw.** 1 000 treatments, uniform over p1, seed 20260823 pinned.
> The `--dry-run` rehearsal wrote nothing and produced a **byte-identical**
> id list to the real run, so the seed did what it claims. Funnel:
> 81 527 → 46 045 (complexity > 0) → 45 935 (not already annotated,
> −110) → **38 303**. Zero newly-flagged merge suspects, so F3's
> evidence-destruction risk did not arise.
>
> **Round 5, not 6.** `round5_manual.txt` already claims that number and
> `default_output_path` returns `production_v4_round5.txt`. The one
> treatment in the manual file is the annotated canary.
>
> **Uniformity verified, not assumed.** Decile counts over the
> population are `[101, 102, 110, 100, 104, 92, 102, 88, 98, 103]`;
> **χ² = 3.46 on 9 df** against a 16.92 critical value. Quantiles track
> the population from p05 to p99. Zero overlap with rounds 1–4.
>
> **Cost: $38.86** — 2 220 583 input tokens, ~1 110 291 estimated
> output. Inside the $22–56 expectation, and the treatment count is
> exactly 1 000, so the p2 exclusion is not broken.
>
> **The `max_tokens` warning below is inverted for this draw.** It
> predicted that a random sample would hit long treatments the biased
> rounds never reached. The reverse holds: the biased rounds selected
> *for* suspected merges, so they carry the long ones. Round 5's largest
> synthetic doc is **7 399 chars** against rounds 1–4's **22 637**, and
> **no treatment exceeds 30 000 chars**. Truncation is not the live risk
> here; keep the check between chunks anyway, but do not expect it to
> fire.
>
> **First round whose provenance was machine-written.**
> `production_v4_round5.meta.json` carries the seed, the funnel, the
> realized slice and the exact argv, and `--round-file` (T0e) will stamp
> `round: 5` onto every candidate and status doc the run creates.

**Pin `--llm-model claude-opus-4-7` deliberately**, against the general
"use the newest model" guidance. Decision 3 is *baseline on the current
prompt*; changing the model perturbs the label distribution as much as
changing the prompt, and the existing 1 582 annotations came from 4-7.
Upgrading is a later, measured step — not an accident of a default.

---

## T3 — API run (unattended) ∥ pathology (free, parallel)

None of the four pathology populations needs an API call, so all of this
runs alongside the run and competes only for operator attention.

### The run

`split -l 250` into four chunks. Bounds blast radius, gives four
checkpoints, allows abort after chunk 1. Keep `--workers 5`. Pin
`--log-file` (default writes `llm_annotate_<epoch>.jsonl` to CWD — there
is already a stray one at repo root).

**No retry code change.** The SDK already retries twice with backoff; a
hand-rolled outer loop would nest backoff inside backoff, and with
`read=600` a hung request already burns up to 30 min across attempts.
Failures land as `status='error'` and default mode re-runs `error`/
`partial` — re-feeding the chunk file resumes exactly the failures.
*Optional one-liner*: `anthropic.Anthropic(api_key=api_key,
max_retries=5)` at `:643`.

Between chunks check **two** counts, not one:
- `status='error'` → re-run that chunk file
- `metrics.stop_reason == 'max_tokens'` → **the likeliest error class**,
  not 429. `_DEFAULT_MAX_TOKENS = 16384`; a random draw will hit long
  treatments the biased rounds never sampled. Losing the longest, most
  vocabulary-rich treatments biases the curve the *same direction as F1*.
  Raise `--max-tokens` for the affected chunk rather than accepting it.

> ### T3 run executed 2026-08-25 — complete, and cheaper than priced
>
> **1 000 of 1 000 processed**, four chunks of 250 at `--workers 5`,
> 17:21–17:46 UTC. After two resume passes: **964 success, 31 partial,
> 5 error**.
>
> | | |
> |---|---|
> | input tokens | 2 220 583 — **exactly** the estimate |
> | output tokens | 626 460 — estimate said 1 110 291 |
> | **actual cost** | **$26.76** against $38.86 priced |
>
> **The estimator over-predicts output by 1.77×.** `count_tokens` gets
> input exactly right; the output figure is the `input / 2` heuristic in
> `estimate_tokens`, and real output ran at **0.28 ×** input. Worth
> recalibrating before the next round is priced — the current heuristic
> turns a $27 run into a $39 decision.
>
> **`max_tokens` never fired — 998 `end_turn`, 2 `refusal`.** The
> warning below is now measured as well as argued: truncation was not
> the risk on a random draw.
>
> **T0e worked end to end**: all 1 000 candidate and status docs carry
> `round: 5, round_provenance: selector`, stamped by `--round-file`.
> Chunk files had to be *named* `production_v4_round5.txt` in separate
> directories, since the flag derives the round from the filename —
> a `part_00` chunk is refused outright, which is the naming rule doing
> its job.
>
> **The 5 permanent failures are 0.5 %**, in two classes: JSON-parse
> failures (§9.1, invalid `\escape`) and refusals with
> `output_tokens: 0`. They are *enriched* for OCR damage but not
> explained by it — two fire `character-substitution` at 5.7 %, about
> 3 × the p90 of 1.95 % over 200 succeeding treatments, while two others
> measure clean.
>
> ### The vocabulary answer, which is what the round was for
>
> | | rounds 1-4 | round 5 | union |
> |---|---:|---:|---:|
> | annotations | 1 582 | 7 486 | 9 068 |
> | distinct labels | 318 | 961 | **1 060** |
>
> **742 labels that did not exist before**, and 99 from the biased
> rounds that round 5 never used. The new terms are real clade
> vocabulary the biased sampling never reached — `Sporidia` (18),
> `Oospores` (13), `Synnemata` (13), `Appressoria` (13),
> `Urediniospores` (9): rusts, oomycetes and mitosporic fungi.
>
> **The singleton fraction did not move: 54 % at n=109 → 52 % at
> n=1 109.** A ten-fold increase in sample left it essentially
> unchanged, which is the signature of a vocabulary nowhere near
> saturation. T4's curve fit will put a β on it, but the headline is
> already legible — and it sharpens §12.1's warning that some of that
> tail is *slots* misfiled as features, which has to be separated before
> the number means what it appears to.

### Pathology, in parallel

**a. p2a — the 7 632 merge suspects, ranked for free.** Measured
2026-08-23: **all 7 632 carry `n_terms_above_5`, none missing**, so the
T0a snapshot is complete and the score is fully recoverable.

| | value |
|---|---|
| range | 10 → **915** |
| median | 22 |
| deciles | 11, 13, 15, 18, 22, 27, 35, 50, 79 |
| ≥ 25 | 3 355 (44 %) |

The distribution is hard right-skewed and **piles up against its own
threshold**: values 10–14 hold 2 112 treatments, 28 % of the population.

**So do not stratify by decile.** Uniform deciles would spend most of a
30-treatment budget confirming the obvious — precision at 915 is not in
doubt — while putting only ~9 treatments in the zone where the decision
actually lives. The question is *"is 10 the right threshold"*, and that is
answered by precision **near 10**:

| stratum | n | what it tests |
|---|---:|---|
| 10–14 | 15 | the decision zone — 28 % of the population sits here |
| 15–50 | 10 | the bulk; confirms the metric behaves monotonically |
| > 50 | 5 | sanity check that extreme scores really are merges |

The question is *"is this two treatments glued together"*, answered by
reading `treatment`, `nomenclature_spans` and `description` — a `fixes/`
script emitting a markdown table, **not a brat round**.

Deliverable: precision of `n_terms_above_5 ≥ 10` as a merge detector, as
a function of score, and a recommended threshold. **If precision at 10–14
is poor, p1 is larger than 38 413** and `select_for_annotation` has been
discarding good treatments — which would also mean every prior round's
sampling frame was smaller than it should have been.

Worth one look on its own: the **maximum, 915**, is a treatment with 915
distinct terms each appearing ≥ 5 times. That is almost certainly a
flora-chapter slice of the `taxon_2b793602` kind, and it belongs in the
fixture as the metric's high-end anchor.

> ### T3a executed 2026-08-26 — the threshold is wrong
>
> 30 read by the operator through `bin/treatment_dossier`. Precision of
> `n_terms_above_5 >= 10` is **51.7 %** pooled, and **28.6 %** in the
> 10–14 decision zone against **100 %** above 50.
>
> **The plan's own conditional fired**: "If precision at 10–14 is poor,
> p1 is larger than 38 413 and `select_for_annotation` has been
> discarding good treatments." It is, and it has — by roughly **3 111
> treatments**, so p1 is about **41 400** and every round since
> 2026-07-01 sampled a frame ~7.5 % too small. The draws remain valid;
> the population they describe was mis-stated.
>
> **Raise `--merge-threshold` to 15** — F1 68.2 → 73.3.
>
> Two things worth more than the threshold. The count of
> `nomenclature_spans`, which I had called "close to decisive", is the
> **worst** predictor tested (recall 6.7 %) — because the merge and the
> swallowed heading are *the same event*, so the second name never
> becomes a second span. And most false positives are genuinely damaged
> treatments, just not merged ones: the metric is a damage detector
> under a merge detector's name. Memo §6.1.

**b. Layout-label queries — the highest-leverage of the four.**
`skol_exp_production_v4_01_00_ann_combined` holds ~21 k annotated
documents. D1–D15 are each described as a label-sequence signature;
operationalise each as a query and count it corpus-wide. This converts a
memo full of anecdotes into a table of *rates*, and rates decide which
grouper bug to fix first. Zero cost. Today's raw attempt showed the
filters need iteration — running heads, real captions and bibliography
dominate the naive query — but iterating a query costs minutes against
~20 min per treatment read.

**c.** Round-4 tail — done at T1a.

### T3e — `bin/treatment_dossier`: put the diagnostic context in front of the human

**The brat export is the wrong instrument for pathology work, and this
session demonstrated it.** Every diagnosis I reached — nine genera fused
in `taxon_8d815304`, the *V. dactylidis* description appended in
`taxon_8ebf437c`, the Notes severed into a `Figure-caption` block — came
from data the reviewer *cannot see*: the layout label each span carried,
the paragraph numbers and the gaps between them, the blocks that were
dropped, `merge_metric`, the triage flags, the source document's
identity. The brat `.txt` shows `=== description ===` and prose. The
signal is all somewhere else, and the human is asked to infer it.

That asymmetry is the reason pathology findings have been costing ~20
minutes each.

**Proposal: one read-only renderer, `bin/treatment_dossier`,** taking
treatment ids (`--doc-id`, or `-` for stdin, per T0d) and emitting a
context page per treatment — HTML by default, since brat is already a
browser workflow and the dossier belongs in the adjacent tab.

Contents, all of it already reachable:

| section | source |
|---|---|
| nomenclature, all prose fields | the treatment doc |
| per span: paragraph no., char range, **layout label**, head text | `span_resolver.resolve_span` + the `.ann` |
| **the gaps** — blocks *between* consecutive spans, with their labels | the `.ann` |
| flags, `merge_metric`, `n_terms_above_5` | `triage_signals`, `merge_metric`, status doc |
| source: journal, volume, `pdf_label`, whole-volume? | the ingest doc |
| sibling treatments from the same source document | `ingest._id` |

**The gaps row is the one that matters most.** "What sat between these
two spans, and what label did it carry" is exactly the question that
exposed the fused headings and the swallowed Notes, and it is a
three-line query given `span_resolver` — which already did the hard part.

This subsumes the one-off scripts: T3a's merge-suspect table and T3d's
p2b cross-tab both become *views over the same renderer* rather than
separate throwaway code.

**Do not enrich the brat `.txt` itself.** Tempting, and wrong here:

* `brat_export` renders the synthetic doc through `render(treatment)` and
  `brat_ingest` **re-renders it** to translate offsets back. Changing the
  format shifts every offset and **invalidates existing `.ann` files** —
  including round 4's and round 5's, mid-flight.
* Context text inside the annotation surface is annotatable text.
  Reviewers would end up labelling material that is not part of the
  treatment.

A side-by-side dossier gets the whole benefit at none of that risk.
`brat` stays the annotation surface; the dossier is the evidence.

**Sequencing:** genuinely optional for T5 (label validation needs prose
only), but it is a prerequisite for T3a and T3d being efficient rather
than artisanal. Build it early in T3, while the API run is unattended.

**d. p2b — cross-tab, not sampling.** Per F4, annotation is structurally
vacuous here. Every field needed is already on the treatment doc:

| signature | verdict |
|---|---|
| `key` non-null, others null | **legitimate** — dichotomous key |
| `materials_examined` non-null, `treatment` a real name | **legitimate** — nomenclature-only entry |
| `synthetic_nomenclature: true` and `treatment` ∈ {`Nomen ignotum`, ""} | **pathology** — invented boundary |
| only `biology` / `figure_captions` populated | **pathology** — non-treatment prose mis-grouped |
| all prose fields null | **extraction failure** — trace to the layout CRF |
| `line_number == 0` while `*_spans[].start_line` ≫ 0 | **pathology** — degenerate anchor |

Then run the **F4 test**: compute `n_terms_above_5` over all 35 482. If
p2b's merge rate is materially elevated versus p1, merges are *causing*
description loss — which unifies p2a and p2b into one root cause and makes
the grouper the top v5 priority.

### Also during the run

**Rebuild and deploy the .deb for `bin/verify_spans` — on *both* hosts.**
Committed (`debian/skol.cron:159-160`, `postinst.template:263`) but
deployed nowhere: confirmed `grep -c verify_spans /etc/cron.d/skol` = 0
on puchpuchobs, and no `verify-spans-*.log` exists. It belongs here
because it unblocks T5 — `brat_export` depends on span resolution, and
the nightly guard should be live before an operator sitting. Its first
run also settles the `--min-pass-rate 90` question with data rather than
inference.

**dev (puchpuchobs)** is the low-risk half. **prod (skol /
synoptickeyof.life)** carries three preconditions that are easy to
forget and expensive to discover afterwards:

* **Revert the TLS stop-gap first.** `verify_ssl_certificates = false`
  is still live at `/data/skol/couchdb/etc/local.ini:108` and `:114`
  (verified 2026-08-23). It was a temporary measure and must not travel
  to production. This gates the prod deploy, not the dev one.
* **Re-check web routing after the install.** Apache on
  synoptickeyof.life maps `/` → CouchDB and `/skol` → Django, and daily
  unattended-upgrades restart Apache and can flip ProxyPass precedence.
  Run `bin/prod_smoke_check.py` after deploying, not before.
* **Confirm prod's `.skol_env` sizing** rather than inheriting dev
  defaults — `env_config.py`'s Spark defaults are sized for a 24-core
  box.

Sequence dev → smoke-check → prod, so a packaging error surfaces on the
machine where it costs nothing.

---

## T4 — Notebook fixes, before reading any curve

In `jupyter/heaps_law_analysis.ipynb`:

1. **Round filter (F2)** — `TREATMENT_ID_FILTER` loaded from
   `production_v4_round5.txt`, applied in `load_candidate_annotations`.
2. **Ordering (F1)** — primary curve in *round-file order* (that is the
   draw order from `select_treatments`), plus a **permutation-averaged
   curve over ~200 permutations with a band**. Permutation averaging is
   the standard estimator for β and removes ordering artefacts by
   construction. Keep the temporal curve as a clearly-labelled secondary
   panel — it answers a different and still-useful question (did the
   model's vocabulary drift *during* the run).
3. **Duplicate-DOI covariate** — Trello #405 says 36.7 % of ingest docs
   share a DOI, so the draw contains near-duplicate treatments and
   re-reading one article inflates apparent saturation. `ingest.doi` is on
   the treatment doc. Report the curve with and without duplicates
   collapsed: that measures how much #405 would move the number, at zero
   risk to the draw.
4. Assert no two treatments share a `created_at` (the cumulative-curve
   loop keys on the timestamp string and would double-count).

---

> ### T4 executed 2026-08-26 — F1 confirmed, and backwards
>
> The curve logic moved out of the notebook into
> `treatments_to_structured/heaps.py` with 21 tests. A notebook cannot
> be tested and this was exactly the logic that was wrong.
>
> **F1 is real and large.** The temporal curve lies **outside the 95 %
> permutation band at 873 of 1 000 points**. But its stated mechanism
> and direction are both wrong: F1 predicted *"label-poor treatments
> cluster at the head, so the curve reads concave-up and β is
> overestimated."* Measured, label-**rich** treatments cluster at the
> head, the temporal curve sits **above** the band throughout, and β
> temporal (0.624) is **lower** than the permutation estimate (0.641).
> The fix was needed; the reasoning behind it was not right.
>
> **The draw-order curve is inside the band at every checkpoint**,
> which is what a uniform draw should look like and is independent
> evidence that round 5 was drawn correctly.
>
> **β ≈ 0.64**, with 957 raw labels from 1 000 treatments and **58 %
> singletons**. Canonicalization collapses only 2.1 % (957 → 937), so
> the drift map as it stands barely dents the tail — which is §12.1's
> point, that much of that tail is *slots misfiled as features* rather
> than vocabulary.
>
> **Two bugs found by running it, not by reasoning.** The old
> `cumulative_distinct_curve` keyed on `created_at` and added labels
> for *every* treatment sharing that string, so a collision both
> mis-attributed and double-counted; measured, there are zero
> collisions, so it was latent, and the new curve never keys on time.
> And my own first `permutation_band` derived its population from the
> treatments that produced labels, silently dropping the 123 that
> produced none — shortening the x-axis and steepening the curve, the
> exact flattery this exercise removes.
>
> **The manual canary leaked in.** `taxon_46ff7dde` is stamped round 5
> with `provenance: manual` but is not in the draw, so it inflated the
> vocabulary to 961. Filtering to the round file's 1 000 gives 957.
> Population mixing, caught by the provenance field that T0e added.

## T5 — Review 50, and report it correctly

Export → review → ingest the **first 50 lines** of the round file, using
the T0d stdin mode:

```sh
head -50 data/annotation_rounds/production_v4_round5.txt > /tmp/r5_review.txt
bin/brat_export --experiment production_v4 --doc-id - --skip-unannotated \
    --output-dir brat/data/skol_segments/production_v4_round5/ < /tmp/r5_review.txt
# review in brat, running brat_ingest at the END OF EACH SITTING
bin/brat_ingest --experiment production_v4 --doc-id - \
    --ann-dir brat/data/skol_segments/production_v4_round5/ < /tmp/r5_review.txt
```

**Commit to "the first 50 lines" *before* the run finishes.** The
tempting move once candidates land — pick the treatments that introduced
new labels — is exactly the selection mechanism that produced round 1's
36.3 % recall.

**`taxon_46ff7dde` is exported alongside but stays out of the
statistics** (operator, 2026-08-24). It is a hand-picked poster child
queued in `production_v4_round5_manual.txt`, so merging it into the draw
would make the round file no longer a uniform sample. Merging happens at
export and the review subset is the first 50 lines of the *selector's*
file, so it would very likely never reach the reviewed set anyway —
but "very likely" is how sampling frames get quietly broken, which is
what this plan exists to stop. Export it, review it, exclude it from
precision/recall.

### 50 is right for precision and hopeless for recall — measured, not assumed

Treatment-level bootstrap over the 85 currently-reviewed treatments
(2026-08-23), against the naive annotation-level Wilson interval at the
same annotation count:

| metric | pooled | naive ± | **clustered ±** | design effect |
|---|---:|---:|---:|---:|
| precision | 98.48 % | 0.90 pp | **1.14 pp** | 1.6× |
| recall | 83.13 % | 2.42 pp | **15.11 pp** | **38.8×** |

Sizing for precision (clustered half-width): n=25 → ±1.61; **n=50 →
±1.12**; n=100 → ±0.80; n=150 → ±0.64. **50 treatments is the right
call** — it lands an interval of roughly [97.4, 99.6], ample to confirm
ceiling behaviour, and doubling to 100 buys only 0.3 pp.

**But recall at n=50 is ±15 pp, which is not a measurement.** 83 % ± 15
spans 68–98 % and supports no conclusion at all. The cause is the shape
of the data, not the design: **53 of 85 treatments add nothing**, one adds
136, top-1 = 52 % and top-5 = 70 % of all additions. A near-zero-inflated
heavy-tailed count cannot be estimated from 50 draws.

**So report the two metrics differently:**

* **Precision** — pooled, with the treatment-level bootstrap CI. This is
  the number that retires label validation.
* **Recall** — **do not report a pooled point estimate with a CI.**
  Report the distribution: median additions per treatment (currently 0),
  the fraction of treatments needing ≥ 1 addition, the top-1/top-5
  concentration, and the raw count. Those are robust; the ratio is not.

### Correction this forces to the round-comparison table

The per-round recall figures in `data/annotation_rounds/README.md`
(R1 36.3 %, R2 78.7 %, R3 99.0 %, R4 94.0 %) carry **annotation-level
Wilson intervals, which are ~6× too narrow.** At a 38.8× design effect
those intervals overlap comfortably, so **the round-to-round recall
differences are largely noise**, not the clean bias gradient they appear
to be. The README needs that caveat added — the *precision* column
survives, the *recall* column does not support the reading it invites.

The underlying claim is unaffected: R1's low recall is one document, and
that is visible directly in the counts without any interval.

### And a correction to "the nominal question is already answered"

Round 3 is **9 treatments**. Its 100 % precision is suggestive, not
conclusive — a treatment-level interval on 9 units is wide. The honest
statement is that round 3 gives no reason to doubt the labels, and
**round 5's 50 treatments is what actually settles it**. That strengthens
the case for T5 rather than weakening the plan, but the earlier phrasing
overstated what 9 treatments can carry.

**Corrected 2026-08-25: round 3 selected 10, not 9.** Its round file
was reconstructed from `.ann` filenames, and `taxon_bc52ee90…`
produced zero annotations — so it left no `.ann` file and vanished from
the reconstruction. Three more went the same way from round 2, which
selected 51 rather than 48; see
`data/annotation_rounds/README.md`. The precision and recall figures
are unchanged, since all four are unreviewed and contribute neither
kept nor added annotations. **The bootstrap denominator is not**: the
treatment-level resample is over **10** units, one of which
contributes nothing, which makes an already-wide interval slightly
wider. Nothing here changes the conclusion that round 5 is what
settles the question.

---

## T6 — Label schema fix (after the baseline)

Target the union of what is now measured: 12 case/plural clusters
(`ConidiogenousCells`, `Chemical Reaction`/`reaction`/`reactions`), the
base+context family (`Asci` / `Asci protologue` / `Asci in culture MEA`;
same for Ascospores, Pseudothecium, Colony, Culture), and the six
round-4 clade-specific-spore cases. **These are one problem** — the label
string carries a qualifier the schema should carry structurally — which is
why the six `Spores` docs should *not* be patched now (§ Backlog).

**And the same problem produces `Squamules`** (`taxon_fa7f4de6`,
2026-08-25): a *property* — fertility — with no structural slot to
occupy, so it is expressed as a new organ-shaped label. Memo §12.1
records the case and the testable consequence: **partition the 322
labels into organ-names and property-names before reading anything into
the 54 % singleton rate.** Schema induction has to happen first, since
a slot masquerading as a feature inflates the vocabulary curve exactly
where it is being measured.

Two traps:

- `annotation_doc_id` builds `_id = <tid>:<label>:<start>` with **no
  deletion pass**, so renaming `Spores` → `Ascospores` at the same offset
  leaves the old doc alive. A re-run into the same DB yields the *union*
  of two prompts' vocabularies — worse than useless for a before/after
  comparison. **The post-fix run must target a new candidate DB.**
- It must re-use the **same round file with `--force`**, not a fresh
  selection: `--exclude-annotated` reads the candidate DB live and would
  exclude precisely the 1 000 treatments being re-measured.

Also needed: **a canonicalization applier**. Nothing applies
`docs/feature_label_canonicalization.json` to any database today — only
the notebook and a guard test read it. That is Track B's missing tool.

---

## Backlog: what is interspersed, and what is deliberately left

**Included, each tied to what it unblocks:** U2 (T0b — unblocks writing
D-items); v3_hand record (T0c — perishable evidence); merge snapshot
(T0a — F3 precondition); stdin mode (T0d — unblocks T5);
`taxon_46ff7dde` (T1b — canary); verify_spans deploy (T3 — unblocks T5
and settles the floor).

**Left alone, with reasons:**

- **U1 (Django `_collect_ann_db_candidates`).** Nothing here touches
  Django search. The one thing that genuinely diverged — attachment-name
  ordering — is already reconciled; the remaining difference (Django also
  probes `ingest.db_name` and the experiment's `databases.annotations`) is
  *deliberate*, so "deduplicate" is really "design a union probe order", a
  design task wearing a cleanup's clothes. Wait for a third caller.
- **Trello #404 / #405.** #405 changes the *composition of the corpus*;
  deduplicating 36.7 % of ingest docs alters the sampling frame and
  retroactively invalidates the round-5 draw. Deferring is not merely
  cheaper — it is required for the baseline to mean anything. The T4
  covariate gets the prioritisation input at zero risk.
- **The six round-4 `Spores` treatments**, and `Squamules` on
  `taxon_fa7f4de6`. They are evidence about the *prompt*; editing them
  now corrupts the baseline. Queue for T6.
- **Credentials.** Operator decision: out of scope.

**Not adopted from review:** renaming `production_v4_round5_manual.txt`.
The `_manual` suffix deliberately does not match the numbering regex
(documented in `data/annotation_rounds/README.md`) so the selector keeps
numbering normally. Renaming would break a working design. The
population-collision concern is real and is handled by the sidecar below.

---

## Round-file convention (prevents the next pooled statistic)

**`_manual` files share their round's number, and that is correct.**
`production_v4_round5_manual.txt` holds treatments to be merged *into*
round 5; it is not a competing round 5. The `_manual` suffix
deliberately fails `default_output_path`'s `round(\d+)\.txt` regex so
the selector still numbers normally — the selector produces
`production_v4_round5.txt` and the two are merged at export.

Round numbers stay a **pure global sequence** across populations —
encoding the population in the filename is what breaks
`default_output_path`'s regex. Provenance lives in the sidecar the
selector now emits (T0e), joined to the data by the `round` field
stamped on each candidate/status doc.

The absent `selection` field is what produced the meaningless pooled
98.5 %/83.1 %, and the absent `round` field on the docs is why the error
was invisible from the database side. T0e closes both.

Add `population` and `purpose` columns to the
`data/annotation_rounds/README.md` provenance table — free, and it stays
the human-facing summary. **Never mix populations in one round file.**

---

## Verification

- **T0d**: `pytest bin/brat_ingest_test.py bin/brat_export_test.py` green,
  including the no-`--doc-id`-with-non-TTY-stdin cron regression;
  `pytest bin/argparse_strict_test.py` still green.
- **T0b**: `grep -c '^### D<n> '` = 1 for n in 1..15; heading order
  numeric; word-count diff ≈ 0 (pure movement).
- **T1b**: candidate + status docs exist for `taxon_46ff7dde` with a
  populated `metrics` sub-doc carrying real token counts.
- **T2**: `--estimate` prints a treatment count ≈ 1 000.
- **T3**: after each chunk, zero `status='error'` remaining after one
  resume pass; `stop_reason == 'max_tokens'` count reported, not ignored.
- **T3d**: the six-way cross-tab sums to 35 482.
- **T4**: permutation band computed over ≥ 100 permutations; curve
  restricted to round 5 only (assert no `created_at` before the run's
  start).
- **T5**: `pytest tests/pathologies_test.py` green; precision/recall
  reported with treatment-level bootstrap CI and top-5-dropped variants.
- **Throughout**: `bin/verify_spans --experiment production_v4` at 100 %.

## The empty-description question — largely answered, and it splits in two

Operator theory (2026-08-23): *"we're generating treatment objects for
pieces of articles that aren't actually taxonomic articles… taxonomic
citations appearing in non-taxonomic articles."* **Tested and confirmed
for a third of the cases — with a second, larger mechanism behind it.**

Grouping the 39 431 empty-description treatments by source document:

| source documents (17 645 producing treatments) | docs | treatments |
|---|---:|---:|
| **every** treatment empty | 8 797 (49.9 %) | 14 143 |
| mixed | 4 450 (25.2 %) | 25 288 empty of 57 817 |
| all have descriptions | 4 398 (24.9 %) | — |

So **35.9 % of empty treatments come from documents that produced no
description at all**, and 64.1 % from documents that produced some.

### The all-empty half: the theory is right

Sampling 400 of each and matching titles against taxonomic keywords
(`sp. nov.`, `new species`, `taxonom*`, `phylogen*`, `revision`, …):

| | all-empty docs | all-full docs |
|---|---:|---:|
| taxonomic title keyword | **7.5 %** | **31.5 %** |
| dominant journals | *Journal of Fungi* (62 %), IMA Fungus | MycoKeys, Mycotaxon |

The journal signature alone tells the story — all-empty is dominated by
*Journal of Fungi*, a broad applied-mycology venue; all-full by MycoKeys
and Mycotaxon, both dedicated taxonomy journals. And the non-matching
titles are unambiguous:

> *"Soil Fungal Community Characteristics at Timberlines…"*,
> *"Calcineurin Inhibitors Synergize with Manogepix to Kill…"*,
> *"Improvement of laccase production by Pleurotus ostreatus…"*,
> *"OMICS and Other Advanced Technologies in Mycological Applications"*

These are ecology, clinical and biotech papers. They contain binomials —
`Pleurotus ostreatus P5`, `Aspergillus fumigatus` — and the grouper built
treatment objects around them. **Exactly the mechanism proposed.**
Corroborating: **54.7 %** of empty treatments carry
`synthetic_nomenclature: true` — the name was invented, not found.

Note the scale implication: *Journal of Fungi* is the corpus's **largest**
source at 8 817 ingest documents, and it is largely not taxonomic
material.

### The mixed half is a different bug

The remaining 64.1 % come from documents that *do* yield descriptions —
real taxonomic articles emitting empty treatments alongside good ones.
That is not "wrong article type"; it is spurious boundaries or lost
descriptions **within** a taxonomic paper, and it is the same suspect as
§6/§12. T3d's cross-tab and the F4 merge-metric test target precisely
this half.

**Implication for T3d:** split the cross-tab by source-document class
(all-empty vs mixed) before interpreting it. Pooling them would mix an
article-selection problem with an extraction problem — the same error as
the pooled precision/recall statistic.

**Cheap follow-on, not scheduled here:** a document-level "is this a
taxonomic article" gate would remove ~14 000 spurious treatments before
extraction. The journal + title-keyword signal above is already a usable
first cut, and it belongs in the v5 discussion rather than this plan.

---

## Amendment to `docs/plans/production-v5-execution.md`

**Applied 2026-08-23** — recorded here as the rationale for the edits
made to that document:

1. **Track A** — replace *"~200-250 reviewer-verified treatments to
   satisfy the Heaps' Law vocabulary-coverage threshold"* with the
   correction above: the curve reads `features_candidate` only, so
   vocabulary coverage is bought with API volume. Keep 200-250 as an
   **M5 training-set** target, labelled as such.
2. **"Heaps' Law dependency"** — M5 still gates on verified volume; the
   *vocabulary curve* does not. Separate the two claims.
3. **"What we're explicitly NOT doing first"** — soften *"More
   vocabulary sampling before terminology dict pass"*: canonicalization
   is post-hoc via the JSON map and the notebook plots raw and canonical
   together, so sampling first is safe and is what decision 3 does.
4. **Add a cross-reference** to `annotation-activity-split.md`, and add a
   change-log entry dated 2026-08-23 recording that Track A's premise was
   corrected and why.
5. **Note the non-taxonomic-article finding** as an input to M3/M4 — a
   document-level taxonomic-article gate is a candidate v5 pipeline
   change, independent of the segment classifier.
