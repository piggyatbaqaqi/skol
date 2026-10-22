# Modal backend for `treatments_to_json`

Status: design, 2026-10-22.  Not yet implemented.

## Why

`production_v4_1` has about 41,700 translatable treatments left to turn
into structured JSON.  On puchpuchobs the constrained-decoding path
(Hugging Face transformers, 4-bit Mistral-7B, Outlines, one record at a
time) runs at about one record per minute: roughly 29 days of
uninterrupted GPU time.  The index has to be built by 2026-11-17 (the
mushroom-club presentation), and the local GPU is also needed for other
development.  So the inference moves to rented GPUs on
[Modal](https://modal.com).

## Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **Everything except inference stays local.**  The local process reads records, applies the untranslatable-text and token-threshold filters, sends `{_id, text}` to Modal, and saves results to CouchDB as they return. | Modal containers cannot reach the local CouchDB, and exposing it is not worth the risk.  Saving locally keeps `--skip-existing`, deterministic doc ids and resumability unchanged. |
| 2 | **vLLM on Modal**, with JSON-schema-guided decoding (xgrammar backend), bf16 weights, continuous batching. | Throughput is the cost lever: GPU time is billed by the second.  Completion matters more than reproducibility (operator decision). |
| 3 | **Mixed outputs are acceptable.**  Records made locally stay; every record carries a provenance field. | Operator decision; provenance keeps the option of regenerating either set later. |
| 4 | **Same prompt and schema as the local path.**  The prompt comes from `ConstrainedDecoder._build_prompt` (system prompt `TreatmentsJSONTranslator.DEFAULT_PROMPT`); the schema from `TaxonomySchema.to_json_schema_inlined()`. | One source of truth, so backend differences are the engine's, not the prompt's. |
| 5 | **Same output contract.**  `max_tokens` = `--max-new-tokens` (pipeline: 3072); output cut off at the cap, or failing `TaxonomySchema.validate`, is saved as `{}` with `json_valid: false`. | Matches the decision for the local path: a retry with the same model would cut off again. |
| 6 | **Ephemeral Modal app.**  `treatments_to_json --backend modal` runs the app for the duration of the command (`app.run()`); no deployed service. | Containers stop when the command stops, so nothing bills after a crash or Ctrl-C beyond Modal's idle timeout. |
| 7 | **Model weights on a Modal Volume**, pinned to a Hugging Face revision; a CPU-only preparation step downloads them if absent. | No GPU time spent downloading; runs and preparation are guaranteed to use the same weights. |
| 8 | **`--dry-run` estimates cost and spends nothing**; throughput comes from a separate, explicit calibration run. | Token counts are exact locally; records/second per GPU type has to be measured, not guessed. |
| 9 | **`--max-cost` is required for any spending run**, and the running cost is tracked so submission stops at the limit. | The Modal workspace limit ($42.50 today) is the backstop, not the plan.  A broken run on rented GPUs costs real money (cf. the silent-`{}` run of 2026-10-16). |
| 10 | **Pipeline default stays local.**  Modal is opted into with a flag. | Development and tests keep working without Modal credentials. |

## Architecture

```
local: bin/treatments_to_json --backend modal
  load prose treatments (Spark) -> drop untranslatable -> drop > N tokens
  -> skip existing
  -> build prompts (ConstrainedDecoder._build_prompt)
  -> submit batches ------------------------------.
  <- results as they finish (unordered) <---------+
  -> validate, save to CouchDB, add provenance    |
                                                  |
Modal (ephemeral app "skol-treatments-json")      |
  Volume "skol-models": HF weights @ pinned rev   |
  class JsonTranslator (GPU, vLLM) <--------------'
    @enter: load vLLM from the Volume (once per container)
    translate(batch) -> [{_id, text, finish_reason, tokens_in, tokens_out}]
  function prepare_models (CPU only)
    download weights into the Volume if missing / wrong revision
```

**Batching.**  The local side submits batches (e.g. 32 records) through
`.map(..., order_outputs=False)` so results stream back as they finish.
vLLM batches requests inside each container; Modal's input concurrency
lets one container work on several batches at once.  Records are saved
one at a time as each batch returns, so an interruption loses at most
the batches in flight.

**What `--gpus N` means.**  The maximum number of GPU containers at
once (Modal `max_containers`).  A second knob, requests per container,
defaults from calibration.

## Command line

```
bin/prepare_modal_models [--model MODEL_ID] [--revision REV] [--check]
    CPU-only.  Ensures the Volume holds the pinned weights; --check
    reports without downloading.  Needs the HF token secret (Mistral is
    gated on Hugging Face; accept the licence on that account first).

bin/treatments_to_json --experiment X --backend modal
    --gpu {T4,L4,A10G,L40S,A100-40GB,A100-80GB,H100,...}
    --gpus N
    --max-cost DOLLARS              # required unless --dry-run
    [--dry-run]                     # cost estimate only; spends nothing
    [--calibrate [N_RECORDS]]       # spend a little to measure throughput
    [existing flags: --skip-existing, --limit, --max-new-tokens,
     --max-description-tokens, ...]
```

`--dry-run` prints: records to process (after filters and
`--skip-existing`), input tokens (exact), estimated output tokens
(production_v4 measurements: median 426, p99 2,659, capped at
`--max-new-tokens`), the calibrated throughput for the chosen GPU, the
wall-clock time at `--gpus N`, and the dollar cost including container
start-up and model load.  Without calibration data for that GPU it says
so instead of guessing.

## Cost estimate

```
gpu_seconds   = output_tokens / calibrated_output_tokens_per_second(gpu)
              + containers * (cold_start + model_load + idle_timeout)
cost          = gpu_seconds * gpu_price(gpu)
              + cpu/memory seconds * their prices
wall_clock    = gpu_seconds / N
```

Output tokens dominate decoding time; input tokens matter for prefill
and are included through calibration.  Prices come from the separate
pricing task (<https://modal.com/pricing>) and live in one table in the
code, with the date they were read.

**Calibration** runs about 50 records drawn from the real distribution
on a chosen GPU, records tokens/second, cut-off rate and per-container
overhead, and stores the result in a JSON file in the repo (see
Calibration storage) for `--dry-run` to read.  Each calibration costs on the
order of a dollar or two.

## Provenance

Every saved record gains:

```json
"provenance": {"backend": "modal", "engine": "vllm", "engine_version": "0.x.y",
               "gpu": "L4", "model": "mistralai/Mistral-7B-Instruct-v0.3",
               "model_revision": "<commit>", "dtype": "bfloat16",
               "decoding": "greedy", "max_tokens": 3072}
```

The local path gets the same field (`backend: local`, `engine:
transformers+outlines`, `dtype: 4bit`).  Records saved before this
change have no field and are known to be local.

## Failures

- Per-call timeout replaces the local 1,200 s worker timeout.
- A failed batch is retried once by Modal; a record that still fails
  is reported and **not saved**, so the next `--skip-existing` run
  retries it.  A record that ran but was cut off or failed validation
  is saved as `{}` (decision 5).
- Losing the local process (suspend, kill) loses only in-flight
  batches; nothing is saved twice because ids are deterministic.
- Stop the local GPU job before starting a Modal run, so two writers
  never work the same records.

## Container image

Pinned versions: vLLM, torch and CUDA matching the chosen GPU, xgrammar.
Built by Modal from the pins, independent of local conda environments.
Locally `skol-vllm312` has vLLM 0.29.1rc1 (dev build), torch
2.13.0+cu130, xgrammar 0.2.6; the image should pin a released vLLM.

**Checked 2026-10-22:** xgrammar 0.2.6 compiles
`TaxonomySchema.to_json_schema_inlined()` at depths 2, 3 and 4.  It
relaxes the schema's `oneOf` to `anyOf`, which is looser;
`TaxonomySchema.validate` still checks every result, and failures are
saved as `{}`.

## Secrets and data

- Modal token: local `~/.modal.toml` (Modal's CLI).  HF token: a Modal
  Secret.  Neither goes in git.
- Description text from published articles, including non-open-access
  journals, is sent to Modal for processing.  Operator accepts this.

## Pricing estimate (2026-10-22)

**Workload** (production_v4_1, measured 2026-10-22 against the
structured DB's 2,613 docs): 42,410 records left after the
untranslatable-text and 4,800-token filters; **50.2 M input tokens**
(mean 1,183 including the 406-token prompt), exact; **21.6 M output
tokens**, estimated from production_v4's 982 outputs (mean 509 capped
at 3,072; median 426, p90 1,009).

**Prices** (<https://modal.com/pricing>, read 2026-10-22; preemptible,
no region selection; region selection costs 1.15-1.75x, non-preemptible
3x).  Starter plan: $30/month free credits.  CPU $0.0000131/core/s,
memory $0.00000222/GiB/s; assumed 2 cores + 16 GiB per container.

**Throughput is estimated, not measured**: aggregate vLLM tokens/s for
Mistral-7B bf16 with continuous batching and guided JSON, as a
low/high range.  T4 is excluded: 14.5 GB of bf16 weights leave no KV
cache on 16 GB.  Costs add decode and prefill time (pessimistic: no
overlap) plus 4 minutes of start-up, load and idle per container.

| GPU | $/s | est. decode tok/s | GPU-hours | cost |
|---|---:|---:|---:|---:|
| H100 | 0.001097 | 2,500-5,000 | 1.5-3.1 | $7-13 |
| A100 40 GB | 0.000583 | 1,200-2,500 | 3.1-6.4 | $7-15 |
| A100 80 GB | 0.000694 | 1,500-3,000 | 2.7-5.4 | $8-15 |
| L40S | 0.000542 | 1,000-2,000 | 3.9-7.7 | $9-17 |
| A10 | 0.000306 | 400-900 | 9.0-19.6 | $12-26 |
| L4 | 0.000222 | 250-600 | 12.8-30.9 | $13-32 |

Faster GPUs are cheaper overall: per-token cost falls faster than the
hourly price rises, and each container's CPU and memory bill by the
second too.  Every option fits the $42.50 workspace limit and finishes
in hours, not weeks, so the deadline stops being the constraint.

**Plan:** calibrate on H100 and A100 40 GB (about $1-2 each), then run
on whichever is cheaper per record, with `--gpus 1` or `2`.

## Calibration storage

Measured throughput goes in a JSON file in the repo, one entry per GPU
type, keyed by what it depends on: engine and version, model revision,
dtype, `max_tokens`, and the image's pins.  Throughput does not depend
on Modal's prices, which live in a separate dated table in the code.
So a price change means editing that table, not recalibrating; an
engine, model or image change means recalibrating.

## Open items

1. **Packaging:** `modal` joins `requirements.txt`; `vllm` lives only
   in the Modal image.
2. **Greedy decoding** (accepted 2026-10-22): temperature 0 with the
   `max_tokens` cap; calibration reports the cut-off rate, and a high
   rate reopens the question.

## Implementation plan

Five tranches, each one test-first cycle (tests, operator review,
commit with xfail, implement).  Ordered so the first spending run comes
early and costs pennies, and the spending guard exists before any large
run.

**Setup (once, operator):** install `modal` in `skol3.14` (pinned in
`requirements.txt` in tranche 1); create the Modal Secret holding the
Hugging Face token; confirm the Mistral licence is accepted on that HF
account; optionally set a Modal workspace spend limit below $42.50.
The Modal token already exists (`~/.modal.toml`).

1. **App shell and model preparation.**  Container image with pinned
   vLLM/torch, the model Volume, a CPU-only `prepare_models` function,
   `bin/prepare_modal_models` (`--check`).  Real run: download the
   pinned weights on CPU (cents).
2. **Inference and spending guard.**  The vLLM GPU class
   (schema-guided, greedy, `max_tokens`); `treatments_to_json
   --backend modal` submitting batches and saving results as they
   arrive; cut-off or invalid output saved as `{}`, failed batches not
   saved; provenance on local and Modal records; `--max-cost` required
   and enforced as elapsed time x GPUs x price (a safe upper bound that
   needs no calibration).  Real run: `--limit 5` (pennies), compared
   with local output.
3. **Calibration.**  `--calibrate N` writes the JSON file keyed by
   engine, model revision, dtype, `max_tokens` and image pins; stale
   entries detected.  Real runs: H100 and A100 40 GB (~$1-2 each).
4. **`--dry-run` estimate.**  Calibration plus the dated price table:
   records, tokens, GPU-hours, wall-clock at `--gpus N`, dollars;
   refuses without calibration for the GPU.  Real run: free.
5. **Production run.**  No new code: stop the local job, dry-run, run
   with `--max-cost` from the estimate, then `assess_treatments`.

Expected: about four working sessions; the production run takes hours.

## Testing

Test-first as usual, with Modal mocked: prompt construction matches the
local path; batching and unordered results; `--dry-run` arithmetic and
its refusal without calibration data; `--max-cost` stops submission;
provenance on saved records; failed records are not saved.  Then a real
smoke run with `--limit 5` before calibration.
