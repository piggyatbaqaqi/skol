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
overhead, and stores the result (e.g. in `skol_experiments` or a small
JSON file) for `--dry-run` to read.  Each calibration costs on the
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

## Open items

1. **Pricing task:** most cost-effective GPU type and count for ~41,700
   records by 2026-11-17, within the budget.
2. **Greedy vs sampling.**  Proposal: greedy (temperature 0) with the
   `max_tokens` cap; calibration reports the cut-off rate, and a high
   rate reopens the question.
3. **Calibration storage:** `skol_experiments` document vs a JSON file
   in the repo.
4. **Packaging:** `modal` joins `requirements.txt`; `vllm` lives only
   in the Modal image.

## Testing

Test-first as usual, with Modal mocked: prompt construction matches the
local path; batching and unordered results; `--dry-run` arithmetic and
its refusal without calibration data; `--max-cost` stops submission;
provenance on saved records; failed records are not saved.  Then a real
smoke run with `--limit 5` before calibration.
