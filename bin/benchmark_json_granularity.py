#!/usr/bin/env python3
"""R1: measure per-clause against per-description Mistral throughput.

Skeleton.  The arithmetic, the report and the B-vs-C comparison live in
``treatments_to_structured.granularity_bench``; this is the GPU shell
around them.

Three arms (see that module for why two are not enough):

  * ``whole``        — one call per description, as the pipeline runs today
  * ``clause_seq``   — one call per clause, sequential
  * ``clause_batch`` — one call per batch of clauses

The one piece of real logic here is :func:`batched_generate`.  The
existing pipeline has never batched: ``_raw_generate`` in
``treatments_json_translator.py`` tokenizes a single prompt.  Batching a
decoder-only model needs left padding, an attention mask, and a
per-sequence slice of the continuation — get any of them wrong and the
benchmark still reports timings, of the wrong thing.

Writes nothing to CouchDB.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from treatments_to_structured.granularity_bench import (
    Timing,
    Workload,
    build_workloads,
    compare_arms,
    run_workload,
    summarize,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import common_parser, get_env_config  # noqa: E402

BASE_MODEL_ID = 'mistralai/Mistral-7B-Instruct-v0.3'

#: Arm A inherits the pipeline's cap.  Arms B and C need a much smaller
#: one or they measure the cap rather than the content, which is why
#: this is a parameter and not a constant.
DEFAULT_MAX_NEW_TOKENS = 1024
# Calibrated 2026-09-10 from a 2-treatment smoke run: at 128 the clause
# arms hit the cap 63.6 % of the time and 3 of 11 outputs were truncated
# mid-JSON, so the arm was timing the cap rather than the work -- the
# confound arm A is run twice to expose.
CLAUSE_MAX_NEW_TOKENS = 384

#: Per-arm generation cap.  ``whole`` keeps the pipeline's 1 024 because
#: it is the incumbent and ``summarize`` uses it as the denominator;
#: ``whole_512`` is the same work at half the budget.  The stored
#: outputs are median 294 / mean 439 tokens, but those are the
#: *extracted* JSON -- ``generate_json`` runs ``_extract_json_object``
#: over the decoded text -- so they do not say how much was generated.
#: If the model emits its JSON and then talks to the cap, arm A's cost
#: is the cap, and the fix is a stop condition rather than a pipeline.
ARM_CAPS: Dict[str, int] = {
    'whole': DEFAULT_MAX_NEW_TOKENS,
    'whole_512': 512,
    'clause_seq': CLAUSE_MAX_NEW_TOKENS,
    'clause_batch': CLAUSE_MAX_NEW_TOKENS,
}


def batched_generate(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    *,
    device: str = 'cuda',
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> List[Optional[str]]:
    """Generate continuations for a batch of prompts.

    Returns one decoded continuation per prompt, in order, with the
    prompt itself stripped.

    Three things that are not optional for a decoder-only model:

    * **Left padding.**  Generation continues from the last position,
      so a right-padded row continues from PAD.  The run would still
      produce timings.
    * **The attention mask.**  Without it the short prompts in a batch
      attend to padding and decode differently than they would alone,
      which would make ``clause_batch``'s outputs incomparable to
      ``clause_seq``'s while appearing to differ only in speed.
    * **A per-sequence slice.**  Left padding makes every row the same
      width so one shared offset happens to work, but it is derived
      from the encoded width rather than assumed, so a change of
      padding side fails loudly instead of silently returning prompts.
    """
    if not prompts:
        return []

    # Mistral ships without a pad token; batching needs one.  The
    # existing pipeline uses EOS, and diverging from it would change
    # what the model sees.
    if getattr(tokenizer, 'pad_token', None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    encoded = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
    prompt_width = len(encoded['input_ids'][0])

    kwargs = {
        'max_new_tokens': max_new_tokens,
        'do_sample': False,                         # matches _raw_generate
        'temperature': None,
        'top_p': None,
    }
    # ``_raw_generate`` hardcodes ``pad_token_id=2``, which is Mistral's
    # EOS.  Read it off the tokenizer instead, and omit it when there is
    # none rather than requiring every tokenizer to carry one.
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_id is not None:
        kwargs['pad_token_id'] = pad_id

    import torch                                    # noqa: PLC0415
    with torch.no_grad():
        output = model.generate(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            **kwargs,
        )

    return [tokenizer.decode(row[prompt_width:], skip_special_tokens=True)
            for row in output]


def extract_json_outputs(
    outputs: Sequence[Optional[str]],
) -> tuple:
    """Map raw generations to the JSON the pipeline would have stored.

    ``batched_generate`` returns what the model said; the pipeline
    stores what ``_extract_json_object`` finds inside it.  Comparing the
    former answers "did batching change the generation", the latter
    "did it change the result" — and only the second decides whether
    arm C is doing the same work.

    ``None`` for a position that carries no JSON, rather than ``""``:
    an empty string would be scored a disagreement, when the honest
    answer is that there is nothing there to compare.  A ``None`` in
    means a call that failed and stays ``None`` out, so it is never
    promoted to a comparable value.
    """
    # Lazy: treatments_json_translator pulls in torch and pyspark, which
    # the tests have no use for.
    from treatments_json_translator import (       # noqa: PLC0415
        _extract_json_object,
    )
    out: List[Optional[str]] = []
    for text in outputs:
        if text is None:
            out.append(None)
            continue
        obj = _extract_json_object(text)
        out.append(None if obj is None
                   else json.dumps(obj, ensure_ascii=False))
    return tuple(out)


def plan_arms(workloads: Sequence[Workload]) -> List[Workload]:
    """Insert the second ``whole`` arm, at half the generation budget.

    The two differ only in the cap, which is the same discipline the
    clause arms follow: change one variable per comparison.  A treatment
    with no description contributes no ``whole`` work, and duplicating
    an empty arm would add a row scoring zero seconds that means
    nothing.
    """
    out: List[Workload] = []
    for w in workloads:
        out.append(w)
        if w.arm == 'whole' and w.items:
            out.append(Workload('whole_512', list(w.items), w.batch_size))
    return out


def cap_hit_rate(
    outputs: Sequence[Optional[str]],
    tokenizer: Any,
    *,
    cap: int,
) -> Optional[float]:
    """Share of generations that ran to ``cap`` rather than stopping.

    This is what decides whether the cap is the cost.  ``None`` entries
    are calls that failed and are excluded from the denominator rather
    than scored as "did not hit the cap", which would make a broken run
    look well behaved; no generations at all returns ``None``, because
    ``0.0`` reads as a finding and nothing-ran is not.
    """
    done = [o for o in outputs if o is not None]
    if not done:
        return None
    # Encode as a batch of one: a tokenizer given a bare string may
    # iterate it character by character, which silently reports every
    # generation as one token long.
    hits = sum(1 for text in done
               if len(tokenizer([text])['input_ids'][0]) >= cap)
    return hits / len(done)


def _load_model(base_model_id: str, device: str):
    """Load tokenizer and 4-bit model, matching the pipeline's setup."""
    import torch                                    # noqa: PLC0415
    from transformers import (                      # noqa: PLC0415
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    )
    tok = AutoTokenizer.from_pretrained(
        base_model_id, padding_side='left', add_eos_token=True)
    tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb, device_map='auto',
        trust_remote_code=True)
    model.eval()
    return tok, model


def _fetch(server: Any, treatments_db: str, features_db: str, n: int,
           seed: int) -> List[tuple]:
    """(treatment, spans) pairs, sampled from treatments that HAVE spans.

    Sampling from the whole corpus would draw mostly span-less
    treatments -- coverage is 3.4 % -- and the clause arms would then be
    measured on almost no work.
    """
    feats = server[features_db]
    by_taxon: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in feats.view('_all_docs', include_docs=True):
        doc = row.doc or {}
        tid = doc.get('treatment_id') or str(row.id).split(':')[0]
        if doc.get('source_text'):
            by_taxon[tid].append(doc)
    ids = sorted(by_taxon)
    random.Random(seed).shuffle(ids)

    out: List[tuple] = []
    tdb = server[treatments_db]
    for tid in ids:
        if len(out) >= n:
            break
        try:
            doc = dict(tdb[tid])
        except Exception:                            # noqa: BLE001
            continue
        if str(doc.get('description') or '').strip():
            out.append((doc, by_taxon[tid]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[common_parser()],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--treatments', type=int, default=50,
                        help='how many treatments to benchmark')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch for the clause_batch arm')
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--base-model', default=BASE_MODEL_ID)
    parser.add_argument('--report', help='write the JSON report here')
    args = parser.parse_args()
    config = get_env_config(cli_args=args)

    import couchdb                                  # noqa: PLC0415
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (config['couchdb_username'],
                                   config['couchdb_password'])
    exp = server['skol_experiments'][config.get('experiment')
                                     or 'production_v4']
    dbs = exp.get('databases') or {}

    pairs = _fetch(server, dbs['treatments_prose'],
                   'skol_exp_production_v4_02_57_features_canonical',
                   args.treatments, args.seed)
    print(f'{len(pairs)} treatments, '
          f'{sum(len(s) for _, s in pairs)} clauses', file=sys.stderr)

    tok, model = _load_model(args.base_model, 'cuda')

    merged: Dict[str, Timing] = {}
    for treatment, spans in pairs:
        for w in plan_arms(build_workloads(treatment, spans,
                                           batch_size=args.batch_size)):
            cap = ARM_CAPS[w.arm]

            def gen(prompts: List[str], _cap: int = cap) -> List[str]:
                return batched_generate(model, tok, prompts,
                                        device='cuda', max_new_tokens=_cap)

            t = run_workload(w, gen)
            prev = merged.get(w.arm)
            merged[w.arm] = t if prev is None else Timing(
                w.arm, prev.items + t.items, prev.calls + t.calls,
                prev.seconds + t.seconds,
                prev.output_chars + t.output_chars,
                prev.failures + t.failures,
                prev.outputs + t.outputs)

    timings = [merged[a] for a in ARM_CAPS if a in merged]
    report: Dict[str, Any] = {
        'treatments': len(pairs),
        'batch_size': args.batch_size,
        'seed': args.seed,
        'model': args.base_model,
        'arms': summarize(timings, treatments=len(pairs)),
        'cap_hits': {t.arm: cap_hit_rate(t.outputs, tok, cap=ARM_CAPS[t.arm])
                     for t in timings},
    }
    if 'clause_seq' in merged and 'clause_batch' in merged:
        seq, bat = merged['clause_seq'], merged['clause_batch']
        report['seq_vs_batch_raw'] = compare_arms(seq, bat)
        # The comparison that decides whether arm C did the same work:
        # raw generations differ on prose the pipeline never stores.
        report['seq_vs_batch_json'] = compare_arms(
            Timing(seq.arm, outputs=extract_json_outputs(seq.outputs)),
            Timing(bat.arm, outputs=extract_json_outputs(bat.outputs)))
    # Persist the generations.  Not having them is why R1's own
    # agree_rate could not be recomputed after the fact.
    report['outputs'] = {t.arm: list(t.outputs) for t in timings}
    for arm in ARM_CAPS:
        if arm not in report['arms']:
            continue
        row = report['arms'][arm]
        print(f"{arm:14s} {row['seconds']:8.1f}s  "
              f"{row.get('seconds_per_treatment', 0):6.2f}s/treatment  "
              f"speedup {row['speedup']}  cap_hits "
              f"{report['cap_hits'][arm]}", file=sys.stderr)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        Path(args.report).write_text(text, encoding='utf-8')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
