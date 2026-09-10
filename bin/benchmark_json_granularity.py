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

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from treatments_to_structured.granularity_bench import Workload

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Arm A inherits the pipeline's cap.  Arms B and C need a much smaller
#: one or they measure the cap rather than the content, which is why
#: this is a parameter and not a constant.
DEFAULT_MAX_NEW_TOKENS = 1024
CLAUSE_MAX_NEW_TOKENS = 128

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


def plan_arms(workloads: Sequence[Workload]) -> List[Workload]:
    """Insert the second ``whole`` arm, at half the generation budget."""
    raise NotImplementedError


def cap_hit_rate(
    outputs: Sequence[Optional[str]],
    tokenizer: Any,
    *,
    cap: int,
) -> Optional[float]:
    """Share of generations that ran to ``cap`` rather than stopping."""
    raise NotImplementedError
