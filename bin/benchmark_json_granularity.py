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
from typing import Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Arm A inherits the pipeline's cap.  Arms B and C need a much smaller
#: one or they measure the cap rather than the content, which is why
#: this is a parameter and not a constant.
DEFAULT_MAX_NEW_TOKENS = 1024
CLAUSE_MAX_NEW_TOKENS = 128


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
    """
    raise NotImplementedError
