#!/usr/bin/env python3
"""Stop generating once the JSON has closed.

R1 measured that **30 % of whole-description generations run to the
full 1 024-token cap** while the extracted JSON is ~294 tokens — the
model emits its JSON and then keeps talking.  Stopping when the object
closes is the one speed-up available that costs no quality, unlike
lowering the cap, which R1 showed buys 1.44x by truncating twice as
many outputs.

**The criterion is "does this parse", not "do the braces balance".**
Hand-rolling a depth scanner means re-implementing string literals,
escaped quotes and escaped backslashes, and getting any of them wrong
stops generation early on text the model was still writing.
``json.JSONDecoder.raw_decode`` already does all of it, parses one
value from a given offset and *ignores whatever follows*, which is
exactly the shape of the problem.

``transformers``' ``stop_strings=["}"]`` is not usable here: it fires
on the *first* closing brace, and this pipeline's output is nested by
design (``{"Aperture": {"Type": [...]}}``).

**Cost.**  Called once per generation step on the text so far, which is
quadratic in principle.  Measured on a realistic 881-character
generation it is **2.7 µs per call — 2.7 ms across a full 1 024-step
generation**, against ~20–30 s of forward passes for the same
generation.  That is ~0.01 %, so no fast-path guard is warranted; one
gating on a trailing brace would also have to be wrong about
``{"a": 1} Hope that helps!``.
"""

import json

_DECODER = json.JSONDecoder()


def json_is_complete(text: str) -> bool:
    """True once ``text`` contains one complete JSON value.

    Scanning starts at the first ``{`` or ``[`` so the model's preamble
    ("Sure! Here it is:") is skipped, and a closing brace with no
    opener before it is never mistaken for a finished value.
    """
    opens = [i for i in (text.find('{'), text.find('[')) if i >= 0]
    if not opens:
        return False
    try:
        _DECODER.raw_decode(text, min(opens))
    except ValueError:
        return False
    return True
