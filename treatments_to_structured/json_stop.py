#!/usr/bin/env python3
"""Stop generating once the JSON has closed.

R1 measured that **30 % of whole-description generations run to the
full 1 024-token cap** while the extracted JSON is ~294 tokens — the
model emits its JSON and then keeps talking.  Stopping when the object
closes is the one speed-up available that costs no quality, unlike
lowering the cap, which R1 showed buys 1.44x by truncating twice as
many outputs.

Skeleton.  See ``json_stop_test.py`` for the cases.
"""

from typing import Any, List, Optional


def json_is_complete(text: str) -> bool:
    """True once ``text`` contains one complete JSON value."""
    raise NotImplementedError
