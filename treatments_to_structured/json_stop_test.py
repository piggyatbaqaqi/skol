#!/usr/bin/env python3
"""Tests for ``treatments_to_structured.json_stop``.

R1 measured that **30 % of whole-description generations run to the
full 1 024-token cap** while the extracted JSON is ~294 tokens: the
model emits its JSON and then keeps talking.  Stopping when the JSON
closes is the one speed-up available that costs no quality — unlike
lowering the cap, which R1 showed buys 1.44x by truncating twice as
many outputs.

``transformers`` offers ``stop_strings=["}"]`` and it is the wrong
tool: it fires on the *first* closing brace, and this pipeline's output
is nested by design (``{"Aperture": {"Type": [...]}}``).  Depth has to
be tracked.

Three things make that fiddly, and each is a test below:

  * **Braces inside string literals do not count** — a description may
    legitimately contain one, and taxonomic prose is full of brackets.
  * **An escaped quote does not end a string**, so ``"a \\" b"`` stays
    inside the literal.
  * **The preamble is skipped.**  Mistral opens with "Sure! Here it
    is:", so depth-zero before the first ``{`` must not read as
    "finished".

The scanner is incremental: generation calls it once per step with the
text so far, so it must be cheap and must never report done early.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treatments_to_structured.json_stop import (  # noqa: E402
    json_is_complete,
)


class TestJsonIsComplete:
    @pytest.mark.parametrize('text', [
        '{}',
        '{"a": 1}',
        '{"a": {"b": 2}}',
        '[{"a": 1}]',
    ])
    def test_a_closed_object_is_complete(self, text: str) -> None:
        assert json_is_complete(text) is True

    @pytest.mark.parametrize('text', [
        '{',
        '{"a": 1',
        '{"a": {"b": 2}',
        '{"a": [1, 2',
    ])
    def test_an_unclosed_object_is_not(self, text: str) -> None:
        assert json_is_complete(text) is False

    def test_nesting_is_tracked_not_just_the_first_brace(self) -> None:
        """`stop_strings=["}"]` would fire here after `"b": 2}` and
        truncate the object.  This is the whole reason for the module."""
        assert json_is_complete('{"Aperture": {"Type": ["Ostiolate"]}') is False

    def test_a_preamble_before_the_open_brace_is_skipped(self) -> None:
        """Depth is zero before the first `{`, which must not read as
        finished. Mistral opens with prose."""
        assert json_is_complete('Sure! Here it is: ') is False
        assert json_is_complete('Sure! Here it is: {"a": 1}') is True

    def test_trailing_prose_after_the_close_still_reads_complete(
            self) -> None:
        """The criterion fires the step the object closes, but the
        predicate must stay true if it is called again."""
        assert json_is_complete('{"a": 1} Hope that helps!') is True

    def test_a_brace_inside_a_string_does_not_count(self) -> None:
        """Taxonomic prose carries brackets, and a value is a verbatim
        source phrase per docs/structured-form-schema.md."""
        assert json_is_complete('{"note": "shaped like a }"') is False
        assert json_is_complete('{"note": "shaped like a }"}') is True

    def test_an_open_brace_inside_a_string_does_not_open_depth(
            self) -> None:
        assert json_is_complete('{"note": "a { here"}') is True

    def test_an_escaped_quote_does_not_end_the_string(self) -> None:
        r"""``"a \" b"`` is one string. Reading the escaped quote as a
        terminator would put the scanner back in object context and the
        next ``}`` would close early."""
        assert json_is_complete('{"n": "a \\" }"') is False
        assert json_is_complete('{"n": "a \\" }"}') is True

    def test_an_escaped_backslash_does_not_escape_the_next_quote(
            self) -> None:
        r"""``"a\\"`` ends the string: the backslash is escaped, not the
        quote."""
        assert json_is_complete('{"n": "a\\\\"}') is True

    def test_empty_text_is_not_complete(self) -> None:
        assert json_is_complete('') is False

    def test_a_bare_array_closes(self) -> None:
        assert json_is_complete('[1, 2]') is True
        assert json_is_complete('[1, 2') is False

    def test_a_close_without_an_open_is_not_complete(self) -> None:
        """Garbage must not be reported as finished, or generation stops
        on the first stray brace the model emits."""
        assert json_is_complete('}') is False
        assert json_is_complete('oh dear }') is False
