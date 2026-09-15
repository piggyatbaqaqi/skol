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
    JsonComplete,
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


class _FakeTok:
    """Decodes one character per token id, ignoring ids below 32."""

    def decode(self, row, skip_special_tokens=True):
        return ''.join(chr(int(t)) for t in row if int(t) >= 32)


def _ids(*rows: str):
    import torch
    width = max(len(r) for r in rows)
    return torch.tensor([[ord(c) for c in r.ljust(width, ' ')] for r in rows])


class TestJsonCompleteCriterion:
    """The ``transformers`` wrapper around :func:`json_is_complete`.

    Two things it must get right, and both are silent when wrong:

    * **Decode only the continuation.**  The prompt now carries
      ``DEFAULT_PROMPT``, which contains JSON examples — scoring the
      whole row would satisfy the criterion before the model has
      written a single token, and every generation would return empty.
    * **Answer per sequence.**  ``transformers`` takes a bool tensor,
      one entry per row.  Returning a scalar would stop the whole batch
      as soon as any member finished and truncate the rest.
    """

    def test_a_finished_row_reports_true(self) -> None:
        ids = _ids('PROMPT{"a": 1}')
        assert bool(JsonComplete(_FakeTok(), 6)(ids, None)[0]) is True

    def test_an_unfinished_row_reports_false(self) -> None:
        ids = _ids('PROMPT{"a": 1')
        assert bool(JsonComplete(_FakeTok(), 6)(ids, None)[0]) is False

    def test_json_in_the_prompt_does_not_satisfy_it(self) -> None:
        """DEFAULT_PROMPT contains JSON examples. Scoring the whole row
        would stop generation at step one, every time."""
        ids = _ids('{"example": 1}')
        assert bool(JsonComplete(_FakeTok(), 14)(ids, None)[0]) is False

    def test_the_answer_is_per_sequence_not_a_scalar(self) -> None:
        """A scalar would stop the whole batch when any member closed,
        truncating the others mid-object."""
        ids = _ids('PROMPT{"a": 1}', 'PROMPT{"b": 2')
        out = JsonComplete(_FakeTok(), 6)(ids, None)
        assert len(out) == 2
        assert bool(out[0]) is True and bool(out[1]) is False

    def test_an_empty_continuation_is_not_complete(self) -> None:
        ids = _ids('PROMPT')
        assert bool(JsonComplete(_FakeTok(), 6)(ids, None)[0]) is False

    def test_the_result_is_a_bool_tensor_on_the_input_device(self) -> None:
        """`transformers` indexes it against the running batch, so a
        list or a float tensor is not interchangeable."""
        import torch
        ids = _ids('PROMPT{}')
        out = JsonComplete(_FakeTok(), 6)(ids, None)
        assert out.dtype == torch.bool
        assert out.device == ids.device
