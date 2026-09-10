#!/usr/bin/env python3
"""Tests for ``bin/benchmark_json_granularity`` (R1's GPU driver).

The arithmetic and the report live in
``treatments_to_structured.granularity_bench`` and are tested there.
What this file covers is the one piece of real logic the driver adds:
**batched generation**, which the existing pipeline has never done.

``_raw_generate`` in ``treatments_json_translator.py`` tokenizes a
single prompt and calls ``model.generate``.  Batching a decoder-only
model is not just passing a list — get it wrong and the benchmark
silently measures something other than what it claims:

  * **Left padding is mandatory.**  Generation continues from the last
    position, so right-padded sequences continue from PAD and produce
    garbage.  The existing tokenizer setup already uses
    ``padding_side="left"``, and this must not diverge from it.
  * **The attention mask must be passed**, or the model attends to pad
    tokens and the shorter prompts in a batch decode differently than
    they would alone — which would make arm C's *outputs* incomparable
    to arm B's while appearing to measure only speed.
  * **Each sequence's continuation must be sliced from its own prompt
    length.**  Decoding the whole row returns the prompt too, and
    ``output_chars`` (which the report normalises by) would then measure
    prompt length rather than generated length.

Fakes throughout: no GPU, no weights.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_json_granularity import (  # type: ignore[import]  # noqa: E402
    batched_generate,
)


class _FakeEncoding(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    """Pads to the longest prompt, one token per character."""

    pad_token = None
    eos_token = '</s>'
    padding_side = 'right'

    def __init__(self):
        self.saw_padding = None
        self.saw_batch = None

    def __call__(self, prompts, return_tensors=None, padding=None,
                 truncation=None, max_length=None):
        self.saw_batch = list(prompts)
        self.saw_padding = padding
        width = max(len(p) for p in prompts)
        ids, mask = [], []
        for p in prompts:
            pad = width - len(p)
            if self.padding_side == 'left':
                ids.append([0] * pad + [ord(c) for c in p])
                mask.append([0] * pad + [1] * len(p))
            else:
                ids.append([ord(c) for c in p] + [0] * pad)
                mask.append([1] * len(p) + [0] * pad)
        return _FakeEncoding(input_ids=ids, attention_mask=mask)

    def decode(self, row, skip_special_tokens=True):
        return ''.join(chr(t) for t in row if t)


class _FakeModel:
    """Echoes the input row and appends a fixed continuation."""

    def __init__(self, continuation='JSON'):
        self.continuation = continuation
        self.saw_kwargs = None

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.saw_kwargs = dict(kwargs)
        self.saw_kwargs['attention_mask'] = attention_mask
        tail = [ord(c) for c in self.continuation]
        return [list(row) + tail for row in input_ids]


class TestBatchedGenerate:
    @pytest.mark.xfail(strict=True, reason="R1 batched_generate not implemented")
    def test_returns_one_string_per_prompt_in_order(self) -> None:
        tok, model = _FakeTokenizer(), _FakeModel()
        out = batched_generate(model, tok, ['abc', 'de', 'fghi'], device='cpu')
        assert len(out) == 3

    @pytest.mark.xfail(strict=True, reason="R1 batched_generate not implemented")
    def test_padding_is_left_not_right(self) -> None:
        """Right padding makes a decoder-only model continue from PAD.
        The benchmark would still produce timings, and they would be
        timings of nonsense."""
        tok, model = _FakeTokenizer(), _FakeModel()
        batched_generate(model, tok, ['abc', 'de'], device='cpu')
        assert tok.padding_side == 'left'

    @pytest.mark.xfail(strict=True, reason="R1 batched_generate not implemented")
    def test_the_attention_mask_reaches_the_model(self) -> None:
        """Without it the shorter prompts attend to padding and decode
        differently in a batch than alone, so arm C would not be
        comparable to arm B."""
        tok, model = _FakeTokenizer(), _FakeModel()
        batched_generate(model, tok, ['abc', 'de'], device='cpu')
        assert model.saw_kwargs['attention_mask'] is not None

    @pytest.mark.xfail(strict=True, reason="R1 batched_generate not implemented")
    def test_only_the_continuation_is_returned_not_the_prompt(self) -> None:
        """`output_chars` normalises the report, so it has to measure
        generated length, not prompt length."""
        tok, model = _FakeTokenizer(), _FakeModel(continuation='XY')
        out = batched_generate(model, tok, ['abcdefgh'], device='cpu')
        assert out == ['XY']

    @pytest.mark.xfail(strict=True, reason="R1 batched_generate not implemented")
    def test_each_sequence_is_sliced_from_its_own_prompt(self) -> None:
        """With left padding every row is the same width, so a single
        shared offset happens to work -- but it must be derived, not
        assumed, or a change of padding side breaks it silently."""
        tok, model = _FakeTokenizer(), _FakeModel(continuation='Z')
        out = batched_generate(model, tok, ['a', 'bbbbbb'], device='cpu')
        assert out == ['Z', 'Z']

    @pytest.mark.xfail(strict=True, reason="R1 batched_generate not implemented")
    def test_a_batch_of_one_still_works(self) -> None:
        tok, model = _FakeTokenizer(), _FakeModel()
        assert len(batched_generate(model, tok, ['solo'], device='cpu')) == 1

    @pytest.mark.xfail(strict=True, reason="R1 batched_generate not implemented")
    def test_an_empty_batch_returns_empty_without_calling_the_model(
            self) -> None:
        tok, model = _FakeTokenizer(), _FakeModel()
        assert batched_generate(model, tok, [], device='cpu') == []
        assert model.saw_kwargs is None

    @pytest.mark.xfail(strict=True, reason="R1 batched_generate not implemented")
    def test_a_pad_token_is_set_when_the_tokenizer_lacks_one(self) -> None:
        """Mistral ships without a pad token; batching needs one or the
        tokenizer raises. The existing pipeline sets it to EOS."""
        tok, model = _FakeTokenizer(), _FakeModel()
        batched_generate(model, tok, ['a', 'bb'], device='cpu')
        assert tok.pad_token == tok.eos_token

    @pytest.mark.xfail(strict=True, reason="R1 batched_generate not implemented")
    def test_generation_is_deterministic(self) -> None:
        """`do_sample=False`, matching `_raw_generate`. A benchmark that
        sampled would vary output length run to run, and output length
        is what an autoregressive decode costs."""
        tok, model = _FakeTokenizer(), _FakeModel()
        batched_generate(model, tok, ['a'], device='cpu')
        assert model.saw_kwargs['do_sample'] is False

    @pytest.mark.xfail(strict=True, reason="R1 batched_generate not implemented")
    def test_max_new_tokens_is_passed_through(self) -> None:
        """Arm A inherits the pipeline's 1024. Arm B and C need a much
        smaller cap or they measure the cap, not the content -- so it
        must be a parameter rather than a constant."""
        tok, model = _FakeTokenizer(), _FakeModel()
        batched_generate(model, tok, ['a'], device='cpu', max_new_tokens=64)
        assert model.saw_kwargs['max_new_tokens'] == 64

    @pytest.mark.parametrize('n', [1, 2, 8])
    @pytest.mark.xfail(strict=True, reason="R1 batched_generate not implemented")
    def test_padding_is_requested_for_every_batch_size(self, n: int) -> None:
        tok, model = _FakeTokenizer(), _FakeModel()
        batched_generate(model, tok, ['x' * (i + 1) for i in range(n)],
                         device='cpu')
        assert tok.saw_padding
