"""Tests for treatments_json_translator helper functions.

Covers ``build_llm_input_text`` — the helper that builds the
text fed to the JSON-translation LLM from a Treatment CouchDB doc.
Per docs/v3_buildout.md and the user directive, the LLM input must
include both Description AND Diagnosis so the annotator sees the
``feature->subfeature->value`` statements that Diagnosis typically
carries.

Also covers the description token threshold: ``make_token_counter``,
``partition_by_token_budget`` and
``TreatmentsJSONTranslator._drop_oversized_descriptions``; skipping
untranslatable descriptions (``partition_untranslatable_descriptions``,
``TreatmentsJSONTranslator._drop_untranslatable_descriptions``); and the
generation token cap in ``_constrained_inference_worker``.

Run with: python -m pytest treatments_json_translator_test.py -v
"""

import contextlib
import io
import math
import unittest
import queue
import sys
import types
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest


from treatments_json_translator import (
    TreatmentsJSONTranslator,
    _chunk_budget_chars,
    _constrained_inference_worker,
    build_llm_input_text,
    make_token_counter,
    partition_by_token_budget,
    partition_untranslatable_descriptions,
)
from treatments_to_structured.modal_calibration import calibration_key
from treatments_to_structured.modal_inference import (
    taxon_doc_id,
)


class TestBuildLlmInputText(unittest.TestCase):
    def test_description_only(self):
        doc = {"description": "Cap red.", "diagnosis": None}
        self.assertEqual(build_llm_input_text(doc), "Cap red.")

    def test_diagnosis_only(self):
        """Orphan-Diagnosis Treatments (synthetic Nomen ignotum stubs
        built from a bare Diagnosis paragraph) still get their content
        to the LLM."""
        doc = {"description": None, "diagnosis": "Differs from X by Y."}
        self.assertEqual(
            build_llm_input_text(doc),
            "Differs from X by Y.",
        )

    def test_both_present_concatenated_with_double_newline(self):
        doc = {
            "description": "Cap red.",
            "diagnosis": "Differs from X by Y.",
        }
        self.assertEqual(
            build_llm_input_text(doc),
            "Cap red.\n\nDiffers from X by Y.",
        )

    def test_both_empty_returns_empty_string(self):
        """Treatments with neither section yield an empty string —
        the LLM annotator is expected to short-circuit on empty input
        rather than the helper itself filtering them out."""
        self.assertEqual(build_llm_input_text({}), "")
        self.assertEqual(
            build_llm_input_text({"description": "", "diagnosis": ""}),
            "",
        )
        self.assertEqual(
            build_llm_input_text({"description": None, "diagnosis": None}),
            "",
        )

    def test_missing_keys_treated_as_absent(self):
        """A doc dict missing the keys entirely behaves the same as
        having them set to None."""
        self.assertEqual(
            build_llm_input_text({"description": "X"}),
            "X",
        )
        self.assertEqual(
            build_llm_input_text({"diagnosis": "Y"}),
            "Y",
        )

    def test_diagnosis_only_no_leading_separator(self):
        """When only Diagnosis is present, the LLM input must not
        start with a stray ``\\n\\n`` separator."""
        out = build_llm_input_text(
            {"description": None, "diagnosis": "D."},
        )
        self.assertFalse(out.startswith("\n"))


class TestChunkBudget(unittest.TestCase):
    """Contract for ``_chunk_budget_chars``.

    A chunk's prompt is ``scaffold + content``. The budget reserves
    ``SCAFFOLD_RESERVE_TOKENS`` for the scaffold and spends the rest on
    content, converting tokens->chars by an implicit chars/token factor.
    For the tokenizer not to exceed its configured ``model_max_length``
    (which triggers the HF "Token indices sequence length is longer than
    the specified maximum" warning), the *worst-case* token count of a
    full-budget chunk plus the scaffold reserve must stay within
    ``max_length``. That holds iff the implicit factor does not exceed
    the corpus's worst-case character density.
    """

    # Token-dense taxonomic text (unicode measurements like "10-15 x 3-4 um",
    # μm, ×, digits, scientific names) packs ~3.0 chars/token in the Mistral
    # tokenizer. The budget's implicit factor must not exceed this floor.
    WORST_CASE_CHARS_PER_TOKEN = 3.0
    # Mirrors the reserve documented in _chunk_budget_chars.
    SCAFFOLD_RESERVE_TOKENS = 768

    def test_budget_stays_within_token_limit(self):
        for max_length in (2048, 4096, 8192):
            with self.subTest(max_length=max_length):
                budget = _chunk_budget_chars(max_length)
                worst_case_content_tokens = math.ceil(
                    budget / self.WORST_CASE_CHARS_PER_TOKEN
                )
                total = worst_case_content_tokens + self.SCAFFOLD_RESERVE_TOKENS
                self.assertLessEqual(
                    total,
                    max_length,
                    f"max_length={max_length}: a full-budget chunk "
                    f"({budget} chars) can reach {worst_case_content_tokens} "
                    f"content tokens; with the {self.SCAFFOLD_RESERVE_TOKENS}-"
                    f"token scaffold that is {total} tokens, exceeding "
                    f"{max_length} and tripping the tokenizer overflow warning.",
                )

    def test_budget_honors_minimum_floor(self):
        """When max_length is small enough that the computed budget would
        fall below 1000 chars, the floor keeps it at 1000."""
        self.assertEqual(_chunk_budget_chars(900), 1000)


class FakeTokenizer:
    """Stands in for a Hugging Face tokenizer: two tokens per word.

    Not one, so an implementation that counts words instead of
    tokens gets different numbers and fails.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, texts: List[str], add_special_tokens: bool = True
                 ) -> Dict[str, List[List[int]]]:
        self.calls.append(
            {"texts": list(texts), "add_special_tokens": add_special_tokens})
        return {"input_ids": [[0] * (2 * len(t.split())) for t in texts]}


def word_counter(texts: List[str]) -> List[int]:
    return [len(t.split()) for t in texts]


def refuse_to_count(texts: List[str]) -> List[int]:
    raise AssertionError("counted tokens although no threshold is set")


def item(doc_id: str, description: Any) -> Dict[str, Any]:
    return {"_id": doc_id, "treatment": f"T {doc_id}",
            "description": description}


class TestMakeTokenCounter(unittest.TestCase):

    def test_counts_each_text_in_order(self):
        count = make_token_counter(FakeTokenizer())
        self.assertEqual(count(["a b c", "d", "e f"]), [6, 2, 4])

    def test_excludes_special_tokens(self):
        """The threshold is about the description's own length; BOS/EOS
        belong to the prompt scaffold."""
        tokenizer = FakeTokenizer()
        make_token_counter(tokenizer)(["a b"])
        self.assertFalse(tokenizer.calls[0]["add_special_tokens"])

    def test_tokenizes_the_whole_list_in_one_call(self):
        """Batching is what keeps 80k descriptions at seconds, not minutes."""
        tokenizer = FakeTokenizer()
        make_token_counter(tokenizer)(["a", "b c", "d e f"])
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertEqual(tokenizer.calls[0]["texts"], ["a", "b c", "d e f"])

    def test_empty_list(self):
        self.assertEqual(make_token_counter(FakeTokenizer())([]), [])


class TestPartitionByTokenBudget(unittest.TestCase):

    def test_none_threshold_keeps_everything_without_counting(self):
        items = [item("a", "one two"), item("b", "x " * 500)]
        kept, dropped = partition_by_token_budget(items, refuse_to_count, None)
        self.assertEqual(kept, items)
        self.assertEqual(dropped, [])

    def test_threshold_is_inclusive(self):
        """A description of exactly max_tokens is kept; one more is dropped."""
        items = [item("at", "a b c"), item("over", "a b c d")]
        kept, dropped = partition_by_token_budget(items, word_counter, 3)
        self.assertEqual([i["_id"] for i in kept], ["at"])
        self.assertEqual(dropped, [("over", 4)])

    def test_kept_items_keep_their_order_and_fields(self):
        items = [item("a", "x"), item("big", "x x x x x"), item("b", "x x"),
                 item("c", "x")]
        items[0]["page_number"] = 7
        kept, _ = partition_by_token_budget(items, word_counter, 2)
        self.assertEqual([i["_id"] for i in kept], ["a", "b", "c"])
        self.assertIs(kept[0], items[0])

    def test_dropped_reports_every_id_with_its_count(self):
        items = [item("a", "x " * 10), item("b", "x"), item("c", "x " * 7)]
        _, dropped = partition_by_token_budget(items, word_counter, 5)
        self.assertEqual(dropped, [("a", 10), ("c", 7)])

    def test_missing_description_counts_as_empty_and_is_kept(self):
        seen: List[List[str]] = []

        def counter(texts: List[str]) -> List[int]:
            seen.append(list(texts))
            return word_counter(texts)

        items = [item("none", None), item("empty", "")]
        kept, dropped = partition_by_token_budget(items, counter, 1)
        self.assertEqual(seen, [["", ""]])
        self.assertEqual(kept, items)
        self.assertEqual(dropped, [])

    def test_counts_all_descriptions_in_one_call(self):
        calls: List[List[str]] = []

        def counter(texts: List[str]) -> List[int]:
            calls.append(list(texts))
            return word_counter(texts)

        partition_by_token_budget(
            [item("a", "x"), item("b", "y z")], counter, 5)
        self.assertEqual(calls, [["x", "y z"]])

    def test_empty_input(self):
        self.assertEqual(
            partition_by_token_budget([], word_counter, 5), ([], []))

    def test_non_positive_threshold_is_rejected(self):
        for bad in (0, -1):
            with self.subTest(max_tokens=bad):
                with self.assertRaises(ValueError):
                    partition_by_token_budget(
                        [item("a", "x")], word_counter, bad)


def make_translator(**kwargs: Any) -> TreatmentsJSONTranslator:
    with contextlib.redirect_stdout(io.StringIO()):
        return TreatmentsJSONTranslator(
            spark=mock.Mock(), couchdb_url="http://localhost:5984",
            base_model_id="example/model", **kwargs)


class TestDropOversizedDescriptions(unittest.TestCase):

    def run_drop(self, translator: TreatmentsJSONTranslator,
                 items: List[Dict[str, Any]]):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            kept = translator._drop_oversized_descriptions(items)
        return kept, out.getvalue()

    def test_no_threshold_returns_input_and_loads_no_tokenizer(self):
        translator = make_translator(token_counter=refuse_to_count)
        items = [item("a", "x " * 100000)]
        with mock.patch("treatments_json_translator.AutoTokenizer") as tok:
            kept, _ = self.run_drop(translator, items)
        self.assertEqual(kept, items)
        tok.from_pretrained.assert_not_called()

    def test_drops_only_oversized(self):
        translator = make_translator(
            max_description_tokens=3, token_counter=word_counter)
        items = [item("small", "a b"), item("big", "a b c d e"),
                 item("edge", "a b c")]
        kept, _ = self.run_drop(translator, items)
        self.assertEqual([i["_id"] for i in kept], ["small", "edge"])

    def test_reports_each_dropped_record_and_a_summary(self):
        translator = make_translator(
            max_description_tokens=3, token_counter=word_counter)
        items = [item("small", "a"), item("taxon_big", "a b c d e f g")]
        _, log = self.run_drop(translator, items)
        self.assertIn("taxon_big", log)
        self.assertIn("7", log)
        self.assertRegex(log, r"[Dd]ropp?ed 1 of 2\b")
        self.assertIn("3", log)

    def test_silent_when_nothing_is_dropped(self):
        translator = make_translator(
            max_description_tokens=10, token_counter=word_counter)
        _, log = self.run_drop(translator, [item("a", "x")])
        self.assertNotRegex(log, r"[Dd]ropp?ed")

    def test_builds_counter_from_base_model_tokenizer_once(self):
        """Counts must come from the configured model's tokenizer:
        switching --base-model changes what a token is.  "x y" is two
        words but four FakeTokenizer tokens, so counting words would keep
        it under the 3-token threshold."""
        translator = make_translator(max_description_tokens=3)
        with mock.patch("treatments_json_translator.AutoTokenizer") as tok:
            tok.from_pretrained.return_value = FakeTokenizer()
            first, _ = self.run_drop(translator, [item("a", "x y")])
            second, _ = self.run_drop(translator, [item("b", "x")])
        self.assertEqual(first, [])
        self.assertEqual([i["_id"] for i in second], ["b"])
        tok.from_pretrained.assert_called_once()
        self.assertEqual(
            tok.from_pretrained.call_args.args[0], "example/model")


REPLACEMENT = "\ufffd"


class TestPartitionUntranslatableDescriptions(unittest.TestCase):
    """43 % of production_v4_1 has no description or diagnosis text; the
    model invents a taxon for each (a frog, for one Nomen ignotum)."""

    def kept_ids(self, *items: Dict[str, Any]) -> List[str]:
        kept, _ = partition_untranslatable_descriptions(list(items))
        return [i["_id"] for i in kept]

    def test_none_empty_and_whitespace_are_dropped(self):
        items = [item("none", None), item("empty", ""),
                 item("blank", "  \n\t "), item("text", "Pileus convex.")]
        kept, dropped = partition_untranslatable_descriptions(items)
        self.assertEqual([i["_id"] for i in kept], ["text"])
        self.assertEqual(dropped, ["none", "empty", "blank"])

    def test_text_without_letters_is_dropped(self):
        self.assertEqual(self.kept_ids(
            item("dots", "... .. ."),
            item("rule", "~" + "-" * 60),
            item("numbers", "2.5-4 \u00d7 0.5-0.8 (14.0)"),  # × not x
        ), [])

    def test_one_letter_is_enough(self):
        """A measurement with a unit has letters: µ and m both count."""
        self.assertEqual(self.kept_ids(item("mu", "2.5-4 µm")), ["mu"])

    def test_replacement_characters_are_not_letters(self):
        self.assertEqual(
            self.kept_ids(item("ocr", REPLACEMENT * 40 + " 53 " +
                               REPLACEMENT * 10)), [])

    def test_mostly_replacement_characters_is_ocr_noise(self):
        """>= 50 % U+FFFD of non-space characters: bibliography shreds
        with 'et al.' surviving, nothing to extract."""
        noisy = REPLACEMENT * 30 + "et al." + REPLACEMENT * 30
        self.assertEqual(self.kept_ids(item("noisy", noisy)), [])

    def test_threshold_is_half_of_non_space_characters(self):
        half = REPLACEMENT * 10 + "abcdefghij"        # exactly 50 %
        under = REPLACEMENT * 9 + "abcdefghij" + " " * 20  # 9 of 19
        self.assertEqual(
            self.kept_ids(item("half", half), item("under", under)),
            ["under"])

    def test_partly_garbled_description_is_kept(self):
        """10-50 % U+FFFD is typical of readable descriptions with a few
        unextractable glyphs (μ, ×, ligatures)."""
        text = ("Basidiomes" + REPLACEMENT * 5 + " separable, gelatinous, "
                "hygrophanous, soft when fresh" + REPLACEMENT * 5)
        self.assertEqual(self.kept_ids(item("ok", text)), ["ok"])

    def test_kept_items_keep_their_order_and_identity(self):
        items = [item("a", "x"), item("e", ""), item("b", "y")]
        kept, _ = partition_untranslatable_descriptions(items)
        self.assertEqual([i["_id"] for i in kept], ["a", "b"])
        self.assertIs(kept[0], items[0])

    def test_empty_input(self):
        self.assertEqual(
            partition_untranslatable_descriptions([]), ([], []))


class TestDropUntranslatableDescriptions(unittest.TestCase):

    def run_drop(self, translator: TreatmentsJSONTranslator,
                 items: List[Dict[str, Any]]):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            kept = translator._drop_untranslatable_descriptions(items)
        return kept, out.getvalue()

    def test_drops_empty_even_without_a_token_threshold(self):
        translator = make_translator()
        kept, _ = self.run_drop(
            translator, [item("e", ""), item("t", "Pileus convex.")])
        self.assertEqual([i["_id"] for i in kept], ["t"])

    def test_needs_no_tokenizer(self):
        translator = make_translator(max_description_tokens=5)
        with mock.patch("treatments_json_translator.AutoTokenizer") as tok:
            self.run_drop(translator, [item("e", ""), item("t", "x")])
        tok.from_pretrained.assert_not_called()

    def test_logs_a_summary_not_every_id(self):
        """Tens of thousands of empties would flood the log."""
        translator = make_translator()
        _, log = self.run_drop(translator, [
            item("taxon_e1", ""), item("taxon_e2", "..."),
            item("taxon_t", "x")])
        self.assertRegex(log, r"[Dd]ropp?ed 2 of 3 ")
        self.assertNotIn("taxon_e1", log)

    def test_silent_when_nothing_is_dropped(self):
        _, log = self.run_drop(make_translator(), [item("t", "x")])
        self.assertNotRegex(log, r"[Dd]ropp?ed")


class FakeConstrainedDecoder:
    """Stands in for skol.constrained_decoder.ConstrainedDecoder."""

    instances: List["FakeConstrainedDecoder"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        FakeConstrainedDecoder.instances.append(self)

    def load_model(self, **kwargs: Any) -> None:
        pass

    def extract_features(self, description: str,
                         **kwargs: Any) -> Dict[str, Any]:
        return {"pileus": {"shape": ["convex"]}}


def fake_decoder_module() -> types.ModuleType:
    """What the worker's ``from skol.constrained_decoder import ...``
    finds.  Under pytest ``skol`` is the repo-root package, so the real
    submodule cannot be patched by name."""
    module = types.ModuleType("skol.constrained_decoder")
    module.ConstrainedDecoder = FakeConstrainedDecoder  # type: ignore[attr-defined]
    module.TaxonomySchema = mock.Mock()  # type: ignore[attr-defined]
    return module


class TestConstrainedWorkerGenerationCap(unittest.TestCase):

    def run_worker(self, max_new_tokens: int):
        FakeConstrainedDecoder.instances = []
        model_config = {
            'base_model_id': 'example/model', 'device': 'cpu',
            'load_in_4bit': False, 'prompt': '',
            'max_new_tokens': max_new_tokens,
        }
        results: "queue.Queue[Any]" = queue.Queue()
        with mock.patch.dict(sys.modules, {
                "skol.constrained_decoder": fake_decoder_module()}), \
                contextlib.redirect_stdout(io.StringIO()):
            _constrained_inference_worker(
                [item("t", "Pileus convex.")], model_config, 10, results)
        return results.get_nowait()

    def test_decoder_is_capped_at_max_new_tokens(self):
        status, _ = self.run_worker(3072)
        self.assertEqual(status, 'success')
        decoder = FakeConstrainedDecoder.instances[0]
        self.assertEqual(decoder.kwargs.get('max_tokens'), 3072)

    def test_worker_still_returns_results(self):
        status, results = self.run_worker(3072)
        self.assertEqual(status, 'success')
        self.assertIn('"pileus"', results['t'])


# ---------------------------------------------------------------------------
# Modal backend: translate_items_modal and local provenance
# ---------------------------------------------------------------------------

class PromptDecoder:
    """Stands in for ConstrainedDecoder: records its system prompt."""

    def __init__(self, system_prompt: str = '', **kwargs: Any) -> None:
        self.system_prompt = system_prompt

    def _build_prompt(self, description: str) -> str:
        return f'[{self.system_prompt}]::{description}'


class StubSchema:
    """Stands in for TaxonomySchema: rejects features holding 'bad'."""

    def __init__(self, max_depth: int = 4, min_depth: int = 2) -> None:
        self.max_depth, self.min_depth = max_depth, min_depth

    def to_json_schema_inlined(self) -> Dict[str, Any]:
        return {'type': 'object', 'max_depth': self.max_depth}

    def validate(self, data: Dict[str, Any]) -> bool:
        if 'bad' in data:
            raise ValueError('schema says no')
        return True


def prompt_decoder_module() -> types.ModuleType:
    module = types.ModuleType("skol.constrained_decoder")
    module.ConstrainedDecoder = PromptDecoder  # type: ignore[attr-defined]
    module.TaxonomySchema = StubSchema  # type: ignore[attr-defined]
    return module


class ModalDB:
    """The slice of couchdb.Database translate_items_modal uses."""

    def __init__(self, docs: Optional[List[Dict[str, Any]]] = None) -> None:
        self.docs = {d['_id']: d for d in docs or []}
        self.saved: List[Dict[str, Any]] = []

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.docs

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.docs.get(doc_id)

    def save(self, doc: Dict[str, Any]) -> None:
        self.saved.append(dict(doc))
        self.docs[doc['_id']] = dict(doc)


class Allow:
    def __init__(self, n: int = 10 ** 6) -> None:
        self.left = n

    def start(self) -> None:
        pass

    def allows_more(self) -> bool:
        self.left -= 1
        return self.left >= 0

    def spent_upper_bound(self) -> float:
        return 0.25


def modal_item(doc_id: str, description: str = 'Pileus convex.',
               treatment: Optional[str] = None) -> Dict[str, Any]:
    return {'_id': doc_id, 'treatment': treatment or f'T {doc_id}',
            'description': description, 'ingest': {}, 'line_number': 1,
            'paragraph_number': 1, 'page_number': '1',
            'empirical_page_number': '1'}


class FakeModalCall:
    """Answers each prompt; ``outcomes`` maps _id to a canned reply."""

    def __init__(self, outcomes: Optional[Dict[str, Any]] = None,
                 fail_ids: Optional[set] = None) -> None:
        self.outcomes = outcomes or {}
        self.fail_ids = fail_ids or set()
        self.batches: List[List[Dict[str, Any]]] = []
        self.schemas: List[Dict[str, Any]] = []
        self.max_tokens: List[int] = []

    def __call__(self, batch: List[Dict[str, Any]], schema: Dict[str, Any],
                 max_tokens: int) -> List[Dict[str, Any]]:
        self.batches.append(batch)
        self.schemas.append(schema)
        self.max_tokens.append(max_tokens)
        if any(b['_id'] in self.fail_ids for b in batch):
            raise RuntimeError('container preempted')
        out = []
        for b in batch:
            text, reason = self.outcomes.get(
                b['_id'], ('{"pileus": {"shape": ["convex"]}}', 'stop'))
            out.append({'_id': b['_id'], 'text': text,
                        'finish_reason': reason, 'tokens_in': 9,
                        'tokens_out': 4})
        return out


class TestTranslateItemsModal:

    def run_modal(self, items: List[Dict[str, Any]], call: FakeModalCall,
                  db: Optional[ModalDB] = None, guard: Any = None,
                  **kwargs: Any) -> Any:
        translator = make_translator(**kwargs.pop('translator', {}))
        db = db if db is not None else ModalDB()
        with mock.patch.dict(sys.modules, {
                "skol.constrained_decoder": prompt_decoder_module()}), \
                contextlib.redirect_stdout(io.StringIO()):
            counts = translator.translate_items_modal(
                items, db, call, guard or Allow(), gpu='H100', **kwargs)
        return translator, db, counts

    def sent_ids(self, call: FakeModalCall) -> List[str]:
        return [b['_id'] for batch in call.batches for b in batch]

    def test_untranslatable_descriptions_are_never_sent(self) -> None:
        call = FakeModalCall()
        _, db, _ = self.run_modal(
            [modal_item('a'), modal_item('e', description=''),
             modal_item('n', description='... --')], call)
        assert self.sent_ids(call) == ['a']
        assert len(db.saved) == 1

    def test_oversized_descriptions_are_never_sent(self) -> None:
        call = FakeModalCall()
        self.run_modal(
            [modal_item('short', 'a b'), modal_item('long', 'a b c d e')],
            call, translator={'max_description_tokens': 3,
                              'token_counter': word_counter})
        assert self.sent_ids(call) == ['short']

    def test_prompts_are_built_as_on_the_local_path(self) -> None:
        call = FakeModalCall()
        translator, _, _ = self.run_modal(
            [modal_item('a', 'Pileus convex.')], call)
        assert call.batches[0] == [
            {'_id': 'a', 'prompt': f'[{translator.prompt}]::Pileus convex.'}]

    def test_sends_schema_and_max_new_tokens(self) -> None:
        call = FakeModalCall()
        translator, _, _ = self.run_modal(
            [modal_item('a')], call, translator={'schema_max_depth': 3})
        assert call.schemas[0] == {'type': 'object', 'max_depth': 3}
        assert call.max_tokens[0] == translator.max_new_tokens

    def test_batches_have_the_requested_size(self) -> None:
        call = FakeModalCall()
        self.run_modal([modal_item(f'i{n}') for n in range(5)], call,
                       modal_batch_size=2)
        # Batches run two at a time in threads, so arrival order varies.
        assert sorted(len(b) for b in call.batches) == [1, 2, 2]

    def test_saves_ok_results_with_modal_provenance(self) -> None:
        call = FakeModalCall()
        item = modal_item('a', 'Pileus convex.', treatment='Russula x')
        _, db, counts = self.run_modal([item], call)
        doc = db.saved[0]
        assert doc['_id'] == taxon_doc_id('Russula x', 'Pileus convex.')
        assert doc['json_annotated'] == {'pileus': {'shape': ['convex']}}
        assert doc['json_valid'] is True
        assert doc['provenance']['backend'] == 'modal'
        assert doc['provenance']['gpu'] == 'H100'
        assert doc['provenance']['engine'] == 'vllm'
        assert counts['ok'] == 1

    def test_cut_off_and_invalid_are_saved_as_empty(self) -> None:
        call = FakeModalCall({'cut': ('{"pileus": {', 'length'),
                              'bad': ('{"bad": ["x"]}', 'stop')})
        _, db, counts = self.run_modal(
            [modal_item('cut'), modal_item('bad')], call)
        assert [d['json_annotated'] for d in db.saved] == [{}, {}]
        assert all(d['json_valid'] is False for d in db.saved)
        assert counts['cut_off'] == 1 and counts['invalid'] == 1

    def test_failed_batch_saves_nothing(self) -> None:
        call = FakeModalCall(fail_ids={'b'})
        _, db, counts = self.run_modal(
            [modal_item('a'), modal_item('b'), modal_item('c')], call,
            modal_batch_size=1)
        saved = {d['treatment'] for d in db.saved}
        assert saved == {'T a', 'T c'}
        assert counts['failed_records'] == 1

    def test_update_keeps_the_revision(self) -> None:
        item = modal_item('a', 'Pileus convex.', treatment='Russula x')
        doc_id = taxon_doc_id('Russula x', 'Pileus convex.')
        db = ModalDB([{'_id': doc_id, '_rev': '2-b'}])
        _, db, _ = self.run_modal([item], FakeModalCall(), db=db)
        assert db.saved[0]['_rev'] == '2-b'

    def test_guard_stops_submission(self) -> None:
        call = FakeModalCall()
        _, db, counts = self.run_modal(
            [modal_item(f'i{n}') for n in range(4)], call,
            guard=Allow(1), modal_batch_size=1, max_in_flight=1)
        assert len(call.batches) == 1
        assert len(db.saved) == 1
        assert counts['not_submitted_records'] == 3
        assert counts['spent_upper_bound'] == 0.25


class TestLocalProvenance:

    def test_constrained_4bit(self) -> None:
        translator = make_translator(use_constrained_decoding=True,
                                     load_in_4bit=True)
        prov = translator._local_provenance()
        assert prov['backend'] == 'local'
        assert prov['engine'] == 'transformers+outlines'
        assert prov['dtype'] == '4bit'
        assert prov['gpu'] is None
        assert prov['model'] == 'example/model'
        assert prov['max_tokens'] == translator.max_new_tokens

    def test_unconstrained(self) -> None:
        prov = make_translator(use_constrained_decoding=False,
                               load_in_4bit=False)._local_provenance()
        assert prov['engine'] == 'transformers'
        assert prov['dtype'] == 'float16'


# ---------------------------------------------------------------------------
# Modal calibration
# ---------------------------------------------------------------------------

class ClockedCall(FakeModalCall):
    """FakeModalCall that advances a fake clock by one second per call."""

    def __init__(self, clock: List[float]) -> None:
        super().__init__()
        self.clock = clock

    def __call__(self, batch: List[Dict[str, Any]], schema: Dict[str, Any],
                 max_tokens: int) -> List[Dict[str, Any]]:
        self.clock[0] += 1.0
        return super().__call__(batch, schema, max_tokens)


def varied_items(n: int) -> List[Dict[str, Any]]:
    """Items whose descriptions run from 1 to n words."""
    return [modal_item(f'i{k:03}', ' '.join(['w'] * (k + 1)))
            for k in range(n)]


class TestCalibrateModal(unittest.TestCase):

    def run_calibration(self, items: List[Dict[str, Any]],
                        batch_sizes: tuple = (2, 4),
                        records_per_config: int = 4) -> Any:
        clock = [100.0]
        call = ClockedCall(clock)
        translator = make_translator(token_counter=word_counter,
                                     max_new_tokens=3072)
        db = ModalDB()
        with mock.patch.dict(sys.modules, {
                "skol.constrained_decoder": prompt_decoder_module()}), \
                contextlib.redirect_stdout(io.StringIO()):
            entry = translator.calibrate_modal(
                items, db, call, Allow(), 'A100-40GB',
                batch_sizes=batch_sizes,
                records_per_config=records_per_config,
                clock=lambda: clock[0])
        return translator, call, db, entry

    def test_warm_up_with_one_record_first(self) -> None:
        _, call, _, entry = self.run_calibration(varied_items(20))
        assert len(call.batches[0]) == 1
        assert entry['startup_s'] == 1.0

    def test_each_batch_size_gets_its_own_slice(self) -> None:
        _, call, _, _ = self.run_calibration(varied_items(20))
        sent = [b['_id'] for batch in call.batches for b in batch]
        assert len(sent) == 1 + 2 * 4
        assert len(set(sent)) == len(sent)

    def test_batch_sizes_are_used_in_order(self) -> None:
        _, call, _, _ = self.run_calibration(varied_items(20))
        after_warm_up = [len(b) for b in call.batches[1:]]
        assert after_warm_up == [2, 2, 4]

    def test_every_result_is_saved(self) -> None:
        """Calibration does real work: nothing it translates is wasted."""
        _, call, db, _ = self.run_calibration(varied_items(20))
        assert len(db.saved) == 9

    def test_entry(self) -> None:
        translator, _, _, entry = self.run_calibration(varied_items(20))
        assert entry['gpu'] == 'A100-40GB'
        assert entry['key'] == calibration_key(translator.max_new_tokens)
        assert entry['measured_at']
        assert [c['batch_size'] for c in entry['configs']] == [2, 4]
        first = entry['configs'][0]
        assert first['records'] == 4
        assert first['elapsed_s'] == 2.0          # two calls of 2
        assert first['output_tokens_per_s'] == 4 * 4 / 2.0

    def test_drops_untranslatable_before_sampling(self) -> None:
        items = varied_items(9) + [modal_item(f'e{k}', '') for k in range(5)]
        _, call, _, _ = self.run_calibration(items)
        sent = {b['_id'] for batch in call.batches for b in batch}
        assert not any(i.startswith('e') for i in sent)

    def test_too_few_records_sends_nothing(self) -> None:
        clock = [0.0]
        call = ClockedCall(clock)
        translator = make_translator(token_counter=word_counter)
        with mock.patch.dict(sys.modules, {
                "skol.constrained_decoder": prompt_decoder_module()}), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, '9'):
                translator.calibrate_modal(
                    varied_items(8), ModalDB(), call, Allow(), 'H100',
                    batch_sizes=(2, 4), records_per_config=4,
                    clock=lambda: clock[0])
        assert call.batches == []


class TestPlanModal(unittest.TestCase):

    @pytest.mark.xfail(strict=True, reason="plan_modal not implemented yet")
    def test_counts_what_would_be_sent(self) -> None:
        translator = make_translator(max_description_tokens=3,
                                     token_counter=word_counter)
        items = [modal_item('a', 'one two'), modal_item('b', 'x y z'),
                 modal_item('e', ''), modal_item('n', '...'),
                 modal_item('long', 'a b c d e')]
        with contextlib.redirect_stdout(io.StringIO()):
            plan = translator.plan_modal(items)
        self.assertEqual(plan, {'records': 2, 'description_tokens': 5})


if __name__ == "__main__":
    unittest.main()
