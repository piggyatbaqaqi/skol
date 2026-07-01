"""Tests for treatments_to_structured.merge_metric."""

from typing import Any, Dict

from treatments_to_structured.merge_metric import (
    STOP_WORDS,
    is_suspected_merge,
    n_terms_above_k,
    treatment_merge_metric,
)


# ---------------------------------------------------------------------------
# n_terms_above_k — the primary metric
# ---------------------------------------------------------------------------


class TestNTermsAboveK:
    """Count distinct non-stop-word terms appearing >= k times."""

    def test_empty_text_returns_zero(self) -> None:
        assert n_terms_above_k('') == 0

    def test_short_text_no_repeats_returns_zero(self) -> None:
        """A treatment with no repeated non-stop words scores 0."""
        text = 'Pileus brown 3 cm wide.'
        assert n_terms_above_k(text, k=5) == 0

    def test_counts_repeated_technical_terms(self) -> None:
        """The core case: a term repeated >= k times gets counted."""
        # 'pileus' appears 5 times → 1 above k=5.
        text = ' '.join(['pileus'] * 5)
        assert n_terms_above_k(text, k=5) == 1

    def test_below_threshold_excluded(self) -> None:
        """A term appearing < k times doesn't count."""
        # 'pileus' × 4 = below k=5.
        text = ' '.join(['pileus'] * 4)
        assert n_terms_above_k(text, k=5) == 0

    def test_stop_words_dropped(self) -> None:
        """'the' × 100 shouldn't count — it's a stop word."""
        text = ' '.join(['the'] * 100)
        assert n_terms_above_k(text, k=5) == 0

    def test_domain_specific_noise_dropped(self) -> None:
        """Common mycology structural terms (mm, wide, cell) are
        stop-listed so they don't produce false positives on any
        long-enough treatment."""
        text = ' '.join(['mm', 'wide', 'cell'] * 20)
        assert n_terms_above_k(text, k=5) == 0

    def test_case_insensitive(self) -> None:
        """'Pileus' × 3 + 'pileus' × 3 aggregates to 6 → above
        k=5 → 1 term counted."""
        text = 'Pileus Pileus Pileus pileus pileus pileus'
        assert n_terms_above_k(text, k=5) == 1

    def test_two_letter_tokens_dropped(self) -> None:
        """Tokenizer requires 3+ letters (drops 'a', 'is', 'to',
        etc. that would otherwise slip through the stop-word
        filter and inflate counts)."""
        text = ' '.join(['ab', 'cd', 'ef'] * 20)
        assert n_terms_above_k(text, k=5) == 0

    def test_numbers_ignored(self) -> None:
        """Measurements like '3-8' or '250' don't count as terms
        (the tokenizer only accepts letter sequences)."""
        text = '5.5 12 100 250 3-8 ' * 20
        assert n_terms_above_k(text, k=5) == 0

    def test_multiple_qualifying_terms_all_counted(self) -> None:
        """Three distinct non-stop words each above k = returns 3."""
        text = ' '.join(
            ['pileus'] * 6 + ['lamellae'] * 6 + ['spores'] * 6,
        )
        assert n_terms_above_k(text, k=5) == 3

    def test_default_k_is_5(self) -> None:
        """Contract check: unspecified k defaults to 5 (the
        calibration value)."""
        # 5 repeats → threshold met with default.
        text = ' '.join(['pileus'] * 5)
        assert n_terms_above_k(text) == 1
        # 4 repeats → threshold NOT met.
        text = ' '.join(['pileus'] * 4)
        assert n_terms_above_k(text) == 0


# ---------------------------------------------------------------------------
# treatment_merge_metric — reads Treatment doc, calls n_terms_above_k
# ---------------------------------------------------------------------------


class TestTreatmentMergeMetric:
    def test_reads_description_and_diagnosis(self) -> None:
        treatment: Dict[str, Any] = {
            'description': 'pileus ' * 5,
            'diagnosis': 'stipe ' * 5,
        }
        # 'pileus' × 5 + 'stipe' × 5 = 2 terms above k=5.
        assert treatment_merge_metric(treatment) == 2

    def test_missing_description_field_ok(self) -> None:
        treatment = {'diagnosis': 'pileus ' * 5}
        assert treatment_merge_metric(treatment) == 1

    def test_missing_diagnosis_field_ok(self) -> None:
        treatment = {'description': 'pileus ' * 5}
        assert treatment_merge_metric(treatment) == 1

    def test_none_fields_treated_as_empty(self) -> None:
        """CouchDB stores unset fields as null; the metric must
        treat null as absent, not raise."""
        treatment = {'description': None, 'diagnosis': None}
        assert treatment_merge_metric(treatment) == 0

    def test_ignores_other_fields(self) -> None:
        """The metric only looks at description + diagnosis.
        Other fields like `key` (dichotomous key repetitions)
        would inflate the count without signaling a description-
        level merge."""
        treatment = {
            'description': None,
            'diagnosis': None,
            'key': 'pileus ' * 100,
            'figure_captions': 'stipe ' * 100,
        }
        assert treatment_merge_metric(treatment) == 0

    def test_k_parameter_forwarded(self) -> None:
        """k passes through so the caller can tune sensitivity."""
        treatment = {'description': 'pileus ' * 3}
        assert treatment_merge_metric(treatment, k=3) == 1
        assert treatment_merge_metric(treatment, k=5) == 0


# ---------------------------------------------------------------------------
# is_suspected_merge — predicate wrapper
# ---------------------------------------------------------------------------


class TestIsSuspectedMerge:
    # Test tokens: 3-letter words guaranteed to survive the
    # regex (letters only, len >= 3) and to NOT be in STOP_WORDS.
    _TERMS = [
        'foo', 'bar', 'baz', 'qux', 'zap',
        'gob', 'hab', 'ile', 'jam', 'kip',
        'lop', 'mud', 'nib', 'oaf', 'pod',
    ]

    def test_below_threshold_false(self) -> None:
        # 5 non-stop terms × 6 repetitions each = 5 terms above k=5;
        # threshold 10 = not suspect.
        parts = [f'{w} ' * 6 for w in self._TERMS[:5]]
        treatment = {'description': ''.join(parts)}
        assert treatment_merge_metric(treatment) == 5
        assert is_suspected_merge(treatment, threshold=10) is False

    def test_at_threshold_true(self) -> None:
        """At-threshold counts as suspect (>=, not >)."""
        parts = [f'{w} ' * 6 for w in self._TERMS[:10]]
        treatment = {'description': ''.join(parts)}
        assert treatment_merge_metric(treatment) == 10
        assert is_suspected_merge(treatment, threshold=10) is True

    def test_above_threshold_true(self) -> None:
        # 15 non-stop terms × 6 = clearly suspect at threshold 10.
        parts = [f'{w} ' * 6 for w in self._TERMS[:15]]
        treatment = {'description': ''.join(parts)}
        assert is_suspected_merge(treatment, threshold=10) is True

    def test_default_threshold_is_10(self) -> None:
        """Contract check: unspecified threshold defaults to 10
        (calibration value)."""
        # 9 above-k terms = should NOT trip default (< 10).
        parts = [f'{w} ' * 6 for w in self._TERMS[:9]]
        treatment = {'description': ''.join(parts)}
        assert is_suspected_merge(treatment) is False


# ---------------------------------------------------------------------------
# STOP_WORDS composition
# ---------------------------------------------------------------------------


class TestStopWords:
    def test_english_stops_present(self) -> None:
        for w in ('the', 'of', 'and'):
            assert w in STOP_WORDS

    def test_mycology_noise_present(self) -> None:
        """Domain-specific additions from the calibration analysis
        must be in the stop set — without them, 'mm', 'wide',
        'thick' etc. would trip the metric on any long treatment
        regardless of merge status."""
        for w in ('mm', 'wide', 'thick', 'cell', 'diam'):
            assert w in STOP_WORDS
