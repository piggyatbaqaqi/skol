"""Tests for treatments_to_structured.triage_signals."""

from treatments_to_structured.triage_signals import (
    count_description_headers,
    count_diagnosis_headers,
    count_key_couplets,
    count_sp_nov,
    desc_starts_mid_sentence,
    latin_block_count,
    predicted_issues,
    treatment_signals,
)


class TestCountDiagnosisHeaders:
    def test_zero(self) -> None:
        assert count_diagnosis_headers('Pileus brown 3 cm.') == 0

    def test_one(self) -> None:
        text = 'Diagnosis: A species with brown pileus.'
        assert count_diagnosis_headers(text) == 1

    def test_two_signals_merge(self) -> None:
        """The taxon_2a9d07e6 pattern: two Diagnosis: headers in
        one description = clear merge signal."""
        text = (
            'Diagnosis: Conidiomata pycnidial ...\n'
            'A. asci euparaphysati ...\n'
            'Teratosphaeria obscuris ...\n'
            'Diagnosis: Leaf spots primarily epiphyllous ...\n'
        )
        assert count_diagnosis_headers(text) == 2

    def test_empty(self) -> None:
        assert count_diagnosis_headers('') == 0


class TestCountDescriptionHeaders:
    def test_zero(self) -> None:
        assert count_description_headers('Pileus brown.') == 0

    def test_two(self) -> None:
        text = 'Description: sp A ...\nDescription: sp B ...\n'
        assert count_description_headers(text) == 2


class TestCountSpNov:
    def test_zero(self) -> None:
        assert count_sp_nov('Pileus brown.') == 0

    def test_sp_nov(self) -> None:
        assert count_sp_nov('Pseudotrichia stromatophila sp. nov.') == 1

    def test_spec_nov(self) -> None:
        assert count_sp_nov('Saccobolus sphaerosporus spec. nov.') == 1

    def test_two_sp_nov_signals_merge(self) -> None:
        """Two 'sp. nov.' in one description = merge."""
        text = (
            'Gymnopilus laeticolor sp. nov.\n'
            'Gymnopilus ornatulus sp. nov.\n'
        )
        assert count_sp_nov(text) == 2

    def test_case_insensitive(self) -> None:
        assert count_sp_nov('X. yz SP. NOV.') == 1


class TestCountKeyCouplets:
    def test_zero(self) -> None:
        text = 'Pileus 3 cm wide. Stipe 5 mm.'
        assert count_key_couplets(text) == 0

    def test_numbered_couplets(self) -> None:
        """The taxon_5b0a8ce7 pattern: numbered key couplets."""
        text = (
            '15. Basal bulb may be elongate ...\n'
            '\n'
            '16. Basal bulb ovoid to ventricose ...\n'
        )
        assert count_key_couplets(text) == 2

    def test_letter_suffix(self) -> None:
        text = '3a. Pileus brown ...\n3b. Pileus white ...\n'
        assert count_key_couplets(text) == 2

    def test_paren_form(self) -> None:
        text = '5) Pileus brown ...\n6) Pileus white ...\n'
        assert count_key_couplets(text) == 2

    def test_ignores_mid_sentence_numbers(self) -> None:
        """A number mid-sentence like '3 cm wide' shouldn't trip
        the pattern — must be at line start."""
        text = 'Pileus 3. cm wide.'
        assert count_key_couplets(text) == 0


class TestDescStartsMidSentence:
    def test_capital_letter_start_is_fine(self) -> None:
        assert not desc_starts_mid_sentence('Pileus brown.')

    def test_semicolon_start_is_mid_sentence(self) -> None:
        """The taxon_acd88732 pattern: starts with '; perithecia'"""
        assert desc_starts_mid_sentence('; perithecia dispersa')

    def test_comma_start(self) -> None:
        assert desc_starts_mid_sentence(', hyaline, thin-walled')

    def test_lowercase_start(self) -> None:
        assert desc_starts_mid_sentence('perithecia dispersa')

    def test_leading_whitespace_stripped(self) -> None:
        assert desc_starts_mid_sentence('   ; perithecia')
        assert desc_starts_mid_sentence('\n\nperithecia')

    def test_empty_is_not_mid_sentence(self) -> None:
        assert not desc_starts_mid_sentence('')


class TestLatinBlockCount:
    def test_pure_english_zero(self) -> None:
        """Long English text with no Latin morphology."""
        text = (
            'The mushroom cap is convex and brown when young. '
            'The stem is white and slightly bulbous at the base. '
            'The gills are attached and turn dark with age. '
        ) * 3
        assert latin_block_count(text) == 0

    def test_single_latin_block(self) -> None:
        """One Latin diagnosis paragraph = 1 block (normal)."""
        text = (
            'Apothecia sessilia, receptaculum globosum, asci '
            'clavati, ascosporae hyalinae globosae 8-spori.\n'
            '\n'
            'The mushroom cap is convex and brown when young. '
            'The stem is white and slightly bulbous at the base.'
        )
        assert latin_block_count(text) == 1

    def test_alternating_latin_signals_merge(self) -> None:
        """The taxon_572d470e pattern: Latin → English → Latin =
        2 Latin blocks = merge."""
        text = (
            'Apothecia sessilia, receptaculum ovoideum, asci '
            'clavati, ascosporae hyalinae globosae.\n'
            '\n'
            'The mushroom is small, brown, with a convex cap. '
            'The base is bulbous and the stem is short.\n'
            '\n'
            'Apothecia globulare, receptaculum hemisphaericum, '
            'asci cylindracei, ascosporae fusiformes.'
        )
        assert latin_block_count(text) == 2

    def test_empty_zero(self) -> None:
        assert latin_block_count('') == 0


class TestTreatmentSignals:
    def test_full_shape(self) -> None:
        """The composed helper returns all the individual signals
        so the caller can write them as CSV columns."""
        t = {
            'description': 'Pileus brown 3 cm. Stipe long.',
            'diagnosis': 'Similar to X.',
            'synthetic_nomenclature': False,
        }
        s = treatment_signals(t)
        expected_keys = {
            'desc_length', 'diag_length',
            'n_diagnosis_headers', 'n_description_headers',
            'n_sp_nov', 'n_key_couplets',
            'desc_starts_mid_sentence', 'latin_block_count',
            'synthetic_nomenclature',
        }
        assert set(s.keys()) == expected_keys

    def test_missing_fields_handled(self) -> None:
        """Description or diagnosis may be None or absent."""
        t = {'description': None, 'diagnosis': None}
        s = treatment_signals(t)
        assert s['desc_length'] == 0
        assert s['diag_length'] == 0
        assert s['n_diagnosis_headers'] == 0

    def test_synthetic_nomenclature_flag(self) -> None:
        s = treatment_signals({'synthetic_nomenclature': True})
        assert s['synthetic_nomenclature'] is True


class TestPredictedIssues:
    def test_clean_treatment_empty(self) -> None:
        """A clean single-species treatment triggers no flags."""
        signals = {
            'desc_length': 2000,
            'n_diagnosis_headers': 1,
            'n_description_headers': 0,
            'n_sp_nov': 1,
            'n_key_couplets': 0,
            'desc_starts_mid_sentence': False,
            'latin_block_count': 1,
            'synthetic_nomenclature': False,
        }
        assert predicted_issues(signals, merge_metric=3) == ''

    def test_metric_flag(self) -> None:
        """merge_metric above threshold triggers a flag."""
        signals = {
            'desc_length': 2000,
            'n_diagnosis_headers': 0,
            'n_description_headers': 0,
            'n_sp_nov': 0,
            'n_key_couplets': 0,
            'desc_starts_mid_sentence': False,
            'latin_block_count': 0,
            'synthetic_nomenclature': False,
        }
        assert '§6:merge_metric=42' in predicted_issues(
            signals, merge_metric=42,
        )

    def test_multiple_flags_pipe_separated(self) -> None:
        """A treatment failing several detections concatenates
        flags with `|`."""
        signals = {
            'desc_length': 2000,
            'n_diagnosis_headers': 2,
            'n_description_headers': 0,
            'n_sp_nov': 2,
            'n_key_couplets': 0,
            'desc_starts_mid_sentence': True,
            'latin_block_count': 2,
            'synthetic_nomenclature': True,
        }
        result = predicted_issues(signals, merge_metric=15)
        assert '§2:synth_nomen' in result
        assert '§10:mid_sentence' in result
        assert '§6:multi_diagnosis' in result
        assert '§6:multi_sp_nov' in result
        assert '§6:latin_alt' in result
        assert '|' in result

    def test_key_short_triggers(self) -> None:
        """The taxon_5b0a8ce7 pattern: short description with
        any numbered couplet triggers the key-content flag."""
        signals = {
            'desc_length': 406,
            'n_diagnosis_headers': 0,
            'n_description_headers': 0,
            'n_sp_nov': 0,
            'n_key_couplets': 2,
            'desc_starts_mid_sentence': False,
            'latin_block_count': 0,
            'synthetic_nomenclature': False,
        }
        result = predicted_issues(signals, merge_metric=0)
        assert '§8:key_content_short' in result
