"""Tests for treatments_to_structured.triage_signals."""

from treatments_to_structured.triage_signals import (
    count_description_headers,
    count_diagnosis_headers,
    count_key_couplets,
    count_sp_nov,
    desc_starts_mid_sentence,
    latin_block_count,
    mid_body_description_header,
    predicted_issues,
    tail_clipped,
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

    def test_em_dash_header(self) -> None:
        """The taxon_8f93bded case: `Diagnosis —` (em-dash)
        should count as a Diagnosis header, not just the
        literal-colon form."""
        text = 'Diagnosis — Robust bright reddish orange basidiomes'
        assert count_diagnosis_headers(text) == 1

    def test_en_dash_and_hyphen_forms(self) -> None:
        """Any of colon, hyphen, en-dash, em-dash after
        `Diagnosis` counts as a header."""
        assert count_diagnosis_headers('Diagnosis- brown') == 1
        assert count_diagnosis_headers('Diagnosis – brown') == 1  # en-dash
        assert count_diagnosis_headers('Diagnosis: brown') == 1

    def test_prose_mention_not_counted(self) -> None:
        """`Diagnosis` used in prose (no header punctuation)
        must not count."""
        text = 'A diagnosis of this species is difficult.'
        assert count_diagnosis_headers(text) == 0


class TestMidBodyDescriptionHeader:
    """Fires when a `Description:` header appears at offset > 0
    inside a description field without a preceding Diagnosis
    header (§6 refinement for taxon_a21a83f4)."""

    def test_no_header_is_false(self) -> None:
        assert not mid_body_description_header('Pileus brown 3 cm.')

    def test_offset_zero_is_false(self) -> None:
        """A single `Description:` at position 0 is the field's
        own header — not a species boundary."""
        text = 'Description: Pileus brown 3 cm.'
        assert not mid_body_description_header(text)

    def test_mid_body_no_diagnosis_fires(self) -> None:
        """The taxon_a21a83f4 case: description opens with
        clipped anatomy, then a mid-body `Description:` starts
        species 2."""
        text = (
            'inconspicuous. Mycelium internal; hyphae branched, '
            'septate.  Later text ...\n'
            'Description: Second species starts here.'
        )
        assert mid_body_description_header(text)

    def test_after_diagnosis_is_false(self) -> None:
        """The taxon_8f93bded-shape case: `Diagnosis —` block
        followed by `Description:` is a legitimate single-species
        two-section structure, NOT a merge."""
        text = (
            'Diagnosis — Robust reddish-orange basidiomes with '
            'plane pileus.\n'
            'Description: Pileus 30-60 mm.'
        )
        assert not mid_body_description_header(text)

    def test_after_diagnosis_colon_is_false(self) -> None:
        """Same as above but with the literal-colon Diagnosis
        form."""
        text = (
            'Diagnosis: Basidiomes reddish orange.\n'
            'Description: Pileus 30-60 mm.'
        )
        assert not mid_body_description_header(text)

    def test_empty_is_false(self) -> None:
        assert not mid_body_description_header('')


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


class TestDiagStartsMidSentence:
    """The head-clip predicate applied to the diagnosis
    field (§10-diag).  Reuses `desc_starts_mid_sentence`
    on the diagnosis text — new key
    `diag_starts_mid_sentence` in the signals dict, not a
    new function.  End-to-end tests via `treatment_signals`
    live in TestTreatmentSignals; the direct-predicate
    tests here confirm the same rule applies to arbitrary
    text regardless of which field it came from."""

    def test_empty_is_false(self) -> None:
        assert not desc_starts_mid_sentence('')

    def test_capital_letter_start_ok(self) -> None:
        assert not desc_starts_mid_sentence(
            'Differs from X by having larger spores.'
        )

    def test_semicolon_start(self) -> None:
        """The taxon_e44e35bc / taxon_9ecad903 shape:
        diagnosis starts abruptly with punctuation."""
        assert desc_starts_mid_sentence(
            '; larger spores, longer stipe.'
        )

    def test_lowercase_start(self) -> None:
        assert desc_starts_mid_sentence(
            'differs from X in having a smaller pileus.'
        )

    def test_leading_whitespace_stripped(self) -> None:
        assert desc_starts_mid_sentence(
            '  ; larger spores'
        )

    def test_capital_diagnosis_not_clipped(self) -> None:
        """A comparative-diagnosis opening with a capital
        letter is not clipped."""
        assert not desc_starts_mid_sentence(
            'Similar to Y in having short conidia; '
            'differs from Y by …'
        )


class TestTailClipped:
    def test_empty_is_false(self) -> None:
        assert not tail_clipped('')

    def test_ends_with_period_ok(self) -> None:
        assert not tail_clipped('Pileus brown.')

    def test_ends_with_question_ok(self) -> None:
        assert not tail_clipped('X shape?')

    def test_ends_with_exclamation_ok(self) -> None:
        assert not tail_clipped('X!')

    def test_trailing_whitespace_tolerated(self) -> None:
        assert not tail_clipped('Pileus brown.  \n')

    def test_mid_word_hyphen(self) -> None:
        """The taxon_9ecad903 canonical case: description
        ends with a hyphen mid-word, signalling a page /
        paragraph break the extractor didn't handle."""
        assert tail_clipped('cinnamon or red-')

    def test_hyphen_at_word_boundary_is_false(self) -> None:
        """A hyphenated compound followed by a period ends
        cleanly — the hyphen is intra-word, not a
        line-break marker."""
        assert not tail_clipped('reddish-brown.')

    def test_ends_with_comma(self) -> None:
        assert tail_clipped('Pileus brown,')

    def test_ends_with_semicolon(self) -> None:
        assert tail_clipped('Pileus brown;')

    def test_ends_with_lowercase_no_punctuation(self) -> None:
        """taxon_ae45a05e-shape tail: description just runs
        out mid-clause without any terminal punctuation."""
        assert tail_clipped('Pileus brown')

    def test_ends_with_word_fragment(self) -> None:
        """taxon_ae45a05e first-line ends `Pil…` — with the
        ellipsis stripped for testing, `Pil` alone is
        clipped."""
        assert tail_clipped('Pil')

    def test_ends_with_ellipsis_ok(self) -> None:
        """Some descriptions legitimately end with `…`
        (stylistic).  Treat it as sentence-final."""
        assert not tail_clipped('Pileus brown …')

    def test_ends_with_period_after_paren_ok(self) -> None:
        """`(Fig. 2).` is a common description ending."""
        assert not tail_clipped('Pileus brown (Fig. 2).')

    def test_ends_with_period_after_close_bracket_ok(self) -> None:
        assert not tail_clipped('Pileus brown [see Fig. 2].')


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
            'mid_body_description_header',
            'tail_clipped',
            'diag_starts_mid_sentence',
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
        assert s['tail_clipped'] is False
        assert s['diag_starts_mid_sentence'] is False

    def test_synthetic_nomenclature_flag(self) -> None:
        s = treatment_signals({'synthetic_nomenclature': True})
        assert s['synthetic_nomenclature'] is True

    def test_tail_clipped_fires_end_to_end(self) -> None:
        """taxon_9ecad903 shape: description ends with a
        mid-word hyphen."""
        t = {
            'description': 'Pileus brown, cinnamon or red-',
            'diagnosis': '',
        }
        s = treatment_signals(t)
        assert s['tail_clipped'] is True

    def test_diag_head_clip_fires_end_to_end(self) -> None:
        """taxon_e44e35bc shape: diagnosis starts abruptly
        with a lowercase letter."""
        t = {
            'description': 'Pileus brown 3 cm.',
            'diagnosis': 'larger spores, longer stipe.',
        }
        s = treatment_signals(t)
        assert s['diag_starts_mid_sentence'] is True

    def test_diag_head_clip_gated_on_non_empty(self) -> None:
        """Empty diagnosis field must not fire the diag-
        head-clip flag (empty ≠ clipped).  Distinguishes
        legitimate diagnosis-less treatments (e.g.,
        §0.5 poster-child taxon_0cfe582f) from clipped
        ones."""
        t = {
            'description': 'Pileus brown 3 cm.',
            'diagnosis': '',
        }
        s = treatment_signals(t)
        assert s['diag_starts_mid_sentence'] is False


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

    def test_mid_body_desc_flag(self) -> None:
        """The taxon_a21a83f4 refinement: signals dict carries
        a `mid_body_description_header` boolean; predicted_issues
        fires §6:mid_body_desc when True."""
        signals = {
            'desc_length': 2000,
            'n_diagnosis_headers': 0,
            'n_description_headers': 1,
            'n_sp_nov': 0,
            'n_key_couplets': 0,
            'desc_starts_mid_sentence': False,
            'latin_block_count': 0,
            'synthetic_nomenclature': False,
            'mid_body_description_header': True,
        }
        result = predicted_issues(signals, merge_metric=5)
        assert '§6:mid_body_desc' in result

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

    def test_tail_clip_flag(self) -> None:
        """taxon_9ecad903 / taxon_ae45a05e shape: tail_clipped
        boolean → §10:tail_clip flag in predicted_issues."""
        signals = {
            'desc_length': 400,
            'n_diagnosis_headers': 0,
            'n_description_headers': 0,
            'n_sp_nov': 0,
            'n_key_couplets': 0,
            'desc_starts_mid_sentence': False,
            'latin_block_count': 0,
            'synthetic_nomenclature': False,
            'tail_clipped': True,
        }
        result = predicted_issues(signals, merge_metric=0)
        assert '§10:tail_clip' in result

    def test_diag_head_clip_flag(self) -> None:
        """taxon_e44e35bc / taxon_8d70e41a shape:
        diag_starts_mid_sentence → §10:diag_head_clip
        flag in predicted_issues."""
        signals = {
            'desc_length': 400,
            'n_diagnosis_headers': 0,
            'n_description_headers': 0,
            'n_sp_nov': 0,
            'n_key_couplets': 0,
            'desc_starts_mid_sentence': False,
            'latin_block_count': 0,
            'synthetic_nomenclature': False,
            'diag_starts_mid_sentence': True,
        }
        result = predicted_issues(signals, merge_metric=0)
        assert '§10:diag_head_clip' in result
