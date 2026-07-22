"""Tests for treatments_to_structured.triage_signals."""

import pytest

from treatments_to_structured.triage_signals import (
    count_description_headers,
    count_diagnosis_headers,
    count_key_couplets,
    count_repeated_section_headers,
    count_repeated_structural_anatomy,
    count_sp_nov,
    desc_starts_mid_sentence,
    latin_between_english,
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

    def test_period_form_header(self) -> None:
        """M2 refinement: `Diagnosis. Something…` (period as
        header terminator) counts as a header when followed by
        whitespace and a capital letter."""
        assert count_diagnosis_headers(
            'Diagnosis. Distinguished by larger spores.',
        ) == 1

    def test_period_form_lowercase_after_not_header(self) -> None:
        """M2 refinement: `Diagnosis. lowercase…` is prose (a
        sentence ending in `Diagnosis`), not a header."""
        assert count_diagnosis_headers(
            'The morphological Diagnosis. more discussion follows.',
        ) == 0

    def test_ufffd_after_diagnosis(self) -> None:
        """M2 refinement: OCR-noise (U+FFFD run) between
        `Diagnosis` and content still counts as a header —
        taxon_e0d2e4bb shape."""
        text = 'Diagnosis��� Species differs by larger spores.'
        assert count_diagnosis_headers(text) == 1


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

    def test_period_form_header(self) -> None:
        """M2 refinement: `Description. Colonies on PDA…`
        (period as header terminator) counts as a header —
        taxon_d65547ed shape."""
        assert count_description_headers(
            'Description. Colonies on PDA reaching 4 cm.',
        ) == 1

    def test_period_form_lowercase_after_not_header(self) -> None:
        """M2 refinement: `Description. lowercase…` is prose,
        not a header."""
        assert count_description_headers(
            'This description. the next sentence continues.',
        ) == 0

    def test_ufffd_after_description(self) -> None:
        """M2 refinement: OCR-noise (U+FFFD run) between
        `Description` and content still counts as a header —
        taxon_e0d2e4bb / taxon_95dbdfb9 shape."""
        text = 'Description��� Leaf spots narrow, oblong.'
        assert count_description_headers(text) == 1

    def test_mid_body_period_form_repeated(self) -> None:
        """Two `Description.` headers in one description →
        multi-species merge signal."""
        text = (
            'Description. Species A pileus 3 cm.\n'
            'Description. Species B pileus 5 cm.'
        )
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


class TestCountRepeatedSectionHeaders:
    """Aggregate detector: counts DISTINCT watchlist section
    headers that appear at least twice in the description.
    §6 idea #3 — a header repetition anywhere in the description
    signals a species boundary.

    Watchlist excludes Description / Diagnosis (handled by
    dedicated counters) and substrate-specific subtypes
    (Cultural characteristics, Colonies on — a single species
    on multiple media legitimately repeats these).
    """

    def test_no_headers_zero(self) -> None:
        assert count_repeated_section_headers(
            'Pileus brown. Stipe long. Spores ellipsoid.'
        ) == 0

    def test_single_occurrence_not_counted(self) -> None:
        """A single header (Observations once) is not a
        repetition."""
        text = 'Pileus brown.\n\nObservations: notable feature X.'
        assert count_repeated_section_headers(text) == 0

    def test_observations_repeated(self) -> None:
        """taxon_592128a8 pattern: two Observations headers
        signal species boundary."""
        text = (
            'Pileus brown.\n\nObservations: notes for species A.'
            '\n\nStipe long.\n\nObservations: notes for species B.'
        )
        assert count_repeated_section_headers(text) == 1

    def test_illustration_repeated(self) -> None:
        """taxon_95dbdfb9 pattern: illustrated-monograph format
        with repeated Illustration headers."""
        text = (
            'Illustration: Braun et al. species A.\n\n'
            'Illustration: species B ref.'
        )
        assert count_repeated_section_headers(text) == 1

    def test_multiple_distinct_repeated_headers(self) -> None:
        """taxon_2a9d07e6 shape: Description and illustration
        + Diagnosis-like repetition patterns combined.  Each
        DISTINCT header keyword repeated counts as 1 toward
        the aggregate."""
        text = (
            'Description and illustration: A ref.\n\n'
            'Etymology: from Greek.\n\n'
            'Description and illustration: B ref.\n\n'
            'Etymology: honoring so-and-so.'
        )
        # Two distinct keywords each repeated → 2.
        assert count_repeated_section_headers(text) == 2

    def test_habitat_repeated(self) -> None:
        """Two species with distinct habitats each headed
        `Habitat:` → repetition."""
        text = (
            'Habitat: on decaying wood.\n\n'
            'Species B.\n\nHabitat: parasitic on Quercus.'
        )
        assert count_repeated_section_headers(text) == 1

    def test_type_repeated(self) -> None:
        text = (
            'Type: USA, holotype.\n\n' * 2
        )
        assert count_repeated_section_headers(text) == 1

    def test_holotype_repeated(self) -> None:
        text = (
            'Holotype: NY 1234.\n\n' * 2
        )
        assert count_repeated_section_headers(text) == 1

    def test_etymology_repeated(self) -> None:
        text = 'Etymology: from Greek.\n\nEtymology: from Latin.'
        assert count_repeated_section_headers(text) == 1

    def test_cultural_characteristics_NOT_in_watchlist(
        self,
    ) -> None:
        """Cultural characteristics is EXCLUDED from the
        watchlist — single-species treatments legitimately
        repeat this header across substrates (PDA, CMA, MEA).
        taxon_b9a6232 false-positive prevention."""
        text = (
            'Cultural characteristics: PDA grows to 4 cm.\n\n'
            'Cultural characteristics: CMA growth differs.'
        )
        assert count_repeated_section_headers(text) == 0

    def test_colonies_on_NOT_in_watchlist(self) -> None:
        """Colonies on is EXCLUDED from the watchlist — a single
        species can have `Colonies on PDA`, `Colonies on MEA`
        legitimately."""
        text = (
            'Colonies on PDA: fluffy white.\n\n'
            'Colonies on MEA: floccose brown.'
        )
        assert count_repeated_section_headers(text) == 0

    def test_description_and_diagnosis_NOT_double_counted(
        self,
    ) -> None:
        """Description and Diagnosis are handled by dedicated
        counters (n_description_headers, n_diagnosis_headers)
        and MUST NOT appear in this aggregate to avoid
        double-flagging the same merge signal."""
        text = (
            'Description: A.\n\nDescription: B.\n\n'
            'Diagnosis: A.\n\nDiagnosis: B.'
        )
        # Watchlist doesn't include Description or Diagnosis
        # → aggregate = 0.  Dedicated counters catch these.
        assert count_repeated_section_headers(text) == 0

    def test_empty_zero(self) -> None:
        assert count_repeated_section_headers('') == 0


class TestCountRepeatedStructuralAnatomy:
    """§6 idea #4 aggregate detector: counts DISTINCT top-level
    fruiting-body / macro-anatomy terms that appear at paragraph
    start ≥2 times.

    Conservative watchlist (fungi-macro terms).  Words that a
    single-species treatment mentions AT MOST ONCE at paragraph
    start — repetition signals a species boundary.

    Paragraph-start = absolute start OR immediately after a
    blank line (`\\n\\s*\\n`).  Same paragraph model
    `latin_block_count` uses.  Fungi-specific vocabulary is a
    known clade-agnostic-design violation; see
    docs/plans/clade-agnostic-detectors.md."""

    def test_no_repetition_zero(self) -> None:
        text = (
            'Ascomata dispersed, black.\n\n'
            'Asci clavati.\n\nParaphyses hyaline.'
        )
        # Ascomata appears once at paragraph start.  Asci and
        # Paraphyses are NOT in the conservative watchlist.
        assert count_repeated_structural_anatomy(text) == 0

    def test_single_occurrence_not_counted(self) -> None:
        """A single Ascomata paragraph-start is not a
        repetition."""
        text = (
            'Some intro text.\n\n'
            'Ascomata dispersed, black.\n\n'
            'Other prose here.'
        )
        assert count_repeated_structural_anatomy(text) == 0

    def test_ascomata_repeated_fires(self) -> None:
        """taxon_173204 shape: 2 similar-species Ascomata
        blocks at English paragraph starts."""
        text = (
            'Ascomata dispersed on the substrate surface, '
            'black in color when mature and dried.\n\n'
            'Additional prose describing anatomy continues '
            'through the treatment body.\n\n'
            'Ascomata scattered and grouped on wood, brown '
            'when young then darkening to black.'
        )
        assert count_repeated_structural_anatomy(text) == 1

    def test_basidiocarps_repeated_fires(self) -> None:
        """taxon_09507677 shape: 3 species with Basidiocarp
        clauses at English paragraph starts."""
        text = (
            'Basidiocarps growing to 0.6 to 1.5 cm across '
            'when fully mature.\n\n'
            'Basidiocarps small and clustered on the wood '
            'substrate surface throughout.\n\n'
            'Basidiocarps large and brightly colored when '
            'fresh and young from the woodland.'
        )
        assert count_repeated_structural_anatomy(text) == 1

    def test_multiple_distinct_repeated_watchlist_words(
        self,
    ) -> None:
        """Ascomata AND Perithecia both repeated → 2."""
        text = (
            'Ascomata black and shining across all the '
            'available surfaces of the wood.\n\n'
            'Perithecia dispersed across the surface and '
            'crowded near the edges of the log.\n\n'
            'Ascomata brown when young and turning darker '
            'with age across all edges of the substrate.\n\n'
            'Perithecia clustered together and grouped in '
            'clusters near the wood substrate edges.'
        )
        assert count_repeated_structural_anatomy(text) == 2

    def test_perithecia_repeated_fires(self) -> None:
        text = (
            'Perithecia dispersed across the substrate '
            'surface and grouped along the wood edges.\n\n'
            'Some intervening prose here to describe the '
            'setting and habitat where they grow.\n\n'
            'Perithecia clustered in groups near the '
            'wood edges throughout the growing season.'
        )
        assert count_repeated_structural_anatomy(text) == 1

    def test_apothecia_repeated_fires(self) -> None:
        text = (
            'Apothecia sessile and scattered on the wood '
            'substrate when they emerge from the bark.\n\n'
            'Additional descriptive prose continues here '
            'to talk about the habitat and location.\n\n'
            'Apothecia stipitate and grouped together '
            'when they are mature and fully developed.'
        )
        assert count_repeated_structural_anatomy(text) == 1

    def test_sporocarp_repeated_fires(self) -> None:
        """Slime mold shape."""
        text = (
            'Sporocarp small and grouped along the wood '
            'log surface when they mature and dry.\n\n'
            'Sporocarp large and yellow when they are '
            'fresh and young in the growing season.'
        )
        assert count_repeated_structural_anatomy(text) == 1

    def test_thallus_repeated_fires(self) -> None:
        """Lichen shape."""
        text = (
            'Thallus foliose and spreading across the '
            'rock surface when growing in the shade.\n\n'
            'Thallus crustose and closely appressed to '
            'the substrate in exposed sun locations.'
        )
        assert count_repeated_structural_anatomy(text) == 1

    def test_micro_anatomy_NOT_in_watchlist(self) -> None:
        """Asci, Paraphyses, Conidia, Ascospores are EXCLUDED
        from the conservative watchlist — single-species
        treatments legitimately discuss these in multiple
        contexts (e.g., macro then microscopic sections).
        taxon_ed2a6f1c would fire on Asci/Paraphyses but that
        FP risk is deferred until §6 idea evaluation shows
        it's worth accepting."""
        text = (
            'Asci clavati.\n\nSome discussion.\n\n'
            'Asci wall structure microscopic.'
        )
        assert count_repeated_structural_anatomy(text) == 0

    def test_mid_sentence_occurrence_not_counted(self) -> None:
        """Mid-paragraph mentions don't count — only paragraph
        starts.  Single-species descriptions can mention the
        top-level structure name multiple times WITHIN a
        paragraph legitimately."""
        text = (
            'Detailed prose here.  The Ascomata are large.\n\n'
            'More prose.  These Ascomata differ from usual.'
        )
        # Neither `Ascomata` is at a paragraph start.
        assert count_repeated_structural_anatomy(text) == 0

    def test_position_zero_counts_as_paragraph_start(self) -> None:
        """Word at text position 0 (no preceding blank line)
        still counts as a paragraph start."""
        text = 'Ascomata early.\n\nAscomata late.'
        assert count_repeated_structural_anatomy(text) == 1

    def test_case_sensitive(self) -> None:
        """Only capitalized Watch words count — matches typical
        section-header capitalization.  Lowercase mentions in
        prose (`these ascomata`) don't fire."""
        text = (
            'ascomata mentioned here.\n\n'
            'ascomata mentioned there.'
        )
        assert count_repeated_structural_anatomy(text) == 0

    def test_plural_and_singular_forms(self) -> None:
        """Basidiome AND Basidiomata are distinct watchlist
        entries — both count.  A treatment mixing them
        (`Basidiome X.\\n\\nBasidiomata Y.`) hits neither
        watchlist entry twice → 0.  Mixed use is unusual;
        typically a treatment sticks with one form."""
        text = (
            'Basidiome small.\n\n'
            'Basidiomata large.'
        )
        # Different watchlist entries, each occurring once.
        assert count_repeated_structural_anatomy(text) == 0

    def test_empty_zero(self) -> None:
        assert count_repeated_structural_anatomy('') == 0

    def test_latin_paragraph_excluded(self) -> None:
        """taxon_d2a4c584 case (2026-07-07 refinement):
        Basidiomata at start of a Latin paragraph +
        Basidiomata at start of an English paragraph is
        the standard Latin+English-pair single-species
        convention, NOT a species boundary."""
        text = (
            'Basidiomata solitaria. Pileus 1.5-3 mm latus, '
            'usque ad 2 mm altus, campanulatus, siccus, '
            'levis, brunneo-pruinosus, margine albus.\n\n'
            'Basidiomata solitary. Pileus 1.5-3 mm broad, '
            'up to 2 mm tall, campanulate, dry, smooth, '
            'brown-pruinose, margin white.'
        )
        assert count_repeated_structural_anatomy(text) == 0

    def test_english_only_repetition_still_fires(self) -> None:
        """taxon_572d470e-like: two English paragraphs each
        starting with Apothecia → fires (species boundary
        within one language)."""
        text = (
            'Apothecia small, black. Discs plane.\n\n'
            'Additional prose here.\n\n'
            'Apothecia larger, brown. Discs concave.'
        )
        assert count_repeated_structural_anatomy(text) == 1

    def test_latin_only_repetition_fires(self) -> None:
        """Two Latin paragraphs each starting with Ascomata
        → 1 (both Latin, same language = merge signal).
        Language-aware counting: repetition within either
        language is a merge signal; cross-language pair
        (one Latin + one English) is NOT."""
        text = (
            'Ascomata perpetua, sessilia, apothecia '
            'globosa, ascosporae hyalinae.\n\n'
            'Ascomata dispersa, asci clavati, ascosporae '
            'ovoidea, paraphyses hyalinae.'
        )
        assert count_repeated_structural_anatomy(text) == 1

    def test_three_paragraphs_mixed_language(self) -> None:
        """E → L → E with Basidiomata at each: only the two
        English mentions count → 1."""
        text = (
            'Basidiomata small, brown. English paragraph.\n\n'
            'Basidiomata parva, brunnea, apothecia globosa, '
            'ascosporae hyalinae ovoidea.\n\n'
            'Basidiomata larger, black. Another English '
            'paragraph continues.'
        )
        assert count_repeated_structural_anatomy(text) == 1


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


class TestLatinBetweenEnglish:
    """Fires when the description has a Latin paragraph
    sandwiched by English paragraphs on BOTH sides
    (E → L → E ordering, §6 idea #1(b)).  Independent of
    `latin_block_count` — fires even when only ONE Latin
    block exists, so long as it sits in a non-terminal
    position."""

    def test_empty_is_false(self) -> None:
        assert not latin_between_english('')

    def test_pure_english_false(self) -> None:
        """Three English paragraphs, no Latin  → False."""
        text = (
            'The mushroom cap is convex and brown.\n\n'
            'The stem is white and slightly bulbous.\n\n'
            'The gills are attached and turn dark.'
        )
        assert not latin_between_english(text)

    def test_pure_latin_false(self) -> None:
        """Contiguous Latin paragraphs, no English between.
        Merges into one block; no sandwich possible."""
        text = (
            'Apothecia sessilia, asci clavati.\n\n'
            'Ascosporae hyalinae globosae.'
        )
        assert not latin_between_english(text)

    def test_latin_first_english_after_false(self) -> None:
        """L → E: normal structure — Latin diagnosis
        followed by English translation or English
        description."""
        text = (
            'Apothecia sessilia, asci clavati, ascosporae '
            'hyalinae globosae.\n\n'
            'The mushroom is small, brown, with a convex '
            'cap. The base is bulbous.'
        )
        assert not latin_between_english(text)

    def test_english_first_latin_after_false(self) -> None:
        """E → L: also legit — English description
        followed by trailing Latin."""
        text = (
            'The mushroom cap is convex and brown when '
            'young. The stem is white.\n\n'
            'Apothecia sessilia, asci clavati, ascosporae '
            'hyalinae globosae.'
        )
        assert not latin_between_english(text)

    def test_english_latin_english_fires(self) -> None:
        """The taxon_9ecad903 canonical shape: Latin
        diagnosis sandwiched between two English
        description paragraphs → merge signal."""
        text = (
            'The mushroom cap is convex and brown when '
            'young. The stem is white.\n\n'
            'Apothecia sessilia, asci clavati, ascosporae '
            'hyalinae globosae.\n\n'
            'The stem is short with a bulbous base. '
            'Spores are ellipsoid.'
        )
        assert latin_between_english(text)

    def test_repeated_pattern_fires(self) -> None:
        """E → L → E → L → E: species-boundary run.  Also
        caught by latin_block_count >= 2, but the E→L→E
        detector fires independently — additive coverage."""
        text = (
            'English description one.  Pileus brown.  '
            'Stipe long and thin.\n\n'
            'Apothecia sessilia asci clavati ascosporae '
            'hyalinae globosae.\n\n'
            'English description two.  Pileus red.  Stipe '
            'short and thick.\n\n'
            'Apothecia globulare asci cylindracei '
            'ascosporae fusiformes.\n\n'
            'English description three.  Pileus yellow.  '
            'Stipe tall.'
        )
        assert latin_between_english(text)

    def test_single_paragraph_false(self) -> None:
        """No paragraph breaks — even if the paragraph
        mixes languages, the detector can't distinguish
        blocks.  Documented limitation."""
        text = (
            'Apothecia sessilia asci clavati mixed with '
            'English text everywhere in one line.'
        )
        assert not latin_between_english(text)

    def test_short_english_before_still_fires(self) -> None:
        """Even brief English paragraphs on each side of
        the Latin block trigger the sandwich detection."""
        text = (
            'Introduction paragraph.\n\n'
            'Apothecia sessilia, asci clavati, ascosporae '
            'hyalinae globosae longa fusiformes.\n\n'
            'Conclusion paragraph.'
        )
        assert latin_between_english(text)


class TestCountDescriptionSpanGaps:
    """Fires on non-contiguous description spans.

    Motivating case: taxon_adcb2fcc (batch-2 §12) has two source
    fragments at lines 11262-11266 and 11282-11283 — a 15-line
    gap between them.  The description text reads as one paragraph
    but was assembled from disjoint regions; downstream review
    depends on the operator catching the gap.
    """

    def test_empty_spans_zero(self) -> None:
        from treatments_to_structured.triage_signals import (
            count_description_span_gaps,
        )
        assert count_description_span_gaps([]) == 0

    def test_single_span_zero(self) -> None:
        from treatments_to_structured.triage_signals import (
            count_description_span_gaps,
        )
        spans = [{'start_line': 10, 'end_line': 25}]
        assert count_description_span_gaps(spans) == 0

    def test_contiguous_pair_zero(self) -> None:
        """Spans separated by a small gap (< default threshold)
        do not fire.  Two-paragraph descriptions with a blank
        line between them typically have gap=2."""
        from treatments_to_structured.triage_signals import (
            count_description_span_gaps,
        )
        spans = [
            {'start_line': 10, 'end_line': 25},
            {'start_line': 27, 'end_line': 40},
        ]
        assert count_description_span_gaps(spans) == 0

    def test_large_gap_pair_one(self) -> None:
        """taxon_adcb2fcc shape: 15-line gap between fragments."""
        from treatments_to_structured.triage_signals import (
            count_description_span_gaps,
        )
        spans = [
            {'start_line': 11262, 'end_line': 11266},
            {'start_line': 11282, 'end_line': 11283},
        ]
        assert count_description_span_gaps(spans) == 1

    def test_three_spans_one_big_gap(self) -> None:
        from treatments_to_structured.triage_signals import (
            count_description_span_gaps,
        )
        spans = [
            {'start_line': 10, 'end_line': 25},
            {'start_line': 27, 'end_line': 40},
            {'start_line': 200, 'end_line': 220},
        ]
        assert count_description_span_gaps(spans) == 1

    def test_three_spans_two_big_gaps(self) -> None:
        from treatments_to_structured.triage_signals import (
            count_description_span_gaps,
        )
        spans = [
            {'start_line': 10, 'end_line': 25},
            {'start_line': 100, 'end_line': 115},
            {'start_line': 300, 'end_line': 320},
        ]
        assert count_description_span_gaps(spans) == 2

    def test_custom_threshold(self) -> None:
        """min_gap kwarg tunes sensitivity."""
        from treatments_to_structured.triage_signals import (
            count_description_span_gaps,
        )
        spans = [
            {'start_line': 10, 'end_line': 25},
            {'start_line': 30, 'end_line': 40},
        ]
        # gap = 30 - 25 = 5
        assert count_description_span_gaps(spans, min_gap=4) == 1
        assert count_description_span_gaps(spans, min_gap=6) == 0

    def test_unsorted_spans_sorted_internally(self) -> None:
        """Real span lists in some CouchDB docs are not ordered;
        detector must sort before pair-wise comparison."""
        from treatments_to_structured.triage_signals import (
            count_description_span_gaps,
        )
        spans = [
            {'start_line': 200, 'end_line': 220},
            {'start_line': 10, 'end_line': 25},
        ]
        assert count_description_span_gaps(spans) == 1

    def test_string_valued_spans_coerced(self) -> None:
        """Some spans in the DB are stored as MapType — start_line
        arrives as a string.  Detector coerces to int."""
        from treatments_to_structured.triage_signals import (
            count_description_span_gaps,
        )
        spans = [
            {'start_line': '10', 'end_line': '25'},
            {'start_line': '200', 'end_line': '220'},
        ]
        assert count_description_span_gaps(spans) == 1

    def test_missing_line_keys_skipped(self) -> None:
        """Spans without line-number keys are ignored — a
        defensive fallback, not a normal path."""
        from treatments_to_structured.triage_signals import (
            count_description_span_gaps,
        )
        spans = [
            {'start_line': 10, 'end_line': 25},
            {'paragraph_number': 33},
            {'start_line': 200, 'end_line': 220},
        ]
        assert count_description_span_gaps(spans) == 1


class TestCountPopulatedFields:
    """Counts non-empty section fields (excluding nomenclature).

    Motivating case: taxon_3e98d44d (batch-2 §11) — a gen. nov.
    treatment whose extracted description is a single clean
    272-char paragraph but etymology / type_designation / notes /
    biology / etc. all read as empty.  n_populated_fields = 1
    on this treatment despite the source PDF having six populated
    sections.  Combined with desc_length in the CSV, operators
    can filter for silent-failure candidates.

    No auto-flag fires yet — see the session decision to defer
    the flag until fixtures carry full section content.
    """

    _FIELDS = (
        'description', 'diagnosis', 'etymology', 'distribution',
        'materials_examined', 'type_designation', 'biology',
        'notes', 'key', 'figure_captions',
    )

    def test_empty_treatment_zero(self) -> None:
        from treatments_to_structured.triage_signals import (
            count_populated_fields,
        )
        assert count_populated_fields({}) == 0

    def test_all_none_zero(self) -> None:
        from treatments_to_structured.triage_signals import (
            count_populated_fields,
        )
        t = {f: None for f in self._FIELDS}
        assert count_populated_fields(t) == 0

    def test_all_empty_string_zero(self) -> None:
        from treatments_to_structured.triage_signals import (
            count_populated_fields,
        )
        t = {f: '' for f in self._FIELDS}
        assert count_populated_fields(t) == 0

    def test_only_description_one(self) -> None:
        """taxon_3e98d44d silent-failure shape."""
        from treatments_to_structured.triage_signals import (
            count_populated_fields,
        )
        t = {
            'description': 'Colonies white, butyrous, smooth.',
            'diagnosis': '',
            'etymology': None,
        }
        assert count_populated_fields(t) == 1

    def test_description_plus_diagnosis_two(self) -> None:
        from treatments_to_structured.triage_signals import (
            count_populated_fields,
        )
        t = {
            'description': 'Colonies white.',
            'diagnosis': 'Similar to X.',
        }
        assert count_populated_fields(t) == 2

    def test_all_ten_populated(self) -> None:
        from treatments_to_structured.triage_signals import (
            count_populated_fields,
        )
        t = {f: 'x' for f in self._FIELDS}
        assert count_populated_fields(t) == 10

    def test_nomenclature_not_counted(self) -> None:
        """The `treatment` (nomenclature) field is NOT one of the
        10 section fields — a treatment with only nomenclature
        populated still reports 0."""
        from treatments_to_structured.triage_signals import (
            count_populated_fields,
        )
        t = {'treatment': 'Foo bar sp. nov.'}
        assert count_populated_fields(t) == 0


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
            'n_repeated_section_headers',
            'n_repeated_structural_anatomy',
            'n_sp_nov', 'n_key_couplets',
            'desc_starts_mid_sentence', 'latin_block_count',
            'latin_between_english',
            'mid_body_description_header',
            'tail_clipped',
            'diag_starts_mid_sentence',
            'authored_binomial_in_desc',
            'synthetic_nomenclature',
            'n_description_span_gaps',
            'n_populated_fields',
            'n_source_anchors',
        }
        assert set(s.keys()) == expected_keys

    def test_description_span_gaps_reads_from_treatment(self) -> None:
        """The composed helper picks up ``description_spans`` from
        the treatment dict and forwards to
        count_description_span_gaps."""
        t = {
            'description': 'Pileus brown 3 cm.',
            'diagnosis': '',
            'description_spans': [
                {'start_line': 10, 'end_line': 25},
                {'start_line': 200, 'end_line': 220},
            ],
        }
        s = treatment_signals(t)
        assert s['n_description_span_gaps'] == 1

    def test_description_span_gaps_absent_zero(self) -> None:
        """No ``description_spans`` key → 0.  Treatments in the
        older schema (pre-span-tracking) do not carry spans."""
        t = {'description': 'Pileus brown 3 cm.', 'diagnosis': ''}
        s = treatment_signals(t)
        assert s['n_description_span_gaps'] == 0

    def test_populated_fields_counts_all_sections(self) -> None:
        """taxon_3e98d44d silent-failure signature: only
        description populated → n_populated_fields = 1."""
        t = {
            'description': 'Colonies white, butyrous, smooth.',
            'diagnosis': '',
            'etymology': None,
            'materials_examined': '',
        }
        s = treatment_signals(t)
        assert s['n_populated_fields'] == 1

    def test_source_anchors_reads_from_treatment(self) -> None:
        """Trello #401 Phase 1: the composed helper reads
        ``source_anchors`` from the treatment dict and reports its
        length as ``n_source_anchors``."""
        t = {
            'description': 'Pileus brown 3 cm.',
            'diagnosis': '',
            'source_anchors': [
                {'kind': 'pdf', 'page': '3', 'label': '3'},
                {'kind': 'plazi', 'uuid': '0A4F6E6CD877...'},
            ],
        }
        s = treatment_signals(t)
        assert s['n_source_anchors'] == 2

    def test_source_anchors_absent_zero(self) -> None:
        """No ``source_anchors`` key → 0.  Legacy treatments
        (pre-Phase-1) do not carry the field."""
        t = {'description': 'Pileus brown 3 cm.', 'diagnosis': ''}
        s = treatment_signals(t)
        assert s['n_source_anchors'] == 0

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

    def test_authored_binomial_flag_flows_through(self) -> None:
        """Pass-through of the caller-supplied authored-binomial
        boolean (§6 idea #2).  gn_client computes it via HTTP;
        treatment_signals just relays."""
        t = {'description': 'Pileus brown 3 cm.', 'diagnosis': ''}
        s = treatment_signals(t, authored_binomial_in_desc=True)
        assert s['authored_binomial_in_desc'] is True

    def test_authored_binomial_default_is_false(self) -> None:
        """Default (kwarg not supplied) → False.  Preserves the
        "not evaluated" == "not fired" behaviour for CLI runs
        where gn services are unavailable."""
        t = {'description': 'Pileus brown 3 cm.', 'diagnosis': ''}
        s = treatment_signals(t)
        assert s['authored_binomial_in_desc'] is False

    def test_latin_between_english_fires_end_to_end(self) -> None:
        """The taxon_9ecad903 shape flows through
        treatment_signals via the description field."""
        t = {
            'description': (
                'The mushroom cap is convex and brown.  '
                'Stipe long.\n\n'
                'Apothecia sessilia, asci clavati, '
                'ascosporae hyalinae globosae.\n\n'
                'The stem is short with a bulbous base. '
                'Spores ellipsoid.'
            ),
            'diagnosis': '',
        }
        s = treatment_signals(t)
        assert s['latin_between_english'] is True


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

    def test_desc_span_gap_flag(self) -> None:
        """taxon_adcb2fcc shape: at least one span-gap fires
        §12:desc_span_gap."""
        signals = {
            'desc_length': 676,
            'n_diagnosis_headers': 0,
            'n_description_headers': 0,
            'n_sp_nov': 0,
            'n_key_couplets': 0,
            'desc_starts_mid_sentence': False,
            'latin_block_count': 0,
            'synthetic_nomenclature': False,
            'n_description_span_gaps': 1,
        }
        result = predicted_issues(signals, merge_metric=0)
        assert '§12:desc_span_gap' in result

    def test_populated_fields_signal_no_flag(self) -> None:
        """n_populated_fields is exposed as a signal for CSV
        filtering but does NOT emit an auto-flag.  Session
        decision: defer the flag until fixtures carry full
        section content across all entries."""
        signals = {
            'desc_length': 272,
            'n_diagnosis_headers': 0,
            'n_description_headers': 0,
            'n_sp_nov': 0,
            'n_key_couplets': 0,
            'desc_starts_mid_sentence': False,
            'latin_block_count': 0,
            'synthetic_nomenclature': False,
            'n_populated_fields': 1,
        }
        result = predicted_issues(signals, merge_metric=0)
        assert 'sparse_treatment' not in result
        assert '§11' not in result

    def test_latin_ele_flag(self) -> None:
        """The taxon_9ecad903 shape: latin_between_english
        True → §6:latin_ele flag.  Independent of
        latin_block_count (can fire when count == 1)."""
        signals = {
            'desc_length': 2000,
            'n_diagnosis_headers': 0,
            'n_description_headers': 0,
            'n_sp_nov': 0,
            'n_key_couplets': 0,
            'desc_starts_mid_sentence': False,
            'latin_block_count': 1,
            'latin_between_english': True,
            'synthetic_nomenclature': False,
        }
        result = predicted_issues(signals, merge_metric=0)
        assert '§6:latin_ele' in result

    def test_authored_binomial_flag(self) -> None:
        """taxon_83e36037 / taxon_2a9d07e6 shape:
        authored_binomial_in_desc True → §6:authored_binomial
        flag.  Detection via gnfinder+gnparser (§6 idea #2)."""
        signals = {
            'desc_length': 2000,
            'n_diagnosis_headers': 0,
            'n_description_headers': 0,
            'n_sp_nov': 0,
            'n_key_couplets': 0,
            'desc_starts_mid_sentence': False,
            'latin_block_count': 0,
            'synthetic_nomenclature': False,
            'authored_binomial_in_desc': True,
        }
        result = predicted_issues(signals, merge_metric=0)
        assert '§6:authored_binomial' in result

    def test_multi_section_header_flag(self) -> None:
        """M2: n_repeated_section_headers >= 1 fires
        §6:multi_section_header.  Independent of dedicated
        Description / Diagnosis counters."""
        signals = {
            'desc_length': 2000,
            'n_diagnosis_headers': 0,
            'n_description_headers': 0,
            'n_repeated_section_headers': 1,
            'n_sp_nov': 0,
            'n_key_couplets': 0,
            'desc_starts_mid_sentence': False,
            'latin_block_count': 0,
            'synthetic_nomenclature': False,
        }
        result = predicted_issues(signals, merge_metric=0)
        assert '§6:multi_section_header' in result

    def test_multi_structural_anatomy_flag(self) -> None:
        """M2 Group B: n_repeated_structural_anatomy >= 1
        fires §6:multi_structural_anatomy."""
        signals = {
            'desc_length': 2000,
            'n_diagnosis_headers': 0,
            'n_description_headers': 0,
            'n_sp_nov': 0,
            'n_key_couplets': 0,
            'desc_starts_mid_sentence': False,
            'latin_block_count': 0,
            'n_repeated_structural_anatomy': 1,
            'synthetic_nomenclature': False,
        }
        result = predicted_issues(signals, merge_metric=0)
        assert '§6:multi_structural_anatomy' in result
