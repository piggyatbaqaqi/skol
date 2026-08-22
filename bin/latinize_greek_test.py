"""Tests for bin/latinize_greek.

Cases are the standard scholarly rules for rendering Greek into
Latin, the convention botanical and mycological nomenclature
follows (ICN Rec. 60A).  Where a rule has a well-known exemplar
word, the exemplar is in the test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latinize_greek import (  # noqa: E402
    combining_form,
    latinize,
    latin_terminations,
    strip_diacritics,
    transliterate,
)


class TestStripDiacritics:
    """Polytonic Greek is precomposed, which is what defeats a
    bare-letter rule table: ὁ, ή, ὐ, ῦ never match ο, η, υ."""

    def test_tonos_removed(self) -> None:
        assert strip_diacritics('λόγος')[0] == 'λογος'

    def test_perispomeni_and_psili_removed(self) -> None:
        assert strip_diacritics('ῦ')[0] == 'υ'
        assert strip_diacritics('ἀ')[0] == 'α'

    def test_iota_subscript_removed(self) -> None:
        assert strip_diacritics('ῳ')[0] == 'ω'

    def test_rough_breathing_reported_not_dropped(self) -> None:
        """Dasia is the one mark that carries sound: it becomes h."""
        bare, rough = strip_diacritics('ὁ')
        assert bare == 'ο' and rough is True

    def test_smooth_breathing_is_silent(self) -> None:
        bare, rough = strip_diacritics('ἀ')
        assert bare == 'α' and rough is False

    def test_final_sigma_left_alone(self) -> None:
        """Sigma form is orthography, not a diacritic; both map
        to 's' at the letter stage."""
        assert strip_diacritics('λόγος')[0].endswith('ος')


class TestTransliterate:
    """Letter and cluster rules, before any Latin terminations."""

    def test_kappa_becomes_c_not_k(self) -> None:
        """The single most common error. Latin has no k:
        κῆπος -> cepus, not kepos."""
        assert transliterate('κακος') == 'cacos'

    def test_aspirates(self) -> None:
        assert transliterate('θ') == 'th'
        assert transliterate('φ') == 'ph'
        assert transliterate('χ') == 'ch'
        assert transliterate('ψ') == 'ps'
        assert transliterate('ξ') == 'x'

    def test_upsilon_is_y(self) -> None:
        assert transliterate('μυκης') == 'myces'

    def test_eta_and_omega_are_plain_e_and_o(self) -> None:
        assert transliterate('ηω') == 'eo'

    def test_diphthong_ai_oi(self) -> None:
        assert transliterate('αι') == 'ae'
        assert transliterate('οι') == 'oe'

    def test_diphthong_ou_is_u(self) -> None:
        assert transliterate('ου') == 'u'

    def test_diphthong_ei_is_i(self) -> None:
        """χείρ -> chir- (chiroptera), not cheir-."""
        assert transliterate('χειρ') == 'chir'

    def test_diphthong_au_eu_keep_u(self) -> None:
        """υ is y alone but u in a diphthong: αυ -> au, not ay."""
        assert transliterate('αυ') == 'au'
        assert transliterate('ευ') == 'eu'

    def test_gamma_nasal_before_velar(self) -> None:
        """γ before γ κ χ ξ is n: ἄγγελος -> angelus,
        ἄγκυρα -> ancora, σφίγξ -> sphinx."""
        assert transliterate('αγγελος') == 'angelos'
        assert transliterate('αγκυρα') == 'ancyra'
        assert transliterate('σφιγξ') == 'sphinx'

    def test_initial_rho_gets_h(self) -> None:
        """Initial ρ is always aspirated: ῥίζα -> rhiza."""
        assert transliterate('ριζα') == 'rhiza'

    def test_double_rho_becomes_rrh(self) -> None:
        """διάρροια -> diarrhoea."""
        assert transliterate('διαρροια') == 'diarrhoea'

    def test_rough_breathing_becomes_h(self) -> None:
        assert transliterate('ὑπο') == 'hypo'
        assert transliterate('ἁλς') == 'hals'

    def test_smooth_breathing_adds_nothing(self) -> None:
        assert transliterate('ἀπο') == 'apo'


class TestLatinTerminations:
    """ICN Rec. 60A-style ending conversion."""

    def test_os_to_us(self) -> None:
        assert latin_terminations('carpos') == 'carpus'

    def test_on_to_um(self) -> None:
        assert latin_terminations('basidion') == 'basidium'

    def test_es_kept(self) -> None:
        assert latin_terminations('myces') == 'myces'

    def test_leaves_other_endings_alone(self) -> None:
        assert latin_terminations('hyphа'.replace('а', 'a')) == 'hypha'


class TestLatinize:
    """End to end, on words this corpus actually needs."""

    def test_basidion(self) -> None:
        assert latinize('βασίδιον') == 'basidium'

    def test_askos(self) -> None:
        assert latinize('ἀσκός') == 'ascus'

    def test_mykes(self) -> None:
        assert latinize('μύκης') == 'myces'

    def test_rhiza(self) -> None:
        assert latinize('ῥίζα') == 'rhiza'

    def test_sporos(self) -> None:
        assert latinize('σπόρος') == 'sporus'

    def test_multiword_headword_handled_per_word(self) -> None:
        """DCC headwords carry several forms: 'ὁ ἡ τό'."""
        assert latinize('ὁ ἡ τό') == 'ho he to'

    def test_no_greek_characters_survive(self) -> None:
        """The bug in the original: unmatched precomposed letters
        passed straight through into the output."""
        for word in ('αὐτός', 'ἔργον', 'ὅς ἥ ὅ', 'ποιέω'):
            assert all(ord(c) < 128 for c in latinize(word)), word


class TestCombiningForm:
    """Systematic names use the combining form far more than the
    nominative: Acanthocystis, not Acanthuscystis.  Validated
    against the Wikipedia systematic-names list, where 26 of the
    Greek rows give a combining form where we gave a nominative."""

    def test_os_stem_takes_o(self) -> None:
        assert combining_form('acanthus') == 'acantho'

    def test_um_stem_takes_o(self) -> None:
        assert combining_form('basidium') == 'basidio'

    def test_a_stem_takes_o(self) -> None:
        assert combining_form('rhiza') == 'rhizo'

    def test_already_combining_is_unchanged(self) -> None:
        assert combining_form('acantho') == 'acantho'

    def test_short_word_left_alone(self) -> None:
        assert combining_form('os') == 'os'


class TestHyphenHandling:
    """The source lists combining forms as -ouchos / actino-.
    Leading and trailing hyphens are notation, not letters."""

    def test_leading_hyphen_stripped(self) -> None:
        assert transliterate('-οῦχος') == 'uchos'

    def test_trailing_hyphen_stripped(self) -> None:
        assert latinize('ἀκτιν-') == 'actin'
