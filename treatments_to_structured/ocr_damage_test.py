#!/usr/bin/env python3
"""Tests for ``treatments_to_structured.ocr_damage``.

Every example here is real text from the corpus, and every threshold
is a measured corpus quantile rather than a guess — see
``docs/data_quality_production_v4_model.md`` §9 and D8.

The point of the module is that OCR damage is **not one thing**.  A
detector tuned on one mode is blind to the others, which is how
``taxon_8d815304`` came to look clean to the rejoin metric while being
one of the worst-damaged treatments in the corpus.
"""

import sys
from pathlib import Path
from typing import Set

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treatments_to_structured.ocr_damage import (  # noqa: E402
    MODE_REPLACEMENT,
    MODE_SPACING,
    MODE_SUBSTITUTION,
    OcrDamage,
)


def _vocab() -> Set[str]:
    """Small vocabulary sufficient for the examples below."""
    return {
        'anamorph', 'fungi', 'fission', 'which', 'straight', 'similar',
        'surface', 'index', 'hyaline', 'with', 'biological', 'slants',
        'spores', 'conidia', 'pileus', 'stipe', 'the', 'and', 'are',
        'basidia', 'smooth', 'brown', 'cells', 'wide', 'long',
    }


def _pad(core: str, total: int = 42) -> str:
    """Pad to just past the minimum-token guard, without adding damage.

    Deliberately tight.  Over-padding dilutes the damage density below
    the measured corpus thresholds, which would force the test to
    weaken a real threshold to pass -- exactly backwards.
    """
    have = len([t for t in core.split() if any(ch.isalpha() for ch in t)])
    return core + ' ' + ' '.join(['spores'] * max(0, total - have))


class TestMeasurability:
    """Short text yields no rates, rather than noisy ones."""

    def test_short_text_is_not_measurable(self) -> None:
        assert OcrDamage('Pileus brown.', vocabulary=_vocab()).measurable \
            is False

    def test_long_enough_text_is_measurable(self) -> None:
        assert OcrDamage(_pad('Pileus brown.'),
                         vocabulary=_vocab()).measurable is True

    def test_rates_are_zero_when_not_measurable(self) -> None:
        """Never return a rate computed from a handful of tokens."""
        d = OcrDamage('short', vocabulary=_vocab())
        assert d.rejoin_rate() == 0.0
        assert d.substitution_rate() == 0.0
        assert d.modes() == ()


class TestReplacementMode:
    """§9 mode A — U+FFFD runs."""

    def test_detects_replacement_characters(self) -> None:
        d = OcrDamage(_pad('Spores ��� globose.'),
                      vocabulary=_vocab())
        assert d.replacement_rate() > 0
        assert MODE_REPLACEMENT in d.modes()

    def test_clean_text_has_no_replacement_chars(self) -> None:
        d = OcrDamage(_pad('Spores globose.'), vocabulary=_vocab())
        assert d.replacement_rate() == 0.0
        assert MODE_REPLACEMENT not in d.modes()


class TestSpacingMode:
    """§9 mode B — spaces landing inside words."""

    def test_pairwise_split_rejoins(self) -> None:
        d = OcrDamage(_pad('demati aceous f ung i wi t h fissi on'),
                      vocabulary=_vocab())
        assert d.rejoin_rate() > 0

    def test_three_way_split_rejoins(self) -> None:
        """`Ana mo rph` needs a 3-token window.

        A pairwise implementation scores this 0 — no adjacent PAIR of
        those fragments is a word — which is why the window matters.
        """
        assert OcrDamage(_pad('Ana mo rph: Geomyces'),
                         vocabulary=_vocab()).rejoin_rate() > 0

    def test_six_way_split_rejoins(self) -> None:
        assert OcrDamage(_pad('A n am o r ph: Paecilomyces'),
                         vocabulary=_vocab()).rejoin_rate() > 0

    def test_clean_text_does_not_rejoin(self) -> None:
        d = OcrDamage(_pad('Spores globose, smooth, brown.'),
                      vocabulary=_vocab())
        assert d.rejoin_rate() == 0.0
        assert MODE_SPACING not in d.modes()


class TestSubstitutionMode:
    """§9 mode C — characters swapped inside words."""

    def test_detects_digit_inside_a_word(self) -> None:
        d = OcrDamage(_pad('Oidiode11dron and Pe11icillium'),
                      vocabulary=_vocab())
        assert d.substitution_rate() > 0
        assert MODE_SUBSTITUTION in d.modes()

    def test_detects_interior_capital(self) -> None:
        assert OcrDamage(_pad('RaiUo and KJocker'),
                         vocabulary=_vocab()).substitution_rate() > 0

    def test_figure_references_are_not_damage(self) -> None:
        """`Fig. 2C` and `6A` are legitimate and must not count.

        This false-positive class is why the raw substitution metric
        ranked modern molecular papers above scanned monographs.
        """
        d = OcrDamage(_pad('Conidia as in Fig. 2C and Figs 6A, 3E.'),
                      vocabulary=_vocab())
        assert d.substitution_rate() == 0.0

    def test_micrometre_measurements_are_not_damage(self) -> None:
        """`3um`, `4-7um`, `10x3um` are measurements, not corruption."""
        d = OcrDamage(_pad('Spores 3um to 4-7um wide, 10x3um.'),
                      vocabulary=_vocab())
        assert d.substitution_rate() == 0.0

    def test_accession_numbers_are_not_damage(self) -> None:
        d = OcrDamage(_pad('GenBank KY784257, ITS1 and rpb2 sequences.'),
                      vocabulary=_vocab())
        assert d.substitution_rate() == 0.0


class TestModesAreIndependent:
    """The load-bearing property: one mode must not mask another."""

    def test_substitution_damage_is_invisible_to_rejoin(self) -> None:
        """The taxon_8d815304 shape.

        Severe character substitution with intact spacing.  Recorded
        because it is exactly the case a rejoin-only detector calls
        clean.
        """
        d = OcrDamage(
            _pad('Coremiellu or Oidiode11dron like, RaiUo, KJocker.'),
            vocabulary=_vocab(),
        )
        assert d.rejoin_rate() == 0.0
        assert d.substitution_rate() > 0
        assert d.modes() == (MODE_SUBSTITUTION,)

    def test_spacing_damage_is_invisible_to_substitution(self) -> None:
        d = OcrDamage(_pad('f ung i wi t h fissi on'), vocabulary=_vocab())
        assert d.substitution_rate() == 0.0
        assert MODE_SPACING in d.modes()

    def test_both_modes_can_fire_together(self) -> None:
        d = OcrDamage(_pad('f ung i and Oidiode11dron RaiUo KJocker'),
                      vocabulary=_vocab())
        assert set(d.modes()) >= {MODE_SPACING, MODE_SUBSTITUTION}


class TestProfile:
    """A single object carrying every rate, for reporting."""

    def test_profile_carries_all_rates_and_modes(self) -> None:
        p = OcrDamage(_pad('Ana mo rph and Oidiode11dron'),
                      vocabulary=_vocab()).profile()
        assert p.n_tokens > 40
        assert p.rejoin_rate > 0
        assert p.substitution_rate > 0
        assert MODE_SPACING in p.modes

    def test_profile_of_clean_text_reports_no_modes(self) -> None:
        p = OcrDamage(_pad('Spores globose, smooth.'),
                      vocabulary=_vocab()).profile()
        assert p.modes == ()

    def test_oov_rate_is_reported_but_not_a_mode(self) -> None:
        """OOV is contaminated by proper nouns, so it never fires a mode.

        Nomenclature-dense text scores 31.6 % legitimately -- genus and
        author names are simply absent from any dictionary.
        """
        d = OcrDamage(_pad('Cephalotheca Pseudogymnoascus Eremascus'),
                      vocabulary=_vocab())
        assert d.oov_rate() > 0
        assert d.modes() == ()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
