#!/usr/bin/env python3
"""Tests for ``treatments_to_structured.display_labels``.

``brat_safe_type`` replaces every character outside ``[A-Za-z0-9_-]``
with an underscore, because brat's storage regex demands it.  That is
right for brat and lossy for the record: ``Schäffer's reaction`` is
stored as ``Sch ffer s reaction``, losing a diacritic and an
apostrophe from a person's name, and ``conidium length/width ratio``
becomes ``Conidium length width ratio``, losing the relation the label
was describing.

**The span is the witness.**  ``source_text`` is stored verbatim, so
the characters brat removed are usually still there.  Recovery matches
the label's words against its own span allowing any non-alphanumeric
run between them, which is evidence rather than inference — the
alternative considered and rejected was a "capitalised word + naked s"
heuristic, measured at **0 correct out of 7** on the span corpus
(2.24 M characters): six were OCR word-splitting (`The s pecies`,
`Hyphae s traight`) and one was `sensu stricto`.  It is also blind to
owners whose name already ends in s — ``Gams' terminology`` sanitizes
with no naked ``s`` at all.

**Case comes from the label, punctuation from the span.**  A label's
capitalisation is the annotator's choice of heading; the span's is an
accident of sentence position.  So recovery splices the label's words
with the span's separators.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treatments_to_structured.display_labels import (  # noqa: E402
    recover_display_label,
)


class TestRecoverDisplayLabel:
    @pytest.mark.parametrize('label,span,expected', [
        ('Sch ffer s reaction',
         'Schäffer’s reaction: positive, reddish orange on dry specimen.',
         'Schäffer’s reaction'),
        ('Conidium length width ratio',
         'the conidium length/width ratio was 3.2',
         'Conidium length/width ratio'),
        ('Spore print', 'Spore-print white to cream.', 'Spore-print'),
        ('Velum Cortina', 'Velum/ Cortina absent.', 'Velum/ Cortina'),
        ('Surface cells', 'surface, cells thin-walled', 'Surface, cells'),
    ])
    def test_punctuation_comes_back(
            self, label: str, span: str, expected: str) -> None:
        assert recover_display_label(label, span) == expected

    def test_case_stays_the_label_s(self) -> None:
        """The span says `conidium`; the label says `Conidium`.  A
        label is a heading and its capitalisation is deliberate, while
        the span's is wherever the sentence put it."""
        got = recover_display_label(
            'Conidium length width ratio',
            'the conidium length/width ratio was 3.2')
        assert got is not None and got.startswith('Conidium')

    def test_a_diacritic_inside_a_word_is_restored(self) -> None:
        """`Sch` + `ä` + `ffer` -- the loss is inside a word, not
        between two of them, so the separator has to be allowed to fall
        anywhere the label has a gap."""
        assert recover_display_label(
            'Sch ffer s reaction', 'Schäffer’s reaction: positive.'
        ) == 'Schäffer’s reaction'

    @pytest.mark.parametrize('label,span', [
        ('Aerial phialides', 'aerial phialides ampulliform'),
        ('Colony reverse', 'Colony reverse ochraceous'),
        ('Pileus trama', 'pileus  trama composed of hyphae'),
    ])
    def test_case_and_whitespace_alone_are_not_a_recovery(
            self, label: str, span: str) -> None:
        """109 labels differ from their span only in case or spacing.
        Those are ``fold_case``'s business, and storing them here would
        put a redundant display form on a seventh of the corpus."""
        assert recover_display_label(label, span) is None

    def test_a_label_absent_from_its_span_recovers_nothing(self) -> None:
        """The annotator's label is often a name for the passage rather
        than a quotation from it."""
        assert recover_display_label(
            'Basidiospores', 'Colonies on MEA reaching 40 mm.') is None

    def test_an_ocr_damaged_span_is_left_alone(self) -> None:
        """`Sabouraud ' s` is what the page says.  The label reports it
        faithfully, and recovery must not invent the apostrophe that
        OCR already destroyed -- that is D8's problem, not this one."""
        got = recover_display_label(
            'Colony Sabouraud s dextrose medium',
            "COLONIES on Sabouraud ' s dextrose medium (Difco)")
        assert got is None or "'" in got

    @pytest.mark.parametrize('label,span', [
        ('', 'anything'),
        ('Pileus', ''),
        ('Pileus', 'Pileus 3 cm diam'),
    ])
    def test_degenerate_inputs_return_none(
            self, label: str, span: str) -> None:
        assert recover_display_label(label, span) is None

    def test_the_separator_run_is_bounded(self) -> None:
        """Without a bound, `Spore print` would match `Spore` and a
        `print` forty characters later and splice nonsense."""
        assert recover_display_label(
            'Spore print',
            'Spore deposit white, and the print taken overnight') is None


class TestRecoveryIsIdempotent:
    def test_an_already_rich_label_recovers_nothing(self) -> None:
        """Running the backfill twice must not churn."""
        assert recover_display_label(
            'Spore-print', 'Spore-print white to cream.') is None
