#!/usr/bin/env python3
"""Tests for ``treatments_to_structured.round_provenance``.

These pin the five design decisions in T0e's stamping work.  Each is
recorded as a test rather than a comment because each is a place where
a plausible alternative would quietly corrupt the provenance record
this module exists to create.

Every xfail here is ``strict``: the head must fail on
``NotImplementedError`` until the implementation lands, so a bisect
never sees a green test for absent code.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treatments_to_structured.round_provenance import (  # noqa: E402
    PROVENANCE_MANUAL,
    PROVENANCE_RECONSTRUCTED,
    PROVENANCE_SELECTOR,
    RoundIdentity,
    RoundProvenanceError,
    read_round_file,
    round_identity,
    stamp_round,
)

pytestmark = pytest.mark.xfail(
    raises=NotImplementedError, strict=True,
    reason='T0e stamping: implementation follows test confirmation',
)


def _round_file(tmp_path: Path, name: str, ids: int = 2) -> Path:
    p = tmp_path / name
    p.write_text(
        ''.join(f'taxon_{i:064x}\n' for i in range(ids)),
        encoding='utf-8',
    )
    return p


def _sidecar(path: Path, **fields: object) -> Path:
    meta = path.with_suffix('.meta.json')
    meta.write_text(json.dumps(fields), encoding='utf-8')
    return meta


# ---------------------------------------------------------------------------
# Decision 1 — the file NAME is authoritative for the round number
# ---------------------------------------------------------------------------


class TestRoundNumberComesFromTheName:
    """``default_output_path`` never reuses a number, so the name is
    the one identifier that cannot drift.  A sidecar can be copied,
    hand-edited, or absent; the name cannot be any of those without
    also changing which file you opened.
    """

    def test_numbered_round_file(self, tmp_path: Path) -> None:
        p = _round_file(tmp_path, 'production_v4_round6.txt')
        assert round_identity(p).round == 6

    def test_experiment_is_parsed_from_the_name(
        self, tmp_path: Path,
    ) -> None:
        p = _round_file(tmp_path, 'production_v4_round6.txt')
        assert round_identity(p).experiment == 'production_v4'

    def test_round_file_stem_is_always_recorded(
        self, tmp_path: Path,
    ) -> None:
        """The stem disambiguates what the number alone cannot.

        ``production_v4_round5`` and ``production_v4_round5_manual``
        share a round number by design, so the number is not a key.
        """
        p = _round_file(tmp_path, 'production_v4_round6.txt')
        assert round_identity(p).round_file == 'production_v4_round6'

    def test_a_name_without_a_round_number_is_an_error(
        self, tmp_path: Path,
    ) -> None:
        """Refuse rather than invent.

        Stamping an unidentifiable file with a guessed number is
        exactly the class of silent error this work removes.
        """
        p = _round_file(tmp_path, 'some_treatments.txt')
        with pytest.raises(RoundProvenanceError):
            round_identity(p)


# ---------------------------------------------------------------------------
# Decision 2 — a `_manual` file belongs to its round, and says so
# ---------------------------------------------------------------------------


class TestManualFiles:
    """``production_v4_round5_manual.txt`` holds treatments that must
    be *included in round 5* (data/annotation_rounds/README.md).  It
    is hand-picked material added to a round, not a round of its own.

    Two regexes are involved and they are deliberately different:
    ``select_for_annotation.default_output_path`` must NOT match
    ``_manual`` — that is what keeps the selector numbering normally —
    while this module MUST match it, or the one treatment in round 5's
    manual file goes unstamped.
    """

    def test_manual_file_carries_its_round_number(
        self, tmp_path: Path,
    ) -> None:
        p = _round_file(tmp_path, 'production_v4_round5_manual.txt')
        assert round_identity(p).round == 5

    def test_manual_file_is_distinguishable_by_its_stem(
        self, tmp_path: Path,
    ) -> None:
        p = _round_file(tmp_path, 'production_v4_round5_manual.txt')
        ident = round_identity(p)
        assert ident.round_file == 'production_v4_round5_manual'

    def test_manual_provenance_is_inferred_from_the_name(
        self, tmp_path: Path,
    ) -> None:
        """No sidecar needed — the suffix is the evidence.

        This matters because ``_manual`` files are written by hand and
        will never have a selector-produced sidecar.
        """
        p = _round_file(tmp_path, 'production_v4_round5_manual.txt')
        assert round_identity(p).provenance == PROVENANCE_MANUAL


# ---------------------------------------------------------------------------
# Decision 3 — a sidecar enriches; a disagreeing sidecar is an error
# ---------------------------------------------------------------------------


class TestSidecar:
    def test_absent_sidecar_is_not_an_error(
        self, tmp_path: Path,
    ) -> None:
        """Rounds 1-4 have no sidecar and must still be stampable.

        Refusing here would make the historical rounds permanently
        unstampable, which is the opposite of the goal.
        """
        p = _round_file(tmp_path, 'production_v4_round4.txt')
        ident = round_identity(p)
        assert ident.round == 4
        assert ident.provenance is None

    def test_sidecar_supplies_provenance_and_selection(
        self, tmp_path: Path,
    ) -> None:
        p = _round_file(tmp_path, 'production_v4_round6.txt')
        _sidecar(p, round=6, provenance=PROVENANCE_SELECTOR,
                 selection='uniform')
        ident = round_identity(p)
        assert ident.provenance == PROVENANCE_SELECTOR
        assert ident.selection == 'uniform'

    def test_reconstructed_sidecar_is_marked_as_such(
        self, tmp_path: Path,
    ) -> None:
        """Backfilled sidecars for rounds 1-4 must not pass as
        first-hand records — their selector invocations are gone.
        """
        p = _round_file(tmp_path, 'production_v4_round2.txt')
        _sidecar(p, round=2, provenance=PROVENANCE_RECONSTRUCTED)
        assert round_identity(p).provenance == PROVENANCE_RECONSTRUCTED

    def test_sidecar_disagreeing_with_the_name_is_an_error(
        self, tmp_path: Path,
    ) -> None:
        """The failure mode is a copied sidecar.

        Silently preferring either source would corrupt precisely the
        provenance record being created, and would do it invisibly.
        """
        p = _round_file(tmp_path, 'production_v4_round6.txt')
        _sidecar(p, round=7, provenance=PROVENANCE_SELECTOR)
        with pytest.raises(RoundProvenanceError, match='7'):
            round_identity(p)

    def test_sidecar_without_a_round_field_still_enriches(
        self, tmp_path: Path,
    ) -> None:
        """Absence is not disagreement.

        ``build_round_metadata`` does not currently emit ``round``, so
        any sidecar written before this change lacks it.
        """
        p = _round_file(tmp_path, 'production_v4_round6.txt')
        _sidecar(p, provenance=PROVENANCE_SELECTOR, selection='uniform')
        ident = round_identity(p)
        assert ident.round == 6
        assert ident.provenance == PROVENANCE_SELECTOR

    def test_malformed_sidecar_is_an_error_not_a_shrug(
        self, tmp_path: Path,
    ) -> None:
        """Do not fall back to the bare name on unreadable JSON.

        A truncated sidecar means something went wrong upstream, and
        continuing would stamp docs with an identity nobody checked.
        """
        p = _round_file(tmp_path, 'production_v4_round6.txt')
        p.with_suffix('.meta.json').write_text('{not json',
                                               encoding='utf-8')
        with pytest.raises(RoundProvenanceError):
            round_identity(p)


# ---------------------------------------------------------------------------
# read_round_file — ids and identity together
# ---------------------------------------------------------------------------


class TestReadRoundFile:
    def test_returns_ids_and_identity(self, tmp_path: Path) -> None:
        p = _round_file(tmp_path, 'production_v4_round6.txt', ids=3)
        ids, ident = read_round_file(p)
        assert len(ids) == 3
        assert ident.round == 6

    def test_blank_lines_are_skipped_and_ids_stripped(
        self, tmp_path: Path,
    ) -> None:
        """Matches ``read_treatment_ids`` so the two input paths agree."""
        p = tmp_path / 'production_v4_round6.txt'
        p.write_text('  taxon_a  \n\n\ntaxon_b\n\n', encoding='utf-8')
        ids, _ = read_round_file(p)
        assert ids == ['taxon_a', 'taxon_b']

    def test_missing_file_raises_file_not_found(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            read_round_file(tmp_path / 'production_v4_round9.txt')

    def test_empty_round_file_is_an_error(
        self, tmp_path: Path,
    ) -> None:
        """An empty round file means the draw failed silently.

        Returning ``[]`` would let the annotator report "No treatments
        to process" and exit 0, which reads as success.
        """
        p = tmp_path / 'production_v4_round6.txt'
        p.write_text('\n\n', encoding='utf-8')
        with pytest.raises(RoundProvenanceError):
            read_round_file(p)


# ---------------------------------------------------------------------------
# Decision 4 — the round goes on the doc, never into the doc `_id`
# Decision 5 — a re-run in a later round overwrites, last write wins
# ---------------------------------------------------------------------------


class TestStampRound:
    def test_round_fields_land_on_the_doc(self) -> None:
        ident = RoundIdentity(
            round=6, round_file='production_v4_round6',
            experiment='production_v4',
            provenance=PROVENANCE_SELECTOR,
        )
        doc = stamp_round({'feature_label': 'Pileus'}, ident)
        assert doc['round'] == 6
        assert doc['round_file'] == 'production_v4_round6'
        assert doc['round_provenance'] == PROVENANCE_SELECTOR

    def test_none_identity_leaves_the_doc_untouched(self) -> None:
        """The cron path passes no round file.

        ``round: null`` would make every cron-produced annotation look
        like it came from an unidentifiable round rather than from no
        round.  Absent means absent.
        """
        doc = stamp_round({'feature_label': 'Pileus'}, None)
        assert doc == {'feature_label': 'Pileus'}

    def test_the_doc_id_is_never_touched(self) -> None:
        """``annotation_doc_id`` is ``<tid>:<label>:<start>`` and must
        stay that way.

        Putting the round in the key would make a re-run create a
        SECOND doc at the same offset instead of replacing the first,
        so a re-measured round would yield the union of two prompts'
        vocabularies — the trap recorded in the T6 plan.
        """
        ident = RoundIdentity(
            round=6, round_file='production_v4_round6',
            experiment='production_v4',
        )
        doc = stamp_round(
            {'_id': 'taxon_a:Pileus:13', 'feature_label': 'Pileus'},
            ident,
        )
        assert doc['_id'] == 'taxon_a:Pileus:13'

    def test_a_later_round_overwrites_an_earlier_one(self) -> None:
        """Last write wins, deliberately.

        A treatment re-annotated in a later round is *now* evidence
        about that round's prompt; the earlier annotation was already
        replaced at the same ``_id``.  Keeping a list would imply the
        old annotation still exists, which it does not.
        """
        old = RoundIdentity(round=4, round_file='production_v4_round4',
                            experiment='production_v4')
        new = RoundIdentity(round=8, round_file='production_v4_round8',
                            experiment='production_v4')
        doc = stamp_round({'feature_label': 'Pileus'}, old)
        doc = stamp_round(doc, new)
        assert doc['round'] == 8
        assert doc['round_file'] == 'production_v4_round8'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
