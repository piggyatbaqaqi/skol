#!/usr/bin/env python3
"""Tests for ``fixes/backfill_round_stamps``.

Two decisions get pinned here, both about a backfill's freedom to be
wrong in ways a live write cannot: it reconstructs history rather than
recording it, so every inference it makes needs a stated reason.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_round_stamps import (  # type: ignore[import]  # noqa: E402
    assign_rounds,
    recover_bands,
    reconstructed_sidecar,
    stamp_docs,
)
from treatments_to_structured.round_provenance import (  # noqa: E402
    PROVENANCE_MANUAL,
    PROVENANCE_RECONSTRUCTED,
)

pytestmark = pytest.mark.xfail(
    raises=NotImplementedError, strict=True,
    reason='backfill: implementation follows test confirmation',
)


# ---------------------------------------------------------------------------
# Decision 6 — in a backfill the LOWEST round wins
# ---------------------------------------------------------------------------


class TestAssignRounds:
    """The live path stamps the round that wrote the doc; a backfill
    has to work out which round that was.

    Those give opposite answers on a treatment listed twice, and the
    backfill's answer is the correct one *here* because nothing was
    ever re-annotated — ``filter_already_annotated`` skips
    ``status='success'``, and all 117 attempted status docs in
    production_v4 read ``attempt_count: 1``.
    """

    def test_single_round_membership(self) -> None:
        got = assign_rounds({'production_v4_round3': ['taxon_a']})
        assert got['taxon_a'].round == 3
        assert got['taxon_a'].round_file == 'production_v4_round3'

    def test_a_treatment_in_two_rounds_takes_the_lower(self) -> None:
        """The real case: taxon_2b793602 is in rounds 1 and 2.

        Round 2's run skipped it because round 1 had already succeeded,
        so round 1's prompt produced the annotations that exist.
        Taking round 2 would attribute 136 of the corpus's 263 hand
        additions to the wrong round.
        """
        got = assign_rounds({
            'production_v4_round2': ['taxon_dup', 'taxon_b'],
            'production_v4_round1': ['taxon_dup'],
        })
        assert got['taxon_dup'].round == 1
        assert got['taxon_b'].round == 2

    def test_order_of_the_input_mapping_does_not_matter(self) -> None:
        """Dict iteration order must not decide provenance."""
        low_first = assign_rounds({
            'production_v4_round1': ['taxon_dup'],
            'production_v4_round2': ['taxon_dup'],
        })
        high_first = assign_rounds({
            'production_v4_round2': ['taxon_dup'],
            'production_v4_round1': ['taxon_dup'],
        })
        assert low_first['taxon_dup'].round == 1
        assert high_first['taxon_dup'].round == 1

    def test_manual_membership_is_marked_manual(self) -> None:
        got = assign_rounds({
            'production_v4_round5_manual': ['taxon_canary'],
        })
        assert got['taxon_canary'].round == 5
        assert got['taxon_canary'].provenance == PROVENANCE_MANUAL

    def test_a_manual_file_does_not_outrank_its_own_round(self) -> None:
        """``round5`` and ``round5_manual`` share a number.

        The tie must resolve on the file, not on the number, or a
        hand-picked addition would masquerade as part of the draw.
        """
        got = assign_rounds({
            'production_v4_round5': ['taxon_x'],
            'production_v4_round5_manual': ['taxon_y'],
        })
        assert got['taxon_x'].round_file == 'production_v4_round5'
        assert got['taxon_y'].round_file == 'production_v4_round5_manual'


# ---------------------------------------------------------------------------
# Decision 7 — record only what is recoverable
# ---------------------------------------------------------------------------


class TestReconstructedSidecar:
    def test_marked_reconstructed(self, tmp_path: Path) -> None:
        p = tmp_path / 'production_v4_round2.txt'
        p.write_text('taxon_a\n', encoding='utf-8')
        meta = reconstructed_sidecar(p, ['taxon_a'])
        assert meta['provenance'] == PROVENANCE_RECONSTRUCTED

    def test_round_and_count_are_recorded(self, tmp_path: Path) -> None:
        p = tmp_path / 'production_v4_round2.txt'
        p.write_text('taxon_a\ntaxon_b\n', encoding='utf-8')
        meta = reconstructed_sidecar(p, ['taxon_a', 'taxon_b'])
        assert meta['round'] == 2
        assert meta['n_selected'] == 2
        assert meta['experiment'] == 'production_v4'

    def test_unknowable_fields_are_absent_not_null(
        self, tmp_path: Path,
    ) -> None:
        """Rounds 1-4's seeds and selector invocations are gone.

        ``"seed": null`` documents an irretrievable gap as though it
        were a recorded fact, and a reader cannot tell it from a run
        that genuinely had no seed.  Absent is the honest form.
        """
        p = tmp_path / 'production_v4_round2.txt'
        p.write_text('taxon_a\n', encoding='utf-8')
        meta = reconstructed_sidecar(p, ['taxon_a'])
        for absent in ('seed', 'selector_argv', 'population_funnel',
                       'band_slices', 'selection'):
            assert absent not in meta

    def test_is_json_serialisable(self, tmp_path: Path) -> None:
        p = tmp_path / 'production_v4_round2.txt'
        p.write_text('taxon_a\n', encoding='utf-8')
        json.dumps(reconstructed_sidecar(p, ['taxon_a']))


# ---------------------------------------------------------------------------
# stamp_docs — and the 7 636 documents it must not touch
# ---------------------------------------------------------------------------


class _FakeRow:
    def __init__(self, _id: str, doc: dict) -> None:
        self.id, self.doc = _id, doc


class _FakeDb:
    def __init__(self, docs: dict) -> None:
        self.docs = docs
        self.saved: list = []

    def view(self, _name: str, **_kw: object) -> object:
        rows = [_FakeRow(k, v) for k, v in self.docs.items()]

        class _V:
            def __init__(self, r: list) -> None:
                self.rows = r

            def __iter__(self) -> object:
                return iter(self.rows)
        return _V(rows)

    def save(self, doc: dict) -> None:
        self.saved.append(dict(doc))


class TestStampDocs:
    @staticmethod
    def _assignments():
        return assign_rounds({'production_v4_round3': ['taxon_a']})

    def test_candidate_docs_are_stamped_by_id_prefix(self) -> None:
        db = _FakeDb({
            'taxon_a:Pileus:13': {'feature_label': 'Pileus'},
            'taxon_a:Stipe:99': {'feature_label': 'Stipe'},
        })
        stamped, _ = stamp_docs(
            db, self._assignments(),
            id_to_treatment=lambda i: i.split(':', 1)[0],
            dry_run=False,
        )
        assert stamped == 2
        assert all(d['round'] == 3 for d in db.saved)

    def test_docs_outside_every_round_file_are_left_alone(self) -> None:
        """The four ad-hoc status docs, and the reason --round-file is
        optional: no round file means no round, and a backfill must not
        invent one either.
        """
        db = _FakeDb({
            'taxon_a:Pileus:13': {'feature_label': 'Pileus'},
            'taxon_orphan:Pileus:0': {'feature_label': 'Pileus'},
        })
        stamped, skipped = stamp_docs(
            db, self._assignments(),
            id_to_treatment=lambda i: i.split(':', 1)[0],
            dry_run=False,
        )
        assert (stamped, skipped) == (1, 1)
        assert [d['_id'] for d in db.saved] == ['taxon_a:Pileus:13']

    def test_skipped_merge_suspects_are_not_stamped(self) -> None:
        """7 636 of the 7 749 status docs, and none of them belongs to
        a round: they were written by the selector to record a
        population decision, before any round annotated anything.
        """
        db = _FakeDb({
            'taxon_a': {'status': 'success'},
            'taxon_sus': {'status': 'skipped_merge_suspect'},
        })
        assignments = assign_rounds({
            'production_v4_round3': ['taxon_a', 'taxon_sus'],
        })
        stamped, _ = stamp_docs(
            db, assignments, id_to_treatment=lambda i: i,
            dry_run=False,
            statuses=frozenset({'success', 'partial', 'error'}),
        )
        assert stamped == 1
        assert [d['_id'] for d in db.saved] == ['taxon_a']

    def test_dry_run_writes_nothing(self) -> None:
        """A backfill touching 1 700 documents gets rehearsed first."""
        db = _FakeDb({'taxon_a:Pileus:13': {'feature_label': 'Pileus'}})
        stamped, _ = stamp_docs(
            db, self._assignments(),
            id_to_treatment=lambda i: i.split(':', 1)[0],
            dry_run=True,
        )
        assert stamped == 1
        assert db.saved == []

    def test_design_docs_are_ignored(self) -> None:
        db = _FakeDb({
            '_design/views': {},
            'taxon_a:Pileus:13': {'feature_label': 'Pileus'},
        })
        stamped, _ = stamp_docs(
            db, self._assignments(),
            id_to_treatment=lambda i: i.split(':', 1)[0],
            dry_run=False,
        )
        assert stamped == 1

    def test_rerunning_the_backfill_is_idempotent(self) -> None:
        """Already-correct docs are not rewritten.

        A second pass that re-saves every doc would burn 1 700
        revisions to change nothing, and would mask a genuine
        disagreement behind an overwrite.
        """
        db = _FakeDb({
            'taxon_a:Pileus:13': {
                'feature_label': 'Pileus', 'round': 3,
                'round_file': 'production_v4_round3',
            },
        })
        stamped, skipped = stamp_docs(
            db, self._assignments(),
            id_to_treatment=lambda i: i.split(':', 1)[0],
            dry_run=False,
        )
        assert (stamped, skipped) == (0, 1)
        assert db.saved == []


# ---------------------------------------------------------------------------
# Decision 8 — knowable-by-derivation is a third category
#
# Decision 7 split fields into recorded and absent.  Round 4's bands
# are neither: they were never logged, but they ARE recoverable, so
# they are recorded together with how they were derived.
# ---------------------------------------------------------------------------


class TestRecoverBands:
    """`select_treatments` emits band-by-band, so a banded round's file
    is band-monotonic in score: every band-0 member precedes every
    band-1 member, and so on.

    That order is the whole signal.  A weaker test -- "do the
    population cut points fall in gaps between sorted sample scores" --
    is **vacuous**, and round 3 proves it: a known-random round passes
    it for k=2, 3 AND 4, because with n sample points every value
    between two adjacent samples lies in some gap.
    """

    @staticmethod
    def _cuts(k):
        # Stand-in population: 300 evenly spaced scores, so the k-band
        # equal-slice cut points are exact and easy to reason about.
        pop = list(range(1, 301))
        return [pop[(i * len(pop)) // k] for i in range(1, k)]

    def test_band_monotonic_file_recovers_k_and_quotas(self) -> None:
        """The round-4 shape: 2 low, 3 mid, 4 high, in that order."""
        ids = [f'taxon_{i}' for i in range(9)]
        scores = [10, 50, 120, 150, 190, 220, 260, 280, 290]
        got = recover_bands(ids, scores, self._cuts)
        assert got is not None
        assert got['k'] == 3
        assert got['band_quotas'] == [2, 3, 4]

    def test_shuffled_order_recovers_nothing(self) -> None:
        """Rounds 1-3 are sorted by treatment id, which destroys the
        band order -- and with it any chance of recovery.  Returning a
        best guess here would put a fabricated draw design into the
        provenance record.
        """
        ids = [f'taxon_{i}' for i in range(9)]
        scores = [290, 50, 220, 10, 260, 150, 120, 280, 190]
        assert recover_bands(ids, scores, self._cuts) is None

    def test_ambiguity_recovers_nothing(self) -> None:
        """If more than one k fits, the answer is unknown, not the
        smallest k.  Round 4 is usable precisely because k=3 is the
        ONLY value that fits; k=2, 4 and 5 are all rejected.
        """
        ids = ['taxon_a', 'taxon_b']
        scores = [10, 290]
        assert recover_bands(ids, scores, self._cuts) is None

    def test_a_single_band_is_not_a_recovery(self) -> None:
        """An unbanded round is trivially monotonic in one band.  That
        is the absence of banding, not evidence of it.
        """
        ids = [f'taxon_{i}' for i in range(4)]
        scores = [10, 20, 30, 40]
        assert recover_bands(ids, scores, self._cuts) is None

    def test_observed_ranges_are_reported_beside_the_cut_points(
        self,
    ) -> None:
        """Both matter and they are different facts: the cut point is
        where the population was divided, the observed range is what
        the draw actually landed on.  Round 4's band 1 is [27, 133]
        against a cut at 146 -- the 13-point margin is what makes the
        recovery checkable.
        """
        ids = [f'taxon_{i}' for i in range(9)]
        scores = [10, 50, 120, 150, 190, 220, 260, 280, 290]
        got = recover_bands(ids, scores, self._cuts)
        first = got['band_slices'][0]
        assert first['observed_min'] == 10
        assert first['observed_max'] == 120
        assert first['cut_max'] == self._cuts(3)[0]


class TestDerivedSidecarFields:
    def test_derivation_note_accompanies_recovered_bands(
        self, tmp_path: Path,
    ) -> None:
        """A derived field that does not say it was derived is
        indistinguishable from a logged one, which is the whole
        distinction decision 8 exists to keep.
        """
        p = tmp_path / 'production_v4_round4.txt'
        p.write_text('taxon_a\n', encoding='utf-8')
        bands = {'k': 3, 'band_quotas': [5, 15, 30], 'band_slices': []}
        meta = reconstructed_sidecar(p, ['taxon_a'], bands=bands)
        assert meta['band_quotas'] == [5, 15, 30]
        assert meta['output_order'] == 'band-by-band'
        assert 'bands_derivation' in meta

    def test_band_names_are_never_invented(
        self, tmp_path: Path,
    ) -> None:
        """Band NAMES carry no meaning -- low/mid/high are decoration
        and the count sets the cut points.  Only quotas are
        recoverable, so a `bands` list of [name, quota] pairs must not
        appear in a reconstructed sidecar.
        """
        p = tmp_path / 'production_v4_round4.txt'
        p.write_text('taxon_a\n', encoding='utf-8')
        bands = {'k': 3, 'band_quotas': [5, 15, 30], 'band_slices': []}
        meta = reconstructed_sidecar(p, ['taxon_a'], bands=bands)
        assert 'bands' not in meta

    def test_no_bands_means_no_band_fields(
        self, tmp_path: Path,
    ) -> None:
        """Rounds 1-3: absent, per decision 7."""
        p = tmp_path / 'production_v4_round2.txt'
        p.write_text('taxon_a\n', encoding='utf-8')
        meta = reconstructed_sidecar(p, ['taxon_a'], bands=None)
        for absent in ('band_quotas', 'band_slices', 'bands_derivation',
                       'output_order'):
            assert absent not in meta


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
