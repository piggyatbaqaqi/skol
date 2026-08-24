"""Tests for bin/select_for_annotation helpers.

The pure ``parse_band_spec`` / ``select_treatments`` logic is
tested in ``treatments_to_structured/select_test.py``.  This file
covers the CLI-side glue: ``_resolve_band_specs`` (CLI-flag
fan-in) and ``score_treatments_in_db`` (CouchDB iteration via a
fake DB stand-in).
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import select_for_annotation  # type: ignore[import]  # noqa: E402
from select_for_annotation import (  # type: ignore[import]  # noqa: E402
    build_round_metadata,
    resolve_seed,
    _resolve_band_specs,
    apply_merge_filter,
    fetch_annotated_treatment_ids,
    fetch_prior_merge_skip_ids,
    filter_excluded,
    score_treatments_in_db,
)
from treatments_to_structured.status import (  # noqa: E402
    STATUS_ERROR,
    STATUS_SKIPPED_MERGE_SUSPECT,
    STATUS_SUCCESS,
)


class _FakeTreatmentsDb:
    """Minimal stand-in for the couchdb.Database iteration / get.

    Iteration yields doc_ids in insertion order; ``db[doc_id]`` looks
    up the doc.  Supports a ``raise_on`` set of IDs that should raise
    on read — used to verify the script's "skip transient failures"
    behaviour.
    """

    def __init__(
        self,
        docs: Dict[str, Dict[str, Any]],
        raise_on: Iterator[str] = (),
    ) -> None:
        self.docs = docs
        self._raise_on = set(raise_on)

    def __iter__(self) -> Iterator[str]:
        return iter(self.docs)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        if key in self._raise_on:
            raise RuntimeError(f"simulated read failure for {key}")
        return self.docs[key]


# ---------------------------------------------------------------------------
# _resolve_band_specs
# ---------------------------------------------------------------------------


class TestResolveBandSpecs:
    """Default → single 'all' band; explicit spec → parsed, validated."""

    def test_no_bands_flag_defaults_to_single_all_band(self) -> None:
        assert _resolve_band_specs(None, 100) == [('all', 100)]
        assert _resolve_band_specs('', 100) == [('all', 100)]

    def test_explicit_bands_parsed(self) -> None:
        assert _resolve_band_specs('low:25,mid:50,high:25', 100) == [
            ('low', 25), ('mid', 50), ('high', 25),
        ]

    def test_quota_mismatch_raises(self) -> None:
        """If --bands sums to 60 but --n is 100, fail loudly."""
        with pytest.raises(ValueError) as exc:
            _resolve_band_specs('low:25,mid:35', 100)
        msg = str(exc.value)
        assert '60' in msg
        assert '100' in msg

    def test_malformed_bands_raises(self) -> None:
        """Errors from parse_band_spec bubble up unchanged."""
        with pytest.raises(ValueError):
            _resolve_band_specs('not a band spec', 100)


# ---------------------------------------------------------------------------
# score_treatments_in_db
# ---------------------------------------------------------------------------


class TestScoreTreatmentsInDb:
    """Iterate, score, filter — without a real CouchDB."""

    def test_skips_design_docs(self) -> None:
        """``_design/*`` docs aren't user-facing data; ignored."""
        db = _FakeTreatmentsDb({
            '_design/views': {'fake': 'design doc'},
            'taxon_a': {'description': 'Pileus brown 3 cm.'},
        })
        scored, _merge_metrics = score_treatments_in_db(db, verbosity=0)
        assert {doc_id for doc_id, _ in scored} == {'taxon_a'}

    def test_filters_zero_score_treatments(self) -> None:
        """Treatments with no description AND no diagnosis can't be
        annotated; drop them from the candidate pool."""
        db = _FakeTreatmentsDb({
            'taxon_empty': {
                'description': None, 'diagnosis': None,
            },
            'taxon_populated': {
                'description': 'Pileus brown 3 cm.',
            },
        })
        scored, _merge_metrics = score_treatments_in_db(db, verbosity=0)
        ids = {doc_id for doc_id, _ in scored}
        assert ids == {'taxon_populated'}

    def test_returns_id_score_pairs(self) -> None:
        """Each entry is (treatment_id, complexity_score)."""
        db = _FakeTreatmentsDb({
            'taxon_a': {'description': 'Pileus brown 3 cm.'},
            'taxon_b': {'description': 'Stipe 5 mm long.'},
        })
        scored, _merge_metrics = score_treatments_in_db(db, verbosity=0)
        assert len(scored) == 2
        for doc_id, score in scored:
            assert doc_id.startswith('taxon_')
            assert isinstance(score, float)
            assert score > 0

    def test_skips_docs_that_raise_on_read(self) -> None:
        """Transient CouchDB errors during iteration are tolerated —
        mirrors build_sources_stats.  The doc is silently skipped."""
        db = _FakeTreatmentsDb(
            {
                'taxon_ok': {'description': 'Pileus brown 3 cm.'},
                'taxon_broken': {'description': 'Stipe 5 mm long.'},
            },
            raise_on={'taxon_broken'},
        )
        scored, _merge_metrics = score_treatments_in_db(db, verbosity=0)
        ids = {doc_id for doc_id, _ in scored}
        assert ids == {'taxon_ok'}

    def test_empty_db_returns_empty_list(self) -> None:
        scored, _merge_metrics = score_treatments_in_db(_FakeTreatmentsDb({}), verbosity=0)
        assert scored == []


# ---------------------------------------------------------------------------
# fetch_annotated_treatment_ids — read distinct treatment_ids out of
# the candidate DB so the --exclude-annotated filter can skip them.
# ---------------------------------------------------------------------------


class _FakeRow:
    """Mimics couchdb-python's view row shape: has an ``.id``."""
    def __init__(self, _id: str) -> None:
        self.id = _id


class _FakeView:
    def __init__(self, rows: Any) -> None:
        self.rows = list(rows)


class _FakeCandidateDb:
    """Stand-in for the candidate annotations DB, supporting the
    ``view('_all_docs')`` call shape used by
    fetch_annotated_treatment_ids."""

    def __init__(self, ids: Any) -> None:
        self._ids = list(ids)

    def view(self, _name: str, **_kwargs: Any) -> _FakeView:
        return _FakeView(_FakeRow(i) for i in self._ids)


class TestFetchAnnotatedTreatmentIds:
    def test_distinct_treatment_ids_extracted(self) -> None:
        """Annotation _ids are ``<tid>:<label>:<offset>`` — the
        prefix before the first ``:`` is the treatment id."""
        db = _FakeCandidateDb([
            'taxon_a:Pileus:48',
            'taxon_a:Stipe:200',
            'taxon_b:Pileus:0',
        ])
        ids = fetch_annotated_treatment_ids(db)
        assert ids == {'taxon_a', 'taxon_b'}

    def test_skips_design_docs(self) -> None:
        """``_design/...`` docs (couchdb views) must not pollute the
        exclusion set."""
        db = _FakeCandidateDb([
            'taxon_a:Pileus:0',
            '_design/some_view',
        ])
        ids = fetch_annotated_treatment_ids(db)
        assert ids == {'taxon_a'}

    def test_skips_keys_without_colon(self) -> None:
        """Defensive: any malformed _id without a ``:`` separator is
        skipped rather than treated as a treatment id."""
        db = _FakeCandidateDb(['malformed_key', 'taxon_a:Pileus:0'])
        ids = fetch_annotated_treatment_ids(db)
        assert ids == {'taxon_a'}

    def test_empty_db_returns_empty_set(self) -> None:
        assert fetch_annotated_treatment_ids(_FakeCandidateDb([])) == set()


# ---------------------------------------------------------------------------
# filter_excluded — drop scored entries by treatment_id
# ---------------------------------------------------------------------------


class TestFilterExcluded:
    def test_drops_excluded_ids(self) -> None:
        scored = [
            ('taxon_a', 0.5),
            ('taxon_b', 0.7),
            ('taxon_c', 0.9),
        ]
        result = filter_excluded(scored, {'taxon_b'})
        assert result == [('taxon_a', 0.5), ('taxon_c', 0.9)]

    def test_preserves_input_order(self) -> None:
        """Banding downstream relies on the score-sorted order
        being intact; the filter must not reshuffle."""
        scored = [
            ('taxon_c', 0.9),
            ('taxon_a', 0.5),
            ('taxon_b', 0.7),
        ]
        result = filter_excluded(scored, {'taxon_a'})
        assert result == [('taxon_c', 0.9), ('taxon_b', 0.7)]

    def test_empty_exclusion_set_passes_through(self) -> None:
        """No exclusions → return the original list (defensive
        fast-path; lets callers always call filter_excluded
        without checking)."""
        scored = [('taxon_a', 0.5), ('taxon_b', 0.7)]
        assert filter_excluded(scored, set()) == scored

    def test_all_excluded_returns_empty(self) -> None:
        """Edge case: every candidate already in the exclusion set.
        Operator sees an empty selection and knows to expand the
        source DB or drop the --exclude flag."""
        scored = [('taxon_a', 0.5), ('taxon_b', 0.7)]
        result = filter_excluded(scored, {'taxon_a', 'taxon_b'})
        assert result == []


# ---------------------------------------------------------------------------
# score_treatments_in_db also returns merge_metrics
# ---------------------------------------------------------------------------


class TestScoreTreatmentsInDbMergeMetrics:
    """Post-2026-07-01: the scorer computes merge metrics in the
    same doc-read pass so the CLI's --exclude-suspected-merges
    filter doesn't need a second scan of the treatments_prose DB."""

    def test_merge_metrics_populated_for_scored_treatments(self) -> None:
        """Every treatment that scores > 0 gets a merge_metric."""
        db = _FakeTreatmentsDb({
            'taxon_a': {'description': 'Pileus brown 3 cm.'},
            'taxon_b': {'description': 'Stipe 5 mm long.'},
        })
        scored, merge_metrics = score_treatments_in_db(db, verbosity=0)
        for tid, _score in scored:
            assert tid in merge_metrics
            assert isinstance(merge_metrics[tid], int)
            assert merge_metrics[tid] >= 0

    def test_high_repetition_treatment_scores_high_metric(
        self,
    ) -> None:
        """A treatment where a technical term is repeated many
        times gets a high merge metric.  Sanity check that the
        pipeline actually invokes the metric function."""
        db = _FakeTreatmentsDb({
            'taxon_merged': {
                'description': 'perithecia ' * 10 + 'spores ' * 10
                    + 'asci ' * 10,
            },
        })
        _scored, merge_metrics = score_treatments_in_db(
            db, verbosity=0,
        )
        # 3 terms each appearing 10 times → 3 above k=5.
        assert merge_metrics['taxon_merged'] == 3


# ---------------------------------------------------------------------------
# fetch_prior_merge_skip_ids — reads previously-flagged treatments
# out of the status DB so re-runs don't re-flag them.
# ---------------------------------------------------------------------------


class _FakeStatusRow:
    def __init__(self, doc: Any) -> None:
        self.doc = doc


class _FakeStatusView:
    def __init__(self, rows: Any) -> None:
        self.rows = list(rows)


class _FakeStatusDb:
    """Stand-in for the status DB supporting
    ``view('_all_docs', include_docs=True)``.  Docs indexed by _id."""

    def __init__(self, docs: Any) -> None:
        self._docs = list(docs)

    def view(self, _name: str, **_kwargs: Any) -> _FakeStatusView:
        return _FakeStatusView(_FakeStatusRow(d) for d in self._docs)


class TestFetchPriorMergeSkipIds:
    def test_only_skip_status_returned(self) -> None:
        """Only status=='skipped_merge_suspect' docs contribute."""
        db = _FakeStatusDb([
            {
                'treatment_id': 'taxon_a',
                'status': STATUS_SKIPPED_MERGE_SUSPECT,
            },
            {
                'treatment_id': 'taxon_b',
                'status': STATUS_SUCCESS,
            },
            {
                'treatment_id': 'taxon_c',
                'status': STATUS_ERROR,
            },
            {
                'treatment_id': 'taxon_d',
                'status': STATUS_SKIPPED_MERGE_SUSPECT,
            },
        ])
        assert fetch_prior_merge_skip_ids(db) == {
            'taxon_a', 'taxon_d',
        }

    def test_skips_none_doc_rows(self) -> None:
        """Defensive: a view row with doc=None (couchdb-python
        can produce this on deleted docs) is silently skipped."""
        class _Row:
            doc = None
        db = _FakeStatusDb([])
        # Inject a None-doc row directly
        db._docs = []
        db.view = lambda *_, **__: _FakeStatusView([_Row()])
        assert fetch_prior_merge_skip_ids(db) == set()

    def test_empty_db_returns_empty_set(self) -> None:
        assert fetch_prior_merge_skip_ids(_FakeStatusDb([])) == set()


# ---------------------------------------------------------------------------
# apply_merge_filter — the core drop-suspected-merges logic
# ---------------------------------------------------------------------------


class TestApplyMergeFilter:
    def test_filters_by_threshold(self) -> None:
        """Treatments with metric >= threshold are moved to
        newly_flagged; others survive."""
        scored = [
            ('taxon_low', 0.5),
            ('taxon_high', 0.7),
        ]
        metrics = {'taxon_low': 3, 'taxon_high': 15}
        surviving, newly_flagged = apply_merge_filter(
            scored, metrics, threshold=10, already_flagged=set(),
        )
        assert surviving == [('taxon_low', 0.5)]
        assert newly_flagged == [('taxon_high', 15)]

    def test_at_threshold_is_flagged(self) -> None:
        """>= threshold — at-boundary counts as suspect."""
        surviving, flagged = apply_merge_filter(
            [('taxon_a', 0.5)], {'taxon_a': 10},
            threshold=10, already_flagged=set(),
        )
        assert surviving == []
        assert flagged == [('taxon_a', 10)]

    def test_already_flagged_excluded_without_reflagging(
        self,
    ) -> None:
        """A treatment in already_flagged is silently dropped
        from the pool AND does NOT reappear in newly_flagged
        (so we don't rewrite its status doc on every run)."""
        scored = [('taxon_prior', 0.5), ('taxon_new', 0.7)]
        metrics = {'taxon_prior': 20, 'taxon_new': 15}
        surviving, newly_flagged = apply_merge_filter(
            scored, metrics, threshold=10,
            already_flagged={'taxon_prior'},
        )
        assert surviving == []
        assert newly_flagged == [('taxon_new', 15)]

    def test_missing_metric_treated_as_zero(self) -> None:
        """Defensive: a treatment without a merge_metric entry
        (e.g., excluded during scoring) shouldn't crash the
        filter — treat as safely-below-threshold."""
        surviving, flagged = apply_merge_filter(
            [('taxon_a', 0.5)], {},
            threshold=10, already_flagged=set(),
        )
        assert surviving == [('taxon_a', 0.5)]
        assert flagged == []

    def test_empty_input_returns_empty(self) -> None:
        surviving, flagged = apply_merge_filter(
            [], {}, threshold=10, already_flagged=set(),
        )
        assert surviving == []
        assert flagged == []

    def test_preserves_input_order_of_survivors(self) -> None:
        """Banding downstream depends on the scored-order being
        intact after filtering."""
        scored = [
            ('taxon_c', 0.9),
            ('taxon_a', 0.5),
            ('taxon_b', 0.7),
        ]
        metrics = {'taxon_c': 3, 'taxon_a': 15, 'taxon_b': 4}
        surviving, _ = apply_merge_filter(
            scored, metrics, threshold=10, already_flagged=set(),
        )
        assert surviving == [('taxon_c', 0.9), ('taxon_b', 0.7)]


class TestDefaultOutputPath:
    """A selection is the only record of which treatments a round
    covered, and `--exclude-annotated` reads the candidate DB live —
    so once the annotator runs, the same seed no longer reproduces the
    same 50.  The selection must therefore persist somewhere durable
    by default, not rely on the operator redirecting stdout."""

    def test_first_round_is_round1(self, tmp_path: Path) -> None:
        p = select_for_annotation.default_output_path(
            'production_v4', tmp_path)
        assert p == tmp_path / 'production_v4_round1.txt'

    def test_next_free_round_number(self, tmp_path: Path) -> None:
        for n in (1, 2, 3):
            (tmp_path / f'production_v4_round{n}.txt').write_text('x')
        p = select_for_annotation.default_output_path(
            'production_v4', tmp_path)
        assert p == tmp_path / 'production_v4_round4.txt'

    def test_never_clobbers_an_existing_selection(
        self, tmp_path: Path
    ) -> None:
        """Overwriting a past round would destroy the only record of
        what it covered."""
        for n in range(1, 6):
            (tmp_path / f'production_v4_round{n}.txt').write_text('x')
        p = select_for_annotation.default_output_path(
            'production_v4', tmp_path)
        assert not p.exists()

    def test_experiments_are_numbered_independently(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / 'production_v4_round1.txt').write_text('x')
        p = select_for_annotation.default_output_path(
            'production_v3_hand', tmp_path)
        assert p == tmp_path / 'production_v3_hand_round1.txt'

    def test_numbering_continues_past_a_gap(self, tmp_path: Path) -> None:
        """Round numbers are sequential history, not free slots.

        An earlier draft filled the lowest gap, which broke on the
        real directory: rounds 1-3 happened but were never captured
        as files, so the first run after round4 landed would have
        been numbered round1.  Continue past the highest instead —
        a missing file means that round's selection wasn't kept, not
        that the number is available.
        """
        for n in (1, 3):
            (tmp_path / f'production_v4_round{n}.txt').write_text('x')
        p = select_for_annotation.default_output_path(
            'production_v4', tmp_path)
        assert p == tmp_path / 'production_v4_round4.txt'


class TestWriteSelection:
    """Writing must not break the documented stdout pipe into
    bin/llm_annotate_features."""

    def test_writes_one_id_per_line(self, tmp_path: Path) -> None:
        out = tmp_path / 'sel.txt'
        select_for_annotation.write_selection(['a', 'b', 'c'], out)
        assert out.read_text() == 'a\nb\nc\n'

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        out = tmp_path / 'nested' / 'deeper' / 'sel.txt'
        select_for_annotation.write_selection(['a'], out)
        assert out.read_text() == 'a\n'


# ---------------------------------------------------------------------------
# resolve_seed / build_round_metadata — machine-written round provenance
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="T0e: implementation pending")
class TestResolveSeed:
    """Every round gets a concrete seed, recorded."""

    def test_explicit_seed_is_returned_unchanged(self) -> None:
        assert resolve_seed(20260823) == (20260823, False)

    def test_zero_is_a_seed_not_an_absence(self) -> None:
        """`if seed:` would silently regenerate on --seed 0."""
        assert resolve_seed(0) == (0, False)

    def test_absent_seed_is_generated_and_flagged(self) -> None:
        """An unrecorded seed makes a draw unreproducible in principle.

        The old behaviour used a bare random.Random(), so the seed was
        unrecoverable; a sidecar saying `"seed": null` would document
        an irretrievable gap rather than prevent one.
        """
        seed, generated = resolve_seed(None)
        assert generated is True
        assert isinstance(seed, int) and seed >= 0

    def test_generated_seeds_differ_between_calls(self) -> None:
        seeds = {resolve_seed(None)[0] for _ in range(8)}
        assert len(seeds) > 1


@pytest.mark.xfail(strict=True, reason="T0e: implementation pending")
class TestBuildRoundMetadata:
    """The sidecar: a funnel, not a boolean."""

    def _meta(self, **over: Any) -> Dict[str, Any]:
        kw: Dict[str, Any] = dict(
            experiment='production_v4',
            seed=20260823,
            seed_generated=False,
            n_requested=1000,
            n_selected=1000,
            band_specs=[('all', 1000)],
            band_rows=[],
            funnel=[{'stage': 'all_treatments', 'n': 81527},
                    {'stage': 'complexity_gt_0', 'n': 46045},
                    {'stage': 'not_merge_suspect', 'n': 38413}],
            merge_threshold=10,
            force_recompute=False,
            selector_argv=['--n', '1000'],
            drawn_at='2026-08-24T00:00:00+00:00',
        )
        kw.update(over)
        return build_round_metadata(**kw)

    def test_single_band_is_recorded_as_uniform(self) -> None:
        meta = self._meta()
        assert meta['selection'] == 'uniform'
        assert meta['output_order'] == 'uniform'
        assert meta['bands'] is None

    def test_multiple_bands_are_stratified_and_band_ordered(self) -> None:
        """output_order is load-bearing: on a banded round the first
        N lines come from the first band, so `head -50` is not a
        random subset."""
        meta = self._meta(
            band_specs=[('low', 25), ('mid', 50), ('high', 25)],
            band_rows=[{'name': 'low', 'quota': 25, 'slice_n': 12768,
                        'score_min': 0.1, 'score_max': 2.4}],
        )
        assert meta['selection'] == 'stratified'
        assert meta['output_order'] == 'band-by-band'
        assert meta['bands'] == [['low', 25], ['mid', 50], ['high', 25]]
        assert meta['band_slices'][0]['score_max'] == 2.4

    def test_funnel_and_seed_provenance_are_carried(self) -> None:
        meta = self._meta(seed_generated=True)
        assert meta['seed'] == 20260823
        assert meta['seed_generated'] is True
        assert meta['population_funnel'][-1]['n'] == 38413
        assert meta['merge_threshold'] == 10

    def test_is_json_serialisable(self) -> None:
        """It is written to disk; a tuple would blow up at dump time."""
        json.dumps(self._meta(
            band_specs=[('low', 5), ('high', 5)], band_rows=[],
        ))
