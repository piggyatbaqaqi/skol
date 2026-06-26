"""Tests for bin/select_for_annotation helpers.

The pure ``parse_band_spec`` / ``select_treatments`` logic is
tested in ``treatments_to_structured/select_test.py``.  This file
covers the CLI-side glue: ``_resolve_band_specs`` (CLI-flag
fan-in) and ``score_treatments_in_db`` (CouchDB iteration via a
fake DB stand-in).
"""

import sys
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from select_for_annotation import (  # type: ignore[import]  # noqa: E402
    _resolve_band_specs,
    score_treatments_in_db,
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
        scored = score_treatments_in_db(db, verbosity=0)
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
        scored = score_treatments_in_db(db, verbosity=0)
        ids = {doc_id for doc_id, _ in scored}
        assert ids == {'taxon_populated'}

    def test_returns_id_score_pairs(self) -> None:
        """Each entry is (treatment_id, complexity_score)."""
        db = _FakeTreatmentsDb({
            'taxon_a': {'description': 'Pileus brown 3 cm.'},
            'taxon_b': {'description': 'Stipe 5 mm long.'},
        })
        scored = score_treatments_in_db(db, verbosity=0)
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
        scored = score_treatments_in_db(db, verbosity=0)
        ids = {doc_id for doc_id, _ in scored}
        assert ids == {'taxon_ok'}

    def test_empty_db_returns_empty_list(self) -> None:
        scored = score_treatments_in_db(_FakeTreatmentsDb({}), verbosity=0)
        assert scored == []
