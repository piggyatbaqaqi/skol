"""Tests for bin/backfill_experiment_golden_fields.py.

Covers the per-experiment golden-DB value map, idempotency, the
"don't overwrite existing values" guarantee, and dry-run mode.
No real CouchDB is touched — a small in-memory dict stub stands in.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_experiment_golden_fields import (  # type: ignore[import]  # noqa: E402
    _EXPERIMENT_GOLDEN_MAP,
    backfill,
    backfill_one_doc,
)


# ---------------------------------------------------------------------------
# The hard-coded per-experiment value map (Step 1.A in docs/golden_v2_plan.md)
# ---------------------------------------------------------------------------


class TestExperimentGoldenMap:
    """The known v1 experiments each have an expected (golden, golden_ann)
    pair.  The JATS-trained experiments are deliberately pointed at the
    JATS silver standard — fixing the latent mis-pairing called out in
    the plan doc."""

    def test_hand_trained_experiments_use_hand_gold(self) -> None:
        for name in ("production", "hand_annotated"):
            golden, ann = _EXPERIMENT_GOLDEN_MAP[name]
            assert golden == "skol_golden"
            assert ann == "skol_golden_ann_hand"

    def test_jats_trained_experiments_use_jats_silver(self) -> None:
        """Per the plan, jats_v1 and the taxpub_v1 family must score against
        skol_golden_ann_jats (the silver they were trained on) — not the
        hand gold the hardcoded literal previously used."""
        for name in ("jats_v1", "taxpub_v1", "taxpub_v1_int8",
                     "taxpub_v1_onnx_int8"):
            golden, ann = _EXPERIMENT_GOLDEN_MAP[name]
            assert golden == "skol_golden"
            assert ann == "skol_golden_ann_jats"

    def test_no_unknown_experiments_in_map(self) -> None:
        """The map covers exactly the six v1 experiments — no more, no less."""
        assert set(_EXPERIMENT_GOLDEN_MAP) == {
            "production", "hand_annotated", "jats_v1",
            "taxpub_v1", "taxpub_v1_int8", "taxpub_v1_onnx_int8",
        }


# ---------------------------------------------------------------------------
# backfill_one_doc: pure function operating on a doc dict
# ---------------------------------------------------------------------------


class TestBackfillOneDoc:
    def test_adds_fields_when_missing(self) -> None:
        doc = {"_id": "production", "databases": {"ingest": "skol_dev"}}
        new, changed = backfill_one_doc(doc)
        assert changed is True
        assert new["databases"]["golden"] == "skol_golden"
        assert new["databases"]["golden_ann"] == "skol_golden_ann_hand"
        # Existing fields preserved.
        assert new["databases"]["ingest"] == "skol_dev"

    def test_preserves_existing_values_idempotent(self) -> None:
        doc = {
            "_id": "production",
            "databases": {
                "golden": "skol_golden_v2",            # already set
                "golden_ann": "skol_golden_ann_hand_v2",
            },
        }
        new, changed = backfill_one_doc(doc)
        assert changed is False
        assert new is doc
        assert new["databases"]["golden"] == "skol_golden_v2"
        assert new["databases"]["golden_ann"] == "skol_golden_ann_hand_v2"

    def test_partial_existing_only_fills_missing(self) -> None:
        """If one of the two fields is already set, only the other is added."""
        doc = {
            "_id": "production",
            "databases": {"golden": "some_custom_golden"},
        }
        new, changed = backfill_one_doc(doc)
        assert changed is True
        assert new["databases"]["golden"] == "some_custom_golden"
        assert new["databases"]["golden_ann"] == "skol_golden_ann_hand"

    def test_jats_v1_gets_silver_not_hand(self) -> None:
        doc = {"_id": "jats_v1", "databases": {}}
        new, _ = backfill_one_doc(doc)
        assert new["databases"]["golden_ann"] == "skol_golden_ann_jats"

    def test_unknown_experiment_returns_unchanged(self) -> None:
        """An experiment doc whose _id is not in the map is left alone (the
        caller decides whether to warn or fail).  This protects future
        experiments and any user-created docs from accidental backfill."""
        doc = {"_id": "ad_hoc_experiment", "databases": {}}
        new, changed = backfill_one_doc(doc)
        assert changed is False
        assert new is doc

    def test_design_doc_unchanged(self) -> None:
        doc = {"_id": "_design/something", "views": {}}
        new, changed = backfill_one_doc(doc)
        assert changed is False
        assert new is doc

    def test_missing_databases_field_creates_one(self) -> None:
        """An experiment doc without a databases field at all (defensive
        edge case): create it on the fly when populating the new keys."""
        doc = {"_id": "production"}
        new, changed = backfill_one_doc(doc)
        assert changed is True
        assert new["databases"] == {
            "golden": "skol_golden",
            "golden_ann": "skol_golden_ann_hand",
        }


# ---------------------------------------------------------------------------
# backfill(): walks the experiments DB and writes back changed docs
# ---------------------------------------------------------------------------


def _make_db_stub(docs: Dict[str, Dict[str, Any]]) -> MagicMock:
    """Return a MagicMock that mimics the CouchDB-python API surface used
    by the backfill script: iteration, __getitem__, save()."""
    db = MagicMock()
    db.__iter__ = MagicMock(return_value=iter(list(docs)))
    db.__getitem__.side_effect = lambda doc_id: dict(docs[doc_id])

    saved: List[Dict[str, Any]] = []

    def _save(doc):
        saved.append(doc)
        docs[doc["_id"]] = dict(doc)
        return doc["_id"], "new-rev"

    db.save.side_effect = _save
    db._saved = saved  # type: ignore[attr-defined]
    return db


class TestBackfillFn:
    def test_updates_changed_docs(self) -> None:
        docs = {
            "production": {"_id": "production", "databases": {}},
            "jats_v1":    {"_id": "jats_v1",    "databases": {}},
        }
        db = _make_db_stub(docs)
        counts = backfill(db, dry_run=False)
        assert counts["updated"] == 2
        assert counts["unchanged"] == 0
        assert counts["skipped"] == 0
        # Verify the values that were written.
        saved = {d["_id"]: d for d in db._saved}
        assert saved["production"]["databases"]["golden_ann"] \
            == "skol_golden_ann_hand"
        assert saved["jats_v1"]["databases"]["golden_ann"] \
            == "skol_golden_ann_jats"

    def test_dry_run_does_not_write(self) -> None:
        docs = {"production": {"_id": "production", "databases": {}}}
        db = _make_db_stub(docs)
        counts = backfill(db, dry_run=True)
        assert counts["updated"] == 1
        # Nothing was actually saved.
        assert db._saved == []
        db.save.assert_not_called()

    def test_idempotent_on_second_pass(self) -> None:
        docs = {"production": {"_id": "production", "databases": {}}}
        db = _make_db_stub(docs)
        first = backfill(db, dry_run=False)
        second = backfill(db, dry_run=False)
        assert first["updated"] == 1
        assert second["updated"] == 0
        assert second["unchanged"] == 1

    def test_skips_design_docs(self) -> None:
        docs = {
            "_design/foo": {"_id": "_design/foo", "views": {}},
            "production":  {"_id": "production",  "databases": {}},
        }
        db = _make_db_stub(docs)
        counts = backfill(db, dry_run=False)
        # _design/foo is not in the map (and starts with '_'); the
        # script should treat it as "skipped", not "updated".
        assert counts["updated"] == 1
        assert counts["skipped"] >= 1

    def test_unknown_experiment_counted_as_skipped(self) -> None:
        docs = {
            "production":          {"_id": "production",          "databases": {}},
            "ad_hoc_experiment":   {"_id": "ad_hoc_experiment",   "databases": {}},
        }
        db = _make_db_stub(docs)
        counts = backfill(db, dry_run=False)
        assert counts["updated"] == 1
        assert counts["skipped"] == 1
        # Only production was saved.
        saved_ids = {d["_id"] for d in db._saved}
        assert saved_ids == {"production"}
