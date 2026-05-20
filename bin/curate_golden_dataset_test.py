"""Tests for bin/curate_golden_dataset.py — Step 3 v2 mode.

Focused on the three new helpers added by Step 3 of
docs/golden_v2_plan.md: output DB-name resolution, hand-source DB
resolution, and the v1 → v2 ID inheritance selector.
"""

import io
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curate_golden_dataset import (  # type: ignore[import]  # noqa: E402
    compute_output_db_names,
    resolve_hand_source_db,
    select_via_reuse_ids,
)


# ---------------------------------------------------------------------------
# compute_output_db_names — golden / golden_ann_hand / golden_ann_jats names
# ---------------------------------------------------------------------------


class TestComputeOutputDbNames:
    def test_v1_default(self) -> None:
        golden, hand, jats = compute_output_db_names("v1")
        assert golden == "skol_golden"
        assert hand == "skol_golden_ann_hand"
        assert jats == "skol_golden_ann_jats"

    def test_v2_uses_suffix(self) -> None:
        golden, hand, jats = compute_output_db_names("v2")
        assert golden == "skol_golden_v2"
        assert hand == "skol_golden_ann_hand_v2"
        assert jats == "skol_golden_ann_jats_v2"

    def test_unknown_version_raises(self) -> None:
        """A typo'd --version should fail fast, not silently use the wrong
        DB names."""
        with pytest.raises(ValueError, match="version"):
            compute_output_db_names("v3")


# ---------------------------------------------------------------------------
# resolve_hand_source_db — where to fetch the hand .ann from
# ---------------------------------------------------------------------------


class TestResolveHandSourceDb:
    def test_v1_defaults_to_skol_training(self) -> None:
        assert resolve_hand_source_db("v1", None) == "skol_training"

    def test_v2_defaults_to_skol_training_v2(self) -> None:
        """v2 mode sources hand .ann from the post-hand-annotation
        skol_training_v2 corpus (Step 3.B of golden_v2_plan)."""
        assert resolve_hand_source_db("v2", None) == "skol_training_v2"

    def test_explicit_override_wins(self) -> None:
        """An explicit --hand-source-db beats both version defaults."""
        assert (
            resolve_hand_source_db("v2", "my_custom_db") == "my_custom_db"
        )
        assert (
            resolve_hand_source_db("v1", "my_custom_db") == "my_custom_db"
        )


# ---------------------------------------------------------------------------
# select_via_reuse_ids — inherit the v1 doc-ID partition for v2
# ---------------------------------------------------------------------------


class _FakeAttachment:
    def __init__(self, content: bytes) -> None:
        self._buf = io.BytesIO(content)

    def read(self) -> bytes:
        return self._buf.read()


class _FakeDb:
    """Minimal CouchDB-python-like stub."""

    def __init__(self, docs: Optional[Dict[str, Dict[str, Any]]] = None,
                 attachments: Optional[Dict[str, Dict[str, bytes]]] = None,
                 ) -> None:
        self.docs: Dict[str, Dict[str, Any]] = docs or {}
        self.attachments: Dict[str, Dict[str, bytes]] = attachments or {}
        self.name = "fake"

    def __iter__(self):
        return iter(list(self.docs))

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        if doc_id not in self.docs:
            raise KeyError(doc_id)
        return dict(self.docs[doc_id])

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.docs

    def get_attachment(self, doc_id: str, name: str):
        data = self.attachments.get(doc_id, {}).get(name)
        return _FakeAttachment(data) if data is not None else None


def _make_server(dbs: Dict[str, _FakeDb]) -> MagicMock:
    """Return a fake server whose __getitem__ returns the named FakeDb."""
    server = MagicMock()
    server.__contains__ = MagicMock(side_effect=lambda n: n in dbs)
    server.__getitem__ = MagicMock(side_effect=lambda n: dbs[n])
    return server


class TestSelectViaReuseIds:
    """Step 3.C — the v2 curator must inherit v1's exact 30+75 doc IDs.
    Given a v1 source golden DB, the resulting (hand_selections,
    jats_selections) pair covers exactly the same doc IDs."""

    def _setup(self) -> MagicMock:
        # v1 ann databases — IDs are the partition signal.
        v1_hand = _FakeDb(docs={
            "hand_a": {"_id": "hand_a"},
            "hand_b": {"_id": "hand_b"},
        })
        v1_jats = _FakeDb(docs={
            "jats_x": {"_id": "jats_x"},
            "jats_y": {"_id": "jats_y"},
            "jats_z": {"_id": "jats_z"},
        })
        v1_golden = _FakeDb(docs={
            "hand_a": {}, "hand_b": {}, "jats_x": {}, "jats_y": {}, "jats_z": {},
        })

        # Hand source carries the .ann text we want to inherit.
        hand_source = _FakeDb(
            docs={
                "hand_a": {"_id": "hand_a", "title": "Mushroom A"},
                "hand_b": {"_id": "hand_b", "title": "Mushroom B"},
            },
            attachments={
                "hand_a": {"article.txt.ann":
                          b"[@Foo bar#Nomenclature*]"},
                "hand_b": {"article.txt.ann":
                          b"[@Baz qux#Nomenclature*]"},
            },
        )

        # skol_dev — where the JATS docs live.
        dev = _FakeDb(docs={
            "jats_x": {"_id": "jats_x", "title": "X", "xml_available": True,
                       "xml_format": "jats"},
            "jats_y": {"_id": "jats_y", "title": "Y", "xml_available": True,
                       "xml_format": "jats"},
            "jats_z": {"_id": "jats_z", "title": "Z", "xml_available": True,
                       "xml_format": "jats"},
        })
        return _make_server({
            "skol_golden":          v1_golden,
            "skol_golden_ann_hand": v1_hand,
            "skol_golden_ann_jats": v1_jats,
            "skol_training_v2":     hand_source,
            "skol_dev":             dev,
        })

    def test_returns_correct_partition(self) -> None:
        server = self._setup()
        hand_sels, jats_sels = select_via_reuse_ids(
            server,
            v1_golden_db_name="skol_golden",
            v1_ann_hand_db_name="skol_golden_ann_hand",
            v1_ann_jats_db_name="skol_golden_ann_jats",
            hand_source_db_name="skol_training_v2",
            dev_db_name="skol_dev",
        )
        hand_ids = {s["doc_id"] for s in hand_sels}
        jats_ids = {s["doc_id"] for s in jats_sels}
        assert hand_ids == {"hand_a", "hand_b"}
        assert jats_ids == {"jats_x", "jats_y", "jats_z"}

    def test_hand_selections_carry_yedda_text(self) -> None:
        """Each hand selection has yedda_text drawn from the
        hand-source DB's article.txt.ann attachment."""
        server = self._setup()
        hand_sels, _ = select_via_reuse_ids(
            server,
            v1_golden_db_name="skol_golden",
            v1_ann_hand_db_name="skol_golden_ann_hand",
            v1_ann_jats_db_name="skol_golden_ann_jats",
            hand_source_db_name="skol_training_v2",
            dev_db_name="skol_dev",
        )
        by_id = {s["doc_id"]: s for s in hand_sels}
        assert by_id["hand_a"]["yedda_text"] == "[@Foo bar#Nomenclature*]"
        assert by_id["hand_b"]["yedda_text"] == "[@Baz qux#Nomenclature*]"

    def test_jats_selections_have_dev_doc(self) -> None:
        server = self._setup()
        _, jats_sels = select_via_reuse_ids(
            server,
            v1_golden_db_name="skol_golden",
            v1_ann_hand_db_name="skol_golden_ann_hand",
            v1_ann_jats_db_name="skol_golden_ann_jats",
            hand_source_db_name="skol_training_v2",
            dev_db_name="skol_dev",
        )
        by_id = {s["doc_id"]: s for s in jats_sels}
        assert by_id["jats_x"]["doc"]["title"] == "X"
        # xml_available is preserved so downstream JATS regeneration
        # knows the doc qualifies.
        assert by_id["jats_x"]["doc"]["xml_available"] is True

    def test_set_equality_with_v1_total(self) -> None:
        """The set of hand ∪ jats IDs equals the set of IDs in the v1
        union golden DB — i.e. we cover everything in v1."""
        server = self._setup()
        hand_sels, jats_sels = select_via_reuse_ids(
            server,
            v1_golden_db_name="skol_golden",
            v1_ann_hand_db_name="skol_golden_ann_hand",
            v1_ann_jats_db_name="skol_golden_ann_jats",
            hand_source_db_name="skol_training_v2",
            dev_db_name="skol_dev",
        )
        v2_ids = (
            {s["doc_id"] for s in hand_sels}
            | {s["doc_id"] for s in jats_sels}
        )
        v1_golden_db = server["skol_golden"]
        v1_ids = set(v1_golden_db.docs)
        assert v2_ids == v1_ids

    def test_missing_hand_source_doc_is_skipped(self) -> None:
        """If the hand-source DB doesn't have a doc that v1's golden hand
        DB names, that hand-selection is dropped (with a clear warning at
        a higher level — the helper just omits it)."""
        # Empty hand_source — none of the hand IDs resolve.
        server = self._setup()
        empty_hand = _FakeDb(docs={}, attachments={})
        server.__getitem__.side_effect = lambda n: (
            empty_hand if n == "skol_training_v2"
            else server.__getitem__.side_effect.__wrapped__(n)
            if False else _fallback(n)
        )

        # Simpler: rebuild server with the same fakes but empty hand_source.
        v1_hand = _FakeDb(docs={
            "hand_a": {"_id": "hand_a"}, "hand_b": {"_id": "hand_b"},
        })
        v1_jats = _FakeDb(docs={"jats_x": {"_id": "jats_x"}})
        v1_golden = _FakeDb(docs={
            "hand_a": {}, "hand_b": {}, "jats_x": {}})
        dev = _FakeDb(docs={"jats_x": {"_id": "jats_x", "xml_available": True}})
        server2 = _make_server({
            "skol_golden":          v1_golden,
            "skol_golden_ann_hand": v1_hand,
            "skol_golden_ann_jats": v1_jats,
            "skol_training_v2":     empty_hand,
            "skol_dev":             dev,
        })
        hand_sels, jats_sels = select_via_reuse_ids(
            server2,
            v1_golden_db_name="skol_golden",
            v1_ann_hand_db_name="skol_golden_ann_hand",
            v1_ann_jats_db_name="skol_golden_ann_jats",
            hand_source_db_name="skol_training_v2",
            dev_db_name="skol_dev",
        )
        assert hand_sels == []  # None resolvable
        # JATS partition still intact.
        assert {s["doc_id"] for s in jats_sels} == {"jats_x"}


def _fallback(n):  # pragma: no cover - test helper only
    raise KeyError(n)
