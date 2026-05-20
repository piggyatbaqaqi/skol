"""Tests for bin/build_combined_training_db.py — Step 2.E of
docs/production_v3_plan.md.

The helper unions docs from N source DBs into a single target DB.
v3's combined-corpus use case: hand training + JATS training where
the two corpora are doc-ID disjoint. The helper enforces that with
a collision error rather than silently overwriting one corpus's
annotations with the other's.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the FakeDb stub from the sibling test file.
from build_no_golden_training_db_test import _FakeDb  # type: ignore[import]  # noqa: E402
from build_combined_training_db import (  # type: ignore[import]  # noqa: E402
    build_combined_db,
)


def _doc(doc_id: str, **extra: Any) -> Any:
    base: Any = {"_id": doc_id}
    base.update(extra)
    return base


class TestBuildCombinedDb:
    def test_union_disjoint_sources(self) -> None:
        """Two disjoint sources → target = union, counts add up."""
        s1 = _FakeDb(
            name="s1",
            docs={"a": _doc("a"), "b": _doc("b")},
            attachments={"a": {"article.txt.ann": b"[@A#x*]"}},
            content_types={"a": {"article.txt.ann": "text/plain"}},
        )
        s2 = _FakeDb(
            name="s2",
            docs={"c": _doc("c")},
            attachments={"c": {"article.txt.ann": b"[@C#y*]"}},
            content_types={"c": {"article.txt.ann": "text/plain"}},
        )
        target = _FakeDb(name="tgt")
        counts = build_combined_db([s1, s2], target)
        assert set(target.docs) == {"a", "b", "c"}
        assert counts == {"copied": 3, "skipped_exists": 0}

    def test_collision_raises(self) -> None:
        """Same doc ID in two sources → ValueError; message names both
        source DBs and the colliding ID."""
        s1 = _FakeDb(name="s1_hand", docs={"a": _doc("a")})
        s2 = _FakeDb(name="s2_jats", docs={"a": _doc("a")})
        target = _FakeDb(name="tgt")
        with pytest.raises(ValueError) as exc:
            build_combined_db([s1, s2], target)
        msg = str(exc.value)
        assert "a" in msg
        assert "s1_hand" in msg
        assert "s2_jats" in msg

    def test_idempotent_on_rerun(self) -> None:
        """Second invocation reports skipped_exists for every doc;
        copied is zero."""
        s1 = _FakeDb(name="s1", docs={"a": _doc("a"), "b": _doc("b")})
        s2 = _FakeDb(name="s2", docs={"c": _doc("c")})
        target = _FakeDb(name="tgt")
        build_combined_db([s1, s2], target)
        counts2 = build_combined_db([s1, s2], target)
        assert counts2 == {"copied": 0, "skipped_exists": 3}
        assert set(target.docs) == {"a", "b", "c"}

    def test_attachments_preserved_across_sources(self) -> None:
        """Attachments from each source land in the target with the
        correct bytes and content_type."""
        s1 = _FakeDb(
            name="s1",
            docs={"a": _doc("a")},
            attachments={"a": {"article.txt.ann": b"FROM_S1"}},
            content_types={"a": {"article.txt.ann": "text/plain"}},
        )
        s2 = _FakeDb(
            name="s2",
            docs={"b": _doc("b")},
            attachments={"b": {"article.txt.ann": b"FROM_S2"}},
            content_types={"b": {"article.txt.ann": "text/x-yedda"}},
        )
        target = _FakeDb(name="tgt")
        build_combined_db([s1, s2], target)
        assert target.get_attachment(
            "a", "article.txt.ann",
        ).read() == b"FROM_S1"
        assert target.get_attachment(
            "b", "article.txt.ann",
        ).read() == b"FROM_S2"
        assert target.content_types["b"]["article.txt.ann"] == (
            "text/x-yedda"
        )

    def test_dry_run_leaves_target_unchanged(self) -> None:
        """dry_run=True returns expected counts but writes nothing."""
        s1 = _FakeDb(name="s1", docs={"a": _doc("a"), "b": _doc("b")})
        s2 = _FakeDb(name="s2", docs={"c": _doc("c")})
        target = _FakeDb(name="tgt")
        counts = build_combined_db([s1, s2], target, dry_run=True)
        assert counts == {"copied": 3, "skipped_exists": 0}
        assert target.docs == {}
        assert target.attachments == {}
