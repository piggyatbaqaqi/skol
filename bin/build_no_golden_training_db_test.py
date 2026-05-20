"""Tests for bin/build_no_golden_training_db.py — Step 2.A/B of
docs/production_v3_plan.md.

The helper copies docs from a source training DB into a target DB,
skipping any doc whose ID also appears in a golden-answer-key DB.
This materialises a contamination-free training corpus per v3
precedence rule "golden docs never appear in training."
"""

import io
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_no_golden_training_db import (  # type: ignore[import]  # noqa: E402
    build_no_golden_db,
)


# ---------------------------------------------------------------------------
# FakeDb stub — mirrors the couchdb-python interface we use:
#   db[id], db.save(doc), db.put_attachment(doc, data, filename=,
#   content_type=), db.get_attachment(id, name), id in db, iter(db).
# Each doc carries its own `_attachments` metadata dict so the helper
# can read content_type from the source side.
# ---------------------------------------------------------------------------


class _FakeAttachment:
    def __init__(self, content: bytes) -> None:
        self._buf = io.BytesIO(content)

    def read(self) -> bytes:
        return self._buf.read()


class _FakeDb:
    def __init__(
        self,
        name: str,
        docs: Optional[Dict[str, Dict[str, Any]]] = None,
        attachments: Optional[Dict[str, Dict[str, bytes]]] = None,
        content_types: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        self.name = name
        self.docs: Dict[str, Dict[str, Any]] = docs or {}
        self.attachments: Dict[str, Dict[str, bytes]] = attachments or {}
        self.content_types: Dict[str, Dict[str, str]] = (
            content_types or {}
        )
        # Populate _attachments metadata so the helper can read
        # content_type from each source doc.
        for doc_id, atts in self.attachments.items():
            if doc_id in self.docs:
                meta: Dict[str, Dict[str, str]] = {}
                for att_name in atts:
                    ct = (
                        self.content_types.get(doc_id, {}).get(
                            att_name, "application/octet-stream",
                        )
                    )
                    meta[att_name] = {"content_type": ct}
                self.docs[doc_id]["_attachments"] = meta

    def __iter__(self) -> Any:
        return iter(list(self.docs))

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.docs

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        if doc_id not in self.docs:
            raise KeyError(doc_id)
        return dict(self.docs[doc_id])

    def __len__(self) -> int:
        return len(self.docs)

    def save(self, doc: Dict[str, Any]) -> Any:
        doc_id = doc["_id"]
        existing = self.docs.get(doc_id, {})
        merged = dict(existing)
        merged.update(doc)
        merged["_rev"] = f"{int(existing.get('_rev', '0').split('-')[0]) + 1}-fake"
        self.docs[doc_id] = merged
        return (doc_id, merged["_rev"])

    def get_attachment(
        self, doc_id: str, name: str,
    ) -> Optional[_FakeAttachment]:
        data = self.attachments.get(doc_id, {}).get(name)
        return _FakeAttachment(data) if data is not None else None

    def put_attachment(
        self,
        doc: Dict[str, Any],
        content: bytes,
        filename: str,
        content_type: str,
    ) -> None:
        doc_id = doc["_id"]
        if doc_id not in self.attachments:
            self.attachments[doc_id] = {}
        self.attachments[doc_id][filename] = content
        if doc_id not in self.content_types:
            self.content_types[doc_id] = {}
        self.content_types[doc_id][filename] = content_type
        # Mirror into _attachments metadata to match couchdb-python.
        if doc_id in self.docs:
            atts = self.docs[doc_id].setdefault("_attachments", {})
            atts[filename] = {"content_type": content_type}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildNoGoldenDb:
    """Step 2.A/B: source - golden = target. Idempotent, dry-runnable."""

    def _setup(self) -> Any:
        """Return (source, golden_ann, target) with a, b, c in source,
        b in golden, a/b/c each carrying an article.txt.ann
        attachment so the attachment-preservation tests have material
        to assert on."""
        source = _FakeDb(
            name="src",
            docs={
                "a": {"_id": "a", "title": "A"},
                "b": {"_id": "b", "title": "B"},
                "c": {"_id": "c", "title": "C"},
            },
            attachments={
                "a": {"article.txt.ann": b"[@A#Nomenclature*]"},
                "b": {"article.txt.ann": b"[@B#Description*]"},
                "c": {"article.txt.ann": b"[@C#Etymology*]"},
            },
            content_types={
                "a": {"article.txt.ann": "text/plain"},
                "b": {"article.txt.ann": "text/plain"},
                "c": {"article.txt.ann": "text/plain"},
            },
        )
        golden = _FakeDb(name="gold", docs={"b": {"_id": "b"}})
        target = _FakeDb(name="tgt")
        return source, golden, target

    def test_basic_filtering(self) -> None:
        """golden = {b} → target = {a, c}; counts add up."""
        source, golden, target = self._setup()
        counts = build_no_golden_db(source, golden, target)
        assert set(target.docs) == {"a", "c"}
        assert counts == {
            "copied": 2,
            "skipped_golden": 1,
            "skipped_exists": 0,
        }

    def test_attachments_preserved(self) -> None:
        """Each copied doc carries its source attachment bytes."""
        source, golden, target = self._setup()
        build_no_golden_db(source, golden, target)
        assert target.get_attachment(
            "a", "article.txt.ann",
        ).read() == b"[@A#Nomenclature*]"
        assert target.get_attachment(
            "c", "article.txt.ann",
        ).read() == b"[@C#Etymology*]"

    def test_attachment_content_type_preserved(self) -> None:
        """content_type from the source's _attachments metadata is
        carried over — we do NOT MIME-guess by filename."""
        source, golden, target = self._setup()
        # Deliberately use a non-default content_type to prove the
        # helper reads it from source metadata, not from filename.
        source.docs["a"]["_attachments"]["article.txt.ann"][
            "content_type"
        ] = "text/x-yedda"
        build_no_golden_db(source, golden, target)
        assert (
            target.content_types["a"]["article.txt.ann"]
            == "text/x-yedda"
        )

    def test_idempotent_on_rerun(self) -> None:
        """Second invocation reports skipped_exists for everything
        previously copied; copied count is zero."""
        source, golden, target = self._setup()
        build_no_golden_db(source, golden, target)
        counts2 = build_no_golden_db(source, golden, target)
        assert counts2 == {
            "copied": 0,
            "skipped_golden": 1,
            "skipped_exists": 2,
        }
        # Target unchanged.
        assert set(target.docs) == {"a", "c"}

    def test_empty_golden_copies_all(self) -> None:
        """No golden IDs → target == source."""
        source, _, target = self._setup()
        empty_golden = _FakeDb(name="gold")
        counts = build_no_golden_db(source, empty_golden, target)
        assert set(target.docs) == {"a", "b", "c"}
        assert counts["copied"] == 3
        assert counts["skipped_golden"] == 0

    def test_all_source_in_golden_yields_empty_target(self) -> None:
        """Every source ID present in golden → target empty."""
        source, _, target = self._setup()
        golden = _FakeDb(
            name="gold",
            docs={
                "a": {"_id": "a"},
                "b": {"_id": "b"},
                "c": {"_id": "c"},
            },
        )
        counts = build_no_golden_db(source, golden, target)
        assert target.docs == {}
        assert counts == {
            "copied": 0,
            "skipped_golden": 3,
            "skipped_exists": 0,
        }

    def test_dry_run_leaves_target_unchanged(self) -> None:
        """dry_run=True returns the same counts but writes nothing."""
        source, golden, target = self._setup()
        counts = build_no_golden_db(
            source, golden, target, dry_run=True,
        )
        assert counts == {
            "copied": 2,
            "skipped_golden": 1,
            "skipped_exists": 0,
        }
        assert target.docs == {}
        assert target.attachments == {}
