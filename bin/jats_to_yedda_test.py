"""Tests for jats_to_yedda.py CLI additions (--exclude-ids,
--include-ids, --taxpub-only)."""

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _import_has_taxpub():  # type: ignore[no-untyped-def]
    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent)
    )
    sys.path.insert(
        0, str(Path(__file__).resolve().parent)
    )
    from jats_to_yedda import _has_taxpub  # type: ignore[import]
    return _has_taxpub


def _import_select_doc_ids():  # type: ignore[no-untyped-def]
    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent)
    )
    sys.path.insert(
        0, str(Path(__file__).resolve().parent)
    )
    from jats_to_yedda import select_doc_ids  # type: ignore[import]
    return select_doc_ids


def test_has_taxpub_with_treatment() -> None:
    """XML with taxon-treatment elements is detected."""
    _has_taxpub = _import_has_taxpub()
    xml = '<article xmlns:tp="http://www.plazi.org/taxpub"><body><tp:taxon-treatment/></body></article>'
    assert _has_taxpub(xml) is True


def test_has_taxpub_namespace_only() -> None:
    """XML declaring TaxPub namespace but lacking treatments is rejected."""
    _has_taxpub = _import_has_taxpub()
    xml = '<article xmlns:tp="http://www.plazi.org/taxpub"><body><sec><p>Hello</p></sec></body></article>'
    assert _has_taxpub(xml) is False


def test_has_taxpub_plain_jats() -> None:
    """Plain JATS without TaxPub is rejected."""
    _has_taxpub = _import_has_taxpub()
    xml = '<article><body><sec><p>Hello</p></sec></body></article>'
    assert _has_taxpub(xml) is False


def test_exclude_ids_loading() -> None:
    """Exclusion file is loaded correctly."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8",
    ) as f:
        f.write("doc_a\n")
        f.write("doc_b\n")
        f.write("\n")  # blank line ignored
        f.write("doc_c\n")
        path = f.name

    with open(path, encoding="utf-8") as fh:
        exclude_ids = {
            line.strip() for line in fh if line.strip()
        }

    assert exclude_ids == {"doc_a", "doc_b", "doc_c"}
    Path(path).unlink()


# ---------------------------------------------------------------------------
# select_doc_ids — used by --all to assemble the work list.
# Tested with a minimal stub that mimics couchdb-python's
# ``db.view("_all_docs", include_docs=True)`` iterator.
# ---------------------------------------------------------------------------


@dataclass
class _Row:
    id: str
    doc: Optional[Dict[str, Any]]


@dataclass
class _FakeDb:
    """Stub source DB that yields ``_Row`` objects for ``_all_docs``."""

    rows: List[_Row] = field(default_factory=list)

    def view(
        self, name: str, include_docs: bool = False,
    ) -> List[_Row]:
        assert name == "_all_docs"
        assert include_docs is True
        return list(self.rows)


def _jats_doc(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a minimal doc that passes the JATS/TaxPub filters."""
    base: Dict[str, Any] = {
        "xml_available": True,
        "xml_format": "jats",
    }
    if extra:
        base.update(extra)
    return base


class TestSelectDocIds:
    """Step 0 of docs/production_v3_plan.md — ``select_doc_ids`` is the
    work-list builder. The new ``include_ids`` parameter restricts the
    output to a specific ID list while preserving the existing JATS /
    TaxPub filters and the exclude_ids behaviour."""

    def _setup_db(self) -> _FakeDb:
        return _FakeDb(rows=[
            _Row("a", _jats_doc()),
            _Row("b", _jats_doc({"xml_format": "taxpub"})),
            _Row("c", _jats_doc()),
            # filtered out: no xml_available
            _Row("d", _jats_doc({"xml_available": False})),
            # filtered out: wrong xml_format
            _Row("e", _jats_doc({"xml_format": "html"})),
            # filtered out: is_taxpub explicitly False
            _Row("f", _jats_doc({"is_taxpub": False})),
            # filtered out: null doc
            _Row("g", None),
        ])

    def test_no_include_no_exclude(self) -> None:
        """Default behaviour (today's --all): returns all docs that
        pass the JATS/TaxPub filters."""
        select_doc_ids = _import_select_doc_ids()
        db = self._setup_db()
        ids = select_doc_ids(
            db, exclude_ids=set(), include_ids=None,
        )
        assert set(ids) == {"a", "b", "c"}

    def test_include_set_intersected_with_filters(self) -> None:
        """When include_ids is given, output is the intersection of the
        filtered docs and the include set. This is the v3 regen use
        case — restrict to the 1 743 original training IDs."""
        select_doc_ids = _import_select_doc_ids()
        db = self._setup_db()
        ids = select_doc_ids(
            db, exclude_ids=set(),
            include_ids={"a", "c", "z_not_in_db"},
        )
        assert set(ids) == {"a", "c"}

    def test_include_and_exclude_overlap_exclude_wins(self) -> None:
        """If an ID is in both include and exclude, exclude wins —
        explicit removal is stronger than explicit inclusion."""
        select_doc_ids = _import_select_doc_ids()
        db = self._setup_db()
        ids = select_doc_ids(
            db, exclude_ids={"b"},
            include_ids={"a", "b", "c"},
        )
        assert set(ids) == {"a", "c"}

    def test_include_ids_not_in_db_silently_ignored(self) -> None:
        """An include-list ID that doesn't exist in the source DB is
        silently dropped; we don't crash or warn at the helper level."""
        select_doc_ids = _import_select_doc_ids()
        db = self._setup_db()
        ids = select_doc_ids(
            db, exclude_ids=set(),
            include_ids={"a", "ghost_id_1", "ghost_id_2"},
        )
        assert set(ids) == {"a"}

    def test_empty_include_set_yields_nothing(self) -> None:
        """An empty include set means 'process no docs' — distinct
        from include_ids=None (which means 'no restriction')."""
        select_doc_ids = _import_select_doc_ids()
        db = self._setup_db()
        ids = select_doc_ids(
            db, exclude_ids=set(), include_ids=set(),
        )
        assert ids == []

    def test_filters_still_apply_to_included_ids(self) -> None:
        """An ID in the include set that fails a filter (no
        xml_available, wrong xml_format, is_taxpub=False) is still
        dropped. The include set narrows the work list; it does not
        bypass the JATS/TaxPub gates."""
        select_doc_ids = _import_select_doc_ids()
        db = self._setup_db()
        ids = select_doc_ids(
            db, exclude_ids=set(),
            include_ids={"a", "d", "e", "f"},
        )
        # Only 'a' is included AND passes filters.
        assert set(ids) == {"a"}
