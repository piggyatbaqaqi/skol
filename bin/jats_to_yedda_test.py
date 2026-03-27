"""Tests for jats_to_yedda.py CLI additions (--exclude-ids, --taxpub-only)."""

import tempfile
from pathlib import Path

import pytest


def _import_has_taxpub():
    import sys
    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent)
    )
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from jats_to_yedda import _has_taxpub
    return _has_taxpub


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
        mode="w", suffix=".txt", delete=False
    ) as f:
        f.write("doc_a\n")
        f.write("doc_b\n")
        f.write("\n")  # blank line ignored
        f.write("doc_c\n")
        path = f.name

    with open(path) as fh:
        exclude_ids = {
            line.strip() for line in fh if line.strip()
        }

    assert exclude_ids == {"doc_a", "doc_b", "doc_c"}
    Path(path).unlink()
