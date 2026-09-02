#!/usr/bin/env python3
"""Tests for ``ingestors.xml_formats``.

The point of the module is that **membership in the JATS family is
encoded once**.  Before it, four sites tested ``xml_format == 'jats'``
and two tested ``xml_format in ('jats', 'taxpub')``, and a stale test
in ``pensoft_test`` asserted that a TaxPub document detects as
``'jats'`` — three different answers to one question.

So these tests are mostly about the registry being the single source
of truth: detection returns only registered names, membership is read
off the registry rather than restated, and adding a format (BITS,
NISO STS) is one entry.
"""

import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestors.xml_formats import (  # noqa: E402
    FORMATS,
    JATS,
    TAXPUB,
    detect,
    is_jats_family,
    is_plain_jats,
    is_taxpub,
)


TAXPUB_HEADER = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n'
    b'<!DOCTYPE article PUBLIC "-//TaxPub//DTD Taxonomic Treatment '
    b'Publishing DTD v1.0 20180101//EN" '
    b'"https://raw.githubusercontent.com/plazi/TaxPub/TaxPubJATS/'
    b'tax-treatment-NS0-v1.dtd">\n'
    b'<article article-type="research-article" '
    b'dtd-version="3.0" xml:lang="en" '
    b'xmlns:mml="http://www.w3.org/1998/Math/MathML" '
    b'xmlns:xlink="http://www.w3.org/1999/xlink" '
    b'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
    b'xmlns:tp="http://www.plazi.org/taxpub">\n'
)


class TestDetect:
    """Every rule that was in ``PensoftIngestor._detect_xml_format``,
    preserved exactly — including the ones that look accidental."""

    @pytest.mark.parametrize('content,expected', [
        (TAXPUB_HEADER, TAXPUB),
        (b'<?xml version="1.0"?>\n<!DOCTYPE article PUBLIC '
         b'"-//NLM//DTD JATS v1.2//EN">\n<article>...', JATS),
        (b'<?xml version="1.0"?>\n<!DOCTYPE article PUBLIC '
         b'"-//NLM//DTD JournalPublishing v3.0//EN">\n<article>...', JATS),
        (b'<?xml version="1.0"?>\n<article '
         b'xmlns="http://example.org/schema">\n<front>...', JATS),
        (b'<?xml version="1.0"?>\n<!DOCTYPE article SYSTEM '
         b'"article.dtd">\n<article>...', JATS),
        (b'<?xml version="1.0"?>\n<root><data>hello</data></root>', None),
        (b'', None),
    ])
    def test_headers_detect_as_expected(
            self, content: bytes, expected: Optional[str]) -> None:
        assert detect(content) == expected

    def test_taxpub_beats_jats_because_it_is_more_specific(self) -> None:
        """TaxPub *is* JATS — the header carries both markers, and the
        registry order is what makes the specific answer win.  This is
        the case the old pensoft test got backwards."""
        assert b'JATS' in TAXPUB_HEADER or b'article' in TAXPUB_HEADER
        assert detect(TAXPUB_HEADER) == TAXPUB

    def test_jats_marker_is_case_sensitive(self) -> None:
        """Preserved from the original: 'JATS' matched case-sensitively
        while 'journalpublishing' did not.  Lower-casing the first
        would newly match any URL containing 'jats'."""
        assert detect(b'<?xml version="1.0"?>\n<jats>no</jats>') is None

    def test_only_the_header_is_scanned(self) -> None:
        """2 000 bytes, as before: a marker in the body must not
        retro-classify a document."""
        content = b'<?xml version="1.0"?>\n' + b'x' * 3000 + b'JATS'
        assert detect(content) is None

    def test_detect_returns_a_registered_name_or_none(self) -> None:
        names = {fmt.name for fmt in FORMATS}
        for content in (TAXPUB_HEADER, b'<article xmlns="x">', b'nope'):
            got = detect(content)
            assert got is None or got in names


class TestMembership:
    """The one question the codebase kept answering three ways."""

    def test_taxpub_is_in_the_jats_family(self) -> None:
        """`is_jats = xml_fmt in ('jats', 'taxpub')`, the line this
        module exists to delete from every caller."""
        assert is_jats_family(TAXPUB) is True

    def test_jats_is_in_the_jats_family(self) -> None:
        assert is_jats_family(JATS) is True

    @pytest.mark.parametrize('value', [None, '', 'bits', 'docbook', 'JATS'])
    def test_unknown_and_missing_formats_are_not_family(
            self, value: Optional[str]) -> None:
        """Callers read `doc.get('xml_format')`, which is routinely
        absent.  Unknown names are not guessed at."""
        assert is_jats_family(value) is False

    def test_membership_is_read_off_the_registry(self) -> None:
        """Not restated here: if a future entry declares itself
        JATS-family, this passes without editing."""
        for fmt in FORMATS:
            assert is_jats_family(fmt.name) is fmt.jats_family

    def test_plain_jats_excludes_taxpub(self) -> None:
        """The four `== 'jats'` sites mean this, and saying so is the
        point -- whether they *should* is a separate question."""
        assert is_plain_jats(JATS) is True
        assert is_plain_jats(TAXPUB) is False

    def test_is_taxpub_is_exact(self) -> None:
        assert is_taxpub(TAXPUB) is True
        assert is_taxpub(JATS) is False
        assert is_taxpub(None) is False


class TestRegistry:
    """Adding BITS or NISO STS should be one entry, so the invariants
    that make that safe are pinned here."""

    def test_names_are_unique(self) -> None:
        names = [fmt.name for fmt in FORMATS]
        assert len(names) == len(set(names))

    def test_more_specific_formats_come_first(self) -> None:
        """Detection walks the registry in order and returns the first
        match, so a profile of a format must precede it.  TaxPub is a
        JATS profile."""
        order = [fmt.name for fmt in FORMATS]
        assert order.index(TAXPUB) < order.index(JATS)

    def test_every_format_declares_its_family(self) -> None:
        for fmt in FORMATS:
            assert isinstance(fmt.jats_family, bool)
