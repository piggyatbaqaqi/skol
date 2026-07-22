"""Tests for django.search.deep_links.

Trello #401 Phase 1 Commit C: resolver turns raw source_anchors
into user-facing deep_links.  Covers priority sorting, per-kind
URL construction, and Tier-2 filtering.
"""

from search.deep_links import resolve_anchors


class TestResolveAnchorsPriority:
    def test_pdf_before_plazi_before_jats_section(self):
        """Kinds are re-sorted by policy regardless of storage
        order.  Extractor may emit in any order; Django renders in
        _KIND_PRIORITY order."""
        anchors = [
            {'kind': 'jats_section', 'doi': '10.0/x',
             'section_id': 'SECID0AAA'},
            {'kind': 'plazi', 'uuid': 'UUID1'},
            {'kind': 'pdf', 'page': '3', 'label': '3'},
        ]
        ingest = {'pdf_url': 'https://e.com/x.pdf',
                  'url': 'https://e.com/article/1/'}
        result = resolve_anchors(anchors, ingest)
        assert [d['kind'] for d in result] == [
            'pdf', 'plazi', 'jats_section',
        ]

    def test_empty_input_yields_empty(self):
        assert resolve_anchors([], {}) == []

    def test_tier2_kinds_filtered_out(self):
        """arpha and mycobank are stored but not rendered until
        their resolvers become useful — see docs/source-anchors.md
        for the credential/content gating rationale."""
        anchors = [
            {'kind': 'arpha', 'uuid': 'BA154B2C-A975-…'},
            {'kind': 'mycobank', 'id': '853632'},
            {'kind': 'pdf', 'page': '3', 'label': '3'},
        ]
        ingest = {'pdf_url': 'https://e.com/x.pdf'}
        result = resolve_anchors(anchors, ingest)
        kinds = [d['kind'] for d in result]
        assert 'arpha' not in kinds
        assert 'mycobank' not in kinds
        assert 'pdf' in kinds


class TestPdfResolver:
    def test_pdf_with_page_builds_fragment(self):
        result = resolve_anchors(
            [{'kind': 'pdf', 'page': '77', 'label': '77'}],
            {'pdf_url': 'https://e.com/x.pdf'},
        )
        assert len(result) == 1
        assert result[0]['href'] == 'https://e.com/x.pdf#page=77'
        assert result[0]['label'] == 'PDF page 77'

    def test_pdf_label_shown_when_differs_from_page(self):
        """PDFs with roman-numeral front matter or other non-numeric
        labels: the label goes into the display, not the page."""
        result = resolve_anchors(
            [{'kind': 'pdf', 'page': '3', 'label': 'iii'}],
            {'pdf_url': 'https://e.com/x.pdf'},
        )
        assert result[0]['label'] == 'PDF page iii'

    def test_pdf_dropped_without_pdf_url(self):
        """No pdf_url in ingest → PDF anchor is silently dropped.
        Better than serving a broken link."""
        result = resolve_anchors(
            [{'kind': 'pdf', 'page': '3', 'label': '3'}],
            {},  # no pdf_url
        )
        assert result == []


class TestPlaziResolver:
    def test_plazi_uses_treatmentbank_url(self):
        """Verified 2026-07-10 that treatment.plazi.org resolves
        free without auth."""
        uuid = '0A4F6E6CD877BD32697F3B6BB9EF2AB5'
        result = resolve_anchors(
            [{'kind': 'plazi', 'uuid': uuid}], {},
        )
        assert result[0]['href'] == (
            f'https://treatment.plazi.org/id/{uuid}'
        )
        assert result[0]['label'] == 'Open at Plazi'

    def test_plazi_multiple_uuids_all_rendered(self):
        """doc.plazi.uuids is article-level; every treatment gets
        the full list.  Each becomes its own deep link."""
        anchors = [
            {'kind': 'plazi', 'uuid': 'UUID1'},
            {'kind': 'plazi', 'uuid': 'UUID2'},
        ]
        result = resolve_anchors(anchors, {})
        assert [d['href'] for d in result] == [
            'https://treatment.plazi.org/id/UUID1',
            'https://treatment.plazi.org/id/UUID2',
        ]


class TestJatsSectionResolver:
    def test_uses_ingest_url_when_available(self):
        """Prefers ingest.url for the article-URL base — works for
        any host (Pensoft/MDPI/etc), no URL-scheme knowledge in
        the resolver."""
        result = resolve_anchors(
            [{'kind': 'jats_section',
              'doi': '10.3897/mycokeys.108.130565',
              'section_id': 'SECID0ELWGK'}],
            {'url': 'https://mycokeys.pensoft.net/article/130565/'},
        )
        assert result[0]['href'] == (
            'https://mycokeys.pensoft.net/article/130565/#SECID0ELWGK'
        )

    def test_appends_trailing_slash_if_missing(self):
        """ingest.url without trailing slash gets one before the
        fragment — Pensoft's article URLs sometimes lack it."""
        result = resolve_anchors(
            [{'kind': 'jats_section',
              'doi': '10.3897/x', 'section_id': 'SECID0AAA'}],
            {'url': 'https://e.com/article/1'},
        )
        assert result[0]['href'] == (
            'https://e.com/article/1/#SECID0AAA'
        )

    def test_falls_back_to_doi_when_no_ingest_url(self):
        """No ingest.url → resolver uses doi.org; browsers follow
        the redirect and preserve the fragment."""
        result = resolve_anchors(
            [{'kind': 'jats_section',
              'doi': '10.3897/mycokeys.108.130565',
              'section_id': 'SECID0EGZGK'}],
            {},
        )
        assert result[0]['href'] == (
            'https://doi.org/10.3897/mycokeys.108.130565'
            '#SECID0EGZGK'
        )

    def test_dropped_when_no_section_id(self):
        """A jats_section anchor without section_id can't build a
        fragment — dropped."""
        result = resolve_anchors(
            [{'kind': 'jats_section', 'doi': '10.0/x'}],
            {'url': 'https://e.com/article/1/'},
        )
        assert result == []


class TestCombinedRendering:
    def test_full_pensoft_treatment_gives_three_links(self):
        """Pensoft is_taxpub treatment with PDF + Plazi + JATS
        section: renders 3 links in policy order.  Tier-2 kinds
        (arpha, mycobank) stored on record but not rendered."""
        anchors = [
            {'kind': 'pdf', 'page': '5', 'label': '5'},
            {'kind': 'plazi', 'uuid': 'UUID_PLAZI'},
            {'kind': 'jats_section',
             'doi': '10.3897/mycokeys.108.130565',
             'section_id': 'SECID0EGZGK'},
            {'kind': 'arpha', 'uuid': 'BA154B2C-A975-…'},
            {'kind': 'mycobank', 'id': '853632'},
        ]
        ingest = {
            'pdf_url': 'https://mycokeys.pensoft.net/article/130565/download/pdf',
            'url': 'https://mycokeys.pensoft.net/article/130565/',
        }
        result = resolve_anchors(anchors, ingest)
        assert len(result) == 3
        assert result[0]['kind'] == 'pdf'
        assert result[1]['kind'] == 'plazi'
        assert result[2]['kind'] == 'jats_section'
