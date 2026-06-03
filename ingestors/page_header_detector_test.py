"""Tests for ingestors/page_header_detector.py.

Pytest-compatible.  Each test class targets one sub-step of the
algorithm described in docs/page-header-detection.md (v4 plan §1.B
sub-rows 1.B.1 through 1.B.5):

    1.B.1 — TestCollectCandidates        — digit-token discovery
    1.B.2 — TestFitSequence              — RANSAC sequence fit
    1.B.3 — TestPartitionAlternation     — recto/verso check
    1.B.4 — TestClusterHeaderText        — journal-name clustering
    1.B.5 — TestRecoverHeaderBlock,      — block recovery + orchestrator
            TestDetectPageHeaders
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestors.page_header_detector import (  # noqa: E402
    AlternationFit,
    HeaderRegion,
    HeaderTextCluster,
    PageNumCandidate,
    SequenceFit,
    cluster_header_text,
    collect_candidates,
    detect_page_headers,
    fit_sequence,
    partition_alternation,
    recover_header_block,
)


def _verso(li: int, value: int, suffix: str) -> PageNumCandidate:
    """Verso (left page) candidate: digit at start, header text after."""
    return PageNumCandidate(
        line_index=li, position='start', value=value,
        raw_token=str(value), prefix='', suffix=suffix,
    )


def _recto(li: int, value: int, prefix: str) -> PageNumCandidate:
    """Recto (right page) candidate: digit at end, header text before."""
    return PageNumCandidate(
        line_index=li, position='end', value=value,
        raw_token=str(value), prefix=prefix, suffix='',
    )


def _cand(li: int, value: int, position: str = 'end') -> PageNumCandidate:
    """Concise factory for sequence-fit test fixtures."""
    return PageNumCandidate(
        line_index=li, position=position, value=value,
        raw_token=str(value), prefix='', suffix='',
    )


class TestCollectCandidates:
    """Per page-header-detection.md §Step 1: digit tokens of 1-4 digits
    at the start or end of ``line.strip()``, year-shaped tokens at
    line-end excluded, 5+ digit tokens excluded outright."""

    def test_digit_at_start(self) -> None:
        """Page number at line-start (left-side / verso convention)."""
        result = collect_candidates(['42  MYCOLOGIA'])
        assert len(result) == 1
        c = result[0]
        assert c.line_index == 0
        assert c.position == 'start'
        assert c.value == 42
        assert c.raw_token == '42'

    def test_digit_at_end(self) -> None:
        """Page number at line-end (right-side / recto convention)."""
        result = collect_candidates(['Smith et al. 2023   17'])
        # '2023' is year-shaped at-end and excluded; '17' is the page.
        positions = sorted((c.position, c.value) for c in result)
        assert ('end', 17) in positions
        assert ('end', 2023) not in positions

    def test_year_shape_rejected_at_end(self) -> None:
        """`(19|20)\\d{2}` at line-end is a year, not a page number."""
        assert collect_candidates(['Some Title 2024']) == []
        assert collect_candidates(['Title 1999']) == []

    def test_year_shape_accepted_at_start(self) -> None:
        """Per §Step 1 we only reject year-shape at line-end.  A
        4-digit page at line-start is rare but accepted; the sequence
        fit in 1.B.2 weeds out false positives anyway."""
        result = collect_candidates(['2024 Random Header'])
        assert len(result) == 1
        assert result[0].position == 'start'
        assert result[0].value == 2024

    def test_five_digit_rejected(self) -> None:
        """Specimen IDs, accession numbers, and other 5+ digit
        tokens are excluded outright (§Step 1)."""
        assert collect_candidates(['12345 Some Text']) == []
        assert collect_candidates(['Some Text 12345']) == []

    def test_embedded_digit_rejected(self) -> None:
        """Digits in the middle of a line are body text, not page
        markers."""
        assert collect_candidates(['Found 42 specimens in study']) == []

    def test_empty_and_whitespace_lines(self) -> None:
        """Blank lines yield no candidates."""
        assert collect_candidates(['']) == []
        assert collect_candidates(['   ']) == []
        assert collect_candidates(['\t\n']) == []

    def test_prefix_and_suffix_captured(self) -> None:
        """The non-numeric portion before/after the digit token must
        be captured for 1.B.4 clustering."""
        result = collect_candidates(['12  MYCOLOGIA volume 5'])
        assert len(result) == 1
        c = result[0]
        assert c.value == 12
        assert c.prefix == ''
        # Strip whitespace from suffix so clustering can use it directly.
        assert c.suffix.strip() == 'MYCOLOGIA volume 5'

    def test_multiple_lines_with_indices(self) -> None:
        """line_index preserves position in the input list."""
        lines = [
            'body text',
            '42  HEADER',
            'more body',
            'HEADER  43',
        ]
        result = collect_candidates(lines)
        indices = sorted(c.line_index for c in result)
        assert indices == [1, 3]


class TestFitSequence:
    """Per §Step 2: RANSAC fit ``value ≈ slope × line_index + intercept``
    with a gap histogram quality score that's high when consecutive
    inlier values differ by 1 (every page numbered) or 2 (every other).

    Tests use a fixed numpy seed via the implementation's ``rng`` kwarg
    so the RANSAC sampler is deterministic."""

    def test_clean_monotonic_sequence(self) -> None:
        """Pages 1..10 on evenly-spaced lines fit perfectly; all 10
        candidates are inliers, gap_histogram peaks at 1, quality
        ≥ 0.8."""
        cands = [_cand(li=10 * (i + 1), value=i + 1) for i in range(10)]
        fit = fit_sequence(cands, seed=42)
        assert fit is not None
        assert isinstance(fit, SequenceFit)
        assert len(fit.inlier_line_indices) == 10
        # gap_histogram: 9 consecutive gaps of 1.
        assert fit.gap_histogram.get(1, 0) == 9
        assert fit.quality_score >= 0.8

    def test_outlier_excluded_from_inliers(self) -> None:
        """An OCR transposition (page 6 misread as 38) gets rejected
        by RANSAC even though it has the right line_index."""
        cands = [_cand(li=10 * (i + 1), value=i + 1) for i in range(10)]
        # Replace the 6th candidate's value with 38 (OCR transposition).
        cands[5] = _cand(li=cands[5].line_index, value=38)
        fit = fit_sequence(cands, seed=42)
        assert fit is not None
        outlier_index = cands[5].line_index
        assert outlier_index not in fit.inlier_line_indices

    def test_gap_two_sequence(self) -> None:
        """Every-other-page numbering (gap 2) is a valid sequence."""
        cands = [
            _cand(li=10 * (i + 1), value=2 * i + 1)
            for i in range(6)  # values 1, 3, 5, 7, 9, 11
        ]
        fit = fit_sequence(cands, seed=42)
        assert fit is not None
        # gap_histogram peak should be at 2.
        assert fit.gap_histogram.get(2, 0) >= 4
        assert fit.quality_score >= 0.6

    def test_random_noise_returns_none_or_low_quality(self) -> None:
        """Five scattered candidates with no underlying sequence
        produce either ``None`` or a low-quality fit (< 0.4)."""
        cands = [
            _cand(li=5, value=42),
            _cand(li=15, value=7),
            _cand(li=25, value=199),
            _cand(li=35, value=23),
            _cand(li=45, value=88),
        ]
        fit = fit_sequence(cands, seed=42)
        if fit is not None:
            assert fit.quality_score < 0.4

    def test_too_few_candidates_returns_none(self) -> None:
        """Below 4 candidates the RANSAC floor kicks in and we
        return ``None`` rather than over-fit on tiny inputs."""
        assert fit_sequence([_cand(li=10, value=1)], seed=42) is None
        assert fit_sequence(
            [_cand(li=10, value=1), _cand(li=20, value=2)], seed=42,
        ) is None
        assert fit_sequence(
            [_cand(li=10, value=1), _cand(li=20, value=2),
             _cand(li=30, value=3)], seed=42,
        ) is None

    def test_empty_returns_none(self) -> None:
        assert fit_sequence([], seed=42) is None


class TestPartitionAlternation:
    """Per §Step 3: recto/verso layout produces page-number sequences
    that interleave by parity (odd values on recto, even on verso, or
    vice versa).  Split candidates by ``value % 2``, fit each subset,
    and score how cleanly the parities alternate when the union of
    candidates is sorted by ``line_index``."""

    def test_strict_alternation_scores_high(self) -> None:
        """Pages 1..10 sorted by line_index → every adjacent pair
        alternates parity → alternation_score = 1.0.  Uses 10 pages
        so each parity subset has 5 candidates (above fit_sequence's
        4-inlier floor) and both verso_fit / recto_fit succeed."""
        cands = [_cand(li=20 * (i + 1), value=i + 1) for i in range(10)]
        fit = partition_alternation(cands, seed=42)
        assert isinstance(fit, AlternationFit)
        assert fit.verso_fit is not None
        assert fit.recto_fit is not None
        assert fit.alternation_score >= 0.8

    def test_all_one_parity_scores_zero(self) -> None:
        """All-odd (or all-even) values have nothing to alternate
        against → alternation_score < 0.3 and the opposite-parity
        fit is None."""
        cands = [_cand(li=20 * (i + 1), value=2 * i + 1) for i in range(6)]
        fit = partition_alternation(cands, seed=42)
        assert fit.alternation_score < 0.3
        # All values are odd; the even/verso subset is empty.
        assert fit.verso_fit is None
        assert fit.recto_fit is not None  # the odd sequence still fits

    def test_only_even_values(self) -> None:
        """Mirror of the above — only-even input populates verso_fit
        and leaves recto_fit empty."""
        cands = [_cand(li=20 * (i + 1), value=2 * (i + 1)) for i in range(6)]
        fit = partition_alternation(cands, seed=42)
        assert fit.alternation_score < 0.3
        assert fit.verso_fit is not None
        assert fit.recto_fit is None

    def test_empty_input(self) -> None:
        """Empty candidate list returns both fits None and score 0."""
        fit = partition_alternation([], seed=42)
        assert fit.verso_fit is None
        assert fit.recto_fit is None
        assert fit.alternation_score == 0.0

    def test_partial_alternation_intermediate_score(self) -> None:
        """A sequence with some parity violations sits between the
        clean-alternation and no-alternation regimes."""
        # Parities sorted by line_index: O E O O O O E -> 3 alternations
        # across 6 adjacent pairs -> score = 0.5.
        cands = [
            _cand(li=20, value=1),
            _cand(li=40, value=2),
            _cand(li=60, value=3),
            _cand(li=80, value=5),
            _cand(li=100, value=7),
            _cand(li=120, value=9),
            _cand(li=140, value=10),
        ]
        fit = partition_alternation(cands, seed=42)
        assert 0.3 <= fit.alternation_score < 0.8


class TestClusterHeaderText:
    """Per §Step 4: the non-numeric portion of each confirmed candidate
    is the journal name, author/title, or similar.  Cluster by
    approximate similarity (difflib >= 0.75) so OCR / abbreviation
    drift / case-flips fold into one cluster."""

    def test_two_distinct_repeated_prefixes(self) -> None:
        """A typical journal with running headers: verso pages show
        the journal name 'MYCOLOGIA', recto pages show the author
        line.  Each text repeats across multiple pages, so two
        clusters surface."""
        cands = [
            _verso(li=20, value=2, suffix='MYCOLOGIA'),
            _verso(li=60, value=4, suffix='MYCOLOGIA'),
            _verso(li=100, value=6, suffix='MYCOLOGIA'),
            _recto(li=40, value=3, prefix='Smith et al. 2023'),
            _recto(li=80, value=5, prefix='Smith et al. 2023'),
            _recto(li=120, value=7, prefix='Smith et al. 2023'),
        ]
        clusters = cluster_header_text(cands)
        # Two non-trivial clusters, each containing three line indices.
        assert len(clusters) == 2
        sizes = sorted(len(c.members) for c in clusters)
        assert sizes == [3, 3]
        canonicals = {c.canonical for c in clusters}
        assert 'MYCOLOGIA' in canonicals
        assert 'Smith et al. 2023' in canonicals

    def test_approximate_match_collapses(self) -> None:
        """Three OCR variants of 'Mycologia' merge into a single
        cluster — caps drop, single-char OCR error tolerated."""
        cands = [
            _verso(li=20, value=2, suffix='Mycologia'),
            _verso(li=40, value=4, suffix='MYCOLOGIA'),
            _verso(li=60, value=6, suffix='Mycoîogia'),
        ]
        clusters = cluster_header_text(cands)
        assert len(clusters) == 1
        assert len(clusters[0].members) == 3

    def test_singletons_filtered_out(self) -> None:
        """Each prefix appears once -> no cluster of size >= 2 -> empty
        result.  We only emit clusters that confirm a header pattern."""
        cands = [
            _verso(li=20, value=2, suffix='alpha'),
            _verso(li=40, value=4, suffix='beta'),
            _verso(li=60, value=6, suffix='gamma'),
        ]
        clusters = cluster_header_text(cands)
        assert clusters == []

    def test_empty_input(self) -> None:
        assert cluster_header_text([]) == []

    def test_cluster_kind_journal_for_short_allcaps(self) -> None:
        """Short, all-caps canonical -> 'journal'."""
        cands = [
            _verso(li=20, value=2, suffix='MYCOLOGIA'),
            _verso(li=40, value=4, suffix='MYCOLOGIA'),
        ]
        clusters = cluster_header_text(cands)
        assert len(clusters) == 1
        assert clusters[0].cluster_kind == 'journal'
        assert isinstance(clusters[0], HeaderTextCluster)


class TestRecoverHeaderBlock:
    """Per §Step 5: extend each confirmed candidate's region outward
    (up to 5 lines per direction) over blank or short adjacent lines
    to capture volume/issue strings and blank separators."""

    def test_single_anchor_extends_to_blank_above(self) -> None:
        """One anchor with a blank line above grows to a 2-line region."""
        lines = [
            'body text body text body text body text body text body',
            '',
            '5  MYCOLOGIA',
            'continuing body text continuing body text body text more',
        ]
        confirmed = [_verso(li=2, value=5, suffix='MYCOLOGIA')]
        regions = recover_header_block(lines, confirmed)
        assert len(regions) == 1
        assert regions[0].start_line == 1
        assert regions[0].end_line == 2
        assert regions[0].anchor_value == 5

    def test_extends_to_short_line(self) -> None:
        """A short line adjacent to the anchor (volume number, etc.)
        gets folded into the region."""
        lines = [
            'A long body line of body text body text body text body more',
            'Vol. 99 Issue 3',
            '7  Journal',
            'long body text following the header body text body more',
        ]
        confirmed = [_verso(li=2, value=7, suffix='Journal')]
        regions = recover_header_block(lines, confirmed)
        assert len(regions) == 1
        # 'Vol. 99 Issue 3' is short (< 30 chars) so it's folded in.
        assert regions[0].start_line == 1

    def test_no_extension_past_body_text(self) -> None:
        """Long body-text lines around the anchor stop region growth."""
        long_body = 'x' * 60
        lines = [long_body, long_body, '5 Journal', long_body, long_body]
        confirmed = [_verso(li=2, value=5, suffix='Journal')]
        regions = recover_header_block(lines, confirmed)
        assert len(regions) == 1
        assert regions[0].start_line == 2
        assert regions[0].end_line == 2

    def test_caps_at_5_lines_back(self) -> None:
        """Even when every preceding line is short/blank, region
        growth caps at 5 lines per direction so we don't gobble
        whole title pages."""
        short = 'short'
        lines = [short] * 10 + ['10 Journal'] + [short] * 10
        confirmed = [_verso(li=10, value=10, suffix='Journal')]
        regions = recover_header_block(lines, confirmed)
        assert len(regions) == 1
        # Cap at 5: anchor is line 10, region.start_line should be 5.
        assert regions[0].start_line == 5
        assert regions[0].end_line == 15

    def test_multiple_anchors_yield_multiple_regions(self) -> None:
        """Two non-overlapping anchors produce two regions."""
        long_body = 'y' * 60
        lines = (
            [long_body] * 5
            + ['3 Journal']           # anchor 1, line 5
            + [long_body] * 10
            + ['Journal  4']          # anchor 2, line 16
            + [long_body] * 5
        )
        confirmed = [
            _verso(li=5, value=3, suffix='Journal'),
            _recto(li=16, value=4, prefix='Journal'),
        ]
        regions = recover_header_block(lines, confirmed)
        assert len(regions) == 2
        anchors = {r.anchor_value for r in regions}
        assert anchors == {3, 4}

    def test_empty_confirmed_yields_empty_regions(self) -> None:
        lines = ['body'] * 5
        assert recover_header_block(lines, []) == []


class TestDetectPageHeaders:
    """The orchestrator runs all five stages and packages the result
    as a JSON-serialisable dict."""

    def test_synthetic_doc_with_planted_headers(self) -> None:
        """Build a doc where lines [2, 7, 12, 17, 22, 27] are headers
        with monotonically-increasing page numbers; detector should
        produce regions covering those lines."""
        body = 'a body text line ' * 6  # long enough to defeat short-line
        lines: List[str] = []
        for page in range(1, 11):
            lines.append(body)
            lines.append(body)
            lines.append(body)
            lines.append(body)
            lines.append(f'{page}  MYCOLOGIA')
        # lines 4, 9, 14, 19, ..., 49 carry the page numbers.
        result = detect_page_headers(lines)
        assert result['schema_version'] == '1'
        assert result['n_lines'] == len(lines)
        assert len(result['regions']) > 0
        # Confidence array length matches.
        assert len(result['per_line_confidence']) == len(lines)
        # At least some of the planted lines should be flagged.
        planted = {4 + 5 * i for i in range(10)}
        flagged = {li for li, c in enumerate(result['per_line_confidence'])
                   if c > 0}
        assert flagged & planted

    def test_doc_with_no_sequence(self) -> None:
        """Plain body text with no page markers -> empty regions,
        all-zero per-line confidence, but the schema is still valid."""
        lines = ['body text without page markers'] * 50
        result = detect_page_headers(lines)
        assert result['schema_version'] == '1'
        assert result['regions'] == []
        assert all(c == 0.0 for c in result['per_line_confidence'])
        assert result['n_lines'] == 50

    def test_result_is_json_serializable(self) -> None:
        """The result must round-trip through json without loss so
        Step 2 / annotate_v4.py can attach it to CouchDB."""
        import json
        lines = ['body text'] * 20 + ['5 Journal', 'body', '6 Journal']
        result = detect_page_headers(lines)
        encoded = json.dumps(result)
        decoded = json.loads(encoded)
        assert decoded['schema_version'] == '1'
        assert decoded['n_lines'] == result['n_lines']

    def test_per_line_confidence_length(self) -> None:
        """Always exactly one confidence value per input line."""
        lines = ['body'] * 13 + ['7 Journal'] + ['body'] * 12
        result = detect_page_headers(lines)
        assert len(result['per_line_confidence']) == 26

    def test_empty_input(self) -> None:
        """Empty line list -> empty everything."""
        result = detect_page_headers([])
        assert result['n_lines'] == 0
        assert result['regions'] == []
        assert result['per_line_confidence'] == []

    def test_header_region_type(self) -> None:
        """HeaderRegion dataclass shape is what the orchestrator
        emits (separately from the JSON-serialised dict form)."""
        r = HeaderRegion(start_line=1, end_line=2, confidence=0.5,
                         anchor_value=42)
        assert r.start_line == 1
        assert r.end_line == 2


class TestDetectPageHeadersWithMarkers:
    """The orchestrator accepts an optional ``pdf_page_markers``
    parameter — a list of ``(line_index, page_number)`` tuples from
    particle_detector's PDF-page-marker spans.  Markers feed into the
    sequence fitter as extra confirmed anchors, rescuing docs where
    natural page-number candidates are absent or too sparse to fit a
    line.  See v4 plan §1.B architectural recommendation."""

    def test_param_is_optional(self) -> None:
        """Pre-marker callers (no ``pdf_page_markers`` kwarg) still
        work exactly as before."""
        result = detect_page_headers(['body'] * 5)
        assert result['n_lines'] == 5

    def test_empty_marker_list_is_noop(self) -> None:
        """Empty list should behave identically to no list at all."""
        lines = ['body'] * 30 + ['7 Journal', 'body', '8 Journal']
        result_no = detect_page_headers(lines, seed=42)
        result_empty = detect_page_headers(
            lines, seed=42, pdf_page_markers=[],
        )
        assert (result_no['per_line_confidence']
                == result_empty['per_line_confidence'])

    def test_markers_rescue_doc_with_no_natural_candidates(self) -> None:
        """Body text with no digit tokens at line ends would normally
        produce zero candidates and no regions.  Inject markers as
        deterministic anchors: even with ``sequence_fit`` still None
        (no *natural* sequence to fit), the markers themselves land
        in flagged regions."""
        lines: List[str] = []
        marker_indices: List[int] = []
        # 5 pages, marker on the first line of each, 9 body lines after.
        for page in range(1, 6):
            marker_indices.append(len(lines))
            lines.append(f'--- PDF Page {page} Label {page} ---')
            for _ in range(9):
                lines.append('plain body text no digits anywhere ok')
        baseline = detect_page_headers(lines, seed=42)
        assert baseline['sequence_fit'] is None, (
            'pre-condition: no natural sequence without markers'
        )
        assert baseline['regions'] == [], (
            'pre-condition: no regions without markers either'
        )

        markers = list(zip(marker_indices, range(1, 6)))
        with_markers = detect_page_headers(
            lines, seed=42, pdf_page_markers=markers,
        )
        # sequence_fit stays None — markers don't participate in the
        # natural-sequence fit (their page_number is a PDF-stream
        # index, not the printed page number).
        assert with_markers['sequence_fit'] is None
        # But the markers do produce regions covering the marker lines.
        assert with_markers['regions'], (
            'markers must produce header regions even without a fit'
        )
        flagged = {
            li for li, c in enumerate(with_markers['per_line_confidence'])
            if c > 0
        }
        assert set(marker_indices) <= flagged, (
            'all marker lines must be flagged when used as anchors'
        )

    def test_markers_augment_existing_sequence(self) -> None:
        """Mixed: natural page numbers at line ends PLUS markers.  The
        combined candidate pool should still fit a clean sequence and
        the marker lines join the flagged set."""
        lines: List[str] = []
        marker_indices: List[int] = []
        for page in range(1, 11):
            marker_indices.append(len(lines))
            lines.append(f'--- PDF Page {page} Label {page} ---')
            lines.append('a body text line that is plenty long enough')
            lines.append('another body text line of similar length')
            lines.append(f'{page}  MYCOLOGIA')  # natural footer
            lines.append('a body text line that is plenty long enough')

        markers = list(zip(marker_indices, range(1, 11)))
        result = detect_page_headers(
            lines, seed=42, pdf_page_markers=markers,
        )
        assert result['sequence_fit'] is not None
        flagged = {
            li for li, c in enumerate(result['per_line_confidence'])
            if c > 0
        }
        # Both the markers and the natural footers should appear.
        assert set(marker_indices) <= flagged
        natural_footers = {i + 3 for i in marker_indices}
        assert natural_footers & flagged

    def test_marker_out_of_range_is_ignored(self) -> None:
        """A marker at an index past the line list shouldn't crash —
        out-of-range markers are silently skipped."""
        lines = ['body'] * 5
        result = detect_page_headers(
            lines, seed=42, pdf_page_markers=[(99, 1), (0, 2)],
        )
        assert result['n_lines'] == 5
