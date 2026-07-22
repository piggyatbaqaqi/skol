"""Tests for bin/triage_treatments."""

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from triage_treatments import (  # noqa: E402
    CSV_COLUMNS,
    _first_nonempty_line,
    build_reviewed_hand_ids,
    build_row,
    review_status,
    sort_rows,
    write_csv,
)


# ---------------------------------------------------------------------------
# Fake CouchDB view row + DB so we can exercise helpers without a
# live server.
# ---------------------------------------------------------------------------


class _FakeRow:
    def __init__(self, rid: str) -> None:
        self.id = rid


class _FakeView:
    def __init__(self, ids: List[str]) -> None:
        self.rows = [_FakeRow(i) for i in ids]


class _FakeDb:
    def __init__(self, ids: List[str]) -> None:
        self._ids = ids

    def view(self, name: str) -> _FakeView:
        assert name == '_all_docs'
        return _FakeView(self._ids)


# ---------------------------------------------------------------------------
# build_reviewed_hand_ids
# ---------------------------------------------------------------------------


class TestBuildReviewedHandIds:
    def test_empty(self) -> None:
        assert build_reviewed_hand_ids(_FakeDb([])) == set()

    def test_prefix_before_colon(self) -> None:
        db = _FakeDb([
            'taxon_abc:Pileus:0',
            'taxon_abc:Stipe:100',
            'taxon_def:Description:0',
        ])
        assert build_reviewed_hand_ids(db) == {
            'taxon_abc', 'taxon_def',
        }

    def test_ignores_bare_ids(self) -> None:
        """A doc ID lacking ':' (e.g., a design doc) is skipped —
        the hand DB only contains annotation docs, but be defensive."""
        db = _FakeDb([
            '_design/whatever',
            'taxon_abc:Pileus:0',
        ])
        assert build_reviewed_hand_ids(db) == {'taxon_abc'}


# ---------------------------------------------------------------------------
# review_status
# ---------------------------------------------------------------------------


class TestReviewStatus:
    def test_unreviewed(self) -> None:
        r = review_status(
            status_doc={'status': 'success', 'annotation_count': 3},
            reviewed_hand_ids=set(),
            treatment_id='taxon_x',
        )
        assert r['reviewed'] is False
        assert r['reviewer'] == ''
        assert r['kept_count'] == 0

    def test_reviewer_action_wins(self) -> None:
        """reviewer_action takes precedence over hand-DB lookup —
        it carries counts the hand-DB path can't provide."""
        r = review_status(
            status_doc={
                'status': 'success',
                'reviewer_action': {
                    'reviewer': 'piggy',
                    'kept_count': 5,
                    'added_count': 2,
                    'deleted_count': 1,
                },
            },
            reviewed_hand_ids={'taxon_x'},
            treatment_id='taxon_x',
        )
        assert r['reviewed'] is True
        assert r['reviewer'] == 'piggy'
        assert r['kept_count'] == 5
        assert r['added_count'] == 2
        assert r['deleted_count'] == 1

    def test_hand_db_fallback(self) -> None:
        """A treatment present in features_hand but with no
        reviewer_action counts as reviewed (predates 2026-07-01)."""
        r = review_status(
            status_doc={'status': 'success'},
            reviewed_hand_ids={'taxon_x'},
            treatment_id='taxon_x',
        )
        assert r['reviewed'] is True
        assert r['reviewer'] == ''
        assert r['kept_count'] == 0

    def test_no_status_doc(self) -> None:
        """A treatment with no status doc is unreviewed by default,
        unless features_hand has entries."""
        r = review_status(
            status_doc=None,
            reviewed_hand_ids=set(),
            treatment_id='taxon_y',
        )
        assert r['reviewed'] is False

    def test_reviewer_action_counts_default_to_zero(self) -> None:
        r = review_status(
            status_doc={
                'reviewer_action': {'reviewer': 'x'},
            },
            reviewed_hand_ids=set(),
            treatment_id='taxon_z',
        )
        assert r['reviewed'] is True
        assert r['kept_count'] == 0
        assert r['added_count'] == 0


# ---------------------------------------------------------------------------
# _first_nonempty_line
# ---------------------------------------------------------------------------


class TestFirstNonemptyLine:
    def test_empty(self) -> None:
        assert _first_nonempty_line('') == ''
        assert _first_nonempty_line(None) == ''  # type: ignore[arg-type]

    def test_first_line(self) -> None:
        assert _first_nonempty_line('Pileus brown.\nStipe long.') \
            == 'Pileus brown.'

    def test_skips_leading_blanks(self) -> None:
        assert _first_nonempty_line('\n\n  \nActual first line.') \
            == 'Actual first line.'

    def test_truncates_long_line(self) -> None:
        text = 'X' * 200
        result = _first_nonempty_line(text, cap=20)
        assert len(result) == 20
        assert result.endswith('…')


# ---------------------------------------------------------------------------
# build_row
# ---------------------------------------------------------------------------


def _clean_treatment() -> Dict[str, Any]:
    return {
        'description': (
            'Pileus 3 cm wide, convex, brown when young. Stipe '
            'central, 5 mm, white. Lamellae adnate, brown. '
            'Spores 8 μm.'
        ),
        'diagnosis': 'Distinguished from X by brown lamellae.',
        'synthetic_nomenclature': False,
        # A clean treatment carries at least one deep-link anchor
        # (Trello #401 Phase 1 Commit C).  Omitting this would
        # fire §13:no_source_anchor and break the "clean → no
        # flags" contract of this test fixture.
        'source_anchors': [
            {'kind': 'pdf', 'page': '1', 'label': '1'},
        ],
    }


def _flagged_treatment() -> Dict[str, Any]:
    """A treatment that trips several detectors: mid-sentence
    start, multiple sp. nov., two Diagnosis: headers."""
    return {
        'description': (
            '; perithecia dispersa, hyaline, thin-walled.\n\n'
            'Diagnosis: Species A, sp. nov.\n\n'
            'Diagnosis: Species B, sp. nov.\n'
        ),
        'diagnosis': '',
        'synthetic_nomenclature': True,
    }


class TestBuildRow:
    def test_clean_treatment_no_flags(self) -> None:
        row = build_row(
            treatment_id='taxon_clean',
            treatment_doc=_clean_treatment(),
            status_doc={
                'status': 'success', 'annotation_count': 8,
            },
            reviewed_hand_ids=set(),
            merge_threshold=10,
        )
        assert row['treatment_id'] == 'taxon_clean'
        assert row['bootstrap_status'] == 'success'
        assert row['claude_annotation_count'] == 8
        assert row['reviewed'] is False
        assert row['predicted_issues'] == ''
        assert row['first_line'].startswith('Pileus')

    def test_flagged_treatment_multiple_flags(self) -> None:
        row = build_row(
            treatment_id='taxon_flagged',
            treatment_doc=_flagged_treatment(),
            status_doc={
                'status': 'success', 'annotation_count': 12,
            },
            reviewed_hand_ids=set(),
            merge_threshold=10,
        )
        # We're not asserting the full string — just that the
        # expected § tags all appear.  Their exact ordering is
        # tested in triage_signals_test.
        assert '§10:mid_sentence' in row['predicted_issues']
        assert '§6:multi_diagnosis' in row['predicted_issues']
        assert '§6:multi_sp_nov' in row['predicted_issues']
        assert '§2:synth_nomen' in row['predicted_issues']

    def test_missing_prose_doc(self) -> None:
        """If the status DB has a treatment but the prose doc is
        gone (or was never there), the row surfaces the gap via
        predicted_issues='no_prose_doc' and zeroed signals."""
        row = build_row(
            treatment_id='taxon_ghost',
            treatment_doc=None,
            status_doc={
                'status': 'error', 'annotation_count': 0,
            },
            reviewed_hand_ids=set(),
            merge_threshold=10,
        )
        assert row['predicted_issues'] == 'no_prose_doc'
        assert row['desc_length'] == 0
        assert row['bootstrap_status'] == 'error'

    def test_reviewer_action_flows_to_row(self) -> None:
        row = build_row(
            treatment_id='taxon_x',
            treatment_doc=_clean_treatment(),
            status_doc={
                'status': 'success',
                'annotation_count': 8,
                'reviewer_action': {
                    'reviewer': 'piggy',
                    'kept_count': 6,
                    'added_count': 1,
                    'deleted_count': 1,
                },
            },
            reviewed_hand_ids={'taxon_x'},
            merge_threshold=10,
        )
        assert row['reviewed'] is True
        assert row['reviewer'] == 'piggy'
        assert row['kept_count'] == 6
        assert row['added_count'] == 1
        assert row['deleted_count'] == 1

    def test_no_status_doc_unknown_status(self) -> None:
        row = build_row(
            treatment_id='taxon_new',
            treatment_doc=_clean_treatment(),
            status_doc=None,
            reviewed_hand_ids=set(),
            merge_threshold=10,
        )
        assert row['bootstrap_status'] == 'unknown'
        assert row['claude_annotation_count'] == 0


# ---------------------------------------------------------------------------
# sort_rows
# ---------------------------------------------------------------------------


class TestSortRows:
    def test_unreviewed_before_reviewed(self) -> None:
        rows = [
            {
                'treatment_id': 'a', 'reviewed': True,
                'merge_metric': 0, 'predicted_issues': '',
            },
            {
                'treatment_id': 'b', 'reviewed': False,
                'merge_metric': 0, 'predicted_issues': '',
            },
        ]
        result = sort_rows(rows)
        assert [r['treatment_id'] for r in result] == ['b', 'a']

    def test_higher_metric_first_within_unreviewed(self) -> None:
        rows = [
            {
                'treatment_id': 'lo', 'reviewed': False,
                'merge_metric': 3, 'predicted_issues': '',
            },
            {
                'treatment_id': 'hi', 'reviewed': False,
                'merge_metric': 42, 'predicted_issues': '',
            },
        ]
        result = sort_rows(rows)
        assert [r['treatment_id'] for r in result] == ['hi', 'lo']

    def test_flagged_before_clean_at_same_metric(self) -> None:
        rows = [
            {
                'treatment_id': 'clean', 'reviewed': False,
                'merge_metric': 5, 'predicted_issues': '',
            },
            {
                'treatment_id': 'flagged', 'reviewed': False,
                'merge_metric': 5,
                'predicted_issues': '§6:multi_diagnosis',
            },
        ]
        result = sort_rows(rows)
        assert [r['treatment_id'] for r in result] \
            == ['flagged', 'clean']

    def test_tiebreak_by_treatment_id(self) -> None:
        rows = [
            {
                'treatment_id': 'zzz', 'reviewed': False,
                'merge_metric': 0, 'predicted_issues': '',
            },
            {
                'treatment_id': 'aaa', 'reviewed': False,
                'merge_metric': 0, 'predicted_issues': '',
            },
        ]
        result = sort_rows(rows)
        assert [r['treatment_id'] for r in result] == ['aaa', 'zzz']


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------


class TestWriteCsv:
    def test_writes_header_and_rows(self, tmp_path: Path) -> None:
        out = tmp_path / 'triage.csv'
        rows = [
            build_row(
                treatment_id='taxon_a',
                treatment_doc=_clean_treatment(),
                status_doc={
                    'status': 'success', 'annotation_count': 5,
                },
                reviewed_hand_ids=set(),
                merge_threshold=10,
            ),
        ]
        write_csv(rows, out)
        with out.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == CSV_COLUMNS
            actual_rows = list(reader)
        assert len(actual_rows) == 1
        assert actual_rows[0]['treatment_id'] == 'taxon_a'
        assert actual_rows[0]['bootstrap_status'] == 'success'

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        """--output pointing at a not-yet-existing subdirectory
        should just work; parent dirs are auto-created."""
        out = tmp_path / 'nested' / 'triage.csv'
        write_csv([], out)
        assert out.exists()
