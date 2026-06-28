"""Tests for bin/brat_ingest CLI helpers.

The pure diff logic is tested in
treatments_to_structured/brat_ingest_test.py.  This file covers
the CLI-side glue: reviewed-DB resolution, .ann file discovery
(--ann-dir / --ann-file / --doc-id filter), candidate fetch from
CouchDB.
"""

import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brat_ingest import (  # type: ignore[import]  # noqa: E402
    discover_ann_files,
    fetch_candidate_anns_for_treatment,
    resolve_reviewed_db_name,
    reviewer_label_from_env,
)


# ---------------------------------------------------------------------------
# resolve_reviewed_db_name
# ---------------------------------------------------------------------------


class TestResolveReviewedDbName:
    """Same fallback shape as candidate / status DBs."""

    def test_uses_databases_features_reviewed_when_set(self) -> None:
        exp = {
            'databases': {
                'features_reviewed':
                    'skol_exp_X_features_reviewed',
            },
        }
        warn = io.StringIO()
        name = resolve_reviewed_db_name(
            'production_v4', exp, verbosity=1, warn_stream=warn,
        )
        assert name == 'skol_exp_X_features_reviewed'
        assert warn.getvalue() == ''

    def test_falls_back_to_naming_convention(self) -> None:
        exp = {'databases': {}}
        warn = io.StringIO()
        name = resolve_reviewed_db_name(
            'production_v4', exp, verbosity=1, warn_stream=warn,
        )
        # 02_50 slot matches candidate + status — keeps the triple
        # together in Fauxton.
        assert name == (
            'skol_exp_production_v4_02_50_features_reviewed'
        )
        assert 'NOTE' in warn.getvalue()

    def test_silent_at_verbosity_zero(self) -> None:
        warn = io.StringIO()
        resolve_reviewed_db_name(
            'x', {'databases': {}}, verbosity=0, warn_stream=warn,
        )
        assert warn.getvalue() == ''


# ---------------------------------------------------------------------------
# discover_ann_files
# ---------------------------------------------------------------------------


class TestDiscoverAnnFiles:
    """Resolve the (treatment_id, path) pairs to process."""

    def test_neither_source_raises(self) -> None:
        with pytest.raises(ValueError) as exc:
            discover_ann_files(None, None, None)
        assert '--ann-dir' in str(exc.value)
        assert '--ann-file' in str(exc.value)

    def test_both_sources_raises(self) -> None:
        with pytest.raises(ValueError) as exc:
            discover_ann_files(
                '/tmp/dir', '/tmp/x.ann', None,
            )
        assert 'mutually exclusive' in str(exc.value)

    def test_ann_file_missing_raises(self) -> None:
        with pytest.raises(ValueError) as exc:
            discover_ann_files(
                None, '/tmp/this_does_not_exist_anywhere.ann',
                None,
            )
        assert 'not found' in str(exc.value)

    def test_ann_file_returns_single_pair(
        self, tmp_path: Path,
    ) -> None:
        f = tmp_path / 'taxon_abc.ann'
        f.write_text('T1\tPileus 0 5\thello\n')
        pairs = discover_ann_files(None, str(f), None)
        assert pairs == [('taxon_abc', str(f))]

    def test_ann_dir_scans_taxon_files(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / 'taxon_a.ann').write_text('')
        (tmp_path / 'taxon_b.ann').write_text('')
        # Non-.ann files are ignored
        (tmp_path / 'README.md').write_text('')
        (tmp_path / 'notes.txt').write_text('')
        pairs = discover_ann_files(str(tmp_path), None, None)
        ids = sorted(tid for tid, _ in pairs)
        assert ids == ['taxon_a', 'taxon_b']

    def test_ann_dir_filter_by_doc_id(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / 'taxon_a.ann').write_text('')
        (tmp_path / 'taxon_b.ann').write_text('')
        (tmp_path / 'taxon_c.ann').write_text('')
        pairs = discover_ann_files(
            str(tmp_path), None, ['taxon_a', 'taxon_c'],
        )
        ids = sorted(tid for tid, _ in pairs)
        assert ids == ['taxon_a', 'taxon_c']

    def test_ann_dir_empty_raises(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError) as exc:
            discover_ann_files(str(tmp_path), None, None)
        assert 'no .ann files' in str(exc.value)

    def test_ann_dir_filter_matches_nothing_raises(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / 'taxon_a.ann').write_text('')
        with pytest.raises(ValueError) as exc:
            discover_ann_files(
                str(tmp_path), None, ['taxon_xyz'],
            )
        assert '--doc-id filter' in str(exc.value)

    def test_ann_dir_not_a_directory_raises(self) -> None:
        with pytest.raises(ValueError) as exc:
            discover_ann_files(
                '/this/does/not/exist/dir', None, None,
            )
        assert 'not a directory' in str(exc.value)


# ---------------------------------------------------------------------------
# fetch_candidate_anns_for_treatment
# ---------------------------------------------------------------------------


class _FakeRow:
    def __init__(self, doc: Dict[str, Any]) -> None:
        self.doc = doc


class _FakeView:
    def __init__(self, rows: List[_FakeRow]) -> None:
        self.rows = rows


class _FakeCandidateDb:
    """Stand-in for couchdb.Database supporting the view+include_docs
    call shape the production code uses."""

    def __init__(self, docs: List[Dict[str, Any]]) -> None:
        self.docs = docs

    def view(self, _name: str, **kwargs: Any) -> _FakeView:
        startkey = kwargs['startkey']
        endkey = kwargs['endkey']
        rows = []
        for d in self.docs:
            _id = d.get('_id', '')
            if startkey <= _id <= endkey:
                rows.append(_FakeRow(d))
        return _FakeView(rows)


class TestFetchCandidateAnns:
    def test_returns_only_treatment_matching_docs(self) -> None:
        db = _FakeCandidateDb([
            {'_id': 'taxon_a:Pileus:0', 'treatment_id': 'taxon_a'},
            {'_id': 'taxon_a:Stipe:100', 'treatment_id': 'taxon_a'},
            {'_id': 'taxon_b:Pileus:0', 'treatment_id': 'taxon_b'},
        ])
        result = fetch_candidate_anns_for_treatment(db, 'taxon_a')
        ids = sorted(d['_id'] for d in result)
        assert ids == ['taxon_a:Pileus:0', 'taxon_a:Stipe:100']

    def test_returns_empty_when_no_match(self) -> None:
        db = _FakeCandidateDb([
            {'_id': 'taxon_a:Pileus:0', 'treatment_id': 'taxon_a'},
        ])
        result = fetch_candidate_anns_for_treatment(db, 'taxon_z')
        assert result == []

    def test_returns_full_doc_dicts(self) -> None:
        """The diff needs more than just _id — it needs model,
        created_at, _rev for the kept docs' provenance pull-through
        and overwrite-with-rev semantics."""
        db = _FakeCandidateDb([{
            '_id': 'taxon_a:Pileus:0',
            'treatment_id': 'taxon_a',
            'feature_label': 'Pileus',
            'field': 'description',
            'start': 0, 'end': 5,
            'model': 'claude-opus-4-7',
            'created_at': '2026-06-28T20:46:16Z',
            '_rev': '1-abc',
        }])
        result = fetch_candidate_anns_for_treatment(db, 'taxon_a')
        assert len(result) == 1
        assert result[0]['model'] == 'claude-opus-4-7'
        assert result[0]['_rev'] == '1-abc'


# ---------------------------------------------------------------------------
# reviewer_label_from_env
# ---------------------------------------------------------------------------


class TestReviewerLabelFromEnv:
    def test_format_is_user_at_host(self) -> None:
        label = reviewer_label_from_env()
        # Format check; can't assert exact values since they
        # depend on the test runner's environment.
        assert '@' in label
        assert len(label) > 1
