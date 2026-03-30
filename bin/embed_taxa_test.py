"""
Tests for embed_taxa helper functions.

These tests cover the pure DataFrame-manipulation logic that builds
per-section embedding inputs, without requiring CouchDB or Redis.

Run with: python -m pytest bin/embed_taxa_test.py -v
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embed_taxa import build_primary_descriptions, build_section_descriptions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skol_taxa_mock(records: List[Dict[str, Any]]):
    """Return a lightweight mock of SKOL_TAXA with .df populated."""

    class _Mock:
        pass

    mock = _Mock()
    mock.df = pd.DataFrame(records)

    # Replicate what get_descriptions() returns (minimal columns).
    def get_descriptions():
        df = mock.df
        filenames = df['ingest'].apply(
            lambda x: x.get('url', 'dummy') if isinstance(x, dict) else 'dummy'
        ) if 'ingest' in df.columns else 'dummy'
        result = pd.DataFrame({
            'source': 'SKOL_TAXA',
            'filename': filenames,
            'row': df.index,
            'description': df.get('description', pd.Series([''] * len(df))),
        })
        return result

    mock.get_descriptions = get_descriptions
    return mock


# ---------------------------------------------------------------------------
# build_primary_descriptions
# ---------------------------------------------------------------------------

class TestBuildPrimaryDescriptions(unittest.TestCase):
    """build_primary_descriptions: combines description + diagnosis."""

    def _mock(self, records):
        return _make_skol_taxa_mock(records)

    def test_no_diagnosis_column_returns_description_unchanged(self):
        """When no diagnosis column, primary text equals description."""
        mock = self._mock([
            {'description': 'Cap convex, white.', 'ingest': {'url': 'http://x'}},
            {'description': 'Gills free.', 'ingest': {'url': 'http://y'}},
        ])
        result = build_primary_descriptions(mock)
        self.assertEqual(result['description'].iloc[0], 'Cap convex, white.')
        self.assertEqual(result['description'].iloc[1], 'Gills free.')

    def test_diagnosis_appended_when_present(self):
        """Non-empty diagnosis is appended to description."""
        mock = self._mock([
            {
                'description': 'Cap convex.',
                'diagnosis': 'Differs from A. muscaria by smaller spores.',
                'ingest': {'url': 'http://x'},
            },
        ])
        result = build_primary_descriptions(mock)
        combined = result['description'].iloc[0]
        self.assertIn('Cap convex.', combined)
        self.assertIn('Differs from A. muscaria by smaller spores.', combined)
        self.assertIn('\n\n', combined)

    def test_null_diagnosis_not_appended(self):
        """Null/NaN diagnosis is skipped; description returned as-is."""
        mock = self._mock([
            {
                'description': 'Cap convex.',
                'diagnosis': None,
                'ingest': {'url': 'http://x'},
            },
        ])
        result = build_primary_descriptions(mock)
        self.assertEqual(result['description'].iloc[0], 'Cap convex.')
        self.assertNotIn('\n\n', result['description'].iloc[0])

    def test_empty_string_diagnosis_not_appended(self):
        """Empty-string diagnosis is treated the same as absent."""
        mock = self._mock([
            {
                'description': 'Cap convex.',
                'diagnosis': '',
                'ingest': {'url': 'http://x'},
            },
        ])
        result = build_primary_descriptions(mock)
        self.assertEqual(result['description'].iloc[0], 'Cap convex.')

    def test_mixed_rows_some_with_diagnosis(self):
        """Rows with and without diagnosis are handled independently."""
        mock = self._mock([
            {
                'description': 'Desc A.',
                'diagnosis': 'Diag A.',
                'ingest': {'url': 'http://a'},
            },
            {
                'description': 'Desc B.',
                'diagnosis': None,
                'ingest': {'url': 'http://b'},
            },
            {
                'description': 'Desc C.',
                'diagnosis': 'Diag C.',
                'ingest': {'url': 'http://c'},
            },
        ])
        result = build_primary_descriptions(mock)
        self.assertIn('Diag A.', result['description'].iloc[0])
        self.assertNotIn('\n\n', result['description'].iloc[1])
        self.assertIn('Diag C.', result['description'].iloc[2])

    def test_original_dataframe_not_mutated(self):
        """The input get_descriptions() result is not modified in place."""
        mock = self._mock([
            {
                'description': 'Cap convex.',
                'diagnosis': 'Diag text.',
                'ingest': {'url': 'http://x'},
            },
        ])
        original = mock.get_descriptions()
        original_desc = original['description'].iloc[0]
        build_primary_descriptions(mock)
        # Re-fetch to confirm the mock source is untouched.
        refetched = mock.get_descriptions()
        self.assertEqual(refetched['description'].iloc[0], original_desc)

    def test_returns_dataframe(self):
        """Return value is a pandas DataFrame."""
        mock = self._mock([{'description': 'x', 'ingest': {'url': 'http://x'}}])
        result = build_primary_descriptions(mock)
        self.assertIsInstance(result, pd.DataFrame)

    def test_result_has_description_column(self):
        """Result DataFrame has a 'description' column."""
        mock = self._mock([{'description': 'x', 'ingest': {'url': 'http://x'}}])
        result = build_primary_descriptions(mock)
        self.assertIn('description', result.columns)


# ---------------------------------------------------------------------------
# build_section_descriptions
# ---------------------------------------------------------------------------

class TestBuildSectionDescriptions(unittest.TestCase):
    """build_section_descriptions: builds per-section embedding inputs."""

    def _mock(self, records):
        return _make_skol_taxa_mock(records)

    def test_missing_column_returns_none(self):
        """Returns None when the requested section column is absent."""
        mock = self._mock([{'description': 'x', 'ingest': {'url': 'http://x'}}])
        result = build_section_descriptions(mock, 'distribution')
        self.assertIsNone(result)

    def test_all_null_values_returns_none(self):
        """Returns None when every row has null/empty for the section."""
        mock = self._mock([
            {'description': 'x', 'distribution': None, 'ingest': {'url': 'http://a'}},
            {'description': 'y', 'distribution': '', 'ingest': {'url': 'http://b'}},
        ])
        result = build_section_descriptions(mock, 'distribution')
        self.assertIsNone(result)

    def test_non_empty_rows_returned(self):
        """Returns DataFrame containing only rows with non-empty section text."""
        mock = self._mock([
            {'description': 'x', 'distribution': 'Europe.', 'ingest': {'url': 'http://a'}},
            {'description': 'y', 'distribution': None, 'ingest': {'url': 'http://b'}},
            {'description': 'z', 'distribution': 'Asia.', 'ingest': {'url': 'http://c'}},
        ])
        result = build_section_descriptions(mock, 'distribution')
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_description_column_contains_section_text(self):
        """The 'description' column in the result holds the section text."""
        mock = self._mock([
            {'description': 'desc', 'biology': 'Saprotrophic.', 'ingest': {'url': 'http://a'}},
        ])
        result = build_section_descriptions(mock, 'biology')
        self.assertIsNotNone(result)
        self.assertEqual(result['description'].iloc[0], 'Saprotrophic.')

    def test_returns_dataframe_when_data_present(self):
        """Return type is DataFrame (not None) when data is present."""
        mock = self._mock([
            {'description': 'x', 'distribution': 'Worldwide.', 'ingest': {'url': 'http://a'}},
        ])
        result = build_section_descriptions(mock, 'distribution')
        self.assertIsInstance(result, pd.DataFrame)

    def test_metadata_columns_preserved(self):
        """Metadata columns from get_descriptions() (e.g., row) are retained."""
        mock = self._mock([
            {'description': 'x', 'distribution': 'Worldwide.', 'ingest': {'url': 'http://a'}},
        ])
        result = build_section_descriptions(mock, 'distribution')
        self.assertIsNotNone(result)
        self.assertIn('row', result.columns)

    def test_biology_column(self):
        """Works for 'biology' section."""
        mock = self._mock([
            {'description': 'x', 'biology': 'Host: oak.', 'ingest': {'url': 'http://a'}},
        ])
        result = build_section_descriptions(mock, 'biology')
        self.assertIsNotNone(result)
        self.assertEqual(result['description'].iloc[0], 'Host: oak.')


if __name__ == '__main__':
    unittest.main()
