"""
Tests that the vocab-tree views take the tree only from the user's
experiment record: ``redis_keys.menus`` names the Redis key holding the
tree, and there is no ``skol:ui:menus_latest`` fallback.

See docs/plans/experiment-record-canonical.md, Step 1.

Run with: python manage.py test search.tests.test_vocab_menus
"""
import fnmatch
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple
from unittest import expectedFailure
from unittest.mock import MagicMock, mock_open, patch

from django.test import Client, SimpleTestCase, override_settings
from django.urls import reverse

# Strict xfail until Step 1 is implemented: unittest reports an
# unexpected success as a failure.
RECORD_MENUS_NOT_YET = expectedFailure

NEW_TREE = 'skol:ui:menus_production_v4_1'
OLD_TREE = 'skol:ui:menus_2026_11_12_22_00'
LATEST = 'skol:ui:menus_latest'


class FakeRedis:
    """The few Redis commands the vocab views use."""

    def __init__(self, data: Optional[Dict[str, str]] = None) -> None:
        self.data: Dict[str, str] = dict(data or {})

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self.data[key] = value
        return True

    def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if k in self.data)

    def scan_iter(self, match: str = '*', count: int = 0) -> Iterator[str]:
        return iter([k for k in self.data if fnmatch.fnmatch(k, match)])


def _tree(version: str, top: Dict[str, Any]) -> str:
    return json.dumps({'version': version, 'created_at': 'now',
                       'tree': top, 'stats': {'total_nodes': len(top)}})


def _store() -> FakeRedis:
    """menus_latest -> an old cron-built tree; the v4.1 tree stored
    under its own key, as on production after the deploy."""
    return FakeRedis({
        LATEST: OLD_TREE,
        OLD_TREE: _tree(OLD_TREE, {'Spores': {'old': {}}}),
        NEW_TREE: _tree(NEW_TREE, {'spores': {'ellipsoid': {}},
                                   'asci': {}}),
    })


def _exp(menus: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    keys = {} if menus is None else {'menus': menus}
    return ('production', {'redis_keys': keys})


class _ViewCase(SimpleTestCase):
    """Patch the Redis client and the user's experiment, then GET."""

    def _get(self, name: str, menus: Optional[str], **params: str) -> Any:
        with patch('search.views.get_redis_client', return_value=_store()), \
                patch('search.views.get_user_experiment',
                      return_value=_exp(menus)):
            return Client().get(reverse(name), params)


@RECORD_MENUS_NOT_YET
class TestVocabTreeServesRecordKey(_ViewCase):
    """The deploy outage: the record named the tree key, the view used
    it as a pointer, and returned 404 with the tree JSON as the key."""

    def test_record_tree_key_serves_that_tree(self) -> None:
        resp = self._get('search:vocab-tree', NEW_TREE)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['version'], NEW_TREE)


@RECORD_MENUS_NOT_YET
class TestVocabTreeNoFallback(_ViewCase):
    """A record without a menus key must not silently serve whatever
    menus_latest points at."""

    def test_missing_menus_key_is_404(self) -> None:
        resp = self._get('search:vocab-tree', None)
        self.assertEqual(resp.status_code, 404)
        self.assertIn('production', resp.json()['error'])


class TestVocabTreeMissingTree(_ViewCase):
    """Already passes: a menus key naming no stored tree is a 404."""

    def test_record_key_without_tree_is_404(self) -> None:
        resp = self._get('search:vocab-tree', 'skol:ui:menus_gone')
        self.assertEqual(resp.status_code, 404)


class TestVocabTreeExplicitVersion(_ViewCase):
    """Already passes: ?version= still selects a specific tree."""

    def test_version_param_wins(self) -> None:
        resp = self._get('search:vocab-tree', NEW_TREE,
                         version='2026_11_12_22_00')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['version'], OLD_TREE)


@RECORD_MENUS_NOT_YET
class TestVocabTreeChildrenUsesRecord(_ViewCase):
    """children/ read menus_latest regardless of the experiment, so
    the dropdown children could come from a different tree."""

    def test_children_come_from_the_record_tree(self) -> None:
        resp = self._get('search:vocab-tree-children', NEW_TREE)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['children'], ['asci', 'spores'])


@RECORD_MENUS_NOT_YET
class TestBuildVocabTreeGetUsesRecord(_ViewCase):
    def test_exists_for_record_tree_key(self) -> None:
        resp = self._get('search:vocab-tree-build', NEW_TREE)
        self.assertEqual(resp.json()['status'], 'exists')
        self.assertEqual(resp.json()['redis_key'], NEW_TREE)

    def test_not_found_without_menus_key(self) -> None:
        resp = self._get('search:vocab-tree-build', None)
        self.assertEqual(resp.json()['status'], 'not_found')


@RECORD_MENUS_NOT_YET
class TestVocabTreeVersionsHasNoLatest(_ViewCase):
    def test_no_latest_field_and_no_pointer_listed(self) -> None:
        resp = self._get('search:vocab-tree-versions', NEW_TREE)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertNotIn('latest', body)
        self.assertEqual(sorted(v['key'] for v in body['versions']),
                         [OLD_TREE, NEW_TREE])


@RECORD_MENUS_NOT_YET
class TestBuildVocabTreePostUsesExperiment(SimpleTestCase):
    """An API-triggered build ran ``--db X`` only: it wrote a
    timestamped key and moved menus_latest instead of rebuilding the
    experiment's tree."""

    def test_post_runs_build_with_the_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as bin_dir:
            (Path(bin_dir) / 'build_vocab_tree').touch()
            with override_settings(SKOL_BIN_PATH=bin_dir), \
                    patch('search.views.get_redis_client',
                          return_value=FakeRedis()), \
                    patch('search.views.get_user_experiment',
                          return_value=_exp(NEW_TREE)), \
                    patch('search.views.open', mock_open(), create=True), \
                    patch('subprocess.Popen',
                          return_value=MagicMock(pid=1234)) as popen:
                resp = Client().post(reverse('search:vocab-tree-build'),
                                     '{}', content_type='application/json')
        self.assertEqual(resp.status_code, 202)
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[cmd.index('--experiment') + 1], 'production')
        self.assertNotIn('--db', cmd)
