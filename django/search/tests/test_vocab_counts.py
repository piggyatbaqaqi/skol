"""
Tests for Proposal 1 of docs/plans/vocab-tree-narrowing.md: the
children view orders and filters by per-node document support, read
from the path -> count hash stored beside the tree at
``<menus key>:counts``, using the user's ``vocab_min_support`` setting
(default 2).

A tree without its counts hash is a bug, not an old format to fall back
on: the children view reports it, and the build endpoint treats the tree
as not built, so the widget's build-and-retry path repairs it.

Run with: python manage.py test search.tests.test_vocab_counts
"""
import fnmatch
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest.mock import MagicMock, mock_open, patch

import redis
from django.contrib.auth.models import User
from django.test import Client, SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from search.models import UserSettings

MENUS = 'skol:ui:menus_production_v4_1'
COUNTS = MENUS + ':counts'

TREE: Dict[str, Any] = {
    'pileus': {'color': {'brown': {}, 'mauve': {}}, 'odour': {}},
    'stipe': {'surface': {}},
    'ascus': {},
}

# Field = JSON-encoded display path, value = documents containing it.
COUNT_MAP: Dict[str, int] = {
    '["pileus"]': 5,
    '["pileus", "color"]': 4,
    '["pileus", "color", "brown"]': 3,
    '["pileus", "color", "mauve"]': 1,
    '["pileus", "odour"]': 1,
    '["stipe"]': 7,
    '["stipe", "surface"]': 1,
    '["ascus"]': 1,
}


class FakeRedis:
    """Strings and hashes, with real Redis's WRONGTYPE on GET of a hash."""

    def __init__(self, data: Optional[Dict[str, str]] = None,
                 hashes: Optional[Dict[str, Dict[str, int]]] = None) -> None:
        self.data: Dict[str, str] = dict(data or {})
        self.hashes: Dict[str, Dict[str, int]] = {
            k: dict(v) for k, v in (hashes or {}).items()}

    def get(self, key: str) -> Optional[str]:
        if key in self.hashes:
            raise redis.exceptions.ResponseError(
                'WRONGTYPE Operation against a key holding the wrong '
                'kind of value')
        return self.data.get(key)

    def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if k in self.data or k in self.hashes)

    def hmget(self, key: str, fields: List[str]) -> List[Optional[str]]:
        h = self.hashes.get(key, {})
        return [None if f not in h else str(h[f]) for f in fields]

    def hget(self, key: str, field: str) -> Optional[str]:
        value = self.hashes.get(key, {}).get(field)
        return None if value is None else str(value)

    def hgetall(self, key: str) -> Dict[str, str]:
        return {f: str(v) for f, v in self.hashes.get(key, {}).items()}

    def type(self, key: str) -> str:
        if key in self.hashes:
            return 'hash'
        return 'string' if key in self.data else 'none'

    def scan_iter(self, match: str = '*', count: int = 0) -> Iterator[str]:
        keys = list(self.data) + list(self.hashes)
        return iter([k for k in keys if fnmatch.fnmatch(k, match)])


def _tree_json() -> str:
    return json.dumps({'version': MENUS, 'created_at': 'now', 'tree': TREE,
                       'stats': {'total_nodes': 8}})


def _store(with_counts: bool = True) -> FakeRedis:
    return FakeRedis({MENUS: _tree_json()},
                     {COUNTS: COUNT_MAP} if with_counts else {})


def _exp() -> Tuple[str, Dict[str, Any]]:
    return ('production', {'redis_keys': {'menus': MENUS}})


def _children(client: Client, store: FakeRedis, path: str = '') -> Any:
    params = {'path': path} if path else {}
    with patch('search.views.get_redis_client', return_value=store), \
            patch('search.views.get_user_experiment', return_value=_exp()):
        return client.get(reverse('search:vocab-tree-children'), params)


class TestChildrenByFrequencyAnonymous(SimpleTestCase):
    """Anonymous users get the default minimum support of 2."""

    def test_ordered_by_count_and_filtered(self) -> None:
        body = _children(Client(), _store()).json()
        self.assertEqual(body['children'], ['stipe', 'pileus'])
        self.assertEqual(body['counts'], {'stipe': 7, 'pileus': 5})
        self.assertEqual(body['count'], 2)

    def test_nested_level_filtered(self) -> None:
        body = _children(Client(), _store(), 'pileus.color').json()
        self.assertEqual(body['children'], ['brown'])
        self.assertEqual(body['counts'], {'brown': 3})

    def test_has_grandchildren_ignores_filtered_grandchildren(self) -> None:
        body = _children(Client(), _store()).json()
        # stipe's only child, surface, is below 2.
        self.assertEqual(body['has_grandchildren'],
                         {'stipe': False, 'pileus': True})

    def test_has_grandchildren_one_level_down(self) -> None:
        body = _children(Client(), _store(), 'pileus').json()
        self.assertEqual(body['children'], ['color'])
        self.assertEqual(body['has_grandchildren'], {'color': True})


class TestChildrenUseTheUsersSetting(TestCase):
    """A signed-in user's ``vocab_min_support`` replaces the default."""

    def setUp(self) -> None:
        self.user = User.objects.create_user('jsmith', 'j@example.com', 'pw')
        self.client = Client()
        self.client.login(username='jsmith', password='pw')

    def test_higher_setting_filters_more(self) -> None:
        UserSettings.objects.create(user=self.user, vocab_min_support=6)
        body = _children(self.client, _store()).json()
        self.assertEqual(body['children'], ['stipe'])

    def test_user_without_settings_gets_default(self) -> None:
        body = _children(self.client, _store()).json()
        self.assertEqual(body['children'], ['stipe', 'pileus'])


class TestMissingCountsIsAnError(SimpleTestCase):
    """10a: a tree without its counts hash is reported, not served
    alphabetical and unfiltered."""

    def test_children_view_returns_500_naming_the_key(self) -> None:
        with self.assertLogs('search.views', level='ERROR'):
            resp = _children(Client(), _store(with_counts=False))
        self.assertEqual(resp.status_code, 500)
        self.assertIn(COUNTS, resp.json()['error'])
        self.assertIn('rebuild', resp.json()['error'].lower())


class TestBuildTreatsMissingCountsAsNotBuilt(SimpleTestCase):
    """10b: the widget POSTs to build/ when children/ fails.  A tree
    without counts must be rebuilt, not answered with 'exists'."""

    def _get(self, store: FakeRedis) -> Any:
        with patch('search.views.get_redis_client', return_value=store), \
                patch('search.views.get_user_experiment',
                      return_value=_exp()):
            return Client().get(reverse('search:vocab-tree-build'))

    def test_get_exists_with_counts(self) -> None:
        """Already passes: a tree with its counts exists."""
        self.assertEqual(self._get(_store()).json()['status'], 'exists')

    def test_get_not_found_without_counts(self) -> None:
        resp = self._get(_store(with_counts=False))
        self.assertEqual(resp.json()['status'], 'not_found')

    def test_post_rebuilds_without_counts(self) -> None:
        with tempfile.TemporaryDirectory() as bin_dir:
            (Path(bin_dir) / 'build_vocab_tree').touch()
            with override_settings(SKOL_BIN_PATH=bin_dir), \
                    patch('search.views.get_redis_client',
                          return_value=_store(with_counts=False)), \
                    patch('search.views.get_user_experiment',
                          return_value=_exp()), \
                    patch('search.views.open', mock_open(), create=True), \
                    patch('subprocess.Popen',
                          return_value=MagicMock(pid=1234)) as popen:
                resp = Client().post(reverse('search:vocab-tree-build'),
                                     '{}', content_type='application/json')
        self.assertEqual(resp.status_code, 202)
        popen.assert_called_once()


class TestVersionsSkipTheCountsHash(SimpleTestCase):
    """11: the counts hash matches ``skol:ui:menus_*``; GET on it raises
    WRONGTYPE, which must not take down the versions listing."""

    def test_listing_survives_and_omits_the_hash(self) -> None:
        with patch('search.views.get_redis_client', return_value=_store()):
            resp = Client().get(reverse('search:vocab-tree-versions'))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual([v['key'] for v in resp.json()['versions']],
                         [MENUS])


class TestVocabMinSupportSetting(TestCase):
    """12: ``UserSettings.vocab_min_support``, default 2, never below the
    build floor of 2, read and written through /api/user-settings/."""

    def setUp(self) -> None:
        self.user = User.objects.create_user('jsmith', 'j@example.com', 'pw')
        self.client = Client()
        self.client.login(username='jsmith', password='pw')
        self.url = reverse('search:user-settings')

    def test_model_default_is_two(self) -> None:
        settings = UserSettings.objects.create(user=self.user)
        self.assertEqual(settings.vocab_min_support, 2)

    def test_get_exposes_it(self) -> None:
        self.assertEqual(self.client.get(self.url).json()['vocab_min_support'],
                         2)

    def test_put_updates_it(self) -> None:
        resp = self.client.put(self.url,
                               json.dumps({'vocab_min_support': 4}),
                               content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['vocab_min_support'], 4)
        self.assertEqual(
            UserSettings.objects.get(user=self.user).vocab_min_support, 4)

    def test_put_below_the_floor_rejected(self) -> None:
        resp = self.client.put(self.url,
                               json.dumps({'vocab_min_support': 1}),
                               content_type='application/json')
        self.assertEqual(resp.status_code, 400)
