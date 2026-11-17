"""
Tests for GET /api/experiments/ and the user's current experiment.

Choosing an unapproved experiment in Settings saves it and reloads the
page; the "Enable all experiments" toggle resets on reload, so the
approved-only listing came back without the chosen experiment and the
selector went blank.  The listing now always includes the signed-in
user's current experiment (still marked ``approved: false``) and names
it as ``current``.

Run with: python manage.py test search.tests.test_experiment_list
"""
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from search.models import UserSettings

DOCS: List[Dict[str, Any]] = [
    {'_id': 'production', 'approved': True, 'status': 'deployed'},
    {'_id': 'production_v4_1', 'approved': False, 'status': 'draft'},
    {'_id': 'taxpub_v1', 'status': 'draft'},
]


def _all_docs() -> MagicMock:
    resp = MagicMock(status_code=200)
    resp.json.return_value = {'rows': [{'doc': d} for d in DOCS]}
    return resp


def _config(name: str) -> Optional[Dict[str, Any]]:
    return next((d for d in DOCS if d['_id'] == name), None)


class TestExperimentListIncludesCurrent(TestCase):

    def setUp(self) -> None:
        self.user = User.objects.create_user('niemand', 'n@example.com', 'pw')
        self.url = reverse('search:experiments')

    def _get(self, client: Client, **params: str) -> Dict[str, Any]:
        with patch('search.views.requests.get', return_value=_all_docs()), \
                patch('search.views.get_experiment_config',
                      side_effect=_config):
            resp = client.get(self.url, params)
        self.assertEqual(resp.status_code, 200)
        body: Dict[str, Any] = resp.json()
        return body

    def _signed_in(self, experiment: str) -> Client:
        UserSettings.objects.create(user=self.user,
                                    default_experiment=experiment)
        client = Client()
        client.login(username='niemand', password='pw')
        return client

    def test_unapproved_current_experiment_is_listed(self) -> None:
        body = self._get(self._signed_in('production_v4_1'))
        by_name = {e['name']: e for e in body['experiments']}
        self.assertEqual(sorted(by_name), ['production', 'production_v4_1'])
        self.assertFalse(by_name['production_v4_1']['approved'])
        self.assertEqual(body['current'], 'production_v4_1')

    def test_listed_once_when_showing_all(self) -> None:
        body = self._get(self._signed_in('production_v4_1'),
                         include_unapproved='true')
        names = [e['name'] for e in body['experiments']]
        self.assertEqual(names.count('production_v4_1'), 1)
        self.assertEqual(sorted(names),
                         ['production', 'production_v4_1', 'taxpub_v1'])
        self.assertEqual(body['current'], 'production_v4_1')

    def test_anonymous_sees_approved_only(self) -> None:
        body = self._get(Client())
        self.assertEqual([e['name'] for e in body['experiments']],
                         ['production'])
        self.assertEqual(body['current'], 'production')
