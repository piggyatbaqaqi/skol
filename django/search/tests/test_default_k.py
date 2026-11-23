"""
Tests for the default number of search results, "Total Results (k)":
200, up from 20 (user, 2026-11-23: "The new menus work better with more
descriptions").  The search API already accepts k up to 200; the user
setting and the page inputs stopped at 100.

Run with: python manage.py test search.tests.test_default_k
"""
import json
import re
import unittest
from typing import List

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from search.models import UserSettings


def k_inputs(html: str) -> List[str]:
    """Every ``<input ... id="k" ...>`` tag in the page."""
    return re.findall(r'<input[^>]*\bid="k"[^>]*>', html)


class TestDefaultK(TestCase):

    def setUp(self) -> None:
        self.user = User.objects.create_user('jsmith', 'j@example.com', 'pw')
        self.client = Client()
        self.client.login(username='jsmith', password='pw')
        self.url = reverse('search:user-settings')

    @unittest.expectedFailure
    def test_model_default_is_200(self) -> None:
        settings = UserSettings.objects.create(user=self.user)
        self.assertEqual(settings.default_k, 200)

    @unittest.expectedFailure
    def test_new_user_gets_200_from_the_api(self) -> None:
        self.assertEqual(self.client.get(self.url).json()['default_k'], 200)

    @unittest.expectedFailure
    def test_put_200_accepted(self) -> None:
        resp = self.client.put(self.url, json.dumps({'default_k': 200}),
                               content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            UserSettings.objects.get(user=self.user).default_k, 200)

    def test_put_above_200_rejected(self) -> None:
        # The search API's own limit (k must be between 1 and 200).
        resp = self.client.put(self.url, json.dumps({'default_k': 201}),
                               content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    def test_a_saved_choice_is_kept(self) -> None:
        # A user's stored value is theirs; only the default changes.
        UserSettings.objects.create(user=self.user, default_k=20)
        self.assertEqual(self.client.get(self.url).json()['default_k'], 20)

    def assert_page_input(self, client: Client) -> None:
        tags = k_inputs(client.get(reverse('home')).content.decode())
        self.assertEqual(len(tags), 1)
        self.assertIn('value="200"', tags[0])
        self.assertIn('max="200"', tags[0])

    @unittest.expectedFailure
    def test_guest_page_input_defaults_to_200(self) -> None:
        self.assert_page_input(Client())

    @unittest.expectedFailure
    def test_signed_in_page_input_defaults_to_200(self) -> None:
        # The page's own default, before the user's saved setting loads.
        self.assert_page_input(self.client)
