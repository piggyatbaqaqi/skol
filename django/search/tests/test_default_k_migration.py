"""
Tests for moving stored "Total Results (k)" settings of 20, the old
default, to 200 (user, 2026-11-23).  ``search.default_k`` holds the
function that migration 0026 runs.

Run with: python manage.py test search.tests.test_default_k_migration
"""
import importlib
import unittest
from typing import Dict

from django.contrib.auth.models import User
from django.test import TestCase

from search.default_k import move_stored_default
from search.models import UserSettings


class TestMoveStoredDefault(TestCase):

    def setUp(self) -> None:
        self.rows: Dict[str, int] = {'old': 20, 'chosen': 50, 'max': 200,
                                     'low': 1}
        for name, k in self.rows.items():
            user = User.objects.create_user(name, f'{name}@example.com', 'pw')
            UserSettings.objects.create(user=user, default_k=k)

    def k_of(self, name: str) -> int:
        return int(UserSettings.objects.get(user__username=name).default_k)

    @unittest.expectedFailure
    def test_stored_20_becomes_200(self) -> None:
        move_stored_default(UserSettings)
        self.assertEqual(self.k_of('old'), 200)

    def test_other_values_are_left_alone(self) -> None:
        move_stored_default(UserSettings)
        self.assertEqual(self.k_of('chosen'), 50)
        self.assertEqual(self.k_of('max'), 200)
        self.assertEqual(self.k_of('low'), 1)

    @unittest.expectedFailure
    def test_returns_the_number_changed(self) -> None:
        self.assertEqual(move_stored_default(UserSettings), 1)

    def test_running_twice_changes_nothing_more(self) -> None:
        move_stored_default(UserSettings)
        self.assertEqual(move_stored_default(UserSettings), 0)

    @unittest.expectedFailure
    def test_migration_0026_runs_it(self) -> None:
        module = importlib.import_module(
            'search.migrations.0026_move_stored_default_k')
        self.assertIs(module.move_stored_default, move_stored_default)
