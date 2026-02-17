#!/usr/bin/env python3
"""
Tests for the daily admin report script.

Run with: cd bin && python -m pytest daily_report_test.py -v
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

# Bootstrap Django before importing anything else
DJANGO_DIR = str(Path(__file__).resolve().parent.parent / 'django')
sys.path.insert(0, DJANGO_DIR)
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skolweb.settings')

import django  # noqa: E402
django.setup()

import pytest  # noqa: E402
from django.contrib.auth.models import User  # noqa: E402
from django.test import TestCase  # noqa: E402

from search.models import Collection, SearchHistory, UserSettings  # noqa: E402

import daily_report  # noqa: E402


@pytest.fixture
def site_url():
    return 'https://example.com/skol'


@pytest.fixture
def now():
    return datetime.now(timezone.utc)


@pytest.fixture
def since(now):
    return now - timedelta(days=1)


# ---------------------------------------------------------------------------
# get_site_url
# ---------------------------------------------------------------------------

class TestGetSiteUrl:
    def test_from_env(self):
        with patch.dict(os.environ, {'SKOL_SITE_URL': 'https://test.com/skol'}):
            assert daily_report.get_site_url() == 'https://test.com/skol'

    def test_strips_trailing_slash(self):
        with patch.dict(os.environ, {'SKOL_SITE_URL': 'https://test.com/skol/'}):
            assert daily_report.get_site_url() == 'https://test.com/skol'

    def test_fallback_to_force_script_name(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('SKOL_SITE_URL', None)
            with patch.object(daily_report.settings, 'FORCE_SCRIPT_NAME', '/skol'):
                url = daily_report.get_site_url()
                assert '/skol' in url


# ---------------------------------------------------------------------------
# Section functions (use Django TestCase for DB access)
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestSectionFlaggedCollections:
    def test_empty_when_none_flagged(self, since, site_url):
        result = daily_report.section_flagged_collections(since, site_url)
        assert result == ''

    def test_shows_flagged_collection(self, since, site_url):
        user = User.objects.create_user('owner1', 'o@test.com', 'pass')
        Collection.objects.create(
            collection_id=111111111,
            name='Flagged One',
            owner=user,
            flagged_by=[1, 2],
        )
        result = daily_report.section_flagged_collections(since, site_url)
        assert 'Flagged One' in result
        assert '111111111' in result
        assert 'Flags: 2' in result

    def test_excludes_hidden_flagged(self, since, site_url):
        user = User.objects.create_user('owner2', 'o2@test.com', 'pass')
        Collection.objects.create(
            collection_id=222222222,
            name='Hidden Flagged',
            owner=user,
            flagged_by=[1],
            hidden=True,
        )
        result = daily_report.section_flagged_collections(since, site_url)
        assert result == ''


@pytest.mark.django_db
class TestSectionFlaggedComments:
    def test_couchdb_error_handled(self, since, site_url):
        with patch('search.comment_service.get_comments_db', side_effect=Exception('conn refused')):
            result = daily_report.section_flagged_comments(since, site_url)
            assert 'CouchDB error' in result

    def test_empty_when_no_flagged(self, since, site_url):
        mock_db = MagicMock()
        mock_db.view.return_value = []
        with patch('search.comment_service.get_comments_db', return_value=mock_db):
            with patch('search.comment_service.ensure_design_docs'):
                result = daily_report.section_flagged_comments(since, site_url)
                assert result == ''


@pytest.mark.django_db
class TestSectionNewUsers:
    def test_empty_when_no_new_users(self, site_url):
        since = datetime.now(timezone.utc) + timedelta(days=1)
        result = daily_report.section_new_users(since, site_url)
        assert result == ''

    def test_shows_new_user(self, since, site_url):
        User.objects.create_user('newbie', 'new@test.com', 'pass')
        very_old = datetime.now(timezone.utc) - timedelta(days=365)
        result = daily_report.section_new_users(very_old, site_url)
        assert 'newbie' in result
        assert 'new@test.com' in result


@pytest.mark.django_db
class TestSectionNewCollections:
    def test_empty_when_none(self, site_url):
        since = datetime.now(timezone.utc) + timedelta(days=1)
        result = daily_report.section_new_collections(since, site_url)
        assert result == ''

    def test_shows_new_collection(self, since, site_url):
        user = User.objects.create_user('owner3', 'o3@test.com', 'pass')
        Collection.objects.create(
            collection_id=333333333,
            name='Fresh Collection',
            owner=user,
        )
        very_old = datetime.now(timezone.utc) - timedelta(days=365)
        result = daily_report.section_new_collections(very_old, site_url)
        assert 'Fresh Collection' in result
        assert '333333333' in result


@pytest.mark.django_db
class TestSectionSearchActivity:
    def test_empty_when_no_searches(self, since, site_url):
        result = daily_report.section_search_activity(since, site_url)
        assert result == ''

    def test_shows_search_stats(self, site_url):
        user = User.objects.create_user('searcher', 'search@test.com', 'pass')
        c = Collection.objects.create(
            collection_id=444444444, name='Search Col', owner=user,
        )
        very_old = datetime.now(timezone.utc) - timedelta(days=365)
        SearchHistory.objects.create(
            collection=c, event_type='search', prompt='test',
            embedding_name='v1', result_references=[], result_count=3,
        )
        result = daily_report.section_search_activity(very_old, site_url)
        assert 'Total searches: 1' in result
        assert 'Search Col' in result


@pytest.mark.django_db
class TestSectionHiddenCollections:
    def test_empty_when_none_hidden(self, since, site_url):
        result = daily_report.section_hidden_collections(since, site_url)
        assert result == ''

    def test_shows_hidden(self, since, site_url):
        user = User.objects.create_user('owner4', 'o4@test.com', 'pass')
        Collection.objects.create(
            collection_id=555555555,
            name='Concealed',
            owner=user,
            hidden=True,
            hidden_by=user,
        )
        result = daily_report.section_hidden_collections(since, site_url)
        assert 'Concealed' in result
        assert 'Hidden by: owner4' in result


@pytest.mark.django_db
class TestSectionSystemTotals:
    def test_shows_counts(self, since, site_url):
        result = daily_report.section_system_totals(since, site_url)
        assert 'Active users:' in result
        assert 'Total collections:' in result
        assert 'All-time searches:' in result


# ---------------------------------------------------------------------------
# Full report generation
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestGenerateReport:
    def test_report_has_header_and_footer(self):
        with patch.dict(os.environ, {'SKOL_SITE_URL': 'https://test.com/skol'}):
            report = daily_report.generate_report(days=7)
            assert 'SKOL Daily Admin Report' in report
            assert 'Generated:' in report
            assert 'Period: last 7 day(s)' in report
            assert 'Admin: https://test.com/skol/admin/' in report


# ---------------------------------------------------------------------------
# Email delivery
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestGetOptedInRecipients:
    def test_returns_opted_in_emails(self):
        u1 = User.objects.create_user('admin1', 'admin1@test.com', 'pass')
        u2 = User.objects.create_user('admin2', 'admin2@test.com', 'pass')
        User.objects.create_user('nope', 'nope@test.com', 'pass')

        UserSettings.objects.create(user=u1, receive_admin_summary=True)
        UserSettings.objects.create(user=u2, receive_admin_summary=True)

        recipients = daily_report.get_opted_in_recipients()
        assert 'admin1@test.com' in recipients
        assert 'admin2@test.com' in recipients
        assert 'nope@test.com' not in recipients

    def test_excludes_users_without_email(self):
        u = User.objects.create_user('noemail', '', 'pass')
        UserSettings.objects.create(user=u, receive_admin_summary=True)

        recipients = daily_report.get_opted_in_recipients()
        assert len(recipients) == 0


class TestSendReport:
    def test_sends_to_all_recipients(self):
        with patch('daily_report.EmailMessage') as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            daily_report.send_report('test report', ['a@b.com', 'c@d.com'])
            assert mock_instance.send.call_count == 2

    def test_handles_send_failure(self, capsys):
        with patch('daily_report.EmailMessage') as mock_cls:
            mock_instance = MagicMock()
            mock_instance.send.side_effect = Exception('SMTP error')
            mock_cls.return_value = mock_instance
            daily_report.send_report('test report', ['fail@test.com'])
            captured = capsys.readouterr()
            assert 'Failed to send' in captured.err
