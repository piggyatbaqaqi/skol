"""Tests for the contact app."""
import json
from unittest.mock import patch, MagicMock

from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model

from .forms import FeedbackForm, ContactForm
from .github import format_feedback_issue


User = get_user_model()


class FeedbackFormTest(TestCase):
    """Tests for FeedbackForm."""

    def test_browser_info_field_exists(self) -> None:
        """Test that browser_info hidden field exists on form."""
        form = FeedbackForm()
        self.assertIn('browser_info', form.fields)

    def test_browser_info_field_is_hidden(self) -> None:
        """Test that browser_info field uses HiddenInput widget."""
        form = FeedbackForm()
        widget = form.fields['browser_info'].widget
        self.assertEqual(widget.input_type, 'hidden')

    def test_browser_info_not_required(self) -> None:
        """Test that browser_info is optional."""
        form = FeedbackForm()
        self.assertFalse(form.fields['browser_info'].required)

    def test_form_valid_with_browser_info(self) -> None:
        """Test form validates with browser_info JSON."""
        browser_info = json.dumps({
            'userAgent': 'Mozilla/5.0',
            'browser': 'Chrome',
            'os': 'Android',
            'isMobile': True,
        })
        form = FeedbackForm(data={
            'feedback_type': 'bug',
            'message': 'Test message',
            'browser_info': browser_info,
        })
        self.assertTrue(form.is_valid())

    def test_form_valid_without_browser_info(self) -> None:
        """Test form validates without browser_info."""
        form = FeedbackForm(data={
            'feedback_type': 'bug',
            'message': 'Test message',
        })
        self.assertTrue(form.is_valid())


class FormatFeedbackIssueTest(TestCase):
    """Tests for format_feedback_issue function."""

    def test_basic_issue_without_browser_info(self) -> None:
        """Test issue formatting without browser info."""
        title, body, labels = format_feedback_issue(
            feedback_type='bug',
            message='Something is broken',
        )
        self.assertIn('Bug Report', title)
        self.assertIn('Something is broken', body)
        self.assertIn('bug', labels)
        self.assertNotIn('## Environment', body)

    def test_issue_with_browser_info(self) -> None:
        """Test issue formatting with browser info."""
        browser_info = {
            'browser': 'Chrome',
            'os': 'Android',
            'osVersion': '14',
            'isMobile': True,
            'screenWidth': 412,
            'screenHeight': 915,
            'viewportWidth': 412,
            'viewportHeight': 800,
            'touchSupport': True,
            'maxTouchPoints': 5,
            'connectionType': '4g',
            'devicePixelRatio': 2.625,
            'userAgent': 'Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36',
        }
        title, body, labels = format_feedback_issue(
            feedback_type='bug',
            message='Layout broken on mobile',
            browser_info=browser_info,
        )

        # Check environment section exists
        self.assertIn('## Environment', body)

        # Check key info is included
        self.assertIn('**Browser:** Chrome', body)
        self.assertIn('**OS:** Android 14', body)
        self.assertIn('**Device:** Mobile', body)
        self.assertIn('**Screen:** 412x915', body)
        self.assertIn('**Viewport:** 412x800', body)
        self.assertIn('**Touch:** Yes (5 touch points)', body)
        self.assertIn('**Connection:** 4g', body)
        self.assertIn('**Pixel Ratio:** 2.625', body)

        # Check user agent is in collapsible section
        self.assertIn('<details>', body)
        self.assertIn('Full User Agent', body)
        self.assertIn('Mozilla/5.0', body)

    def test_issue_with_desktop_browser_info(self) -> None:
        """Test issue formatting with desktop browser info."""
        browser_info = {
            'browser': 'Firefox',
            'os': 'Linux',
            'isMobile': False,
            'screenWidth': 1920,
            'screenHeight': 1080,
            'viewportWidth': 1800,
            'viewportHeight': 900,
            'touchSupport': False,
            'devicePixelRatio': 1,
        }
        title, body, labels = format_feedback_issue(
            feedback_type='feature',
            message='Add dark mode',
            browser_info=browser_info,
        )

        self.assertIn('**Device:** Desktop', body)
        self.assertIn('**Browser:** Firefox', body)
        self.assertIn('**OS:** Linux', body)
        self.assertNotIn('**Touch:**', body)  # No touch on desktop

    def test_partial_browser_info(self) -> None:
        """Test issue formatting with partial browser info."""
        browser_info = {
            'userAgent': 'Mozilla/5.0',
            'isMobile': True,
        }
        title, body, labels = format_feedback_issue(
            feedback_type='usability',
            message='Hard to tap buttons',
            browser_info=browser_info,
        )

        self.assertIn('## Environment', body)
        self.assertIn('**Device:** Mobile', body)
        # Should not crash on missing fields
        self.assertNotIn('**Browser:** None', body)


class FeedbackViewTest(TestCase):
    """Tests for feedback_view."""

    def setUp(self) -> None:
        """Set up test client and user."""
        self.client = Client()
        self.url = reverse('contact:feedback')

    @patch('contact.views.EmailMessage')
    def test_feedback_with_browser_info(self, mock_email_class: MagicMock) -> None:
        """Test feedback submission includes browser info in email."""
        mock_email = MagicMock()
        mock_email_class.return_value = mock_email

        browser_info = json.dumps({
            'browser': 'Safari',
            'os': 'iOS',
            'osVersion': '17.2',
            'isMobile': True,
            'screenWidth': 390,
            'screenHeight': 844,
            'viewportWidth': 390,
            'viewportHeight': 664,
            'touchSupport': True,
            'userAgent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2)',
        })

        response = self.client.post(self.url, {
            'feedback_type': 'bug',
            'message': 'Page not scrolling properly on iPhone',
            'browser_info': browser_info,
        })

        # Check email was created with browser info in body
        mock_email_class.assert_called_once()
        call_kwargs = mock_email_class.call_args[1]
        body = call_kwargs['body']

        self.assertIn('Browser Environment:', body)
        self.assertIn('Browser: Safari', body)
        self.assertIn('OS: iOS 17.2', body)
        self.assertIn('Mobile: Yes', body)
        self.assertIn('Screen: 390x844', body)

    @patch('contact.views.EmailMessage')
    def test_feedback_without_browser_info(self, mock_email_class: MagicMock) -> None:
        """Test feedback submission works without browser info."""
        mock_email = MagicMock()
        mock_email_class.return_value = mock_email

        response = self.client.post(self.url, {
            'feedback_type': 'feature',
            'message': 'Please add export functionality',
        })

        mock_email_class.assert_called_once()
        call_kwargs = mock_email_class.call_args[1]
        body = call_kwargs['body']

        # Should not have browser environment section
        self.assertNotIn('Browser Environment:', body)

    @patch('contact.views.EmailMessage')
    def test_feedback_with_invalid_json_browser_info(
        self, mock_email_class: MagicMock
    ) -> None:
        """Test feedback handles invalid browser_info JSON gracefully."""
        mock_email = MagicMock()
        mock_email_class.return_value = mock_email

        response = self.client.post(self.url, {
            'feedback_type': 'bug',
            'message': 'Test message',
            'browser_info': 'not valid json{',
        })

        # Should still send email without crashing
        mock_email_class.assert_called_once()
        call_kwargs = mock_email_class.call_args[1]
        body = call_kwargs['body']

        # Should not have browser environment section
        self.assertNotIn('Browser Environment:', body)
